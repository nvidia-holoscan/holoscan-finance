# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Holoscan network -> quote bars -> concurrent CPU/GPU risk analytics -> HTTP.

With ``--ingest gpu`` the graph is the shipped operators, GpuMitchSourceOp ->
GpuQuoteBarsOp -> GpuRollingRiskOp, plus this application's dashboard publisher and
CPU cross-check. The host-memory ingest paths keep their own bars and analytics
operators below as the baseline.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import multiprocessing
import os
import threading
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from holoscan.marketdata.bars import QuoteBars
from holoscan.marketdata.wire import combine_batches

from .analytics import agreement, cpu_window, gpu_window, keep_off_cores
from .capture import LatencyCapture
from .mitch import PROTOCOL, TIMEZONE, day_start_ns, decode_packet


class _PacketView:
    """Minimal ReplayBatch-shaped record for LatencyCapture.record_packet."""

    __slots__ = (
        "events",
        "generation",
        "sequence",
        "sequence_count",
        "recv_wall_ns",
    )

    def __init__(self, events, generation, sequence, sequence_count, recv_wall_ns):
        self.events, self.generation, self.sequence = events, generation, sequence
        self.sequence_count, self.recv_wall_ns = sequence_count, recv_wall_ns


def number(value):
    return float(value) if np.isfinite(value) else None


class EquityState:
    def __init__(
        self,
        *,
        bar_seconds=5,
        window_seconds=300,
        stale_seconds=30,
        min_bars=20,
        max_spread_bps=100,
        ingest="async",
        cpu_threads=0,
        gpu_device=0,
        interval=0.05,
        analytics_backend="compare",
        replay_control=None,
    ):
        self.settings = dict(
            bar_seconds=bar_seconds,
            window_seconds=window_seconds,
            stale_seconds=stale_seconds,
            min_bars=min_bars,
            max_spread_bps=max_spread_bps,
        )
        QuoteBars([], **self.settings)  # validate before opening ports
        if interval <= 0:
            raise ValueError("interval must be positive")
        if analytics_backend not in ("compare", "cpu", "gpu"):
            raise ValueError("analytics backend must be compare, cpu or gpu")
        if ingest == "gpu" and analytics_backend == "cpu":
            raise ValueError("--ingest gpu runs GPU analytics: use backend compare or gpu")
        self.analytics_backend = analytics_backend
        # Reentrant: set_manifest resets the GPU bars operator, which takes this lock too.
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.ingest, self.cpu_threads, self.gpu_device, self.interval = (
            ingest,
            cpu_threads,
            gpu_device,
            interval,
        )
        self.source = self.bars = self.manifest = None
        self.bars_op = None  # GpuQuoteBarsOp with --ingest gpu; it owns self.bars
        self.generation = self.expected_sequence = self.last_taken = 0
        self.last_started = 0.0
        self.graph_status, self.error = "starting", None
        self.comparison_failure = None
        self.control = dict(speed=10.0, rate=50000.0, paused=False, seek=48600.0, seek_seq=1)
        self.sender = self.live = self.comparison = None
        self.comparisons = self.stale_packets = self.sequence_gaps = 0
        self.history = deque(maxlen=300)
        self.timings = deque(maxlen=120)
        self.capture = None
        self.used_sender_ports = set()
        # Full-day burst: the replay controller tcpreplays the captured day at line rate.
        self.replay_control, self.fullday = replay_control, None
        self.generation_first_ns = self.generation_last_ns = None
        # Full-day stall monitors (installed by start_heartbeat, reset at each burst):
        # longest garbage-collection pause, longest oversleep of a 5 ms ticker thread
        # (a process-wide GIL/CPU stall) and slowest GPU analytics window.
        self.gc_pause = dict(max_ms=0.0, generation=None, started=0)
        self.heartbeat = dict(max_ms=0.0, at_s=None)
        self.analytics_max_ms = 0.0

    def start_heartbeat(self):
        """Once per process (main): the GC callback runs in whichever thread collects, so
        it takes no locks."""
        gc.callbacks.append(self._gc_timer)

        def tick():
            while True:
                t = time.perf_counter_ns()
                if self.stop.wait(0.005):
                    return
                late = (time.perf_counter_ns() - t) / 1e6 - 5.0
                if late > 50 and self.source is not None and hasattr(self.source, "stall_event"):
                    self.source.stall_event("heartbeat", late)
                if late > self.heartbeat["max_ms"]:
                    self.heartbeat.update(max_ms=round(late, 1), at_s=round(time.monotonic(), 1))

        threading.Thread(target=tick, daemon=True, name="stall-heartbeat").start()

    def _gc_timer(self, phase, info):
        g = self.gc_pause
        if phase == "start":
            g["started"] = time.perf_counter_ns()
        elif g["started"]:
            ms = (time.perf_counter_ns() - g["started"]) / 1e6
            if ms > g["max_ms"]:
                g.update(max_ms=ms, generation=info.get("generation"))

    def decode(self, data, recv_ns, peer=None):
        with self.lock:
            manifest = self.manifest
            if not manifest or peer is None or peer[1] != manifest["sender_port"]:
                self.stale_packets += 1
                return [], 0
            generation = self.generation
        batches, count = decode_packet(
            data,
            recv_ns,
            generation=generation,
            midnight_ns=day_start_ns(manifest["date"]),
            universe=len(manifest["symbols"]),
        )
        capture = self.capture
        if capture is not None:
            for batch in batches:
                capture.record_packet(batch)
        return batches, count

    def decode_gpu(self, slots, lens, src_port, recv_ns):
        """GPU counterpart of :meth:`decode` for one staged device slot buffer."""
        from holoscan.marketdata.gpu_bars import GpuReplayBatch
        from holoscan.marketdata.gpu_mitch import decode_packets

        with self.lock:
            manifest = self.manifest
            generation = self.generation
            port = manifest["sender_port"] if manifest else None
            stale = np.ones(len(src_port), bool) if port is None else src_port != port
            self.stale_packets += int(stale.sum())
        keep = ~stale
        if not keep.any():
            return [], 0, 0
        dec = decode_packets(
            slots,
            lens,
            midnight_ns=day_start_ns(manifest["date"]),
            universe=len(manifest["symbols"]),
            keep=keep,
        )
        ok = (dec.status == 0) & keep
        malformed = int(((dec.status != 0) & keep).sum())
        idx = np.flatnonzero(ok)
        if not len(idx):
            return [], 0, malformed
        batch = GpuReplayBatch(
            dec.columns,
            generation,
            dec.sequence[idx],
            dec.wire_count[idx],
            dec.offsets[idx],
            recv_ns[idx],
            int(recv_ns[idx[-1]]),
        )
        capture = self.capture
        if capture is not None:
            ends = np.r_[batch.packet_offsets[1:], batch.n_events]
            for k in range(batch.n_packets):
                capture.record_packet(
                    _PacketView(
                        range(int(ends[k] - batch.packet_offsets[k])),
                        generation,
                        int(batch.sequence[k]),
                        int(batch.wire_count[k]),
                        int(batch.packet_recv_ns[k]),
                    )
                )
        return [batch], int(dec.wire_count[idx].sum()), malformed

    def start_capture(self, payload):
        seconds = float(payload.get("seconds", 120))
        with self.lock:
            if self.capture and not self.capture.closed:
                raise ValueError("finish the previous capture before starting another")
            if (
                not self.live
                or self.live["bars"]
                < self.settings["window_seconds"] // self.settings["bar_seconds"]
            ):
                raise ValueError("wait for a full return window before capturing")
            self.capture = LatencyCapture(seconds, self.generation, backend=self.analytics_backend)
            return self.capture.status()

    def finish_capture(self):
        if self.capture is None:
            raise ValueError("no capture has been started")
        result = self.capture.finish()
        result["ingest"] = self.ingest
        result["settings"] = self.settings
        result["wire_protocol"] = PROTOCOL
        result["minimum_update_interval_seconds"] = self.interval
        return result

    def set_manifest(self, payload):
        symbols = payload.get("symbols")
        generation = payload.get("generation")
        if (
            payload.get("format") != "nyq-equity-replay-v1"
            or payload.get("wire_protocol") != PROTOCOL
            or payload.get("market_timezone") != TIMEZONE
            or type(payload.get("sender_port")) is not int
            or not 1 <= payload["sender_port"] <= 65535
            or not isinstance(symbols, list)
            or not 1 <= len(symbols) <= 10000
            or not all(isinstance(s, str) and 0 < len(s) <= 128 for s in symbols)
            or len(set(symbols)) != len(symbols)
            or type(generation) is not int
            or generation <= 0
            or not isinstance(payload.get("date"), str)
            or type(payload.get("start_ns")) is not int
            or type(payload.get("end_ns")) is not int
            or payload["start_ns"] >= payload["end_ns"]
            or type(payload.get("events")) is not int
            or payload["events"] < 1
            or not isinstance(payload.get("profile"), list)
        ):
            raise ValueError("invalid NYQ directory or generation")
        with self.lock:
            if generation <= self.generation:
                return {"generation": self.generation}
            if payload["sender_port"] in self.used_sender_ports:
                raise ValueError("MITCH replay sender port must be new for each generation")
            day_start_ns(payload["date"])  # validate before replacing the active generation
            self.used_sender_ports.add(payload["sender_port"])
            self.manifest = payload.copy()
            if self.capture and self.capture.includes(time.time_ns()):
                self.capture.generation_changed = True
            self.generation = generation
            self.generation_first_ns = self.generation_last_ns = None
            if self.fullday and self.fullday["status"] == "starting":
                s = self.source
                self.fullday.update(
                    status="running", generation=generation, sender_port=payload["sender_port"],
                    tape_seconds=(payload["end_ns"] - payload["start_ns"]) / 1e9,
                    base=(s.packets, s.dropped, s.ring_dropped) if s else (0, 0, 0),
                )  # fmt: skip
                if s is not None:
                    s.reset_stalls()
                self.gc_pause.update(max_ms=0.0, generation=None)
                self.heartbeat.update(max_ms=0.0, at_s=None)
                self.analytics_max_ms = 0.0
            if self.ingest == "gpu":
                self.bars = None
                if self.bars_op is not None:
                    self.bars_op.reset(symbols, generation)
                    self.bars = self.bars_op.bars
            else:
                self.bars = QuoteBars(symbols, **self.settings)
            self.live = self.comparison = None
            self.last_taken = 0
            self.expected_sequence = 1
            self.history.clear()
            self.timings.clear()
            return {"generation": generation}

    def attach_bars(self, op):
        """GPU ingest: the graph's GpuQuoteBarsOp owns the bars. Start it on the current
        generation if the directory arrived before the graph was built."""
        with self.lock:
            self.bars_op = op
            if self.manifest:
                op.reset(self.manifest["symbols"], self.generation)
                self.bars = op.bars

    def validation_allowed(self):
        """The CPU comparator competes with ingest for cores and memory bandwidth: skip
        validation while a full-day burst is in flight."""
        return self.analytics_backend != "compare" or not (
            self.fullday and self.fullday["status"] in ("starting", "running")
        )

    def feed_control(self, payload=None):
        with self.lock:
            if payload is not None:
                if set(payload) - {"speed", "rate", "paused", "seek"}:
                    raise ValueError("unsupported replay control")
                c = {**self.control, **payload}
                if (
                    not 0 <= float(c["speed"]) <= 100
                    or not 1 <= float(c["rate"]) <= 200000
                    or type(c["paused"]) is not bool
                    or not 0 <= float(c["seek"]) < 86400
                ):
                    raise ValueError("invalid replay control")
                if "seek" in payload:
                    c["seek_seq"] += 1
                self.control = c
            return dict(self.control)

    def _arrived(self, batches):
        """First/last arrival of the current generation (caller holds the lock)."""
        if any(b.generation == self.generation for b in batches):
            now = time.monotonic_ns()
            self.generation_first_ns = self.generation_first_ns or now
            self.generation_last_ns = now

    def start_fullday(self, pps):
        """Pause the paced sender and tcpreplay the captured day at ``pps`` (replay controller)."""
        if not self.replay_control:
            raise ValueError("no replay controller configured (--replay-control)")
        with self.lock:
            if self.fullday and self.fullday["status"] in ("starting", "running"):
                raise ValueError("a full-day replay is already running")
            self.fullday = dict(status="starting", pps=pps, requested=time.time())
        self.feed_control({"paused": True})
        reply = replay_request(self.replay_control, dict(pps=pps), timeout=60)
        with self.lock:
            if reply.get("error") and self.fullday["status"] == "starting":
                self.fullday.update(status="error", error=reply["error"])
            self.fullday["controller"] = reply
            return dict(self.fullday)

    def _stalls(self, source):
        """Copies: the drain thread keeps appending while the snapshot serializes."""
        st = getattr(source, "stalls", None) or {}
        return dict(
            st,
            burst_gaps_ms=sorted(st.get("burst_gaps_ms", [])),
            events=list(st.get("events", [])),
            heartbeat=dict(self.heartbeat),
            gc_max_ms=round(self.gc_pause["max_ms"], 1),
            gc_generation=self.gc_pause["generation"],
            analytics_max_ms=round(self.analytics_max_ms, 1),
        )

    def fullday_status(self):
        """Caller holds the lock. Times are first to last packet arrival of the burst generation."""
        f = self.fullday
        if not f or f["status"] != "running" or self.generation != f.get("generation"):
            return f
        s = self.source
        p0, g0, r0 = f["base"]
        packets, total = (
            s.packets - p0,
            f["controller"].get("packets") if f.get("controller") else None,
        )
        now = time.monotonic_ns()
        op = self.bars_op
        first, last = (
            (op.first_batch_ns, op.last_batch_ns)
            if op is not None
            else (self.generation_first_ns, self.generation_last_ns)
        )
        seconds = (last - first) / 1e9 if first and last and last > first else None
        f.update(
            packets=packets, total_packets=total, sequence_gaps=s.dropped - g0,
            ring_dropped=s.ring_dropped - r0, seconds=seconds,
            events=self.bars.events if self.bars is not None else None,
            completed_bars=self.bars.completed_bars if self.bars is not None else None,
            first_s=round(first / 1e9, 3) if first else None,
            stalls=self._stalls(s),
        )  # fmt: skip
        pending = s.pending_packets() if hasattr(s, "pending_packets") else 0
        if last is not None and now - last > 2e9 and not pending:  # the burst has ended
            f["status"] = "done"
            f["x_realtime"] = f["tape_seconds"] / seconds if seconds else None
            # NIC queue overflow appears in DAQIRI's stats-thread log ("Rx: Dropped N packets");
            # daqiri.print_stats() is not safe while the engine runs.
            worst = {k: v for k, v in f["stalls"].items() if k != "events"}
            print(
                f"full-day replay done: {f['packets']} packets, gaps {f['sequence_gaps']}, "
                f"ring drops {f['ring_dropped']}, stalls {worst}",
                flush=True,
            )
        elif last is None and time.time() - f["requested"] > 120:
            f.update(status="error", error="no packets arrived from the replay controller")
        return f

    def observe(self, batches):
        with self.lock:
            if self.bars is None:
                return
            self._arrived(batches)
            pending = []
            for batch in batches:
                if batch.generation != self.generation or batch.sequence < self.expected_sequence:
                    self.stale_packets += 1
                    continue
                if self.expected_sequence and batch.sequence > self.expected_sequence:
                    if pending:
                        self.bars.observe(combine_batches(pending))
                        pending.clear()
                    self.sequence_gaps += batch.sequence - self.expected_sequence
                    # Unknown missing updates: require refreshed marks and a new return endpoint.
                    self.bars.mid[:] = np.nan
                    self.bars.previous[:] = np.nan
                self.expected_sequence = batch.sequence + batch.sequence_count
                pending.append(batch)
            if pending:
                self.bars.observe(combine_batches(pending))

    def take_work(self):
        with self.lock:
            if self.bars is None or self.bars.completed_bars == self.last_taken:
                return None
            if time.monotonic() - self.last_started < self.interval:
                return None
            window = self.bars.window()
            if window is None:
                return None
            self.last_taken = self.bars.completed_bars
            self.last_started = time.monotonic()
            work = dict(
                rows=window[0],
                times=window[1],
                generation=self.generation,
                recv_ns=self.bars.bar_recv_ns,
                validate=self.validation_allowed(),
            )
            return work

    def publish_gpu(self, work, result):
        with self.lock:
            if work["generation"] != self.generation:
                return
            original = result
            result = {k: v for k, v in result.items() if k not in ("matrix", "covariance_probes")}
            self.analytics_max_ms = max(self.analytics_max_ms, result.get("milliseconds") or 0.0)
            now = time.time_ns()
            result.update(
                timestamp_ns=int(work["times"][-1]),
                generation=self.generation,
                published_wall_ns=now,
                receive_to_result_ms=(now - work["recv_ns"]) / 1e6,
            )
            self.live = result
            if self.capture is not None:
                if self.analytics_backend == "compare":
                    self.capture.record_gpu(work, result)
                else:
                    self.capture.record_single(work, result, original)
            self.history.append(
                [
                    int(work["times"][-1]) // 10**9,
                    float(np.median(result["annual_vol"])) if result["annual_vol"] else None,
                    result["mean_correlation"],
                ]
            )

    def publish_comparison(self, work, cpu, gpu):
        with self.lock:
            if work["generation"] != self.generation:
                return
            ok = agreement(cpu, gpu)
            # Same generation, identical windows and identical incremental additions/expirations.
            matched = all(
                cpu[k] == gpu[k]
                for k in ("method", "added_bars", "expired_bars", "bars", "universe")
            )
            usable = ok and matched and gpu["eligible"] >= 2
            if self.capture is not None:
                self.capture.record_cpu(work, cpu, gpu, ok and matched)
            now = time.time_ns()
            self.comparisons += 1
            self.comparison = dict(
                identifier=self.comparisons,
                generation=self.generation,
                timestamp_ns=int(work["times"][-1]),
                eligible=gpu["eligible"],
                pairs=gpu["pairs"],
                cpu_ms=cpu["milliseconds"],
                gpu_ms=gpu["milliseconds"],
                speedup=cpu["milliseconds"] / gpu["milliseconds"] if usable else None,
                agreement=ok,
                matched_work=matched,
                cpu_threads=cpu["threads"],
                cpu_affinity=cpu.get("cpu_affinity"),
                cpu_thread_trials_ms=cpu.get("cpu_thread_trials_ms"),
                cpu_tuned_bars=cpu.get("cpu_tuned_bars"),
                gpu_name=gpu.get("gpu_name"),
                blas=cpu["blas"],
                method=gpu["method"],
                added_bars=gpu["added_bars"],
                expired_bars=gpu["expired_bars"],
                cpu_receive_to_result_ms=(cpu["finished_wall_ns"] - work["recv_ns"]) / 1e6,
                published_wall_ns=now,
                validation="all eligible volatilities, 32x32 covariance probes and displayed correlations",
            )
            if usable and gpu["method"] == "incremental":
                self.timings.append([cpu["milliseconds"], gpu["milliseconds"]])
            if not ok or not matched:
                self.comparison_failure = {
                    "comparison": self.comparison.copy(),
                    "eligible_indices_same": cpu["eligible_indices"] == gpu["eligible_indices"],
                    "heatmap_indices_same": cpu["heatmap_indices"] == gpu["heatmap_indices"],
                    "cpu": {
                        k: cpu[k]
                        for k in (
                            "method",
                            "added_bars",
                            "expired_bars",
                            "bars",
                            "universe",
                            "heatmap_indices",
                            "heatmap",
                            "annual_vol",
                            "eligible_indices",
                            "covariance_probes",
                        )
                    },
                    "gpu": {
                        k: gpu[k]
                        for k in (
                            "method",
                            "added_bars",
                            "expired_bars",
                            "bars",
                            "universe",
                            "heatmap_indices",
                            "heatmap",
                            "annual_vol",
                            "eligible_indices",
                            "covariance_probes",
                        )
                    },
                }
                self.error = "CPU/GPU comparison failed; inspect /snapshot before using timings"

    def network(self):
        s = self.source
        if s is None:
            return {"ready": False, "status": "starting", "ingest": self.ingest}
        op = self.bars_op
        now = time.time_ns()
        buckets = s.feed_buckets(last=100)
        boundary = now // 100000000 * 100
        age = (now - s.last_packet_wall_ns) / 1e9 if s.last_packet_wall_ns else None
        return dict(
            ready=(
                s._sock is not None
                if self.ingest == "async"
                else (
                    getattr(s, "_daq", None) is not None
                    and (
                        s.drain_alive()
                        if hasattr(s, "drain_alive")
                        else getattr(s, "_thread", None) is not None and s._thread.is_alive()
                    )
                    and not getattr(s, "error", None)
                )
            ),
            status="receiving"
            if age is not None and age < 2
            else "idle"
            if age is not None
            else "waiting",
            ingest=self.ingest,
            transport="MITCH (MIT303 11.9) / UDP · NYQ replay mapping",
            wire_protocol=PROTOCOL,
            message_count_scope="MITCH messages including Time; replay rate is source observations/s",
            packets=s.packets,
            received_messages=s.received_messages,
            decoded_messages=s.messages,
            messages_per_second=sum(n for t, n in buckets if boundary - 1000 <= t < boundary),
            sequence_gaps=s.dropped,
            analytics_sequence_gaps=self.sequence_gaps + (op.sequence_gaps if op else 0),
            ring_dropped=s.ring_dropped,
            ring_messages=s.ring_fill(),
            malformed_packets=s.malformed_packets,
            out_of_order_packets=s.out_of_order_packets,
            stale_packets=self.stale_packets + (op.stale_packets if op else 0),
            error=getattr(s, "error", None),
            buckets=buckets,
            gpu_ingest=dict(
                bursts=s.bursts,
                non_udp_frames=s.non_udp_frames,
                drain_host_ms=s.drain_host_ns / 1e6,
                frame_fetch=s.frame_fetch,
                ring_peak=s.ring_peak(),
                payload_path="NIC -> GPU memory (GPUDirect) -> GPU gather -> GPU MITCH decode",
            )
            if self.ingest == "gpu"
            else None,
        )

    def snapshot(self):
        with self.lock:
            result = dict(
                graph_status=self.graph_status,
                error=self.error,
                comparison_failure=self.comparison_failure,
                settings=self.settings,
                minimum_update_interval_seconds=self.interval,
                analytics_backend=self.analytics_backend,
                feed_control=dict(self.control),
                sender_status=self.sender,
                generation=self.generation,
                live=self.live,
                comparison=self.comparison,
                comparisons=self.comparisons,
                history=list(self.history),
                capture=self.capture.status() if self.capture else None,
            )
            result["profile"] = (
                {k: self.manifest[k] for k in ("date", "events", "profile", "start_ns", "end_ns")}
                if self.manifest
                else {}
            )
            result["universe"] = self.bars.symbols if self.bars else []
            if self.bars:
                b = self.bars
                result["observations"] = dict(
                    events=b.events,
                    quotes=b.quotes,
                    trades=b.trades,
                    invalid_quotes=b.invalid_quotes,
                    late_events=b.late_events,
                    completed_bars=b.completed_bars,
                )
                if self.live:
                    indices = self.live["eligible_indices"]
                    ranking = np.argsort(-np.asarray(self.live["annual_vol"]))[:30]
                    mid, trade, volume, quote_time = b.mid, b.trade, b.volume, b.quote_time
                    result["prices"] = [
                        dict(
                            symbol=b.symbols[indices[j]],
                            mid=number(mid[indices[j]]),
                            last_trade=number(trade[indices[j]]),
                            observed_volume=number(volume[indices[j]]),
                            annual_vol=self.live["annual_vol"][j],
                            quote_age_s=(b.last_event_ns - quote_time[indices[j]]) / 1e9,
                        )
                        for j in ranking
                    ]
            if len(self.timings):
                timing = np.asarray(self.timings)
                result["rolling_timing"] = dict(
                    samples=len(timing),
                    cpu_median_ms=float(np.median(timing[:, 0])),
                    gpu_median_ms=float(np.median(timing[:, 1])),
                    speedup=float(np.median(timing[:, 0]) / np.median(timing[:, 1])),
                )
        result["network"] = self.network()
        with self.lock:
            f = self.fullday_status()
            result["fullday"] = dict(f, base=None) if f else None
        result["fullday_available"] = bool(self.replay_control)
        return result


def make_graph(state, host, port, source_kwargs=None, batch_size=16384):
    from holoscan.conditions import BooleanCondition, PeriodicCondition
    from holoscan.core import Application, ConditionType, Operator
    from holoscan.schedulers import MultiThreadScheduler

    reserved = tuple(
        c
        for c in ((source_kwargs or {}).get(k) for k in ("master_core", "rx_core"))
        if c is not None
    )
    window_bars = state.settings["window_seconds"] // state.settings["bar_seconds"]

    def cpu_pool():
        """The CPU comparator: one spawned process, kept off the DAQIRI cores."""
        return ProcessPoolExecutor(
            max_workers=1,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=keep_off_cores,
            initargs=(reserved,),
        )

    def stop_requested(alive, source):
        if state.stop.is_set():
            alive.disable_tick()
            source._stop.set()
            source._eos.set()
            return True
        return False

    class Graph(Application):
        def compose(self):
            if state.ingest == "gpu":
                self.compose_gpu()
            else:
                self.compose_host()
            self.scheduler(
                MultiThreadScheduler(
                    self,
                    # GPU ingest: source, bars and the synchronous analytics each need a
                    # worker, plus the dashboard publisher.
                    worker_thread_number=4 if state.ingest == "gpu" else 3,
                    stop_on_deadlock=True,
                    stop_on_deadlock_timeout=1000,
                    name="equity_scheduler",
                )
            )

        def compose_gpu(self):
            from holoscan.marketdata import GpuMitchSourceOp, GpuQuoteBarsOp, GpuRollingRiskOp

            source = GpuMitchSourceOp(
                self,
                port=port,
                batch_size=batch_size,
                keep_alive=True,
                batch_decoder=state.decode_gpu,
                # ~5 s of a 200k pkt/s burst (2 GiB of device memory): the native drain
                # keeps receiving into it while Python threads stall.
                ring_size=1 << 20,
                gpu_device=state.gpu_device,
                name="nyq_network",
                **(source_kwargs or {}),
            )
            state.source = source
            bars = GpuQuoteBarsOp(self, **state.settings, lock=state.lock, name="quote_bars")
            state.attach_bars(bars)
            # Compare mode: the CPU comparison in flight. Each validated GPU window waits
            # for its comparison, so every published GPU result has a CPU pair.
            pending = {}

            def validate_next():
                if state.analytics_backend != "compare" or not state.validation_allowed():
                    return False
                pending["claimed"] = True
                return True

            risk = GpuRollingRiskOp(
                self,
                bar_seconds=state.settings["bar_seconds"],
                gpu_device=state.gpu_device,
                min_interval_s=state.interval,
                ready=lambda: not pending,
                host_variance=validate_next,
                name="rolling_risk",
            )

            class PublishOp(Operator):
                """Dashboard publication and, in compare mode, the CPU cross-check."""

                def __init__(self, fragment):
                    self.alive = BooleanCondition(fragment, enable_tick=True, name="publish_alive")
                    super().__init__(
                        fragment,
                        self.alive,
                        PeriodicCondition(fragment, recess_period=5000000, name="publish_clock"),
                        name="publish",
                    )
                    self.pool = None

                def setup(self, spec):
                    spec.input("risk").condition(ConditionType.NONE)

                def start(self):
                    if state.analytics_backend == "compare":
                        self.pool = cpu_pool()

                def stop(self):
                    if self.pool:
                        self.pool.shutdown(wait=True, cancel_futures=True)

                def compute(self, op_input, op_output, context):
                    if stop_requested(self.alive, source):
                        return
                    try:
                        future = pending.get("future")
                        if future is not None and future.done():
                            state.publish_comparison(
                                pending["work"], future.result(), pending["gpu"]
                            )
                            pending.clear()
                        update = op_input.receive("risk")
                        if update is None:
                            return
                        w = update.window
                        work = dict(
                            rows=w.rows, times=w.times, generation=w.generation, recv_ns=w.recv_ns
                        )
                        if update.rows_host is not None:
                            work["rows_host"] = update.rows_host
                        state.publish_gpu(work, update.result)
                        if update.rows_host is not None and self.pool is not None:
                            pending.update(
                                work=work,
                                gpu=update.result,
                                future=self.pool.submit(
                                    cpu_window,
                                    update.rows_host,
                                    w.times,
                                    w.generation,
                                    state.settings["bar_seconds"],
                                    state.cpu_threads,
                                    window_bars,
                                ),
                            )
                    except Exception as exc:
                        state.error = f"Analytics failed: {exc}"
                        self.alive.disable_tick()

            self.add_flow(source, bars, {("tick", "events")})
            self.add_flow(bars, risk, {("window", "window")})
            self.add_flow(risk, PublishOp(self), {("risk", "risk")})

        def compose_host(self):
            from .host_sources import AnoMitchSourceOp, AsyncUdpMitchSourceOp

            cls = AnoMitchSourceOp if state.ingest == "ano" else AsyncUdpMitchSourceOp
            source = cls(
                self,
                host=host,
                port=port,
                batch_size=batch_size,
                keep_alive=True,
                datagram_decoder=state.decode,
                ring_size=65536,
                name="nyq_network",
                **(source_kwargs or {}),
            )
            state.source = source

            class BarsOp(Operator):
                def setup(self, spec):
                    spec.input("events")

                def compute(self, op_input, op_output, context):
                    state.observe(op_input.receive("events"))

            class RiskOp(Operator):
                def __init__(self, fragment):
                    self.alive = BooleanCondition(fragment, enable_tick=True, name="risk_alive")
                    super().__init__(
                        fragment,
                        self.alive,
                        PeriodicCondition(fragment, recess_period=5000000, name="risk_clock"),
                        name="rolling_risk",
                    )
                    self.future = self.pool = self.work = self.gpu = None
                    self.gpu_future = self.gpu_pool = None

                def setup(self, spec):
                    pass

                def start(self):
                    if state.analytics_backend in ("compare", "cpu"):
                        self.pool = cpu_pool()
                    if state.analytics_backend in ("compare", "gpu"):
                        # A separate process keeps CUDA work off the decoder's interpreter lock.
                        self.gpu_pool = ProcessPoolExecutor(
                            max_workers=1, mp_context=multiprocessing.get_context("spawn")
                        )

                def stop(self):
                    if self.pool:
                        self.pool.shutdown(wait=True, cancel_futures=True)
                    if self.gpu_pool:
                        self.gpu_pool.shutdown(wait=True, cancel_futures=True)

                def compute(self, op_input, op_output, context):
                    if stop_requested(self.alive, source):
                        return
                    try:
                        if self.gpu_future:
                            if not self.gpu_future.done():
                                return
                            self.gpu = self.gpu_future.result()
                            state.publish_gpu(self.work, self.gpu)
                            self.gpu_future = None
                        if self.future:
                            if not self.future.done():
                                return
                            if state.analytics_backend == "compare":
                                state.publish_comparison(self.work, self.future.result(), self.gpu)
                            else:
                                state.publish_gpu(self.work, self.future.result())
                            self.future = None
                        work = state.take_work()
                        if work is None:
                            return
                        self.work = work
                        if self.pool is not None and work["validate"]:
                            self.future = self.pool.submit(
                                cpu_window,
                                work["rows"],
                                work["times"],
                                work["generation"],
                                state.settings["bar_seconds"],
                                state.cpu_threads,
                                window_bars,
                            )
                        if self.gpu_pool is not None:
                            self.gpu_future = self.gpu_pool.submit(
                                gpu_window,
                                work["rows"],
                                work["times"],
                                work["generation"],
                                state.settings["bar_seconds"],
                                state.gpu_device,
                            )
                    except Exception as exc:
                        state.error = f"Analytics failed: {exc}"
                        self.alive.disable_tick()

            self.add_flow(source, BarsOp(self, name="quote_bars"), {("tick", "events")})
            self.add_operator(RiskOp(self))

    return Graph()


def web_app(state):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse

    app = FastAPI(title="NYQ equity price, volatility and correlation")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (Path(__file__).parent / "webui/index.html").read_text()

    @app.get("/snapshot")
    def snapshot():
        return state.snapshot()

    def mutate(fn, payload):
        try:
            return fn(payload)
        except (ValueError, TypeError, KeyError) as exc:
            raise HTTPException(422, detail=str(exc)) from exc

    @app.post("/manifest")
    def manifest(payload: dict):
        return mutate(state.set_manifest, payload)

    @app.post("/capture")
    def capture(payload: dict):
        return mutate(state.start_capture, payload)

    @app.get("/capture")
    def capture_status():
        return state.capture.status() if state.capture else {}

    @app.post("/capture/finish")
    def capture_finish():
        return mutate(lambda _: state.finish_capture(), {})

    @app.post("/replay/fullday")
    def fullday(payload: dict):
        return mutate(lambda p: state.start_fullday(int(p.get("pps", 200_000))), payload)

    @app.post("/replay/resume")
    def resume(payload: dict):
        """Back to the paced sender: unpause and restart it at the session open (new generation)."""
        start = (state.manifest or {}).get("start_ns")
        seek = start // 10**9 % 86400 if start else state.control["seek"]
        return mutate(state.feed_control, {"paused": False, "seek": float(seek)})

    @app.get("/feedrate")
    def feedrate():
        return state.feed_control()

    @app.post("/feedrate")
    def control(payload: dict):
        return mutate(state.feed_control, payload)

    @app.post("/feedstatus")
    def status(payload: dict):
        with state.lock:
            if (
                payload.get("generation") == state.generation
                and type(payload.get("seek_seq")) is int
                and 0 <= payload["seek_seq"] <= state.control["seek_seq"]
            ):
                state.sender = payload
        return {"status": "received"}

    if os.environ.get("HOLOSCAN_FINANCE_FAULTS") == "1":

        @app.post("/debug/hold_gil")
        def hold_gil(payload: dict):
            """Fault injection: hold the interpreter lock for ``seconds``, as a Python thread
            blocked in the kernel does (the full-day stalls). ctypes.PyDLL keeps the GIL."""
            seconds = min(float(payload.get("seconds", 1.0)), 10.0)
            ctypes.PyDLL(None).usleep(int(seconds * 1e6))
            return {"held_s": seconds}

    return app


def replay_request(url, payload, timeout=5):
    """Ask the replay controller (deploy/igx_thor/gpu_ingest_bench/replay_control.py) to start."""
    import json
    import urllib.request

    try:
        req = urllib.request.Request(
            url.rstrip("/") + "/start",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception as exc:  # the dashboard shows it; ingest is unaffected
        return dict(error=str(exc))


def main():
    import sys

    import uvicorn

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--udp-host", default="127.0.0.1")
    p.add_argument("--udp-port", type=int, default=26400)
    p.add_argument(
        "--ingest",
        choices=("async", "ano", "gpu"),
        default="async",
        help="async: kernel UDP socket; ano: DAQIRI kernel bypass into host memory, Python decode; "
        "gpu: DAQIRI GPUDirect into GPU memory, GPU gather + MITCH decode + quote bars",
    )
    p.add_argument("--ano-pcie", default="0004:03:00.0")
    p.add_argument("--ano-config")
    p.add_argument("--ano-master-core", type=int, default=3)
    p.add_argument("--ano-rx-core", type=int, default=9)
    p.add_argument(
        "--ano-num-bufs",
        type=int,
        default=114688,
        help="GPU ingest NIC receive buffers: ~0.57 s of a 200k pkt/s burst; the GPU's BAR1 "
        "window bounds it (256 MiB holds ~120k 2 KiB buffers); ignored with --ano-config",
    )
    p.add_argument("--bar-seconds", type=int, default=5)
    p.add_argument("--window-seconds", type=int, default=300)
    p.add_argument("--stale-seconds", type=int, default=30)
    p.add_argument("--min-bars", type=int, default=20)
    p.add_argument("--max-spread-bps", type=float, default=100)
    p.add_argument("--cpu-threads", type=int, default=0)
    p.add_argument("--gpu-device", type=int, default=0)
    p.add_argument("--interval", type=float, default=0.05)
    p.add_argument(
        "--batch-messages",
        type=int,
        default=None,
        help="maximum messages staged per source compute() call; default 131072 for --ingest gpu "
        "(amortizes per-batch launches), 16384 otherwise. Sources take what is queued and never "
        "wait to fill a batch, so a larger limit adds no latency at low rates.",
    )
    p.add_argument("--analytics-backend", choices=("compare", "cpu", "gpu"), default="compare")
    p.add_argument(
        "--replay-control",
        help="replay controller URL (tcpreplay of a captured day) for the dashboard's full-day button",
    )
    a = p.parse_args()
    kwargs = dict(vars(a))
    for name in (
        "host",
        "port",
        "udp_host",
        "udp_port",
        "ano_pcie",
        "ano_config",
        "ano_master_core",
        "ano_rx_core",
        "ano_num_bufs",
        "batch_messages",
    ):
        kwargs.pop(name)
    state = EquityState(**kwargs)
    source = {}
    if a.ingest in ("ano", "gpu"):
        from holoscan.marketdata.daqiri import require_daqiri

        require_daqiri()
        source = dict(
            pcie_address=a.ano_pcie,
            daqiri_config=a.ano_config,
            master_core=a.ano_master_core,
            rx_core=a.ano_rx_core,
        )
    if a.ingest == "gpu":
        from .warmup import warmup

        source["num_bufs"] = a.ano_num_bufs

        started = time.perf_counter()
        warmup(state.settings, device=a.gpu_device)
        print(f"GPU decode/bars kernels ready in {time.perf_counter() - started:.1f} s", flush=True)
    sys.setswitchinterval(0.001)
    batch = a.batch_messages or (131072 if a.ingest == "gpu" else 16384)
    graph = make_graph(state, a.udp_host, a.udp_port, source, batch_size=batch)

    def run():
        state.graph_status = "running"
        try:
            graph.run()
        except Exception as exc:
            state.error = str(exc)
        finally:
            state.graph_status = "stopped"

    state.start_heartbeat()
    worker = threading.Thread(target=run, daemon=True, name="nyq-holoscan")
    worker.start()
    try:
        print(
            f"Equity analytics: http://{a.host}:{a.port}; UDP {a.udp_host}:{a.udp_port}", flush=True
        )
        uvicorn.run(web_app(state), host=a.host, port=a.port, log_level="warning")
    finally:
        state.stop.set()
        worker.join(timeout=30)


if __name__ == "__main__":
    main()
