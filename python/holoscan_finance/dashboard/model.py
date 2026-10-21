# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ``/snapshot`` JSON: every 4.3 key and value, built from host objects only.

4.3 held ``state.lock`` (shared with the bars operator) across four D2H copies per request;
here the bars lane publishes exact counters per batch and a symbol snapshot at most every
250 ms, so a snapshot (about 1-3 ms of Python) never touches CUDA, never takes a
data-plane lock, and reads the directory without a lock. The ``network`` constants are
4.3's verbatim (``index.html``, ``measure.py`` and ``compare_runs.py`` read them); the
``engine`` block is new and additive. From 4.3 ``EquityState.snapshot``, ``network``,
``start_capture`` and ``finish_capture``.
"""

from __future__ import annotations

import os
import time
from typing import Any

import numpy as np

from ..nyq.mapping import PROTOCOL
from ..settings import GPU_INGEST_MODES, LEGACY_INGEST_TOKEN

TRANSPORT = "MITCH (MIT303 11.9) / UDP · NYQ replay mapping"
MESSAGE_COUNT_SCOPE = "MITCH messages including Time; replay rate is source observations/s"
PAYLOAD_PATHS = {
    "gpudirect": "NIC -> GPU memory (GPUDirect) -> GPU gather -> GPU MITCH decode",
    "replay": "pcap frames in GPU memory -> GPU gather -> GPU MITCH decode",
}
PROFILE_KEYS = ("date", "events", "profile", "start_ns", "end_ns")
OBSERVATION_KEYS = ("events", "quotes", "trades", "invalid_quotes", "late_events", "completed_bars")


def _number(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


class DashboardModel:
    """Read side of the dashboard over an ``EquityContext`` (or any object with its
    attribute names: ``args``, ``settings``, ``directory``, ``feed``, ``results``, ``tap``,
    ``ingest_stats``, ``bars_stats``, ``risk_stats``, ``fullday``, ``status``,
    ``plan_summary``). ``args.ingest`` is the canonical ingest mode."""

    def __init__(self, ctx: Any) -> None:
        self.ctx = ctx

    def snapshot(self) -> dict:
        """``GET /snapshot``: every 4.3 key, plus ``engine``."""
        c = self.ctx
        d = c.directory.current  # one lock-free read: every part below uses this generation
        view = c.results.view()
        errors = (c.status.error, view["error"], c.risk_stats.error, c.ingest_stats.error)
        result = dict(
            graph_status=c.status.graph_status,
            error=next((e for e in errors if e is not None), None),
            comparison_failure=view["comparison_failure"],
            settings=c.settings.as_params(),
            minimum_update_interval_seconds=c.args.interval,
            analytics_backend=c.args.analytics_backend,
            feed_control=c.feed.control(),
            sender_status=c.feed.sender,
            generation=d.generation if d is not None else 0,
            live=view["live"],
            comparison=view["comparison"],
            comparisons=view["comparisons"],
            history=view["history"],
            capture=c.tap.status(),
            profile={k: d.manifest[k] for k in PROFILE_KEYS} if d is not None else {},
            universe=list(d.symbols) if d is not None else [],
        )
        if d is not None:  # 4.3 had bars (and so observations) from the first manifest on
            # One read: the bars lane replaces the counters per batch and drops them at a
            # generation change. Before this generation's first batch, 4.3's fresh zeros.
            counters = c.bars_stats.counters
            current = counters is not None and counters.generation == d.generation
            result["observations"] = {
                k: getattr(counters, k) if current else 0 for k in OBSERVATION_KEYS
            }
            prices = self._prices(d.generation, view["live"])
            if prices is not None:
                result["prices"] = prices
        if view["rolling_timing"] is not None:
            result["rolling_timing"] = view["rolling_timing"]
        result["network"] = self.network()
        f = c.fullday.status()
        result["fullday"] = dict(f, base=None) if f else None
        result["fullday_available"] = c.fullday.available()
        result["engine"] = self._engine()
        return result

    def _prices(self, generation: int, live: dict | None) -> list[dict] | None:
        """The 30 most volatile eligible instruments, as 4.3 ranked them."""
        snap = self.ctx.bars_stats.symbol_snapshot
        if (
            not live
            or live["generation"] != generation
            or snap is None
            or snap.generation != generation
        ):
            return None
        indices = live["eligible_indices"]
        ranking = np.argsort(-np.asarray(live["annual_vol"]))[:30]
        return [
            dict(
                symbol=snap.symbols[indices[j]],
                mid=_number(snap.mid[indices[j]]),
                last_trade=_number(snap.trade[indices[j]]),
                observed_volume=_number(snap.volume[indices[j]]),
                annual_vol=live["annual_vol"][j],
                quote_age_s=float(snap.last_event_ns - snap.quote_time[indices[j]]) / 1e9,
            )
            for j in ranking
        ]

    def network(self) -> dict:
        """``/snapshot.network``: receive counters, the arrival-rate buckets and the 4.3
        constants. Before the source opens it is exactly 4.3's starting dict."""
        c = self.ctx
        s, bars, mode = c.ingest_stats, c.bars_stats, c.args.ingest
        token = LEGACY_INGEST_TOKEN[mode]
        if not s.opened:
            return {"ready": False, "status": "starting", "ingest": token}
        now = time.time_ns()
        buckets = s.feed_buckets(last=100)
        boundary = now // 100_000_000 * 100
        age = (now - s.last_packet_wall_ns) / 1e9 if s.last_packet_wall_ns else None
        return dict(
            ready=s.ready(),
            status="receiving"
            if age is not None and age < 2
            else "idle"
            if age is not None
            else "waiting",
            ingest=token,
            ingest_mode=mode,
            transport=TRANSPORT,
            wire_protocol=PROTOCOL,
            message_count_scope=MESSAGE_COUNT_SCOPE,
            packets=s.packets,
            received_messages=s.received_messages,
            decoded_messages=s.messages,
            messages_per_second=sum(n for t, n in buckets if boundary - 1000 <= t < boundary),
            sequence_gaps=s.dropped,
            analytics_sequence_gaps=bars.sequence_gaps,
            ring_dropped=s.ring_dropped,
            ring_messages=s.ring_fill(),
            malformed_packets=s.malformed_packets + s.decode_malformed,
            out_of_order_packets=s.out_of_order_packets,
            stale_packets=s.stale_packets + bars.stale_packets,
            error=s.error,
            buckets=buckets,
            gpu_ingest=dict(
                bursts=s.bursts,
                non_udp_frames=s.non_udp_frames,
                drain_host_ms=s.drain_host_ns / 1e6,
                frame_fetch=s.frame_fetch,
                ring_peak=s.ring_peak(),
                payload_path=PAYLOAD_PATHS[mode],
            )
            if mode in GPU_INGEST_MODES
            else None,
        )

    def _engine(self) -> dict:
        """What the Holoscan 5 engine runs: lanes, edges and clocks from the compiled plan,
        termination, lane placement and the scheduling counters."""
        c = self.ctx
        plan = c.plan_summary or {}
        ingest, risk = c.ingest_stats, c.risk_stats
        replay = ingest.replay
        return dict(
            sdk=os.environ.get("HOLOSCAN_SDK_VERSION"),
            lanes=plan.get("lanes"),
            edges=plan.get("edges"),
            clocks=plan.get("clocks"),
            termination=c.status.termination,
            lane_cpus=dict(
                ingest=list(ingest.lane_cpus),
                bars=list(c.bars_stats.lane_cpus),
                risk=list(risk.lane_cpus),
            ),
            risk=dict(
                windows_superseded=risk.windows_superseded,
                gate_waits=risk.gate_waits,
                late_activations=risk.late_activations,
            ),
            source=dict(retries=ingest.retries, held=ingest.held),
            replay=dict(replay) if replay is not None else None,
        )

    def start_capture(self, payload: dict) -> dict:
        """``POST /capture``: needs a full return window, as 4.3 did. ValueError otherwise."""
        c = self.ctx
        seconds = float(payload.get("seconds", 120))
        if c.tap.active:
            raise ValueError("finish the previous capture before starting another")
        live = c.results.live
        if not live or live["bars"] < c.settings.window_bars:
            raise ValueError("wait for a full return window before capturing")
        return c.tap.start(seconds, c.directory.generation, c.args.analytics_backend)

    def finish_capture(self) -> dict:
        """``POST /capture/finish``: the capture's result plus 4.3's run description
        (``compare_runs.py`` reads ``ingest``). ValueError if no capture was started."""
        c = self.ctx
        result = c.tap.finish()
        result["ingest"] = LEGACY_INGEST_TOKEN[c.args.ingest]
        result["settings"] = c.settings.as_params()
        result["wire_protocol"] = PROTOCOL
        result["minimum_update_interval_seconds"] = c.args.interval
        return result
