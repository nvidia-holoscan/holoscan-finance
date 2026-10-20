# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact empirical latency percentiles for a bounded, explicitly timed capture.

``percentiles`` and ``LatencyCapture`` are verbatim from 4.3 ``equities.capture``: every
packet in the interval is counted by its receive stamp and every completed calculation is
kept, with no rolling averages or HTTP sampling, so the published receive-to-result
percentiles stay comparable. ``CaptureTap`` holds the active capture for the decode sites
(``record_*``), ``ResultsStore`` and the HTTP endpoints (was ``EquityState.capture``).
"""

from __future__ import annotations

import hashlib
import threading
import time
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..marketdata.gpu_bars import GpuReplayBatch
    from ..marketdata.records import ReplayBatch
    from ..nyq.manifest import FeedDirectory


def percentiles(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return {"samples": 0, "p50": None, "p99": None, "max": None}
    p50, p99 = np.percentile(a, [50, 99])
    return dict(samples=len(a), p50=float(p50), p99=float(p99), max=float(a.max()))


class LatencyCapture:
    """Count every packet and retain every completed calculation in the interval.

    Membership uses the receiver timestamp, including for the packet triggering an
    analytics job. No rolling-percentile averaging or HTTP sampling is involved.
    More than ``max_packets`` packets marks the capture invalid.
    """

    def __init__(
        self,
        seconds,
        generation,
        *,
        start_ns=None,
        max_packets=3_000_000,
        backend="compare",
    ):
        if not 1 <= seconds <= 600:
            raise ValueError("capture duration must be 1..600 seconds")
        self.start_ns = time.time_ns() if start_ns is None else start_ns
        if backend not in ("compare", "cpu", "gpu"):
            raise ValueError("backend must be compare, cpu or gpu")
        self.backend = backend
        self.packet_count = self.wire_messages = 0
        self.end_ns = self.start_ns + int(seconds * 1e9)
        self.generation = generation
        self.max_packets = max_packets
        self.lock = threading.Lock()
        self.gpu = []
        self.cpu = []
        self.single = []
        self.events = self.sequence_gaps = self.reordered = 0
        self.expected_sequence = None
        self.overflow = self.generation_changed = self.closed = False

    def includes(self, recv_ns):
        return not self.closed and self.start_ns <= recv_ns < self.end_ns

    def record_packet(self, batch):
        if not self.includes(batch.recv_wall_ns):
            return
        with self.lock:
            if self.closed:
                return
            if batch.generation != self.generation:
                self.generation_changed = True
                return
            count = len(batch.events)
            self.events += count
            self.packet_count += 1
            self.wire_messages += batch.sequence_count
            if self.expected_sequence is not None:
                self.sequence_gaps += max(0, batch.sequence - self.expected_sequence)
                self.reordered += int(batch.sequence < self.expected_sequence)
            self.expected_sequence = max(
                self.expected_sequence or 0, batch.sequence + batch.sequence_count
            )
            if self.packet_count > self.max_packets:
                self.overflow = True

    def record_gpu(self, work, result):
        if not self.includes(work["recv_ns"]):
            return
        with self.lock:
            if not self.closed:
                self.gpu.append(
                    dict(
                        bar_ns=int(work["times"][-1]),
                        generation=work["generation"],
                        eligible=result["eligible"],
                        method=result["method"],
                        receive_to_result_ms=result["receive_to_result_ms"],
                        analytics_ms=result["milliseconds"],
                    )
                )

    def record_cpu(self, work, cpu, gpu, valid):
        if not self.includes(work["recv_ns"]):
            return
        with self.lock:
            if not self.closed:
                self.cpu.append(
                    dict(
                        bar_ns=int(work["times"][-1]),
                        generation=work["generation"],
                        valid=valid,
                        eligible=cpu["eligible"],
                        method=cpu["method"],
                        receive_to_result_ms=(cpu["finished_wall_ns"] - work["recv_ns"]) / 1e6,
                        analytics_ms=cpu["milliseconds"],
                        gpu_analytics_ms=gpu["milliseconds"],
                        threads=cpu["threads"],
                    )
                )

    def status(self):
        with self.lock:
            return dict(
                start_ns=self.start_ns,
                end_ns=self.end_ns,
                generation=self.generation,
                packets=self.packet_count,
                calculations=len(self.gpu) + len(self.single),
                analytics_backend=self.backend,
                closed=self.closed,
                overflow=self.overflow,
            )

    def record_single(self, work, published, result):
        """Keep validation outputs for comparison after separately executed runs.

        Input hashing and capture serialization are outside the analytics timer.
        """
        if not self.includes(work["recv_ns"]):
            return
        digest = hashlib.sha256(np.asarray(work["times"], dtype="<i8").tobytes())
        rows = work.get("rows_host", work["rows"])
        rows = rows.get() if hasattr(rows, "get") and not isinstance(rows, np.ndarray) else rows
        digest.update(np.asarray(rows, dtype="<f8").tobytes())
        with self.lock:
            if self.closed:
                return
            self.single.append(
                dict(
                    bar_ns=int(work["times"][-1]),
                    backend=result["backend"],
                    input_sha256=digest.hexdigest(),
                    analytics_ms=result["milliseconds"],
                    receive_to_result_ms=published["receive_to_result_ms"],
                    **{
                        k: result[k]
                        for k in (
                            "method",
                            "added_bars",
                            "expired_bars",
                            "bars",
                            "universe",
                            "eligible",
                        )
                    },
                    validation={
                        k: result[k]
                        for k in (
                            "eligible_indices",
                            "annual_vol",
                            "heatmap_indices",
                            "heatmap",
                            "covariance_probes",
                        )
                    },
                )
            )

    def finish(self, *, now_ns=None):
        now = time.time_ns() if now_ns is None else now_ns
        if now < self.end_ns:
            raise ValueError("capture is still running")
        with self.lock:
            self.closed = True
            gpu, cpu = list(self.gpu), list(self.cpu)
            single = list(self.single)
        if self.backend != "compare":
            seconds = (self.end_ns - self.start_ns) / 1e9
            valid = bool(
                self.packet_count
                and single
                and all(r["backend"] == self.backend and r["eligible"] >= 2 for r in single)
                and not (
                    self.overflow or self.generation_changed or self.sequence_gaps or self.reordered
                )
            )
            return dict(
                start_ns=self.start_ns,
                end_ns=self.end_ns,
                finished_ns=now,
                seconds=seconds,
                generation=self.generation,
                valid=valid,
                analytics_backend=self.backend,
                validation_scope="Transport/capture checks only; compare input hashes and numerical outputs after both separate runs.",
                observations=self.events,
                packets=self.packet_count,
                wire_messages=self.wire_messages,
                received_observations_per_second=self.events / seconds,
                sequence_gaps=self.sequence_gaps,
                reordered_packets=self.reordered,
                overflow=self.overflow,
                generation_changed=self.generation_changed,
                percentile_method="numpy.percentile, linear interpolation; all captured samples",
                analytics_ms=percentiles([r["analytics_ms"] for r in single]),
                receive_to_result_ms=percentiles([r["receive_to_result_ms"] for r in single]),
                single_calculations=single,
            )
        matching = {(c["generation"], c["bar_ns"]) for c in cpu}
        complete = all((g["generation"], g["bar_ns"]) in matching for g in gpu)
        valid = bool(
            self.packet_count
            and gpu
            and cpu
            and complete
            and not (
                self.overflow or self.generation_changed or self.sequence_gaps or self.reordered
            )
            and all(c["valid"] and c["eligible"] >= 2 for c in cpu)
        )
        seconds = (self.end_ns - self.start_ns) / 1e9
        result = dict(
            start_ns=self.start_ns,
            end_ns=self.end_ns,
            finished_ns=now,
            seconds=seconds,
            generation=self.generation,
            valid=valid,
            observations=self.events,
            packets=self.packet_count,
            wire_messages=self.wire_messages,
            received_observations_per_second=self.events / seconds,
            sequence_gaps=self.sequence_gaps,
            reordered_packets=self.reordered,
            overflow=self.overflow,
            generation_changed=self.generation_changed,
            all_gpu_results_compared=complete,
            percentile_method="numpy.percentile, linear interpolation; all captured samples",
            gpu_receive_to_result_ms=percentiles([g["receive_to_result_ms"] for g in gpu]),
            gpu_analytics_ms=percentiles([g["analytics_ms"] for g in gpu]),
            cpu_receive_to_result_ms=percentiles([c["receive_to_result_ms"] for c in cpu]),
            cpu_analytics_ms=percentiles([c["analytics_ms"] for c in cpu]),
            gpu_calculations=gpu,
            cpu_comparisons=cpu,
        )
        matched = [c for c in cpu if c["valid"] and c["eligible"] >= 2]
        result["matched_analytics_speedup"] = (
            float(
                np.median([c["analytics_ms"] for c in matched])
                / np.median([c["gpu_analytics_ms"] for c in matched])
            )
            if matched and valid
            else None
        )
        return result


class CaptureTap:
    """Thread-safe holder for the active capture."""

    capture: LatencyCapture | None

    def __init__(self) -> None:
        raise NotImplementedError

    @property
    def active(self) -> bool:
        """A capture exists and is not closed."""
        raise NotImplementedError

    def start(self, seconds: float, generation: int, backend: str) -> dict:
        """Start a new capture (ValueError while one is open); returns its status."""
        raise NotImplementedError

    def finish(self) -> dict:
        """Close the capture and return its result (ValueError if none was started)."""
        raise NotImplementedError

    def status(self) -> dict | None:
        raise NotImplementedError

    def on_generation(self, d: FeedDirectory) -> None:
        """Directory listener: a new generation during the capture interval invalidates it."""
        raise NotImplementedError

    def record_host_batches(self, batches: list[ReplayBatch]) -> None:
        raise NotImplementedError

    def record_gpu_batch(self, batch: GpuReplayBatch) -> None:
        """Count every packet of a device batch (the 4.3 ``_PacketView`` loop)."""
        raise NotImplementedError
