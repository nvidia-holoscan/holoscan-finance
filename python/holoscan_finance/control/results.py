# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Published analytics: the dashboard's ``live``, ``comparison``, history and timings.

``CallbackSinkOp`` hands each ``RiskUpdate`` to ``on_risk`` (analytics lane) and each
``RiskComparison`` to ``on_comparison`` (crosscheck lane), which releases the cross-check
gate only after the comparison is published, so every validated GPU result has its CPU
pair. Publication is on the risk lane right after compute, and ``receive_to_result_ms``
is the publication wall time minus the receive stamp of the packet that closed the newest
bar. ``lock`` covers dict building only; the generation filter reads the directory's knob,
never its lock (the lock order in docs/ARCHITECTURE.md).
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..nyq.manifest import FeedDirectory
    from ..payloads import RiskComparison, RiskUpdate
    from ..settings import BarSettings
    from ..telemetry.capture import CaptureTap
    from .directory import ReplayDirectory
    from .gate import CrossCheckGate

logger = logging.getLogger(__name__)

# Keys of a comparison failure's per-backend detail.
_FAILURE_KEYS = (
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


class ResultsStore:
    """The analytics as the dashboard shows them, written by the two sink callbacks.

    ``backend`` (``compare``, ``gpu`` or ``cpu``) decides how an active capture records a
    result; ``interval_s`` and ``settings`` describe the run alongside its results."""

    lock: threading.RLock
    live: dict | None
    comparison: dict | None
    comparisons: int
    history: deque  # [bar_epoch_s, median_vol, mean_corr], maxlen 300
    timings: deque  # [cpu_ms, gpu_ms] of usable incremental comparisons, maxlen 120
    comparison_failure: dict | None
    error: str | None
    analytics_max_ms: float
    published: int
    received_risk: int  # every on_risk call, filtered ones included (drained predicate)
    received_comparisons: int  # every on_comparison call
    latencies_ms: deque  # receive_to_result_ms of published results, maxlen 100_000
    disabled: bool  # set by an analytics failure: nothing more is published

    def __init__(
        self,
        *,
        backend: str,
        interval_s: float,
        settings: BarSettings,
        directory: ReplayDirectory,
        tap: CaptureTap,
        gate: CrossCheckGate | None = None,
    ) -> None:
        self.backend, self.interval_s, self.settings = backend, interval_s, settings
        self.directory, self.tap, self.gate = directory, tap, gate
        self.lock = threading.RLock()
        self.live = self.comparison = self.comparison_failure = self.error = None
        self.comparisons = self.published = 0
        self.received_risk = self.received_comparisons = 0
        self.analytics_max_ms = 0.0
        self.history = deque(maxlen=300)
        self.timings = deque(maxlen=120)
        self.latencies_ms = deque(maxlen=100_000)
        self.disabled = False

    def on_risk(self, update: RiskUpdate) -> None:
        """Publish a risk result of the current generation; never raises.

        A failure never releases the gate: a validated update's comparison still comes and
        releases it in ``on_comparison``; an earlier release would free risk's next claim
        while this comparison is still in flight."""
        try:
            with self.lock:
                self.received_risk += 1
                # Under our lock: a reset_generation listener that already ran has also
                # moved the knob, so a result of the old generation can never follow it.
                if self.disabled or update.window.generation != self.directory.generation:
                    return
                result = self._publish_risk(update)
                capture = self.tap.capture
            # After our lock (dict building only): record_single copies the window to the
            # host and hashes it. LatencyCapture has its own lock.
            if capture is not None:
                if self.backend == "compare":
                    capture.record_gpu(_work(update), result)
                else:
                    capture.record_single(_work(update), result, update.result)
        except Exception as exc:
            self._fail(f"Analytics failed: {exc}")

    def _publish_risk(self, update: RiskUpdate) -> dict:
        w, original = update.window, update.result
        result = {k: v for k, v in original.items() if k not in ("matrix", "covariance_probes")}
        now = time.time_ns()
        result.update(
            timestamp_ns=int(w.times[-1]),
            generation=w.generation,
            published_wall_ns=now,
            receive_to_result_ms=(now - w.recv_ns) / 1e6,
        )
        # Built before anything is published, so a malformed result publishes nothing.
        point = [
            int(w.times[-1]) // 10**9,
            float(np.median(result["annual_vol"])) if result["annual_vol"] else None,
            result["mean_correlation"],
        ]
        self.analytics_max_ms = max(self.analytics_max_ms, result.get("milliseconds") or 0.0)
        self.live = result
        self.history.append(point)
        self.published += 1
        self.latencies_ms.append(result["receive_to_result_ms"])
        return result

    def on_comparison(self, cmp: RiskComparison) -> None:
        """Publish a CPU/GPU comparison, then release the gate; never raises."""
        try:
            with self.lock:
                self.received_comparisons += 1
                if cmp.error is not None:
                    self._fail(f"Analytics failed: {cmp.error}")
                elif not self.disabled and (
                    cmp.update.window.generation == self.directory.generation
                ):
                    self._publish_comparison(cmp)
        except Exception as exc:
            self._fail(f"Analytics failed: {exc}")
        finally:
            # Exactly once per comparison: a second release could free the claim risk took
            # for its next window while that comparison is still in flight.
            if self.gate is not None:
                self.gate.release()

    def _publish_comparison(self, cmp: RiskComparison) -> None:
        w, cpu, gpu = cmp.update.window, cmp.cpu, cmp.update.result
        ok, matched, usable = cmp.agreement, cmp.matched_work, cmp.usable
        capture = self.tap.capture
        if capture is not None:
            capture.record_cpu(_work(cmp.update), cpu, gpu, ok and matched)
        now = time.time_ns()
        self.comparisons += 1
        self.comparison = dict(
            identifier=self.comparisons,
            generation=w.generation,
            timestamp_ns=int(w.times[-1]),
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
            cpu_receive_to_result_ms=(cpu["finished_wall_ns"] - w.recv_ns) / 1e6,
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
                "cpu": {k: cpu[k] for k in _FAILURE_KEYS},
                "gpu": {k: gpu[k] for k in _FAILURE_KEYS},
            }
            self.error = "CPU/GPU comparison failed; inspect /snapshot before using timings"

    def _fail(self, msg: str) -> None:
        # Publication stops at the first exception.
        logger.error("%s", msg)
        with self.lock:
            self.error = msg
            self.disabled = True

    def reset_generation(self, d: FeedDirectory) -> None:
        """Directory listener: results of the previous generation are gone."""
        with self.lock:
            self.live = self.comparison = None
            self.history.clear()
            self.timings.clear()

    def set_error(self, msg: str) -> None:
        with self.lock:
            self.error = msg

    def rolling_timing(self) -> dict | None:
        """Medians of the matched incremental comparisons; None before the first."""
        with self.lock:
            if not self.timings:
                return None
            timing = np.asarray(self.timings)
        cpu, gpu = float(np.median(timing[:, 0])), float(np.median(timing[:, 1]))
        return dict(samples=len(timing), cpu_median_ms=cpu, gpu_median_ms=gpu, speedup=cpu / gpu)

    def view(self) -> dict:
        """Copies of everything ``/snapshot`` shows. Published dicts are replaced, never
        mutated, so they are shared as they are."""
        with self.lock:
            return dict(
                error=self.error,
                comparison_failure=self.comparison_failure,
                live=self.live,
                comparison=self.comparison,
                comparisons=self.comparisons,
                history=list(self.history),
                rolling_timing=self.rolling_timing(),
            )


def _work(update: RiskUpdate) -> dict:
    """The work dict ``LatencyCapture`` records from. ``rows`` was materialized (and
    cached on the window) by the risk operator, so this costs no GPU work."""
    w = update.window
    work = dict(rows=w.rows, times=w.times, generation=w.generation, recv_ns=w.recv_ns)
    if update.rows_host is not None:
        work["rows_host"] = update.rows_host
    return work
