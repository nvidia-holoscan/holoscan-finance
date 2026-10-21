# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rolling volatility, covariance and correlation of the newest bar window: the processor of
``RollingRiskOp``.

A one-window slot keeps only the newest window, and ``poll`` analyzes it at most once per
``min_interval_ns`` and only while no CPU comparison is in flight: 4.3's run-on-arrival
timing (``MessageAvailable`` + ``PeriodicCondition``) with a 1 ms poll measured a 0.99 ms
pick-up (J1), where a 50 ms clock added 28 ms at p50. Inline it is 4.3
``GpuRollingRiskOp.compute`` (a throwaway engine per generation, host-variance ranking of
validated windows); with ``executor="process"`` it is the host paths' 4.3 ``RiskOp``
(``gpu_window``/``cpu_window`` in one spawned worker, off the decoder's GIL). Never imports
Holoscan; CuPy only inside the inline GPU path.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ProcessPoolExecutor
from typing import TYPE_CHECKING

import numpy as np

from ..marketdata.covariance import CovarianceEngine, gpu_name
from ..payloads import BarWindow, RiskUpdate
from ..telemetry.stats import RiskStats
from . import workers

if TYPE_CHECKING:
    from ..control.gate import CrossCheckGate

BACKENDS = ("gpu", "cpu")
EXECUTORS = ("inline", "process")


class RollingRisk:
    """The newest window, at most every ``min_interval_ns``, one validated window in flight.

    With ``cross_check`` each analysis first claims ``gate`` (``try_claim``); a claimed
    window is ``validated`` and carries a host copy of its rows for the CPU comparison, and
    while a claim is open no window is analyzed (``ResultsStore`` releases it after
    publishing the comparison). ``host_variance`` ranks every inline analysis by the NumPy
    variance of a host copy, so a NumPy reference picks identical instruments.
    """

    def __init__(
        self,
        *,
        backend: str = "gpu",
        executor: str = "inline",
        bar_seconds: int = 5,
        gpu_device: int = 0,
        min_interval_ns: int = 50_000_000,
        cross_check: bool = False,
        gate: CrossCheckGate | None = None,
        host_variance: bool = False,
        cpu_threads: int = 0,
        full_window_bars: int = 60,
        worker_cpus: Sequence[int] = (),
        stats: RiskStats | None = None,
        clock: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if backend not in BACKENDS or executor not in EXECUTORS:
            raise ValueError(f"backend must be one of {BACKENDS}, executor one of {EXECUTORS}")
        if cross_check and gate is None:
            raise ValueError("cross_check requires a CrossCheckGate")
        if host_variance and executor == "process":
            raise ValueError("host_variance needs the inline executor (workers rank on their own)")
        self.backend, self.executor = backend, executor
        self.stats = stats if stats is not None else RiskStats()
        self._bar_seconds, self._gpu_device = bar_seconds, gpu_device
        self._min_interval_ns, self._clock = min_interval_ns, clock
        self._cross_check, self._gate, self._host_variance = cross_check, gate, host_variance
        self._cpu_threads, self._full_window_bars = cpu_threads, full_window_bars
        self._worker_cpus = list(worker_cpus)
        self._slot: BarWindow | None = None
        self._last_start: int | None = None
        self._engine: CovarianceEngine | None = None
        self._generation: int | None = None
        self._pool: ProcessPoolExecutor | None = None
        self._future: Future | None = None
        self._job: tuple[BarWindow, bool, int, np.ndarray] | None = None
        self._failed = False

    def start(self) -> None:
        """Pre-spawn the worker (process), or activate the device on this thread (inline GPU)."""
        if self.executor == "process":
            self._pool = workers.start_worker_pool(self._worker_cpus)
        elif self.backend == "gpu":
            import cupy as cp

            cp.cuda.Device(self._gpu_device).use()
            # The first query reads ~100 driver attributes under the GIL: never per window.
            gpu_name(cp, self._gpu_device)

    def stop(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None
        self._future = self._job = self._engine = None
        self.stats.busy = False

    def offer(self, window: BarWindow) -> None:
        """Make ``window`` the next one to analyze; a waiting older window is superseded."""
        self.stats.windows_received += 1
        if self._slot is not None:
            self.stats.windows_superseded += 1
        self._slot = window

    def poll(self) -> RiskUpdate | None:
        """A finished analysis, or None. Inline, the analysis runs in this call; the process
        executor collects a finished future and submits the next window in the same call."""
        if self.executor == "process":
            return self._poll_process()
        job = self._take()
        return None if job is None else self._analyze(*job)

    def _take(self) -> tuple[BarWindow, bool, int] | None:
        """``(window, validated, started)`` once the interval has passed and the gate is free."""
        if self._slot is None:
            return None
        now = self._clock()
        if self._last_start is not None and now - self._last_start < self._min_interval_ns:
            return None
        if self._gate is not None and not self._gate.ready():
            self.stats.gate_waits += 1
            return None
        window, self._slot, self._last_start = self._slot, None, now
        validated = self._gate.try_claim() if self._cross_check else False
        return window, validated, now

    def _analyze(self, window: BarWindow, validated: bool, started: int) -> RiskUpdate:
        if self.backend == "gpu":
            import cupy as cp

            with cp.cuda.Device(self._gpu_device):
                result, rows_host = self._update(window, validated)
                result["gpu_name"] = gpu_name(cp, self._gpu_device)
        else:
            result, rows_host = self._update(window, validated)
        result["finished_wall_ns"] = time.time_ns()
        return self._finish(window, result, rows_host, validated, started)

    def _update(self, window: BarWindow, validated: bool) -> tuple[dict, np.ndarray | None]:
        rows = window.rows
        rows_host = observed = None
        if validated or self._host_variance:
            rows_host = _host(rows)
            observed = np.var(rows_host, axis=0, ddof=1)
        if self._engine is None or window.generation != self._generation:
            # First window of a generation: a throwaway engine compiles the kernels and
            # allocates workspaces, so the kept engine's timings exclude that cost.
            self._new_engine().update(rows, window.times, observed_variance=observed)
            self._engine, self._generation = self._new_engine(), window.generation
        result = self._engine.update(rows, window.times, observed_variance=observed)
        return result, rows_host

    def _new_engine(self) -> CovarianceEngine:
        return CovarianceEngine(
            self.backend, bar_seconds=self._bar_seconds, gpu_device=self._gpu_device
        )

    def _poll_process(self) -> RiskUpdate | None:
        update = None
        if self._future is not None:
            if not self._future.done():
                return None
            update = self._collect()
        if not self._failed and (job := self._take()) is not None:
            self._submit(*job)
        return update

    def _submit(self, window: BarWindow, validated: bool, started: int) -> None:
        rows, times, generation = _host(window.rows), window.times, window.generation
        if self.backend == "gpu":
            future = self._pool.submit(
                workers.gpu_window, rows, times, generation, self._bar_seconds, self._gpu_device
            )
        else:
            future = self._pool.submit(
                workers.cpu_window,
                rows,
                times,
                generation,
                self._bar_seconds,
                self._cpu_threads,
                self._full_window_bars,
            )
        self._future, self._job = future, (window, validated, started, rows)
        self.stats.busy = True

    def _collect(self) -> RiskUpdate | None:
        future, (window, validated, started, rows) = self._future, self._job
        self._future = self._job = None
        self.stats.busy = False
        try:
            result = future.result()
        except Exception as exc:
            # 4.3 RiskOp: record the error and stop analyzing. No comparison will come for
            # this window, so its claim is void.
            self.stats.error = f"Analytics failed: {exc}"
            self._failed = True
            if validated:
                self._gate.release()
            return None
        return self._finish(window, result, rows if validated else None, validated, started)

    def _finish(
        self,
        window: BarWindow,
        result: dict,
        rows_host: np.ndarray | None,
        validated: bool,
        started: int,
    ) -> RiskUpdate:
        st = self.stats
        ms = (self._clock() - started) / 1e6
        st.analyzed += 1
        st.last_completed_bars, st.last_generation = window.completed_bars, window.generation
        st.last_window_key = (window.generation, window.completed_bars)
        st.last_ms, st.max_ms = ms, max(st.max_ms, ms)
        return RiskUpdate(window, result, rows_host, self.backend, validated)


def _host(rows: object) -> np.ndarray:
    """Window rows on the host (a D2H copy of a device window)."""
    return rows if isinstance(rows, np.ndarray) else rows.get()
