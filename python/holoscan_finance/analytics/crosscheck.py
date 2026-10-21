# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The CPU recomputation of validated GPU windows: the processor of ``RiskCrossCheckOp``.

Each validated ``RiskUpdate`` carries a host copy of its window; ``cpu_window`` recomputes it
in one spawned, pinned, BLAS-tuned worker, and the comparison applies
``agreement.compare_results``. The dashboard's live CPU/GPU speedup comes from these pairs
(separate CPU-only and GPU-only runs of the same workers give the separate-runs speedup,
docs/METHODOLOGY.md). The gate keeps one update in flight; the queue stays FIFO anyway.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from concurrent.futures import BrokenExecutor, Future, ProcessPoolExecutor

from ..payloads import RiskComparison, RiskUpdate
from ..telemetry.stats import RiskStats
from . import workers
from .agreement import compare_results


class CpuCrossCheck:
    """FIFO of validated updates, one ``cpu_window`` job at a time."""

    def __init__(
        self,
        *,
        bar_seconds: int = 5,
        cpu_threads: int = 0,
        full_window_bars: int = 60,
        worker_cpus: Sequence[int] = (),
        stats: RiskStats | None = None,
    ) -> None:
        self.stats = stats if stats is not None else RiskStats()
        self._bar_seconds, self._cpu_threads = bar_seconds, cpu_threads
        self._full_window_bars, self._worker_cpus = full_window_bars, list(worker_cpus)
        self._queue: deque[RiskUpdate] = deque()
        self._pool: ProcessPoolExecutor | None = None
        self._future: Future | None = None
        self._running: RiskUpdate | None = None

    def start(self) -> None:
        self._pool = workers.start_worker_pool(self._worker_cpus)

    def stop(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None
        self._queue.clear()
        self._future = self._running = None

    def submit(self, update: RiskUpdate) -> None:
        """Queue a validated update (it must carry ``rows_host``)."""
        if update.rows_host is None:
            raise ValueError("the CPU cross-check needs a validated update with rows_host")
        self._queue.append(update)

    def poll(self) -> RiskComparison | None:
        """The finished comparison, if any; then the next queued update goes to the worker."""
        comparison = None
        if self._future is not None:
            if not self._future.done():
                return None
            comparison = self._collect()
        if self._queue:
            update = self._running = self._queue.popleft()
            w = update.window
            try:
                self._future = self._pool.submit(
                    workers.cpu_window,
                    update.rows_host,
                    w.times,
                    w.generation,
                    self._bar_seconds,
                    self._cpu_threads,
                    self._full_window_bars,
                )
            except BrokenExecutor as exc:
                # The worker died while idle: an error comparison at the next poll, as for a
                # failed job.
                self._future = Future()
                self._future.set_exception(exc)
            self.stats.crosschecks_submitted += 1
        return comparison

    def _collect(self) -> RiskComparison:
        future, update = self._future, self._running
        self._future = self._running = None
        self.stats.crosschecks_done += 1
        try:
            cpu = future.result()
        except Exception as exc:
            # Published as an error comparison: ResultsStore records it and releases the gate.
            return RiskComparison(update, None, False, False, False, f"{type(exc).__name__}: {exc}")
        return RiskComparison(update, cpu, *compare_results(cpu, update.result))
