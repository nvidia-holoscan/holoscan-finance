# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The CPU recomputation of validated GPU windows: the processor of ``RiskCrossCheckOp``.

Each validated ``RiskUpdate`` carries a host copy of its window; ``cpu_window`` recomputes it
in one spawned, pinned, BLAS-tuned worker, and the comparison applies
``agreement.compare_results``. As ``RollingRisk`` does for its own worker, a window that
slides from the one sent last travels as its new bars only (the full window after a gap, a
new generation or a lost worker state), so the worker's engine copies in only the new bars,
as the GPU engine does. The dashboard's live CPU/GPU speedup comes from these pairs
(separate CPU-only and GPU-only runs of the same workers give the separate-runs speedup,
docs/METHODOLOGY.md). The gate keeps one update in flight; the queue stays FIFO anyway.
``prepare`` tunes the worker's BLAS threads for a universe when its manifest arrives
(``tune_cpu``), long before its first window, or in ``RiskCrossCheckOp.start`` when the
universe is known before the graph runs (replay), so no GPU result waits for the tuning.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from concurrent.futures import BrokenExecutor, Future, ProcessPoolExecutor

import numpy as np

from ..marketdata.covariance import slide
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
        self._sent: tuple[int, np.ndarray] | None = None  # the worker's window: generation, times
        self._tuning = workers.UniverseTuning(
            "CPU cross-check worker", full_window_bars, bar_seconds, cpu_threads
        )

    def start(self) -> None:
        self._pool = workers.start_worker_pool(self._worker_cpus)

    def stop(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None
        self._queue.clear()
        self._future = self._running = self._sent = None
        self._tuning.reset()

    @property
    def tuning(self) -> dict | None:
        """The worker's latest ``tune_cpu`` result."""
        return self._tuning.result

    @property
    def tuning_pending(self) -> bool:
        """A ``prepare`` tuning has not been collected by ``poll`` yet."""
        return self._tuning.pending

    def prepare(self, width: int, *, wait: bool = False) -> None:
        """Tune the worker for a universe of ``width`` instruments before its first window,
        once per width, when the worker is idle (call it again until then); ``poll``
        collects it. A validated update that arrives meanwhile queues behind it. ``wait``:
        return once the tuning is done (``RiskCrossCheckOp.start``, when the universe is
        known before the graph runs)."""
        if self._pool is not None and self._future is None and not self._queue:
            self._tuning.submit(self._pool, width)
        if wait:
            self._tuning.wait()

    def submit(self, update: RiskUpdate) -> None:
        """Queue a validated update (it must carry ``rows_host``)."""
        if update.rows_host is None:
            raise ValueError("the CPU cross-check needs a validated update with rows_host")
        self._queue.append(update)

    def poll(self) -> RiskComparison | None:
        """The finished comparison, if any; then the next queued update goes to the worker."""
        if not self._tuning.collect():
            return None
        comparison = None
        if self._future is not None:
            if not self._future.done():
                return None
            comparison = self._collect()
        if comparison is None and self._future is not None:  # resubmitted in full
            return None
        if self._queue:
            self._running = self._queue.popleft()
            self._submit(self._running)
            self.stats.crosschecks_submitted += 1
        return comparison

    def _submit(self, update: RiskUpdate, *, full: bool = False) -> None:
        """The validated window's new bars (all of them unless it slides from the window
        sent last) to the worker."""
        w = update.window
        new = len(w.times)
        if not full and self._sent is not None and self._sent[0] == w.generation:
            step = slide(self._sent[1], w.times)
            new = len(w.times) if step is None else step[1]
        self._sent = (w.generation, w.times)
        try:
            self._future = self._pool.submit(
                workers.cpu_window,
                update.rows_host[len(w.times) - new :],
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

    def _collect(self) -> RiskComparison | None:
        future, update = self._future, self._running
        self._future = None
        try:
            cpu = future.result()
        except workers.WindowUnknown:
            # The worker lost its window (a new worker process): send it in full.
            self._submit(update, full=True)
            return None
        except Exception as exc:
            self._running = None
            self.stats.crosschecks_done += 1
            # Published as an error comparison: ResultsStore records it and releases the gate.
            return RiskComparison(update, None, False, False, False, f"{type(exc).__name__}: {exc}")
        self._running = None
        self.stats.crosschecks_done += 1
        return RiskComparison(update, cpu, *compare_results(cpu, update.result))
