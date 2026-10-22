# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Spawn-worker functions: the CPU comparator and the separate-process GPU worker.

``cpu_window`` and ``gpu_window`` each keep one persistent incremental engine per process,
reset on a new generation, so a worker pays the full rebuild only on a new generation. Each
keeps the window it analyzed last, so a caller may send only the new bars of a window that
slides from it (``WindowUnknown`` asks for the full window); the engine then copies in only
those bars.
The CPU worker tunes its BLAS thread count over {1, 2, 4, 8, all} once per universe width
(``tune_cpu``: a run of one-bar slides of a full window of that width, the steady workload,
timed in a second pass; the work is a rank-k update of the whole ``(N+1) x (N+1)`` matrix,
so it does not depend on the returns). It keeps the fewest threads within ``TUNE_MARGIN``
of the fastest: extra threads that do not clearly pay cost far more once the pipeline's own
threads compete for the cores. The tuned limit stays in force for the process, so a window
pays no thread-pool calls.
The callers send ``tune_cpu`` when a manifest announces the universe, before its first
window (in their ``start`` when the universe is known before the graph runs);
``cpu_window`` tunes itself only for a width it was not tuned for (the separate-runs
CPU/GPU method, docs/METHODOLOGY.md). Workers run in spawned processes so BLAS
and CUDA work never holds the pipeline's GIL (in-process, 30% decoder duty added 45% to the
engine time); ``start_worker_pool`` starts one before data flows. Its initializer leaves
SIGINT and SIGTERM to the parent, pins the worker (``pin_worker``) and ends it with the
parent.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import signal
import threading
import time
from collections.abc import Sequence
from concurrent.futures import BrokenExecutor, Future, ProcessPoolExecutor
from concurrent.futures import wait as futures_wait

import numpy as np

from ..ingest.affinity import pin_worker
from ..marketdata.covariance import CovarianceEngine, gpu_name, slide

__all__ = [
    "ARRAY_KEYS",
    "UniverseTuning",
    "WindowUnknown",
    "cpu_window",
    "gpu_window",
    "pin_worker",
    "ping",
    "reset",
    "start_worker_pool",
    "tune_cpu",
]

logger = logging.getLogger(__name__)

# The tuning keeps the fewest BLAS threads whose steady one-bar slide is within this fraction
# of the fastest.
TUNE_MARGIN = 0.10
TUNE_SLIDES = 8  # one-bar slides timed per thread count; the first two warm up

# Result vectors returned as NumPy arrays with ``arrays=True`` (cheaper to pickle).
ARRAY_KEYS = ("eligible_indices", "annual_vol", "heatmap_indices", "heatmap", "covariance_probes")


class WindowUnknown(LookupError):
    """Only new bars arrived, but this worker does not hold the window they extend."""


def ping() -> int:
    """The worker's pid; submitting it once starts the worker before data flows."""
    return os.getpid()


def start_worker_pool(
    worker_cpus: Sequence[int] = (), *, gpu_device: int | None = None
) -> ProcessPoolExecutor:
    """One spawned worker pinned to ``worker_cpus`` (``[]`` inherits), already running.

    Spawn, never fork: the parent runs runtime and CUDA threads. The ``ping`` round trip
    takes the interpreter start (and the re-import of an app script's top-level imports)
    out of the first window's latency. With ``gpu_device`` (a ``gpu_window`` worker) the
    worker also loads the risk kernels on that device before it reports ready.
    """
    pool = ProcessPoolExecutor(
        max_workers=1,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_init_worker,
        initargs=(list(worker_cpus), gpu_device),
    )
    try:
        pool.submit(ping).result()
    except BaseException:
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    return pool


def _init_worker(cpus: Sequence[int], gpu_device: int | None = None) -> None:
    """Spawn initializer: the parent owns SIGINT and SIGTERM; then ``pin_worker``, and the
    risk kernels on ``gpu_device`` unless it is None.

    A terminal Ctrl-C (or ``timeout``) signals the whole process group. With Python's default
    handlers a busy worker failed its job with ``KeyboardInterrupt`` (a BaseException: the
    compute failed and the run ended FAILED) and an idle one died, breaking the pool. The
    parent's handler stops the run and the operators' ``stop()`` shuts the pool down. A
    worker that ignores SIGTERM must not outlive its parent, so a daemon thread exits it when
    the parent's end of the spawn pipe closes (also after a SIGKILL).
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    pin_worker(cpus)
    parent = multiprocessing.parent_process()
    if parent is not None:  # created after pinning, so it inherits the worker's mask
        threading.Thread(
            target=_exit_with_parent, args=(parent,), name="hf-worker-parent", daemon=True
        ).start()
    if gpu_device is not None:
        from ..marketdata.warmup import warm_risk_kernels

        warm_risk_kernels(gpu_device)


def _exit_with_parent(parent: multiprocessing.process.BaseProcess) -> None:
    parent.join()  # returns once the parent died, also by SIGKILL
    os._exit(1)


_cpu_engine = None
_cpu_generation = None
_cpu_threads = None
_cpu_tuned_bars = 0
_cpu_tuned_width = None  # the universe width the thread limit was tuned for
_cpu_tunings = 0  # tunings in this process
_cpu_thread_trials = None
_cpu_blas = None  # the worker's BLAS libraries, read once at tuning
_cpu_limits = None  # the tuned BLAS thread limit, in force until the next tuning


_windows: dict[str, tuple[int, np.ndarray, tuple]] = {}  # engine -> generation, times, rows


def _window(kind: str, rows, times, generation) -> tuple:
    """The window's per-bar rows: ``rows`` itself when it has a row per time, else the new
    bars of a window sliding from the one this worker analyzed last."""
    times = np.asarray(times, dtype=np.int64)
    if len(rows) == len(times):
        parts = tuple(np.asarray(rows, dtype=np.float64))
    else:
        known = _windows.get(kind)
        step = slide(known[1], times) if known is not None and known[0] == generation else None
        if step is None or step[1] != len(rows):
            raise WindowUnknown(f"{kind} worker holds no window that {len(rows)} new bars extend")
        parts = known[2][step[0] :] + tuple(np.asarray(rows, dtype=np.float64))
    _windows[kind] = (generation, times, parts)
    return parts


def _arrays(result: dict, arrays: bool) -> dict:
    if arrays:
        for key in ARRAY_KEYS:
            result[key] = np.asarray(result[key])
    return result


def tune_cpu(width, full_window_bars=60, bar_seconds=5, threads=0) -> dict:
    """Tune this worker's BLAS thread limit for ``width`` instruments ahead of the universe's
    first window: the median steady one-bar slide of a ``full_window_bars`` window of
    synthetic returns at each of {1, 2, 4, 8, all} threads (only ``threads`` unless 0), in
    a second pass after one that warms up, and the fewest threads within ``TUNE_MARGIN`` of
    the fastest. Returns the tuning."""
    started = time.perf_counter()
    _tune(int(width), int(full_window_bars), bar_seconds, threads)
    return dict(
        width=_cpu_tuned_width,
        threads=_cpu_threads,
        cpu_thread_trials_ms=_cpu_thread_trials,
        cpu_tuned_bars=_cpu_tuned_bars,
        cpu_tunings=_cpu_tunings,
        seconds=time.perf_counter() - started,
    )


def _tune(width: int, bars: int, bar_seconds, threads: int) -> None:
    global _cpu_threads, _cpu_tuned_bars, _cpu_tuned_width, _cpu_tunings, _cpu_thread_trials
    global _cpu_blas, _cpu_limits
    from threadpoolctl import threadpool_info, threadpool_limits

    available = (
        len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    )
    candidates = (
        [threads]
        if threads
        else sorted({1, min(2, available), min(4, available), min(8, available), available})
    )
    if any(x < 1 or x > available for x in candidates):
        raise ValueError(f"CPU threads must be 1..{available} (or 0 for tuning)")
    bars = max(bars, 2)
    rows = np.random.default_rng(0).normal(0.0, 1e-3, (bars + TUNE_SLIDES, width))
    times = np.arange(bars + TUNE_SLIDES, dtype=np.int64) * int(bar_seconds * 10**9)
    timings = {}
    # Two passes over the candidates, the second's times kept: a process's first engine
    # slides faster than every later one, the engine the worker keeps included, so a single
    # pass understated the first candidate (1 thread at 1,000 symbols: 0.74 against 1.19 ms
    # steady) and kept it where 4 threads were 1.8x faster. A forced count needs no pick.
    for _ in range(1 if len(candidates) == 1 else 2):
        for nt in candidates:
            with threadpool_limits(limits=nt, user_api="blas"):
                engine = CovarianceEngine(bar_seconds=bar_seconds, window_bars=bars)
                engine.update(rows[:bars], times[:bars])  # allocates the workspaces, rebuilds
                slides = [
                    engine.update(rows[k : k + bars], times[k : k + bars])["milliseconds"]
                    for k in range(1, TUNE_SLIDES + 1)
                ]
                timings[nt] = float(np.median(slides[2:]))
    fastest = min(timings.values())
    _cpu_threads = min(nt for nt, ms in timings.items() if ms <= (1 + TUNE_MARGIN) * fastest)
    _cpu_tuned_bars, _cpu_tuned_width, _cpu_thread_trials = bars, width, timings
    _cpu_tunings += 1
    _cpu_blas = [
        {"name": p["internal_api"], "version": p["version"]}
        for p in threadpool_info()
        if p["user_api"] == "blas"
    ]
    if _cpu_limits is not None:
        _cpu_limits.restore_original_limits()
    _cpu_limits = threadpool_limits(limits=_cpu_threads, user_api="blas")


class UniverseTuning:
    """A spawned CPU worker's ``tune_cpu``, once per universe width, for the parent that
    owns the worker (``RollingRisk``, ``CpuCrossCheck``): ``submit`` while the worker is
    idle, then ``collect`` before the next window goes out."""

    def __init__(self, name: str, full_window_bars: int, bar_seconds: int, threads: int) -> None:
        self._name, self._args = name, (full_window_bars, bar_seconds, threads)
        self.reset()

    def reset(self) -> None:
        """Forget the worker's tuning (it was shut down)."""
        self._future: Future | None = None
        self._width: int | None = None
        self.result: dict | None = None  # the latest tune_cpu result

    @property
    def pending(self) -> bool:
        """A tuning is in flight or not collected yet."""
        return self._future is not None

    def submit(self, pool: ProcessPoolExecutor, width: int) -> None:
        """Tune the (idle) worker for ``width`` instruments, unless it is tuned for that
        width already or a tuning is in flight."""
        if width == self._width or self._future is not None:
            return
        try:
            self._future = pool.submit(tune_cpu, width, *self._args)
        except BrokenExecutor:
            return  # the next window reports the broken worker
        self._width = width

    def wait(self) -> None:
        """Block until the tuning in flight, if any, finishes, and collect it."""
        if self._future is not None:
            futures_wait([self._future])
        self.collect()

    def collect(self) -> bool:
        """True when no tuning is in flight (a finished one is collected and logged)."""
        if self._future is None:
            return True
        if not self._future.done():
            return False
        future, self._future = self._future, None
        try:
            self.result = t = future.result()
        except Exception as exc:  # cpu_window tunes at the universe's first window instead
            logger.warning("%s tuning failed: %s", self._name, exc)
            return True
        logger.info(
            "%s tuned for %d instruments: %d BLAS threads (%.1f s)",
            self._name,
            t["width"],
            t["threads"],
            t["seconds"],
        )
        return True


def cpu_window(
    rows, times, generation, bar_seconds, threads=0, full_window_bars=60, *, arrays=False
):
    """Spawned worker with persistent incremental state and a tuned BLAS thread count.
    ``rows``: the window, or only its new bars (see the module docstring);
    ``full_window_bars``: the full window's length (the engine's workspaces hold it, and the
    tuning times it); ``arrays``: return the ``ARRAY_KEYS`` vectors as NumPy arrays. A width
    ``tune_cpu`` did not tune for is tuned here first (``cpu_tunings`` counts them)."""
    global _cpu_engine, _cpu_generation

    parts = _window("cpu", rows, times, generation)
    width = len(parts[0])
    if _cpu_threads is None or (not threads and _cpu_tuned_width != width):
        _tune(width, full_window_bars, bar_seconds, threads)
    if _cpu_engine is None:
        _cpu_engine = CovarianceEngine(bar_seconds=bar_seconds, window_bars=full_window_bars)
    if generation != _cpu_generation:
        if _cpu_engine.width != width:
            _cpu_engine.update(None, times, parts=parts)  # allocates the workspaces: untimed
        _cpu_engine.reset()
        _cpu_generation = generation
    result = _arrays(_cpu_engine.update(None, times, parts=parts), arrays)
    result["threads"] = _cpu_threads
    result["cpu_thread_trials_ms"] = _cpu_thread_trials
    result["cpu_tuned_bars"] = _cpu_tuned_bars
    result["cpu_tunings"] = _cpu_tunings
    result["cpu_affinity"] = (
        sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    )
    result["blas"] = _cpu_blas
    result["finished_wall_ns"] = time.time_ns()
    return result


def reset() -> None:
    """Forget this process's engines and BLAS tuning (benchmarks and tests run several
    universes in one process)."""
    global _cpu_engine, _cpu_generation, _cpu_threads, _cpu_tuned_bars, _cpu_thread_trials
    global _cpu_blas, _cpu_limits, _cpu_tuned_width, _cpu_tunings, _gpu_engine, _gpu_generation
    if _cpu_limits is not None:
        _cpu_limits.restore_original_limits()
    _cpu_engine = _cpu_generation = _cpu_threads = _cpu_thread_trials = None
    _cpu_blas = _cpu_limits = _cpu_tuned_width = None
    _cpu_tuned_bars = _cpu_tunings = 0
    _gpu_engine = _gpu_generation = None
    _windows.clear()


_gpu_engine = None
_gpu_generation = None


def gpu_window(rows, times, generation, bar_seconds, device=0, window_bars=None, *, arrays=False):
    """Persistent CUDA worker, isolated from the network decoder's Python GIL. One engine
    per process, reset on a new generation; the first window of a universe allocates the
    workspaces (for ``window_bars``-bar windows) and the CUDA graphs, and runs once untimed.
    ``rows`` and ``arrays`` as for ``cpu_window``."""
    global _gpu_engine, _gpu_generation
    parts = _window("gpu", rows, times, generation)
    if _gpu_engine is None:
        _gpu_engine = CovarianceEngine(
            "gpu", bar_seconds=bar_seconds, gpu_device=device, window_bars=window_bars
        )
    if generation != _gpu_generation:
        if _gpu_engine.width != len(parts[0]):
            _gpu_engine.update(None, times, parts=parts)
        _gpu_engine.reset()
        _gpu_generation = generation
    result = _arrays(_gpu_engine.update(None, times, parts=parts), arrays)
    result["gpu_name"] = gpu_name(_gpu_engine.xp, device)
    result["finished_wall_ns"] = time.time_ns()
    return result
