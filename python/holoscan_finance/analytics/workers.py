# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Spawn-worker functions: the CPU comparator and the separate-process GPU worker.

``cpu_window`` and ``gpu_window`` are verbatim from 4.3 ``equities.analytics``. Each keeps a
persistent incremental engine per process and generation, so a worker pays the full rebuild
only on a new generation; ``cpu_window`` tunes the BLAS thread count over {1, 2, 4, 8, all}
at start and at the first full window (the 4.3 9.55x CPU/GPU method). Workers run in
spawned processes so BLAS and CUDA work never holds the pipeline's GIL (in-process, 30%
decoder duty added 45% to the engine time: J2); ``pin_worker`` is their initializer (the
4.3 ``keep_off_cores``) and ``start_worker_pool`` starts one before data flows.
"""

from __future__ import annotations

import multiprocessing
import os
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from ..ingest.affinity import pin_worker
from ..marketdata.covariance import CovarianceEngine, gpu_name

__all__ = ["cpu_window", "gpu_window", "pin_worker", "ping", "start_worker_pool"]


def ping() -> int:
    """The worker's pid; submitting it once starts the worker before data flows."""
    return os.getpid()


def start_worker_pool(worker_cpus: Sequence[int] = ()) -> ProcessPoolExecutor:
    """One spawned worker pinned to ``worker_cpus`` (``[]`` inherits), already running.

    Spawn, never fork: the parent runs runtime and CUDA threads. The ``ping`` round trip
    takes the interpreter start (and the re-import of an app script's top-level imports)
    out of the first window's latency.
    """
    pool = ProcessPoolExecutor(
        max_workers=1,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=pin_worker,
        initargs=(list(worker_cpus),),
    )
    try:
        pool.submit(ping).result()
    except BaseException:
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    return pool


_cpu_engine = None
_cpu_generation = None
_cpu_threads = None
_cpu_tuned_bars = 0
_cpu_thread_trials = None


def cpu_window(rows, times, generation, bar_seconds, threads=0, full_window_bars=60):
    """Spawned worker with persistent incremental state; calibrates BLAS thread count."""
    global _cpu_engine, _cpu_generation, _cpu_threads, _cpu_tuned_bars, _cpu_thread_trials
    from threadpoolctl import threadpool_info, threadpool_limits

    if _cpu_threads is None or (not threads and _cpu_tuned_bars < full_window_bars <= len(times)):
        available = (
            len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else os.cpu_count() or 1
        )
        candidates = (
            [threads]
            if threads
            else sorted({1, min(2, available), min(4, available), min(8, available), available})
        )
        if any(x < 1 or x > available for x in candidates):
            raise ValueError(f"CPU threads must be 1..{available} (or 0 for tuning)")
        timings = {}
        for nt in candidates:
            with threadpool_limits(limits=nt, user_api="blas"):
                engine = CovarianceEngine(bar_seconds=bar_seconds)
                engine.update(rows, times)
                timings[nt] = float(
                    np.median([_calibration_update(engine, rows, times) for _ in range(3)])
                )
        _cpu_threads = min(timings, key=timings.get)
        _cpu_tuned_bars = len(times)
        _cpu_thread_trials = timings
    if _cpu_engine is None or generation != _cpu_generation:
        _cpu_engine = CovarianceEngine(bar_seconds=bar_seconds)
        _cpu_generation = generation
    with threadpool_limits(limits=_cpu_threads, user_api="blas"):
        result = _cpu_engine.update(rows, times)
    result["threads"] = _cpu_threads
    result["cpu_thread_trials_ms"] = _cpu_thread_trials
    result["cpu_tuned_bars"] = _cpu_tuned_bars
    result["cpu_affinity"] = (
        sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    )
    result["blas"] = [
        {"name": p["internal_api"], "version": p["version"]}
        for p in threadpool_info()
        if p["user_api"] == "blas"
    ]
    result["finished_wall_ns"] = time.time_ns()
    return result


def _calibration_update(engine, rows, times):
    """Tune the steady rolling workload with actual observations, outside timings."""
    if len(times) < 3:
        return engine.update(rows, times, force_rebuild=True)["milliseconds"]
    engine.update(rows[:-1], times[:-1], force_rebuild=True)
    return engine.update(rows[1:], times[1:])["milliseconds"]


_gpu_engine = None
_gpu_generation = None


def gpu_window(rows, times, generation, bar_seconds, device=0):
    """Persistent CUDA worker, isolated from the network decoder's Python GIL."""
    global _gpu_engine, _gpu_generation
    if _gpu_engine is None or generation != _gpu_generation:
        warm = CovarianceEngine("gpu", bar_seconds=bar_seconds, gpu_device=device)
        warm.update(rows, times)
        _gpu_engine = CovarianceEngine("gpu", bar_seconds=bar_seconds, gpu_device=device)
        _gpu_generation = generation
    result = _gpu_engine.update(rows, times)
    result["gpu_name"] = gpu_name(_gpu_engine.xp, device)
    result["finished_wall_ns"] = time.time_ns()
    return result
