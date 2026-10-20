# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Analytics workers for the dashboard: the CPU comparator, the separate-process GPU
worker of the host-ingest paths, and the CPU/GPU agreement check. The engine itself is
:class:`holoscan.marketdata.covariance.CovarianceEngine`.
"""

from __future__ import annotations

import os
import time

import numpy as np

from holoscan.marketdata.covariance import CovarianceEngine, gpu_name


def agreement(cpu, gpu):
    same = cpu["eligible_indices"] == gpu["eligible_indices"]
    vol_ok = same and np.allclose(cpu["annual_vol"], gpu["annual_vol"], rtol=1e-8, atol=1e-10)
    cov_ok = same and np.allclose(
        cpu["covariance_probes"], gpu["covariance_probes"], rtol=1e-8, atol=1e-14
    )
    corr_ok = (
        same
        and cpu["heatmap_indices"] == gpu["heatmap_indices"]
        and np.allclose(cpu["heatmap"], gpu["heatmap"], rtol=1e-8, atol=1e-8)
    )
    return bool(vol_ok and cov_ok and corr_ok)


def keep_off_cores(reserved):
    """CPU analytics worker initializer: leave the NIC's DAQIRI master/RX cores to the NIC.

    Every existing thread (BLAS pools start at import) and later threads get the mask."""
    if not reserved or not hasattr(os, "sched_setaffinity"):
        return
    cores = set(os.sched_getaffinity(0)) - set(reserved)
    if not cores:
        return
    for tid in os.listdir("/proc/self/task"):
        try:
            os.sched_setaffinity(int(tid), cores)
        except OSError:
            pass


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
