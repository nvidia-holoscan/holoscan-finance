# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU spawn workers: the persistent engine across a roll and a seek (from 4.3
``test_equities``), the pre-spawn ``ping``, and the pinned worker reporting its CPU set."""

from __future__ import annotations

import os

import numpy as np
import pytest

from holoscan_finance.analytics.workers import cpu_window, ping, start_worker_pool

pytest.importorskip("scipy")
pytest.importorskip("threadpoolctl")


def test_persistent_cpu_worker_matches_after_roll_and_seek():
    rows = np.random.default_rng(42).normal(0, 0.001, (30, 12))
    rows[2:4, 3] = np.nan
    with start_worker_pool() as pool:
        for generation, end, method in (
            (1, 10, "rebuild"),
            (1, 13, "incremental"),
            (2, 10, "rebuild"),
        ):
            sample, times = rows[end - 10 : end], np.arange(end - 10, end)
            result = pool.submit(cpu_window, sample, times, generation, 5, 1).result(timeout=60)
            assert result["method"] == method
            assert result["threads"] == 1
            if method == "incremental":
                assert result["added_bars"] == result["expired_bars"] == 3
            oracle = np.cov(sample[:, result["eligible_indices"]], rowvar=False)
            np.testing.assert_allclose(result["covariance_probes"], oracle, rtol=1e-10, atol=1e-16)
            assert result["finished_wall_ns"] > 0 and result["blas"]


def test_ping_runs_in_a_spawned_worker():
    with start_worker_pool() as pool:
        assert pool.submit(ping).result(timeout=60) != os.getpid()


def test_pinned_worker_reports_its_cpu_set_and_tunes_within_it():
    pin = sorted(os.sched_getaffinity(0))[:2]
    rows = np.random.default_rng(7).normal(0, 0.001, (12, 8))
    with start_worker_pool(pin) as pool:
        result = pool.submit(cpu_window, rows, np.arange(12), 1, 5).result(timeout=60)
    assert result["cpu_affinity"] == pin
    # threads=0 tunes over {1, 2, 4, 8, all}, capped by the pinned set.
    assert set(result["cpu_thread_trials_ms"]) == set(range(1, len(pin) + 1))
    assert result["threads"] in result["cpu_thread_trials_ms"]
