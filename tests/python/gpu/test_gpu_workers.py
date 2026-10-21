# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Spawned CPU and GPU workers agree after a roll and a seek (4.3
``test_persistent_workers_match_after_roll_and_seek``, which failed on 5.0 under the old
``holoscan.equities`` package and passes here), and ``gpu_window`` runs in a pinned worker."""

from __future__ import annotations

import os

import numpy as np
import pytest

from holoscan_finance.analytics.agreement import agreement
from holoscan_finance.analytics.workers import cpu_window, gpu_window, start_worker_pool

pytestmark = pytest.mark.gpu
pytest.importorskip("scipy")
pytest.importorskip("threadpoolctl")


def test_persistent_workers_match_after_roll_and_seek():
    rows = np.random.default_rng(42).normal(0, 0.001, (30, 12))
    rows[2:4, 3] = np.nan
    with start_worker_pool() as cpu, start_worker_pool() as gpu:
        for generation, end, method in (
            (1, 10, "rebuild"),
            (1, 13, "incremental"),
            (2, 10, "rebuild"),
        ):
            sample, times = rows[end - 10 : end], np.arange(end - 10, end)
            cf = cpu.submit(cpu_window, sample, times, generation, 5, 1)
            gf = gpu.submit(gpu_window, sample, times, generation, 5)
            cr, gr = cf.result(timeout=60), gf.result(timeout=60)
            assert cr["method"] == gr["method"] == method
            assert cr["added_bars"] == gr["added_bars"]
            assert cr["expired_bars"] == gr["expired_bars"]
            assert agreement(cr, gr)
            oracle = np.cov(sample[:, cr["eligible_indices"]], rowvar=False)
            np.testing.assert_allclose(gr["covariance_probes"], oracle, rtol=1e-10, atol=1e-16)


def test_gpu_window_in_a_pinned_worker():
    pin = sorted(os.sched_getaffinity(0))[:2]
    rows = np.random.default_rng(8).normal(0, 0.001, (20, 6))
    with start_worker_pool(pin) as pool:
        assert sorted(pool.submit(os.sched_getaffinity, 0).result(timeout=60)) == pin
        result = pool.submit(gpu_window, rows, np.arange(20), 1, 5).result(timeout=60)
    assert result["gpu_name"] and result["method"] == "rebuild"
    oracle = np.cov(rows[:, result["eligible_indices"]], rowvar=False)
    np.testing.assert_allclose(result["covariance_probes"], oracle, rtol=1e-10, atol=1e-16)
