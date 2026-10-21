# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``CovarianceEngine`` on the CPU against an independent NumPy oracle, settling to zero
eligibility, and the cached GPU name (from 4.3 ``test_equities``; the ``gpu`` param of the
oracle test is ``gpu/test_gpu_covariance.py``)."""

from __future__ import annotations

import numpy as np
import pytest

from holoscan_finance.marketdata import covariance
from holoscan_finance.marketdata.covariance import CovarianceEngine


def check_incremental_covariance_matches_numpy_oracle(backend):
    """Shared with the GPU test: rolling, rebuilt and settled windows on ``backend``."""
    rng = np.random.default_rng(120)
    rows = rng.normal(0, 0.001, (50, 12))
    rows[3:7, 1] = np.nan
    rows[:, 2] = 0
    rows[18:20, 4] = np.nan
    engine = CovarianceEngine(backend)
    for end in (10, 11, 15, 23, 30, 45):
        sample = rows[end - 10 : end]
        result = engine.update(sample, np.arange(end - 10, end), return_matrix=True)
        eligible = np.flatnonzero(
            np.isfinite(sample).all(axis=0) & (np.nanvar(sample, axis=0) > 1e-20)
        )
        assert result["eligible_indices"] == eligible.tolist()
        oracle = np.cov(sample[:, eligible], rowvar=False, ddof=1)
        np.testing.assert_allclose(result["matrix"], oracle, rtol=1e-10, atol=1e-16)
        np.testing.assert_allclose(
            result["annual_vol"], np.sqrt(oracle.diagonal() * 252 * 23400 / 5), rtol=1e-10
        )
    changed = rows[35:45].copy()
    changed[0, 0] += 0.01
    with pytest.raises(ValueError, match="completed bars changed"):
        engine.update(changed, np.arange(35, 45))
    reset = engine.update(rows[:10], np.arange(10), return_matrix=True)
    assert reset["method"] == "rebuild"
    # All movement expires before the periodic rebuild. Subtraction roundoff
    # must not leave phantom volatility or inconsistent CPU/GPU eligibility.
    settling = rng.normal(0, 0.1, (120, 12))
    settling[60:] = 0
    engine = CovarianceEngine(backend)
    for end in range(60, 121):
        result = engine.update(settling[end - 60 : end], np.arange(end - 60, end))
    assert result["eligible_indices"] == []
    assert result["mean_correlation"] is None


def test_incremental_covariance_matches_independent_numpy_oracle():
    check_incremental_covariance_matches_numpy_oracle("cpu")


def test_gpu_name_is_queried_once():
    class Runtime:
        calls = 0

        def getDeviceProperties(self, device):
            Runtime.calls += 1
            return {"name": b"Test GPU"}

    class Xp:
        class cuda:
            runtime = Runtime()

    covariance._gpu_names.pop(99, None)
    assert [covariance.gpu_name(Xp, 99) for _ in range(3)] == ["Test GPU"] * 3
    assert Runtime.calls == 1
