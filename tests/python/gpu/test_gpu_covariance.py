# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``CovarianceEngine("gpu")``: the NumPy oracle test on the GPU backend, and device windows
consumed in place (from 4.3 ``test_equities`` and ``test_equities_gpu``)."""

from __future__ import annotations

import numpy as np
import pytest
from library.test_covariance import check_incremental_covariance_matches_numpy_oracle

from holoscan_finance.marketdata.covariance import CovarianceEngine

pytestmark = pytest.mark.gpu
cp = pytest.importorskip("cupy")


def test_incremental_covariance_matches_independent_numpy_oracle():
    check_incremental_covariance_matches_numpy_oracle("gpu")


def test_covariance_engine_consumes_a_device_window_in_place():
    """GPU ingest feeds analytics a CuPy window; results must match the host-window
    path, including incremental updates, a supplied variance and a completed-bar check."""
    rng = np.random.default_rng(5)
    rows = rng.normal(0, 0.001, (90, 40))
    rows[3:9, 2] = np.nan
    rows[:, 7] = 0
    host_engine, dev_engine = CovarianceEngine("gpu"), CovarianceEngine("gpu")
    for end in (60, 61, 62, 70, 90):
        window, times = rows[end - 60 : end], np.arange(end - 60, end)
        h = host_engine.update(window, times, return_matrix=True)
        d = dev_engine.update(cp.asarray(window), times, return_matrix=True)
        assert (h["method"], h["added_bars"], h["expired_bars"]) == (
            d["method"],
            d["added_bars"],
            d["expired_bars"],
        )
        assert h["eligible_indices"] == d["eligible_indices"]
        assert h["heatmap_indices"] == d["heatmap_indices"]
        np.testing.assert_allclose(h["matrix"], d["matrix"], rtol=1e-10, atol=1e-16)
        np.testing.assert_allclose(h["annual_vol"], d["annual_vol"], rtol=1e-10)
    supplied = dev_engine.update(
        cp.asarray(rows[30:90]),
        np.arange(30, 90),
        observed_variance=np.var(rows[30:90], axis=0, ddof=1),
    )
    assert supplied["eligible_indices"] == d["eligible_indices"]
    changed = rows[30:90].copy()
    changed[0, 0] += 0.01
    with pytest.raises(ValueError, match="completed bars changed"):
        dev_engine.update(cp.asarray(changed), np.arange(30, 90))
