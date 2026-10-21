# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU risk: heatmap selection independent of the BLAS backend, and ``RollingRisk(gpu)`` on
device windows matching the engine, validated or not."""

from __future__ import annotations

import numpy as np
import pytest

from holoscan_finance.analytics.agreement import agreement
from holoscan_finance.analytics.rolling import RollingRisk
from holoscan_finance.control.gate import CrossCheckGate
from holoscan_finance.marketdata.covariance import CovarianceEngine
from holoscan_finance.payloads import BarWindow

pytestmark = pytest.mark.gpu
cp = pytest.importorskip("cupy")
pytest.importorskip("scipy")  # the NumPy engine's dgemm


def test_equal_volatility_heatmap_selection_is_backend_independent():
    rng = np.random.default_rng(912)
    rows = rng.normal(0, 0.01, (80, 40))
    rows[20:] = np.tile(rng.normal(0, 0.001, (60, 1)), (1, 40))
    cpu, gpu = CovarianceEngine(), CovarianceEngine("gpu")
    for end in range(60, 81):
        a = cpu.update(rows[end - 60 : end], np.arange(end - 60, end))
        b = gpu.update(rows[end - 60 : end], np.arange(end - 60, end))
        assert agreement(a, b)
    assert a["heatmap_indices"] == list(range(24))


def _device_windows(rows: np.ndarray) -> list[BarWindow]:
    return [
        BarWindow(
            tuple(cp.asarray(r) for r in rows[i : i + 60]),
            np.arange(i, i + 60),
            1,
            60 + i,
            0,
            device=True,
        )
        for i in (0, 1)
    ]


@pytest.mark.parametrize("validate", [True, False])
def test_rolling_risk_on_device_windows_matches_the_engine(validate):
    rng = np.random.default_rng(3)
    rows = rng.normal(0, 0.001, (61, 40))
    rows[5:9, 2] = np.nan
    rows[:, 7] = 0
    gate = CrossCheckGate()
    risk = RollingRisk(backend="gpu", min_interval_ns=0, cross_check=validate, gate=gate)
    risk.start()
    engine = CovarianceEngine("gpu")
    for w in _device_windows(rows):
        risk.offer(w)
        update = risk.poll()
        gate.release()  # stands in for the published comparison
        host = rows[w.times[0] : w.times[-1] + 1]
        observed = np.var(host, axis=0, ddof=1) if validate else None
        want = engine.update(host, w.times, observed_variance=observed)
        assert update.window is w and update.validated is validate and update.backend == "gpu"
        if validate:
            np.testing.assert_array_equal(update.rows_host, host)
        else:
            assert update.rows_host is None
        for key in ("method", "added_bars", "expired_bars", "eligible_indices", "heatmap_indices"):
            assert update.result[key] == want[key], key
        np.testing.assert_allclose(update.result["annual_vol"], want["annual_vol"], rtol=1e-10)
        assert update.result["gpu_name"] and update.result["finished_wall_ns"] > 0
    assert update.result["method"] == "incremental"  # second window: one bar added, one expired
    risk.stop()
