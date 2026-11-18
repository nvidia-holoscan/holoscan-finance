# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU/GPU agreement tolerances and the comparison rules of ``compare_results``."""

from __future__ import annotations

import numpy as np
import pytest

from holoscan_finance.analytics.agreement import agreement, compare_results
from holoscan_finance.marketdata.covariance import CovarianceEngine

pytest.importorskip("scipy")


def _result(seed: int = 61, shape: tuple[int, int] = (10, 5)) -> dict:
    rows = np.random.default_rng(seed).normal(size=shape)
    return CovarianceEngine().update(rows, np.arange(shape[0]))


def test_agreement_requires_the_same_displayed_instruments():
    result = _result()
    wrong_labels = {**result, "heatmap_indices": list(reversed(result["heatmap_indices"]))}
    assert not agreement(result, wrong_labels)


def test_agreement_tolerates_rounding_but_not_a_different_volatility():
    result = _result()
    rounded = {**result, "annual_vol": [v * (1 + 1e-12) for v in result["annual_vol"]]}
    assert agreement(result, rounded)
    off = {**result, "annual_vol": [v * 1.001 for v in result["annual_vol"]]}
    assert not agreement(result, off)


def test_identical_results_agree_match_and_are_usable():
    result = _result()
    assert compare_results(result, dict(result)) == (True, True, True)


@pytest.mark.parametrize("key", ["method", "added_bars", "expired_bars", "bars", "universe"])
def test_different_work_is_not_matched_and_not_usable(key):
    result = _result()
    other = {**result, key: "other"}
    assert compare_results(other, result) == (True, False, False)


def test_fewer_than_two_eligible_names_is_not_usable():
    rows = np.zeros((10, 3))
    rows[:, 0] = np.random.default_rng(3).normal(size=10)
    result = CovarianceEngine().update(rows, np.arange(10))
    assert result["eligible"] == 1
    assert compare_results(result, dict(result)) == (True, True, False)


def test_a_disagreeing_result_is_not_usable():
    result = _result()
    off = {**result, "covariance_probes": (np.asarray(result["covariance_probes"]) * 2).tolist()}
    assert compare_results(off, result) == (False, True, False)
