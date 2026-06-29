# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guard the small, intentional top-level API."""

import inspect

import holoscan.finance as finance

PUBLIC_NAMES = {
    "MarketDataReplayOp",
    "MarketFeatures",
    "MarketTick",
    "PortfolioRiskOp",
    "PortfolioSnapshot",
    "Position",
    "PositionMark",
    "RiskBreach",
    "RiskLimits",
    "RollingStatisticsOp",
    "load_market_ticks_csv",
}


def test_public_api_is_explicit_and_complete():
    assert set(finance.__all__) == PUBLIC_NAMES
    assert all(hasattr(finance, name) for name in PUBLIC_NAMES)


def test_public_types_and_loader_have_the_expected_kinds():
    for name in PUBLIC_NAMES - {"load_market_ticks_csv"}:
        assert inspect.isclass(getattr(finance, name)), name
    assert inspect.isfunction(finance.load_market_ticks_csv)


def test_internal_stream_validator_is_not_public():
    assert "validate_market_ticks" not in finance.__all__
