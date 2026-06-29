# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration coverage for the checked-in intraday-risk reference application."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from conftest import USING_HOLOSCAN_STUB, invoke

from holoscan.finance import MarketDataReplayOp, PortfolioRiskOp, RollingStatisticsOp

PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = PROJECT_ROOT / "applications" / "intraday_risk_pipeline"
APP_PATH = APP_ROOT / "python" / "intraday_risk_pipeline.py"
DATA_PATH = APP_ROOT / "data" / "synthetic_market_data.csv"


def _load_application_module():
    spec = importlib.util.spec_from_file_location("intraday_risk_pipeline", APP_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"unable to load reference application from {APP_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_checked_in_pipeline_fixture_produces_expected_breaches(fragment):
    app_module = _load_application_module()
    ticks = app_module.load_market_ticks_csv(DATA_PATH)
    replay = MarketDataReplayOp(fragment, ticks=ticks)
    statistics = RollingStatisticsOp(fragment, window_size=3, min_periods=2)
    risk = PortfolioRiskOp(
        fragment,
        positions=app_module.DEFAULT_POSITIONS,
        limits=app_module.DEFAULT_RISK_LIMITS,
    )

    snapshots = []
    for _ in range(replay.record_count):
        tick, _ = invoke(replay).emitted
        features, _ = invoke(statistics, port="tick", value=tick).emitted
        snapshot, _ = invoke(risk, port="features", value=features).emitted
        snapshots.append(snapshot)

    assert len(snapshots) == 8
    assert snapshots[0].unpriced_symbols == ("BETA",)
    assert all(snapshot.complete for snapshot in snapshots[1:])
    assert [tuple(breach.code for breach in snapshot.breaches) for snapshot in snapshots] == [
        (),
        (),
        (),
        (),
        ("GROSS_EXPOSURE", "ABS_NET_EXPOSURE"),
        ("GROSS_EXPOSURE",),
        ("UNREALIZED_LOSS",),
        ("UNREALIZED_LOSS",),
    ]

    captured = []
    sink = app_module.ConsoleRiskSinkOp(fragment, snapshots=captured)
    for snapshot in snapshots:
        invoke(sink, port="snapshot", value=snapshot)
    assert captured == snapshots


def test_reference_application_composes_with_the_public_operators():
    app_module = _load_application_module()
    app = app_module.IntradayRiskApp()

    app.compose()


@pytest.mark.skipif(USING_HOLOSCAN_STUB, reason="Holoscan SDK is not installed")
def test_reference_application_runs_a_real_holoscan_graph():
    app_module = _load_application_module()
    snapshots = []
    app = app_module.IntradayRiskApp(snapshots=snapshots)

    app.run()

    assert len(snapshots) == 8
    assert snapshots[0].complete is False
    assert snapshots[-1].breaches[0].code == "UNREALIZED_LOSS"
