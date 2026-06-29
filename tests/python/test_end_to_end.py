# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Manual in-process chain proving the three operators interoperate."""

from decimal import Decimal

from conftest import invoke

from holoscan.finance import (
    MarketDataReplayOp,
    PortfolioRiskOp,
    Position,
    RiskLimits,
    RollingStatisticsOp,
    load_market_ticks_csv,
)


def test_csv_replay_statistics_and_risk_chain(fragment, tmp_path):
    csv_path = tmp_path / "market.csv"
    csv_path.write_text(
        "timestamp,sequence,symbol,price,currency\n"
        "2026-06-27T14:30:00+00:00,1,ALPHA,100,USD\n"
        "2026-06-27T14:30:01+00:00,2,BETA,200,USD\n"
        "2026-06-27T14:30:02+00:00,3,ALPHA,110,USD\n"
        "2026-06-27T14:30:03+00:00,4,BETA,180,USD\n",
        encoding="utf-8",
    )
    replay = MarketDataReplayOp(fragment, ticks=load_market_ticks_csv(csv_path))
    statistics = RollingStatisticsOp(fragment, window_size=3, min_periods=2)
    risk = PortfolioRiskOp(
        fragment,
        positions=(
            Position("ALPHA", Decimal("10"), Decimal("100")),
            Position("BETA", Decimal("-5"), Decimal("200")),
        ),
        limits=RiskLimits(max_gross_exposure=Decimal("2500")),
    )

    snapshots = []
    feature_messages = []
    for _ in range(replay.record_count):
        tick, tick_port = invoke(replay).emitted
        assert tick_port == "tick"
        feature, feature_port = invoke(statistics, port=tick_port, value=tick).emitted
        assert feature_port == "features"
        feature_messages.append(feature)
        snapshot, snapshot_port = invoke(risk, port=feature_port, value=feature).emitted
        assert snapshot_port == "snapshot"
        snapshots.append(snapshot)

    assert [feature.price_observations for feature in feature_messages] == [1, 1, 2, 2]
    assert feature_messages[2].moving_average == Decimal("105")
    assert feature_messages[3].moving_average == Decimal("190")
    assert snapshots[0].unpriced_symbols == ("BETA",)
    assert all(snapshot.complete for snapshot in snapshots[1:])

    final = snapshots[-1]
    assert final.sequence == 4
    assert final.gross_exposure == Decimal("2000")
    assert final.net_exposure == Decimal("200")
    assert final.unrealized_pnl == Decimal("200")
    assert final.breaches == ()
