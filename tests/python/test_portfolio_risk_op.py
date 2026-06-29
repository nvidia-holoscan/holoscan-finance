# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for deterministic marked-to-market portfolio risk."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest
from conftest import invoke

from holoscan.core import OperatorSpec
from holoscan.finance import (
    MarketFeatures,
    MarketTick,
    PortfolioRiskOp,
    PortfolioSnapshot,
    Position,
    RiskLimits,
)

START = datetime(2026, 6, 27, 14, 30, tzinfo=timezone.utc)


def position(
    symbol: str,
    quantity: str,
    average_price: str,
    currency: str = "USD",
) -> Position:
    return Position(
        symbol=symbol,
        quantity=Decimal(quantity),
        average_price=Decimal(average_price),
        currency=currency,
    )


def features(
    sequence: int,
    symbol: str,
    price: str,
    currency: str = "USD",
    timestamp: datetime | None = None,
) -> MarketFeatures:
    tick = MarketTick(
        timestamp=timestamp or START + timedelta(microseconds=sequence),
        sequence=sequence,
        symbol=symbol,
        price=Decimal(price),
        currency=currency,
    )
    return MarketFeatures(
        tick=tick,
        simple_return=None,
        log_return=None,
        moving_average=None,
        return_deviation=None,
        price_observations=1,
        return_observations=0,
    )


def mark(op, value: MarketFeatures) -> PortfolioSnapshot:
    output = invoke(op, port="features", value=value)
    assert len(output.emissions) == 1
    snapshot, port = output.emissions[0]
    assert port == "snapshot"
    assert isinstance(snapshot, PortfolioSnapshot)
    return snapshot


def test_portfolio_risk_declares_features_and_snapshot_ports(fragment):
    op = PortfolioRiskOp(
        fragment,
        positions=[position("NVDA", "1", "100")],
        limits=RiskLimits(),
    )
    spec = OperatorSpec()

    op.setup(spec)

    assert set(spec.inputs) == {"features"}
    assert set(spec.outputs) == {"snapshot"}


def test_portfolio_marks_long_and_short_positions_and_reports_partial_state(fragment):
    positions = [
        position("ALPHA", "10", "100"),
        position("BETA", "-5", "200"),
    ]
    op = PortfolioRiskOp(fragment, positions=positions, limits=RiskLimits())

    partial = mark(op, features(1, "ALPHA", "110"))

    assert partial.complete is False
    assert partial.currency == "USD"
    assert partial.unpriced_symbols == ("BETA",)
    assert partial.gross_exposure == Decimal("1100")
    assert partial.net_exposure == Decimal("1100")
    assert partial.unrealized_pnl == Decimal("100")
    assert partial.breaches == ()
    assert len(partial.marks) == 1
    assert partial.marks[0].position is positions[0]

    complete = mark(op, features(2, "BETA", "180"))

    assert complete.complete is True
    assert complete.unpriced_symbols == ()
    assert [item.position.symbol for item in complete.marks] == ["ALPHA", "BETA"]
    assert [item.market_value for item in complete.marks] == [
        Decimal("1100"),
        Decimal("-900"),
    ]
    assert [item.unrealized_pnl for item in complete.marks] == [
        Decimal("100"),
        Decimal("100"),
    ]
    assert complete.gross_exposure == Decimal("2000")
    assert complete.net_exposure == Decimal("200")
    assert complete.unrealized_pnl == Decimal("200")
    assert complete.timestamp == START + timedelta(microseconds=2)
    assert complete.sequence == 2


def test_partial_snapshot_never_evaluates_limits(fragment):
    op = PortfolioRiskOp(
        fragment,
        positions=[position("ALPHA", "100", "100"), position("BETA", "1", "1")],
        limits=RiskLimits(
            max_gross_exposure=Decimal("1"),
            max_abs_net_exposure=Decimal("1"),
            max_unrealized_loss=Decimal("1"),
        ),
    )

    snapshot = mark(op, features(1, "ALPHA", "50"))

    assert snapshot.complete is False
    assert snapshot.breaches == ()


@pytest.mark.parametrize(
    ("limit_name", "boundary", "breached", "expected_fragment"),
    [
        ("max_gross_exposure", "1100", "1099.99", "gross"),
        ("max_abs_net_exposure", "1100", "1099.99", "net"),
        ("max_unrealized_loss", "100", "99.99", "loss"),
    ],
)
def test_risk_limits_are_strict_and_do_not_breach_at_the_boundary(
    fragment, limit_name, boundary, breached, expected_fragment
):
    kwargs = {limit_name: Decimal(boundary)}
    at_boundary = PortfolioRiskOp(
        fragment,
        positions=[position("NVDA", "-10", "100")],
        limits=RiskLimits(**kwargs),
    )
    boundary_snapshot = mark(at_boundary, features(1, "NVDA", "110"))

    if limit_name == "max_unrealized_loss":
        # A short marked above its average price has a 100-unit unrealized loss.
        assert boundary_snapshot.unrealized_pnl == Decimal("-100")
    assert boundary_snapshot.breaches == ()

    kwargs[limit_name] = Decimal(breached)
    over_limit = PortfolioRiskOp(
        fragment,
        positions=[position("NVDA", "-10", "100")],
        limits=RiskLimits(**kwargs),
    )
    breached_snapshot = mark(over_limit, features(1, "NVDA", "110"))

    assert len(breached_snapshot.breaches) == 1
    breach = breached_snapshot.breaches[0]
    assert expected_fragment in breach.code.lower()
    assert breach.limit == Decimal(breached)
    assert breach.observed == (
        Decimal("100") if limit_name == "max_unrealized_loss" else Decimal("1100")
    )
    assert breach.message


def test_absolute_net_limit_applies_to_negative_net_exposure(fragment):
    op = PortfolioRiskOp(
        fragment,
        positions=[position("NVDA", "-10", "100")],
        limits=RiskLimits(max_abs_net_exposure=Decimal("1000")),
    )

    snapshot = mark(op, features(1, "NVDA", "110"))

    assert snapshot.net_exposure == Decimal("-1100")
    assert snapshot.breaches[0].observed == Decimal("1100")


def test_portfolio_rejects_duplicate_symbols(fragment):
    with pytest.raises(ValueError, match="duplicate|symbol"):
        PortfolioRiskOp(
            fragment,
            positions=[position("NVDA", "1", "100"), position("NVDA", "2", "100")],
            limits=RiskLimits(),
        )


def test_portfolio_rejects_mixed_position_currencies(fragment):
    with pytest.raises(ValueError, match="currency"):
        PortfolioRiskOp(
            fragment,
            positions=[position("NVDA", "1", "100"), position("SAP", "1", "100", "EUR")],
            limits=RiskLimits(),
        )


def test_portfolio_rejects_a_mark_in_the_wrong_currency(fragment):
    op = PortfolioRiskOp(
        fragment,
        positions=[position("NVDA", "1", "100")],
        limits=RiskLimits(),
    )

    with pytest.raises(ValueError, match="currency"):
        mark(op, features(1, "NVDA", "110", currency="EUR"))


def test_portfolio_rejects_duplicate_and_late_features(fragment):
    op = PortfolioRiskOp(
        fragment,
        positions=[position("NVDA", "1", "100")],
        limits=RiskLimits(),
    )
    mark(op, features(2, "NVDA", "110"))

    with pytest.raises(ValueError, match="sequence|increasing"):
        mark(op, features(2, "NVDA", "90"))
    with pytest.raises(ValueError, match="timestamp|non-decreasing"):
        mark(
            op,
            features(
                3,
                "NVDA",
                "90",
                timestamp=START - timedelta(seconds=1),
            ),
        )


def test_portfolio_start_clears_all_cached_marks(fragment):
    op = PortfolioRiskOp(
        fragment,
        positions=[position("ALPHA", "1", "100"), position("BETA", "1", "200")],
        limits=RiskLimits(),
    )
    mark(op, features(1, "ALPHA", "110"))
    assert mark(op, features(2, "BETA", "210")).complete is True

    op.start()
    after_reset = mark(op, features(3, "BETA", "205"))

    assert after_reset.complete is False
    assert after_reset.unpriced_symbols == ("ALPHA",)
