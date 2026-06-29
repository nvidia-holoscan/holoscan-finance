# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for immutable messages shared by the finance operators."""

from dataclasses import FrozenInstanceError, is_dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from holoscan.finance import (
    MarketFeatures,
    MarketTick,
    PortfolioSnapshot,
    Position,
    PositionMark,
    RiskBreach,
    RiskLimits,
)

NOW = datetime(2026, 6, 27, 14, 30, tzinfo=timezone.utc)


def test_market_tick_is_an_immutable_decimal_value_object():
    tick = MarketTick(
        timestamp=NOW,
        sequence=42,
        symbol="NVDA",
        price=Decimal("157.2500"),
        currency="USD",
    )

    assert is_dataclass(tick)
    assert tick.timestamp == NOW
    assert tick.sequence == 42
    assert tick.symbol == "NVDA"
    assert tick.price == Decimal("157.2500")
    assert tick.currency == "USD"
    with pytest.raises(FrozenInstanceError):
        tick.price = Decimal("1")


def test_market_tick_defaults_to_us_dollars():
    tick = MarketTick(timestamp=NOW, sequence=0, symbol="NVDA", price=Decimal("157.25"))

    assert tick.currency == "USD"


def test_market_tick_normalizes_timezone_symbol_and_currency():
    eastern = timezone(timedelta(hours=-4))
    tick = MarketTick(
        timestamp=datetime(2026, 6, 27, 10, 30, tzinfo=eastern),
        sequence=1,
        symbol=" nvda ",
        price=Decimal("157.25"),
        currency="usd",
    )

    assert tick.timestamp == NOW
    assert tick.timestamp.tzinfo is timezone.utc
    assert tick.symbol == "NVDA"
    assert tick.currency == "USD"


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"timestamp": datetime(2026, 6, 27, 14, 30)}, "timestamp|timezone|UTC"),
        ({"timestamp": "2026-06-27T14:30:00Z"}, "timestamp"),
        ({"sequence": -1}, "sequence"),
        ({"sequence": 1.5}, "sequence"),
        ({"sequence": True}, "sequence"),
        ({"symbol": ""}, "symbol"),
        ({"symbol": "   "}, "symbol"),
        ({"symbol": 123}, "symbol"),
        ({"price": 157.25}, "price"),
        ({"price": Decimal("0")}, "price"),
        ({"price": Decimal("-0.01")}, "price"),
        ({"price": Decimal("NaN")}, "price"),
        ({"price": Decimal("Infinity")}, "price"),
        ({"currency": "US"}, "currency"),
    ],
)
def test_market_tick_rejects_invalid_financial_data(overrides, match):
    values = dict(
        timestamp=NOW,
        sequence=1,
        symbol="NVDA",
        price=Decimal("157.25"),
        currency="USD",
    )
    values.update(overrides)

    with pytest.raises((TypeError, ValueError), match=match):
        MarketTick(**values)


def test_market_features_retain_the_original_tick():
    tick = MarketTick(timestamp=NOW, sequence=1, symbol="NVDA", price=Decimal("102"))
    features = MarketFeatures(
        tick=tick,
        simple_return=0.009900990099009901,
        log_return=0.00985229644301164,
        moving_average=Decimal("101"),
        return_deviation=0.005,
        price_observations=3,
        return_observations=2,
    )

    assert is_dataclass(features)
    assert features.tick is tick
    assert features.price_observations == 3
    assert features.return_observations == 2
    assert features.moving_average == Decimal("101")


def test_market_features_allow_unavailable_warmup_metrics():
    tick = MarketTick(timestamp=NOW, sequence=1, symbol="NVDA", price=Decimal("100"))
    features = MarketFeatures(
        tick=tick,
        simple_return=None,
        log_return=None,
        moving_average=None,
        return_deviation=None,
        price_observations=1,
        return_observations=0,
    )

    assert features.simple_return is None
    assert features.log_return is None
    assert features.moving_average is None
    assert features.return_deviation is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"price_observations": 0},
        {"return_observations": -1},
        {"moving_average": Decimal("0")},
        {"return_deviation": -0.001},
        {"simple_return": float("nan")},
        {"simple_return": -1.0},
        {"simple_return": 0.01, "log_return": None},
    ],
)
def test_market_features_reject_impossible_statistics(overrides):
    tick = MarketTick(timestamp=NOW, sequence=1, symbol="NVDA", price=Decimal("102"))
    values = dict(
        tick=tick,
        simple_return=0.01,
        log_return=0.00995,
        moving_average=Decimal("101"),
        return_deviation=0.005,
        price_observations=3,
        return_observations=2,
    )
    values.update(overrides)

    with pytest.raises((TypeError, ValueError)):
        MarketFeatures(**values)


def test_position_supports_long_and_short_books():
    long_position = Position(symbol="NVDA", quantity=Decimal("10"), average_price=Decimal("100"))
    short_position = Position(
        symbol="TSLA",
        quantity=Decimal("-5"),
        average_price=Decimal("250"),
        currency="USD",
    )

    assert long_position.quantity == Decimal("10")
    assert long_position.currency == "USD"
    assert short_position.quantity == Decimal("-5")


@pytest.mark.parametrize(
    "overrides",
    [
        {"symbol": ""},
        {"quantity": Decimal("0")},
        {"quantity": Decimal("NaN")},
        {"quantity": 10.0},
        {"average_price": Decimal("0")},
        {"average_price": Decimal("Infinity")},
        {"currency": "US"},
    ],
)
def test_position_validation(overrides):
    values = dict(
        symbol="NVDA",
        quantity=Decimal("10"),
        average_price=Decimal("100"),
        currency="USD",
    )
    values.update(overrides)

    with pytest.raises((TypeError, ValueError)):
        Position(**values)


def test_position_normalizes_symbol_and_currency():
    value = Position(
        symbol=" nvda ",
        quantity=Decimal("10"),
        average_price=Decimal("100"),
        currency="usd",
    )

    assert value.symbol == "NVDA"
    assert value.currency == "USD"


def test_derived_risk_messages_are_immutable_and_complete():
    position = Position(symbol="NVDA", quantity=Decimal("2"), average_price=Decimal("100"))
    mark = PositionMark(
        position=position,
        market_price=Decimal("110"),
        market_value=Decimal("220"),
        unrealized_pnl=Decimal("20"),
    )
    breach = RiskBreach(
        code="MAX_GROSS_EXPOSURE",
        observed=Decimal("220"),
        limit=Decimal("200"),
        message="gross exposure exceeds its configured limit",
    )
    snapshot = PortfolioSnapshot(
        timestamp=NOW,
        sequence=42,
        currency="USD",
        marks=(mark,),
        gross_exposure=Decimal("220"),
        net_exposure=Decimal("220"),
        unrealized_pnl=Decimal("20"),
        unpriced_symbols=(),
        breaches=(breach,),
    )

    assert all(is_dataclass(value) for value in (mark, breach, snapshot))
    assert snapshot.complete is True
    assert snapshot.marks == (mark,)
    assert snapshot.breaches == (breach,)
    with pytest.raises(FrozenInstanceError):
        snapshot.sequence = 1


def test_snapshot_is_partial_when_any_position_is_unpriced():
    snapshot = PortfolioSnapshot(
        timestamp=NOW,
        sequence=1,
        currency="USD",
        marks=(),
        gross_exposure=Decimal("0"),
        net_exposure=Decimal("0"),
        unrealized_pnl=Decimal("0"),
        unpriced_symbols=("NVDA",),
        breaches=(),
    )

    assert snapshot.complete is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_gross_exposure": Decimal("0")},
        {"max_abs_net_exposure": Decimal("-1")},
        {"max_unrealized_loss": Decimal("NaN")},
        {"max_unrealized_loss": Decimal("Infinity")},
        {"max_gross_exposure": 1000.0},
    ],
)
def test_risk_limits_must_be_positive_and_finite(kwargs):
    with pytest.raises((TypeError, ValueError)):
        RiskLimits(**kwargs)


def test_risk_limits_may_be_independently_disabled():
    limits = RiskLimits(max_gross_exposure=Decimal("1000"))

    assert limits.max_gross_exposure == Decimal("1000")
    assert limits.max_abs_net_exposure is None
    assert limits.max_unrealized_loss is None
