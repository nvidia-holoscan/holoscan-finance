# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for per-symbol rolling market statistics."""

import math
import statistics
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest
from conftest import invoke

from holoscan.core import OperatorSpec
from holoscan.finance import MarketFeatures, MarketTick, RollingStatisticsOp

START = datetime(2026, 6, 27, 14, 30, tzinfo=timezone.utc)


def make_tick(
    sequence: int,
    price: str,
    *,
    symbol: str = "NVDA",
    currency: str = "USD",
    timestamp: datetime | None = None,
) -> MarketTick:
    return MarketTick(
        timestamp=timestamp or START + timedelta(microseconds=sequence),
        sequence=sequence,
        symbol=symbol,
        price=Decimal(price),
        currency=currency,
    )


def calculate(op, tick: MarketTick) -> MarketFeatures:
    output = invoke(op, port="tick", value=tick)
    assert len(output.emissions) == 1
    features, port = output.emissions[0]
    assert port == "features"
    assert isinstance(features, MarketFeatures)
    return features


def test_rolling_operator_declares_tick_and_features_ports(fragment):
    op = RollingStatisticsOp(fragment, window_size=3, min_periods=2)
    spec = OperatorSpec()

    op.setup(spec)

    assert set(spec.inputs) == {"tick"}
    assert set(spec.outputs) == {"features"}


@pytest.mark.parametrize(
    ("window_size", "min_periods"),
    [(0, 1), (-1, 1), (3, 0), (3, -1), (3, 4), (2.5, 1), (3, 1.5)],
)
def test_rolling_constructor_validates_window_configuration(fragment, window_size, min_periods):
    with pytest.raises((TypeError, ValueError), match="window_size|min_periods"):
        RollingStatisticsOp(fragment, window_size=window_size, min_periods=min_periods)


def test_rolling_emits_warmup_features_with_unavailable_metrics_as_none(fragment):
    op = RollingStatisticsOp(fragment, window_size=4, min_periods=3)

    first = calculate(op, make_tick(1, "100"))
    second = calculate(op, make_tick(2, "110"))

    assert first.tick.price == Decimal("100")
    assert first.price_observations == 1
    assert first.return_observations == 0
    assert first.simple_return is None
    assert first.log_return is None
    assert first.moving_average is None
    assert first.return_deviation is None

    assert second.price_observations == 2
    assert second.return_observations == 1
    assert second.simple_return == pytest.approx(0.1)
    assert second.log_return == pytest.approx(math.log(1.1))
    assert second.moving_average is None
    assert second.return_deviation is None


def test_rolling_window_evicts_old_prices_and_returns(fragment):
    op = RollingStatisticsOp(fragment, window_size=3, min_periods=2)
    prices = ("100", "110", "121", "133.1")

    results = [calculate(op, make_tick(index, price)) for index, price in enumerate(prices, 1)]

    assert results[1].moving_average == Decimal("105")
    assert results[2].moving_average == Decimal("110.3333333333333333333333333")
    assert results[3].moving_average == Decimal("121.3666666666666666666666667")
    assert results[3].price_observations == 3
    assert results[3].return_observations == 3
    assert results[3].simple_return == pytest.approx(0.1)
    assert results[3].return_deviation == pytest.approx(0.0, abs=1e-15)


def test_return_deviation_is_unannualized_sample_stdev_of_log_returns(fragment):
    op = RollingStatisticsOp(fragment, window_size=3, min_periods=2)
    calculate(op, make_tick(1, "100"))
    calculate(op, make_tick(2, "110"))

    result = calculate(op, make_tick(3, "110"))

    expected = statistics.stdev([math.log(1.1), 0.0])
    assert result.return_deviation == pytest.approx(expected)


def test_rolling_preserves_returns_smaller_than_float_spacing_around_one(fragment):
    op = RollingStatisticsOp(fragment, window_size=3, min_periods=2)
    calculate(op, make_tick(1, "1"))

    result = calculate(op, make_tick(2, "1.000000000000000000000000001"))

    assert result.simple_return == pytest.approx(1e-27)
    assert result.log_return == pytest.approx(math.log1p(1e-27))


def test_rolling_state_is_isolated_by_symbol(fragment):
    op = RollingStatisticsOp(fragment, window_size=3, min_periods=2)

    alpha_first = calculate(op, make_tick(1, "100", symbol="ALPHA"))
    beta_first = calculate(op, make_tick(2, "200", symbol="BETA"))
    alpha_second = calculate(op, make_tick(3, "110", symbol="ALPHA"))
    beta_second = calculate(op, make_tick(4, "180", symbol="BETA"))

    assert alpha_first.simple_return is None
    assert beta_first.simple_return is None
    assert alpha_second.simple_return == pytest.approx(0.1)
    assert beta_second.simple_return == pytest.approx(-0.1)
    assert alpha_second.moving_average == Decimal("105")
    assert beta_second.moving_average == Decimal("190")
    assert alpha_second.price_observations == beta_second.price_observations == 2


def test_rolling_rejects_a_non_increasing_global_sequence(fragment):
    op = RollingStatisticsOp(fragment, window_size=3, min_periods=2)
    calculate(op, make_tick(2, "100", symbol="ALPHA"))

    with pytest.raises(ValueError, match="sequence|order"):
        calculate(op, make_tick(1, "200", symbol="BETA"))


def test_rolling_rejects_a_globally_decreasing_timestamp(fragment):
    op = RollingStatisticsOp(fragment, window_size=3, min_periods=2)
    calculate(op, make_tick(1, "100", symbol="ALPHA", timestamp=START))

    with pytest.raises(ValueError, match="timestamp|order|decreasing"):
        calculate(
            op,
            make_tick(
                2,
                "200",
                symbol="BETA",
                timestamp=START - timedelta(microseconds=1),
            ),
        )


def test_rolling_rejects_currency_changes_for_a_symbol(fragment):
    op = RollingStatisticsOp(fragment, window_size=3, min_periods=2)
    calculate(op, make_tick(1, "100", currency="USD"))

    with pytest.raises(ValueError, match="currency"):
        calculate(op, make_tick(2, "101", currency="EUR"))


def test_rolling_start_clears_all_symbol_history(fragment):
    op = RollingStatisticsOp(fragment, window_size=3, min_periods=2)
    calculate(op, make_tick(1, "100"))
    before_reset = calculate(op, make_tick(2, "110"))
    assert before_reset.simple_return == pytest.approx(0.1)

    op.start()
    after_reset = calculate(op, make_tick(1, "200"))

    assert after_reset.simple_return is None
    assert after_reset.price_observations == 1
