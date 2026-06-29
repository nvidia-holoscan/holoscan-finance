# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-symbol rolling market statistics for Holoscan pipelines."""

from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from holoscan.core import (
    ExecutionContext,
    Fragment,
    InputContext,
    Operator,
    OperatorSpec,
    OutputContext,
)
from holoscan.finance.models import MarketFeatures, MarketTick


@dataclass(slots=True)
class _SymbolState:
    prices: deque[Decimal]
    log_returns: deque[float]
    currency: str
    last_order: tuple[datetime, int] | None = None


class RollingStatisticsOp(Operator):
    """Compute independent rolling statistics for each market symbol.

    ``return_deviation`` is the sample standard deviation of per-observation
    log returns. It is deliberately not annualized and is not VaR or a
    volatility forecast.

    Ports
    -----
    tick : input
        A validated :class:`~holoscan.finance.MarketTick`.
    features : output
        A :class:`~holoscan.finance.MarketFeatures` for every input tick.
    """

    def __init__(
        self,
        fragment: Fragment,
        *args,
        window_size: int = 20,
        min_periods: int | None = None,
        name: str = "rolling_statistics",
        **kwargs,
    ) -> None:
        if isinstance(window_size, bool) or not isinstance(window_size, int) or window_size < 2:
            raise ValueError("window_size must be an integer of at least 2")
        if min_periods is None:
            min_periods = window_size
        if isinstance(min_periods, bool) or not isinstance(min_periods, int):
            raise TypeError("min_periods must be an integer")
        if not 2 <= min_periods <= window_size:
            raise ValueError("min_periods must be between 2 and window_size")

        self.window_size = window_size
        self.min_periods = min_periods
        self._states: dict[str, _SymbolState] = {}
        self._last_global_timestamp: datetime | None = None
        self._last_global_sequence: int | None = None
        super().__init__(fragment, *args, name=name, **kwargs)

    def setup(self, spec: OperatorSpec) -> None:
        spec.input("tick")
        spec.output("features")

    def start(self) -> None:
        self._states.clear()
        self._last_global_timestamp = None
        self._last_global_sequence = None

    def _state_for(self, tick: MarketTick) -> _SymbolState:
        symbol = tick.symbol
        state = self._states.get(symbol)
        if state is None:
            state = _SymbolState(
                prices=deque(maxlen=self.window_size),
                log_returns=deque(maxlen=self.window_size),
                currency=tick.currency,
            )
            self._states[symbol] = state
        elif state.currency != tick.currency:
            raise ValueError(f"currency changed for symbol {symbol}")
        return state

    def compute(
        self,
        op_input: InputContext,
        op_output: OutputContext,
        context: ExecutionContext,
    ) -> None:
        del context
        tick = op_input.receive("tick")
        if not isinstance(tick, MarketTick):
            raise TypeError("tick input must be a MarketTick")

        if self._last_global_sequence is not None:
            if tick.sequence <= self._last_global_sequence:
                raise ValueError("market tick sequence must be globally increasing")
            if tick.timestamp < self._last_global_timestamp:
                raise ValueError("market tick timestamp must be globally non-decreasing")

        state = self._state_for(tick)
        order = (tick.timestamp, tick.sequence)
        if state.last_order is not None and order <= state.last_order:
            raise ValueError(f"non-monotonic market tick for symbol {tick.symbol}")

        simple_return: float | None = None
        log_return: float | None = None
        if state.prices:
            previous_price = state.prices[-1]
            decimal_return = (tick.price - previous_price) / previous_price
            simple_return = float(decimal_return)
            if not math.isfinite(simple_return) or simple_return <= -1.0:
                raise ValueError("price return cannot be represented as a finite float")
            # log1p preserves small relative moves that would disappear if the
            # Decimal price ratio were rounded to float near 1.0 first.
            log_return = math.log1p(simple_return)
            state.log_returns.append(log_return)

        state.prices.append(tick.price)
        state.last_order = order
        self._last_global_timestamp = tick.timestamp
        self._last_global_sequence = tick.sequence

        moving_average = None
        if len(state.prices) >= self.min_periods:
            moving_average = sum(state.prices, start=Decimal(0)) / Decimal(len(state.prices))

        return_deviation = None
        if len(state.log_returns) >= self.min_periods:
            return_deviation = statistics.stdev(state.log_returns)

        features = MarketFeatures(
            tick=tick,
            simple_return=simple_return,
            log_return=log_return,
            moving_average=moving_average,
            return_deviation=return_deviation,
            price_observations=len(state.prices),
            return_observations=len(state.log_returns),
        )
        op_output.emit(features, "features")


__all__ = ["RollingStatisticsOp"]
