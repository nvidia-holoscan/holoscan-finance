# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static-portfolio marking and deterministic risk-limit checks."""

from __future__ import annotations

from collections.abc import Sequence
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
from holoscan.finance.models import (
    MarketFeatures,
    PortfolioSnapshot,
    Position,
    PositionMark,
    RiskBreach,
    RiskLimits,
)


class PortfolioRiskOp(Operator):
    """Mark a static, single-currency portfolio and evaluate simple limits.

    Ports
    -----
    features : input
        :class:`~holoscan.finance.MarketFeatures` carrying the latest price.
    snapshot : output
        One :class:`~holoscan.finance.PortfolioSnapshot` per input message.

    Limit evaluation begins only after every configured position has received
    at least one price. Equal-to-limit values are allowed; a breach requires a
    strict ``observed > limit`` comparison.
    """

    def __init__(
        self,
        fragment: Fragment,
        *args,
        positions: Sequence[Position],
        limits: RiskLimits | None = None,
        name: str = "portfolio_risk",
        **kwargs,
    ) -> None:
        self._positions = tuple(positions)
        if not self._positions:
            raise ValueError("positions must contain at least one Position")
        if any(not isinstance(position, Position) for position in self._positions):
            raise TypeError("positions must contain only Position values")
        symbols = [position.symbol for position in self._positions]
        if len(symbols) != len(set(symbols)):
            raise ValueError("positions must have unique symbols")
        currencies = {position.currency for position in self._positions}
        if len(currencies) != 1:
            raise ValueError(
                "all positions must use one currency; FX conversion is not implemented"
            )
        if limits is not None and not isinstance(limits, RiskLimits):
            raise TypeError("limits must be a RiskLimits value or None")

        self.currency = currencies.pop()
        self.limits = limits or RiskLimits()
        self._positions_by_symbol = {position.symbol: position for position in self._positions}
        self._latest_prices: dict[str, Decimal] = {}
        self._last_timestamp: datetime | None = None
        self._last_sequence: int | None = None
        super().__init__(fragment, *args, name=name, **kwargs)

    def setup(self, spec: OperatorSpec) -> None:
        spec.input("features")
        spec.output("snapshot")

    def start(self) -> None:
        self._latest_prices.clear()
        self._last_timestamp = None
        self._last_sequence = None

    def _breaches(
        self,
        gross_exposure: Decimal,
        net_exposure: Decimal,
        unrealized_pnl: Decimal,
    ) -> tuple[RiskBreach, ...]:
        breaches: list[RiskBreach] = []
        if (
            self.limits.max_gross_exposure is not None
            and gross_exposure > self.limits.max_gross_exposure
        ):
            breaches.append(
                RiskBreach(
                    code="GROSS_EXPOSURE",
                    observed=gross_exposure,
                    limit=self.limits.max_gross_exposure,
                    message="gross exposure exceeded its configured limit",
                )
            )

        absolute_net = abs(net_exposure)
        if (
            self.limits.max_abs_net_exposure is not None
            and absolute_net > self.limits.max_abs_net_exposure
        ):
            breaches.append(
                RiskBreach(
                    code="ABS_NET_EXPOSURE",
                    observed=absolute_net,
                    limit=self.limits.max_abs_net_exposure,
                    message="absolute net exposure exceeded its configured limit",
                )
            )

        unrealized_loss = max(-unrealized_pnl, Decimal(0))
        if (
            self.limits.max_unrealized_loss is not None
            and unrealized_loss > self.limits.max_unrealized_loss
        ):
            breaches.append(
                RiskBreach(
                    code="UNREALIZED_LOSS",
                    observed=unrealized_loss,
                    limit=self.limits.max_unrealized_loss,
                    message="unrealized loss exceeded its configured limit",
                )
            )
        return tuple(breaches)

    def compute(
        self,
        op_input: InputContext,
        op_output: OutputContext,
        context: ExecutionContext,
    ) -> None:
        del context
        features = op_input.receive("features")
        if not isinstance(features, MarketFeatures):
            raise TypeError("features input must be a MarketFeatures value")
        tick = features.tick
        if self._last_sequence is not None:
            if tick.sequence <= self._last_sequence:
                raise ValueError("market feature sequence must be globally increasing")
            if tick.timestamp < self._last_timestamp:
                raise ValueError("market feature timestamp must be globally non-decreasing")
        if tick.currency != self.currency:
            raise ValueError(
                f"market tick currency {tick.currency} does not match portfolio {self.currency}"
            )
        if tick.symbol in self._positions_by_symbol:
            self._latest_prices[tick.symbol] = tick.price
        self._last_timestamp = tick.timestamp
        self._last_sequence = tick.sequence

        marks: list[PositionMark] = []
        unpriced_symbols: list[str] = []
        for position in self._positions:
            market_price = self._latest_prices.get(position.symbol)
            if market_price is None:
                unpriced_symbols.append(position.symbol)
                continue
            market_value = position.quantity * market_price
            marks.append(
                PositionMark(
                    position=position,
                    market_price=market_price,
                    market_value=market_value,
                    unrealized_pnl=position.quantity * (market_price - position.average_price),
                )
            )

        gross_exposure = sum((abs(mark.market_value) for mark in marks), start=Decimal(0))
        net_exposure = sum((mark.market_value for mark in marks), start=Decimal(0))
        unrealized_pnl = sum((mark.unrealized_pnl for mark in marks), start=Decimal(0))
        breaches = ()
        if not unpriced_symbols:
            breaches = self._breaches(gross_exposure, net_exposure, unrealized_pnl)

        snapshot = PortfolioSnapshot(
            timestamp=tick.timestamp,
            sequence=tick.sequence,
            currency=self.currency,
            marks=tuple(marks),
            gross_exposure=gross_exposure,
            net_exposure=net_exposure,
            unrealized_pnl=unrealized_pnl,
            unpriced_symbols=tuple(unpriced_symbols),
            breaches=breaches,
        )
        op_output.emit(snapshot, "snapshot")


__all__ = ["PortfolioRiskOp"]
