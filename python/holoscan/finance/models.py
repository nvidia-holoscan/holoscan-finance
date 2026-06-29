# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validated, immutable message contracts for :mod:`holoscan.finance`."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from math import isfinite


def _normalize_symbol(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("symbol must be a string")
    normalized = value.strip().upper()
    if not normalized:
        raise ValueError("symbol must not be empty")
    return normalized


def _normalize_currency(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("currency must be a string")
    normalized = value.strip().upper()
    if len(normalized) != 3 or not normalized.isascii() or not normalized.isalpha():
        raise ValueError("currency must be a three-letter ISO-style code")
    return normalized


def _require_decimal(value: Decimal, field_name: str, *, positive: bool = False) -> None:
    if not isinstance(value, Decimal):
        raise TypeError(f"{field_name} must be a Decimal")
    if not value.is_finite():
        raise ValueError(f"{field_name} must be finite")
    if positive and value <= 0:
        raise ValueError(f"{field_name} must be greater than zero")


def _require_optional_metric(value: float | None, field_name: str) -> None:
    if value is not None and (not isinstance(value, float) or not isfinite(value)):
        raise ValueError(f"{field_name} must be a finite float or None")


@dataclass(frozen=True, slots=True)
class MarketTick:
    """One normalized market-price observation.

    ``sequence`` is a strictly increasing replay identifier. ``timestamp`` must
    include a timezone and is normalized to UTC.
    """

    timestamp: datetime
    sequence: int
    symbol: str
    price: Decimal
    currency: str = "USD"

    def __post_init__(self) -> None:
        if not isinstance(self.timestamp, datetime):
            raise TypeError("timestamp must be a datetime")
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int):
            raise TypeError("sequence must be an integer")
        if self.sequence < 0:
            raise ValueError("sequence must be non-negative")
        _require_decimal(self.price, "price", positive=True)
        object.__setattr__(self, "timestamp", self.timestamp.astimezone(timezone.utc))
        object.__setattr__(self, "symbol", _normalize_symbol(self.symbol))
        object.__setattr__(self, "currency", _normalize_currency(self.currency))


@dataclass(frozen=True, slots=True)
class MarketFeatures:
    """Rolling features derived from a :class:`MarketTick`."""

    tick: MarketTick
    simple_return: float | None
    log_return: float | None
    moving_average: Decimal | None
    return_deviation: float | None
    price_observations: int
    return_observations: int

    def __post_init__(self) -> None:
        if not isinstance(self.tick, MarketTick):
            raise TypeError("tick must be a MarketTick")
        _require_optional_metric(self.simple_return, "simple_return")
        _require_optional_metric(self.log_return, "log_return")
        _require_optional_metric(self.return_deviation, "return_deviation")
        if self.simple_return is not None and self.simple_return <= -1.0:
            raise ValueError("simple_return must be greater than -1")
        if self.return_deviation is not None and self.return_deviation < 0:
            raise ValueError("return_deviation must be non-negative")
        if (self.simple_return is None) != (self.log_return is None):
            raise ValueError("simple_return and log_return must warm up together")
        if self.moving_average is not None:
            _require_decimal(self.moving_average, "moving_average", positive=True)
        for field_name, value in (
            ("price_observations", self.price_observations),
            ("return_observations", self.return_observations),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer")
            minimum = 1 if field_name == "price_observations" else 0
            if value < minimum:
                raise ValueError(f"{field_name} must be at least {minimum}")


@dataclass(frozen=True, slots=True)
class Position:
    """Static portfolio position used by :class:`PortfolioRiskOp`."""

    symbol: str
    quantity: Decimal
    average_price: Decimal
    currency: str = "USD"

    def __post_init__(self) -> None:
        _require_decimal(self.quantity, "quantity")
        if self.quantity == 0:
            raise ValueError("quantity must not be zero")
        _require_decimal(self.average_price, "average_price", positive=True)
        object.__setattr__(self, "symbol", _normalize_symbol(self.symbol))
        object.__setattr__(self, "currency", _normalize_currency(self.currency))


@dataclass(frozen=True, slots=True)
class PositionMark:
    """Marked value and unrealized P&L for one position."""

    position: Position
    market_price: Decimal
    market_value: Decimal
    unrealized_pnl: Decimal

    def __post_init__(self) -> None:
        if not isinstance(self.position, Position):
            raise TypeError("position must be a Position")
        _require_decimal(self.market_price, "market_price", positive=True)
        _require_decimal(self.market_value, "market_value")
        _require_decimal(self.unrealized_pnl, "unrealized_pnl")


@dataclass(frozen=True, slots=True)
class RiskLimits:
    """Optional positive limits evaluated on complete portfolio snapshots."""

    max_gross_exposure: Decimal | None = None
    max_abs_net_exposure: Decimal | None = None
    max_unrealized_loss: Decimal | None = None

    def __post_init__(self) -> None:
        for field_name, value in (
            ("max_gross_exposure", self.max_gross_exposure),
            ("max_abs_net_exposure", self.max_abs_net_exposure),
            ("max_unrealized_loss", self.max_unrealized_loss),
        ):
            if value is not None:
                _require_decimal(value, field_name, positive=True)


@dataclass(frozen=True, slots=True)
class RiskBreach:
    """A deterministic record of one breached portfolio limit."""

    code: str
    observed: Decimal
    limit: Decimal
    message: str

    def __post_init__(self) -> None:
        if not isinstance(self.code, str) or not self.code.strip():
            raise ValueError("code must not be empty")
        if not isinstance(self.message, str) or not self.message.strip():
            raise ValueError("message must not be empty")
        _require_decimal(self.observed, "observed")
        if self.observed < 0:
            raise ValueError("observed must be non-negative")
        _require_decimal(self.limit, "limit", positive=True)
        object.__setattr__(self, "code", self.code.strip().upper())


@dataclass(frozen=True, slots=True)
class PortfolioSnapshot:
    """Latest marked portfolio state emitted for one market observation."""

    timestamp: datetime
    sequence: int
    currency: str
    marks: tuple[PositionMark, ...]
    gross_exposure: Decimal
    net_exposure: Decimal
    unrealized_pnl: Decimal
    unpriced_symbols: tuple[str, ...]
    breaches: tuple[RiskBreach, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.timestamp, datetime):
            raise TypeError("timestamp must be a datetime")
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int):
            raise TypeError("sequence must be an integer")
        if self.sequence < 0:
            raise ValueError("sequence must be non-negative")
        _require_decimal(self.gross_exposure, "gross_exposure")
        if self.gross_exposure < 0:
            raise ValueError("gross_exposure must be non-negative")
        _require_decimal(self.net_exposure, "net_exposure")
        _require_decimal(self.unrealized_pnl, "unrealized_pnl")
        marks = tuple(self.marks)
        if any(not isinstance(mark, PositionMark) for mark in marks):
            raise TypeError("marks must contain only PositionMark values")
        breaches = tuple(self.breaches)
        if any(not isinstance(breach, RiskBreach) for breach in breaches):
            raise TypeError("breaches must contain only RiskBreach values")
        object.__setattr__(self, "timestamp", self.timestamp.astimezone(timezone.utc))
        object.__setattr__(self, "currency", _normalize_currency(self.currency))
        object.__setattr__(self, "marks", marks)
        object.__setattr__(self, "breaches", breaches)
        object.__setattr__(
            self,
            "unpriced_symbols",
            tuple(_normalize_symbol(symbol) for symbol in self.unpriced_symbols),
        )

    @property
    def complete(self) -> bool:
        """Return ``True`` when every configured position has a market price."""

        return not self.unpriced_symbols


def validate_market_ticks(ticks: Iterable[MarketTick]) -> tuple[MarketTick, ...]:
    """Validate and freeze a globally ordered market-tick stream."""

    try:
        validated = tuple(ticks)
    except TypeError as exc:
        raise TypeError("ticks must be an iterable of MarketTick values") from exc
    if not validated:
        raise ValueError("ticks must contain at least one MarketTick")

    previous: MarketTick | None = None
    for index, tick in enumerate(validated):
        if not isinstance(tick, MarketTick):
            raise TypeError(f"ticks[{index}] must be a MarketTick")
        if previous is not None:
            if tick.sequence <= previous.sequence:
                raise ValueError("market tick sequences must be strictly increasing")
            if tick.timestamp < previous.timestamp:
                raise ValueError("market tick timestamps must be non-decreasing")
        previous = tick
    return validated


__all__ = [
    "MarketFeatures",
    "MarketTick",
    "PortfolioSnapshot",
    "Position",
    "PositionMark",
    "RiskBreach",
    "RiskLimits",
    "validate_market_ticks",
]
