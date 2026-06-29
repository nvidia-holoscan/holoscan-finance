# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public API for the Holoscan Finance module."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

from .io import load_market_ticks_csv
from .models import (
    MarketFeatures,
    MarketTick,
    PortfolioSnapshot,
    Position,
    PositionMark,
    RiskBreach,
    RiskLimits,
)

if TYPE_CHECKING:
    from .market_data_replay_op import MarketDataReplayOp
    from .portfolio_risk_op import PortfolioRiskOp
    from .rolling_statistics_op import RollingStatisticsOp

_LAZY_OPERATOR_EXPORTS = {
    "MarketDataReplayOp": (".market_data_replay_op", "MarketDataReplayOp"),
    "PortfolioRiskOp": (".portfolio_risk_op", "PortfolioRiskOp"),
    "RollingStatisticsOp": (".rolling_statistics_op", "RollingStatisticsOp"),
}


def __getattr__(name: str) -> Any:
    """Load Holoscan-dependent operator classes only when requested."""

    target = _LAZY_OPERATOR_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__all__ = [
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
]
