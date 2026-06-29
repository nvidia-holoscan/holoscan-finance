# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline intraday portfolio-risk pipeline using deterministic market data."""

from __future__ import annotations

import logging
from collections.abc import MutableSequence, Sequence
from decimal import Decimal
from pathlib import Path

from holoscan.core import (
    Application,
    ExecutionContext,
    Fragment,
    InputContext,
    Operator,
    OperatorSpec,
    OutputContext,
)
from holoscan.finance import (
    MarketDataReplayOp,
    MarketTick,
    PortfolioRiskOp,
    PortfolioSnapshot,
    Position,
    RiskLimits,
    RollingStatisticsOp,
    load_market_ticks_csv,
)

logger = logging.getLogger(__name__)


DEFAULT_POSITIONS = (
    Position(
        symbol="ALPHA",
        quantity=Decimal("100"),
        average_price=Decimal("100"),
        currency="USD",
    ),
    Position(
        symbol="BETA",
        quantity=Decimal("-40"),
        average_price=Decimal("200"),
        currency="USD",
    ),
)

DEFAULT_RISK_LIMITS = RiskLimits(
    max_gross_exposure=Decimal("19000"),
    max_abs_net_exposure=Decimal("3000"),
    max_unrealized_loss=Decimal("1000"),
)
DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "synthetic_market_data.csv"


class ConsoleRiskSinkOp(Operator):
    """Log complete/partial portfolio snapshots and every limit breach.

    Parameters
    ----------
    snapshots
        Optional mutable collection populated with each received snapshot.
        This makes the example reusable in tests without coupling the finance
        package to a particular logging or persistence system.

    Ports
    -----
    snapshot : PortfolioSnapshot
        A partial or complete snapshot from :class:`PortfolioRiskOp`.
    """

    def __init__(
        self,
        fragment: Fragment,
        *args,
        snapshots: MutableSequence[PortfolioSnapshot] | None = None,
        name: str = "console_risk_sink",
        **kwargs,
    ) -> None:
        self.snapshots = snapshots if snapshots is not None else []
        super().__init__(fragment, *args, name=name, **kwargs)

    def setup(self, spec: OperatorSpec) -> None:
        """Declare the snapshot input port."""
        spec.input("snapshot")

    def compute(
        self,
        op_input: InputContext,
        op_output: OutputContext,
        context: ExecutionContext,
    ) -> None:
        """Record and log one portfolio snapshot."""
        del op_output, context
        snapshot = op_input.receive("snapshot")
        if not isinstance(snapshot, PortfolioSnapshot):
            raise TypeError("snapshot input must be a PortfolioSnapshot")
        self.snapshots.append(snapshot)

        status = "complete" if snapshot.complete else "partial"
        logger.info("%s portfolio snapshot: %s", status, snapshot)
        for breach in snapshot.breaches:
            logger.warning("risk limit breach: %s", breach)


class IntradayRiskApp(Application):
    """Replay market ticks and calculate rolling portfolio-risk snapshots."""

    def __init__(
        self,
        *,
        ticks: Sequence[MarketTick] | None = None,
        positions: Sequence[Position] | None = None,
        limits: RiskLimits | None = None,
        snapshots: MutableSequence[PortfolioSnapshot] | None = None,
    ) -> None:
        super().__init__()
        self._ticks = (
            tuple(ticks) if ticks is not None else load_market_ticks_csv(DEFAULT_DATA_PATH)
        )
        self._positions = tuple(positions) if positions is not None else DEFAULT_POSITIONS
        self._limits = limits if limits is not None else DEFAULT_RISK_LIMITS
        self.snapshots = snapshots if snapshots is not None else []

    def compose(self) -> None:
        """Create the finite, in-process operator graph."""
        replay = MarketDataReplayOp(
            self,
            ticks=self._ticks,
            name="market_data_replay",
        )
        statistics = RollingStatisticsOp(
            self,
            window_size=3,
            min_periods=2,
            name="rolling_statistics",
        )
        risk = PortfolioRiskOp(
            self,
            positions=self._positions,
            limits=self._limits,
            name="portfolio_risk",
        )
        sink = ConsoleRiskSinkOp(
            self,
            snapshots=self.snapshots,
            name="console_risk_sink",
        )

        self.add_flow(replay, statistics, {("tick", "tick")})
        self.add_flow(statistics, risk, {("features", "features")})
        self.add_flow(risk, sink, {("snapshot", "snapshot")})


def main() -> None:
    """Run the checked-in offline example."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    app = IntradayRiskApp()
    app.run()


if __name__ == "__main__":
    main()
