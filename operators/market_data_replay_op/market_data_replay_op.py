# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Finite, deterministic market-data replay for Holoscan pipelines."""

from __future__ import annotations

from collections.abc import Sequence

from holoscan.conditions import CountCondition
from holoscan.core import (
    ExecutionContext,
    Fragment,
    InputContext,
    Operator,
    OperatorSpec,
    OutputContext,
)
from holoscan.finance.models import MarketTick, validate_market_ticks


class MarketDataReplayOp(Operator):
    """Emit a finite, validated sequence of market ticks in source order.

    Ports
    -----
    tick : output
        One :class:`~holoscan.finance.MarketTick` per compute invocation.

    Parameters
    ----------
    ticks
        Non-empty sequence with strictly increasing sequence numbers and
        non-decreasing timestamps.

    Notes
    -----
    The operator installs a ``CountCondition`` equal to ``len(ticks)``. Other
    scheduling conditions, such as ``PeriodicCondition``, may be supplied as
    positional arguments and compose with the finite replay condition.
    """

    def __init__(
        self,
        fragment: Fragment,
        *args,
        ticks: Sequence[MarketTick],
        name: str = "market_data_replay",
        **kwargs,
    ) -> None:
        self._ticks = validate_market_ticks(ticks)
        self._index = 0
        replay_count = CountCondition(fragment, count=len(self._ticks))
        super().__init__(fragment, replay_count, *args, name=name, **kwargs)

    @property
    def record_count(self) -> int:
        """Number of ticks in this replay."""

        return len(self._ticks)

    def setup(self, spec: OperatorSpec) -> None:
        spec.output("tick")

    def start(self) -> None:
        self._index = 0

    def compute(
        self,
        op_input: InputContext,
        op_output: OutputContext,
        context: ExecutionContext,
    ) -> None:
        del op_input, context
        if self._index >= len(self._ticks):
            raise RuntimeError("market replay invoked after its finite sequence was exhausted")
        op_output.emit(self._ticks[self._index], "tick")
        self._index += 1


__all__ = ["MarketDataReplayOp"]
