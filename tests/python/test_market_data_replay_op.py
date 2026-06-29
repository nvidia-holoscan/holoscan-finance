# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the finite market-data replay source."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest
from conftest import invoke

from holoscan.core import Operator, OperatorSpec
from holoscan.finance import MarketDataReplayOp, MarketTick

START = datetime(2026, 6, 27, 14, 30, tzinfo=timezone.utc)


def make_tick(sequence: int, symbol: str = "NVDA") -> MarketTick:
    return MarketTick(
        timestamp=START + timedelta(microseconds=sequence),
        sequence=sequence,
        symbol=symbol,
        price=Decimal("100") + sequence,
    )


def test_replay_operator_declares_only_the_tick_output(fragment):
    op = MarketDataReplayOp(fragment, ticks=[make_tick(1)])
    spec = OperatorSpec()

    op.setup(spec)

    assert set(spec.inputs) == set()
    assert set(spec.outputs) == {"tick"}
    assert isinstance(op, Operator)


def test_replay_emits_ticks_once_in_caller_supplied_order(fragment):
    ticks = [make_tick(10), make_tick(20, "MSFT"), make_tick(30)]
    op = MarketDataReplayOp(fragment, ticks=ticks)

    emitted = [invoke(op).emitted for _ in ticks]

    assert emitted == [(tick, "tick") for tick in ticks]


def test_replay_raises_if_manually_invoked_after_exhaustion(fragment):
    op = MarketDataReplayOp(fragment, ticks=[make_tick(1)])

    assert invoke(op).emitted == (make_tick(1), "tick")
    with pytest.raises(RuntimeError, match="exhaust|tick"):
        invoke(op)


def test_replay_takes_a_defensive_snapshot_of_the_input_sequence(fragment):
    ticks = [make_tick(1)]
    op = MarketDataReplayOp(fragment, ticks=ticks)
    ticks.append(make_tick(2))

    assert invoke(op).emitted == (make_tick(1), "tick")
    with pytest.raises(RuntimeError, match="exhaust|tick"):
        invoke(op)


def test_replay_start_resets_the_finite_source(fragment):
    tick = make_tick(1)
    op = MarketDataReplayOp(fragment, ticks=[tick])
    assert invoke(op).emitted == (tick, "tick")

    op.start()

    assert invoke(op).emitted == (tick, "tick")


@pytest.mark.parametrize("ticks", [None, [], [object()]])
def test_replay_rejects_invalid_tick_collections(fragment, ticks):
    with pytest.raises((TypeError, ValueError), match="ticks|MarketTick"):
        MarketDataReplayOp(fragment, ticks=ticks)


def test_replay_rejects_non_increasing_sequences(fragment):
    with pytest.raises(ValueError, match="sequence|increasing"):
        MarketDataReplayOp(fragment, ticks=[make_tick(2), make_tick(1)])


def test_replay_rejects_decreasing_timestamps(fragment):
    later = make_tick(1)
    earlier = make_tick(2)
    earlier = MarketTick(
        timestamp=later.timestamp - timedelta(seconds=1),
        sequence=earlier.sequence,
        symbol=earlier.symbol,
        price=earlier.price,
    )

    with pytest.raises(ValueError, match="timestamp|order|decreasing"):
        MarketDataReplayOp(fragment, ticks=[later, earlier])
