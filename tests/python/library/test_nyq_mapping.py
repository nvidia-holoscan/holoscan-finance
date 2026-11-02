# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""equity observations to MITCH and back: round trip, lossy-field policy, huge prices and the
replay-specific rejections."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from holoscan_finance.marketdata import mitch_codec as mitch
from holoscan_finance.marketdata.bars import QuoteBars
from holoscan_finance.marketdata.records import (
    CONDITION,
    EVENT_DTYPE,
    HALTED,
    QUOTE,
    TRADE,
    ReplayBatch,
)
from holoscan_finance.nyq import mapping as replay

MIDNIGHT = replay.day_start_ns("2025-04-01")
OPEN = MIDNIGHT + 34200 * 10**9


def rows(kinds=(QUOTE, TRADE, CONDITION)):
    out = np.zeros(len(kinds), dtype=EVENT_DTYPE)
    out["kind"] = kinds
    out["timestamp_ns"] = OPEN + np.arange(len(kinds)) * 10**9
    out["bid"], out["ask"], out["price"] = 100, 101, 100.5
    out["bid_size"], out["ask_size"], out["volume"] = 3, 4, 5
    return out


def decode(data, universe=1):
    return replay.decode_packet(data, 1000, generation=1, midnight_ns=MIDNIGHT, universe=universe)


def test_day_start_is_new_york_midnight_across_dst():
    assert replay.day_start_ns("2025-04-01") == 1743480000 * 10**9  # EDT, UTC-4
    assert replay.day_start_ns("2025-01-02") == 1735794000 * 10**9  # EST, UTC-5


def test_round_trip_every_kind_and_time_repeats_per_second():
    rng = np.random.default_rng(4)
    n = 32
    source = np.zeros(n, EVENT_DTYPE)
    source["timestamp_ns"] = OPEN + np.sort(rng.integers(0, 3 * 10**9, n))
    source["symbol_id"] = rng.integers(0, 40, n)
    source["kind"] = rng.choice([QUOTE, TRADE, CONDITION], n)
    for f in ("bid", "ask", "bid_size", "ask_size", "price", "volume"):
        source[f] = np.nan
    q, t, c = (source["kind"] == k for k in (QUOTE, TRADE, CONDITION))
    source["bid"][q] = np.round(rng.uniform(1, 500, q.sum()), 4)
    source["ask"][q] = np.round(source["bid"][q] + 0.01, 4)
    source["bid_size"][q] = rng.integers(1, 2000, q.sum())
    source["ask_size"][q] = rng.integers(1, 2000, q.sum())
    source["price"][t] = np.round(rng.uniform(1, 900_000, t.sum()), 6)
    source["volume"][t] = rng.integers(1, 10_000, t.sum())
    source["flags"][c] = rng.integers(0, 2, c.sum()) * HALTED
    data, count = replay.pack_events(source, 41, MIDNIGHT)
    seconds = (source["timestamp_ns"] - MIDNIGHT) // 10**9
    assert count == n + len(np.unique(seconds))  # one Time per second change
    (batch,), decoded_count = decode(data, universe=40)
    assert decoded_count == count == batch.sequence_count and batch.sequence == 41
    got = batch.events
    for name in ("timestamp_ns", "symbol_id", "kind", "flags", "internal_sequence"):
        np.testing.assert_array_equal(got[name], source[name], err_msg=name)
    for name in ("bid", "ask", "bid_size", "ask_size", "price", "volume"):
        np.testing.assert_array_equal(got[name], source[name], err_msg=name)


def test_messages_preserve_analytics_inputs():
    source = rows()
    source["flags"][-1] = HALTED
    data, _ = replay.pack_events(source, 100, MIDNIGHT)
    batches, count = decode(data)
    assert count == 6 and batches[0].sequence_count == 6
    a, b = QuoteBars(["A"]), QuoteBars(["A"])
    a.observe(ReplayBatch(source, 1, 1000, 100))
    b.observe(batches[0])
    np.testing.assert_equal(a.mid, b.mid)
    np.testing.assert_equal(a.trade, b.trade)
    np.testing.assert_equal(a.volume, b.volume)


def test_missing_values_and_unrepresentable_quantities():
    source = rows((QUOTE, TRADE))
    source["bid"][0] = np.nan
    source["volume"][1] = np.nan
    data, _ = replay.pack_events(source, 1, MIDNIGHT)
    events = decode(data)[0][0].events
    assert events["bid"][0] == 0 and events["volume"][1] == 0
    for invalid in (1.5, 2**32):
        source["ask_size"][0] = invalid
        with pytest.raises(ValueError, match="UInt32"):
            replay.pack_events(source, 1, MIDNIGHT)


def test_halted_quotes_carry_no_prices():
    source = rows((QUOTE,))
    source["flags"] = HALTED
    events = decode(replay.pack_events(source, 1, MIDNIGHT)[0])[0][0].events
    assert (events["bid"][0], events["ask"][0], events["flags"][0]) == (0, 0, 0)


@pytest.mark.parametrize("magnitude", [2**53 + 1, 2**60 + 12345, 2**63 - 1, 123456789012345678])
def test_huge_prices_round_like_python_integer_division(magnitude):
    unit = bytearray(replay.pack_events(rows((QUOTE,)), 7, MIDNIGHT)[0])
    struct.pack_into("<Q", unit, 24, magnitude | (1 << 63))  # bid, negative
    struct.pack_into("<Q", unit, 36, magnitude)  # ask
    events = decode(bytes(unit))[0][0].events
    assert events["bid"][0] == -(magnitude / mitch.SCALE)
    assert events["ask"][0] == magnitude / mitch.SCALE


def test_packing_rejects_what_the_replay_cannot_carry():
    with pytest.raises(ValueError, match="1..32"):
        replay.pack_events(rows(()), 1, MIDNIGHT)
    with pytest.raises(ValueError, match="1..32"):
        replay.pack_events(np.concatenate([rows()] * 11), 1, MIDNIGHT)
    unordered = rows((QUOTE, QUOTE))[::-1]
    with pytest.raises(ValueError, match="ordered"):
        replay.pack_events(unordered, 1, MIDNIGHT)
    late = rows((QUOTE,))
    late["timestamp_ns"] = MIDNIGHT + 86400 * 10**9
    with pytest.raises(ValueError, match="ordered"):
        replay.pack_events(late, 1, MIDNIGHT)
    unknown = rows((QUOTE,))
    unknown["kind"] = 9
    with pytest.raises(ValueError, match="kind"):
        replay.pack_events(unknown, 1, MIDNIGHT)


def mixed_rows(rng, n, unit_events):
    """Every kind and every lossy field case: one-sided, negative, infinite and halted
    quotes, zero and missing sizes, huge trade prices, halts and resumes, several Time
    seconds per unit and a timestamp step back at every unit boundary."""
    out = np.zeros(n, EVENT_DTYPE)
    steps = rng.integers(0, 7 * 10**8, n)
    steps[::unit_events] = rng.integers(0, 3 * 10**9, len(steps[::unit_events]))
    out["timestamp_ns"] = OPEN + np.cumsum(steps)
    out["symbol_id"] = rng.integers(0, 3000, n)
    out["kind"] = rng.choice([QUOTE, QUOTE, TRADE, CONDITION], n)
    for f in ("bid", "ask", "bid_size", "ask_size", "price", "volume"):
        out[f] = np.nan
    q, t, c = (out["kind"] == k for k in (QUOTE, TRADE, CONDITION))
    out["bid"][q] = np.round(rng.uniform(-5, 500, q.sum()), 4)
    out["ask"][q] = np.round(rng.uniform(1, 500, q.sum()), 4)
    u = rng.random(n)
    out["bid"][q & (u < 0.05)] = np.nan
    out["ask"][q & (u > 0.97)] = np.inf
    out["bid"][q & (u > 0.5) & (u < 0.52)] = -0.0
    out["bid_size"][q] = rng.integers(0, 2000, q.sum())
    out["ask_size"][q] = rng.integers(0, 2000, q.sum())
    out["ask_size"][q & (u > 0.9) & (u < 0.92)] = -3
    out["price"][t] = np.round(rng.uniform(1, 9e7, t.sum()), 6)
    out["price"][t & (u < 0.02)] = 2**62 / mitch.SCALE
    out["volume"][t] = rng.integers(0, 10_000, t.sum())
    out["flags"][(c | q) & (u < 0.3)] = HALTED
    return out


@pytest.mark.parametrize("unit_events", [32, 7, 1])
def test_pack_units_writes_every_unit_as_pack_events_does(unit_events):
    rng = np.random.default_rng(unit_events)
    for n in (1, unit_events, unit_events + 1, 1000):
        source = mixed_rows(rng, n, unit_events)
        sequence = int(rng.integers(1, 10**6))
        data, offsets, lengths, counts = replay.pack_units(source, sequence, MIDNIGHT, unit_events)
        assert offsets[-1] + lengths[-1] == len(data) and len(offsets) == -(-n // unit_events)
        expected = sequence
        for k, lo in enumerate(range(0, n, unit_events)):
            unit, count = replay.pack_events(source[lo : lo + unit_events], expected, MIDNIGHT)
            assert data[offsets[k] : offsets[k] + lengths[k]].tobytes() == unit, (n, k)
            assert counts[k] == count
            expected += count
    # Trade IDs carry the sequence: the second unit's differ from a fresh pack of its rows.
    data, offsets, lengths, counts = replay.pack_units(rows((TRADE, TRADE)), 10, MIDNIGHT, 1)
    assert (
        data[offsets[1] :].tobytes()
        != replay.pack_events(rows((TRADE, TRADE))[1:], 10, MIDNIGHT)[0]
    )
    assert (
        data[offsets[1] :].tobytes()
        == replay.pack_events(rows((TRADE, TRADE))[1:], 10 + counts[0], MIDNIGHT)[0]
    )


def test_pack_units_rejects_what_pack_events_rejects():
    def both(source, match, unit_events=32):
        with pytest.raises(ValueError, match=match):
            replay.pack_events(source[:unit_events], 1, MIDNIGHT)
        with pytest.raises(ValueError, match=match):
            replay.pack_units(source, 1, MIDNIGHT, unit_events)

    source = rows((QUOTE, TRADE))
    source["ask_size"][0] = 1.5
    both(source, "UInt32")
    source = rows((QUOTE, TRADE))
    source["volume"][1] = 2**32
    both(source, "UInt32")
    both(rows((QUOTE, QUOTE))[::-1].copy(), "ordered")
    late = rows((QUOTE,))
    late["timestamp_ns"] = MIDNIGHT + 86400 * 10**9
    both(late, "ordered")
    unknown = rows((QUOTE,))
    unknown["kind"] = 9
    both(unknown, "kind")
    huge = rows((QUOTE,))
    huge["bid"] = 1e12
    both(huge, "sign-magnitude")
    both(rows(()), "1..32")
    with pytest.raises(ValueError, match="1..32"):
        replay.pack_units(rows(), 1, MIDNIGHT, 33)
    with pytest.raises(ValueError, match="sequence"):
        replay.pack_units(rows(), 2**32, MIDNIGHT)
    # A step back across a unit boundary is allowed, as pack_events never sees it.
    back = np.concatenate([rows((QUOTE,)), rows((QUOTE,))])
    back["timestamp_ns"][1] -= 1
    assert len(replay.pack_units(back, 1, MIDNIGHT, 1)[1]) == 2


def test_decoding_rejects_other_groups_instruments_and_trade_semantics():
    data, _ = replay.pack_events(rows(), 1, MIDNIGHT)
    with pytest.raises(ValueError, match="group"):
        decode(mitch.pack_unit(mitch.parse_unit(data)[2], 1, group=ord("B")))
    with pytest.raises(ValueError, match="instrument"):
        decode(data, universe=0)
    trade = bytearray(replay.pack_events(rows((TRADE,)), 1, MIDNIGHT)[0])
    trade[8 + 6 + 32] = 1  # cross flag of the P message
    with pytest.raises(ValueError, match="trade semantics"):
        decode(bytes(trade))
