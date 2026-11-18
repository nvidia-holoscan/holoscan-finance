# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Equity observations to SBE entries and back: exact round trip (nulls as NaN, flags),
PriceNULL4/QtyNULL limits and the replay-specific rejections."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from holoscan_finance.equity import mapping as replay
from holoscan_finance.marketdata import sbe_codec as sbe
from holoscan_finance.marketdata.bars import QuoteBars
from holoscan_finance.marketdata.records import (
    CONDITION,
    EVENT_DTYPE,
    HALTED,
    QUOTE,
    TRADE,
    ReplayBatch,
)

MIDNIGHT = replay.day_start_ns("2026-06-01")
OPEN = MIDNIGHT + 34200 * 10**9
MAX = replay.MAX_PACKET_EVENTS
FIELDS = ("bid", "ask", "bid_size", "ask_size", "price", "volume")


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
    assert replay.day_start_ns("2026-06-01") == 1780286400 * 10**9  # EDT, UTC-4
    assert replay.day_start_ns("2025-01-02") == 1735794000 * 10**9  # EST, UTC-5


def random_rows(rng, n, universe=40):
    source = np.zeros(n, EVENT_DTYPE)
    source["timestamp_ns"] = OPEN + np.sort(rng.integers(0, 3 * 10**9, n))
    source["symbol_id"] = rng.integers(0, universe, n)
    source["kind"] = rng.choice([QUOTE, TRADE, CONDITION], n)
    for f in FIELDS:
        source[f] = np.nan
    q, t, c = (source["kind"] == k for k in (QUOTE, TRADE, CONDITION))
    source["bid"][q] = np.round(rng.uniform(1, 500, q.sum()), 4)
    source["bid"][np.flatnonzero(q)[0]] = -2.5  # negative prices are legal in PriceNULL4
    source["ask"][q] = np.round(source["bid"][q] + 0.01, 4)
    source["bid"][q & (rng.random(n) < 0.1)] = np.nan  # one-sided: null on the wire
    source["bid_size"][q] = rng.integers(0, 2000, q.sum())
    source["ask_size"][q] = rng.integers(1, 2000, q.sum())
    source["ask_size"][q & (rng.random(n) < 0.1)] = np.nan
    source["price"][t] = np.round(rng.uniform(1, 400_000, t.sum()), 4)
    source["volume"][t] = rng.integers(0, 10_000, t.sum())
    source["flags"][c] = rng.integers(0, 2, c.sum()) * HALTED
    source["flags"][q & (rng.random(n) < 0.2)] = HALTED  # halted quotes keep their prices
    source["flags"][t & (rng.random(n) < 0.2)] = HALTED
    return source


def test_round_trip_is_exact_and_an_observation_is_one_entry():
    source = random_rows(np.random.default_rng(4), MAX)
    data, count = replay.pack_events(source, 41, MIDNIGHT)
    assert count == MAX and len(data) == 24 + MAX * 48 <= sbe.MAX_PAYLOAD
    (batch,), decoded_count = decode(data, universe=40)
    assert decoded_count == count == batch.sequence_count and batch.sequence == 41
    got = batch.events
    for name in EVENT_DTYPE.names:
        np.testing.assert_array_equal(got[name], source[name], err_msg=name)
    assert np.isnan(got["bid"]).any() and (got["flags"][got["kind"] == QUOTE] == HALTED).any()
    assert (got["bid"][got["kind"] == QUOTE] < 0).any()
    # Every entry carries its own time and instrument, so a quote decodes without a partner.
    seq, entries = sbe.parse_packet(data)
    assert seq == 41 and len(entries) == MAX
    assert (entries["EntryType"] == source["kind"]).all()
    assert (entries["SecurityID"] == source["symbol_id"]).all()


def test_entries_of_is_the_record_the_packet_carries():
    source = random_rows(np.random.default_rng(9), 7)
    entries = replay.entries_of(source, MIDNIGHT)
    data, _ = replay.pack_events(source, 3, MIDNIGHT)
    assert data[24:] == entries.tobytes()
    halted = (source["flags"] & HALTED) != 0
    status = entries["SecurityTradingStatus"]
    assert (status[halted] == sbe.TRADING_HALT).all()
    assert (status[~halted & (source["kind"] == CONDITION)] == sbe.READY_TO_TRADE).all()
    assert (status[~halted & (source["kind"] != CONDITION)] == sbe.STATUS_NULL).all()
    assert (entries["LastPx"][source["kind"] == QUOTE] == sbe.PRICE_NULL).all()
    assert (entries["BidSize"][source["kind"] != QUOTE] == sbe.QTY_NULL).all()


def test_entries_preserve_analytics_inputs():
    source = rows()
    source["flags"][-1] = HALTED
    data, _ = replay.pack_events(source, 100, MIDNIGHT)
    batches, count = decode(data)
    assert count == 3 and batches[0].sequence_count == 3
    a, b = QuoteBars(["A"]), QuoteBars(["A"])
    a.observe(ReplayBatch(source, 1, 1000, 100))
    b.observe(batches[0])
    np.testing.assert_equal(a.mid, b.mid)
    np.testing.assert_equal(a.trade, b.trade)
    np.testing.assert_equal(a.volume, b.volume)


def test_missing_values_are_null_and_come_back_as_nan():
    source = rows((QUOTE, TRADE))
    source["bid"][0] = np.nan
    source["ask_size"][0] = np.nan
    source["volume"][1] = np.nan
    data, _ = replay.pack_events(source, 1, MIDNIGHT)
    seq, entries = sbe.parse_packet(data)
    assert entries["BidPx"][0] == sbe.PRICE_NULL and entries["OfferSize"][0] == sbe.QTY_NULL
    assert entries["LastQty"][1] == sbe.QTY_NULL
    events = decode(data)[0][0].events
    assert np.isnan(events["bid"][0]) and np.isnan(events["ask_size"][0])
    assert np.isnan(events["volume"][1]) and events["bid_size"][0] == 3
    # Fields that do not belong to the kind decode as NaN whatever the wire carries.
    loose = bytearray(data)
    struct.pack_into("<q", loose, 24 + 24, 123456)  # the quote's LastPx
    struct.pack_into("<I", loose, 24 + 48 + 32, 9)  # the trade's BidSize
    events = decode(bytes(loose))[0][0].events
    assert np.isnan(events["price"][0]) and np.isnan(events["bid_size"][1])


def test_unrepresentable_quantities_and_prices_are_rejected():
    for invalid, match in ((1.5, "QtyNULL"), (2**32 - 1, "QtyNULL"), (-3, "QtyNULL")):
        source = rows((QUOTE,))
        source["ask_size"][0] = invalid
        with pytest.raises(ValueError, match=match):
            replay.pack_events(source, 1, MIDNIGHT)
    for invalid in (np.inf, -np.inf, 1e15, -1e15):
        source = rows((TRADE,))
        source["price"][0] = invalid
        with pytest.raises(ValueError, match="Price|finite"):
            replay.pack_events(source, 1, MIDNIGHT)
    # A trade's quote fields and a quote's trade fields are not read, so anything goes there.
    source = rows((TRADE,))
    source["bid_size"][0] = 1.5
    assert replay.pack_events(source, 1, MIDNIGHT)[1] == 1


def test_halted_quotes_keep_prices_and_flag():
    source = rows((QUOTE,))
    source["flags"] = HALTED
    events = decode(replay.pack_events(source, 1, MIDNIGHT)[0])[0][0].events
    assert (events["bid"][0], events["ask"][0], events["flags"][0]) == (100, 101, HALTED)


def test_price_extremes_decode_as_float_then_divide():
    unit = bytearray(replay.pack_events(rows((QUOTE,)), 7, MIDNIGHT)[0])
    for bid, ask in ((2**63 - 1, sbe.PRICE_NULL + 1), (2**53 + 1, -(2**53) - 1), (1, -1)):
        struct.pack_into("<q", unit, 24 + 8, bid)
        struct.pack_into("<q", unit, 24 + 16, ask)
        events = decode(bytes(unit))[0][0].events
        assert events["bid"][0] == float(bid) / 10000.0
        assert events["ask"][0] == float(ask) / 10000.0
    struct.pack_into("<q", unit, 24 + 8, sbe.PRICE_NULL)
    assert np.isnan(decode(bytes(unit))[0][0].events["bid"][0])


def test_packing_rejects_what_the_replay_cannot_carry():
    with pytest.raises(ValueError, match=f"1..{MAX}"):
        replay.pack_events(rows(()), 1, MIDNIGHT)
    with pytest.raises(ValueError, match=f"1..{MAX}"):
        replay.pack_events(np.concatenate([rows()] * 11), 1, MIDNIGHT)
    unordered = rows((QUOTE, QUOTE))[::-1]
    with pytest.raises(ValueError, match="ordered"):
        replay.pack_events(unordered, 1, MIDNIGHT)
    late = rows((QUOTE,))
    late["timestamp_ns"] = MIDNIGHT + 86400 * 10**9
    with pytest.raises(ValueError, match="ordered"):
        replay.pack_events(late, 1, MIDNIGHT)
    early = rows((QUOTE,))
    early["timestamp_ns"] = MIDNIGHT - 1
    with pytest.raises(ValueError, match="ordered"):
        replay.pack_events(early, 1, MIDNIGHT)
    unknown = rows((QUOTE,))
    unknown["kind"] = 9
    with pytest.raises(ValueError, match="kind"):
        replay.pack_events(unknown, 1, MIDNIGHT)
    far = rows((QUOTE,))
    far["symbol_id"] = 0x10000
    with pytest.raises(ValueError, match="SecurityID"):
        replay.pack_events(far, 1, MIDNIGHT)
    for sequence in (0, 2**63 - 1):
        with pytest.raises(ValueError, match="sequence"):
            replay.pack_events(rows(), sequence, MIDNIGHT)


def mixed_rows(rng, n, unit_events):
    """Every kind and every field case: one-sided, crossed, negative and halted quotes, zero
    and missing sizes, signed zeros, extreme trade prices, halts and resumes, several
    seconds per packet and a timestamp step back at every packet boundary."""
    out = np.zeros(n, EVENT_DTYPE)
    steps = rng.integers(0, 7 * 10**8, n)
    steps[::unit_events] = rng.integers(0, 3 * 10**9, len(steps[::unit_events]))
    out["timestamp_ns"] = OPEN + np.cumsum(steps)
    out["symbol_id"] = rng.integers(0, 3000, n)
    out["kind"] = rng.choice([QUOTE, QUOTE, TRADE, CONDITION], n)
    for f in FIELDS:
        out[f] = np.nan
    q, t, c = (out["kind"] == k for k in (QUOTE, TRADE, CONDITION))
    out["bid"][q] = np.round(rng.uniform(-10, 500, q.sum()), 4)
    out["ask"][q] = np.round(rng.uniform(1, 500, q.sum()), 4)
    u = rng.random(n)
    out["bid"][q & (u < 0.05)] = np.nan
    out["bid"][q & (u > 0.5) & (u < 0.52)] = -0.0
    out["bid_size"][q] = rng.integers(0, 2000, q.sum())
    out["ask_size"][q] = rng.integers(0, 2000, q.sum())
    out["ask_size"][q & (u > 0.9) & (u < 0.92)] = np.nan
    out["price"][t] = np.round(rng.uniform(1, 400_000, t.sum()), 4)
    out["price"][t & (u < 0.02)] = (2**53) / sbe.PRICE_SCALE
    out["volume"][t] = rng.integers(0, 10_000, t.sum())
    out["volume"][t & (u > 0.98)] = sbe.QTY_NULL - 1
    out["flags"][(c | q) & (u < 0.3)] = HALTED
    return out


@pytest.mark.parametrize("unit_events", [MAX, 7, 1])
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
        assert expected == sequence + n  # one sequence number per observation
    # The sequence travels: the second packet's differs from a fresh pack of its rows.
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
    def both(source, match, unit_events=MAX):
        with pytest.raises(ValueError, match=match):
            replay.pack_events(source[:unit_events], 1, MIDNIGHT)
        with pytest.raises(ValueError, match=match):
            replay.pack_units(source, 1, MIDNIGHT, unit_events)

    source = rows((QUOTE, TRADE))
    source["ask_size"][0] = 1.5
    both(source, "QtyNULL")
    source = rows((QUOTE, TRADE))
    source["volume"][1] = 2**32 - 1
    both(source, "QtyNULL")
    source = rows((QUOTE,))
    source["bid_size"][0] = -1
    both(source, "QtyNULL")
    both(rows((QUOTE, QUOTE))[::-1].copy(), "ordered")
    late = rows((QUOTE,))
    late["timestamp_ns"] = MIDNIGHT + 86400 * 10**9
    both(late, "ordered")
    unknown = rows((QUOTE,))
    unknown["kind"] = 9
    both(unknown, "kind")
    huge = rows((QUOTE,))
    huge["bid"] = 1e15
    both(huge, "Price")
    infinite = rows((TRADE,))
    infinite["price"] = np.inf
    both(infinite, "finite")
    far = rows((QUOTE,))
    far["symbol_id"] = 0x10000
    both(far, "SecurityID")
    both(rows(()), f"1..{MAX}")
    with pytest.raises(ValueError, match=f"1..{MAX}"):
        replay.pack_units(rows(), 1, MIDNIGHT, MAX + 1)
    for sequence in (0, 2**63 - 1):
        with pytest.raises(ValueError, match="sequence"):
            replay.pack_units(rows(), sequence, MIDNIGHT)
    # A step back across a packet boundary is allowed, as pack_events never sees it.
    back = np.concatenate([rows((QUOTE,)), rows((QUOTE,))])
    back["timestamp_ns"][1] -= 1
    assert len(replay.pack_units(back, 1, MIDNIGHT, 1)[1]) == 2


def test_decoding_rejects_other_schemas_instruments_and_trading_states():
    data, _ = replay.pack_events(rows(), 1, MIDNIGHT)
    other = bytearray(data)
    struct.pack_into("<H", other, 4, sbe.SCHEMA_ID + 1)
    with pytest.raises(ValueError, match="MarketDataIncrementalRefresh"):
        decode(bytes(other))
    with pytest.raises(ValueError, match="instrument"):
        decode(data, universe=0)
    action = bytearray(replay.pack_events(rows((CONDITION,)), 1, MIDNIGHT)[0])
    action[24 + 47] = 21  # pre-open: not a state this replay uses
    with pytest.raises(ValueError, match="trading status"):
        decode(bytes(action))
    action[24 + 47] = sbe.STATUS_NULL  # a status entry needs a status
    with pytest.raises(ValueError, match="trading status"):
        decode(bytes(action))
    quote = bytearray(data)
    quote[24 + 46] = 4  # an entry type the schema does not define
    with pytest.raises(ValueError, match="entry type"):
        decode(bytes(quote))
    # A halted trade is a trade with the halt flag; a quote at ReadyToTrade has none.
    quote[24 + 46] = TRADE
    quote[24 + 47] = sbe.TRADING_HALT
    quote[24 + 48 + 47] = sbe.READY_TO_TRADE
    events = decode(bytes(quote))[0][0].events
    assert events["flags"].tolist() == [HALTED, 0, 0] and events["kind"][0] == TRADE
