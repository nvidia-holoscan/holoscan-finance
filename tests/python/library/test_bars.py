# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Event-time quote bars and the host-mirror methods: ``window_parts``, ``reset_marks``,
``counts`` and ``symbol_arrays``."""

from __future__ import annotations

import numpy as np
import pytest

from holoscan_finance.marketdata.bars import QuoteBars
from holoscan_finance.marketdata.covariance import CovarianceEngine
from holoscan_finance.marketdata.records import (
    CONDITION,
    EVENT_DTYPE,
    HALTED,
    QUOTE,
    TRADE,
    ReplayBatch,
    combine_batches,
)


def event(ts, mid=100.0, kind=QUOTE, flags=0, symbol=0, size=10):
    row = np.zeros(1, dtype=EVENT_DTYPE)
    row["timestamp_ns"] = int(ts * 1e9)
    row["symbol_id"] = symbol
    row["kind"], row["flags"] = kind, flags
    row["bid"], row["ask"] = mid - 0.01, mid + 0.01
    row["bid_size"] = row["ask_size"] = size
    row["price"], row["volume"] = mid, 2
    return row


def batch(rows, sequence=1, gen=1):
    return ReplayBatch(np.concatenate(rows), gen, 200, sequence)


def test_quote_returns_exclude_trades_and_stale_marks():
    bars = QuoteBars(["A"], window_seconds=20, min_bars=2, stale_seconds=5)
    bars.observe(
        batch(
            [
                event(1, 100),
                event(6, 110),
                event(7, 999, TRADE),
                event(11, 121),
                event(16, kind=CONDITION),
                event(21, kind=CONDITION),
            ]
        )
    )
    np.testing.assert_allclose(np.array(bars.rows)[1:3, 0], np.log(1.1))
    assert np.isnan(bars.rows[-1][0])
    assert bars.trade[0] == 999 and bars.volume[0] == 2


def test_halts_invalid_quotes_and_packet_boundaries_do_not_change_bars():
    rows = [
        event(1),
        event(2, kind=CONDITION, flags=HALTED),
        event(3, 101),
        event(6, 105),
        event(7, kind=CONDITION, flags=HALTED),
        event(11, 110),
        event(12, 111, size=0),
        event(16, 120),
        event(21, 123),
        event(26, kind=CONDITION),
    ]
    one = QuoteBars(["A"], window_seconds=30, min_bars=2)
    split = QuoteBars(["A"], window_seconds=30, min_bars=2)
    one.observe(batch(rows))
    packets = []
    for i, row in enumerate(rows):
        b = ReplayBatch(row, 1, i * 100 + 10, i + 1)
        packets.append(b)
        split.observe(b)
    merged = QuoteBars(["A"], window_seconds=30, min_bars=2)
    merged.observe(combine_batches(packets))
    np.testing.assert_equal(np.asarray(one.rows), np.asarray(split.rows))
    np.testing.assert_equal(np.asarray(one.rows), np.asarray(merged.rows))
    assert merged.bar_recv_ns == split.bar_recv_ns == 910
    assert one.marks[0][0] == 101
    assert np.isnan(one.marks[1][0]) and np.isnan(one.marks[2][0])
    assert one.invalid_quotes == 1


def test_missing_oldest_bar_does_not_shorten_an_established_window():
    bars = QuoteBars(["A"], window_seconds=20, min_bars=2)
    bars.observe(
        batch(
            [
                event(1),
                event(6, size=0),
                event(11, 102),
                event(16, 103),
                event(21, 104),
                event(26, 105),
            ]
        )
    )
    rows, times = bars.window()
    assert len(rows) == 4
    assert times.tolist() == [10_000_000_000, 15_000_000_000, 20_000_000_000, 25_000_000_000]
    assert np.isnan(rows[0, 0])
    assert CovarianceEngine().update(rows, times)["eligible"] == 0


def test_invalid_settings_are_rejected():
    for bad in (
        dict(bar_seconds=0),
        dict(window_seconds=301),
        dict(window_seconds=5),
        dict(stale_seconds=4),
        dict(max_spread_bps=float("nan")),
        dict(max_spread_bps=0),
        dict(min_bars=1),
        dict(min_bars=61),
    ):
        with pytest.raises(ValueError):
            QuoteBars(["A"], **bad)


def test_window_parts_is_the_window_without_stacking():
    bars = QuoteBars(["A", "B"], window_seconds=20, min_bars=2)
    assert bars.window_parts() is None
    for t in range(1, 40, 5):
        bars.observe(batch([event(t, 100 + t), event(t, 50 + t, symbol=1)]))
        stacked, parts = bars.window(), bars.window_parts()
        assert (stacked is None) == (parts is None)
        if parts is None:
            continue
        rows, times = parts
        newest = list(bars.rows)[-len(rows) :]
        assert isinstance(rows, list) and all(r is a for r, a in zip(rows, newest))
        np.testing.assert_array_equal(np.stack(rows), stacked[0])
        np.testing.assert_array_equal(times, stacked[1])
        times[0] = -1  # an owned copy
        assert bars.times[-len(rows)] != -1


def test_reset_marks_forces_a_fresh_return_endpoint():
    bars = QuoteBars(["A"], window_seconds=20, min_bars=2)
    bars.observe(batch([event(1, 100), event(6, 110)]))
    stored = [r.copy() for r in bars.rows]
    bars.reset_marks()
    assert np.isnan(bars.mid).all() and np.isnan(bars.previous).all()
    for a, b in zip(stored, bars.rows):
        np.testing.assert_array_equal(a, b)  # closed bars are untouched
    bars.observe(batch([event(11, 120), event(16, 130), event(21, 140)], sequence=3))
    assert bars.times[-3] == 10 * 10**9 and np.isnan(bars.rows[-3][0])  # 110 was forgotten
    assert np.isfinite(bars.rows[-1][0])


def test_counts_and_symbol_arrays_are_host_copies():
    bars = QuoteBars(["A", "B"], window_seconds=20, min_bars=2)
    bars.observe(
        batch([event(1, 100), event(2, 50, TRADE, symbol=1), event(3, 101, size=0), event(4)])
    )
    bars.observe(batch([event(3.5, 99)], sequence=5))  # late: before last_event_ns
    counts = bars.counts()
    assert counts.dtype == np.uint64
    assert counts.tolist() == [4, 3, 1, 1, 1]
    assert counts.tolist() == [
        bars.events,
        bars.quotes,
        bars.trades,
        bars.invalid_quotes,
        bars.late_events,
    ]
    mid, trade, volume, quote_time = bars.symbol_arrays()
    for got, state in zip(
        (mid, trade, volume, quote_time), (bars.mid, bars.trade, bars.volume, bars.quote_time)
    ):
        np.testing.assert_array_equal(got, state)
        assert got is not state and not np.shares_memory(got, state)
    assert quote_time.dtype == np.int64 and quote_time[0] == 4 * 10**9
    assert trade[1] == 50 and np.isnan(trade[0])
