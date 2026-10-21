# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host packet-sequence rules: seek and sequence-loss invalidation (from 4.3 ``test_equities``),
gap flushes before the mark reset, and stale packets."""

from __future__ import annotations

import numpy as np

from holoscan_finance.marketdata.bars import QuoteBars
from holoscan_finance.marketdata.records import CONDITION, EVENT_DTYPE, QUOTE, ReplayBatch
from holoscan_finance.marketdata.sequence import SequenceTracker


def event(ts, mid=100.0, kind=QUOTE):
    row = np.zeros(1, dtype=EVENT_DTYPE)
    row["timestamp_ns"] = int(ts * 1e9)
    row["kind"] = kind
    row["bid"], row["ask"] = mid - 0.01, mid + 0.01
    row["bid_size"] = row["ask_size"] = 10
    return row


def batch(rows, sequence=1, gen=1):
    return ReplayBatch(np.concatenate(rows), gen, 200, sequence)


class Recorder:
    """Records what the tracker feeds the bars."""

    def __init__(self):
        self.calls = []

    def observe(self, b):
        offsets = b.packet_offsets if b.packet_offsets is not None else [0]
        self.calls.append(("observe", b.sequence, len(offsets)))

    def reset_marks(self):
        self.calls.append(("reset",))


def test_seek_and_sequence_loss_invalidate_unknown_marks():
    bars = QuoteBars(["LONG_INSTRUMENT_NAME"], window_seconds=20, min_bars=2)
    tracker = SequenceTracker(generation=1)
    tracker.observe(bars, [batch([event(1), event(6, 110)])])
    assert tracker.expected == 3 and bars.events == 2
    tracker.observe(bars, [batch([event(11, kind=CONDITION)], sequence=4)])
    assert tracker.sequence_gaps == 1
    assert np.isnan(bars.mid).all()
    # Seek: a new generation starts a new tracker; queued packets of the old one are stale.
    bars = QuoteBars(["LONG_INSTRUMENT_NAME"], window_seconds=20, min_bars=2)
    tracker = SequenceTracker(generation=2)
    tracker.observe(bars, [batch([event(16, 120)], sequence=5)])
    assert bars.events == 0 and tracker.stale_packets == 1
    assert bars.window() is None
    assert tracker.expected == 1


def test_a_gap_flushes_pending_packets_before_resetting_marks():
    rec, tracker = Recorder(), SequenceTracker(generation=1)
    packets = [batch([event(t)], sequence=s) for t, s in ((1, 1), (2, 2), (3, 5), (4, 4), (5, 6))]
    tracker.observe(rec, packets)
    assert rec.calls == [("observe", 1, 2), ("reset",), ("observe", 5, 2)]
    assert (tracker.expected, tracker.sequence_gaps, tracker.stale_packets) == (7, 2, 1)
    tracker.observe(rec, [batch([event(6)], sequence=7), batch([event(7)], sequence=8, gen=3)])
    assert rec.calls[-1] == ("observe", 7, 1) and tracker.stale_packets == 2
    tracker.observe(rec, [])
    assert len(rec.calls) == 4


def test_wire_counts_advance_the_expected_sequence():
    rec, tracker = Recorder(), SequenceTracker(generation=1, expected=10)
    one = ReplayBatch(np.concatenate([event(1), event(2)]), 1, 0, 10, wire_count=4)
    tracker.observe(rec, [one, batch([event(3)], sequence=14)])
    assert tracker.expected == 15 and tracker.sequence_gaps == 0


def test_expected_zero_accepts_any_first_sequence():
    rec, tracker = Recorder(), SequenceTracker(generation=1, expected=0)
    tracker.observe(rec, [batch([event(1)], sequence=40)])
    assert (tracker.expected, tracker.sequence_gaps, rec.calls) == (41, 0, [("observe", 40, 1)])
