# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stats objects: ``IngestStats`` buckets, ring views, readiness and the drained predicate;
the stall helpers; ``BarsStats`` and ``RiskStats``."""

from __future__ import annotations

import pytest

from holoscan_finance.payloads import BarCounters
from holoscan_finance.telemetry.stats import BarsStats, IngestStats, RiskStats, SourceProbe


def probe(*, alive=True, pending=0, peak=0):
    calls = []
    return (
        SourceProbe(
            alive=lambda: alive,
            pending=lambda: pending,
            ring_peak=lambda: peak,
            reset_peak=lambda: calls.append("reset_peak"),
        ),
        calls,
    )


def test_reset_zeroes_every_counter_for_a_new_run():
    st = IngestStats()
    st.packets = st.messages = st.dropped = st.retries = 5
    st.error, st.eof, st.opened = "boom", True, True
    with st.lock:
        st.ring_packets = st.ring_msgs = 3
        st.add_bucket(10**9, 7)
    st.reset(front_end="replay")
    assert st.front_end == "replay"
    assert (st.packets, st.messages, st.dropped, st.retries) == (0, 0, 0, 0)
    assert (st.error, st.eof, st.opened, st.replay) == (None, False, False, None)
    assert st.ring_fill() == 0 and st.pending_packets() == 0 and st.feed_buckets() == []


def test_arrival_buckets():
    st = IngestStats()
    with st.lock:
        for recv_ns, msgs in ((0, 1), (99_999_999, 2), (100_000_000, 4), (350_000_000, 8)):
            st.add_bucket(recv_ns, msgs)
    assert st.feed_buckets() == [[0, 3], [100, 4], [300, 8]]
    assert st.feed_buckets(last=2) == [[100, 4], [300, 8]]
    with st.lock:
        for b in range(700):
            st.add_bucket(b * IngestStats.BUCKET_NS, 1)
    assert len(st.buckets) == 600 and st.feed_buckets()[-1] == [699 * 100, 1]


def test_ring_views_prefer_the_receiver_probe():
    st = IngestStats()
    with st.lock:
        st.ring_packets, st.ring_msgs, st.ring_peak_local = 2, 9, 4
    assert (st.pending_packets(), st.ring_fill(), st.ring_peak()) == (2, 9, 4)
    assert not st.drain_alive()
    p, calls = probe(pending=7, peak=11)
    st.attach(p)
    assert (st.pending_packets(), st.ring_peak(), st.drain_alive()) == (7, 11, True)
    assert st.ring_fill() == 9  # messages staged on the host side stay the stats' own
    st.reset_stalls()
    assert calls == ["reset_peak"] and st.ring_peak_local == 0
    st.detach()
    assert st.pending_packets() == 2 and st.ring_peak() == 0


def test_ready_and_drained():
    st = IngestStats()
    p, _ = probe()
    assert not st.ready()
    st.opened = True
    st.attach(p)
    assert st.ready()
    st.error = "rx failed"
    assert not st.ready()
    st.error = None
    assert not st.drained()
    st.eof = True
    assert st.drained()
    st.held = 1
    assert not st.drained()
    st.held = 0
    st.count_packet_batch()
    assert st.packet_batches == 1 and not st.drained()  # a batch not decoded yet
    st.packet_batches_decoded = 1
    assert st.drained()
    busy, _ = probe(pending=3)
    st.attach(busy)
    assert not st.drained()


def test_stall_events_keep_long_stalls_and_gaps_only():
    st = IngestStats()
    st.stall_event("heartbeat", 10.0)
    st.stall_event("heartbeat", 51.26, at_s=1)
    st.stall_event("gap", 1.0)
    events = st.stalls["events"]
    assert [(e["kind"], e["ms"]) for e in events] == [("heartbeat", 51.3), ("gap", 1.0)]
    assert events[0]["at_s"] == 1 and "at" in events[0]
    for _ in range(100):
        st.stall_event("gap", 1.0)
    assert len(st.stalls["events"]) == 64


def test_burst_gaps_keep_the_eight_longest_over_20_ms():
    st = IngestStats()
    t = 0
    st.burst_gap(t)
    for gap_ms in (5, 25, 30, 10, 40, 21, 22, 23, 24, 26, 27, 50):
        t += gap_ms * 1_000_000
        st.burst_gap(t)
    gaps = st.stalls_view()["burst_gaps_ms"]
    assert gaps == sorted(gaps) and len(gaps) == 8
    assert gaps == [23.0, 24.0, 25.0, 26.0, 27.0, 30.0, 40.0, 50.0]


def test_timed_keeps_the_worst_pass_with_step_split():
    st = IngestStats()
    ms = 1_000_000
    st.timed("burst", [0, 2 * ms, 5 * ms], ("frames", "sync"), n=3)
    assert st.stalls["worst_burst"] == dict(ms=5.0, frames=2.0, sync=3.0, n=3)
    st.timed("burst", [0, 1 * ms, 2 * ms], ("frames", "sync"), n=4)
    assert st.stalls["worst_burst"]["ms"] == 5.0
    st.timed("compute", [0, 60 * ms, 70 * ms], ("decode", "release"))
    assert st.stalls["worst_compute"]["decode"] == 60.0
    assert [e["kind"] for e in st.stalls["events"]] == ["compute"]  # only passes over 50 ms
    view = st.stalls_view()
    view["events"].append("x")
    assert len(st.stalls["events"]) == 1  # the view holds copies
    st.reset_stalls()
    assert st.stalls == dict(burst_gaps_ms=[], worst_burst=None, worst_compute=None, events=[])


def counters(generation=1, events=10):
    return BarCounters(generation, events, 6, 3, 1, 0, 4, 99, 1, 2)


def test_bars_stats_observations_and_reset():
    bars = BarsStats()
    assert bars.observations() is None
    bars.reset(3, ["A", "B"], device=True)
    assert (bars.generation, bars.symbols, bars.device, bars.expected_sequence) == (
        3,
        ("A", "B"),
        True,
        1,
    )
    bars.counters = counters()
    assert bars.observations() == dict(
        events=10, quotes=6, trades=3, invalid_quotes=1, late_events=0, completed_bars=4
    )
    bars.batches_in, bars.stale_packets, bars.windows_emitted = 5, 2, 1
    bars.last_window_key = (3, 4)
    bars.reset(4, ("C",), device=False)
    assert bars.counters is None and bars.symbol_snapshot is None
    # Run totals and the last window key survive a generation change.
    assert (bars.batches_in, bars.stale_packets, bars.windows_emitted) == (5, 2, 1)
    assert bars.last_window_key == (3, 4)


def test_bar_counters_are_frozen_and_name_their_timing_fields():
    c = counters()
    with pytest.raises(AttributeError):
        c.events = 1
    assert BarCounters.TIMING_FIELDS == ("first_batch_ns", "last_batch_ns")
    assert c == counters() and c != counters(events=11)


def test_risk_stats_defaults():
    risk = RiskStats()
    assert (risk.analyzed, risk.last_window_key, risk.error, risk.lane_cpus) == (0, None, None, [])
    assert RiskStats().lane_cpus is not risk.lane_cpus
