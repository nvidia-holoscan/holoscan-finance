# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``BarBuilder`` with the cpu backend: generation resets, stale generations, gaps, window
emission and the immutable counters and symbol snapshots the bars lane publishes."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from holoscan_finance.equity.manifest import MANIFEST_FORMAT, validate_manifest
from holoscan_finance.equity.mapping import PROTOCOL, TIMEZONE, day_start_ns, decode_packet
from holoscan_finance.equity.synthetic import synthetic_stream
from holoscan_finance.marketdata.bar_builder import BarBuilder
from holoscan_finance.marketdata.bars import QuoteBars
from holoscan_finance.marketdata.sequence import SequenceTracker
from holoscan_finance.payloads import BarCounters, BarWindow
from holoscan_finance.settings import BarSettings

MIDNIGHT = day_start_ns("2026-06-01")
UNIVERSE = 50
SETTINGS = BarSettings(bar_seconds=5, window_seconds=60, stale_seconds=30, min_bars=4)


def directory(generation=1, port=40000):
    return validate_manifest(
        dict(
            format=MANIFEST_FORMAT,
            wire_protocol=PROTOCOL,
            market_timezone=TIMEZONE,
            symbols=[f"S{i}" for i in range(UNIVERSE)],
            generation=generation,
            sender_port=port,
            date="2026-06-01",
            start_ns=MIDNIGHT,
            end_ns=MIDNIGHT + 86400 * 10**9,
            events=100,
            profile=[],
        )
    )


def decoded(payload, recv_ns, generation=1):
    (batch,), _ = decode_packet(
        payload, recv_ns, generation=generation, midnight_ns=MIDNIGHT, universe=UNIVERSE
    )
    return batch


def host_groups(generation=1, packets=900, group=97):
    """The synthetic stream of the GPU tests, decoded on the host in 97-packet groups, with a
    three-packet sequence gap."""
    payloads = synthetic_stream(np.random.default_rng(99), packets, UNIVERSE, MIDNIGHT)
    del payloads[400:403]
    batches = [decoded(p, i + 1, generation) for i, p in enumerate(payloads)]
    return [batches[lo : lo + group] for lo in range(0, len(batches), group)]


def test_nothing_is_built_before_a_directory():
    builder = BarBuilder(SETTINGS, backend="cpu")
    assert builder.engine is None and builder.generation is None
    assert builder.observe(host_groups()[0]) is None
    assert builder.counters() is None and builder.symbol_snapshot() is None
    assert not builder.needs_reset(None) and builder.needs_reset(directory())
    with pytest.raises(ValueError, match="backend"):
        BarBuilder(SETTINGS, backend="tpu")


def test_matches_quote_bars_behind_the_sequence_tracker_with_stale_generations():
    builder = BarBuilder(SETTINGS, backend="cpu")
    d = directory()
    builder.reset(d)
    assert not builder.needs_reset(d) and builder.needs_reset(directory(2, 40001))
    reference = QuoteBars(d.symbols, **SETTINGS.as_params())
    tracker = SequenceTracker(d.generation)
    stale = [dataclasses.replace(b, generation=7) for b in host_groups()[0][:3]]
    emitted = 0
    for group in host_groups():
        window = builder.observe(stale + group)
        before = reference.completed_bars
        tracker.observe(reference, stale + group)
        parts = reference.window_parts()
        if window is None:
            assert reference.completed_bars == before or parts is None
            continue
        emitted += 1
        assert isinstance(window, BarWindow) and not window.device
        assert (window.generation, window.completed_bars) == (1, reference.completed_bars)
        assert window.recv_ns == reference.bar_recv_ns
        np.testing.assert_array_equal(window.times, parts[1])
        np.testing.assert_array_equal(window.rows, np.stack(parts[0]))
        assert not window.times.flags.writeable
        assert all(not p.flags.writeable for p in window.parts)
    assert emitted >= 5
    assert builder.sequence_gaps == tracker.sequence_gaps > 0
    assert builder.stale_packets == tracker.stale_packets == 3 * len(host_groups())
    assert builder.expected == tracker.expected


def test_a_new_generation_starts_a_new_engine_and_keeps_run_totals():
    builder = BarBuilder(SETTINGS, backend="cpu")
    builder.reset(directory())
    for group in host_groups():
        builder.observe(group)
    first = builder.engine
    gaps, completed = builder.sequence_gaps, builder.counters().completed_bars
    assert gaps > 0 and completed > 10
    builder.reset(directory(2, 40001))
    assert builder.engine is not first and builder.engine.completed_bars == 0
    assert (builder.generation, builder.expected) == (2, 1)
    assert builder.counters().first_batch_ns is None
    # The old generation's packets are stale now; the totals carry over.
    assert builder.observe(host_groups()[0]) is None
    assert builder.stale_packets == len(host_groups()[0]) and builder.sequence_gaps == gaps
    assert builder.counters().events == 0
    builder.observe(host_groups(generation=2)[0])
    assert builder.counters().events > 0 and builder.expected > 1


def test_counters_and_symbol_snapshot_are_immutable_host_mirrors():
    builder = BarBuilder(SETTINGS, backend="cpu")
    builder.reset(directory())
    groups = host_groups()
    builder.observe(groups[0])
    first = builder.counters().first_batch_ns
    for group in groups[1:]:
        builder.observe(group)
    c, engine = builder.counters(), builder.engine
    assert isinstance(c, BarCounters) and c.generation == 1
    assert [c.events, c.quotes, c.trades, c.invalid_quotes, c.late_events] == [
        engine.events,
        engine.quotes,
        engine.trades,
        engine.invalid_quotes,
        engine.late_events,
    ]
    assert (c.completed_bars, c.last_event_ns) == (engine.completed_bars, engine.last_event_ns)
    assert c.first_batch_ns == first <= c.last_batch_ns
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.events = 0
    snap = builder.symbol_snapshot()
    assert snap.generation == 1 and snap.symbols == directory().symbols
    assert snap.last_event_ns == engine.last_event_ns and snap.taken_ns > 0
    for got, state in zip(
        (snap.mid, snap.trade, snap.volume, snap.quote_time),
        (engine.mid, engine.trade, engine.volume, engine.quote_time),
    ):
        np.testing.assert_array_equal(got, state)
        assert not got.flags.writeable and not np.shares_memory(got, state)


def test_wrong_batch_types_are_rejected():
    builder = BarBuilder(SETTINGS, backend="cpu")
    builder.reset(directory())
    with pytest.raises(TypeError, match="cpu bars take"):
        builder.observe([{"events": []}])
