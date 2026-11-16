# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Device-resident quote bars match the host ``QuoteBars`` through a sequence gap, on the
fused and the general path, and the batched host mirrors ``counts()`` and
``symbol_arrays()`` are exact."""

from __future__ import annotations

import numpy as np
import pytest
from test_gpu_decode import MIDNIGHT, UNIVERSE, to_slots

from holoscan_finance.equity import mapping as replay
from holoscan_finance.equity.synthetic import synthetic_stream
from holoscan_finance.marketdata.bars import QuoteBars
from holoscan_finance.marketdata.gpu_bars import GpuQuoteBars, GpuReplayBatch, apply_sequence
from holoscan_finance.marketdata.gpu_sbe import decode_packets
from holoscan_finance.marketdata.sequence import SequenceTracker

pytestmark = pytest.mark.gpu
cp = pytest.importorskip("cupy")

SETTINGS = dict(
    bar_seconds=5, window_seconds=60, stale_seconds=30, max_spread_bps=100.0, min_bars=4
)
SYMBOLS = [f"S{i}" for i in range(UNIVERSE)]


def ulps(a, b):
    """Distance in units in the last place between float64 arrays."""
    ai, bi = a.view(np.int64), b.view(np.int64)
    ai = np.where(ai < 0, np.int64(-(2**63)) - ai, ai)
    bi = np.where(bi < 0, np.int64(-(2**63)) - bi, bi)
    return np.abs(ai - bi)


def stream_groups(packets=900, group=97):
    """The synthetic stream with a three-packet wire loss, in 97-packet groups."""
    payloads = synthetic_stream(np.random.default_rng(99), packets, UNIVERSE, MIDNIGHT)
    del payloads[400:403]  # wire loss: the gap path must reset marks identically
    for lo in range(0, len(payloads), group):
        part = payloads[lo : lo + group]
        yield part, np.arange(lo, lo + len(part), dtype=np.int64) + 1


def host_batches(part, recv):
    """The host decoder's batch for each packet."""
    out = []
    for data, recv_ns in zip(part, recv):
        (batch,), _ = replay.decode_packet(
            data, int(recv_ns), generation=1, midnight_ns=MIDNIGHT, universe=UNIVERSE
        )
        out.append(batch)
    return out


def device_batch(part, recv, generation=1):
    slots, lens = to_slots(part)
    dec = decode_packets(slots, lens, midnight_ns=MIDNIGHT, universe=UNIVERSE)
    return GpuReplayBatch(
        dec.columns, generation, dec.sequence, dec.wire_count, dec.offsets, recv, int(recv[-1])
    )


@pytest.mark.parametrize("path", ["fused", "general"])
def test_gpu_bars_match_host_bars_through_a_sequence_gap(path, monkeypatch):
    if path == "general":
        monkeypatch.setattr(GpuQuoteBars, "_observe_fused", lambda self, batch: False)
    hb, gb = QuoteBars(SYMBOLS, **SETTINGS), GpuQuoteBars(SYMBOLS, **SETTINGS)
    tracker = SequenceTracker(generation=1)
    g_expected, g_gaps = 1, 0
    for part, recv in stream_groups():
        tracker.observe(hb, host_batches(part, recv))
        g_expected, _, gaps = apply_sequence(gb, device_batch(part, recv), g_expected)
        g_gaps += gaps
    assert tracker.sequence_gaps > 0 and tracker.sequence_gaps == g_gaps
    assert tracker.expected == g_expected
    assert hb.completed_bars == gb.completed_bars > 10
    for name in ("events", "quotes", "trades", "invalid_quotes", "late_events", "bar_recv_ns"):
        assert getattr(hb, name) == getattr(gb, name), name
    for name in ("mid", "trade", "volume", "previous"):
        np.testing.assert_array_equal(
            getattr(hb, name).view(np.int64), getattr(gb, name).view(np.int64)
        )
    np.testing.assert_array_equal(hb.quote_time, gb.quote_time)
    assert list(hb.times) == list(gb.times)
    for a, b in zip(hb.marks, gb.marks):
        assert isinstance(b, cp.ndarray)  # bars stay in GPU memory
        np.testing.assert_array_equal(a.view(np.int64), b.get().view(np.int64))
    for a, b in zip(hb.rows, gb.rows):
        b = b.get()
        np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
        f = np.isfinite(a)
        assert (ulps(a[f], b[f]) <= 1).all()  # CUDA vs NumPy log: <= 1 ULP
    wa, wb = hb.window(), gb.window()
    assert isinstance(wb[0], cp.ndarray)
    np.testing.assert_array_equal(wa[1], wb[1])
    np.testing.assert_array_equal(np.isnan(wa[0]), np.isnan(wb[0].get()))
    parts, times = gb.window_parts()
    assert all(isinstance(p, cp.ndarray) for p in parts)
    np.testing.assert_array_equal(cp.stack(parts).get(), wb[0].get())
    np.testing.assert_array_equal(times, wb[1])


def test_counts_and_symbol_arrays_mirror_the_host_bars_exactly():
    hb, gb = QuoteBars(SYMBOLS, **SETTINGS), GpuQuoteBars(SYMBOLS, **SETTINGS)
    tracker, expected = SequenceTracker(generation=1), 1
    for part, recv in stream_groups(packets=300):
        tracker.observe(hb, host_batches(part, recv))
        expected = apply_sequence(gb, device_batch(part, recv), expected)[0]
    counts = gb.counts()
    assert isinstance(counts, np.ndarray) and counts.dtype == np.uint64
    np.testing.assert_array_equal(counts, hb.counts())
    host, device = hb.symbol_arrays(), gb.symbol_arrays()
    for name, a, b in zip(("mid", "trade", "volume", "quote_time"), host, device):
        assert isinstance(b, np.ndarray) and b.dtype == a.dtype, name
        np.testing.assert_array_equal(a.view(np.int64), b.view(np.int64), err_msg=name)
    # quote_time travels as int64 bits: epoch nanoseconds (~2^61) survive exactly.
    assert (device[3] % 256 != 0).any()


def test_reset_marks_never_alters_a_stored_bar():
    gb = GpuQuoteBars(SYMBOLS, **SETTINGS)
    expected = 1
    for part, recv in stream_groups(packets=200):
        expected = apply_sequence(gb, device_batch(part, recv), expected)[0]
    stored = [m.get() for m in gb.marks]
    gb.reset_marks()
    assert np.isnan(gb.mid).all() and np.isnan(gb.previous).all()
    for a, b in zip(stored, gb.marks):
        np.testing.assert_array_equal(a.view(np.int64), b.get().view(np.int64))
