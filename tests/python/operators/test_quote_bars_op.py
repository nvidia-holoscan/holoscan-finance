# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``QuoteBarsOp`` under the operator harness.

cpu: windows equal ``BarBuilder`` on the same input; a knob generation change resets the bars
and older batches turn stale, but messages of the bars' generation still in flight when the
knob moved are built first; ``emit_windows=False`` declares no ``window`` port; counters are
mirrored per batch and the symbol snapshot at its cadence; ``stop()`` publishes the final
state. gpu: windows equal ``GpuQuoteBars`` + ``apply_sequence`` through a stale generation, a
gap and a reset, with device-resident parts; the bar kernels warm in ``start()``
unless ``warmup=False``; the cpu backend fed the same ``GpuReplayBatch``es builds the same
bars."""

from __future__ import annotations

import dataclasses

import holoscan as hs
import numpy as np
import pytest
from holoscan.testing import OperatorTestHarness

from holoscan_finance.equity.manifest import validate_manifest
from holoscan_finance.equity.mapping import PROTOCOL, TIMEZONE, day_start_ns, decode_packet
from holoscan_finance.equity.synthetic import synthetic_stream
from holoscan_finance.marketdata.bar_builder import BarBuilder
from holoscan_finance.operators import QuoteBarsOp
from holoscan_finance.operators import quote_bars as quote_bars_module
from holoscan_finance.payloads import BarCounters
from holoscan_finance.ref import Ref
from holoscan_finance.settings import BarSettings
from holoscan_finance.telemetry.stats import BarsStats

pytestmark = pytest.mark.sdk

SETTINGS = BarSettings(bar_seconds=5, window_seconds=60, stale_seconds=30, min_bars=4)
DATE, UNIVERSE = "2026-06-01", 50
MIDNIGHT = day_start_ns(DATE)
PAYLOADS = synthetic_stream(np.random.default_rng(99), 900, UNIVERSE, MIDNIGHT, 34200)
del PAYLOADS[400:403]  # a sequence gap


def feed(generation: int):
    return validate_manifest(
        dict(
            format="equity-replay-v1",
            wire_protocol=PROTOCOL,
            market_timezone=TIMEZONE,
            sender_port=40000 + generation,
            symbols=[f"S{i}" for i in range(UNIVERSE)],
            generation=generation,
            date=DATE,
            start_ns=MIDNIGHT,
            end_ns=MIDNIGHT + 86_400 * 10**9,
            events=1,
            profile=[],
        )
    )


class ScriptedKnob:
    """``Knob`` protocol: the operator reads it once per batch, so the n-th batch sees the n-th
    value (the last one repeats)."""

    def __init__(self, *values) -> None:
        self.values = list(values)

    def get(self):
        return self.values.pop(0) if len(self.values) > 1 else self.values[0]


class RecordingBarsStats(BarsStats):
    """Records every published ``counters`` and ``symbol_snapshot`` object."""

    def __init__(self) -> None:
        super().__init__()
        self.published: dict[str, list] = {"counters": [], "symbol_snapshot": []}

    def __setattr__(self, name: str, value: object) -> None:
        if (
            name in ("counters", "symbol_snapshot")
            and value is not None
            and hasattr(self, "published")
        ):
            self.published[name].append(value)
        super().__setattr__(name, value)


def host_messages(generation: int, per_message: int = 10) -> list[list]:
    """``SbeDecodeOp``-shaped input: one ``ReplayBatch`` per packet, ``per_message`` packets
    per message."""
    batches = []
    for k, p in enumerate(PAYLOADS):
        decoded, _ = decode_packet(
            p, k + 1, generation=generation, midnight_ns=MIDNIGHT, universe=UNIVERSE
        )
        batches += decoded
    return [batches[i : i + per_message] for i in range(0, len(batches), per_message)]


def run_bars(messages: list, *, knob, stats: BarsStats, **params) -> list:
    """The windows ``QuoteBarsOp`` emits for ``messages`` (``stats`` holds the rest)."""
    params = {**SETTINGS.as_params(), "directory": Ref(knob), "stats": Ref(stats), **params}
    with OperatorTestHarness(QuoteBarsOp, params=params) as h:
        h.push_many("events", messages)
        result = h.run()
    return result.values("window") if params.get("emit_windows", True) else []


def reference(messages: list, directory) -> tuple[list, BarBuilder]:
    builder = BarBuilder(SETTINGS, backend="cpu")
    builder.reset(directory)
    windows = [w for m in messages if (w := builder.observe(m)) is not None]
    return windows, builder


def meta(w) -> tuple:
    return w.generation, w.completed_bars, w.recv_ns, w.device


def assert_same_windows(got: list, want: list) -> None:
    assert len(got) == len(want) > 5
    for a, b in zip(got, want):
        assert meta(a) == meta(b)
        np.testing.assert_array_equal(a.times, b.times)
        assert not a.times.flags.writeable
        np.testing.assert_array_equal(a.rows.view(np.int64), b.rows.view(np.int64))  # bitwise


def timeless(c: BarCounters) -> dict:
    d = dataclasses.asdict(c)
    for name in BarCounters.TIMING_FIELDS:
        d.pop(name)
    return d


def assert_same_snapshot(a, b) -> None:
    assert (a.generation, a.symbols, a.last_event_ns) == (b.generation, b.symbols, b.last_event_ns)
    for name in ("mid", "trade", "volume", "quote_time"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
        assert not getattr(a, name).flags.writeable


def test_cpu_windows_and_final_state_equal_the_bar_builder():
    messages, directory = host_messages(1), feed(1)
    stats = RecordingBarsStats()
    windows = run_bars(messages, knob=ScriptedKnob(directory), stats=stats, backend="cpu")
    want, builder = reference(messages, directory)
    assert_same_windows(windows, want)
    assert stats.windows_emitted == len(windows) and stats.batches_in == len(messages)
    assert stats.last_window_key == (1, windows[-1].completed_bars) == (1, stats.last_window_bars)
    assert (stats.generation, stats.symbols, stats.device) == (1, directory.symbols, False)
    assert (stats.sequence_gaps, stats.stale_packets) == (
        builder.sequence_gaps,
        builder.stale_packets,
    )
    assert stats.sequence_gaps > 0 and stats.expected_sequence == builder.expected
    # Counters every batch, plus the final state published by stop().
    assert len(stats.published["counters"]) == len(messages) + 1
    assert timeless(stats.counters) == timeless(builder.counters())
    assert_same_snapshot(stats.symbol_snapshot, builder.symbol_snapshot())


def test_the_bars_synchronize_before_each_window_is_emitted(monkeypatch):
    """The producer's half of the publish contract: risk reads a window's device parts on
    its own stream, which only this synchronize orders after the bar kernels. Per window:
    synchronize, then the drained key, the emit and the emitted count."""
    order = []

    class Builder(BarBuilder):
        def synchronize(self) -> None:
            order.append("synchronize")
            super().synchronize()

    class Stats(BarsStats):
        def __setattr__(self, name: str, value: object) -> None:
            if name in ("last_window_key", "windows_emitted") and value:
                order.append(name)
            super().__setattr__(name, value)

    monkeypatch.setattr(quote_bars_module, "BarBuilder", Builder)
    stats = Stats()
    windows = run_bars(host_messages(1), knob=ScriptedKnob(feed(1)), stats=stats, backend="cpu")
    assert len(windows) > 5
    assert order == ["synchronize", "last_window_key", "windows_emitted"] * len(windows)


@pytest.mark.parametrize(("period_ms", "snapshots"), [(60_000, 2), (0, 1)])
def test_symbol_snapshot_cadence(period_ms, snapshots):
    stats = RecordingBarsStats()
    run_bars(
        host_messages(1)[:20],
        knob=ScriptedKnob(feed(1)),
        stats=stats,
        backend="cpu",
        snapshot_period_ms=period_ms,
    )
    # A long period: the first batch and the final state; 0: only the final state.
    assert len(stats.published["symbol_snapshot"]) == snapshots


def test_a_new_generation_resets_the_bars_and_older_batches_turn_stale():
    old, new = host_messages(1)[:40], host_messages(2)[:40]
    late = old[10:12]  # generation-1 packets arriving after the switch
    messages = old + new[:20] + late + new[20:]
    knob = ScriptedKnob(*[feed(1)] * len(old), feed(2))
    stats = BarsStats()
    windows = run_bars(messages, knob=knob, stats=stats, backend="cpu")
    want_old, _ = reference(old, feed(1))
    want_new, builder = reference(new, feed(2))
    assert_same_windows(windows, want_old + want_new)
    assert stats.generation == 2 and stats.stale_packets == sum(len(m) for m in late)
    assert stats.counters.generation == stats.symbol_snapshot.generation == 2
    assert timeless(stats.counters) == timeless(builder.counters())
    assert stats.windows_emitted == len(windows)  # run totals survive the reset


def test_messages_in_flight_when_the_knob_moves_are_built_before_the_reset():
    """A replay loop moves the knob once the ring is released, before the bars took the pass's
    last decoded messages: they are built under their own generation, not stale, and the bars
    reset at the first message decoded under the new one."""
    old, new = host_messages(1)[:40], host_messages(2)[:40]
    knob = ScriptedKnob(*[feed(1)] * (len(old) - 2), feed(2))  # moves with two still queued
    stats = BarsStats()
    windows = run_bars(old + new, knob=knob, stats=stats, backend="cpu")
    want_old, _ = reference(old, feed(1))
    want_new, builder = reference(new, feed(2))
    assert_same_windows(windows, want_old + want_new)
    assert stats.generation == 2 and stats.stale_packets == 0
    assert timeless(stats.counters) == timeless(builder.counters())
    assert stats.windows_emitted == len(windows) and stats.batches_in == len(old) + len(new)


def test_without_windows_no_window_port_is_declared():
    g = hs.Graph("bars", default_clock=hs.RealtimeClock)
    assert "window" in g.op(QuoteBarsOp, "with", backend="cpu").ports
    assert list(g.op(QuoteBarsOp, "without", backend="cpu", emit_windows=False).ports) == ["events"]
    stats = BarsStats()
    run_bars(
        host_messages(1)[:30],
        knob=ScriptedKnob(feed(1)),
        stats=stats,
        backend="cpu",
        emit_windows=False,
    )
    assert stats.batches_in == 30 and stats.windows_emitted == 0
    assert stats.counters.completed_bars > 4


def test_without_a_directory_input_is_counted_and_ignored():
    stats = BarsStats()
    windows = run_bars(host_messages(1)[:10], knob=ScriptedKnob(None), stats=stats, backend="cpu")
    assert windows == [] and stats.batches_in == 10
    assert stats.counters is None and stats.symbol_snapshot is None


def test_invalid_bar_settings_are_rejected_at_authoring():
    g = hs.Graph("bars", default_clock=hs.RealtimeClock)
    with pytest.raises(hs.errors.HoloscanError, match="invalid bar"):
        g.op(QuoteBarsOp, "bars", backend="cpu", window_seconds=7)


# --- GPU ------------------------------------------------------------------------------------


def _to_slots(cp, payloads: list[bytes]):
    from holoscan_finance.marketdata.gpu_sbe import SLOT_BYTES

    host = np.zeros((len(payloads), SLOT_BYTES), np.uint8)
    for i, p in enumerate(payloads):
        host[i, : len(p)] = np.frombuffer(p, np.uint8)
    return cp.asarray(host), cp.asarray(np.array([len(p) for p in payloads], np.int32))


def gpu_batches(cp, generation: int = 1) -> list:
    """The synthetic stream decoded on the GPU in 97-packet batches."""
    from holoscan_finance.marketdata.gpu_bars import GpuReplayBatch
    from holoscan_finance.marketdata.gpu_sbe import decode_packets

    batches = []
    for lo in range(0, len(PAYLOADS), 97):
        part = PAYLOADS[lo : lo + 97]
        recv = np.arange(lo, lo + len(part), dtype=np.int64) + 1
        slots, lens = _to_slots(cp, part)
        dec = decode_packets(slots, lens, midnight_ns=MIDNIGHT, universe=UNIVERSE)
        batches.append(
            GpuReplayBatch(
                dec.columns,
                generation,
                dec.sequence,
                dec.wire_count,
                dec.offsets,
                recv,
                int(recv[-1]),
            )
        )
    return batches


@pytest.mark.gpu
def test_gpu_windows_match_gpu_quote_bars_through_stale_batches_a_gap_and_a_reset():
    cp = pytest.importorskip("cupy")
    from holoscan_finance.marketdata.gpu_bars import GpuQuoteBars, apply_sequence

    batches, stale = gpu_batches(cp), gpu_batches(cp, generation=7)[0]
    messages = [[stale, b] for b in batches] + [[b] for b in gpu_batches(cp, generation=2)]
    knob = ScriptedKnob(*[feed(1)] * len(batches), feed(2))
    stats = BarsStats()
    windows = run_bars(messages, knob=knob, stats=stats, backend="gpu")
    want, gaps = [], 0
    for generation in (1, 2):  # the reset: a fresh engine for generation 2
        ref, expected, emitted = (
            GpuQuoteBars(list(feed(generation).symbols), **SETTINGS.as_params()),
            1,
            0,
        )
        for b in batches:
            expected, _, g = apply_sequence(ref, b, expected)
            gaps += g
            if ref.completed_bars != emitted and (w := ref.window()) is not None:
                want.append(((generation, ref.completed_bars, ref.bar_recv_ns, True), w))
            emitted = ref.completed_bars
    assert len(windows) == len(want) > 10 and gaps > 0
    for got, (expected_meta, (rows, times)) in zip(windows, want):
        assert meta(got) == expected_meta
        assert all(isinstance(p, cp.ndarray) for p in got.parts)
        np.testing.assert_array_equal(got.times, times)
        np.testing.assert_array_equal(got.rows.get().view(np.int64), rows.get().view(np.int64))
    assert stats.sequence_gaps == gaps and stats.stale_packets == stale.n_packets * len(batches)
    assert stats.counters.generation == 2 and stats.device
    assert stats.counters.first_batch_ns <= stats.counters.last_batch_ns


@pytest.mark.gpu
@pytest.mark.parametrize("warmup", [True, False])
def test_gpu_bar_kernels_warm_in_start_unless_disabled(monkeypatch, warmup):
    pytest.importorskip("cupy")
    calls = []
    monkeypatch.setattr(
        quote_bars_module, "warm_bars_kernels", lambda settings, device: calls.append(settings)
    )
    run_bars([], knob=ScriptedKnob(feed(1)), stats=BarsStats(), backend="gpu", warmup=warmup)
    assert calls == ([SETTINGS.as_params()] if warmup else [])


@pytest.mark.gpu
def test_the_cpu_backend_fed_gpu_batches_builds_the_same_bars():
    cp = pytest.importorskip("cupy")
    messages = [[b] for b in gpu_batches(cp)]
    gpu_stats, cpu_stats = BarsStats(), BarsStats()
    gpu_windows = run_bars(messages, knob=ScriptedKnob(feed(1)), stats=gpu_stats, backend="gpu")
    cpu_windows = run_bars(messages, knob=ScriptedKnob(feed(1)), stats=cpu_stats, backend="cpu")
    assert len(gpu_windows) == len(cpu_windows) > 5
    for g, c in zip(gpu_windows, cpu_windows):
        assert meta(g)[:3] == meta(c)[:3] and g.device and not c.device
        np.testing.assert_array_equal(g.times, c.times)
        a, b = g.rows.get(), c.rows
        np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
        ulps = np.abs(a.view(np.int64) - b.view(np.int64))[~np.isnan(a)]
        assert ulps.max(initial=0) <= 1  # CUDA log vs NumPy log: within 1 ULP
    assert timeless(gpu_stats.counters) == timeless(cpu_stats.counters)
    assert_same_snapshot(gpu_stats.symbol_snapshot, cpu_stats.symbol_snapshot)
