# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run verdicts: window pairing, ULP and NaN rules, final bar state, the host reference and
every bench gate (``validation.checkers``, ``validation.compare.HostReference``)."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np
import pytest

from holoscan_finance.marketdata.bar_builder import BarBuilder
from holoscan_finance.marketdata.framing import PcapFrames
from holoscan_finance.nyq.manifest import validate_manifest
from holoscan_finance.nyq.mapping import PROTOCOL, TIMEZONE, decode_packet
from holoscan_finance.nyq.synthetic import synthetic_capture, synthetic_tape
from holoscan_finance.payloads import BarCounters, BarWindow, SymbolSnapshot
from holoscan_finance.settings import BarSettings
from holoscan_finance.telemetry.stats import BarsStats
from holoscan_finance.validation import compare
from holoscan_finance.validation.checkers import (
    LOSS_KEYS,
    BarsChecker,
    Gates,
    bench_verdict,
    final_state_equal,
)
from holoscan_finance.validation.compare import HostReference, host_decode, ulps

BAR_NS = 5 * 10**9


def window(completed, *, generation=1, n_bars=3, universe=4, shift=0.0, parts=None):
    """A window whose newest bar is bar ``completed``; bar k's returns are k + shift."""
    times = np.arange(completed - n_bars + 1, completed + 1, dtype=np.int64) * BAR_NS
    if parts is None:
        parts = tuple(
            np.full(universe, k + shift) for k in range(completed - n_bars + 1, completed + 1)
        )
    return BarWindow(tuple(parts), times, generation, completed, 0)


def test_ulps_counts_representable_steps_across_zero():
    a = np.array([1.0, -0.0, 1.0, -1.0])
    b = np.array([np.nextafter(1.0, 2.0), 0.0, 1.0, np.nextafter(-1.0, 0.0)])
    assert ulps(a, b).tolist() == [1, 0, 0, 1]


def test_pairs_windows_by_key_whatever_the_arrival_order():
    c = BarsChecker()
    for k in (3, 4, 6):
        c.left(window(k))
    for k in (6, 3, 4):  # the other lane's order does not matter
        c.right(window(k))
    s = c.summary()
    assert s == dict(
        windows_left=3,
        windows_right=3,
        windows_compared=3,
        times_equal=True,
        nan_pattern_equal=True,
        max_ulps=0,
    )
    assert c.count_left() == c.count_right() == 3


def test_one_ulp_is_reported_and_nan_patterns_must_match():
    c = BarsChecker()
    c.left(window(3))
    nudged = tuple(np.nextafter(p, np.inf) for p in window(3).parts)
    c.right(window(3, parts=nudged))
    assert c.summary()["max_ulps"] == 1 and c.summary()["nan_pattern_equal"]

    c = BarsChecker()
    c.left(window(3))
    holed = [p.copy() for p in window(3).parts]
    holed[-1][0] = np.nan
    c.right(window(3, parts=holed))
    s = c.summary()
    assert not s["nan_pattern_equal"] and s["times_equal"]

    c = BarsChecker()
    inf = [p.copy() for p in window(3).parts]
    inf[0][1] = np.inf
    c.left(window(3, parts=inf))
    c.right(window(3))
    assert not c.summary()["nan_pattern_equal"]


def test_a_bar_newer_than_the_last_pair_is_compared_even_if_not_newest():
    # Two bars close between windows 3 and 5: bar 4 must be compared too.
    c = BarsChecker()
    c.left(window(3))
    c.left(window(5))
    c.right(window(3))
    bad = [p.copy() for p in window(5).parts]
    bad[1][2] += 1.0  # bar 4, not the newest
    c.right(window(5, parts=bad))
    assert c.summary()["max_ulps"] > 1


@pytest.mark.parametrize("fault", ["missing", "extra", "duplicate", "times", "generation"])
def test_unpaired_or_misaligned_windows_fail_times_equal(fault):
    c = BarsChecker()
    c.left(window(3))
    c.left(window(4))
    if fault == "missing":
        c.right(window(3))
    elif fault == "extra":
        c.right(window(3))
        c.right(window(4))
        c.right(window(5))
    elif fault == "duplicate":
        c.right(window(3))
        c.right(window(4))
        c.right(window(4))
    elif fault == "times":
        c.right(window(3))
        shifted = window(4)
        c.right(BarWindow(shifted.parts, shifted.times + 1, 1, 4, 0))
    else:
        c.right(window(3))
        c.right(window(4, generation=2))
    assert not c.summary()["times_equal"]


def test_callbacks_from_two_lanes_are_all_counted():
    c = BarsChecker()
    w = window(3)

    def feed(side):
        for _ in range(2000):
            side(w)

    threads = [threading.Thread(target=feed, args=(s,)) for s in (c.left, c.right, c.left)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert (c.count_left(), c.count_right()) == (4000, 2000)


def fake_reference(n_bars=10, universe=4, generation=1):
    times = np.arange(1, n_bars + 1, dtype=np.int64) * BAR_NS
    returns = np.stack([np.full(universe, float(k)) for k in range(1, n_bars + 1)])
    return SimpleNamespace(
        closed_bars=lambda: (times, returns), directory=SimpleNamespace(generation=generation)
    )


def test_compare_reference_needs_consecutive_reference_bars():
    ref = fake_reference()
    c = BarsChecker()
    out = c.compare_reference(ref, [window(3), window(4), window(7)])
    assert out == dict(closed_bars_compared=7, times_equal=True, nan_pattern_equal=True, max_ulps=0)
    skipped = window(7)
    gap = BarWindow(skipped.parts, np.array([4, 6, 7]) * BAR_NS, 1, 7, 0)  # bar 5 missing
    assert not c.compare_reference(ref, [gap])["times_equal"]
    assert not c.compare_reference(ref, [window(11)])["times_equal"]  # beyond the reference
    assert not c.compare_reference(ref, [window(4, generation=2)])["times_equal"]
    assert c.compare_reference(ref, [window(4, shift=1e-15)])["max_ulps"] > 1


def bars_stats(*, events=10, first=1, mid=(1.0, np.nan), quote_time=(5, 6)):
    s = BarsStats()
    s.counters = BarCounters(1, events, 6, 3, 1, 0, 2, 99, first, first + 10)
    s.symbol_snapshot = SymbolSnapshot(
        1,
        ("A", "B"),
        np.array(mid),
        np.array([2.0, np.nan]),
        np.array([3.0, 0.0]),
        np.array(quote_time, np.int64),
        99,
        first,
    )
    return s


def test_final_state_ignores_arrival_times_only():
    assert final_state_equal(bars_stats(first=1), bars_stats(first=12345))
    assert not final_state_equal(bars_stats(), bars_stats(events=11))
    assert not final_state_equal(bars_stats(), bars_stats(mid=(1.0, 0.0)))  # NaN pattern
    assert not final_state_equal(bars_stats(), bars_stats(mid=(np.nextafter(1.0, 2), np.nan)))
    assert not final_state_equal(bars_stats(), bars_stats(quote_time=(5, 7)))
    other_nan = np.array([1.0, np.nan])
    other_nan.view(np.int64)[1] |= 1  # another NaN payload is still NaN
    assert final_state_equal(bars_stats(), bars_stats(mid=other_nan))
    empty = BarsStats()
    assert not final_state_equal(bars_stats(), empty)
    no_snapshot = bars_stats()
    no_snapshot.symbol_snapshot = None
    assert not final_state_equal(bars_stats(), no_snapshot)


def good_report():
    return {
        "termination": "STOPPED",
        "stop_reason": "drained",
        "retries": 0,
        "ring_waits": 0,
        "msgs_per_s": 2e7,
        "lag_p99_ms": 3.0,
        **dict.fromkeys(LOSS_KEYS, 0),
        "check": {
            "windows_compared": 5,
            "times_equal": True,
            "nan_pattern_equal": True,
            "max_ulps": 1,
            "final_state_equal": True,
            "reference": {
                "closed_bars_compared": 50,
                "times_equal": True,
                "nan_pattern_equal": True,
                "max_ulps": 0,
                "counters_equal": True,
                "decode_counts_equal": True,
            },
        },
    }


ALL_GATES = Gates(True, True, True, min_msgs_per_s=1.32e7, max_lag_p99_ms=5)


def test_verdict_passes_a_good_report_on_every_gate():
    assert bench_verdict(good_report(), ALL_GATES) == (True, [])
    assert bench_verdict({"termination": "STOPPED", "stop_reason": "drained"}, Gates()) == (
        True,
        [],
    )


@pytest.mark.parametrize(
    "path, value, gates",
    [
        ("termination", "FAILED", Gates()),
        ("stop_reason", "timeout", Gates()),
        ("stop_reason", None, Gates()),
        *[(key, 1, Gates(require_lossless=True)) for key in LOSS_KEYS],
        ("nic_dropped", None, Gates(require_lossless=True)),
        ("retries", 2, Gates(require_realtime=True)),
        ("ring_waits", 1, Gates(require_realtime=True)),
        ("check.windows_compared", 0, Gates(check=True)),
        ("check.times_equal", False, Gates(check=True)),
        ("check.nan_pattern_equal", False, Gates(check=True)),
        ("check.max_ulps", 2, Gates(check=True)),
        ("check.final_state_equal", False, Gates(check=True)),
        ("check.reference.closed_bars_compared", 0, Gates(check=True)),
        ("check.reference.times_equal", False, Gates(check=True)),
        ("check.reference.nan_pattern_equal", False, Gates(check=True)),
        ("check.reference.max_ulps", 3, Gates(check=True)),
        ("check.reference.counters_equal", False, Gates(check=True)),
        ("check.reference.decode_counts_equal", False, Gates(check=True)),
        ("msgs_per_s", 1.3e7, Gates(min_msgs_per_s=1.32e7)),
        ("lag_p99_ms", 5.5, Gates(max_lag_p99_ms=5)),
        ("lag_p99_ms", None, Gates(max_lag_p99_ms=5)),
    ],
)
def test_each_gate_fails_on_a_bad_value(path, value, gates):
    report = good_report()
    *parents, leaf = path.split(".")
    node = report
    for part in parents:
        node = node[part]
    node[leaf] = value
    ok, failures = bench_verdict(report, gates)
    assert not ok and len(failures) == 1 and failures[0].startswith(path)


def test_numpy_scalars_are_numbers_and_booleans_are_not():
    report = good_report()
    report.update(sequence_gaps=np.int64(0), msgs_per_s=np.float32(2e7), retries=np.uint8(0))
    assert bench_verdict(report, ALL_GATES) == (True, [])
    report.update(sequence_gaps=False)  # a flag where a count belongs
    ok, failures = bench_verdict(report, ALL_GATES)
    assert not ok and failures == ["sequence_gaps = False: --require-lossless"]


def test_gates_that_are_off_ignore_their_values():
    report = good_report()
    report.update(sequence_gaps=7, retries=3, msgs_per_s=1.0, lag_p99_ms=99.0, check=None)
    assert bench_verdict(report, Gates()) == (True, [])


@pytest.mark.parametrize("missing", ["termination", "sequence_gaps", "check", "lag_p99_ms"])
def test_a_missing_report_key_is_a_failure_not_an_error(missing):
    report = good_report()
    del report[missing]
    ok, failures = bench_verdict(report, ALL_GATES)
    assert not ok and failures and all("missing" in f for f in failures)
    assert any(f.startswith(missing) for f in failures)


# ---- HostReference against the CPU bar builder on a synthetic capture --------------------


def capture(tmp_path, **kwargs):
    tape = tmp_path / "tape"
    manifest = synthetic_tape(tape, symbols=40, packets=1500, seconds=900, seed=11)
    synthetic_capture(tape, tmp_path / "day.pcap", sender_port=40000, **kwargs)
    directory = validate_manifest(
        {
            **manifest,
            "generation": 5,
            "sender_port": 40000,
            "wire_protocol": PROTOCOL,
            "market_timezone": TIMEZONE,
        }
    )
    return directory, PcapFrames.read(tmp_path / "day.pcap")


def builder_windows(directory, frames, batch_packets):
    """CPU ``BarBuilder`` windows over the capture in batches of ``batch_packets``: what the
    graph's bars would emit, batched differently from the reference."""
    b = BarBuilder(BarSettings(), backend="cpu")
    b.reset(directory)
    day = dict(
        generation=directory.generation,
        midnight_ns=directory.midnight_ns,
        universe=directory.universe,
    )
    batches = []
    for k, data in enumerate(frames.payloads()):
        batches.extend(decode_packet(data, k, **day)[0])
    windows = []
    for lo in range(0, len(batches), batch_packets):
        if (w := b.observe(batches[lo : lo + batch_packets])) is not None:
            windows.append(w)
    return b, windows


@pytest.mark.parametrize("drop_every", [0, 97])
def test_host_reference_matches_the_bar_builder_bar_for_bar(tmp_path, drop_every):
    directory, frames = capture(tmp_path, drop_every=drop_every)
    ref = HostReference(directory, BarSettings())
    ref.feed(frames, workers=2)  # the fork pool path
    builder, windows = builder_windows(directory, frames, batch_packets=200)
    times, returns = ref.closed_bars()
    assert len(times) == ref.bars.completed_bars > BarSettings().min_bars
    assert returns.shape == (len(times), directory.universe)
    assert np.all(np.diff(times) == BAR_NS)
    out = BarsChecker().compare_reference(ref, windows)
    assert out["times_equal"] and out["nan_pattern_equal"] and out["max_ulps"] == 0
    assert out["closed_bars_compared"] == times[-1] // BAR_NS - windows[0].times[0] // BAR_NS + 1
    counters = builder.counters()
    assert ref.counters() == dict(
        events=counters.events,
        quotes=counters.quotes,
        trades=counters.trades,
        invalid_quotes=counters.invalid_quotes,
        late_events=counters.late_events,
        completed_bars=counters.completed_bars,
        sequence_gaps=builder.sequence_gaps,
        stale_packets=builder.stale_packets,
    )
    assert (ref.tracker.sequence_gaps > 0) == bool(drop_every)
    decoded = ref.decode_counts()
    assert decoded["packets"] == len(frames) and decoded["rejected_packets"] == 0
    assert decoded["messages"] == decoded["received_messages"]


def test_host_reference_is_the_same_decoded_in_chunks(tmp_path, monkeypatch):
    directory, frames = capture(tmp_path, drop_every=97)
    whole = HostReference(directory, BarSettings())
    whole.feed(frames, workers=1)
    monkeypatch.setattr(compare, "_CHUNK_PACKETS", 128)  # bars and gaps span chunk edges
    chunked = HostReference(directory, BarSettings())
    chunked.feed(frames, workers=1)
    for a, b in zip(whole.closed_bars(), chunked.closed_bars(), strict=True):
        assert np.array_equal(a, b, equal_nan=True)
    assert whole.counters() == chunked.counters()
    assert whole.decode_counts() == chunked.decode_counts()


def test_host_decode_is_the_same_in_process_and_in_a_fork_pool(tmp_path):
    directory, frames = capture(tmp_path)
    payloads = frames.payloads()[:300]
    one = host_decode(payloads, directory.midnight_ns, directory.universe, workers=1)
    pool = host_decode(payloads, directory.midnight_ns, directory.universe, workers=3)
    for a, b in zip(one, pool, strict=True):
        assert a.dtype == b.dtype and np.array_equal(a.view(np.uint8), b.view(np.uint8))
