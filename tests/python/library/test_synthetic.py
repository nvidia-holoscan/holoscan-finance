# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic data: the stream and fuzz generators reproduce their expected digests,
deterministic tapes, captures that carry exactly the tape, ``drop_every`` gaps, and the
``synth`` CLI with ``--reuse``."""

from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
import sys

import numpy as np
import pytest

from holoscan_finance.equity.manifest import validate_manifest
from holoscan_finance.equity.mapping import MAX_PACKET_EVENTS as MAX_EVENTS
from holoscan_finance.equity.mapping import PROTOCOL, TIMEZONE, day_start_ns, decode_packet
from holoscan_finance.equity.synthetic import (
    fuzz,
    synthetic_capture,
    synthetic_stream,
    synthetic_tape,
)
from holoscan_finance.equity.tape import open_tape
from holoscan_finance.marketdata import sbe_codec as sbe
from holoscan_finance.marketdata.bars import QuoteBars
from holoscan_finance.marketdata.framing import PcapFrames, strip_udp_datagram
from holoscan_finance.marketdata.gpu_sbe import PACKET_HEADER
from holoscan_finance.marketdata.records import CONDITION, HALTED, QUOTE, TRADE, ReplayBatch
from holoscan_finance.tools import synth

MIDNIGHT = day_start_ns("2026-06-01")
# Expected SHA-256 of the generators' output for these arguments: the generators must
# reproduce them byte for byte.
STREAM_DIGEST = "04fd5eb920549b7a9fe8890b899673fe513628a0c5e709979b20b24ab16d72f5"
FUZZ_DIGEST = "bcb6c46a2c6698d11ed9db370616f7e96f896579864ac49c6c1dd82443bac855"


def digest(units):
    return hashlib.sha256(b"".join(len(u).to_bytes(4, "little") + u for u in units)).hexdigest()


def decode(unit, universe, recv_ns=0):
    return decode_packet(unit, recv_ns, generation=1, midnight_ns=MIDNIGHT, universe=universe)


def test_stream_and_fuzz_reproduce_the_expected_digests():
    stream = synthetic_stream(np.random.default_rng(20260930), 400, 50, MIDNIGHT, 34200)
    assert digest(stream) == STREAM_DIGEST
    assert digest(fuzz(stream, 6000, seed=11)) == FUZZ_DIGEST


def test_stream_units_are_valid_contiguous_and_cover_every_kind():
    stream = synthetic_stream(np.random.default_rng(5), 300, 50, MIDNIGHT)
    expected, kinds, halted = 1, set(), 0
    for unit in stream:
        (batch,), count = decode(unit, 50)
        assert batch.sequence == expected
        expected += count
        kinds |= set(batch.events["kind"].tolist())
        halted += int((batch.events["flags"] == HALTED).sum())
    assert kinds == {QUOTE, TRADE, CONDITION} and halted > 0


def test_fuzz_exercises_every_rejection_stage():
    stream = synthetic_stream(np.random.default_rng(6), 300, 50, MIDNIGHT)
    mutated = fuzz(stream, 1200, seed=3)
    assert mutated == fuzz(stream, 1200, seed=3) and mutated != fuzz(stream, 1200, seed=4)
    stages = set()
    for unit in mutated:
        try:
            sbe.inspect_packet(unit)
        except (ValueError, struct.error):
            stages.add("framing")
            continue
        try:
            decode(unit, 50)
            stages.add("decoded")
        except (ValueError, struct.error):
            stages.add("content")
    assert stages == {"framing", "content", "decoded"}


@pytest.fixture(scope="module")
def tape(tmp_path_factory):
    out = tmp_path_factory.mktemp("synth") / "tape"
    manifest = synthetic_tape(out, symbols=40, packets=300, seconds=120, seed=11)
    return out, manifest


def test_tape_format_and_content(tape):
    out, manifest = tape
    m, rows = open_tape(out)
    assert m == manifest and m["events"] == len(rows) == 300 * MAX_EVENTS
    assert len(m["symbols"]) == len(set(m["symbols"])) == 40
    assert m["start_ns"] == MIDNIGHT + 34200 * 10**9 and m["end_ns"] == m["start_ns"] + 120 * 10**9
    ts = rows["timestamp_ns"]
    assert (np.diff(ts) >= 0).all() and ts[0] >= m["start_ns"] and ts[-1] < m["end_ns"]
    kinds = rows["kind"]
    assert m["kinds"] == {
        "quotes": int((kinds == QUOTE).sum()),
        "trades": int((kinds == TRADE).sum()),
        "conditions": int((kinds == CONDITION).sum()),
    }
    assert sum(n for _, n in m["profile"]) == len(rows) and len(m["profile"]) == 2
    assert m["profile"][0][0] == m["start_ns"] // 10**9 % 86400
    directory = validate_manifest(
        {
            **m,
            "generation": 1,
            "wire_protocol": PROTOCOL,
            "market_timezone": TIMEZONE,
            "sender_port": 1,
        }
    )
    assert directory.universe == 40 and directory.midnight_ns == MIDNIGHT


def test_tape_quotes_are_mostly_valid_with_rare_anomalies(tmp_path):
    synthetic_tape(tmp_path, symbols=30, packets=2000, seconds=600, seed=2)
    _, rows = open_tape(tmp_path)
    q = rows[rows["kind"] == QUOTE]
    one_sided, crossed = np.isnan(q["bid"]), q["bid"] > q["ask"]
    halted = (q["flags"] & HALTED) != 0
    assert one_sided.any() and crossed.any() and halted.any()
    assert (one_sided | crossed | halted).mean() < 0.01
    assert (q["bid"][~one_sided] > 0).all()
    ok = ~(one_sided | crossed | halted)
    spread_bps = (q["ask"][ok] - q["bid"][ok]) / ((q["ask"][ok] + q["bid"][ok]) / 2) * 1e4
    assert (spread_bps > 0).all() and (spread_bps <= 25).all()
    conditions = rows[rows["kind"] == CONDITION]
    assert set(conditions["flags"].tolist()) == {0, HALTED}
    trades = rows[rows["kind"] == TRADE]
    assert (trades["price"] > 0).all() and (trades["volume"] >= 1).all()
    # Most symbols keep a complete return window.
    bars = QuoteBars([f"S{i}" for i in range(30)], window_seconds=300, min_bars=20)
    bars.observe(ReplayBatch(np.asarray(rows), 1, 0, 1))
    window, _ = bars.window()
    assert np.isfinite(window).all(axis=0).mean() > 0.6


def test_tapes_are_deterministic_by_seed(tmp_path):
    for name, seed in (("a", 3), ("b", 3), ("c", 4)):
        synthetic_tape(tmp_path / name, symbols=20, packets=50, seconds=60, seed=seed)
    read = lambda name: (tmp_path / name / "events.bin").read_bytes()  # noqa: E731
    assert read("a") == read("b") != read("c")
    assert (tmp_path / "a/manifest.json").read_text() == (tmp_path / "b/manifest.json").read_text()


def test_capture_carries_the_tape_as_replay_units(tape, tmp_path):
    out, manifest = tape
    _, rows = open_tape(out)
    counts = synthetic_capture(out, tmp_path / "day.pcap", sender_port=40123, start_seq=5)
    frames = PcapFrames.read(tmp_path / "day.pcap")
    assert len(frames) == counts["packets"] == 300 + 1  # full units plus the end marker
    expected, events, messages = 5, [], 0
    for i in range(len(frames)):
        frame = frames.frame(i)
        payload, peer = strip_udp_datagram(frame, minimum_payload=PACKET_HEADER)
        assert peer[1] == 40123 and struct.unpack_from(">H", frame, 36)[0] == 26400
        (batch,), count = decode(payload, 40, recv_ns=i)
        assert batch.sequence == expected
        expected += count
        messages += count
        events.append(batch.events)
        assert frames.ts_ns[i] == batch.events["timestamp_ns"][-1]  # record time = event time
    decoded = np.concatenate(events)
    for name in ("timestamp_ns", "symbol_id", "kind"):
        np.testing.assert_array_equal(decoded[name][:-1], rows[name], err_msg=name)
    # Every entry carries the directory index of its instrument.
    _, entries = sbe.parse_packet(strip_udp_datagram(frames.frame(0))[0])
    assert (entries["SecurityID"] < len(manifest["symbols"])).all()
    assert len(entries) == MAX_EVENTS
    marker = decoded[-1]
    assert (marker["timestamp_ns"], marker["kind"], marker["flags"]) == (
        manifest["end_ns"],
        CONDITION,
        0,
    )
    for name in ("bid", "ask", "bid_size", "ask_size", "price", "volume", "flags"):
        np.testing.assert_array_equal(decoded[name][:-1], rows[name], err_msg=name)
    assert messages == len(rows) + 1  # one entry per observation
    assert counts == dict(
        packets=301,
        events=len(rows) + 1,
        messages=messages,
        dropped_packets=0,
        dropped_events=0,
        dropped_messages=0,
        first_sequence=5,
        next_sequence=expected,
        sender_port=40123,
        dst_port=26400,
        encoder="numpy",
    )


def test_drop_every_leaves_sequence_gaps_but_keeps_the_marker(tape, tmp_path):
    out, manifest = tape
    full = synthetic_capture(out, tmp_path / "full.pcap")
    counts = synthetic_capture(out, tmp_path / "drops.pcap", drop_every=7)
    assert counts["dropped_packets"] == 300 // 7 and counts["packets"] == 301 - 300 // 7
    assert counts["messages"] + counts["dropped_messages"] == full["messages"]
    assert counts["events"] + counts["dropped_events"] == full["events"]
    assert counts["next_sequence"] == full["next_sequence"]
    expected, gaps = 1, 0
    for payload in PcapFrames.read(tmp_path / "drops.pcap").payloads():
        (batch,), count = decode(payload, 40)
        gaps += batch.sequence - expected
        expected = batch.sequence + count
    assert gaps == counts["dropped_messages"] > 0
    assert batch.events["timestamp_ns"][-1] == manifest["end_ns"]


def test_invalid_arguments_are_rejected(tape, tmp_path):
    out, _ = tape
    for bad in (
        dict(symbols=0),
        dict(seconds=86400 - 34200),
        dict(max_events_per_packet=MAX_EVENTS + 1),
    ):
        with pytest.raises(ValueError):
            synthetic_tape(tmp_path / "bad", **bad)
    for bad in (
        dict(drop_every=-1),
        dict(sender_port=0),
        dict(dst_port=65536),
        dict(start_seq=0),
        dict(encoder="fpga"),
    ):
        with pytest.raises(ValueError):
            synthetic_capture(out, tmp_path / "bad.pcap", **bad)


def run_synth(*args):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    return subprocess.run(
        [sys.executable, "-m", "holoscan_finance.tools.synth", *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_synth_cli_writes_a_data_set_and_reuses_it(tmp_path):
    out = tmp_path / "s"
    args = ["--out", str(out), "--symbols", "20", "--packets", "200", "--seconds", "120"]
    first = run_synth(*args)
    assert first.returncode == 0, first.stderr
    summary = json.loads((out / "synth.json").read_text())
    assert summary["args"]["packets"] == 200 and summary["args"]["seed"] == 7
    assert summary["tape"]["events"] == 200 * MAX_EVENTS and summary["capture"]["packets"] == 201
    assert summary["files"]["day.pcap"] == (out / "day.pcap").stat().st_size
    stamps = {p: (out / p).stat().st_mtime_ns for p in synth._FILES}
    reused = run_synth(*args, "--reuse")
    assert reused.returncode == 0 and "--reuse" in reused.stdout
    assert {p: (out / p).stat().st_mtime_ns for p in synth._FILES} == stamps
    assert json.loads((out / "synth.json").read_text()) == summary
    # Other arguments regenerate the data set.
    assert synth.main([*args, "--drop-every", "5", "--reuse"]) == 0
    changed = json.loads((out / "synth.json").read_text())
    assert changed["capture"]["dropped_packets"] == 40
    # A data set whose files changed is not reused.
    (out / "day.pcap").write_bytes(b"")
    assert synth.main([*args, "--drop-every", "5", "--reuse"]) == 0
    assert (out / "day.pcap").stat().st_size == changed["files"]["day.pcap"]
    bad = run_synth("--out", str(tmp_path / "bad"), "--symbols", "0")
    assert bad.returncode != 0 and "symbols" in bad.stderr
