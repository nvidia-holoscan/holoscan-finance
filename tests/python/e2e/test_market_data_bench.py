# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``market_data_bench --check`` end to end on synthetic captures: the whole GPU chain runs,
drains and stops by itself, and every bar matches the CPU backend and the host decode,
also on a capture with missing packets.

Each run is a subprocess, as from the command line: the runner sets the process's CPU mask,
switch interval, logging and signal handlers.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from apphelpers import REPO, child_env

pytestmark = [pytest.mark.sdk, pytest.mark.gpu, pytest.mark.slow]

BENCH = REPO / "applications" / "market_data_bench" / "market_data_bench.py"


def run(*argv: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, *argv], env=child_env(), capture_output=True, text=True, timeout=900
    )


def synth(out: Path, *flags: str) -> dict:
    sizes = ["--symbols", "300", "--packets", "20000"]
    done = run("-m", "holoscan_finance.tools.synth", "--out", str(out), *sizes, *flags)
    assert done.returncode == 0, done.stderr
    return json.loads((out / "synth.json").read_text())


def bench(data: Path, report: Path, *flags: str) -> tuple[int, dict, str]:
    capture = ["--pcap", str(data / "day.pcap"), "--tape", str(data / "tape")]
    done = run(str(BENCH), *capture, "--pps", "0", "--check", "--report", str(report), *flags)
    return done.returncode, json.loads(report.read_text()), done.stderr[-4000:]


def assert_identical(report: dict) -> None:
    check = report["check"]
    assert check["windows_compared"] > 0 and check["max_ulps"] <= 1
    assert check["times_equal"] and check["nan_pattern_equal"] and check["final_state_equal"]
    ref = check["reference"]
    assert ref["closed_bars_compared"] > 0 and ref["max_ulps"] <= 1
    assert all(
        ref[k]
        for k in ("times_equal", "nan_pattern_equal", "counters_equal", "decode_counts_equal")
    )


def test_lossless_check_on_a_synthetic_capture(tmp_path):
    summary = synth(tmp_path / "data")
    rc, report, log = bench(tmp_path / "data", tmp_path / "check.json", "--require-lossless")
    assert rc == 0 and report["ok"], (report["failures"], log)
    assert (report["termination"], report["stop_reason"]) == ("STOPPED", "drained")
    assert_identical(report)
    assert report["packets"] == summary["capture"]["packets"]  # the units and the end marker
    assert report["sequence_gaps"] == report["nic_dropped"] == report["ring_dropped"] == 0
    assert report["completed_bars"] > 0 and report["msgs_per_s"] > 0
    assert set(report["lanes"]) == {"ingest", "reference"}


def test_check_holds_on_a_capture_with_missing_packets(tmp_path):
    summary = synth(tmp_path / "drops", "--drop-every", "997")
    rc, report, log = bench(tmp_path / "drops", tmp_path / "drops.json")
    assert rc == 0 and report["ok"], (report["failures"], log)
    assert_identical(report)  # counters_equal: the bars' gaps equal the reference's
    capture = summary["capture"]
    assert capture["dropped_messages"] > 0
    # The receive accounting counts every missing message, as the synthesizer dropped them.
    assert report["sequence_gaps"] == capture["dropped_messages"]
    assert report["packets"] == capture["packets"]


def test_looped_unpaced_replay_stays_lossless_at_the_bars(tmp_path):
    """Each pass is a new generation, and the loop boundary waits until the bars took every
    batch of the pass: none is counted stale, so --require-lossless holds over three passes.
    A small capture is one burst per pass, which the bars used to see only after the next
    generation was published."""
    data, report_path = tmp_path / "loops", tmp_path / "loops.json"
    summary = synth(data, "--symbols", "200", "--packets", "3000")
    capture = ["--pcap", str(data / "day.pcap"), "--tape", str(data / "tape")]
    flags = ["--pps", "0", "--loops", "3", "--require-lossless", "--report", str(report_path)]
    done = run(str(BENCH), *capture, *flags)
    report = json.loads(report_path.read_text())
    assert done.returncode == 0 and report["ok"], (report["failures"], done.stderr[-4000:])
    assert report["stale_packets"] == 0 and report["packets"] == 3 * summary["capture"]["packets"]
