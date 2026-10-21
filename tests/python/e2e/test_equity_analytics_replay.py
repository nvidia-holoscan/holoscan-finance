# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ``smoke`` mode end to end: a synthetic day through the whole GPU chain with the CPU
cross-check, headless, stopped once drained.

Synthetic tape and capture -> ``GpuMitchReplaySourceOp`` (pcap frames in GPU memory, GPU
gather and decode) -> ``QuoteBarsOp(gpu)`` -> ``RollingRiskOp`` -> results, with every
validated window recomputed by ``RiskCrossCheckOp`` on the CPU. The run must stop by itself
once drained, analyze windows, publish agreeing comparisons and report latencies.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from apphelpers import REPO, child_env

pytestmark = [pytest.mark.sdk, pytest.mark.gpu, pytest.mark.slow]


def run(*argv: str, timeout: float) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, *argv], capture_output=True, text=True, timeout=timeout, env=child_env()
    )


def test_smoke_mode_drains_analyzes_and_agrees(tmp_path: Path) -> None:
    data, report = tmp_path / "synth-smoke", tmp_path / "equity_smoke.json"
    synth = run(
        "-m", "holoscan_finance.tools.synth", "--out", str(data), "--symbols", "200",
        "--packets", "20000", "--seconds", "900", "--reuse",
        timeout=300,
    )  # fmt: skip
    assert synth.returncode == 0, synth.stderr[-3000:]

    done = run(
        str(REPO / "applications/equity_analytics/equity_analytics.py"),
        "--ingest", "replay", "--pcap", str(data / "day.pcap"), "--tape", str(data / "tape"),
        "--replay-pps", "0", "--port", "0", "--exit-when-drained", "--timeout-s", "300",
        "--analytics-backend", "compare", "--expect-windows", "1", "--report", str(report),
        timeout=420,
    )  # fmt: skip
    assert done.returncode == 0, done.stderr[-5000:]

    r = json.loads(report.read_text())
    assert r["termination"] == "STOPPED" and r["stop_reason"] == "drained"
    assert r["errors"] == [] and r["unmet_expectations"] == []
    ingest = r["ingest"]
    capture = json.loads((data / "synth.json").read_text())["capture"]
    assert ingest["front_end"] == "replay" and ingest["packets"] == capture["packets"]
    assert ingest["messages"] == ingest["received_messages"] > 0
    for loss in (
        "sequence_gaps",
        "ring_dropped",
        "stale_packets",
        "malformed_packets",
        "decode_malformed",
    ):
        assert ingest[loss] == 0, loss
    assert r["bars"]["windows_emitted"] >= 1 and r["bars"]["observations"]["completed_bars"] > 0
    assert r["risk"]["analyzed"] >= 1
    assert r["results"]["comparisons"] >= 1 and r["results"]["agreement"] is True
    latency = r["results"]["receive_to_result_ms"]
    assert latency["samples"] >= 1 and latency["p50"] is not None
    assert {lane["name"] for lane in r["plan"]["lanes"]} == {"ingest", "analytics", "crosscheck"}
