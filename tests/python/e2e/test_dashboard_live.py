# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The live dashboard end to end: ``equity_analytics`` replaying synthetic data behind uvicorn.

A client polls ``/snapshot`` at 2 Hz like the browser and exercises ``/manifest``,
``/feedrate``, ``/capture``, ``/capture/finish`` and ``/plan`` while the NIC-free replay runs
through the production GPU chain; ``--exit-when-drained`` ends the server once everything
was analyzed, with exit code 0. Needs the SDK, a GPU and every work package.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from holoscan_finance.nyq.mapping import PROTOCOL, TIMEZONE
from holoscan_finance.nyq.tape import open_tape

httpx = pytest.importorskip("httpx")

pytestmark = [pytest.mark.sdk, pytest.mark.gpu, pytest.mark.slow]

REPO = Path(__file__).parents[3]
APP = REPO / "applications/equity_analytics/equity_analytics.py"
WINDOW_BARS = 12  # --window-seconds 60 / --bar-seconds 5: a full window within seconds
SNAPSHOT_KEYS = {
    "graph_status", "error", "comparison_failure", "settings",
    "minimum_update_interval_seconds", "analytics_backend", "feed_control", "sender_status",
    "generation", "live", "comparison", "comparisons", "history", "capture", "profile",
    "universe", "network", "fullday", "fullday_available", "engine",
}  # fmt: skip
FINISH_ADDED = {"ingest", "settings", "wire_protocol", "minimum_update_interval_seconds"}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_until(predicate, timeout, interval=0.5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    return None


def test_dashboard_contract_while_replaying(tmp_path):
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join([str(REPO / "python"), os.environ.get("PYTHONPATH", "")]),
    )
    synth = tmp_path / "synth"
    subprocess.run(
        [sys.executable, "-m", "holoscan_finance.tools.synth", "--out", str(synth),
         "--symbols", "200", "--packets", "20000", "--seconds", "900"],
        check=True, env=env, cwd=REPO, timeout=300,
    )  # fmt: skip
    port, report = free_port(), tmp_path / "report.json"
    app = subprocess.Popen(
        [sys.executable, str(APP), "--ingest", "replay", "--pcap", str(synth / "day.pcap"),
         "--tape", str(synth / "tape"), "--replay-pps", "1000", "--analytics-backend", "compare",
         "--window-seconds", "60", "--min-bars", "5", "--host", "127.0.0.1", "--port", str(port),
         "--exit-when-drained", "--report", str(report)],
        env=env, cwd=REPO,
    )  # fmt: skip
    base = f"http://127.0.0.1:{port}"
    try:
        with httpx.Client(base_url=base, timeout=10) as client:

            def snapshot():
                try:
                    r = client.get("/snapshot")
                except httpx.TransportError:
                    return None  # not listening yet
                return r.json() if r.status_code == 200 else None

            s = wait_until(snapshot, timeout=180)
            assert s is not None, "the dashboard never answered"
            assert SNAPSHOT_KEYS <= set(s)
            plan = client.get("/plan").json()
            assert {lane["name"] for lane in plan["lanes"]} >= {"ingest", "analytics", "crosscheck"}

            # The tools' calls: re-posting the current generation is a no-op.
            def registered():
                x = snapshot()
                return x if x and x["generation"] else None

            s = wait_until(registered, timeout=60)
            assert s is not None, "the replay never registered a generation"
            payload = {
                **open_tape(str(synth / "tape"))[0],
                "format": "nyq-equity-replay-v1",
                "wire_protocol": PROTOCOL,
                "market_timezone": TIMEZONE,
                "sender_port": 1,
                "generation": s["generation"],
            }
            r = client.post("/manifest", json=payload)
            assert r.status_code == 200 and r.json() == {"generation": s["generation"]}
            assert client.get("/feedrate").json()["seek_seq"] == 1
            assert client.post("/feedrate", json={"speed": 25}).json()["speed"] == 25

            # A latency capture over a full window, as measure.py takes one.
            def full_window():
                x = snapshot()
                return x if x and x["live"] and x["live"]["bars"] >= WINDOW_BARS else None

            s = wait_until(full_window, timeout=120)
            assert s is not None, "no full return window was analyzed"
            assert s["network"]["ingest"] == "replay"
            assert s["engine"]["lanes"] and s["engine"]["termination"] == "RUNNING"
            assert s["error"] is None and s["observations"]["events"] > 0
            status = client.post("/capture", json={"seconds": 1}).json()
            assert status["analytics_backend"] == "compare" and not status["closed"]
            time.sleep(1.2)
            result = client.post("/capture/finish").json()
            assert FINISH_ADDED <= set(result) and result["ingest"] == "replay"
            assert result["wire_protocol"] == PROTOCOL
            assert client.get("/capture").json()["closed"] is True

            # Browser-rate polling until the replay drains and the server exits.
            deadline = time.monotonic() + 240
            while app.poll() is None and time.monotonic() < deadline:
                x = snapshot()
                if x is not None:
                    assert x["error"] is None, x["error"]
                time.sleep(0.5)
        assert app.wait(timeout=30) == 0
    finally:
        if app.poll() is None:
            app.terminate()
            app.wait(timeout=30)
    assert isinstance(json.loads(report.read_text()), dict)  # written after the drained stop
