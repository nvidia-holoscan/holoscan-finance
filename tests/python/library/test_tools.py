# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The command-line tools under ``holoscan_finance.tools``: the replay sender's wire packets,
``bench_replay inspect``/``register``, and every CLI's ``--help`` without the SDK."""

from __future__ import annotations

import ast
import json
import re
import socket
import subprocess
import sys
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest
from apphelpers import child_env

from holoscan_finance.marketdata.framing import PcapFrames, udp_frame, write_pcap
from holoscan_finance.marketdata.records import EVENT_DTYPE, QUOTE
from holoscan_finance.nyq.mapping import PROTOCOL, TIMEZONE, day_start_ns, pack_events
from holoscan_finance.nyq.synthetic import synthetic_capture, synthetic_tape
from holoscan_finance.tools import bench_replay, replay_sender, risk_bench

MIDNIGHT = day_start_ns("2025-04-01")


@contextmanager
def control_api(seek: float, *, ready_after: int = 0, graph: str = "running"):
    """A stand-in for the dashboard's control endpoints: ``/healthz`` answers 503 (the graph
    ``graph``) to the first ``ready_after`` requests, then 200; ``/manifest`` accepts every
    generation,
    ``/feedrate`` asks for an unpaced 20k observations/s from ``seek``. ``posted["order"]``
    lists every request's path."""
    posted: dict[str, list] = {"/manifest": [], "/feedstatus": [], "order": []}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 (http.server API)
            posted["order"].append(self.path)
            if self.path == "/healthz":
                ready = posted["order"].count("/healthz") > ready_after
                body = {"ready": ready, "graph_status": graph, "termination": None}
                self.reply(body, 200 if ready else 503)
                return
            assert self.path == "/feedrate"
            self.reply({"speed": 0, "rate": 20000, "paused": False, "seek": seek, "seek_seq": 0})

        def do_POST(self):  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            posted["order"].append(self.path)
            posted[self.path].append(body)
            self.reply({"generation": body["generation"]} if self.path == "/manifest" else {})

        def reply(self, payload, status=200):
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", posted
    finally:
        server.shutdown()
        server.server_close()


def test_replay_sender_sends_what_synthetic_capture_writes(tmp_path):
    manifest = synthetic_tape(tmp_path / "tape", symbols=20, packets=150, seconds=120, seed=3)
    counts = synthetic_capture(tmp_path / "tape", tmp_path / "day.pcap")
    expected = PcapFrames.read(tmp_path / "day.pcap").payloads()
    assert len(expected) == counts["packets"] == 151  # 150 units + the end-of-session marker

    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 22)
    rx.bind(("127.0.0.1", 0))
    rx.settimeout(10)
    received: list[bytes] = []
    reader = threading.Thread(target=lambda: received.extend(rx.recv(4096) for _ in expected))
    reader.start()
    seek = manifest["start_ns"] // 10**9 % 86400  # the session start: the first observation
    argv = ["--data", str(tmp_path / "tape"), "--port", str(rx.getsockname()[1]), "--loops", "1"]
    with control_api(seek, ready_after=3) as (url, posted):
        rc = replay_sender.main([*argv, "--control-url", url])
    reader.join(15)
    rx.close()
    assert rc == 0 and received == expected
    # It waited for the receiver's front end: three 503s, the 200, then the first generation.
    assert posted["order"][:5] == ["/healthz"] * 4 + ["/feedrate"]
    (registered,) = posted["/manifest"]
    assert registered["wire_protocol"] == PROTOCOL and registered["market_timezone"] == TIMEZONE
    assert registered["symbols"] == manifest["symbols"] and registered["generation"] > 0


def test_replay_sender_gives_up_on_a_receiver_that_never_gets_ready_or_stopped(tmp_path):
    synthetic_tape(tmp_path / "tape", symbols=5, packets=10, seconds=60)
    argv = ["--data", str(tmp_path / "tape"), "--ready-timeout", "0.3"]
    with control_api(0, ready_after=10**6) as (url, posted):
        with pytest.raises(RuntimeError, match="not ready in 0.3 s: HTTP 503, graph running"):
            replay_sender.main([*argv, "--control-url", url])
    assert posted["/manifest"] == [] and set(posted["order"]) == {"/healthz"}
    with control_api(0, ready_after=10**6, graph="stopped") as (url, posted):
        with pytest.raises(RuntimeError, match="has stopped"):
            replay_sender.wait_ready(url, 60)
    with pytest.raises(RuntimeError, match="not ready in 0.2 s"):
        replay_sender.wait_ready("http://127.0.0.1:9", 0.2)  # nothing listens


def unit(sequence: int, count: int = 2) -> bytes:
    rows = np.zeros(count, EVENT_DTYPE)
    rows["timestamp_ns"] = MIDNIGHT + 34200 * 10**9 + np.arange(count) * 10**6
    rows["kind"] = QUOTE
    rows["bid"], rows["ask"], rows["bid_size"], rows["ask_size"] = 10.0, 10.01, 5, 5
    for f in ("price", "volume"):
        rows[f] = np.nan
    return pack_events(rows, sequence, MIDNIGHT)[0]


def test_bench_replay_inspect_reads_ports_and_sequences_past_ip_options(tmp_path):
    # Port 40001 sends 3 units (2 quotes + 1 Time = 3 messages each), then a gap of 3
    # messages; port 40002 interleaves. IHL 6 moves every UDP header by 4 bytes.
    frames = [
        udp_frame(unit(1), src_port=40001, ihl=6),
        udp_frame(unit(4), src_port=40001, ihl=6),
        udp_frame(unit(1), src_port=40002),
        udp_frame(unit(10), src_port=40001, ihl=6),
        udp_frame(unit(4), src_port=40002, ihl=7),
    ]
    write_pcap(tmp_path / "c.pcap", frames)
    out = bench_replay.inspect(str(tmp_path / "c.pcap"), 40001)
    assert out["ports_in_order"] == [40001, 40002, 40001, 40002]
    assert out["packets_per_port"] == {40001: 3, 40002: 2}
    assert (out["packets"], out["first_sequence"], out["messages"]) == (3, 1, 9)
    assert out["discontinuities"] == 1
    assert "first_sequence" not in bench_replay.inspect(str(tmp_path / "c.pcap"))
    assert bench_replay.inspect(str(tmp_path / "c.pcap"), 40002)["discontinuities"] == 0


def test_risk_bench_cpu_agrees_with_the_oracle_and_reports_each_window(tmp_path):
    pytest.importorskip("scipy")
    pytest.importorskip("threadpoolctl")
    out = tmp_path / "risk_bench.json"
    argv = ["--sizes", "30,45", "--backends", "cpu", "--windows", "4", "--cpu-threads", "1"]
    assert risk_bench.main([*argv, "--check", "--out", str(out)]) == 0
    report = json.loads(out.read_text())
    for size in ("30", "45"):
        entry = report["sizes"][size]
        assert entry["check"] == {"cpu_vs_oracle": "5/5"}
        assert entry["cpu"]["engine_ms"]["n"] == 4 and entry["cpu"]["eligible"] == int(size)
        assert entry["cpu"]["first_window"]["method"] == "rebuild"


def test_risk_bench_phases_time_every_window_kind_and_agree_with_the_oracle(tmp_path):
    """``--phases``: the windows of a live run after a start or seek, each phase timed apart,
    every window checked against the oracle."""
    pytest.importorskip("scipy")
    pytest.importorskip("threadpoolctl")
    out = tmp_path / "phases.json"
    argv = ["--sizes", "30", "--backends", "cpu", "--phases", "2", "--cpu-threads", "1"]
    assert risk_bench.main([*argv, "--check", "--out", str(out)]) == 0
    entry = json.loads(out.read_text())["sizes"]["30"]
    windows = len(risk_bench.phases())
    assert entry["check"] == {"cpu_vs_oracle": f"{windows}/{windows}"}
    phases = {k: v for k, v in entry["cpu"].items() if k != "threads"}
    assert {k: v["methods"] for k, v in phases.items()} == {
        "allocate": ["rebuild"],
        "first": ["rebuild"],
        "growth": ["incremental"],
        "slide1": ["incremental"],
        "slide2": ["incremental"],
        "slide3": ["incremental"],
        "rebuild": ["rebuild"],
    }
    assert phases["slide2"]["engine_ms"]["n"] == 20 and phases["first"]["engine_ms"]["n"] == 1


KIND_WORK = {  # what each --kinds window does: method, added, expired and window bars
    "first": ("rebuild", 20, 0, 20),
    "growth": ("incremental", 1, 0, 21),
    "rebuild": ("rebuild", 60, 0, 60),
    "slide1": ("incremental", 1, 1, 60),
    "slide2": ("incremental", 2, 2, 60),
    "slide3": ("incremental", 3, 3, 60),
}


def test_risk_bench_kinds_times_every_window_kind_back_to_back_and_at_a_cadence(tmp_path):
    """``--kinds``: every window kind ``C`` times per pass (back to back, then with idle
    time before every window), each kind doing its own work, every kept result checked
    against the oracle, the CPU worker tuned once ahead of the first window, and the table."""
    pytest.importorskip("scipy")
    pytest.importorskip("threadpoolctl")
    out, table = tmp_path / "kinds.json", tmp_path / "kinds.md"
    argv = ["--sizes", "30", "--backends", "cpu", "--kinds", "3", "--cadence-ms", "1"]
    argv += ["--cadence-cycles", "2", "--check", "--check-cycles", "2", "--cpu-threads", "1"]
    assert risk_bench.main([*argv, "--out", str(out), "--table", str(table)]) == 0
    report = json.loads(out.read_text())
    entry = report["sizes"]["30"]
    assert list(entry["passes"]) == ["back_to_back", "cadence_1ms"]
    for name, cycles in (("back_to_back", 3), ("cadence_1ms", 2)):
        cpu = entry["passes"][name]["cpu"]
        assert cpu["cpu_tunings"] == 1 and cpu["threads"] == 1 and cpu["eligible"] == 30
        assert set(cpu["warmup"]) == set(KIND_WORK)
        for kind, work in KIND_WORK.items():
            stats = cpu["kinds"][kind]
            assert stats["engine_ms"]["n"] == stats["wall_ms"]["n"] == cycles
            assert stats["work"] == [dict(zip(risk_bench.WORK, work))], kind
        assert entry["check"][f"{name}:cpu_vs_oracle"] == f"{2 * 6}/{2 * 6}"
    assert report["verdict"] == {"cells": 0, "gpu_below_cpu": 0, "not_below": []}
    rows = table.read_text().splitlines()
    assert rows[0].startswith("| symbols (eligible) | pass | kind | cpu |") and len(rows) == 2 + 12
    assert rows[2].startswith("| 30 (30) | back_to_back | first | ")


def test_risk_bench_kinds_refuses_the_other_modes_options():
    for argv in (["--phases", "1"], ["--gap-ms", "270"]):
        with pytest.raises(SystemExit, match="--cadence-ms"):
            risk_bench.main(["--sizes", "30", "--backends", "cpu", "--kinds", "1", *argv])
    with pytest.raises(SystemExit, match="choose from"):
        risk_bench.main(["--sizes", "30", "--backends", "compare", "--windows", "1"])


def test_risk_bench_derives_the_days_returns_from_a_tape(tmp_path):
    """Every bar of the session from host QuoteBars, in order; the first has no previous
    mark. Batching by 30 bars gives the bars of one batch."""
    from holoscan_finance.marketdata.bars import QuoteBars
    from holoscan_finance.marketdata.records import ReplayBatch
    from holoscan_finance.nyq.tape import open_tape
    from holoscan_finance.settings import BarSettings

    manifest = synthetic_tape(tmp_path / "tape", symbols=12, packets=400, seconds=600, seed=5)
    returns, times = risk_bench.tape_returns(str(tmp_path / "tape"))
    assert len(times) == returns.shape[0] > 100 and returns.shape[1] == 12
    np.testing.assert_array_equal(np.diff(times), 5 * 10**9)
    assert times[0] == manifest["start_ns"] + 5 * 10**9 and np.isnan(returns[0]).all()
    assert times[-1] <= manifest["end_ns"] < times[-1] + 5 * 10**9
    _, events = open_tape(tmp_path / "tape")
    bars = QuoteBars(manifest["symbols"], **BarSettings().as_params())
    bars.observe(ReplayBatch(np.asarray(events), 1, 0, 0))  # one batch: the last 60 bars
    at = np.searchsorted(times, np.asarray(bars.times))
    np.testing.assert_array_equal(times[at], np.asarray(bars.times))
    np.testing.assert_array_equal(returns[at], np.asarray(bars.rows))
    saved = tmp_path / "bars.npz"
    risk_bench.day_returns.cache_clear()
    day = risk_bench.day_returns(str(tmp_path / "tape"), str(saved))
    np.testing.assert_array_equal(np.load(saved)["returns"], day)


def test_bench_replay_register_posts_a_new_generation_for_the_port(tmp_path):
    manifest = synthetic_tape(tmp_path / "tape", symbols=5, packets=10, seconds=60)
    with control_api(0) as (url, posted):
        reply = bench_replay.register(40123, url, str(tmp_path / "tape"))
    (payload,) = posted["/manifest"]
    assert reply == {"generation": payload["generation"]}
    assert payload["sender_port"] == 40123 and payload["symbols"] == manifest["symbols"]
    assert payload["wire_protocol"] == PROTOCOL and payload["market_timezone"] == TIMEZONE


CPU_TOOLS = (
    "replay_sender",
    "synth",
    "bench_replay",
    "measure",
    "compare_runs",
    "decode_bench",
    "gpu_validate",
    "risk_bench",
)


@pytest.mark.parametrize("tool", CPU_TOOLS)
def test_help_needs_neither_holoscan_nor_cupy(tool):
    code = (
        "import runpy, sys; sys.modules['holoscan'] = sys.modules['cupy'] = None; "
        f"sys.argv = ['{tool}', '--help']; "
        f"runpy.run_module('holoscan_finance.tools.{tool}', run_name='__main__')"
    )
    done = subprocess.run(
        [sys.executable, "-P", "-c", code],
        env=child_env(),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert done.returncode == 0, done.stderr
    assert "usage:" in done.stdout


def test_usage_examples_name_the_documented_data_layout():
    """Usage text names the synthetic capture and tape generated by ``tools.synth``."""
    repo = Path(__file__).resolve().parents[3]
    files = sorted((repo / "python/holoscan_finance/tools").glob("*.py"))
    files.append(repo / "deploy/igx_thor/gpu_ingest_bench/replay_control.py")
    examples = []
    for path in files:
        doc = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8"))) or ""
        assert "base.pcap" not in doc, path.name
        examples += [(path.name, *m) for m in re.findall(r"--(pcap|tape) (\S+)", doc)]
    assert examples  # gpu_validate, decode_bench and replay_control show both
    for name, flag, value in examples:
        assert value == {"pcap": "data/synth/day.pcap", "tape": "data/synth/tape"}[flag], name
