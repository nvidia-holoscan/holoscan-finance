# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The command-line tools under ``holoscan_finance.tools``: the replay sender's wire packets,
``bench_replay inspect``/``register``, and every CLI's ``--help`` without the SDK."""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest
from apphelpers import child_env

from holoscan_finance.marketdata.framing import PcapFrames, udp_frame, write_pcap
from holoscan_finance.marketdata.records import EVENT_DTYPE, QUOTE
from holoscan_finance.nyq.mapping import PROTOCOL, TIMEZONE, day_start_ns, pack_events
from holoscan_finance.nyq.synthetic import synthetic_capture, synthetic_tape
from holoscan_finance.tools import bench_replay, replay_sender

MIDNIGHT = day_start_ns("2025-04-01")


@contextmanager
def control_api(seek: float):
    """A stand-in for the dashboard's control endpoints: ``/manifest`` accepts every
    generation, ``/feedrate`` asks for an unpaced 20k observations/s from ``seek``."""
    posted: dict[str, list] = {"/manifest": [], "/feedstatus": []}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 (http.server API)
            assert self.path == "/feedrate"
            self.reply({"speed": 0, "rate": 20000, "paused": False, "seek": seek, "seek_seq": 0})

        def do_POST(self):  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            posted[self.path].append(body)
            self.reply({"generation": body["generation"]} if self.path == "/manifest" else {})

        def reply(self, payload):
            data = json.dumps(payload).encode()
            self.send_response(200)
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
    with control_api(seek) as (url, posted):
        rc = replay_sender.main([*argv, "--control-url", url])
    reader.join(15)
    rx.close()
    assert rc == 0 and received == expected
    (registered,) = posted["/manifest"]
    assert registered["wire_protocol"] == PROTOCOL and registered["market_timezone"] == TIMEZONE
    assert registered["symbols"] == manifest["symbols"] and registered["generation"] > 0


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
    "prepare_tape",
    "bench_replay",
    "measure",
    "compare_runs",
    "decode_bench",
    "gpu_validate",
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
