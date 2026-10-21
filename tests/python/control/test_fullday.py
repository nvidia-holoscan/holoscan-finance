# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FullDayController: start/running/done/error, exact counters, validation paused in a burst."""

from __future__ import annotations

import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace

import pytest

from holoscan_finance.control import fullday as fullday_module
from holoscan_finance.control.directory import ReplayDirectory
from holoscan_finance.control.feed import FeedControl
from holoscan_finance.control.fullday import FullDayController, replay_request
from holoscan_finance.control.gate import CrossCheckGate
from holoscan_finance.control.results import ResultsStore
from holoscan_finance.nyq.mapping import PROTOCOL, TIMEZONE, day_start_ns
from holoscan_finance.payloads import BarCounters
from holoscan_finance.ref import Knob
from holoscan_finance.settings import BarSettings
from holoscan_finance.telemetry.capture import CaptureTap
from holoscan_finance.telemetry.stalls import StallMonitor
from holoscan_finance.telemetry.stats import BarsStats, IngestStats

MIDNIGHT = day_start_ns("2025-04-01")
URL = "http://replay-control.invalid:8095"


def manifest(generation, port):
    return dict(
        format="nyq-equity-replay-v1",
        wire_protocol=PROTOCOL,
        market_timezone=TIMEZONE,
        symbols=["A", "B"],
        generation=generation,
        sender_port=port,
        date="2025-04-01",
        start_ns=MIDNIGHT + 34200 * 10**9,
        end_ns=MIDNIGHT + 57600 * 10**9,  # a 6.5-hour session
        events=1000,
        profile=[],
    )


@pytest.fixture
def controller(monkeypatch):
    """A controller on real control objects; replay_request answers like replay_control.py."""
    requests = []

    def fake_request(url, payload, timeout=5):
        requests.append((url, payload, timeout))
        return dict(pps=payload["pps"], sender_port=40000, started=True, packets=5000)

    monkeypatch.setattr(fullday_module, "replay_request", fake_request)
    directory = ReplayDirectory(Knob())
    directory.set_manifest(manifest(1, 50001))
    ingest, bars = IngestStats(), BarsStats()
    results = ResultsStore(
        backend="compare",
        interval_s=0.05,
        settings=BarSettings(),
        directory=directory,
        tap=CaptureTap(),
    )
    feed = FeedControl(directory)
    fd = FullDayController(
        replay_control=URL,
        feed=feed,
        ingest=ingest,
        bars=bars,
        results=results,
        stalls=StallMonitor(ingest),
        backend="compare",
    )
    directory.add_listener(results.reset_generation)
    directory.add_listener(fd.on_generation)
    return SimpleNamespace(
        fd=fd, directory=directory, feed=feed, ingest=ingest, bars=bars, results=results,
        requests=requests,
    )  # fmt: skip


def counters(generation, *, first, last, events=900, bars=12):
    return BarCounters(generation, events, 800, 100, 3, 0, bars, 0, first, last)


def test_without_a_controller_nothing_can_start():
    ingest = IngestStats()
    fd = FullDayController(
        replay_control=None,
        feed=FeedControl(ReplayDirectory(Knob())),
        ingest=ingest,
        bars=BarsStats(),
        results=SimpleNamespace(),
        stalls=StallMonitor(ingest),
        backend="compare",
    )
    assert not fd.available() and fd.status() is None
    with pytest.raises(ValueError, match="--replay-control"):
        fd.start(200_000)


@pytest.mark.parametrize("pps", [0, -1, 250_001, 2.0e5, "200000", True])
def test_a_bad_rate_is_refused_before_the_sender_pauses(controller, pps):
    c = controller
    with pytest.raises(ValueError, match="pps"):
        c.fd.start(pps)
    assert not c.feed.control()["paused"] and c.requests == [] and c.fd.status() is None


def test_start_pauses_the_sender_and_asks_the_controller(controller):
    c = controller
    f = c.fd.start(200_000)
    assert c.fd.available() and c.feed.control()["paused"] is True
    assert c.requests == [(URL, {"pps": 200_000}, 60)]
    assert f["status"] == "starting" and f["pps"] == 200_000 and f["controller"]["packets"] == 5000
    with pytest.raises(ValueError, match="already running"):
        c.fd.start(100_000)


def test_a_controller_error_ends_the_burst(controller, monkeypatch):
    monkeypatch.setattr(fullday_module, "replay_request", lambda *a, **k: {"error": "refused"})
    f = controller.fd.start(200_000)
    assert (f["status"], f["error"]) == ("error", "refused")
    assert controller.fd.validation_allowed()
    assert controller.fd.start(200_000)["status"] == "error"  # a new attempt is allowed


def test_the_burst_manifest_starts_the_measurement(controller):
    c = controller
    c.ingest.packets, c.ingest.dropped, c.ingest.ring_dropped = 1000, 7, 3
    c.ingest.stall_event("compute", 80.0)
    c.fd.stalls.heartbeat.update(max_ms=99.0, at_s=1.0)
    c.results.analytics_max_ms = 42.0
    c.fd.start(200_000)
    c.directory.set_manifest(manifest(2, 40000))
    f = c.fd.status()
    assert f["status"] == "running" and f["generation"] == 2 and f["sender_port"] == 40000
    assert f["tape_seconds"] == 23400.0 and f["base"] == (1000, 7, 3)
    assert c.ingest.stalls["events"] == [] and c.fd.stalls.heartbeat["max_ms"] == 0.0
    assert c.results.analytics_max_ms == 0.0
    c.directory.set_manifest(manifest(3, 40001))  # not starting any more: no new base
    assert c.fd.status()["generation"] == 2


def test_status_reads_exact_counters_until_the_burst_is_done(controller, caplog):
    c = controller
    c.fd.start(200_000)
    c.directory.set_manifest(manifest(2, 40000))
    now = time.monotonic_ns()
    c.ingest.packets, c.ingest.dropped, c.ingest.ring_dropped = 4000, 2, 1
    c.bars.counters = counters(2, first=now - 1_000_000_000, last=now)
    f = c.fd.status()
    assert f["status"] == "running"
    assert (f["packets"], f["total_packets"], f["sequence_gaps"], f["ring_dropped"]) == (
        4000,
        5000,
        2,
        1,
    )
    assert (f["events"], f["completed_bars"], f["seconds"]) == (900, 12, 1.0)
    assert f["first_s"] == round((now - 1_000_000_000) / 1e9, 3)
    assert {"heartbeat", "gc_max_ms", "analytics_max_ms", "events"} <= set(f["stalls"])
    assert not c.fd.validation_allowed()
    # The last batch is over 2 s old and nothing is pending: the burst is done.
    c.bars.counters = counters(2, first=now - 4_000_000_000, last=now - 3_000_000_000)
    with caplog.at_level(logging.INFO, logger="holoscan_finance.control.fullday"):
        f = c.fd.status()
    assert f["status"] == "done" and f["x_realtime"] == 23400.0
    assert "full-day replay done: 4000 packets, gaps 2, ring drops 1" in caplog.text
    assert c.fd.validation_allowed()
    f["status"] = "edited"
    assert c.fd.status()["status"] == "done"  # status() returns copies


def test_counters_of_another_generation_are_not_the_burst(controller):
    c = controller
    c.fd.start(200_000)
    c.directory.set_manifest(manifest(2, 40000))
    c.bars.counters = counters(1, first=1, last=2)
    f = c.fd.status()
    assert f["status"] == "running" and f["events"] is None and f["seconds"] is None


def test_a_burst_without_packets_fails_after_two_minutes(controller, monkeypatch):
    c = controller
    c.fd.start(200_000)
    c.directory.set_manifest(manifest(2, 40000))
    later = time.time() + 121
    clock = SimpleNamespace(time=lambda: later, monotonic_ns=time.monotonic_ns)
    monkeypatch.setattr(fullday_module, "time", clock)
    f = c.fd.status()
    assert (f["status"], f["error"]) == ("error", "no packets arrived from the replay controller")


def test_status_is_frozen_once_another_generation_takes_over(controller):
    c = controller
    c.fd.start(200_000)
    c.directory.set_manifest(manifest(2, 40000))
    c.directory.set_manifest(manifest(3, 40001))  # e.g. the paced sender resumed
    c.ingest.packets = 999
    assert "packets" not in c.fd.status()


def test_cpu_validation_pauses_during_a_full_day_burst(controller):
    # From 4.3 test_equities: the CPU comparator competes with ingest during a burst.
    c = controller
    gate = CrossCheckGate(allowed=c.fd.validation_allowed)
    assert gate.try_claim()
    gate.release()
    c.fd.start(200_000)
    assert not gate.try_claim()  # starting
    c.directory.set_manifest(manifest(2, 40000))
    assert not gate.try_claim()  # running
    assert gate.ready()  # risk keeps analyzing, unvalidated


@pytest.mark.parametrize("backend", ["gpu", "cpu"])
def test_single_backend_runs_never_validate(backend, monkeypatch):
    monkeypatch.setattr(fullday_module, "replay_request", lambda *a, **k: {"packets": 1})
    ingest = IngestStats()
    fd = FullDayController(
        replay_control=URL,
        feed=FeedControl(ReplayDirectory(Knob())),
        ingest=ingest,
        bars=BarsStats(),
        results=SimpleNamespace(),
        stalls=StallMonitor(ingest),
        backend=backend,
    )
    assert fd.start(200_000)["status"] == "starting" and fd.validation_allowed()


def test_replay_request_posts_json_to_the_controller():
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            received.append((self.path, self.headers["Content-Type"], json.loads(body)))
            reply = json.dumps({"pps": 1000, "started": True, "packets": 7}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/"
        assert replay_request(url, {"pps": 1000}) == {"pps": 1000, "started": True, "packets": 7}
    finally:
        server.shutdown()
        server.server_close()
    assert received == [("/start", "application/json", {"pps": 1000})]


def test_replay_request_reports_failures_instead_of_raising():
    reply = replay_request("http://127.0.0.1:9", {"pps": 1}, timeout=1)
    assert set(reply) == {"error"} and reply["error"]
