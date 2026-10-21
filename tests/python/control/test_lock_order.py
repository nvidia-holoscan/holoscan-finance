# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The lock order (docs/ARCHITECTURE.md) under stress: manifests from two writers against the data plane.

Two threads register 1,000 generations (one like ``POST /manifest``, one like the replay
loop's ``ReplayGenerations``) while a 1 kHz thread publishes risk results and comparisons
and a snapshot thread reads the dashboard (50 Hz, above the browser's 2 Hz), with a
full-day burst and a capture in flight so every listener does its work. A 30 s watchdog
fails the test on a hang; listeners must see the generations in increasing order.
"""

from __future__ import annotations

import faulthandler
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from holoscan_finance.control import fullday as fullday_module
from holoscan_finance.control.directory import ReplayDirectory, ReplayGenerations
from holoscan_finance.control.feed import FeedControl
from holoscan_finance.control.fullday import FullDayController
from holoscan_finance.control.gate import CrossCheckGate
from holoscan_finance.control.results import ResultsStore
from holoscan_finance.control.status import RunStatus
from holoscan_finance.dashboard.model import DashboardModel
from holoscan_finance.marketdata.covariance import CovarianceEngine
from holoscan_finance.nyq.mapping import PROTOCOL, TIMEZONE, day_start_ns
from holoscan_finance.payloads import BarWindow, RiskComparison, RiskUpdate
from holoscan_finance.ref import Knob
from holoscan_finance.settings import BarSettings
from holoscan_finance.telemetry.capture import CaptureTap
from holoscan_finance.telemetry.stalls import StallMonitor
from holoscan_finance.telemetry.stats import BarsStats, IngestStats, RiskStats

pytest.importorskip("scipy")

WATCHDOG_S = 30.0
MANIFESTS_PER_WRITER = 500
MIDNIGHT = day_start_ns("2025-04-01")
SYMBOLS = [f"S{i}" for i in range(16)]
BASE = dict(
    format="nyq-equity-replay-v1",
    wire_protocol=PROTOCOL,
    market_timezone=TIMEZONE,
    symbols=SYMBOLS,
    date="2025-04-01",
    start_ns=MIDNIGHT + 34200 * 10**9,
    end_ns=MIDNIGHT + 57600 * 10**9,
    events=100,
    profile=[],
)


def make_ctx():
    settings = BarSettings(window_seconds=40, min_bars=2)
    directory = ReplayDirectory(Knob())
    ingest, bars = IngestStats(), BarsStats()
    tap = CaptureTap()
    feed = FeedControl(directory)
    monitor = StallMonitor(ingest)
    ctx = SimpleNamespace(
        args=SimpleNamespace(ingest="replay", analytics_backend="compare", interval=0.05),
        settings=settings, directory=directory, feed=feed, ingest_stats=ingest,
        bars_stats=bars, risk_stats=RiskStats(), tap=tap, stall_monitor=monitor,
        status=RunStatus(), plan_summary=None,
    )  # fmt: skip
    # The gate asks the full-day controller, which needs the results store, which needs the
    # gate: the gate looks the controller up when it is asked.
    ctx.gate = CrossCheckGate(allowed=lambda: ctx.fullday.validation_allowed())
    ctx.results = ResultsStore(
        backend="compare",
        interval_s=0.05,
        settings=settings,
        directory=directory,
        tap=tap,
        gate=ctx.gate,
    )
    ctx.fullday = FullDayController(
        replay_control="http://controller.invalid",
        feed=feed,
        ingest=ingest,
        bars=bars,
        results=ctx.results,
        stalls=monitor,
        backend="compare",
    )
    return ctx


@pytest.fixture
def fine_grained_switching():
    # Switch threads every 10 us instead of 5 ms: many more interleavings per run.
    interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-5)
    yield
    sys.setswitchinterval(interval)


def test_manifests_listeners_and_publication_never_deadlock(monkeypatch, fine_grained_switching):
    manifest_arrived = threading.Event()

    def controller(url, payload, timeout=5):
        # The burst's manifest arrives while it is starting: the full-day listener then
        # takes its lock and the results lock inside the listener chain, under load.
        manifest_arrived.clear()
        manifest_arrived.wait(timeout)
        return {"packets": 1}

    monkeypatch.setattr(fullday_module, "replay_request", controller)
    ctx = make_ctx()
    seen: list[int] = []

    def record(d):
        seen.append(d.generation)
        manifest_arrived.set()

    for listener in (
        ctx.results.reset_generation,
        ctx.tap.on_generation,
        ctx.fullday.on_generation,
        record,
    ):
        ctx.directory.add_listener(listener)
    ctx.directory.set_manifest({**BASE, "sender_port": 40000, "generation": 1})
    ctx.tap.start(600, 1, "compare")
    model = DashboardModel(ctx)

    rng = np.random.default_rng(3)
    parts = tuple(rng.normal(0.0, 1e-3, len(SYMBOLS)) for _ in range(8))
    times = MIDNIGHT + (34200 + 5 * np.arange(1, 9, dtype=np.int64)) * 10**9
    probe = BarWindow(parts, times, 1, 8, 0)
    gpu = {
        **CovarianceEngine("cpu", bar_seconds=5).update(probe.rows, times),
        "gpu_name": "Test GPU",
        "finished_wall_ns": time.time_ns(),
    }
    cpu = {**gpu, "threads": 1, "blas": [], "cpu_affinity": [0], "method": gpu["method"]}

    done = threading.Event()
    errors: list[BaseException] = []
    counts = dict(risk=0, snapshots=0)

    def guarded(fn):
        def run():
            try:
                fn()
            except BaseException as exc:  # reported by the main thread
                errors.append(exc)

        return run

    def http_writer():
        for i in range(MANIFESTS_PER_WRITER):
            generation = max(ctx.directory.generation + 1, time.time_ns())
            ctx.directory.set_manifest({**BASE, "sender_port": 10000 + i, "generation": generation})
            if i % 50 == 0:
                try:
                    ctx.fullday.start(200_000)
                except ValueError:  # the burst is running: POST /replay/fullday would 422
                    pass

    def replay_writer():
        loops = ReplayGenerations(ctx.directory, BASE, first_port=30000)
        for k in range(1, MANIFESTS_PER_WRITER + 1):
            loops(k)

    def data_plane():  # the analytics and crosscheck lanes' callbacks at 1 kHz
        while not done.is_set():
            w = BarWindow(parts, times, ctx.directory.generation, 8, time.time_ns())
            update = RiskUpdate(w, gpu, w.rows, "gpu", True)
            ctx.results.on_risk(update)
            if ctx.gate.try_claim():
                ctx.results.on_comparison(RiskComparison(update, cpu, True, True, True))
            counts["risk"] += 1
            time.sleep(0.001)

    def dashboard():
        while not done.is_set():
            model.snapshot()
            counts["snapshots"] += 1
            time.sleep(0.02)

    writers = [
        threading.Thread(target=guarded(f), daemon=True) for f in (http_writer, replay_writer)
    ]
    readers = [threading.Thread(target=guarded(f), daemon=True) for f in (data_plane, dashboard)]
    for t in readers + writers:
        t.start()
    deadline = time.monotonic() + WATCHDOG_S
    for t in writers:
        t.join(max(0.0, deadline - time.monotonic()))
    time.sleep(0.1)  # a few more data-plane rounds on the final generation
    done.set()
    for t in readers:
        t.join(max(0.0, deadline - time.monotonic()))
    if any(t.is_alive() for t in writers + readers):
        faulthandler.dump_traceback(file=sys.stderr)
        pytest.fail(f"lock-order deadlock: no progress within {WATCHDOG_S} s")
    assert not errors, errors

    assert len(seen) >= MANIFESTS_PER_WRITER  # every replay pass registered a generation
    assert all(a < b for a, b in zip(seen, seen[1:])), "listeners ran out of order"
    assert seen[-1] == ctx.directory.generation
    assert ctx.results.received_risk == counts["risk"] and counts["snapshots"] >= 2
    assert ctx.results.error is None and not ctx.results.disabled
    assert ctx.gate.ready() and ctx.gate.claims == ctx.gate.releases
    assert ctx.results.received_comparisons == ctx.gate.claims
    assert ctx.results.published >= 1  # the final generation's results were published
    assert ctx.fullday.status()["status"] == "running"  # its listener ran under load
