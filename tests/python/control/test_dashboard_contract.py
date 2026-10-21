# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The /snapshot contract: 4.3 key sets per ingest mode, backend and state, and pinned values.

The golden lists come from the 4.3 code paths (``EquityState.snapshot`` and ``network``);
``engine`` and ``network.ingest_mode`` are the only additions. Tools read the pinned values:
``measure.py`` and ``ano_runtime.wait`` read ``network.ingest``, ``index.html`` prints
``network.transport``. Every field ``index.html`` reads must be present. The context is the
``EquityContext`` attribute set over the real control and telemetry objects.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from holoscan_finance.control import fullday as fullday_module
from holoscan_finance.control.directory import ReplayDirectory
from holoscan_finance.control.feed import FeedControl
from holoscan_finance.control.fullday import FullDayController
from holoscan_finance.control.gate import CrossCheckGate
from holoscan_finance.control.results import ResultsStore
from holoscan_finance.control.status import RunStatus
from holoscan_finance.dashboard.model import DashboardModel
from holoscan_finance.marketdata.covariance import CovarianceEngine
from holoscan_finance.nyq.mapping import PROTOCOL, TIMEZONE, day_start_ns
from holoscan_finance.payloads import (
    BarCounters,
    BarWindow,
    RiskComparison,
    RiskUpdate,
    SymbolSnapshot,
)
from holoscan_finance.ref import Knob
from holoscan_finance.settings import BarSettings
from holoscan_finance.telemetry.capture import CaptureTap
from holoscan_finance.telemetry.stalls import StallMonitor
from holoscan_finance.telemetry.stats import BarsStats, IngestStats, RiskStats, SourceProbe

pytest.importorskip("scipy")

INDEX_HTML = Path(__file__).parents[3] / "python/holoscan_finance/dashboard/webui/index.html"
MIDNIGHT = day_start_ns("2025-04-01")
SYMBOLS = [f"SYM{i}" for i in range(40)]

# ---- golden 4.3 key sets ------------------------------------------------------------------
TOP = {
    "graph_status", "error", "comparison_failure", "settings",
    "minimum_update_interval_seconds", "analytics_backend", "feed_control", "sender_status",
    "generation", "live", "comparison", "comparisons", "history", "capture", "profile",
    "universe", "network", "fullday", "fullday_available",
}  # fmt: skip
ADDED_TOP = {"engine"}
SETTINGS = {"bar_seconds", "window_seconds", "stale_seconds", "min_bars", "max_spread_bps"}
FEED_CONTROL = {"speed", "rate", "paused", "seek", "seek_seq"}
PROFILE = {"date", "events", "profile", "start_ns", "end_ns"}
OBSERVATIONS = {"events", "quotes", "trades", "invalid_quotes", "late_events", "completed_bars"}
PRICE = {"symbol", "mid", "last_trade", "observed_volume", "annual_vol", "quote_age_s"}
ROLLING_TIMING = {"samples", "cpu_median_ms", "gpu_median_ms", "speedup"}
NETWORK = {
    "ready", "status", "ingest", "transport", "wire_protocol", "message_count_scope",
    "packets", "received_messages", "decoded_messages", "messages_per_second",
    "sequence_gaps", "analytics_sequence_gaps", "ring_dropped", "ring_messages",
    "malformed_packets", "out_of_order_packets", "stale_packets", "error", "buckets",
    "gpu_ingest",
}  # fmt: skip
ADDED_NETWORK = {"ingest_mode"}
GPU_INGEST = {
    "bursts", "non_udp_frames", "drain_host_ms", "frame_fetch", "ring_peak", "payload_path",
}  # fmt: skip
ENGINE_OUTPUT = {
    "backend", "milliseconds", "method", "added_bars", "expired_bars", "bars", "universe",
    "eligible_indices", "eligible", "pairs", "annual_vol", "mean_correlation",
    "heatmap_indices", "heatmap",
}  # fmt: skip
PUBLISHED = {"timestamp_ns", "generation", "published_wall_ns", "receive_to_result_ms"}
GPU_EXTRA = {"gpu_name", "finished_wall_ns"}
CPU_EXTRA = {
    "threads", "cpu_thread_trials_ms", "cpu_tuned_bars", "cpu_affinity", "blas",
    "finished_wall_ns",
}  # fmt: skip
COMPARISON = {
    "identifier", "generation", "timestamp_ns", "eligible", "pairs", "cpu_ms", "gpu_ms",
    "speedup", "agreement", "matched_work", "cpu_threads", "cpu_affinity",
    "cpu_thread_trials_ms", "cpu_tuned_bars", "gpu_name", "blas", "method", "added_bars",
    "expired_bars", "cpu_receive_to_result_ms", "published_wall_ns", "validation",
}  # fmt: skip
FULLDAY_RUNNING = {
    "status", "pps", "requested", "controller", "generation", "sender_port",
    "tape_seconds", "base", "packets", "total_packets", "sequence_gaps", "ring_dropped",
    "seconds", "events", "completed_bars", "first_s", "stalls",
}  # fmt: skip
STALLS = {
    "burst_gaps_ms", "worst_burst", "worst_compute", "events", "heartbeat", "gc_max_ms",
    "gc_generation", "analytics_max_ms",
}  # fmt: skip
ENGINE = {
    "sdk", "lanes", "edges", "clocks", "termination", "lane_cpus", "risk", "source", "replay",
}  # fmt: skip

# ---- pinned values (4.3 constants, verbatim) ----------------------------------------------
LEGACY_TOKEN = {"udp": "async", "daqiri-host": "ano", "gpudirect": "gpu", "replay": "replay"}
TRANSPORT = "MITCH (MIT303 11.9) / UDP · NYQ replay mapping"
WIRE_PROTOCOL = "mit303-11.9-nyq-replay-v1"
MESSAGE_COUNT_SCOPE = "MITCH messages including Time; replay rate is source observations/s"
PAYLOAD_PATH = {
    "gpudirect": "NIC -> GPU memory (GPUDirect) -> GPU gather -> GPU MITCH decode",
    "replay": "pcap frames in GPU memory -> GPU gather -> GPU MITCH decode",
}
FRONT_END = {
    "udp": "socket",
    "daqiri-host": "daqiri-host",
    "gpudirect": "gpudirect",
    "replay": "replay",
}

MODES = [
    (mode, backend)
    for mode in ("udp", "daqiri-host", "gpudirect", "replay")
    for backend in ("compare", "gpu", "cpu")
    if not (mode in ("gpudirect", "replay") and backend == "cpu")  # rejected by cli.normalize
]
STATES = ("before_manifest", "after_manifest", "live", "live_comparison")
# Single-backend runs have no comparisons: their last state is "live".
CASES = [
    (mode, backend, state)
    for mode, backend in MODES
    for state in STATES
    if state != "live_comparison" or backend == "compare"
]
PLAN = {
    "lanes": [{"name": "ingest", "operators": ["bars", "source"], "cpus": [0], "wake_mode": 0}],
    "edges": [{"from": "bars.window", "to": "risk.window", "policy": "DROP_OLDEST", "depth": 1}],
    "clocks": [{"operator": "risk", "period_us": 1000}],
    "automatic_partitions": [],
    "sdk": "5.0.0",
}


def manifest(generation=1, port=50001):
    return dict(
        format="nyq-equity-replay-v1",
        wire_protocol=PROTOCOL,
        market_timezone=TIMEZONE,
        symbols=SYMBOLS,
        generation=generation,
        sender_port=port,
        date="2025-04-01",
        start_ns=MIDNIGHT + 34200 * 10**9,
        end_ns=MIDNIGHT + 57600 * 10**9,
        events=5000,
        profile=[[34200, 100], [34260, 80]],
    )


def make_ctx(mode, backend, replay_control=None):
    settings = BarSettings(window_seconds=40, min_bars=2)
    knob = Knob()
    directory = ReplayDirectory(knob)
    ingest, bars, risk = IngestStats(), BarsStats(), RiskStats()
    tap, gate = CaptureTap(), CrossCheckGate()
    results = ResultsStore(
        backend=backend,
        interval_s=0.05,
        settings=settings,
        directory=directory,
        tap=tap,
        gate=gate if backend == "compare" else None,
    )
    feed = FeedControl(directory)
    monitor = StallMonitor(ingest)
    fullday = FullDayController(
        replay_control=replay_control,
        feed=feed,
        ingest=ingest,
        bars=bars,
        results=results,
        stalls=monitor,
        backend=backend,
    )
    for listener in (results.reset_generation, tap.on_generation, fullday.on_generation):
        directory.add_listener(listener)
    return SimpleNamespace(
        args=SimpleNamespace(ingest=mode, analytics_backend=backend, interval=0.05),
        settings=settings, directory_knob=knob, directory=directory, feed=feed,
        ingest_stats=ingest, bars_stats=bars, risk_stats=risk, tap=tap, gate=gate,
        results=results, stall_monitor=monitor, fullday=fullday, status=RunStatus(),
        plan_summary=PLAN, stop_reason=None,
    )  # fmt: skip


def open_source(ctx):
    """What a source's start() and first bursts leave in IngestStats."""
    s = ctx.ingest_stats
    s.reset(front_end=FRONT_END[ctx.args.ingest])
    s.attach(SourceProbe(lambda: True, lambda: 0, lambda: 17, lambda: None))
    s.opened = True
    now = time.time_ns()
    with s.lock:
        s.add_bucket(now - 500_000_000, 1200)
        s.ring_msgs = 33
    s.packets, s.received_messages, s.messages, s.last_packet_wall_ns = 40, 1200, 1100, now
    s.stale_packets, s.malformed_packets, s.decode_malformed = 2, 1, 1
    s.lane_cpus = [0, 1]
    ctx.status.graph_status = "running"


def bars_window(ctx, bars=8):
    d = ctx.directory.current
    rng = np.random.default_rng(d.generation)
    parts = tuple(rng.normal(0.0, 1e-3, d.universe) for _ in range(bars))
    times = MIDNIGHT + (34200 + 5 * np.arange(1, bars + 1, dtype=np.int64)) * 10**9
    return BarWindow(parts, times, d.generation, bars, time.time_ns() - 2_000_000)


def publish_bars(ctx, w):
    """The bars lane's per-batch counters and its symbol snapshot."""
    d, n = ctx.directory.current, ctx.directory.current.universe
    b = ctx.bars_stats
    b.reset(d.generation, d.symbols, device=ctx.args.ingest in ("gpudirect", "replay"))
    last = int(w.times[-1])
    b.counters = BarCounters(d.generation, 900, 800, 90, 12, 0, w.completed_bars, last, 1, 2)
    mid = np.linspace(10.0, 50.0, n)
    mid[3] = np.nan
    snap = SymbolSnapshot(
        d.generation, d.symbols, mid, mid + 0.01, np.full(n, 100.0),
        np.full(n, last - 1_500_000_000, dtype=np.int64), last, time.monotonic_ns(),
    )  # fmt: skip
    for a in (snap.mid, snap.trade, snap.volume, snap.quote_time):
        a.flags.writeable = False
    b.symbol_snapshot = snap


def risk_result(ctx, w, backend):
    result = CovarianceEngine("cpu", bar_seconds=5).update(w.rows, w.times)
    result["finished_wall_ns"] = time.time_ns()
    if backend == "cpu":
        result.update(
            threads=2,
            cpu_thread_trials_ms={"1": 1.0},
            cpu_tuned_bars=8,
            cpu_affinity=[0, 1],
            blas=[{"name": "openblas", "version": "0.3"}],
        )
    else:
        result["gpu_name"] = "Test GPU"
    return result


def go_live(ctx):
    w = bars_window(ctx)
    publish_bars(ctx, w)
    backend = ctx.args.analytics_backend
    update = RiskUpdate(
        w, risk_result(ctx, w, backend), w.rows, "cpu" if backend == "cpu" else "gpu",
        validated=backend == "compare",
    )  # fmt: skip
    ctx.results.on_risk(update)
    return update


def compare(ctx, update):
    cpu = risk_result(ctx, update.window, "cpu")
    gpu = dict(update.result, method="incremental")
    cmp = RiskComparison(RiskUpdate(update.window, gpu, update.rows_host), cpu, True, True, True)
    ctx.results.on_comparison(cmp)


def reach(ctx, state):
    if state == "before_manifest":
        return
    ctx.directory.set_manifest(manifest())
    open_source(ctx)
    if state == "after_manifest":
        return
    update = go_live(ctx)
    if state == "live_comparison":
        compare(ctx, update)


def expected_top(ctx, state):
    keys = TOP | ADDED_TOP
    if state != "before_manifest":
        keys |= {"observations"}
    if state in ("live", "live_comparison"):
        keys |= {"prices"}
    if state == "live_comparison":
        keys |= {"rolling_timing"}
    return keys


def roundtrip(snapshot):
    """What the browser and the tools receive."""
    return json.loads(json.dumps(snapshot))


@pytest.mark.parametrize(("mode", "backend", "state"), CASES)
def test_snapshot_key_sets(mode, backend, state):
    ctx = make_ctx(mode, backend)
    reach(ctx, state)
    s = roundtrip(DashboardModel(ctx).snapshot())
    assert set(s) == expected_top(ctx, state)
    assert set(s["settings"]) == SETTINGS and set(s["feed_control"]) == FEED_CONTROL
    assert set(s["engine"]) == ENGINE
    assert set(s["engine"]["lane_cpus"]) == {"ingest", "bars", "risk"}
    assert set(s["engine"]["risk"]) == {"windows_superseded", "gate_waits", "late_activations"}
    assert set(s["engine"]["source"]) == {"retries", "held"}
    assert s["analytics_backend"] == backend and s["minimum_update_interval_seconds"] == 0.05
    assert s["fullday"] is None and s["fullday_available"] is False
    if state == "before_manifest":
        assert s["network"] == {"ready": False, "status": "starting", "ingest": LEGACY_TOKEN[mode]}
        assert (s["generation"], s["profile"], s["universe"]) == (0, {}, [])
        assert s["live"] is None and s["comparisons"] == 0 and s["history"] == []
        assert s["graph_status"] == "starting" and s["error"] is None
        return
    n = s["network"]
    assert set(n) == NETWORK | ADDED_NETWORK
    assert n["ingest_mode"] == mode and n["ready"] is True and n["status"] == "receiving"
    if mode in ("gpudirect", "replay"):
        assert set(n["gpu_ingest"]) == GPU_INGEST
        assert n["gpu_ingest"]["payload_path"] == PAYLOAD_PATH[mode]
        assert n["gpu_ingest"]["ring_peak"] == 17
    else:
        assert n["gpu_ingest"] is None
    assert s["generation"] == 1 and s["universe"] == SYMBOLS
    assert set(s["profile"]) == PROFILE and s["profile"]["profile"] == [[34200, 100], [34260, 80]]
    assert set(s["observations"]) == OBSERVATIONS
    if state == "after_manifest":
        assert s["observations"] == dict.fromkeys(OBSERVATIONS, 0)  # a fresh generation
        assert s["live"] is None
        return
    extra = CPU_EXTRA if backend == "cpu" else GPU_EXTRA
    assert set(s["live"]) == ENGINE_OUTPUT | extra | PUBLISHED
    assert s["observations"]["events"] == 900 and s["observations"]["completed_bars"] == 8
    assert 0 < len(s["prices"]) <= 30 and all(set(p) == PRICE for p in s["prices"])
    assert len(s["history"]) == 1 and len(s["history"][0]) == 3
    if state == "live":
        assert s["comparison"] is None and "rolling_timing" not in s
        return
    assert set(s["comparison"]) == COMPARISON and s["comparisons"] == 1
    assert set(s["rolling_timing"]) == ROLLING_TIMING and s["comparison_failure"] is None


@pytest.mark.parametrize("mode", ["udp", "daqiri-host", "gpudirect", "replay"])
def test_pinned_network_values(mode):
    ctx = make_ctx(mode, "gpu")
    model = DashboardModel(ctx)
    assert model.network() == {"ready": False, "status": "starting", "ingest": LEGACY_TOKEN[mode]}
    open_source(ctx)
    n = model.network()
    assert n["ingest"] == LEGACY_TOKEN[mode]
    assert n["transport"] == TRANSPORT
    assert n["wire_protocol"] == WIRE_PROTOCOL == PROTOCOL
    assert n["message_count_scope"] == MESSAGE_COUNT_SCOPE


def test_network_counters_follow_the_4_3_definitions():
    ctx = make_ctx("udp", "compare")
    open_source(ctx)
    ctx.ingest_stats.dropped, ctx.ingest_stats.ring_dropped = 5, 6
    ctx.bars_stats.sequence_gaps, ctx.bars_stats.stale_packets = 3, 4
    n = DashboardModel(ctx).network()
    assert n["decoded_messages"] == 1100 and n["received_messages"] == 1200
    assert (n["sequence_gaps"], n["analytics_sequence_gaps"], n["ring_dropped"]) == (5, 3, 6)
    assert n["stale_packets"] == 2 + 4 and n["malformed_packets"] == 1 + 1
    assert n["ring_messages"] == 33 and n["messages_per_second"] == 1200
    assert n["buckets"] and all(len(b) == 2 for b in n["buckets"])
    ctx.ingest_stats.last_packet_wall_ns = time.time_ns() - 5 * 10**9
    assert DashboardModel(ctx).network()["status"] == "idle"
    ctx.ingest_stats.last_packet_wall_ns = 0
    assert DashboardModel(ctx).network()["status"] == "waiting"


def test_prices_rank_the_most_volatile_eligible_instruments():
    ctx = make_ctx("replay", "compare")
    reach(ctx, "live")
    s = DashboardModel(ctx).snapshot()
    live, snap = s["live"], ctx.bars_stats.symbol_snapshot
    vols = [p["annual_vol"] for p in s["prices"]]
    assert vols == sorted(vols, reverse=True) and len(vols) == min(30, live["eligible"])
    top = int(np.argmax(live["annual_vol"]))
    i = live["eligible_indices"][top]
    p = s["prices"][0]
    assert p["symbol"] == SYMBOLS[i] and p["annual_vol"] == live["annual_vol"][top]
    assert p["mid"] == (None if i == 3 else float(snap.mid[i]))  # NaN marks show as null
    assert p["quote_age_s"] == 1.5 and type(p["quote_age_s"]) is float


def test_stale_bars_and_results_never_mix_generations():
    ctx = make_ctx("replay", "compare")
    reach(ctx, "live")
    ctx.directory.set_manifest(manifest(2, 50002))
    s = DashboardModel(ctx).snapshot()
    assert s["generation"] == 2 and s["live"] is None and "prices" not in s
    assert s["observations"] == dict.fromkeys(OBSERVATIONS, 0)  # counters are generation 1's


def test_observations_come_from_one_read_of_the_counters():
    ctx = make_ctx("replay", "compare")
    reach(ctx, "live")
    bars = ctx.bars_stats
    reads = iter([bars.counters])

    class Resetting:  # the bars lane resets for a new generation right after the first read
        observations = BarsStats.observations

        def __getattr__(self, name):
            return getattr(bars, name)

        @property
        def counters(self):
            return next(reads, None)

    ctx.bars_stats = Resetting()
    observations = DashboardModel(ctx).snapshot()["observations"]
    assert observations == dict(
        events=900, quotes=800, trades=90, invalid_quotes=12, late_events=0, completed_bars=8
    )


def test_capture_status_and_error_precedence():
    ctx = make_ctx("replay", "compare")
    reach(ctx, "live")
    model = DashboardModel(ctx)
    assert model.snapshot()["capture"] is None
    status = model.start_capture({"seconds": 5})
    assert model.snapshot()["capture"] == status and status["analytics_backend"] == "compare"
    ctx.ingest_stats.error = "drain error"
    ctx.risk_stats.error = "risk error"
    assert model.snapshot()["error"] == "risk error"
    ctx.results.set_error("Analytics failed: x")
    assert model.snapshot()["error"] == "Analytics failed: x"
    ctx.status.error = "risk: RuntimeError: boom"
    assert model.snapshot()["error"] == "risk: RuntimeError: boom"


def test_fullday_keys_while_running(monkeypatch):
    monkeypatch.setattr(fullday_module, "replay_request", lambda *a, **k: {"packets": 10})
    ctx = make_ctx("gpudirect", "compare", replay_control="http://controller.invalid")
    reach(ctx, "live")
    ctx.fullday.start(200_000)
    ctx.directory.set_manifest(manifest(2, 50002))
    s = roundtrip(DashboardModel(ctx).snapshot())
    assert s["fullday_available"] is True
    assert set(s["fullday"]) == FULLDAY_RUNNING and s["fullday"]["base"] is None
    assert STALLS <= set(s["fullday"]["stalls"])


def test_engine_block():
    ctx = make_ctx("replay", "gpu")
    open_source(ctx)
    ctx.ingest_stats.replay = {"thread_cpu_s": 0.5, "eof": False}
    ctx.ingest_stats.retries, ctx.risk_stats.windows_superseded = 3, 9
    ctx.status.termination = "RUNNING"
    e = DashboardModel(ctx).snapshot()["engine"]
    assert e["lanes"] == PLAN["lanes"] and e["edges"] == PLAN["edges"]
    assert e["clocks"] == PLAN["clocks"] and e["termination"] == "RUNNING"
    assert e["lane_cpus"]["ingest"] == [0, 1] and e["source"] == {"retries": 3, "held": 0}
    assert e["risk"]["windows_superseded"] == 9
    assert e["replay"] == ctx.ingest_stats.replay and e["replay"] is not ctx.ingest_stats.replay
    ctx.plan_summary = None  # before compile_plan ran
    e = DashboardModel(ctx).snapshot()["engine"]
    assert (e["lanes"], e["edges"], e["clocks"]) == (None, None, None)


# What dashboard/webui/index.html reads, by the JavaScript name it reads it through.
INDEX_READS = {
    "s": [
        "feed_control", "error", "observations", "profile", "sender_status", "live", "settings",
        "universe", "prices", "comparison", "rolling_timing", "analytics_backend", "network",
        "history", "fullday_available", "fullday", "comparison_failure",
    ],
    "n": [
        "status", "messages_per_second", "sequence_gaps", "ring_dropped", "packets",
        "transport", "ingest", "ring_messages", "buckets",
    ],
    "l": [
        "eligible", "universe", "pairs", "annual_vol", "mean_correlation", "bars",
        "heatmap_indices", "heatmap", "timestamp_ns", "generation", "receive_to_result_ms",
        "milliseconds",
    ],
    "c": [
        "cpu_ms", "gpu_ms", "speedup", "agreement", "matched_work", "gpu_name", "cpu_threads",
        "cpu_receive_to_result_ms", "identifier", "method", "added_bars", "expired_bars",
        "eligible",
    ],
    "r": ["cpu_median_ms", "gpu_median_ms", "speedup", "samples"],
    "p": ["date", "events", "profile", "start_ns", "end_ns"],
    "x": ["symbol", "mid", "last_trade", "annual_vol", "quote_age_s"],
}  # fmt: skip
INDEX_NESTED = {
    "feed_control": ["paused"],
    "observations": ["events", "invalid_quotes", "completed_bars"],
    "sender_status": ["seek_seq", "position"],
    "settings": ["bar_seconds"],
}
FULLDAY_READS = [
    "status", "seconds", "x_realtime", "tape_seconds", "packets", "total_packets", "events",
    "completed_bars", "sequence_gaps", "ring_dropped",
]  # fmt: skip


def test_every_field_index_html_reads_is_present(monkeypatch):
    html = INDEX_HTML.read_text()
    for var, names in INDEX_READS.items():
        for name in names:
            assert f"{var}.{name}" in html, f"index.html no longer reads {var}.{name}"
    ctx = make_ctx("replay", "compare", replay_control="http://controller.invalid")
    reach(ctx, "live_comparison")
    ctx.feed.sender_status(dict(seek_seq=1, generation=1, position=48600.0, sent=1, loops=0))
    s = roundtrip(DashboardModel(ctx).snapshot())
    views = dict(
        s=s, n=s["network"], l=s["live"], c=s["comparison"], r=s["rolling_timing"],
        p=s["profile"], x=s["prices"][0],
    )  # fmt: skip
    for var, names in INDEX_READS.items():
        missing = [name for name in names if name not in views[var]]
        assert not missing, f"{var}: {missing}"
    for key, names in INDEX_NESTED.items():
        assert set(names) <= set(s[key]), key
    monkeypatch.setattr(fullday_module, "replay_request", lambda *a, **k: {"packets": 10})
    ctx.fullday.start(200_000)
    ctx.directory.set_manifest(manifest(2, 50002))
    ctx.bars_stats.counters = BarCounters(2, 1, 1, 0, 0, 0, 1, 0, 1, 2)  # long finished
    f = DashboardModel(ctx).snapshot()["fullday"]
    assert f["status"] == "done" and set(FULLDAY_READS) <= set(f)
    assert all(f"f.{name}" in html for name in FULLDAY_READS)


def test_snapshot_imports_no_cupy(monkeypatch):
    ctx = make_ctx("replay", "compare")
    reach(ctx, "live_comparison")
    monkeypatch.setitem(sys.modules, "cupy", None)  # any `import cupy` now raises
    s = DashboardModel(ctx).snapshot()
    assert s["live"] is not None and "prices" in s
