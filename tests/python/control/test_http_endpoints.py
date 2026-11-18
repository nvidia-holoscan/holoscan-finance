# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every dashboard endpoint through FastAPI's test client: bodies, 422 paths, added routes.

The context is the ``EquityContext`` attribute set over the real control objects.
``serve()`` (``sdk``) runs a tiny graph behind uvicorn: exit codes and stop reasons.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from apphelpers import equity_args

from holoscan_finance.app.context import EquityContext
from holoscan_finance.app.runner import cpu_sets
from holoscan_finance.control import fullday as fullday_module
from holoscan_finance.control.directory import ReplayDirectory
from holoscan_finance.control.feed import FeedControl
from holoscan_finance.control.fullday import FullDayController
from holoscan_finance.control.gate import CrossCheckGate
from holoscan_finance.control.results import ResultsStore
from holoscan_finance.control.status import RunStatus
from holoscan_finance.dashboard.model import DashboardModel
from holoscan_finance.equity.mapping import PROTOCOL, TIMEZONE, day_start_ns
from holoscan_finance.ref import Knob
from holoscan_finance.settings import BarSettings
from holoscan_finance.telemetry.capture import CaptureTap, LatencyCapture
from holoscan_finance.telemetry.stalls import StallMonitor
from holoscan_finance.telemetry.stats import BarsStats, IngestStats, RiskStats, SourceProbe

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from holoscan_finance.dashboard.server import make_app  # noqa: E402

REPO = Path(__file__).parents[3]
MIDNIGHT = day_start_ns("2026-06-01")
PLAN = {"lanes": [], "edges": [], "clocks": [], "automatic_partitions": [], "sdk": "5.0.0"}
FINISH_COMPARE = {
    "start_ns", "end_ns", "finished_ns", "seconds", "generation", "valid", "observations",
    "packets", "wire_messages", "received_observations_per_second", "sequence_gaps",
    "reordered_packets", "overflow", "generation_changed", "all_gpu_results_compared",
    "percentile_method", "gpu_receive_to_result_ms", "gpu_analytics_ms",
    "cpu_receive_to_result_ms", "cpu_analytics_ms", "gpu_calculations", "cpu_comparisons",
    "matched_analytics_speedup",
}  # fmt: skip
FINISH_ADDED = {"ingest", "settings", "wire_protocol", "minimum_update_interval_seconds"}


def manifest(generation=1, port=50001):
    return dict(
        format="equity-replay-v1",
        wire_protocol=PROTOCOL,
        market_timezone=TIMEZONE,
        symbols=["A", "B", "C"],
        generation=generation,
        sender_port=port,
        date="2026-06-01",
        start_ns=MIDNIGHT + 34200 * 10**9,
        end_ns=MIDNIGHT + 57600 * 10**9,
        events=10,
        profile=[[34200, 10]],
    )


def make_ctx(mode="replay", backend="compare", replay_control=None):
    settings = BarSettings(window_seconds=40, min_bars=2)
    directory = ReplayDirectory(Knob())
    ingest, bars = IngestStats(), BarsStats()
    tap = CaptureTap()
    results = ResultsStore(
        backend=backend,
        interval_s=0.05,
        settings=settings,
        directory=directory,
        tap=tap,
        gate=CrossCheckGate(),
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
    directory.add_listener(results.reset_generation)
    directory.add_listener(fullday.on_generation)
    return SimpleNamespace(
        args=SimpleNamespace(ingest=mode, analytics_backend=backend, interval=0.05),
        settings=settings, directory=directory, feed=feed, ingest_stats=ingest,
        bars_stats=bars, risk_stats=RiskStats(), tap=tap, results=results,
        stall_monitor=monitor, fullday=fullday,
        status=RunStatus(receiving=ingest.ready, input_ended=lambda: ingest.eof),
        plan_summary=PLAN,
    )  # fmt: skip


def client_for(ctx):
    return TestClient(make_app(DashboardModel(ctx), ctx))


def test_index_serves_the_dashboard_page():
    r = client_for(make_ctx()).get("/")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    assert 'id="networkdetail"' in r.text and "n.ingest==='replay'" in r.text


def test_snapshot_before_the_source_opens():
    s = client_for(make_ctx("udp")).get("/snapshot").json()
    assert s["network"] == {"ready": False, "status": "starting", "ingest": "udp"}
    assert s["graph_status"] == "starting" and s["generation"] == 0


def test_plan_and_healthz():
    ctx = make_ctx()
    client = client_for(ctx)
    assert client.get("/plan").json() == PLAN
    r = client.get("/healthz")
    assert r.status_code == 503
    assert r.json() == {
        "ready": False, "graph_status": "starting", "termination": None, "input_ended": False
    }  # fmt: skip
    ctx.status.graph_status, ctx.status.termination = "stopped", "FAILED"
    r = client.get("/healthz")
    assert r.status_code == 503
    assert r.json() == {
        "ready": False, "graph_status": "stopped", "termination": "FAILED", "input_ended": False
    }  # fmt: skip


@pytest.mark.parametrize("mode", ["udp", "replay"])
def test_healthz_answers_200_only_while_the_ingest_front_end_receives(mode):
    """The graph runs some time before the source has bound its socket (or started its
    drain): a sender started on a 200 then lost its first packets. /healthz is 503 until
    the front end receives, as /snapshot's network.ready says, and again after an error."""
    ctx = make_ctx(mode)
    client = client_for(ctx)
    ctx.status.graph_status, ctx.status.termination = "running", "RUNNING"
    for _ in range(2):  # the graph runs, the source has not opened (then: its drain is down)
        r = client.get("/healthz")
        assert r.status_code == 503 and r.json()["ready"] is False
        assert client.get("/snapshot").json()["network"]["ready"] is False
        ctx.ingest_stats.opened = True
    alive = [True]
    ctx.ingest_stats.attach(SourceProbe(lambda: alive[0], lambda: 0, lambda: 0, lambda: None))
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {
        "ready": True, "graph_status": "running", "termination": "RUNNING", "input_ended": False
    }  # fmt: skip
    assert client.get("/snapshot").json()["network"]["ready"] is True
    ctx.ingest_stats.error = "UDP RX failed: boom"
    assert client.get("/healthz").status_code == 503
    ctx.ingest_stats.error = None
    alive[0] = False
    assert client.get("/healthz").status_code == 503


def test_healthz_tells_a_finished_replay_from_one_not_started():
    """A finite replay's drain ends after its last pass (``--replay-loops 1`` by default)
    while the graph and the dashboard keep running: /healthz is 503 then, as nothing is
    received any more, and its body says the input ended, which a replay that has not
    started does not say (a short capture can end before a poller ever sees the 200)."""
    ctx = make_ctx("replay")
    client = client_for(ctx)
    ctx.status.graph_status, ctx.status.termination = "running", "RUNNING"
    assert client.get("/healthz").json()["input_ended"] is False  # not started
    alive = [True]
    ctx.ingest_stats.opened = True
    ctx.ingest_stats.attach(SourceProbe(lambda: alive[0], lambda: 0, lambda: 0, lambda: None))
    assert client.get("/healthz").status_code == 200
    alive[0], ctx.ingest_stats.eof = False, True  # the last pass played, everything emitted
    r = client.get("/healthz")
    assert r.status_code == 503
    assert r.json() == {
        "ready": False, "graph_status": "running", "termination": "RUNNING", "input_ended": True
    }  # fmt: skip
    assert client.get("/").status_code == 200 and client.get("/snapshot").status_code == 200


def test_the_apps_run_status_reads_the_end_of_input_from_its_ingest_stats():
    args = equity_args("--ingest", "udp")
    ctx = EquityContext.from_args(args, cpu_sets(args))
    assert ctx.status.input_ended() is False
    ctx.ingest_stats.eof = True
    assert ctx.status.input_ended() is True


def test_a_failed_session_shows_in_healthz_and_snapshot():
    def diagnostic(severity, token, message=None, where=None):
        detail = SimpleNamespace(error=SimpleNamespace(message=message), operator_path=where)
        return SimpleNamespace(severity=SimpleNamespace(name=severity), token=token, detail=detail)

    session = SimpleNamespace(
        termination=SimpleNamespace(name="FAILED"),
        is_running=False,
        diagnostics=[
            diagnostic("WARNING", "MAX_START_DELAY_MISS"),
            diagnostic(
                "ERROR",
                "RUNTIME_OPERATOR_CALLBACK_FAILED",
                'Traceback (most recent call last):\n  File "x.py", line 1\nRuntimeError: boom\n',
                "risk",
            ),
        ],
    )
    ctx = make_ctx()
    ctx.status.graph_status = "running"
    ctx.status.from_session(session)
    client = client_for(ctx)
    r = client.get("/healthz")
    assert r.status_code == 503
    assert r.json() == {
        "ready": False, "graph_status": "stopped", "termination": "FAILED", "input_ended": False
    }  # fmt: skip
    assert client.get("/snapshot").json()["error"] == "risk: RuntimeError: boom"


def test_manifest_idempotence_and_422s():
    client = client_for(make_ctx())
    assert client.post("/manifest", json=manifest(5)).json() == {"generation": 5}
    assert client.post("/manifest", json=manifest(5, 50002)).json() == {"generation": 5}
    assert client.post("/manifest", json=manifest(4, 50003)).json() == {"generation": 5}
    r = client.post("/manifest", json=manifest(6, 50001))
    assert r.status_code == 422 and "port must be new" in r.json()["detail"]
    r = client.post("/manifest", json={**manifest(7, 50004), "symbols": []})
    assert r.status_code == 422 and r.json()["detail"] == "invalid replay directory or generation"
    assert client.post("/manifest", json=[1, 2]).status_code == 422  # not an object
    assert client.get("/snapshot").json()["generation"] == 5


def test_feedrate_rules():
    client = client_for(make_ctx())
    assert client.get("/feedrate").json() == dict(
        speed=10.0, rate=50000.0, paused=False, seek=48600.0, seek_seq=1
    )
    assert client.post("/feedrate", json={"seek": 40000}).json()["seek_seq"] == 2
    assert client.post("/feedrate", json={"speed": 25, "rate": 5000}).json()["speed"] == 25
    for body, detail in (
        ({"speed": 101}, "invalid replay control"),
        ({"loops": 1}, "unsupported replay control"),
        ({"paused": "yes"}, "invalid replay control"),
        ({"speed": "10"}, "invalid replay control"),  # the sender reads it back as a number
    ):
        r = client.post("/feedrate", json=body)
        assert (r.status_code, r.json()["detail"]) == (422, detail)
    assert client.post("/feedrate", json={"speed": None}).status_code == 422  # TypeError
    assert client.get("/feedrate").json()["speed"] == 25


def test_feedstatus_and_resume():
    client = client_for(make_ctx())
    client.post("/manifest", json=manifest(1))
    report = dict(seek_seq=1, generation=1, position=48700.0, sent=5, loops=0)
    assert client.post("/feedstatus", json=report).json() == {"status": "received"}
    assert client.post("/feedstatus", json=dict(report, generation=9)).json() == {
        "status": "received"
    }
    assert client.get("/snapshot").json()["sender_status"] == report
    client.post("/feedrate", json={"paused": True})
    c = client.post("/replay/resume", json={}).json()
    assert (c["paused"], c["seek"], c["seek_seq"]) == (False, 48600.0, 2)


def test_capture_lifecycle_and_finish_contract():
    ctx = make_ctx("gpudirect")
    client = client_for(ctx)
    assert client.get("/capture").json() == {}
    r = client.post("/capture/finish")
    assert (r.status_code, r.json()["detail"]) == (422, "no capture has been started")
    client.post("/manifest", json=manifest(1))
    r = client.post("/capture", json={"seconds": 1})
    assert (r.status_code, r.json()["detail"]) == (
        422,
        "wait for a full return window before capturing",
    )
    ctx.results.live = {"bars": ctx.settings.window_bars - 1}
    assert client.post("/capture", json={"seconds": 1}).status_code == 422
    ctx.results.live = {"bars": ctx.settings.window_bars}
    for body in ({"seconds": 0}, {"seconds": "x"}):
        assert client.post("/capture", json=body).status_code == 422
    status = client.post("/capture", json={"seconds": 1}).json()
    assert status["generation"] == 1 and status["analytics_backend"] == "compare"
    assert client.get("/capture").json() == status
    r = client.post("/capture", json={"seconds": 1})
    assert (r.status_code, r.json()["detail"]) == (
        422,
        "finish the previous capture before starting another",
    )
    r = client.post("/capture/finish")
    assert (r.status_code, r.json()["detail"]) == (422, "capture is still running")
    time.sleep(max(0.0, (status["end_ns"] - time.time_ns()) / 1e9) + 0.05)
    result = client.post("/capture/finish").json()
    assert set(result) == FINISH_COMPARE | FINISH_ADDED
    assert result["ingest"] == "gpudirect" and result["wire_protocol"] == PROTOCOL
    assert result["settings"] == ctx.settings.as_params()
    assert result["minimum_update_interval_seconds"] == 0.05
    assert client.get("/capture").json()["closed"] is True


@pytest.mark.parametrize("mode", ["udp", "daqiri-host", "gpudirect", "replay"])
def test_finish_capture_reports_the_ingest_mode(mode):
    ctx = make_ctx(mode, "gpu")
    ctx.tap.capture = LatencyCapture(1, 1, start_ns=time.time_ns() - 2 * 10**9, backend="gpu")
    result = client_for(ctx).post("/capture/finish").json()
    assert result["ingest"] == mode  # compare_runs.py reads cpu["ingest"]
    assert FINISH_ADDED <= set(result) and result["analytics_backend"] == "gpu"


def test_fullday_endpoint(monkeypatch):
    client = client_for(make_ctx())
    r = client.post("/replay/fullday", json={})
    assert r.status_code == 422 and "--replay-control" in r.json()["detail"]
    calls = []

    def fake_request(url, payload, timeout=5):
        try:
            asyncio.get_running_loop()
            on_event_loop = True
        except RuntimeError:
            on_event_loop = False
        calls.append((on_event_loop, payload))
        return {"pps": payload["pps"], "packets": 100}

    monkeypatch.setattr(fullday_module, "replay_request", fake_request)
    client = client_for(make_ctx(replay_control="http://controller.invalid"))
    assert client.post("/replay/fullday", json={"pps": "fast"}).status_code == 422
    f = client.post("/replay/fullday", json={}).json()
    assert f["status"] == "starting" and f["pps"] == 200_000
    assert calls == [(False, {"pps": 200_000})]  # a worker thread, not the event loop
    assert client.get("/feedrate").json()["paused"] is True
    r = client.post("/replay/fullday", json={"pps": 1000})
    assert r.status_code == 422 and "already running" in r.json()["detail"]
    s = client.get("/snapshot").json()
    assert s["fullday_available"] is True and s["fullday"]["base"] is None


BIG = int("9" * 400)  # valid JSON; float() raises OverflowError


def raw(client, path: str, body: bytes):
    """POST a JSON body as bytes: Python's parser also reads 1e999 (inf) and NaN."""
    return client.post(path, content=body, headers={"content-type": "application/json"})


def test_numbers_out_of_range_are_422_and_change_nothing():
    """A huge integer, or a number only JSON extensions parse (1e999, NaN), is a 422."""
    ctx = make_ctx()
    client = client_for(ctx)
    client.post("/manifest", json=manifest(1))
    ctx.results.live = {"bars": ctx.settings.window_bars}
    before = client.get("/feedrate").json()
    for key in ("speed", "rate", "seek"):
        assert client.post("/feedrate", json={key: BIG}).status_code == 422
        r = raw(client, "/feedrate", f'{{"{key}": 1e999}}'.encode())
        assert r.status_code == 422, key
    assert client.get("/feedrate").json() == before
    for body in ({"seconds": BIG}, {"seconds": "5"}, {"seconds": True}):
        assert client.post("/capture", json=body).status_code == 422, body
    assert raw(client, "/capture", b'{"seconds": NaN}').status_code == 422
    assert client.get("/capture").json() == {}


def test_fullday_pps_must_be_an_integer(monkeypatch):
    calls = []
    monkeypatch.setattr(fullday_module, "replay_request", lambda u, p, timeout=5: calls.append(p))
    client = client_for(make_ctx(replay_control="http://controller.invalid"))
    for body in ({"pps": "200000"}, {"pps": 2.7}, {"pps": True}, {"pps": BIG}):
        assert client.post("/replay/fullday", json=body).status_code == 422, body
    assert raw(client, "/replay/fullday", b'{"pps": 1e999}').status_code == 422
    assert calls == [] and client.get("/feedrate").json()["paused"] is False


def test_a_non_finite_sender_report_is_dropped_and_the_snapshot_stays_up():
    client = client_for(make_ctx())
    client.post("/manifest", json=manifest(1))
    for value in (b"1e999", b"NaN", b"-Infinity", b"true"):
        body = b'{"seek_seq": 1, "generation": 1, "sent": 1, "loops": 0, "position": %s}' % value
        assert raw(client, "/feedstatus", body).status_code == 200
        r = client.get("/snapshot")
        assert r.status_code == 200, value
        assert r.json()["sender_status"] == {"seek_seq": 1, "generation": 1, "sent": 1, "loops": 0}


def test_a_manifest_the_snapshot_could_not_serve_is_rejected():
    """A lone surrogate symbol (a JSON escape) or a non-finite profile entry is a 422, and
    the snapshot keeps serving the current generation."""
    client = client_for(make_ctx())
    client.post("/manifest", json=manifest(1))
    base = json.dumps(manifest(2, 50002))
    for bad in (
        base.replace('"A"', '"\\ud800"'),
        base.replace("[[34200, 10]]", "[[1e999, 10]]"),
        base.replace("[[34200, 10]]", "[[NaN, 10]]"),
        base.replace("[[34200, 10]]", "[[34200, true]]"),
        base.replace("[[34200, 10]]", "[[34200]]"),
    ):
        assert raw(client, "/manifest", bad.encode()).status_code == 422, bad
    r = client.get("/snapshot")
    assert r.status_code == 200 and r.json()["generation"] == 1


def test_hold_gil_exists_only_with_fault_injection(monkeypatch):
    monkeypatch.delenv("HOLOSCAN_FINANCE_FAULTS", raising=False)
    assert client_for(make_ctx()).post("/debug/hold_gil", json={}).status_code == 404
    monkeypatch.setenv("HOLOSCAN_FINANCE_FAULTS", "1")
    client = client_for(make_ctx())
    started = time.perf_counter()
    assert client.post("/debug/hold_gil", json={"seconds": 0.05}).json() == {"held_s": 0.05}
    assert time.perf_counter() - started >= 0.05
    # usleep() takes an unsigned count: a negative request must not wrap to an hour-long hold.
    assert client.post("/debug/hold_gil", json={"seconds": -1}).json() == {"held_s": 0.0}


# ---- serve(): the run loop behind uvicorn (sdk) ---------------------------------------------
# serve() installs signal handlers and runs uvicorn on the main thread, so each case runs in a
# fresh interpreter on a tiny graph (a 1 kHz counter into a sink) and prints a JSON line.
SERVE_CASES = textwrap.dedent(
    """
    import json, os, signal, socket, sys, threading, time, urllib.request
    from types import SimpleNamespace

    import holoscan as hs

    from holoscan_finance.control.status import RunStatus
    from holoscan_finance.dashboard import server as server_module
    from holoscan_finance.dashboard.server import serve
    from holoscan_finance.ref import Ref

    COUNT = 20


    class Counter(hs.Operator):  # emits 1..count at 1 kHz, then idles; raises at fail_at
        out = hs.Output("out", hs.pyobject)
        count = hs.Param(COUNT)
        fail_at = hs.Param(0)

        def contract(self):
            return hs.Contract().trigger(hs.OnClock(period=hs.time.ms(1)))

        def start(self):
            self.n = 0

        def compute(self, ctx):
            if self.n < self.count:
                self.n += 1
                if self.n == self.fail_at:
                    raise RuntimeError("injected failure")
                self.out.emit(self.n)


    class Sink(hs.Operator):  # a stop_ms stop() appends one item at the end (then fails)
        items = hs.Input("items", hs.pyobject)
        seen = hs.Param(Ref())
        stop_ms = hs.Param(0)
        stop_fails = hs.Param(False)

        def contract(self):
            return hs.Contract().trigger(hs.OnEach(self.items))

        def compute(self, ctx):
            self.seen.obj.append(self.items.receive_data())

        def stop(self):
            if self.stop_ms:
                time.sleep(self.stop_ms / 1000)
                self.seen.obj.append("final")
                if self.stop_fails:
                    raise RuntimeError("injected stop failure")


    def healthz_once_running(port, out):
        # What a launcher sees mid-run, then SIGTERM as a process manager sends it.
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=1) as r:
                    out.update(json.loads(r.read()))
                if out["termination"] == "RUNNING":
                    break
            except OSError:
                pass  # not listening yet
            time.sleep(0.05)
        os.kill(os.getpid(), signal.SIGTERM)


    def main(case, tmp):
        seen, run_async_calls = [], []
        run_async = hs.run_async

        def counted_run_async(plan):
            run_async_calls.append(1)
            if case == "start_fails":
                raise RuntimeError("injected start failure")
            return run_async(plan)

        hs.run_async = counted_run_async
        server_module.STOP_WARNING_S = 0.2  # the slow stop()s below take 1 s
        g = hs.Graph("serve", default_clock=hs.RealtimeClock)
        counter = g.op(Counter, "counter", fail_at=5 if case == "failed" else 0)
        slow = case.startswith("slow_stop")
        sink = g.op(Sink, "sink", seen=Ref(seen), stop_ms=1000 if slow else 0,
                    stop_fails=case == "slow_stop_fails")
        g.add_flow(counter.out, sink.items, queue_policy=hs.QueuePolicy.BLOCKING, queue_depth=8)
        ctx = SimpleNamespace(status=RunStatus(), stall_monitor=None, plan_summary=None,
                              stop_reason=None, drained=lambda: len(seen) >= COUNT)
        ctx.report_dict = lambda s: dict(termination=s.termination.name, items=len(seen))
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        if case == "port_busy":  # another process already listens on the dashboard port
            busy = socket.create_server(("127.0.0.1", 0))
            port = busy.getsockname()[1]
        healthz = {}
        if case == "signal":
            threading.Thread(target=healthz_once_running, args=(port, healthz), daemon=True).start()
        expect = (lambda: ["fewer windows than expected"]) if case == "expectation" else None
        report = os.path.join(tmp, "report.json")
        rc = serve(hs.compile(g), ctx, host="127.0.0.1", port=port,
                   exit_when_drained=case != "signal", report=report, expect=expect)
        print(json.dumps(dict(
            rc=int(rc), stop_reason=ctx.stop_reason, status=vars(ctx.status), healthz=healthz,
            report=json.load(open(report)) if os.path.exists(report) else None,
            sigterm_restored=signal.getsignal(signal.SIGTERM) is signal.SIG_DFL,
            run_async_calls=len(run_async_calls),
        )))


    if __name__ == "__main__":
        main(sys.argv[1], sys.argv[2])
    """
)


def serve_case(case: str, tmp_path: Path) -> dict:
    script = tmp_path / "serve_cases.py"
    script.write_text(SERVE_CASES)
    path = os.pathsep.join(p for p in (os.environ.get("PYTHONPATH"), str(REPO / "python")) if p)
    done = subprocess.run(
        [sys.executable, str(script), case, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=120,
        env=dict(os.environ, PYTHONPATH=path),
    )
    lines = [line for line in done.stdout.splitlines() if line.startswith("{")]
    assert lines, done.stderr[-3000:]
    return dict(json.loads(lines[-1]), stderr=done.stderr)


@pytest.mark.sdk
@pytest.mark.parametrize(
    ("case", "rc", "stop_reason", "termination"),
    [
        ("drained", 0, "drained", "STOPPED"),
        ("failed", 1, "terminated", "FAILED"),
        ("signal", 0, "signal", "STOPPED"),
        ("expectation", 5, "drained", "STOPPED"),
    ],
)
def test_serve_exit_codes_and_stop_reasons(case, rc, stop_reason, termination, tmp_path):
    r = serve_case(case, tmp_path)
    assert (r["rc"], r["stop_reason"]) == (rc, stop_reason)
    assert r["status"]["graph_status"] == "stopped"
    assert r["status"]["termination"] == r["report"]["termination"] == termination
    assert r["sigterm_restored"]  # serve leaves the process's handlers as it found them
    if case == "failed":
        assert r["status"]["error"] == "counter: RuntimeError: injected failure"
    else:
        assert r["status"]["error"] is None
    if case == "signal":  # a graph without an ingest front end is ready once it runs
        assert r["healthz"] == {
            "ready": True, "graph_status": "running", "termination": "RUNNING",
            "input_ended": False,
        }  # fmt: skip


@pytest.mark.sdk
@pytest.mark.parametrize(
    ("case", "rc", "termination"),
    [("slow_stop", 0, "STOPPED"), ("slow_stop_fails", 1, "FAILED")],
)
def test_serve_waits_for_a_slow_stop(case, rc, termination, tmp_path):
    # A stop() that outlasts STOP_WARNING_S (0.2 s here): the status, report and exit code
    # still come from the stopped graph, with what stop() added and how it ended.
    r = serve_case(case, tmp_path)
    assert "graph still stopping after 0.2 s" in r["stderr"]  # the bound expired first
    assert (r["rc"], r["stop_reason"]) == (rc, "drained")
    assert r["status"]["graph_status"] == "stopped"
    assert r["status"]["termination"] == r["report"]["termination"] == termination
    assert r["report"]["items"] == 21  # the 20 counted, then the one stop() appends
    if case == "slow_stop_fails":
        assert "injected stop failure" in r["status"]["error"]


@pytest.mark.sdk
@pytest.mark.parametrize("case", ["port_busy", "start_fails"])
def test_serve_startup_failures_exit_run_failed(case, tmp_path):
    # RUN_FAILED, never uvicorn's own status 3 (our PLAN_REJECTED); a busy port starts nothing.
    r = serve_case(case, tmp_path)
    assert (r["rc"], r["stop_reason"], r["report"]) == (1, None, None)
    assert r["run_async_calls"] == (0 if case == "port_busy" else 1)
    assert r["status"]["graph_status"] == "starting" and r["sigterm_restored"]
