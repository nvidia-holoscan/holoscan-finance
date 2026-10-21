# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The runner: process set-up, compile refusals, the headless loop's stop reasons and exit codes.

``prepare_process`` and ``run_headless`` change the switch interval, the CPU mask, logging
and signal handlers of the calling process, so those cases run in a subprocess on a tiny
stub graph (a 1 kHz counter into a sink) and print a JSON result line.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from apphelpers import child_env, equity_args

from holoscan_finance.app import runner
from holoscan_finance.app.runner import ExitCode

CASES = textwrap.dedent(
    """
    import json, os, signal, sys, threading
    from types import SimpleNamespace

    from holoscan_finance.app import runner


    def graph(case, seen, count):
        import holoscan as hs

        from holoscan_finance.app.placement import Lanes
        from holoscan_finance.ingest.affinity import CpuSets
        from holoscan_finance.ref import Ref

        class Counter(hs.Operator):  # emits 1..count at 1 kHz, then idles; raises at fail_at
            out = hs.Output("out", hs.pyobject)
            count = hs.Param(20)
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

        class Sink(hs.Operator):
            items = hs.Input("items", hs.pyobject)
            seen = hs.Param(Ref())

            def contract(self):
                return hs.Contract().trigger(hs.OnEach(self.items))

            def compute(self, ctx):
                self.seen.obj.append(self.items.receive_data())

        g = hs.Graph("runner", default_clock=hs.RealtimeClock)
        counter = g.op(Counter, "counter", count=count, fail_at=5 if case == "failed" else 0)
        sink = g.op(Sink, "sink", seen=Ref(seen))
        g.add_flow(counter.out, sink.items, queue_policy=hs.QueuePolicy.BLOCKING, queue_depth=8)
        lanes = Lanes(CpuSets.compute()).add("ingest", counter, sink)
        lanes.apply(g)
        return g, lanes


    class Status:  # RunStatus stand-in
        graph_status, termination = "starting", None

        def from_session(self, session):
            self.graph_status, self.termination = "stopped", session.termination.name


    class Monitor:  # StallMonitor stand-in
        def __init__(self):
            self.calls = []

        def start(self):
            self.calls.append("start")

        def stop(self):
            self.calls.append("stop")


    def run(case, tmp):
        seen, count = [], 20
        g, lanes = graph(case, seen, count)
        drained = lambda: len(seen) >= count
        drained.why = lambda: f"{len(seen)} of {count} items"
        monitor = Monitor() if case == "monitor" else None
        ctx = SimpleNamespace(status=Status(), drained=drained, stop_reason=None,
                              stall_monitor=monitor, plan_summary=None)
        ctx.report_dict = lambda s: dict(termination=s.termination.name,
                                         stop_reason=ctx.stop_reason, items=len(seen))
        plan = runner.compile_plan(g, lanes, ctx=ctx)
        kw = dict(until=ctx.drained, report=os.path.join(tmp, "report.json"))
        if case == "expectation":
            kw["expect"] = lambda: ["fewer windows than expected"]
        elif case == "timeout":
            kw.update(until=lambda: False, timeout_s=0.3)
        elif case == "signal":
            kw["until"] = None
            threading.Timer(0.3, os.kill, (os.getpid(), signal.SIGTERM)).start()
        rc = runner.run_headless(plan, ctx, **kw)
        print(json.dumps(dict(
            rc=int(rc), stop_reason=ctx.stop_reason, status=vars(ctx.status), items=len(seen),
            monitor=monitor.calls if monitor else None, lanes=ctx.plan_summary["lanes"],
            sigterm_restored=signal.getsignal(signal.SIGTERM) is signal.SIG_DFL,
        )))


    def prepare(case):
        release = threading.Event()
        early = threading.Thread(target=release.wait, name="early")  # exists before the sweep
        early.start()
        mask = sorted(os.sched_getaffinity(0))
        args = SimpleNamespace(ingest_cpus=None, ingest_busy_poll=False,
                               nic_master_core=mask[-2], nic_rx_core=mask[-1])
        if case != "bench":  # the bench's args have no --ingest
            args.ingest = "daqiri-host" if case.startswith("daqiri") else "udp"
        if case.endswith("busypoll"):  # a busy-polled ingest CPU just below the lcores
            args.ingest_cpus = str(mask[-3] if case.startswith("daqiri") else mask[-1])
            args.ingest_busy_poll = True
        cpus = runner.prepare_process(args)
        result = dict(
            switch=sys.getswitchinterval(), mask=mask, reserved=sorted(cpus.reserved),
            app=sorted(cpus.app), helper=sorted(cpus.helper),
            main=sorted(os.sched_getaffinity(0)),
            early=sorted(os.sched_getaffinity(early.native_id)),
        )
        release.set()
        early.join()
        print(json.dumps(result))


    if __name__ == "__main__":
        case, tmp = sys.argv[1], sys.argv[2]
        if case in ("udp", "daqiri", "bench", "busypoll", "daqiri-busypoll"):
            prepare(case)
        else:
            run(case, tmp)
    """
)


def run_case(case: str, tmp_path: Path) -> tuple[dict, str]:
    """Run one case in a fresh interpreter; its result and its stderr."""
    script = tmp_path / "cases.py"
    script.write_text(CASES)
    done = subprocess.run(
        [sys.executable, str(script), case, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=120,
        env=child_env(),
    )
    lines = [line for line in done.stdout.splitlines() if line.startswith("{")]
    assert done.returncode == 0 and lines, done.stderr[-3000:]
    return json.loads(lines[-1]), done.stderr


def mask_of_at_least(n: int) -> list[int]:
    mask = sorted(os.sched_getaffinity(0))
    if len(mask) < n:
        pytest.skip(f"needs {n} CPUs")
    return mask


@pytest.mark.parametrize("case", ["udp", "bench"])
def test_prepare_process_without_reserved_cpus_keeps_the_mask(case: str, tmp_path: Path) -> None:
    mask_of_at_least(3)
    out, _ = run_case(case, tmp_path)
    assert out["switch"] == pytest.approx(0.001)
    assert out["reserved"] == [] and out["app"] == out["mask"]
    assert out["main"] == out["mask"] and out["early"] == out["mask"]


def test_prepare_process_moves_the_process_off_the_daqiri_lcores(tmp_path: Path) -> None:
    mask = mask_of_at_least(3)
    out, _ = run_case("daqiri", tmp_path)
    lcores = mask[-2:]
    assert out["switch"] == pytest.approx(0.001)
    assert out["reserved"] == lcores
    assert out["main"] == out["app"] == mask[:-2]  # what later threads inherit
    assert out["early"] == mask[:-2]  # the 4.3 sweep moved a thread started before


@pytest.mark.parametrize("case", ["busypoll", "daqiri-busypoll"])
def test_prepare_process_keeps_the_process_off_a_busy_polled_ingest_cpu(
    case: str, tmp_path: Path
) -> None:
    """Under ``--ingest-cpus N --ingest-busy-poll`` only the ingest lane may run on N: the main
    thread (and so uvicorn, the stall heartbeat and the runtime's threads) is masked to
    ``helper``, and threads started before are swept off N as off the DAQIRI lcores."""
    mask = mask_of_at_least(4)
    out, _ = run_case(case, tmp_path)
    expected = mask[:-3] if case == "daqiri-busypoll" else mask[:-1]
    assert out["app"] == (mask[:-2] if case == "daqiri-busypoll" else mask)
    assert out["helper"] == expected
    assert out["main"] == expected  # what later threads inherit
    assert out["early"] == expected


def test_cpu_sets_refuse_an_ingest_cpu_the_app_may_not_use(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An ingest lane on a DAQIRI lcore is a usage error, not a traceback."""
    mask = mask_of_at_least(3)
    lcores = ("--nic-master-core", str(mask[-2]), "--nic-rx-core", str(mask[-1]))
    args = equity_args("--ingest", "daqiri-host", *lcores, "--ingest-cpus", str(mask[-1]))
    with pytest.raises(SystemExit) as exc:
        runner.cpu_sets(args)
    assert exc.value.code == ExitCode.USAGE
    assert f"error: ingest CPUs [{mask[-1]}] are not all in" in capsys.readouterr().err
    args.ingest_cpus = str(mask[0])
    cpus = runner.cpu_sets(args)
    assert cpus.reserved == set(mask[-2:]) and cpus.ingest == {mask[0]}
    assert sorted(os.sched_getaffinity(0)) == mask  # the calling process is left alone


@pytest.mark.sdk
def test_run_headless_stops_once_drained(tmp_path: Path) -> None:
    out, _ = run_case("drained", tmp_path)
    assert out["rc"] == ExitCode.OK and out["stop_reason"] == "drained"
    assert out["status"] == {"graph_status": "stopped", "termination": "STOPPED"}
    assert out["items"] >= 20 and out["monitor"] is None
    assert out["sigterm_restored"]
    report = json.loads((tmp_path / "report.json").read_text())
    assert report == {"termination": "STOPPED", "stop_reason": "drained", "items": out["items"]}
    assert out["lanes"][0]["name"] == "ingest"  # compile_plan stored the plan summary


@pytest.mark.sdk
def test_run_headless_reports_a_failed_run(tmp_path: Path) -> None:
    out, err = run_case("failed", tmp_path)
    assert out["rc"] == ExitCode.RUN_FAILED
    assert out["stop_reason"] == "terminated"  # the session ended by itself
    assert out["status"]["termination"] == "FAILED"
    assert "RUNTIME_OPERATOR_CALLBACK_FAILED [counter]" in err and "injected failure" in err
    assert json.loads((tmp_path / "report.json").read_text())["termination"] == "FAILED"


@pytest.mark.sdk
def test_run_headless_unmet_expectations(tmp_path: Path) -> None:
    out, err = run_case("expectation", tmp_path)
    assert out["rc"] == ExitCode.EXPECTATION and out["stop_reason"] == "drained"
    assert "expectation not met: fewer windows than expected" in err


@pytest.mark.sdk
def test_run_headless_timeout_fails_and_says_what_was_pending(tmp_path: Path) -> None:
    out, err = run_case("timeout", tmp_path)
    assert out["rc"] == ExitCode.RUN_FAILED and out["stop_reason"] == "timeout"
    assert "not drained after 0.3 s: " in err and " of 20 items" in err


@pytest.mark.sdk
def test_run_headless_stops_on_sigterm(tmp_path: Path) -> None:
    out, _ = run_case("signal", tmp_path)
    assert out["rc"] == ExitCode.OK and out["stop_reason"] == "signal"
    assert out["status"]["termination"] == "STOPPED" and out["sigterm_restored"]


@pytest.mark.sdk
def test_run_headless_runs_the_stall_monitor_around_the_session(tmp_path: Path) -> None:
    out, _ = run_case("monitor", tmp_path)
    assert out["rc"] == ExitCode.OK and out["monitor"] == ["start", "stop"]


def stub_graph(*, uncovered: bool = False, export: bool = False):
    """A clock source into a sink in lane ``ingest``; optionally an extra operator in no
    lane, or a String output exported over holomq."""
    import holoscan as hs
    from holoscan.schema import String

    from holoscan_finance.app.placement import Lanes
    from holoscan_finance.ingest.affinity import CpuSets

    class Source(hs.Operator):
        out = hs.Output("out", hs.pyobject)
        summary = hs.Output("summary", hs.payload.schema.String)

        def contract(self) -> hs.Contract:
            return hs.Contract().trigger(hs.OnClock(period=timedelta(milliseconds=1)))

        def compute(self, ctx: hs.ExecutionContext) -> None:
            self.out.emit(1)
            self.summary.emit(String(data="{}"))

    class Sink(hs.Operator):
        items = hs.Input("items", hs.pyobject)

        def contract(self) -> hs.Contract:
            return hs.Contract().trigger(hs.OnEach(self.items))

        def compute(self, ctx: hs.ExecutionContext) -> None:
            self.items.receive_data()

    g = hs.Graph("stub", default_clock=hs.RealtimeClock)
    source, sink = g.op(Source, "source"), g.op(Sink, "sink")
    g.add_flow(source.out, sink.items, queue_policy=hs.QueuePolicy.BLOCKING, queue_depth=4)
    if export:
        topic = g.export_topic(
            "holomq://stub.summary", hs.payload.schema.String, endpoint_name="stub"
        )
        g.add_flow(
            source.summary, topic.input, queue_policy=hs.QueuePolicy.DROP_OLDEST, queue_depth=1
        )
    if uncovered:
        g.op(Sink, "orphan")
    lanes = Lanes(CpuSets.compute()).add("ingest", source, sink)
    lanes.apply(g)
    return g, lanes


@pytest.mark.sdk
def test_compile_plan_refuses_a_rejected_plan(capsys: pytest.CaptureFixture[str]) -> None:
    g, lanes = stub_graph(uncovered=True)
    with pytest.raises(SystemExit) as exc:
        runner.compile_plan(g, lanes)
    assert exc.value.code == ExitCode.PLAN_REJECTED == 3
    assert "PARTITION_COVER_INVALID" in capsys.readouterr().err


@pytest.mark.sdk
def test_compile_plan_refuses_a_plan_that_breaks_a_rule(capsys: pytest.CaptureFixture[str]) -> None:
    g, lanes = stub_graph()
    with pytest.raises(SystemExit) as exc:
        runner.compile_plan(g, lanes, rules=[lambda view: [], lambda view: ["broken promise"]])
    assert exc.value.code == ExitCode.PLAN_RULES == 4
    assert "plan rule violated: broken promise" in capsys.readouterr().err


@pytest.mark.sdk
def test_compile_plan_writes_plan_json_catalog_and_summary(tmp_path: Path) -> None:
    g, lanes = stub_graph(export=True)
    ctx = SimpleNamespace(plan_summary=None)
    plan_json = tmp_path / "out" / "plan.json"
    catalog = tmp_path / "missing" / "dir" / "endpoints.json"  # save_json makes no directory
    plan = runner.compile_plan(
        g, lanes, ctx=ctx, plan_json=str(plan_json), topics_catalog=str(catalog)
    )
    assert plan.ok
    assert json.loads(plan_json.read_text())["payload_edges"]
    assert json.loads(catalog.read_text())["format"] == "holoscan.channel-endpoints"
    assert [lane["name"] for lane in ctx.plan_summary["lanes"]] == ["ingest"]


@pytest.mark.sdk
def test_print_plan_prints_the_summary(capsys: pytest.CaptureFixture[str]) -> None:
    g, lanes = stub_graph()
    assert runner.print_plan(runner.compile_plan(g, lanes)) == ExitCode.OK
    summary = json.loads(capsys.readouterr().out)
    assert summary["lanes"][0]["operators"] == ["sink", "source"]
    assert {"from": "source.out", "to": "sink.items", "policy": "BLOCKING", "depth": 4} in summary[
        "edges"
    ]


def test_write_report_writes_the_contexts_report(tmp_path: Path) -> None:
    report = {"stop_reason": "drained", "packets": np.int64(7), "lag": np.float64(0.5)}
    ctx = SimpleNamespace(report_dict=lambda session: report)
    path = tmp_path / "reports" / "run.json"
    assert runner.write_report(str(path), ctx, session=None) is report
    assert json.loads(path.read_text()) == {"stop_reason": "drained", "packets": 7, "lag": 0.5}
    assert runner.write_report(None, ctx, session=None) is report  # no path: nothing written


def test_exit_codes_are_stable() -> None:
    """Launchers and CI read these numbers."""
    assert [(c.name, int(c)) for c in ExitCode] == [
        ("OK", 0),
        ("RUN_FAILED", 1),
        ("USAGE", 2),
        ("PLAN_REJECTED", 3),
        ("PLAN_RULES", 4),
        ("EXPECTATION", 5),
    ]
