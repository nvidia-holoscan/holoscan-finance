# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every ``market_data_bench`` composition compiles to the intended lanes, edges, clocks and
placement, and passes its own plan rules; its flags are checked and its report has the
documented keys and exit codes.

Compiled on the CPU-only tier: ``--plan-only`` contexts register no replay and read no file,
and nothing runs. The flags crossed are ``--check`` x ``--risk {none,gpu}`` x
``--ingest-busy-poll`` (with a one-CPU ``--ingest-cpus``).
"""

from __future__ import annotations

import itertools
import json
import os

import holoscan as hs
import pytest
from apphelpers import cpu_sets, load_app

from holoscan_finance.app import bench
from holoscan_finance.app.bench import BenchContext
from holoscan_finance.app.plan import Edge, PlanView, check_plan, plan_summary
from holoscan_finance.app.runner import ExitCode
from holoscan_finance.operators import GpuMitchReplaySourceOp
from holoscan_finance.payloads import BarCounters

pytestmark = pytest.mark.sdk

APP = load_app("market_data_bench")
CPUS = sorted(os.sched_getaffinity(0))


class RecordingSource(GpuMitchReplaySourceOp):
    """The bench's source, recording the ``helper_cpus`` it was given at ``graph.op``."""

    helper_cpus_seen: list[list[int]] = []

    def __init__(self) -> None:
        super().__init__()
        RecordingSource.helper_cpus_seen.append(list(self.helper_cpus))


def compile_bench(monkeypatch, *flags: str):
    args = bench.normalize_bench(bench.bench_parser().parse_args(["--plan-only", *flags]))
    cpus = cpu_sets(args)
    ctx = BenchContext.from_args(args, cpus)
    RecordingSource.helper_cpus_seen.clear()
    monkeypatch.setattr(APP, "GpuMitchReplaySourceOp", RecordingSource)
    g, lanes = APP.compose(args, ctx)
    plan = hs.compile(g, lanes.compile_options())
    assert plan.ok, plan.summary()
    return args, ctx, cpus, plan


CASES = list(itertools.product((False, True), ("none", "gpu"), (False, True)))


@pytest.mark.parametrize("check, risk, busy_poll", CASES)
def test_bench_plan(monkeypatch, check, risk, busy_poll):
    if busy_poll and len(CPUS) < 2:
        pytest.skip("a busy-polled ingest CPU needs another CPU for the other lanes")
    flags = ["--risk", risk] + (["--check"] if check else [])
    if busy_poll:
        flags += ["--ingest-cpus", str(CPUS[-1]), "--ingest-busy-poll"]
    args, ctx, cpus, plan = compile_bench(monkeypatch, *flags)
    view = PlanView(plan)

    lanes = {"ingest": ["bars", "source"]}
    edges = [Edge("source.events", "bars.events", "BLOCKING", 4)]
    if check:
        lanes["ingest"] = ["bars", "source", "windows"]
        lanes["reference"] = ["ref_bars", "ref_windows"]
        edges += [
            Edge("bars.window", "windows.items", "BLOCKING", 64),
            Edge("source.events", "ref_bars.events", "BLOCKING", 8),
            Edge("ref_bars.window", "ref_windows.items", "BLOCKING", 64),
        ]
    if risk == "gpu":
        lanes["analytics"] = ["publish", "risk"]
        edges += [
            Edge("bars.window", "risk.window", "DROP_OLDEST", 1),
            Edge("risk.risk", "publish.items", "BLOCKING", 8),
        ]
    assert view.lanes == lanes and view.automatic == []
    assert sorted(view.edges, key=str) == sorted(edges, key=str)
    clocks = {"source": 500_000} | ({"risk": 1_000_000} if risk == "gpu" else {})
    assert view.clocks == clocks
    # Without windows to emit, the GPU bars declare no window port.
    assert view.outputs.get("bars", []) == (["window"] if check or risk == "gpu" else [])

    assert set(view.placements) == set(lanes)
    for lane, placed in view.placements.items():
        assert placed["cpus"] == sorted(cpus.ingest if lane == "ingest" else cpus.helper), lane
    wake = {lane["name"]: lane["wake_mode"] for lane in plan_summary(plan)["lanes"]}
    assert (wake["ingest"] == "BUSY_POLL") == busy_poll
    # hf-replay is created under the helper set: never on the busy-polled ingest CPU.
    assert RecordingSource.helper_cpus_seen and all(
        seen == sorted(cpus.helper) for seen in RecordingSource.helper_cpus_seen
    )
    if busy_poll:
        assert CPUS[-1] not in cpus.helper and sorted(cpus.ingest) == [CPUS[-1]]
    assert view.unavailable("runtime_updates") == [] and view.runtime_updates == []
    assert check_plan(plan, APP.rules(args, ctx)) == []


def test_rules_catch_a_broken_composition(monkeypatch):
    """The bench's rules are not vacuous: a plan without the check lanes fails the check rules."""
    args, ctx, _, plan = compile_bench(monkeypatch, "--risk", "gpu")
    args.check = True
    assert check_plan(plan, APP.rules(args, ctx))


def test_plan_only_reads_no_capture():
    args = bench.normalize_bench(bench.bench_parser().parse_args(["--plan-only", "--check"]))
    assert (args.pcap, args.tape) == ("none", "none")
    ctx = BenchContext.from_args(args, cpu_sets(args))
    assert ctx.replay_generations is None and ctx.directory.current is None


def bench_args(*flags: str):
    return bench.normalize_bench(bench.bench_parser().parse_args(list(flags)))


@pytest.mark.parametrize(
    "flags",
    [
        ["--tape", "t"],  # no capture
        ["--pcap", "p"],  # no tape
        ["--pcap", "p", "--tape", "t", "--loops", "0"],  # the bench is finite
        ["--pcap", "p", "--tape", "t", "--check", "--loops", "2"],
        ["--pcap", "p", "--tape", "t", "--require-realtime", "--pps", "0"],
        ["--pcap", "p", "--tape", "t", "--require-realtime", "--check"],
        ["--pcap", "p", "--tape", "t", "--bar-seconds", "7"],  # 300 s is not a multiple
        ["--pcap", "p", "--tape", "t", "--interval", "0"],
        ["--pcap", "p", "--tape", "t", "--ingest-cpus", "x"],
    ],
)
def test_flags_that_cannot_run_exit_with_usage(flags):
    with pytest.raises(SystemExit) as exc:
        bench_args(*flags)
    assert exc.value.code == ExitCode.USAGE


def test_gates_follow_the_flags():
    args = bench_args("--plan-only", "--check", "--require-lossless", "--max-lag-p99-ms", "5")
    gates = args.gates
    assert (gates.check, gates.require_lossless, gates.require_realtime) == (True, True, False)
    assert (gates.min_msgs_per_s, gates.max_lag_p99_ms) == (0.0, 5.0)


REPORT_KEYS = [
    "ok", "failures", "termination", "stop_reason", "config", "packets", "received_messages",
    "messages", "events", "completed_bars", "seconds", "msgs_per_s", "packets_per_s",
    "requested_pps", "achieved_pps", "sequence_gaps", "ring_dropped", "nic_dropped",
    "stale_packets", "malformed_packets", "decode_malformed", "ring_peak", "ring_waits",
    "ring_wait_ms", "lag_p50_ms", "lag_p99_ms", "lag_max_ms", "retries", "replay_thread_cpu_s",
    "worst_compute_ms", "worst_burst_ms", "lanes", "risk", "check",
]  # fmt: skip


def finished(tmp_path, *flags: str, session_rc=ExitCode.OK, termination="STOPPED"):
    """``finish_report`` of a run that never started (empty stats): the verdict alone."""
    report = tmp_path / "out" / "bench.json"
    args = bench_args("--plan-only", "--report", str(report), *flags)
    ctx = BenchContext.from_args(args, cpu_sets(args))
    ctx.status.termination, ctx.stop_reason = termination, "drained"
    rc = ctx.finish_report(session_rc)
    return rc, json.loads(report.read_text())


def test_report_keys_and_exit_codes(tmp_path, capsys):
    rc, report = finished(tmp_path)
    assert rc == ExitCode.OK and report["ok"] and list(report) == REPORT_KEYS
    assert json.loads(capsys.readouterr().out) == report
    assert report["config"]["pps"] == 200000 and "gates" not in report["config"]
    # Gates never pass on values the run did not produce: no replay stats, no NIC drop count.
    rc, report = finished(tmp_path, "--require-lossless")
    assert rc == ExitCode.EXPECTATION and report["failures"][0].startswith("nic_dropped")
    rc, report = finished(tmp_path, session_rc=ExitCode.RUN_FAILED, termination="FAILED")
    assert rc == ExitCode.RUN_FAILED and not report["ok"] and len(report["failures"]) == 2
    rc, report = finished(tmp_path, "--risk", "gpu")
    assert report["risk"]["analyzed"] == 0 and report["risk"]["receive_to_result_ms"]["p50"] is None


@pytest.mark.parametrize("loops, seconds", [("1", 2.0), ("2", None)])
def test_rates_span_one_generation(loops, seconds):
    """The bar counters restart with every generation: a run of several passes reports no
    rates instead of the run's messages over the last pass's span."""
    args = bench_args("--plan-only", "--loops", loops)
    ctx = BenchContext.from_args(args, cpu_sets(args))
    ctx.ingest_stats.received_messages = 10
    ctx.bars_stats.counters = BarCounters(
        generation=1, events=8, quotes=6, trades=2, invalid_quotes=0, late_events=0,
        completed_bars=1, last_event_ns=0, first_batch_ns=1_000_000_000,
        last_batch_ns=3_000_000_000,
    )  # fmt: skip
    report = ctx.report_dict()
    assert report["seconds"] == seconds
    assert report["msgs_per_s"] == (5.0 if seconds else None)
