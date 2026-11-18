# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every ``equity_analytics`` mode and option compiles to exactly the lanes, edges, clocks and
placement of docs/ARCHITECTURE.md, with no I/O and no GPU, so it runs on the CPU-only tier.

Ingest {udp, daqiri-host, gpudirect, replay} x analytics {compare, gpu, cpu} (minus the two
GPU front ends with CPU analytics) x ``--topics`` x ``--ingest-busy-poll``. In DAQIRI modes
the two highest CPUs of the process stand in for the lcores, so the "no lane on a reserved
CPU" rule is exercised whatever cpuset the test runs under.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import holoscan as hs
import pytest
from apphelpers import REPO, child_env, dry_equity_context, equity_args, load_app

from holoscan_finance.app.plan import POLICIES, Edge, PlanView, check_plan
from holoscan_finance.settings import DAQIRI_INGEST_MODES, GPU_INGEST_MODES, INGEST_MODES

pytestmark = pytest.mark.sdk

MODES = [
    (ingest, backend)
    for ingest in INGEST_MODES
    for backend in ("compare", "gpu", "cpu")
    if not (ingest in GPU_INGEST_MODES and backend == "cpu")
]
BUSY_POLL = hs.execution.WakeMode.BUSY_POLL.value
SLEEPING = hs.execution.WakeMode.SLEEPING_WAITSET.value


@pytest.fixture(scope="module")
def app() -> ModuleType:
    return load_app("equity_analytics")


@pytest.fixture(scope="module")
def mask() -> list[int]:
    cpus = sorted(os.sched_getaffinity(0))
    if len(cpus) < 4:
        pytest.skip("needs 4 CPUs: an ingest CPU, two stand-in lcores and the rest")
    return cpus


class RecordingGraph(hs.Graph):
    """``hs.Graph`` that remembers the Params each operator was declared with."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.params: dict[str, dict] = {}

    def op(
        self, type_or_name: type[hs.Operator], instance_name: str, **params: object
    ) -> hs.operators.OperatorHandle:
        self.params[instance_name] = params
        return super().op(type_or_name, instance_name, **params)


def flags(
    ingest: str, backend: str, mask: list[int], *, topics: bool = False, busy_poll: bool = False
) -> list[str]:
    out = ["--ingest", ingest, "--analytics-backend", backend, "--plan-only"]
    if ingest in DAQIRI_INGEST_MODES:  # stand-in lcores inside this process's mask
        out += ["--nic-master-core", str(mask[-2]), "--nic-rx-core", str(mask[-1])]
    if topics:
        out.append("--topics")
    if busy_poll:
        out += ["--ingest-cpus", str(mask[0]), "--ingest-busy-poll"]
    return out


def compose(app: ModuleType, monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> tuple:
    """``app.compose`` on a dry context; returns (args, ctx, plan, Params per operator)."""
    args = equity_args(*argv)
    ctx = dry_equity_context(args)
    monkeypatch.setattr(hs, "Graph", RecordingGraph)
    g, lanes = app.compose(args, ctx)
    plan = hs.compile(g, lanes.compile_options())
    assert plan.ok, plan.summary()
    return args, ctx, plan, g.params


def expected_edges(ingest: str, backend: str, topics: bool) -> set[Edge]:
    if ingest in GPU_INGEST_MODES:
        edges = {Edge("source.events", "bars.events", "BLOCKING", 4)}
    else:
        edges = {
            Edge("source.packets", "decode.packets", "BLOCKING", 16),
            Edge("decode.events", "bars.events", "BLOCKING", 16),
        }
    edges |= {
        Edge("bars.window", "risk.window", "DROP_OLDEST", 1),
        Edge("risk.risk", "publish.items", "BLOCKING", 8),
    }
    if backend == "compare":
        edges |= {
            Edge("risk.validate", "crosscheck.risk", "BLOCKING", 2),
            Edge("crosscheck.comparison", "publish_comparison.items", "BLOCKING", 2),
        }
    if topics:
        edges |= {
            Edge("risk.risk", "risk_summary.risk", "DROP_OLDEST", 1),
            Edge("risk_summary.summary", "@export.external", "DROP_OLDEST", 1),
        }
    return edges


@pytest.mark.parametrize("busy_poll", [False, True], ids=["sleep", "busy_poll"])
@pytest.mark.parametrize("topics", [False, True], ids=["no_topics", "topics"])
@pytest.mark.parametrize(("ingest", "backend"), MODES)
def test_every_combination_compiles_as_designed(
    app: ModuleType,
    mask: list[int],
    monkeypatch: pytest.MonkeyPatch,
    ingest: str,
    backend: str,
    topics: bool,
    busy_poll: bool,
) -> None:
    argv = flags(ingest, backend, mask, topics=topics, busy_poll=busy_poll)
    args, ctx, plan, params = compose(app, monkeypatch, argv)
    view = PlanView(plan)
    gpu = ingest in GPU_INGEST_MODES

    lanes = {
        "ingest": ["bars", "source"] if gpu else ["bars", "decode", "source"],
        "analytics": ["publish", "risk"],
    }
    if backend == "compare":
        lanes["crosscheck"] = ["crosscheck", "publish_comparison"]
    if topics:
        lanes["topics"] = ["risk_summary"]
    assert view.lanes == lanes and view.automatic == []

    assert set(view.edges) == expected_edges(ingest, backend, topics)
    assert len(view.edges) == len(set(view.edges))

    clocks = {"source": 500_000, "risk": 250_000}
    if backend == "compare":
        clocks["crosscheck"] = 5_000_000
    assert view.clocks == clocks

    cpus = ctx.cpus
    for lane, placement in view.placements.items():
        if lane == "ingest":
            wake = BUSY_POLL if busy_poll else SLEEPING
            assert placement == {"cpus": sorted(cpus.ingest), "wake_mode": wake}
        else:
            assert placement == {"cpus": sorted(cpus.helper), "wake_mode": SLEEPING}
        assert not cpus.reserved & set(placement["cpus"])
    assert set(view.placements) == set(lanes)
    assert params["source"]["helper_cpus"] == sorted(cpus.helper)  # drains never on the ingest lane
    if busy_poll:
        carved = mask[0]
        assert cpus.ingest == {carved}
        assert all(
            carved not in view.placements[lane]["cpus"] for lane in lanes if lane != "ingest"
        )
        assert carved not in params["source"]["helper_cpus"]
    if ingest in DAQIRI_INGEST_MODES:
        assert cpus.reserved == set(mask[-2:])
        # daqiri_init runs on the full mask
        assert params["source"]["init_cpus"] == sorted(cpus.all)
        assert params["source"]["master_core"] == mask[-2]

    risk = params["risk"]
    assert risk["executor"] == ("inline" if gpu else "process")
    assert risk["backend"] == ("cpu" if backend == "cpu" else "gpu")
    assert risk["cross_check"] is (backend == "compare")
    # The CPU workers tune for each universe as its manifest arrives, before its first window.
    assert risk["directory"].obj is ctx.directory_knob
    if backend == "compare":
        assert params["crosscheck"]["directory"].obj is ctx.directory_knob
    assert params["bars"]["backend"] == ("gpu" if gpu else "cpu")

    assert view.runtime_updates == []
    assert check_plan(plan, app.rules(ctx)) == []


def test_risk_polls_every_250_us_by_default(
    app: ModuleType, mask: list[int], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, plan, _ = compose(app, monkeypatch, flags("replay", "compare", mask))
    assert PlanView(plan).clocks["risk"] == 250_000


def test_the_5ms_risk_poll_fallback_compiles_and_keeps_the_rules(
    app: ModuleType, mask: list[int], monkeypatch: pytest.MonkeyPatch
) -> None:
    argv = flags("replay", "compare", mask) + ["--risk-poll-us", "5000"]
    _, ctx, plan, _ = compose(app, monkeypatch, argv)
    assert PlanView(plan).clocks["risk"] == 5_000_000
    assert check_plan(plan, app.rules(ctx)) == []


def test_rules_catch_a_risk_poll_slower_than_promised(
    app: ModuleType, mask: list[int], monkeypatch: pytest.MonkeyPatch
) -> None:
    argv = flags("replay", "gpu", mask) + ["--risk-poll-us", "5000"]
    args, ctx, plan, _ = compose(app, monkeypatch, argv)
    args.risk_poll_us = 1000  # what the rules promise vs the graph that was built
    assert check_plan(plan, app.rules(ctx)) == [
        "clock_period_max: risk polls every 5000 us > 1000 us"
    ]


def test_queue_policy_ints_are_calibrated() -> None:
    """plan.json's queue_policy ints, read from a two-edge graph with known policies."""
    from holoscan_finance.operators.edges import LATEST, lossless

    class Source(hs.Operator):
        a = hs.Output("a", hs.pyobject)
        b = hs.Output("b", hs.pyobject)

        def contract(self) -> hs.Contract:
            return hs.Contract().trigger(hs.OnClock(period=hs.time.ms(1)))

        def compute(self, ctx: hs.ExecutionContext) -> None:
            pass

    class Sink(hs.Operator):
        x = hs.Input("x", hs.pyobject)

        def contract(self) -> hs.Contract:
            return hs.Contract().trigger(hs.OnEach(self.x))

        def compute(self, ctx: hs.ExecutionContext) -> None:
            pass

    g = hs.Graph("calibrate", default_clock=hs.RealtimeClock)
    source, blocking, latest = g.op(Source, "source"), g.op(Sink, "blocking"), g.op(Sink, "latest")
    g.add_flow(source.a, blocking.x, **lossless(3))
    g.add_flow(source.b, latest.x, **LATEST)
    doc = json.loads(hs.compile(g).json())
    ports = {p["id"]: p["local_name"] for p in doc["ports"]}
    policy = {
        ports[e["source"]]: (e["queue_policy"], e["queue_depth"]) for e in doc["payload_edges"]
    }
    assert policy == {"a": (0, 3), "b": (1, 1)}
    assert POLICIES == {0: "BLOCKING", 1: "DROP_OLDEST"}


def run_app(*argv: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(REPO / "examples/equity_analytics/equity_analytics.py"), *argv],
        capture_output=True,
        text=True,
        timeout=120,
        env=child_env(),
    )


def test_plan_only_prints_the_plan_and_writes_plan_json(tmp_path: Path) -> None:
    """``main()`` with the real context: compile, check the rules, print, start nothing."""
    plan_json = tmp_path / "plan" / "equity_analytics.plan.json"
    done = run_app(
        "--ingest", "replay", "--pcap", "none", "--tape", "none", "--plan-only",
        "--plan-json", str(plan_json),
    )  # fmt: skip
    assert done.returncode == 0, done.stderr[-3000:]
    summary = json.loads(done.stdout)
    assert {lane["name"] for lane in summary["lanes"]} == {"ingest", "analytics", "crosscheck"}
    assert json.loads(plan_json.read_text())["payload_edges"]


def test_plan_only_with_topics_writes_the_endpoint_catalog(tmp_path: Path) -> None:
    catalog = tmp_path / "new" / "dir" / "ep.json"  # save_json makes no directory
    done = run_app(
        "--ingest", "replay", "--pcap", "none", "--tape", "none", "--plan-only", "--topics",
        "--topics-catalog", str(catalog),
    )  # fmt: skip
    assert done.returncode == 0, done.stderr[-3000:]
    endpoints = json.loads(catalog.read_text())
    assert "finance.risk" in json.dumps(endpoints["services"])
