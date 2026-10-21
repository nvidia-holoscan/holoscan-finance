# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plan rules on small stub graphs: each holds and each catches its violation; a missing
``plan.json`` key is reported, never raised; ``plan_summary`` has the documented shape."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from datetime import timedelta

import holoscan as hs
import pytest
from holoscan.schema import String

from holoscan_finance.app import plan as rules
from holoscan_finance.app.placement import Lanes
from holoscan_finance.app.plan import Edge, PlanView, check_plan, plan_summary
from holoscan_finance.ingest.affinity import CpuSets

pytestmark = pytest.mark.sdk


class Source(hs.Operator):
    out = hs.Output("out", hs.pyobject)

    def contract(self) -> hs.Contract:
        return hs.Contract().trigger(hs.OnClock(period=timedelta(microseconds=500)))

    def compute(self, ctx: hs.ExecutionContext) -> None:
        self.out.emit(1)


class Bars(hs.Operator):  # one wired output, one optional output nobody reads
    events = hs.Input("events", hs.pyobject)
    window = hs.Output("window", hs.pyobject)
    spare = hs.Output("spare", hs.pyobject)

    def contract(self) -> hs.Contract:
        return hs.Contract().trigger(hs.OnEach(self.events))

    def compute(self, ctx: hs.ExecutionContext) -> None:
        self.window.emit(self.events.receive_data())


class Risk(hs.Operator):  # a clock-paced consumer with a String output for the export
    window = hs.Input("window", hs.pyobject)
    summary = hs.Output("summary", hs.payload.schema.String)

    def contract(self) -> hs.Contract:
        return (
            hs.Contract()
            .trigger(hs.OnClock(period=timedelta(milliseconds=1)))
            .input(self.window, hs.DrainAll())
        )

    def compute(self, ctx: hs.ExecutionContext) -> None:
        if len(self.window.selected_batch()):
            self.summary.emit(String(data="{}"))


class TwoInputs(hs.Operator):  # its second input stays unwired: it compiles, then cannot run
    a = hs.Input("a", hs.pyobject)
    b = hs.Input("b", hs.pyobject)

    def compute(self, ctx: hs.ExecutionContext) -> None:
        pass


@pytest.fixture(scope="module")
def cpus() -> CpuSets:
    return CpuSets.compute()


def compile_graph(cpus: CpuSets, *, lanes: bool = True, unwired_input: bool = False) -> hs.Plan:
    """source -> bars (BLOCKING/4) -> risk (DROP_OLDEST/1) -> holomq export (DROP_OLDEST/1);
    lanes ingest={source, bars}, analytics={risk} unless ``lanes`` is False."""
    g = hs.Graph("rules", default_clock=hs.RealtimeClock)
    source, bars, risk = g.op(Source, "source"), g.op(Bars, "bars"), g.op(Risk, "risk")
    g.add_flow(source.out, bars.events, queue_policy=hs.QueuePolicy.BLOCKING, queue_depth=4)
    g.add_flow(bars.window, risk.window, queue_policy=hs.QueuePolicy.DROP_OLDEST, queue_depth=1)
    topic = g.export_topic(
        "holomq://rules.risk", hs.payload.schema.String, endpoint_name="rules.risk"
    )
    g.add_flow(risk.summary, topic.input, queue_policy=hs.QueuePolicy.DROP_OLDEST, queue_depth=1)
    ingest = [source, bars]
    if unwired_input:
        two = g.op(TwoInputs, "two")
        g.add_flow(bars.spare, two.a, queue_policy=hs.QueuePolicy.BLOCKING, queue_depth=2)
        ingest.append(two)
    if not lanes:
        return hs.compile(g)
    placed = Lanes(cpus).add("ingest", *ingest).add("analytics", risk)
    placed.apply(g)
    plan = hs.compile(g, placed.compile_options())
    assert plan.ok, plan.summary()
    return plan


@pytest.fixture(scope="module")
def view(cpus: CpuSets) -> PlanView:
    return PlanView(compile_graph(cpus))


class EditedPlan:
    """A compiled plan whose ``plan.json()`` is edited, as a later SDK might change it."""

    def __init__(self, plan: hs.Plan, edit: Callable[[dict], None]) -> None:
        self.partitions = plan.partitions
        self._doc = json.loads(plan.json())
        edit(self._doc)

    def json(self) -> str:
        return json.dumps(self._doc)


def test_plan_view_joins_edges_ports_and_operators(view: PlanView) -> None:
    assert view.lanes == {"ingest": ["bars", "source"], "analytics": ["risk"]}
    assert set(view.edges) == {
        Edge("source.out", "bars.events", "BLOCKING", 4),
        Edge("bars.window", "risk.window", "DROP_OLDEST", 1),
        Edge("risk.summary", "@export.external", "DROP_OLDEST", 1),  # target owner: an endpoint
    }
    assert len(view.edges) == 3
    assert view.clocks == {"source": 500_000, "risk": 1_000_000}
    assert view.outputs == {"source": ["out"], "bars": ["spare", "window"]}  # not risk's String
    assert view.inputs == {"bars": ["events"], "risk": ["window"]}
    assert view.runtime_updates == []
    assert view.lane_of("bars") == "ingest" and view.lane_of("nope") is None


def test_same_lane(view: PlanView) -> None:
    assert rules.same_lane("source", "bars")(view) == []
    assert rules.same_lane("source", "risk")(view) == [
        "same_lane: source (ingest), risk (analytics) are not in one authored lane"
    ]
    assert rules.same_lane("source", "ghost")(view) == [
        "same_lane: source (ingest), ghost (no authored lane) are not in one authored lane"
    ]


def test_edge(view: PlanView) -> None:
    assert rules.edge("bars.window", "risk.window", policy="DROP_OLDEST", depth=1)(view) == []
    assert rules.edge("source.out", "bars.events", policy="BLOCKING", min_depth=4)(view) == []
    assert rules.edge("risk.summary", "*", policy="DROP_OLDEST", depth=1)(view) == []
    assert rules.edge("bars.window", "risk.window", policy="BLOCKING", depth=4)(view) == [
        "edge bars.window -> risk.window: policy DROP_OLDEST, expected BLOCKING",
        "edge bars.window -> risk.window: depth 1, expected 4",
    ]
    assert rules.edge("source.out", "*", min_depth=16)(view) == [
        "edge source.out -> bars.events: depth 4, expected >= 16"
    ]
    assert rules.edge("bars.spare", "*")(view) == ["edge bars.spare -> *: no such edge"]


def test_clock_period_max(view: PlanView) -> None:
    assert rules.clock_period_max("risk", 1000)(view) == []
    assert rules.clock_period_max("risk", 500)(view) == [
        "clock_period_max: risk polls every 1000 us > 500 us"
    ]
    assert rules.clock_period_max("bars", 1000)(view) == [
        "clock_period_max: bars has no clock trigger"
    ]


def test_no_automatic_partitions(cpus: CpuSets, view: PlanView) -> None:
    assert rules.no_automatic_partitions()(view) == []
    unplaced = PlanView(compile_graph(cpus, lanes=False))
    assert unplaced.lanes == {} and unplaced.automatic
    violations = rules.no_automatic_partitions()(unplaced)
    assert violations and all(v.startswith("automatic partition [") for v in violations)


def test_outputs_connected(view: PlanView) -> None:
    assert rules.outputs_connected()(view) == ["output bars.spare is not connected"]


def test_inputs_connected(cpus: CpuSets, view: PlanView) -> None:
    assert rules.inputs_connected()(view) == []
    unwired = PlanView(compile_graph(cpus, unwired_input=True))
    assert rules.inputs_connected()(unwired) == ["input two.b is not connected"]
    assert rules.outputs_connected()(unwired) == []


def test_lanes_avoid_cpus(cpus: CpuSets, view: PlanView) -> None:
    used = min(cpus.app)
    outside = max(os.sched_getaffinity(0)) + 1
    assert rules.lanes_avoid_cpus([])(view) == []
    assert rules.lanes_avoid_cpus([outside])(view) == []
    assert sorted(rules.lanes_avoid_cpus([used])(view)) == [
        f"lane analytics runs on reserved CPUs [{used}]",
        f"lane ingest runs on reserved CPUs [{used}]",
    ]
    unplaced = EditedPlan(compile_graph(cpus), lambda doc: doc["partition_placements"].pop())
    lost = next(
        name for name in ("ingest", "analytics") if name not in PlanView(unplaced).placements
    )
    assert f"lane {lost} has no CPU placement" in " ".join(
        rules.lanes_avoid_cpus([outside])(PlanView(unplaced))
    )


def test_no_runtime_parameter_updates(cpus: CpuSets, view: PlanView) -> None:
    assert rules.no_runtime_parameter_updates()(view) == []

    def enable(doc: dict) -> None:
        for op in doc["operator_parameters"]:
            op["runtime_updates_enabled"] = op["operator_path"] == "risk"

    edited = PlanView(EditedPlan(compile_graph(cpus), enable))
    assert rules.no_runtime_parameter_updates()(edited) == [
        "risk has runtime parameter updates enabled"
    ]


@pytest.mark.parametrize(
    ("edit", "rule", "message"),
    [
        (lambda d: d.pop("payload_edges"), rules.edge("a.b", "*"), "payload_edges"),
        (lambda d: d.pop("clock_triggers"), rules.clock_period_max("risk", 1000), "clock_triggers"),
        (lambda d: d.pop("ports"), rules.outputs_connected(), "ports"),
        (lambda d: d.pop("ports"), rules.inputs_connected(), "ports"),
        (
            lambda d: d.pop("partition_placements"),
            rules.lanes_avoid_cpus([0]),
            "partition_placements",
        ),
        (
            lambda d: d.pop("operator_parameters"),
            rules.no_runtime_parameter_updates(),
            "operator_parameters",
        ),
        (
            lambda d: [e.pop("queue_policy") for e in d["payload_edges"]],
            rules.edge("a.b", "*"),
            "queue_policy",
        ),
    ],
)
def test_a_missing_plan_json_key_is_a_violation_not_an_exception(
    cpus: CpuSets, edit, rule, message: str
) -> None:
    plan = EditedPlan(compile_graph(cpus), edit)
    assert rule(PlanView(plan)) == [f"plan.json key {message} unavailable"]
    assert check_plan(plan, [rule]) == [f"plan.json key {message} unavailable"]


def test_check_plan_collects_every_rule(cpus: CpuSets) -> None:
    plan = compile_graph(cpus)
    assert (
        check_plan(plan, [rules.same_lane("source", "bars"), rules.no_automatic_partitions()]) == []
    )
    assert check_plan(plan, [rules.outputs_connected(), rules.clock_period_max("risk", 100)]) == [
        "output bars.spare is not connected",
        "clock_period_max: risk polls every 1000 us > 100 us",
    ]


def test_plan_summary_shape(cpus: CpuSets) -> None:
    summary = plan_summary(compile_graph(cpus))
    assert set(summary) == {"lanes", "edges", "clocks", "automatic_partitions", "sdk"}
    app = sorted(cpus.app)
    assert sorted(summary["lanes"], key=lambda lane: lane["name"]) == [
        {"name": "analytics", "operators": ["risk"], "cpus": app, "wake_mode": "SLEEPING_WAITSET"},
        {
            "name": "ingest",
            "operators": ["bars", "source"],
            "cpus": app,
            "wake_mode": "SLEEPING_WAITSET",
        },
    ]
    assert {
        "from": "risk.summary",
        "to": "@export.external",
        "policy": "DROP_OLDEST",
        "depth": 1,
    } in (summary["edges"])
    assert sorted(summary["clocks"], key=lambda c: c["operator"]) == [
        {"operator": "risk", "period_us": 1000.0},
        {"operator": "source", "period_us": 500.0},
    ]
    assert summary["automatic_partitions"] == []
    assert summary["sdk"] == os.environ.get("HOLOSCAN_SDK_VERSION")
    json.dumps(summary)  # served as-is by GET /plan
