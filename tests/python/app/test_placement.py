# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``Lanes``: one partition per lane, ``ingest`` on its own CPU set, busy-polled only on request."""

from __future__ import annotations

import os
from datetime import timedelta

import holoscan as hs
import pytest

from holoscan_finance.app.placement import Lanes
from holoscan_finance.app.plan import PlanView
from holoscan_finance.ingest.affinity import CpuSets

pytestmark = pytest.mark.sdk


class Source(hs.Operator):
    out = hs.Output("out", hs.pyobject)

    def contract(self) -> hs.Contract:
        return hs.Contract().trigger(hs.OnClock(period=timedelta(milliseconds=1)))

    def compute(self, ctx: hs.ExecutionContext) -> None:
        self.out.emit(1)


class Each(hs.Operator):
    inp = hs.Input("inp", hs.pyobject)
    out = hs.Output("out", hs.pyobject)

    def contract(self) -> hs.Contract:
        return hs.Contract().trigger(hs.OnEach(self.inp))

    def compute(self, ctx: hs.ExecutionContext) -> None:
        self.out.emit(self.inp.receive_data())


class Poll(hs.Operator):
    inp = hs.Input("inp", hs.pyobject)

    def contract(self) -> hs.Contract:
        return (
            hs.Contract()
            .trigger(hs.OnClock(period=timedelta(milliseconds=1)))
            .input(self.inp, hs.DrainAll())
        )

    def compute(self, ctx: hs.ExecutionContext) -> None:
        self.inp.selected_batch()


def carved_cpus() -> CpuSets:
    """A busy-polled ingest CPU carved out of the process mask, as ``--ingest-cpus`` does."""
    mask = sorted(os.sched_getaffinity(0))
    if len(mask) < 2:
        pytest.skip("needs two CPUs")
    return CpuSets.compute(ingest=mask[:1], busy_poll=True)


def three_lane_graph(lanes: Lanes) -> hs.Graph:
    g = hs.Graph("lanes", default_clock=hs.RealtimeClock)
    source, bars, risk = g.op(Source, "source"), g.op(Each, "bars"), g.op(Poll, "risk")
    g.add_flow(source.out, bars.inp, queue_policy=hs.QueuePolicy.BLOCKING, queue_depth=4)
    g.add_flow(bars.out, risk.inp, queue_policy=hs.QueuePolicy.DROP_OLDEST, queue_depth=1)
    lanes.add("ingest", source).add("analytics", risk).add("ingest", bars)
    lanes.apply(g)
    return g


class RecordingGraph:
    """Stands in for ``hs.Graph``: records ``partition(name).add(*ops)`` calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple]] = []

    def partition(self, name: str) -> RecordingGraph.Partition:
        return self.Partition(self, name)

    class Partition:
        def __init__(self, graph: RecordingGraph, name: str) -> None:
            self.graph, self.name = graph, name

        def add(self, *ops: object) -> None:
            self.graph.calls.append((self.name, ops))


def test_a_repeated_name_extends_the_lane() -> None:
    g = hs.Graph("names", default_clock=hs.RealtimeClock)
    a, b, c = g.op(Source, "a"), g.op(Each, "b"), g.op(Poll, "c")
    lanes = Lanes(CpuSets.compute()).add("ingest", a).add("analytics", c).add("ingest", b)
    assert lanes.names == ["ingest", "analytics"]
    assert lanes.members("ingest") == ["a", "b"] and lanes.members("analytics") == ["c"]


def test_apply_authors_each_lane_once_and_only_once() -> None:
    lanes = Lanes(CpuSets.compute()).add("ingest", "a").add("analytics", "c").add("ingest", "b")
    graph = RecordingGraph()
    lanes.apply(graph)
    assert graph.calls == [("ingest", ("a", "b")), ("analytics", ("c",))]
    with pytest.raises(RuntimeError, match="already applied"):
        lanes.apply(graph)
    with pytest.raises(RuntimeError, match="already applied"):
        lanes.add("late", "d")
    assert len(graph.calls) == 2


def test_every_lane_is_placed_on_the_app_cpus_by_default() -> None:
    cpus = CpuSets.compute()
    lanes = Lanes(cpus)
    plan = hs.compile(three_lane_graph(lanes), lanes.compile_options())
    assert plan.ok, plan.summary()
    view = PlanView(plan)
    assert view.lanes == {"ingest": ["bars", "source"], "analytics": ["risk"]}
    assert view.automatic == []
    app = sorted(cpus.app)
    assert view.placements == {
        "ingest": {"cpus": app, "wake_mode": hs.execution.WakeMode.SLEEPING_WAITSET.value},
        "analytics": {"cpus": app, "wake_mode": hs.execution.WakeMode.SLEEPING_WAITSET.value},
    }
    assert not [d.token for d in plan.warnings]  # ALLOW: the lane split is intended


def test_ingest_gets_its_cpus_and_busy_poll_only_when_asked() -> None:
    cpus = carved_cpus()
    for busy in (False, True):
        lanes = Lanes(cpus, ingest_busy_poll=busy)
        plan = hs.compile(three_lane_graph(lanes), lanes.compile_options())
        assert plan.ok, plan.summary()
        wake = hs.execution.WakeMode.BUSY_POLL if busy else hs.execution.WakeMode.SLEEPING_WAITSET
        assert PlanView(plan).placements == {
            "ingest": {"cpus": sorted(cpus.ingest), "wake_mode": wake.value},
            "analytics": {
                "cpus": sorted(cpus.helper),
                "wake_mode": hs.execution.WakeMode.SLEEPING_WAITSET.value,
            },
        }
        assert not cpus.ingest & cpus.helper


def test_placed_lanes_run_on_their_cpus() -> None:
    """The placement reaches the lane threads (``os.sched_getaffinity(0)`` in compute)."""
    cpus = carved_cpus()
    seen: dict[str, set[int]] = {}

    class Probe(hs.Operator):
        inp = hs.Input("inp", hs.pyobject)

        def contract(self) -> hs.Contract:
            return hs.Contract().trigger(hs.OnEach(self.inp))

        def compute(self, ctx: hs.ExecutionContext) -> None:
            self.inp.receive_data()
            seen.setdefault(ctx.operator_path, set()).update(os.sched_getaffinity(0))

    class ProbeSource(Source):
        def compute(self, ctx: hs.ExecutionContext) -> None:
            seen.setdefault(ctx.operator_path, set()).update(os.sched_getaffinity(0))
            self.out.emit(1)

    g = hs.Graph("placed", default_clock=hs.RealtimeClock)
    source, sink = g.op(ProbeSource, "source"), g.op(Probe, "sink")
    g.add_flow(source.out, sink.inp, queue_policy=hs.QueuePolicy.DROP_OLDEST, queue_depth=1)
    lanes = Lanes(cpus).add("ingest", source).add("analytics", sink)
    lanes.apply(g)
    with hs.run_async(hs.compile(g, lanes.compile_options())) as session:
        session.wait(0.2)
        session.request_stop()
        session.wait(5)
    assert seen == {"source": set(cpus.ingest), "sink": set(cpus.helper)}
