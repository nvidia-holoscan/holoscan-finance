# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The lane rule, for whoever edits a composition (one entry per lane, plus convexity).

An authored lane has exactly one entry operator: a clock-triggered one, or one fed only from
other lanes. Every other member needs an input edge from inside the lane (more inputs from
other lanes are fine). A layout whose path leaves a lane and comes back compiles while every
operator is authored, but is rejected as soon as anything is partitioned automatically.
"""

from __future__ import annotations

from datetime import timedelta

import holoscan as hs
import pytest

pytestmark = pytest.mark.sdk

ALLOW = hs.CompileOptions(
    serialization_domain_splits=hs.execution.SerializationDomainSplitPolicy.ALLOW
)
BLOCKING = {"queue_policy": hs.QueuePolicy.BLOCKING, "queue_depth": 8}
LATEST = {"queue_policy": hs.QueuePolicy.DROP_OLDEST, "queue_depth": 1}


class Source(hs.Operator):  # a source: OnClock, no inputs
    out = hs.Output("out", hs.pyobject)

    def contract(self) -> hs.Contract:
        return hs.Contract().trigger(hs.OnClock(period=timedelta(milliseconds=1)))

    def compute(self, ctx: hs.ExecutionContext) -> None:
        self.out.emit(1)


class Poll(hs.Operator):  # a clock-paced consumer: OnClock + DrainAll (risk, cross-check)
    inp = hs.Input("inp", hs.pyobject)
    out = hs.Output("out", hs.pyobject)
    side = hs.Output("side", hs.pyobject)

    def contract(self) -> hs.Contract:
        return (
            hs.Contract()
            .trigger(hs.OnClock(period=timedelta(milliseconds=2)))
            .input(self.inp, hs.DrainAll())
        )

    def compute(self, ctx: hs.ExecutionContext) -> None:
        batch = self.inp.selected_batch()
        if len(batch):
            self.out.emit(1)
            self.side.emit(1)


class Sink(hs.Operator):  # OnAnyInput over two inputs (a publisher fed by two stages)
    a = hs.Input("a", hs.pyobject)
    b = hs.Input("b", hs.pyobject)

    def compute(self, ctx: hs.ExecutionContext) -> None:
        for port in (self.a, self.b):
            port.receive()


def test_two_entry_operators_in_one_lane_are_rejected() -> None:
    """A source and a clock-paced consumer in one lane: each is an entry."""
    g = hs.Graph("two_entries", default_clock=hs.RealtimeClock)
    source, poll = g.op(Source, "source"), g.op(Poll, "poll")
    sink = g.op(Sink, "sink")
    g.add_flow(source.out, poll.inp, **LATEST)
    g.add_flow(poll.out, sink.a, **BLOCKING)
    g.add_flow(poll.side, sink.b, **BLOCKING)
    g.partition("one").add(source, poll, sink)
    plan = hs.compile(g, ALLOW)
    assert not plan.ok
    assert "PARTITION_COVER_INVALID" in [d.token for d in plan.errors]


def test_a_member_fed_from_inside_and_outside_its_lane_is_accepted() -> None:
    """The sink takes the poll's output (its own lane) and the source's (the ingest lane):
    the poll is the analytics lane's only entry."""
    g = hs.Graph("mixed_inputs", default_clock=hs.RealtimeClock)
    source, poll, sink = g.op(Source, "source"), g.op(Poll, "poll"), g.op(Sink, "sink")
    g.add_flow(source.out, poll.inp, **LATEST)
    g.add_flow(source.out, sink.b, **BLOCKING)
    g.add_flow(poll.out, sink.a, **BLOCKING)
    g.partition("ingest").add(source)
    g.partition("analytics").add(poll, sink)
    plan = hs.compile(g, ALLOW)
    assert plan.ok, plan.summary()
    lanes = {p.authored_name: sorted(p.operator_paths) for p in plan.partitions}
    assert lanes == {"ingest": ["source"], "analytics": ["poll", "sink"]}


def test_a_member_fed_only_from_another_lane_is_a_second_entry() -> None:
    """A sink fed only by the source cannot join the poll's lane; in a lane of its own it is
    that lane's entry."""

    def plan(*, own_lane: bool) -> hs.Plan:
        g = hs.Graph("outside_only", default_clock=hs.RealtimeClock)
        source, poll, lone = g.op(Source, "source"), g.op(Poll, "poll"), g.op(Sink, "lone")
        g.add_flow(source.out, poll.inp, **LATEST)
        g.add_flow(source.out, lone.a, **BLOCKING)
        g.add_flow(source.out, lone.b, **BLOCKING)
        g.partition("ingest").add(source)
        if own_lane:
            g.partition("analytics").add(poll)
            g.partition("tail").add(lone)
        else:
            g.partition("analytics").add(poll, lone)
        return hs.compile(g, ALLOW)

    assert plan(own_lane=True).ok
    rejected = plan(own_lane=False)
    assert not rejected.ok
    assert "PARTITION_COVER_INVALID" in [d.token for d in rejected.errors]


def non_convex(*, automatic: bool) -> hs.Plan:
    """analytics = {risk, publish}, crosscheck = {xc}: risk -> xc -> publish leaves the
    analytics lane and comes back (the layout the final design moved away from)."""
    g = hs.Graph("non_convex", default_clock=hs.RealtimeClock)
    source, risk = g.op(Source, "source"), g.op(Poll, "risk")
    xc, publish = g.op(Poll, "xc"), g.op(Sink, "publish")
    g.add_flow(source.out, risk.inp, **LATEST)
    g.add_flow(risk.out, publish.a, **BLOCKING)
    g.add_flow(risk.side, xc.inp, **BLOCKING)
    g.add_flow(xc.out, publish.b, **BLOCKING)
    g.partition("ingest").add(source)
    g.partition("analytics").add(risk, publish)
    g.partition("crosscheck").add(xc)
    if automatic:
        g.auto_partition_remaining()
    return hs.compile(g, ALLOW)


def test_a_non_convex_layout_breaks_once_anything_is_automatic() -> None:
    assert non_convex(automatic=False).ok  # it compiles, but only while everything is authored
    plan = non_convex(automatic=True)
    assert not plan.ok
    assert "PARTITION_ANCHOR_NON_CONVEX" in [d.token for d in plan.errors]
