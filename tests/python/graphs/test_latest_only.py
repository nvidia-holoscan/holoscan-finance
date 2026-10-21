# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Slow analytics skip windows without ever blocking the bars.

The real ``QuoteBarsOp(cpu)`` in lane ``ingest`` closes one bar per batch and feeds a 30 ms
clocked consumer in lane ``analytics`` over the production ``LATEST`` edge. The bars must
never be paced by the consumer, and the consumer must end on the newest window, with gaps.
The feed never ends: a finite source would end the clocked consumer's activations."""

from __future__ import annotations

import time

import holoscan as hs
import numpy as np
import pytest

from holoscan_finance.marketdata.records import EVENT_DTYPE, QUOTE, ReplayBatch
from holoscan_finance.nyq.manifest import validate_manifest
from holoscan_finance.nyq.mapping import PROTOCOL, TIMEZONE, day_start_ns
from holoscan_finance.operators import QuoteBarsOp
from holoscan_finance.operators.edges import LATEST, lossless
from holoscan_finance.ref import Knob, Ref
from holoscan_finance.settings import BarSettings
from holoscan_finance.telemetry.stats import BarsStats

pytestmark = pytest.mark.sdk

N, SYMBOLS = 40, 4
SETTINGS = BarSettings(bar_seconds=1, window_seconds=10, stale_seconds=5, min_bars=2)
MIDNIGHT = day_start_ns("2025-04-01")


def directory() -> Knob:
    return Knob(
        validate_manifest(
            dict(
                format="nyq-equity-replay-v1",
                wire_protocol=PROTOCOL,
                market_timezone=TIMEZONE,
                sender_port=40000,
                symbols=[f"S{i}" for i in range(SYMBOLS)],
                generation=1,
                date="2025-04-01",
                start_ns=MIDNIGHT,
                end_ns=MIDNIGHT + 86_400 * 10**9,
                events=1,
                profile=[],
            )
        )
    )


def bar_batches(n: int) -> list[list[ReplayBatch]]:
    """Batch i quotes every symbol in second i of the session, so each batch closes one bar."""
    mids = 100 + np.random.default_rng(1).normal(0, 0.1, (n, SYMBOLS)).cumsum(axis=0)
    out = []
    for i in range(n):
        ev = np.zeros(SYMBOLS, EVENT_DTYPE)
        ev["timestamp_ns"] = MIDNIGHT + (34_200 + i) * 10**9 + np.arange(SYMBOLS)
        ev["symbol_id"] = np.arange(SYMBOLS)
        ev["kind"] = QUOTE
        ev["bid"], ev["ask"] = mids[i] - 0.01, mids[i] + 0.01
        ev["bid_size"] = ev["ask_size"] = 100
        ev["price"] = ev["volume"] = np.nan
        out.append([ReplayBatch(ev, 1, i + 1, 1 + i * SYMBOLS)])
    return out


class BarFeed(hs.Operator):
    """Emits one staged batch per 1 ms tick, then keeps ticking without emitting."""

    out = hs.Output("out", hs.pyobject)
    batches = hs.Param(Ref())
    sent = hs.Param(Ref())  # list: monotonic emit times

    def __init__(self) -> None:
        self._next = 0

    def contract(self) -> hs.Contract:
        return hs.Contract().trigger(hs.OnClock(period=hs.time.ms(1)))

    def compute(self, ctx: hs.ExecutionContext) -> None:
        if self._next < len(self.batches.obj):
            self.out.emit(self.batches.obj[self._next])
            self._next += 1
            self.sent.obj.append(time.monotonic_ns())


class SlowRisk(hs.Operator):
    """Stand-in for risk: the RollingRiskOp contract, 30 ms of work per window it takes."""

    window = hs.Input("window", hs.pyobject)
    seen = hs.Param(Ref())  # list: (window taken, newest window emitted by then)
    bars = hs.Param(Ref())  # the producer's BarsStats

    def contract(self) -> hs.Contract:
        return (
            hs.Contract()
            .trigger(hs.OnClock(period=hs.time.ms(1)))
            .input(self.window, hs.DrainAll())
        )

    def compute(self, ctx: hs.ExecutionContext) -> None:
        batch = self.window.selected_batch()
        if len(batch):
            newest = batch[len(batch) - 1].data
            self.seen.obj.append((newest.completed_bars, self.bars.obj.last_window_bars))
            time.sleep(0.03)


def test_slow_risk_skips_windows_without_blocking_bars():
    sent, seen, stats = [], [], BarsStats()
    g = hs.Graph("latest_only", default_clock=hs.RealtimeClock)
    feed = g.op(BarFeed, "source", batches=Ref(bar_batches(N)), sent=Ref(sent))
    bars = g.op(
        QuoteBarsOp,
        "bars",
        backend="cpu",
        **SETTINGS.as_params(),
        directory=Ref(directory()),
        stats=Ref(stats),
    )
    slow = g.op(SlowRisk, "risk", seen=Ref(seen), bars=Ref(stats))
    g.add_flow(feed.out, bars.events, **lossless(16))
    g.add_flow(bars.window, slow.window, **LATEST)  # bars never wait for analytics
    g.partition("ingest").add(feed, bars)
    g.partition("analytics").add(slow)
    split = hs.execution.SerializationDomainSplitPolicy.ALLOW
    plan = hs.compile(g, hs.CompileOptions(serialization_domain_splits=split))
    assert plan.ok, plan.summary()
    with hs.run_async(plan) as session:
        deadline = time.monotonic() + 30
        while len(sent) < N and time.monotonic() < deadline and not session.wait(0.01):
            pass
        session.wait(0.1)  # longer than one 30 ms analysis: the newest window gets picked up
        session.request_stop()
        session.wait(10)
    assert session.termination is hs.RunTermination.STOPPED, session.summary()
    assert [d.token for d in session.diagnostics] == []  # no BackpressuredError anywhere
    c = stats.counters
    assert stats.batches_in == N and c.completed_bars == N - 1
    bars_span_s = (c.last_batch_ns - c.first_batch_ns) / 1e9
    assert bars_span_s < 0.5 * N * 0.03  # never paced by the 30 ms consumer
    taken = [w for w, _ in seen]
    assert taken[-1] == stats.last_window_bars  # the newest window, though it came last
    assert all(w <= newest for w, newest in seen)
    assert len(taken) < stats.windows_emitted and taken == sorted(set(taken))  # skipped, in order
