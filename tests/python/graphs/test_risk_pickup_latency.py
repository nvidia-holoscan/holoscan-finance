# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Risk keeps 4.3's run-on-arrival latency: a window reaching ``RollingRiskOp`` over the
``LATEST`` edge is picked up by the 1 ms poll, not on a 50 ms grid (the 1 ms poll measured a
0.99 ms p50; a 50 ms clock added 28 ms). Pick-up delay = receive-to-result minus the engine
time, over at least 30 windows arriving every 100 ms at a random phase. Timing pass only."""

from __future__ import annotations

import time

import holoscan as hs
import numpy as np
import pytest
from fakes.stores import ResultsStoreFake

from holoscan_finance.operators import CallbackSinkOp, RollingRiskOp
from holoscan_finance.operators.edges import LATEST, lossless
from holoscan_finance.payloads import BarWindow
from holoscan_finance.ref import Ref

pytestmark = [pytest.mark.sdk, pytest.mark.slow, pytest.mark.timing]
pytest.importorskip("scipy")

WINDOWS = 32
ROWS = np.random.default_rng(2).normal(0, 0.001, (20 + WINDOWS, 8))


class WindowFeed(hs.Operator):
    """A tiny window every 100 ms plus up to 1 ms of jitter, stamped with its send time."""

    out = hs.Output("out", hs.pyobject)

    def __init__(self) -> None:
        self._rng = np.random.default_rng(9)
        self._due = self._sent = 0

    def contract(self) -> hs.Contract:
        return hs.Contract().trigger(hs.OnClock(period=hs.time.us(100)))

    def compute(self, ctx: hs.ExecutionContext) -> None:
        now = time.monotonic_ns()
        if self._sent == WINDOWS or now < self._due:
            return
        k = self._sent
        times = np.arange(k, k + 20, dtype=np.int64) * 5_000_000_000
        self.out.emit(BarWindow(tuple(ROWS[k : k + 20]), times, 1, k + 20, time.time_ns()))
        self._sent += 1
        self._due = now + 100_000_000 + int(self._rng.integers(0, 1_000_000))


def test_risk_picks_up_a_window_within_two_ms_at_p50():
    store = ResultsStoreFake()
    g = hs.Graph("pickup", default_clock=hs.RealtimeClock)
    feed = g.op(WindowFeed, "bars")
    risk = g.op(RollingRiskOp, "risk", backend="cpu", executor="inline", min_interval_us=50_000)
    publish = g.op(CallbackSinkOp, "publish", callback=Ref(store.on_risk))
    g.add_flow(feed.out, risk.window, **LATEST)
    g.add_flow(risk.risk, publish.items, **lossless(8))
    g.partition("ingest").add(feed)
    g.partition("analytics").add(risk, publish)
    split = hs.execution.SerializationDomainSplitPolicy.ALLOW
    plan = hs.compile(g, hs.CompileOptions(serialization_domain_splits=split))
    assert plan.ok, plan.summary()
    with hs.run_async(plan) as session:
        deadline = time.monotonic() + 30
        while (
            store.received_risk < WINDOWS and time.monotonic() < deadline and not session.wait(0.05)
        ):
            pass
        session.request_stop()
        session.wait(10)
    assert store.received_risk == WINDOWS, session.summary()
    # The first window of a generation also runs a throwaway engine: not a pick-up delay.
    delays = [
        (u.result["finished_wall_ns"] - u.window.recv_ns) / 1e6 - u.result["milliseconds"]
        for u in store.risk[1:]
    ]
    assert len(delays) >= 30 and float(np.median(delays)) <= 2.0, sorted(delays)
