# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``RiskCrossCheckOp``: a validated GPU update comes back as a ``RiskComparison`` whose CPU
side is the spawned ``cpu_window`` result and which agrees with the GPU side; a universe
registered before the run (replay) is tuned in ``start``, before any window.

The comparison is collected at a later poll than the one that received the update, and a
clocked operator is not activated again once its staged harness input ends, so this
runs the production wiring in a small graph whose test source never ends."""

from __future__ import annotations

import time
from types import SimpleNamespace

import holoscan as hs
import numpy as np
import pytest
from fakes.stores import ResultsStoreFake

from holoscan_finance.analytics import workers
from holoscan_finance.analytics.workers import cpu_window, start_worker_pool
from holoscan_finance.control.gate import CrossCheckGate
from holoscan_finance.marketdata.covariance import CovarianceEngine
from holoscan_finance.operators import CallbackSinkOp, RiskCrossCheckOp
from holoscan_finance.operators.edges import lossless
from holoscan_finance.payloads import BarWindow, RiskUpdate
from holoscan_finance.ref import Knob, Ref
from holoscan_finance.telemetry.stats import RiskStats

pytestmark = pytest.mark.sdk
pytest.importorskip("scipy")
pytest.importorskip("threadpoolctl")


class OneShotSource(hs.Operator):
    """Emits ``item`` once, then keeps ticking: no end of stream reaches the cross-check.
    ``on_emit``: called as it emits."""

    out = hs.Output("out", hs.pyobject)
    item = hs.Param(Ref())
    on_emit = hs.Param(Ref())

    def __init__(self) -> None:
        self._sent = False

    def contract(self) -> hs.Contract:
        return hs.Contract().trigger(hs.OnClock(period=hs.time.ms(1)))

    def compute(self, ctx: hs.ExecutionContext) -> None:
        if not self._sent:
            if self.on_emit:
                self.on_emit.obj()
            self.out.emit(self.item.obj)
            self._sent = True


def validated_update() -> RiskUpdate:
    rows = np.random.default_rng(21).normal(0, 0.001, (20, 16))
    rows[3:5, 4] = np.nan
    times = np.arange(20, dtype=np.int64) * 5_000_000_000
    observed = np.var(rows, axis=0, ddof=1)
    result = CovarianceEngine().update(rows, times, observed_variance=observed)  # the GPU side
    return RiskUpdate(BarWindow(tuple(rows), times, 1, 20, 0), result, rows, "gpu", True)


def run_crosscheck(update: RiskUpdate, gate: CrossCheckGate, **params) -> ResultsStoreFake:
    """The update from a source in another lane through ``RiskCrossCheckOp`` (``params``)
    to the publishing sink, until its comparison is published; ``on_emit`` goes to the
    source."""
    store = ResultsStoreFake(gate)
    g = hs.Graph("crosscheck", default_clock=hs.RealtimeClock)
    source = g.op(OneShotSource, "risk", item=Ref(update), on_emit=params.pop("on_emit", Ref()))
    crosscheck = g.op(RiskCrossCheckOp, "crosscheck", **params)
    publish = g.op(CallbackSinkOp, "publish_comparison", callback=Ref(store.on_comparison))
    g.add_flow(source.out, crosscheck.risk, **lossless(2))
    g.add_flow(crosscheck.comparison, publish.items, **lossless(2))
    g.partition("analytics").add(source)
    g.partition("crosscheck").add(crosscheck, publish)
    split = hs.execution.SerializationDomainSplitPolicy.ALLOW
    plan = hs.compile(g, hs.CompileOptions(serialization_domain_splits=split))
    assert plan.ok, plan.summary()
    with hs.run_async(plan) as session:
        deadline = time.monotonic() + 60
        while not store.comparisons and time.monotonic() < deadline and not session.wait(0.01):
            pass
        session.request_stop()
        session.wait(10)
    assert session.termination is hs.RunTermination.STOPPED, session.summary()
    return store


def test_a_validated_update_returns_an_agreeing_comparison():
    update, stats, gate = validated_update(), RiskStats(), CrossCheckGate()
    assert gate.try_claim()  # as risk did when it validated the window
    store = run_crosscheck(update, gate, cpu_threads=1, full_window_bars=20, stats=Ref(stats))
    (comparison,) = store.comparisons
    assert comparison.update is update and comparison.error is None
    assert comparison.agreement and comparison.matched_work and comparison.usable
    assert not gate.in_flight  # released after publication
    with start_worker_pool() as pool:
        direct = pool.submit(cpu_window, update.rows_host, update.window.times, 1, 5, 1).result(
            timeout=60
        )
    np.testing.assert_array_equal(comparison.cpu["annual_vol"], direct["annual_vol"])
    assert (
        comparison.cpu["eligible_indices"]
        == direct["eligible_indices"]
        == update.result["eligible_indices"]
    )
    assert (stats.crosschecks_submitted, stats.crosschecks_done) == (1, 1)


def test_a_universe_registered_before_the_run_is_tuned_in_start(monkeypatch):
    """Replay registers its universe before the graph runs and starts its drain with the
    source, so a tuning left to ``compute`` ran beside the first windows: the first
    validated one waited for it, and with it the next GPU result (0.6 s at 2,886 symbols).
    ``start`` tunes such a universe and returns once the tuning is collected, before any
    operator computes; the window then computes without tuning."""
    events = []
    collect = workers.UniverseTuning.collect

    def traced(self):
        pending = self.pending
        done = collect(self)
        if pending and done:
            events.append(("tuned", self.result["width"]))
        return done

    monkeypatch.setattr(workers.UniverseTuning, "collect", traced)
    update, gate = validated_update(), CrossCheckGate()
    assert gate.try_claim()
    store = run_crosscheck(
        update,
        gate,
        full_window_bars=20,
        directory=Ref(Knob(SimpleNamespace(universe=16))),
        on_emit=Ref(lambda: events.append(("window",))),
    )
    assert events[:2] == [("tuned", 16), ("window",)]
    (comparison,) = store.comparisons
    assert comparison.error is None and comparison.agreement and comparison.matched_work
    assert comparison.cpu["cpu_tunings"] == 1  # the window did not tune


def test_worker_cpus_are_checked_at_authoring():
    g = hs.Graph("crosscheck", default_clock=hs.RealtimeClock)
    with pytest.raises(hs.errors.HoloscanError, match="worker_cpus"):
        g.op(RiskCrossCheckOp, "crosscheck", worker_cpus=[True])
