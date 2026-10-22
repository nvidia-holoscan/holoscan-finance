# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``RollingRiskOp`` under the operator harness: the newest of several queued windows is the
one analyzed, ``validate`` exists only with ``cross_check``, Params are checked at
authoring, a CPU worker is tuned in ``start`` for a universe registered before the run,
late poll ticks are counted, and on the GPU the update matches the engine with and without
host-variance ranking.

The slot, interval, gate and process-executor rules are tested on ``RollingRisk`` itself
(``library/test_rolling_risk.py``): clocked operators stop being activated once staged
harness input ends."""

from __future__ import annotations

import time
from types import SimpleNamespace

import holoscan as hs
import numpy as np
import pytest
from holoscan.diagnostics import DiagnosticSeverity
from holoscan.testing import OperatorTestHarness, TestResult, TestRunOptions

from holoscan_finance.analytics import workers
from holoscan_finance.analytics.rolling import RollingRisk
from holoscan_finance.control.gate import CrossCheckGate
from holoscan_finance.marketdata.covariance import CovarianceEngine
from holoscan_finance.operators import CallbackSinkOp, RollingRiskOp
from holoscan_finance.operators import rolling_risk as rolling_risk_module
from holoscan_finance.operators.edges import LATEST, lossless
from holoscan_finance.payloads import BarWindow
from holoscan_finance.ref import Knob, Ref
from holoscan_finance.telemetry.stats import RiskStats

pytestmark = pytest.mark.sdk
pytest.importorskip("scipy")

ROWS = np.random.default_rng(3).normal(0, 0.001, (70, 40))
ROWS[5:9, 2] = np.nan
ROWS[:, 7] = 0


def windows(count: int, *, device: bool = False) -> list[BarWindow]:
    xp = pytest.importorskip("cupy") if device else np
    return [
        BarWindow(
            tuple(xp.asarray(r) for r in ROWS[i : i + 60]),
            np.arange(i, i + 60),
            1,
            60 + i,
            0,
            device=device,
        )
        for i in range(count)
    ]


def run_until_stopped(params: dict, staged: list, *, outputs: int = 1, **flow) -> TestResult:
    """Stage ``staged`` on ``window``, run until ``outputs`` updates (at most 2 s) and 100 ms
    more, then stop: a clocked operator never finishes on its own."""
    with OperatorTestHarness(RollingRiskOp, params=params) as h:
        if flow:
            h.set_input_flow("window", **flow)
        h.push_many("window", staged)
        with h.run_async(TestRunOptions(expect_stop=True)) as running:
            running.wait_for_output("risk", outputs, timeout_ms=2000)
            time.sleep(0.1)
            running.request_stop()
            return running.finish()


def test_the_newest_of_queued_windows_is_analyzed():
    stats, staged = RiskStats(), windows(5)
    params = dict(backend="cpu", poll_period_us=20_000, min_interval_us=1000, stats=Ref(stats))
    flow = dict(queue_policy=hs.QueuePolicy.DROP_OLDEST, queue_depth=1)
    updates = run_until_stopped(params, staged, **flow).values("risk")
    assert updates and updates[-1].window is staged[-1]
    assert stats.last_window_key == (1, staged[-1].completed_bars)
    assert stats.analyzed == len(updates)
    assert stats.lane_cpus and not updates[-1].validated and updates[-1].backend == "cpu"


def test_validate_is_declared_only_with_cross_check():
    g = hs.Graph("risk", default_clock=hs.RealtimeClock)
    assert sorted(g.op(RollingRiskOp, "plain").ports) == ["risk", "window"]
    checked = g.op(RollingRiskOp, "checked", cross_check=True, gate=Ref(CrossCheckGate()))
    assert sorted(checked.ports) == ["risk", "validate", "window"]


@pytest.mark.parametrize(
    ("params", "message"),
    [
        (dict(cross_check=True), "cross_check requires a gate"),
        (dict(cross_check=True, gate=Ref(CrossCheckGate()), backend="cpu"), "backend='gpu'"),
        (dict(host_variance=True, executor="process"), "executor='inline'"),
        (dict(worker_cpus=["x"]), "worker_cpus"),
        (dict(worker_cpus=[-1]), "worker_cpus"),
    ],
)
def test_invalid_params_are_rejected_at_authoring(params, message):
    g = hs.Graph("risk", default_clock=hs.RealtimeClock)
    with pytest.raises(hs.errors.HoloscanError, match=message):
        g.op(RollingRiskOp, "risk", **params)


def test_worker_cpus_reach_the_processor_sorted_and_unique(monkeypatch):
    seen = {}

    class Recording(RollingRisk):
        def __init__(self, **kwargs):
            seen.update(kwargs)
            super().__init__(**kwargs)

    monkeypatch.setattr(rolling_risk_module, "RollingRisk", Recording)
    run_until_stopped(dict(backend="cpu", worker_cpus=(1, 0, 1)), [], outputs=0)
    assert seen["worker_cpus"] == [0, 1] and seen["min_interval_ns"] == 50_000_000


class OneShotWindow(hs.Operator):
    """Emits ``item`` once (recording it in ``events``), then keeps ticking: a clocked
    operator fed by the harness stops being activated once its staged input ends, and the
    process executor collects a window at a later poll."""

    out = hs.Output("out", hs.pyobject)
    item = hs.Param(Ref())
    events = hs.Param(Ref())

    def __init__(self) -> None:
        self._sent = False

    def contract(self) -> hs.Contract:
        return hs.Contract().trigger(hs.OnClock(period=hs.time.ms(1)))

    def compute(self, ctx: hs.ExecutionContext) -> None:
        if not self._sent:
            self.events.obj.append(("window",))
            self.out.emit(self.item.obj)
            self._sent = True


def test_a_cpu_worker_is_tuned_in_start_for_a_universe_registered_before_the_run(monkeypatch):
    """Process + cpu: replay registers its universe before the graph runs, so ``start``
    tunes the worker and returns once the tuning is collected, before any operator computes
    (a tuning left to ``compute`` made the first window wait for it)."""
    events, updates = [], []
    collect = workers.UniverseTuning.collect

    def traced(self):
        pending = self.pending
        done = collect(self)
        if pending and done:
            events.append(("tuned", self.result["width"]))
        return done

    monkeypatch.setattr(workers.UniverseTuning, "collect", traced)
    (window,) = windows(1)
    knob = Knob(SimpleNamespace(universe=ROWS.shape[1]))
    g = hs.Graph("risk", default_clock=hs.RealtimeClock)
    source = g.op(OneShotWindow, "bars", item=Ref(window), events=Ref(events))
    risk = g.op(
        RollingRiskOp,
        "risk",
        backend="cpu",
        executor="process",
        min_interval_us=1000,
        directory=Ref(knob),
    )
    publish = g.op(CallbackSinkOp, "publish", callback=Ref(updates.append))
    g.add_flow(source.out, risk.window, **LATEST)
    g.add_flow(risk.risk, publish.items, **lossless(8))
    g.partition("bars").add(source)
    g.partition("analytics").add(risk, publish)
    split = hs.execution.SerializationDomainSplitPolicy.ALLOW
    plan = hs.compile(g, hs.CompileOptions(serialization_domain_splits=split))
    assert plan.ok, plan.summary()
    with hs.run_async(plan) as session:
        deadline = time.monotonic() + 60
        while not updates and time.monotonic() < deadline and not session.wait(0.01):
            pass
        session.request_stop()
        session.wait(10)
    assert session.termination is hs.RunTermination.STOPPED, session.summary()
    assert events[:2] == [("tuned", ROWS.shape[1]), ("window",)]
    (update,) = updates
    assert update.window is window and update.backend == "cpu"
    assert update.result["cpu_tunings"] == 1  # the window did not tune


def test_late_poll_ticks_are_counted_with_a_deadline():
    stats = RiskStats()
    result = run_until_stopped(dict(backend="cpu", deadline_us=1, stats=Ref(stats)), windows(1))
    assert result.ok and stats.late_activations > 0
    # Each late tick leaves a MAX_START_DELAY_MISS warning; none is an error.
    assert [d.token for d in result.diagnostics if d.severity == DiagnosticSeverity.ERROR] == []


@pytest.mark.gpu
@pytest.mark.parametrize("host_variance", [True, False])
def test_gpu_update_matches_the_engine(host_variance):
    (w,) = windows(1, device=True)
    result = run_until_stopped(dict(backend="gpu", host_variance=host_variance), [w])
    (update,) = result.values("risk")
    host = ROWS[:60]
    observed = np.var(host, axis=0, ddof=1) if host_variance else None
    want = CovarianceEngine("gpu").update(host, w.times, observed_variance=observed)
    assert update.window is w and update.backend == "gpu" and not update.validated
    if host_variance:
        np.testing.assert_array_equal(update.rows_host, host)
    else:
        assert update.rows_host is None
    for key in ("method", "added_bars", "expired_bars", "eligible_indices", "heatmap_indices"):
        assert update.result[key] == want[key], key
    np.testing.assert_allclose(update.result["annual_vol"], want["annual_vol"], rtol=1e-10)
    assert update.result["gpu_name"] and update.result["finished_wall_ns"] > 0
