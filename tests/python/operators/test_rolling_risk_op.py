# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``RollingRiskOp`` under the operator harness: the newest of several queued windows is the
one analyzed (J9), ``validate`` exists only with ``cross_check``, Params are checked at
authoring, late poll ticks are counted, and on the GPU the update matches the engine with
and without host-variance ranking (from 4.3 ``test_rolling_risk_op_matches_the_engine``).

The slot, interval, gate and process-executor rules are tested on ``RollingRisk`` itself
(``library/test_rolling_risk.py``): clocked operators stop being activated once staged
harness input ends (J6)."""

from __future__ import annotations

import time

import holoscan as hs
import numpy as np
import pytest
from holoscan.diagnostics import DiagnosticSeverity
from holoscan.testing import OperatorTestHarness, TestResult, TestRunOptions

from holoscan_finance.analytics.rolling import RollingRisk
from holoscan_finance.control.gate import CrossCheckGate
from holoscan_finance.marketdata.covariance import CovarianceEngine
from holoscan_finance.operators import RollingRiskOp
from holoscan_finance.operators import rolling_risk as rolling_risk_module
from holoscan_finance.payloads import BarWindow
from holoscan_finance.ref import Ref
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
