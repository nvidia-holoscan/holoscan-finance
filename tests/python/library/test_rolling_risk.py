# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``RollingRisk`` on the CPU with an injected clock: the newest-window slot, the minimum
interval, the cross-check gate, validated host copies, one engine per generation, and the
spawned-worker executor.

Clocked ``RollingRiskOp`` activations stop at the end of staged harness input, so these
rules are tested here on the processor rather than through the operator."""

from __future__ import annotations

import os
import signal
import time

import numpy as np
import pytest
from fakes.stores import GateFake, ResultsStoreFake

from holoscan_finance.analytics import rolling, workers
from holoscan_finance.analytics.rolling import RollingRisk
from holoscan_finance.control.gate import CrossCheckGate
from holoscan_finance.marketdata.covariance import CovarianceEngine
from holoscan_finance.payloads import BarWindow
from holoscan_finance.telemetry.stats import RiskStats

pytest.importorskip("scipy")

MS = 1_000_000
ROWS = np.random.default_rng(11).normal(0, 0.001, (40, 9))
ROWS[3:6, 2] = np.nan
ROWS[:, 5] = 0


class Clock:
    def __init__(self) -> None:
        self.ns = 0

    def __call__(self) -> int:
        return self.ns


def window(end: int, *, generation: int = 1, size: int = 20, rows: np.ndarray = ROWS) -> BarWindow:
    """Bars ``end - size .. end - 1`` of ``rows``; bar k closes at k * 5 s."""
    times = np.arange(end - size, end, dtype=np.int64) * 5_000_000_000
    times.flags.writeable = False
    return BarWindow(tuple(rows[end - size : end]), times, generation, end, 1000 + end)


def risk(clock: Clock | None = None, **kwargs) -> RollingRisk:
    """A CPU inline risk with the default 50 ms minimum interval on a fake clock."""
    return RollingRisk(backend="cpu", clock=clock or Clock(), stats=RiskStats(), **kwargs)


def test_the_slot_keeps_only_the_newest_window():
    r = risk()
    r.offer(window(20))
    r.offer(window(21))
    update = r.poll()
    assert update.window.completed_bars == 21 and update.backend == "cpu"
    assert (r.stats.windows_received, r.stats.windows_superseded, r.stats.analyzed) == (2, 1, 1)
    assert r.poll() is None  # the slot is empty: nothing is analyzed twice


def test_the_minimum_interval_is_respected_and_the_newest_window_then_runs():
    clock = Clock()
    r = risk(clock)
    r.offer(window(20))
    assert r.poll() is not None
    clock.ns = 10 * MS
    r.offer(window(21))
    r.offer(window(22))
    assert r.poll() is None
    clock.ns = 49 * MS
    assert r.poll() is None
    clock.ns = 50 * MS
    update = r.poll()
    assert update.window.completed_bars == 22
    assert r.stats.last_window_key == (1, 22) and r.stats.last_completed_bars == 22


def test_the_gate_blocks_until_the_comparison_is_published():
    clock, gate = Clock(), GateFake()
    store = ResultsStoreFake(gate)
    r = risk(clock, cross_check=True, gate=gate)
    r.offer(window(20))
    first = r.poll()
    assert first.validated and gate.in_flight and gate.calls == ["ready", "try_claim"]
    clock.ns = 100 * MS
    r.offer(window(21))
    assert r.poll() is None and r.poll() is None
    assert r.stats.gate_waits == 2 and r.stats.analyzed == 1
    assert "release" not in gate.calls  # only the publication of the comparison releases it
    store.on_comparison(object())
    second = r.poll()
    assert second.validated and second.window.completed_bars == 21 and gate.claims == 2


def test_a_validated_window_carries_its_rows_and_ranks_by_its_own_variance():
    """As the CPU comparison does: no supplied variance, so both time the same work."""
    r = risk(cross_check=True, gate=CrossCheckGate())
    w = window(20)
    r.offer(w)
    update = r.poll()
    host = np.stack(w.parts)
    np.testing.assert_array_equal(update.rows_host, host)
    want = CovarianceEngine().update(host, w.times)
    for key in ("method", "eligible_indices", "heatmap_indices", "bars", "universe"):
        assert update.result[key] == want[key], key
    np.testing.assert_allclose(update.result["annual_vol"], want["annual_vol"], rtol=1e-12)
    assert update.result["finished_wall_ns"] > 0 and "gpu_name" not in update.result


def test_validation_not_allowed_runs_unvalidated():
    gate = GateFake(allowed=lambda: False)
    r = risk(cross_check=True, gate=gate)
    r.offer(window(20))
    update = r.poll()
    assert not update.validated and update.rows_host is None and gate.claims == 0


def test_host_variance_without_validation_keeps_the_host_copy():
    r = risk(host_variance=True)
    r.offer(window(20))
    update = r.poll()
    assert not update.validated and update.rows_host is not None


def test_one_engine_is_reset_per_generation_and_allocates_once_per_universe(monkeypatch):
    built, updates = [], []

    class Counting(CovarianceEngine):
        def __init__(self, *args, **kwargs):
            built.append(args)
            super().__init__(*args, **kwargs)

        def update(self, *args, **kwargs):
            updates.append(self.width)
            return super().update(*args, **kwargs)

    monkeypatch.setattr(rolling, "CovarianceEngine", Counting)
    clock = Clock()
    r = risk(clock)
    methods = []
    reseeded = np.random.default_rng(12).normal(0, 0.001, ROWS.shape)
    for generation, end, rows in ((1, 20, ROWS), (1, 21, ROWS), (2, 21, reseeded)):
        clock.ns += 100 * MS
        r.offer(window(end, generation=generation, rows=rows))
        methods.append(r.poll().result["method"])
    # Generation 2 reuses generation 1's bar times with other returns: only a reset engine
    # accepts that (without it the engine would raise "completed bars changed").
    assert methods == ["rebuild", "incremental", "rebuild"]
    assert len(built) == 1
    # The first window of the universe runs once untimed to allocate; later ones once.
    assert updates == [None, 9, 9, 9]
    assert r.stats.last_generation == 2


def test_stats_time_the_analysis_with_the_injected_clock():
    clock = Clock()

    def ticking() -> int:
        clock.ns += 3 * MS
        return clock.ns

    r = RollingRisk(backend="cpu", clock=ticking, stats=RiskStats())
    r.offer(window(20))
    r.poll()
    assert r.stats.last_ms == r.stats.max_ms == 3.0


def test_invalid_configurations_are_rejected():
    with pytest.raises(ValueError, match="CrossCheckGate"):
        RollingRisk(cross_check=True)
    with pytest.raises(ValueError, match="inline"):
        RollingRisk(executor="process", host_variance=True)
    with pytest.raises(ValueError, match="backend"):
        RollingRisk(backend="tpu")


def _wait_for_update(r: RollingRisk, timeout_s: float = 60.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if (update := r.poll()) is not None:
            return update
        time.sleep(0.002)
    raise AssertionError("no analysis finished")


def test_process_executor_round_trip():
    pytest.importorskip("threadpoolctl")  # cpu_window imports it in the worker
    gate = CrossCheckGate()
    r = RollingRisk(
        backend="cpu",
        executor="process",
        cpu_threads=1,
        cross_check=True,
        gate=gate,
        stats=RiskStats(),
    )
    r.start()
    try:
        w = window(20)
        r.offer(w)
        assert r.poll() is None and r.stats.busy  # submitted
        update = _wait_for_update(r)
        assert not r.stats.busy and update.window is w and update.validated
        np.testing.assert_array_equal(update.rows_host, np.stack(w.parts))
        assert update.result["threads"] == 1 and update.result["method"] == "rebuild"
        assert r.stats.last_window_key == (1, 20) and r.stats.analyzed == 1
    finally:
        r.stop()


def test_process_executor_failure_records_the_error_and_stops_submitting():
    pytest.importorskip("threadpoolctl")  # else the worker fails on the import, not the window
    gate = CrossCheckGate()
    r = RollingRisk(backend="cpu", executor="process", cpu_threads=1, cross_check=True, gate=gate)
    r.start()
    try:
        bad = BarWindow(tuple(ROWS[:3]), np.array([3, 2, 1], np.int64), 1, 3, 0)
        r.offer(bad)
        r.poll()
        deadline = time.monotonic() + 60
        while r.stats.error is None and time.monotonic() < deadline:
            r.poll()
            time.sleep(0.002)
        assert r.stats.error.startswith("Analytics failed: ")
        assert not gate.in_flight  # the void claim is released
        r.offer(window(20))
        assert r.poll() is None and not r.stats.busy and r.stats.analyzed == 0
    finally:
        r.stop()


def test_a_worker_that_died_while_idle_is_recorded_like_a_failed_job():
    """The next submit raises BrokenProcessPool; it must not escape ``poll`` (and fail the
    run) but void the claim and stop analyzing, as a failed job does."""
    pytest.importorskip("threadpoolctl")
    gate = CrossCheckGate()
    r = RollingRisk(backend="cpu", executor="process", cpu_threads=1, cross_check=True, gate=gate)
    r.start()
    try:
        os.kill(r._pool.submit(workers.ping).result(timeout=60), signal.SIGKILL)
        time.sleep(0.5)  # the pool's manager thread marks it broken
        r.offer(window(20))
        deadline = time.monotonic() + 60
        while r.stats.error is None and time.monotonic() < deadline:
            r.poll()
            time.sleep(0.002)
        assert r.stats.error.startswith("Analytics failed: ")
        assert not gate.in_flight and not r.stats.busy and r.stats.analyzed == 0
    finally:
        r.stop()


def test_a_window_the_worker_lost_is_resent_in_full_and_published():
    """WindowUnknown (a worker that lost its window) resends the window in full; a newer
    window waiting meanwhile is analyzed after it, never in its place."""
    pytest.importorskip("threadpoolctl")
    r = RollingRisk(backend="cpu", executor="process", cpu_threads=1, min_interval_ns=0)
    r.start()
    try:
        r.offer(window(20))
        _wait_for_update(r)
        r._pool.submit(workers.reset).result(timeout=60)  # the worker forgets its window
        r.offer(window(21))
        assert r.poll() is None  # submitted: the new bar only
        while not r._future.done():
            time.sleep(0.002)
        r.offer(window(22))  # waits in the slot
        assert r.poll() is None  # WindowUnknown: 21 resent in full, 22 still waiting
        first = _wait_for_update(r)
        second = _wait_for_update(r)
        assert [first.window.completed_bars, second.window.completed_bars] == [21, 22]
        assert first.result["method"] == "rebuild" and second.result["method"] == "incremental"
        assert r.stats.error is None
    finally:
        r.stop()
