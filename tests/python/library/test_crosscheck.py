# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``CpuCrossCheck``: validated updates are recomputed in FIFO order by a spawned
``cpu_window`` worker and compared with the GPU-side result; a worker exception becomes an
error comparison (which ``ResultsStore`` publishes and which releases the gate), and so does
a worker that died while idle. The worker tunes for a universe before its first window, so
no validated window waits for a tuning."""

from __future__ import annotations

import os
import signal
import time

import numpy as np
import pytest

from holoscan_finance.analytics import workers
from holoscan_finance.analytics.crosscheck import CpuCrossCheck
from holoscan_finance.analytics.workers import cpu_window, ping, start_worker_pool
from holoscan_finance.marketdata.covariance import CovarianceEngine
from holoscan_finance.payloads import BarWindow, RiskUpdate
from holoscan_finance.telemetry.stats import RiskStats

pytest.importorskip("scipy")
pytest.importorskip("threadpoolctl")

ROWS = np.random.default_rng(5).normal(0, 0.001, (30, 10))
ROWS[4:7, 1] = np.nan


def validated(*windows: tuple[int, int], size: int = 12) -> list[RiskUpdate]:
    """Validated updates of the windows ``(end, generation)``, in order; the GPU side is
    stood in for by one NumPy engine per generation, as risk keeps one."""
    engines: dict[int, CovarianceEngine] = {}
    updates = []
    for end, generation in windows:
        rows = ROWS[end - size : end]
        times = np.arange(end - size, end, dtype=np.int64) * 5_000_000_000
        observed = np.var(rows, axis=0, ddof=1)
        engine = engines.setdefault(generation, CovarianceEngine())
        result = engine.update(rows, times, observed_variance=observed)
        window = BarWindow(tuple(rows), times, generation, end, 0)
        updates.append(RiskUpdate(window, result, rows.copy(), "gpu", True))
    return updates


def drain(xc: CpuCrossCheck, count: int, timeout_s: float = 60.0) -> list:
    out, deadline = [], time.monotonic() + timeout_s
    while len(out) < count and time.monotonic() < deadline:
        if (comparison := xc.poll()) is not None:
            out.append(comparison)
        time.sleep(0.002)
    assert len(out) == count, f"{len(out)} of {count} comparisons"
    return out


@pytest.fixture
def xc():
    checker = CpuCrossCheck(cpu_threads=1, full_window_bars=12, stats=RiskStats())
    checker.start()
    yield checker
    checker.stop()


def test_comparisons_follow_submission_order_and_agree_with_a_direct_cpu_window(xc):
    updates = validated((12, 1), (15, 1), (12, 2))
    for update in updates:
        xc.submit(update)
    comparisons = drain(xc, 3)
    assert [c.update for c in comparisons] == updates
    assert all(c.agreement and c.matched_work and c.usable and c.error is None for c in comparisons)
    with start_worker_pool() as pool:  # the same worker function in a fresh process
        w = updates[0].window
        direct = pool.submit(cpu_window, updates[0].rows_host, w.times, 1, 5, 1).result(timeout=60)
    cpu = comparisons[0].cpu
    assert cpu["eligible_indices"] == direct["eligible_indices"]
    np.testing.assert_array_equal(cpu["annual_vol"], direct["annual_vol"])
    assert [c.cpu["method"] for c in comparisons] == ["rebuild", "incremental", "rebuild"]
    assert (xc.stats.crosschecks_submitted, xc.stats.crosschecks_done) == (3, 3)


def test_a_worker_exception_becomes_an_error_comparison(xc):
    (good,) = validated((12, 1))
    bad = RiskUpdate(
        BarWindow(tuple(ROWS[:3]), np.array([3, 2, 1], np.int64), 1, 3, 0),
        good.result,
        ROWS[:3],
        "gpu",
        True,
    )
    xc.submit(bad)
    xc.submit(good)
    failed, ok = drain(xc, 2)
    assert failed.update is bad and failed.cpu is None and failed.error.startswith("ValueError")
    assert not (failed.agreement or failed.matched_work or failed.usable)
    assert ok.update is good and ok.agreement  # the worker survives a failed job


def test_a_worker_that_died_while_idle_becomes_an_error_comparison(xc):
    """The next submit raises BrokenProcessPool; it must not escape ``poll`` (and fail the
    run): the update is published as an error comparison, which releases the gate."""
    os.kill(xc._pool.submit(ping).result(timeout=60), signal.SIGKILL)
    time.sleep(0.5)  # the pool's manager thread marks it broken
    (update,) = validated((12, 1))
    xc.submit(update)
    (failed,) = drain(xc, 1)
    assert failed.update is update and failed.cpu is None
    assert failed.error.startswith("BrokenProcessPool")
    assert (xc.stats.crosschecks_submitted, xc.stats.crosschecks_done) == (1, 1)


def test_an_unvalidated_update_is_refused():
    (update,) = validated((12, 1))
    with pytest.raises(ValueError, match="rows_host"):
        CpuCrossCheck().submit(RiskUpdate(update.window, update.result))


def _sent_rows(xc: CpuCrossCheck) -> list[int]:
    """Records the number of rows each job sends the worker."""
    sent, submit = [], xc._pool.submit

    def recording(fn, *args, **kwargs):
        if fn is workers.cpu_window:
            sent.append(len(args[0]))
        return submit(fn, *args, **kwargs)

    xc._pool.submit = recording
    return sent


def test_a_sliding_window_travels_as_its_new_bars(xc):
    """As RollingRisk sends its own worker: only the bars a validated window adds to the
    one sent last, the full window after a gap or in a new generation; the CPU engine then
    does the GPU's work (same method and bar counts) on every window."""
    sent = _sent_rows(xc)
    updates = validated((12, 1), (13, 1), (16, 1), (30, 1), (13, 2))
    for update in updates:
        xc.submit(update)
    comparisons = drain(xc, 5)
    assert sent == [12, 1, 3, 12, 12]
    assert all(c.agreement and c.matched_work for c in comparisons)
    assert [c.cpu["method"] for c in comparisons] == [
        "rebuild", "incremental", "incremental", "rebuild", "rebuild"
    ]  # fmt: skip


def test_a_worker_that_lost_its_window_gets_the_full_window(xc):
    """A worker without the window the new bars extend (a new worker process) answers
    WindowUnknown; the window is sent again in full and its comparison still comes."""
    sent = _sent_rows(xc)
    first, second = validated((12, 1), (13, 1))
    xc.submit(first)
    drain(xc, 1)
    xc._pool.submit(workers.reset).result(timeout=60)  # the worker forgets its window
    xc.submit(second)
    (comparison,) = drain(xc, 1)
    assert sent == [12, 1, 12]
    assert comparison.update is second and comparison.error is None and comparison.agreement
    assert (xc.stats.crosschecks_submitted, xc.stats.crosschecks_done) == (2, 2)


def test_the_worker_tunes_before_the_first_window_so_no_validated_window_waits_for_it(xc):
    """The universe reaches the idle worker as ``tune_cpu`` when its manifest arrives
    (``RiskCrossCheckOp`` calls ``prepare``), once per width and never while a window is in
    flight. Every validated window after it, the first full window included, is computed
    without tuning: the gate a validated window holds never waits for one (the re-tuning at
    the first full window delayed the next GPU result by about 0.35 s)."""
    (busy,) = validated((12, 1))
    xc.submit(busy)
    xc.poll()  # the job is in flight
    xc.prepare(10)
    assert not xc.tuning_pending  # not while the worker computes a window
    drain(xc, 1)
    xc.prepare(10)
    assert xc.tuning_pending
    deadline = time.monotonic() + 60
    while xc.tuning_pending and time.monotonic() < deadline:
        assert xc.poll() is None
        time.sleep(0.002)
    tuning = xc.tuning
    assert tuning["width"] == 10 and tuning["threads"] == 1  # the fixture's cpu_threads
    xc.prepare(10)
    assert not xc.tuning_pending  # once per width
    engine = CovarianceEngine()
    for end in range(6, 17):  # growth to the full window (12 bars), then slides, generation 2
        start = max(0, end - 12)
        rows = ROWS[start:end]
        times = np.arange(start, end, dtype=np.int64) * 5_000_000_000
        result = engine.update(rows, times, observed_variance=np.var(rows, axis=0, ddof=1))
        window = BarWindow(tuple(rows), times, 2, end, 0)
        xc.submit(RiskUpdate(window, result, rows.copy(), "gpu", True))
        (comparison,) = drain(xc, 1)
        assert comparison.agreement and comparison.matched_work, end
        assert comparison.cpu["cpu_tunings"] == tuning["cpu_tunings"], end  # none since
