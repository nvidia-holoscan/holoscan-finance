# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU spawn workers: the persistent engine across a roll and a seek, windows sent as their
new bars only, the BLAS tuning once per universe ahead of its first window, the pre-spawn
``ping``, the pinned worker reporting its CPU set, and the worker leaving SIGINT/SIGTERM to
the parent without outliving it."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from holoscan_finance.analytics import workers
from holoscan_finance.analytics.rolling import RollingRisk
from holoscan_finance.analytics.workers import WindowUnknown, cpu_window, ping, start_worker_pool
from holoscan_finance.control.gate import CrossCheckGate
from holoscan_finance.marketdata.covariance import CovarianceEngine, slide
from holoscan_finance.payloads import BarWindow
from holoscan_finance.telemetry.stats import RiskStats

pytest.importorskip("scipy")
pytest.importorskip("threadpoolctl")


def test_persistent_cpu_worker_matches_after_roll_and_seek():
    rows = np.random.default_rng(42).normal(0, 0.001, (30, 12))
    rows[2:4, 3] = np.nan
    with start_worker_pool() as pool:
        for generation, end, method in (
            (1, 10, "rebuild"),
            (1, 13, "incremental"),
            (2, 10, "rebuild"),
        ):
            sample, times = rows[end - 10 : end], np.arange(end - 10, end)
            result = pool.submit(cpu_window, sample, times, generation, 5, 1).result(timeout=60)
            assert result["method"] == method
            assert result["threads"] == 1
            if method == "incremental":
                assert result["added_bars"] == result["expired_bars"] == 3
            oracle = np.cov(sample[:, result["eligible_indices"]], rowvar=False)
            np.testing.assert_allclose(result["covariance_probes"], oracle, rtol=1e-10, atol=1e-16)
            assert result["finished_wall_ns"] > 0 and result["blas"]


def window_schedule():
    """``(generation, start, end)`` windows: ramp-up, slides of one and three bars
    (superseded windows), more calls than the periodic rebuild needs, a gap and a new
    generation."""
    out, end = [], 12
    while end < 400:
        out.append((1, max(0, end - 20), end))
        end += 3 if len(out) % 7 == 0 else 1
    return [*out, (1, 400, 420), (2, 0, 20), (2, 1, 21)]


def check_new_bars_only_match_full_windows(window_fn, backend="cpu", **kwargs):
    """A worker fed only the new bars of each sliding window (as ``RollingRisk`` sends
    them) holds the parent's window and returns what an engine fed full windows returns;
    after ``reset()`` new bars alone raise ``WindowUnknown``."""
    rows = np.random.default_rng(14).normal(0, 0.001, (430, 10))
    rows[50:53, 2] = np.nan
    workers.reset()
    previous, methods = None, []
    for generation, start, end in window_schedule():
        times = np.arange(start, end) * 5
        step = slide(previous[1], times) if previous and previous[0] == generation else None
        new = end - start if step is None else step[1]
        if previous is None or previous[0] != generation:
            reference = CovarianceEngine(backend)
        result = window_fn(rows[end - new : end], times, generation, 5, **kwargs)
        want = reference.update(rows[start:end], times)
        np.testing.assert_array_equal(np.stack(workers._windows[backend][2]), rows[start:end])
        for key in ("method", "added_bars", "expired_bars", "eligible_indices", "heatmap_indices"):
            assert result[key] == want[key], (generation, start, end, key)
        np.testing.assert_allclose(result["annual_vol"], want["annual_vol"], rtol=1e-10)
        methods.append(result["method"])
        previous = (generation, times)
    assert methods.count("rebuild") == 4  # the first window, the 256th call, the gap, gen 2
    workers.reset()
    with pytest.raises(WindowUnknown):
        window_fn(rows[21:22], np.arange(2, 22) * 5, 2, 5, **kwargs)


def test_a_cpu_worker_fed_new_bars_only_matches_full_windows():
    check_new_bars_only_match_full_windows(cpu_window, threads=1)


def test_a_worker_that_lost_its_window_gets_the_full_window():
    """The parent sends new bars only; a worker that does not hold the window they extend
    (a new worker process) answers ``WindowUnknown`` and the parent sends it all."""
    rows = np.random.default_rng(15).normal(0, 0.001, (40, 7))
    r = RollingRisk(
        backend="cpu",
        executor="process",
        cpu_threads=1,
        min_interval_ns=0,
        cross_check=True,
        gate=CrossCheckGate(),
        stats=RiskStats(),
    )
    r.start()
    try:
        results = []
        for end in (20, 21, 22, 23):
            if end == 22:
                r._pool.submit(workers.reset).result(timeout=60)  # the worker loses its window
            times = np.arange(end - 20, end, dtype=np.int64) * 5_000_000_000
            r.offer(BarWindow(tuple(rows[end - 20 : end]), times, 1, end, 0))
            deadline = time.monotonic() + 60
            while (update := r.poll()) is None and time.monotonic() < deadline:
                time.sleep(0.002)
                r._gate.release()
            results.append(update)
        assert [u.result["method"] for u in results] == [
            "rebuild", "incremental", "rebuild", "incremental"
        ]  # fmt: skip
        for u in results:
            want = CovarianceEngine().update(np.stack(u.window.parts), u.window.times)
            assert u.result["eligible_indices"] == want["eligible_indices"]
            assert isinstance(u.result["annual_vol"], list)
            np.testing.assert_allclose(u.result["annual_vol"], want["annual_vol"], rtol=1e-10)
            np.testing.assert_array_equal(u.rows_host, np.stack(u.window.parts))
        assert r.stats.error is None
    finally:
        r.stop()


def test_the_cpu_worker_tunes_once_per_universe_and_no_window_after_tune_cpu_tunes():
    """``tune_cpu`` (sent when the manifest announces the universe) tunes for its width ahead
    of the first window; no window of that universe tunes again, the first full window
    included (it used to re-tune there, inside a compare-mode job that holds the gate). A
    width it was not tuned for is tuned at its first window."""
    rows = np.random.default_rng(8).normal(0, 0.001, (40, 10))
    workers.reset()
    try:
        tuning = workers.tune_cpu(10, 12, 5, 0)
        assert tuning["width"] == 10 and tuning["cpu_tuned_bars"] == 12
        assert tuning["cpu_tunings"] == 1
        assert tuning["threads"] in tuning["cpu_thread_trials_ms"]
        for end in range(6, 30):  # growth to the full window, then slides
            start = max(0, end - 12)
            result = cpu_window(rows[start:end], np.arange(start, end), 1, 5, 0, 12)
            assert result["cpu_tunings"] == 1 and result["threads"] == tuning["threads"], end
        other = cpu_window(rows[:12, :7], np.arange(12), 2, 5, 0, 12)
        assert other["cpu_tunings"] == 2 and other["cpu_tuned_bars"] == 12
    finally:
        workers.reset()


def fake_blas(monkeypatch, milliseconds) -> tuple[list[int], list]:
    """Stand in for threadpoolctl and the engine: an engine built under a BLAS limit reports
    ``milliseconds(limit, engines built so far)`` per slide. Returns the limits engines
    were built under, in order (the tuning builds one per candidate and pass), and the
    latest limit set (``[limit]``)."""
    import threadpoolctl

    limit, built = [None], []

    class Limits:
        def __init__(self, limits, user_api):
            limit[0] = limits

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def restore_original_limits(self):
            pass

    class Engine:
        def __init__(self, **kwargs):
            built.append(limit[0])
            self.ms = milliseconds(limit[0], len(built))

        def update(self, rows, times):
            return {"milliseconds": self.ms}

    monkeypatch.setattr(threadpoolctl, "threadpool_limits", Limits)
    monkeypatch.setattr(threadpoolctl, "threadpool_info", lambda: [])
    monkeypatch.setattr(workers, "CovarianceEngine", Engine)
    monkeypatch.setattr(workers.os, "sched_getaffinity", lambda pid: set(range(12)))
    return built, limit


def test_the_tuning_keeps_the_fewest_threads_within_the_margin_of_the_fastest(monkeypatch):
    """Steady one-bar slides at each candidate; a thread count is worth it only when it is
    clearly faster: on a pipeline host, threads that win by noise in an idle tuning stall
    once other threads compete for the cores."""
    steady = {1: 1.00, 2: 0.95, 4: 0.50, 8: 0.48, 12: 0.47}  # ms per slide at each limit
    _, limit = fake_blas(monkeypatch, lambda nt, n: steady[nt])
    workers.reset()
    try:
        tuning = workers.tune_cpu(50, 60, 5, 0)
        assert tuning["cpu_thread_trials_ms"] == steady
        assert tuning["threads"] == 4 and limit[0] == 4  # 8 and 12 are within 10% of 4
        assert workers.TUNE_MARGIN == 0.10
    finally:
        workers.reset()


def test_the_tuning_keeps_the_times_of_a_second_pass(monkeypatch):
    """A process's first engine slides faster than every later one, the engine the worker
    keeps included, and the first candidate (1 thread) always got it: at 1,000 symbols its
    one trial read 0.70-0.82 ms against 1.19 ms steady, so the margin kept 1 thread where 4
    ran 1.8x faster. The first pass only warms up; the pick and the reported trials come
    from the second. A forced thread count picks nothing, so it is timed once."""
    steady = {1: 1.19, 2: 0.80, 4: 0.66, 8: 0.70, 12: 0.90}
    built, limit = fake_blas(monkeypatch, lambda nt, n: 0.70 if n == 1 else steady[nt])
    workers.reset()
    try:
        tuning = workers.tune_cpu(1000, 60, 5, 0)
        assert built == [1, 2, 4, 8, 12] * 2
        assert tuning["cpu_thread_trials_ms"] == steady
        assert tuning["threads"] == 4 and limit[0] == 4
        built.clear()
        assert workers.tune_cpu(1000, 60, 5, 2)["threads"] == 2 and built == [2]
    finally:
        workers.reset()


def test_a_cpu_risk_worker_is_tuned_for_the_universe_before_its_first_window():
    """``prepare`` (``RollingRiskOp`` calls it with the directory's universe) has the spawned
    worker tune while idle, once per width; windows wait for it, and none tunes again."""
    rows = np.random.default_rng(16).normal(0, 0.001, (30, 9))
    r = RollingRisk(
        backend="cpu", executor="process", min_interval_ns=0, full_window_bars=12, stats=RiskStats()
    )
    r.start()
    try:
        r.prepare(9)
        assert r.tuning_pending
        r.prepare(9)  # once per width
        updates = []
        for end in (8, 12, 13):
            start = max(0, end - 12)
            times = np.arange(start, end, dtype=np.int64) * 5_000_000_000
            r.offer(BarWindow(tuple(rows[start:end]), times, 1, end, 0))
            deadline = time.monotonic() + 60
            while (update := r.poll()) is None and time.monotonic() < deadline:
                time.sleep(0.002)
            updates.append(update)
        assert not r.tuning_pending
        assert [u.result["cpu_tunings"] for u in updates] == [1, 1, 1]
        assert [u.result["method"] for u in updates] == ["rebuild", "incremental", "incremental"]
        assert r.stats.error is None
    finally:
        r.stop()


def test_ping_runs_in_a_spawned_worker():
    with start_worker_pool() as pool:
        assert pool.submit(ping).result(timeout=60) != os.getpid()


def test_pinned_worker_reports_its_cpu_set_and_tunes_within_it():
    pin = sorted(os.sched_getaffinity(0))[:2]
    rows = np.random.default_rng(7).normal(0, 0.001, (12, 8))
    with start_worker_pool(pin) as pool:
        result = pool.submit(cpu_window, rows, np.arange(12), 1, 5).result(timeout=60)
    assert result["cpu_affinity"] == pin
    # threads=0 tunes over {1, 2, 4, 8, all}, capped by the pinned set.
    assert set(result["cpu_thread_trials_ms"]) == set(range(1, len(pin) + 1))
    assert result["threads"] in result["cpu_thread_trials_ms"]


def test_a_process_group_interrupt_neither_fails_the_job_nor_breaks_the_pool():
    """A terminal Ctrl-C or ``timeout`` signals the whole group; the parent stops the run.
    Before, a busy worker failed its job with KeyboardInterrupt and an idle one died."""
    with start_worker_pool() as pool:
        pid = pool.submit(ping).result(timeout=60)
        busy = pool.submit(time.sleep, 0.5)
        time.sleep(0.1)
        for sig in (signal.SIGINT, signal.SIGTERM):
            os.kill(pid, sig)
        assert busy.exception(timeout=60) is None
        for sig in (signal.SIGINT, signal.SIGTERM):  # now idle
            os.kill(pid, sig)
        time.sleep(0.2)
        assert pool.submit(ping).result(timeout=60) == pid


def _running(pid: int) -> bool:
    """Neither gone nor a zombie (an orphan's zombie waits for the container's init)."""
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except (FileNotFoundError, ProcessLookupError):
        return False


def test_the_worker_exits_with_its_parent_also_after_a_sigkill(tmp_path):
    """Ignoring SIGTERM must not leave an orphan holding BLAS threads or a CUDA context."""
    code = (
        "import os, signal\n"
        "from holoscan_finance.analytics.workers import ping, start_worker_pool\n"
        "pool = start_worker_pool()  # referenced: a collected pool shuts its worker down\n"
        "print(pool.submit(ping).result(), flush=True)\n"
        "os.kill(os.getpid(), signal.SIGKILL)\n"
    )
    package_root = Path(workers.__file__).resolve().parents[2]
    path = [str(package_root), os.environ.get("PYTHONPATH", "")]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in path if p))
    out = tmp_path / "pid"
    with out.open("w") as f:  # a pipe would stay open in the worker
        subprocess.run([sys.executable, "-c", code], stdout=f, env=env, timeout=120)
    pid = int(out.read_text())
    try:
        deadline = time.monotonic() + 10
        while _running(pid) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not _running(pid), "the worker outlived its parent"
    finally:
        if _running(pid):
            os.kill(pid, signal.SIGKILL)
