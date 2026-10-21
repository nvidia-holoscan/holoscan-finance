# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU spawn workers: the persistent engine across a roll and a seek, the pre-spawn ``ping``,
the pinned worker reporting its CPU set, and the worker leaving SIGINT/SIGTERM to the parent
without outliving it."""

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
from holoscan_finance.analytics.workers import cpu_window, ping, start_worker_pool

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
