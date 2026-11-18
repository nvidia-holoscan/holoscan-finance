# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU sets and thread masks. Mask changes run in helper threads, spawned workers or a child
process, so the test process keeps its own mask."""

from __future__ import annotations

import json
import multiprocessing
import os
import subprocess
import sys
import textwrap
import threading
from concurrent.futures import ProcessPoolExecutor

import pytest

from holoscan_finance.ingest import affinity
from holoscan_finance.ingest.affinity import (
    CpuSets,
    helper_mask,
    normalize_cpu_list,
    parse_cpu_list,
    pin_worker,
    process_mask,
    thread_mask,
)

CPUS = sorted(os.sched_getaffinity(0))
needs_two = pytest.mark.skipif(len(CPUS) < 2, reason="needs at least two CPUs")


def in_thread(fn):
    """Run ``fn`` in a new thread (whose mask changes die with it); return or raise its result."""
    out = {}

    def run():
        try:
            out["value"] = fn()
        except BaseException as exc:  # re-raised in the test thread
            out["error"] = exc

    t = threading.Thread(target=run)
    t.start()
    t.join()
    if "error" in out:
        raise out["error"]
    return out["value"]


def test_parse_cpu_list():
    assert parse_cpu_list("0-2,4") == {0, 1, 2, 4}
    assert parse_cpu_list(" 7 , 3-3") == {3, 7}
    for bad in ("", "a", "3-1", "1-", "-2", "1,,2"):
        with pytest.raises(ValueError, match="CPU list"):
            parse_cpu_list(bad)


def test_normalize_cpu_list():
    assert normalize_cpu_list((3, 1, 1), name="helper_cpus") == [1, 3]
    assert normalize_cpu_list([], name="helper_cpus") == []
    for bad in (["a"], [-1], [True], [1.0], "0-2", None, 3):
        with pytest.raises(ValueError, match="worker_cpus"):
            normalize_cpu_list(bad, name="worker_cpus")


def test_cpu_sets_default_to_the_whole_process_mask():
    sets = CpuSets.compute()
    assert sets.all == frozenset(CPUS) and sets.reserved == frozenset()
    assert sets.app == sets.ingest == sets.helper == sets.worker == sets.all
    with pytest.raises(ValueError, match="no CPUs left"):
        CpuSets.compute(reserved=CPUS)


@needs_two
def test_cpu_sets_keep_every_set_off_the_reserved_cpus():
    sets = CpuSets.compute(reserved=[CPUS[0], max(CPUS) + 100])  # one is not ours: ignored
    assert sets.reserved == {CPUS[0]} and sets.app == frozenset(CPUS[1:])
    assert sets.ingest == sets.helper == sets.worker == sets.app


@needs_two
def test_cpu_sets_carve_helpers_out_only_for_a_busy_polled_ingest_lane():
    lane = [CPUS[-1]]
    sets = CpuSets.compute(ingest=lane)
    assert sets.ingest == set(lane) and sets.helper == sets.worker == sets.app
    sets = CpuSets.compute(ingest=lane, busy_poll=True)
    assert sets.helper == sets.worker == sets.app - set(lane)
    whole = CpuSets.compute(ingest=CPUS, busy_poll=True)
    assert whole.helper == whole.app  # nothing left to carve out
    with pytest.raises(ValueError, match="ingest CPUs"):
        CpuSets.compute(reserved=lane, ingest=lane)


@needs_two
def test_thread_mask_restores_a_mask_a_callee_repinned():
    def body():
        saved = os.sched_getaffinity(0)
        with thread_mask():
            os.sched_setaffinity(0, {CPUS[-1]})  # what an EAL init does to its caller
        return saved, os.sched_getaffinity(0)

    saved, after = in_thread(body)
    assert after == saved


@needs_two
def test_thread_mask_asserts_the_restore(monkeypatch):
    real = os.sched_setaffinity

    def body():
        with thread_mask():
            real(0, {CPUS[-1]})
            monkeypatch.setattr(affinity.os, "sched_setaffinity", lambda pid, cpus: None)

    with pytest.raises(RuntimeError, match="not restored"):
        in_thread(body)


@needs_two
def test_thread_mask_widens_for_the_body_and_its_threads():
    def body():
        os.sched_setaffinity(0, {CPUS[0]})
        with thread_mask(CPUS[:2]):
            inside = os.sched_getaffinity(0)
            inherited = in_thread(lambda: os.sched_getaffinity(0))
        return inside, inherited, os.sched_getaffinity(0)

    inside, inherited, after = in_thread(body)
    assert inside == inherited == set(CPUS[:2]) and after == {CPUS[0]}


@needs_two
def test_helper_mask_defaults_to_the_process_mask_not_the_lane_mask():
    def body():
        os.sched_setaffinity(0, {CPUS[0]})  # a narrowed lane thread
        return helper_mask([]), sorted(process_mask()), helper_mask([CPUS[1], CPUS[0]])

    default, process, explicit = in_thread(body)
    assert default == process == CPUS
    assert explicit == sorted(CPUS[:2])


@needs_two
def test_keep_off_cores_leaves_threads_pinned_inside_reserved_alone():
    code = textwrap.dedent(
        """
        import json, os, threading
        from holoscan_finance.ingest.affinity import keep_off_cores
        r = max(os.sched_getaffinity(0))
        pinned, release, masks = threading.Event(), threading.Event(), {}

        def lcore():
            os.sched_setaffinity(0, {r})
            pinned.set()
            release.wait()
            masks["lcore"] = sorted(os.sched_getaffinity(0))

        def helper():
            release.wait()
            masks["helper"] = sorted(os.sched_getaffinity(0))

        threads = [threading.Thread(target=f) for f in (lcore, helper)]
        for t in threads:
            t.start()
        pinned.wait()
        moved = keep_off_cores({r})
        release.set()
        for t in threads:
            t.join()
        print(json.dumps(dict(r=r, moved=moved, main=sorted(os.sched_getaffinity(0)), **masks)))
        """
    )
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=60
    )
    assert out.returncode == 0, out.stderr
    got = json.loads(out.stdout)
    r = got["r"]
    assert got["lcore"] == [r]
    assert r not in got["main"] and r not in got["helper"] and got["moved"] >= 2
    assert affinity.keep_off_cores([]) == 0


@needs_two
def test_pin_worker_masks_a_spawned_worker():
    spawn = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        1, mp_context=spawn, initializer=pin_worker, initargs=([CPUS[1]],)
    ) as pool:
        assert pool.submit(os.sched_getaffinity, 0).result(timeout=60) == {CPUS[1]}
    with ProcessPoolExecutor(1, mp_context=spawn, initializer=pin_worker, initargs=([],)) as pool:
        assert pool.submit(os.sched_getaffinity, 0).result(timeout=60) == set(CPUS)
