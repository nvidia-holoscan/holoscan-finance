# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU sets and thread masks: the DAQIRI lcores run only DPDK, helpers never inherit a lane mask.

Every thread of the process stays off the DPDK master and RX lcores (4.3 ``_isolate_cores``
and the analytics worker initializer). The Holoscan 5 runtime never re-applies a lane's
placement, so a mask change made in ``start()`` persists for the whole lane (c3): around
``daqiri_init``, ``thread_mask`` always restores and checks the caller's mask, and drain
and receiver threads are created under ``helper_mask``, which defaults to the process
(main-thread) mask rather than a narrowed or busy-polled lane mask. Stdlib only, so every
package can import it at once.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass


def parse_cpu_list(text: str) -> frozenset[int]:
    """``"0-2,4"`` -> ``{0, 1, 2, 4}`` (the cpuset list syntax); ValueError if malformed."""
    cpus: set[int] = set()
    for token in text.split(","):
        lo, dash, hi = token.strip().partition("-")
        if not lo.isdecimal() or (dash and (not hi.isdecimal() or int(hi) < int(lo))):
            raise ValueError(f"invalid CPU list {text!r}: expected a list such as '0-2,4'")
        cpus.update(range(int(lo), int(hi or lo) + 1))
    return frozenset(cpus)


def normalize_cpu_list(value: object, *, name: str) -> list[int]:
    """A CPU-list Param as a sorted, de-duplicated list; ValueError unless it is a list or
    tuple of ints >= 0 (``hs.Param([])`` accepts any content and keeps a tuple a tuple)."""
    if not isinstance(value, (list, tuple)) or not all(
        isinstance(c, int) and not isinstance(c, bool) and c >= 0 for c in value
    ):
        raise ValueError(f"{name}: expected a list of CPU numbers >= 0, got {value!r}")
    return sorted(set(value))


@dataclass(frozen=True)
class CpuSets:
    """The process's CPU sets, computed once at start-up."""

    all: frozenset[int]  # the start-up mask
    reserved: frozenset[int]  # the DAQIRI lcores in DAQIRI modes
    app: frozenset[int]  # all - reserved: every pipeline thread
    ingest: frozenset[int]  # the ingest lane
    helper: frozenset[int]  # other lanes, drains, receivers
    worker: frozenset[int]  # spawned analytics workers

    @classmethod
    def compute(
        cls,
        *,
        reserved: Iterable[int] = (),
        ingest: Iterable[int] | None = None,
        busy_poll: bool = False,
    ) -> CpuSets:
        """``ingest`` defaults to ``app``. Only a busy-polled, narrowed ingest set is carved
        out of ``helper`` (= ``worker``), so by default every set but ``ingest`` is ``app``,
        the 4.3 ``keep_off_cores`` set."""
        start = frozenset(os.sched_getaffinity(0))
        lcores = frozenset(reserved) & start
        app = start - lcores
        if not app:
            raise ValueError(f"no CPUs left beside the reserved {sorted(lcores)}")
        lane = frozenset(ingest) if ingest is not None else frozenset()
        if lane and not lane <= app:
            raise ValueError(f"ingest CPUs {sorted(lane)} are not all in {sorted(app)}")
        lane = lane or app
        helper = app - lane if busy_poll and app - lane else app
        return cls(all=start, reserved=lcores, app=app, ingest=lane, helper=helper, worker=helper)


def process_mask() -> frozenset[int]:
    """The main thread's mask: ``sched_getaffinity(getpid())`` is that thread's on Linux, also
    when called from a lane thread (c3), never the caller's own mask."""
    return frozenset(os.sched_getaffinity(os.getpid()))


def helper_mask(cpus: Sequence[int]) -> list[int]:
    """The mask to create a helper thread under: ``cpus``, or the process mask if empty."""
    return sorted(cpus) if cpus else sorted(process_mask())


@contextmanager
def thread_mask(cpus: Iterable[int] | None = None) -> Iterator[None]:
    """Run the body on ``cpus`` (if given and non-empty), then restore the calling thread's mask.

    The saved mask is always restored and checked, also when nothing was widened: this
    undoes EAL pinning the caller inside ``daqiri_init``, which would otherwise move the
    whole lane onto a DPDK lcore. Threads started in the body inherit ``cpus``.
    """
    saved = os.sched_getaffinity(0)
    wanted = set(cpus) if cpus is not None else set()
    if wanted:
        os.sched_setaffinity(0, wanted)
    try:
        yield
    finally:
        os.sched_setaffinity(0, saved)
        if os.sched_getaffinity(0) != saved:
            raise RuntimeError(f"thread mask not restored to {sorted(saved)}")


def keep_off_cores(reserved: Iterable[int]) -> int:
    """Move this process's threads off the DPDK lcores, leaving the lcore threads (pinned
    inside ``reserved``) alone; threads created later inherit the mask. A Python or CUDA
    thread scheduled onto the RX core takes CPU time from the NIC poll loop.

    Returns the number of threads whose mask changed (4.3 ``_isolate_cores``; threads already
    off the lcores are left untouched instead of re-set to the same mask).
    """
    reserved = set(reserved)
    if not reserved or not hasattr(os, "sched_setaffinity"):
        return 0
    moved = 0
    for tid in os.listdir("/proc/self/task"):
        try:
            current = os.sched_getaffinity(int(tid))
            kept = current - reserved
            if kept and kept != current:
                os.sched_setaffinity(int(tid), kept)
                moved += 1
        except OSError:
            pass
    return moved


def pin_worker(cpus: Sequence[int]) -> None:
    """Spawn-worker initializer: move every thread of the child onto ``cpus`` (BLAS pools
    start at import; later threads inherit). ``[]`` keeps the inherited mask."""
    if not cpus or not hasattr(os, "sched_setaffinity"):
        return
    target = set(cpus)
    for tid in os.listdir("/proc/self/task"):
        try:
            os.sched_setaffinity(int(tid), target)
        except OSError:
            pass
