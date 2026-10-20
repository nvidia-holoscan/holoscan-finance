# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helpers every operator shares: Ref resolution, hold-and-retry emits, device scope, lane CPUs.

``PendingEmits`` is the backpressure rule for outputs that may meet a full ``BLOCKING`` edge
the operator cannot rely on being drained in-lane (the sources, risk, cross-check): on
``BackpressuredError`` the item is held and retried at the next activation, in order, and a
compute that finds a backlog flushes it and returns without producing, so each port emits
at most once per compute and nothing is dropped. Operators that feed same-lane ``OnEach``
consumers or ``DROP_OLDEST`` edges are throttled by the runtime instead and do not use it.
"""

from __future__ import annotations

import os
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import holoscan as hs

from ..ref import Ref


def ref_obj[T](ref: Ref, factory: Callable[[], T] | None = None, *, required: str = "") -> T | None:
    """``ref.obj``; else ``factory()`` if given; else ValueError naming ``required`` if set;
    else None. Stats Refs pass a factory, so every operator also runs standalone."""
    if ref.obj is not None:
        return ref.obj
    if factory is not None:
        return factory()
    if required:
        raise ValueError(f"{required} is required")
    return None


class PendingEmits:
    """Hold-and-retry queue for BLOCKING outputs."""

    def __init__(self) -> None:
        self._items: deque[tuple[hs.OperatorPort, object, Callable[[], None] | None]] = deque()
        self.retries = 0  # BackpressuredError count since start()

    def __len__(self) -> int:
        return len(self._items)

    def add(
        self, port: hs.OperatorPort, item: object, on_sent: Callable[[], None] | None = None
    ) -> None:
        """Queue ``item`` for ``port``; ``on_sent`` runs once it is emitted."""
        self._items.append((port, item, on_sent))

    def flush(self) -> bool:
        """Emit in order and stop at the first ``BackpressuredError``; True when empty."""
        while self._items:
            port, item, on_sent = self._items[0]
            try:
                port.emit(item)
            except hs.errors.BackpressuredError:
                self.retries += 1
                return False
            self._items.popleft()
            if on_sent is not None:
                on_sent()
        return True


@contextmanager
def device_scope(gpu_device: int) -> Iterator[None]:
    """``with cupy.cuda.Device(gpu_device)``: a lane thread's current device is 0 otherwise."""
    import cupy as cp

    with cp.cuda.Device(gpu_device):
        yield


def lane_cpus() -> list[int]:
    """The calling lane thread's CPUs, recorded in its stats as evidence of placement."""
    return sorted(os.sched_getaffinity(0))
