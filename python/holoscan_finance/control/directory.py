# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The replay directory: which sender, symbols and market day the current generation has.

``ReplayDirectory`` accepts manifests (``POST /manifest``, and the replay loop through
``ReplayGenerations``) and publishes each as an immutable ``FeedDirectory`` in a ``Knob``;
decode, bars and every generation filter read it without a lock. Global lock order (one
order, so no ABBA deadlock between manifests and the analytics lane):
``ReplayDirectory.lock`` -> ``_listener_lock`` -> ``FullDayController`` -> ``ResultsStore``
-> ``CaptureTap`` -> ``IngestStats``. Listeners run after ``lock`` is released, in
publication order.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from ..equity.manifest import validate_manifest

if TYPE_CHECKING:
    from ..equity.manifest import FeedDirectory
    from ..ref import Knob

# Ports a replay generation may take (no privileged ports).
_PORT_LO, _PORT_HI = 1024, 65535


class ReplayDirectory:
    """Generations of the replayed feed; one accepted manifest per generation."""

    lock: threading.Lock  # validation and publication only
    used_sender_ports: set[int]

    def __init__(self, knob: Knob[FeedDirectory]) -> None:
        self.knob = knob
        self.lock = threading.Lock()
        self._listener_lock = threading.Lock()  # serializes listener runs, in publication order
        self.used_sender_ports = set()
        self._listeners: list[Callable[[FeedDirectory], None]] = []

    @property
    def current(self) -> FeedDirectory | None:
        return self.knob.get()

    @property
    def generation(self) -> int:
        """0 before the first manifest."""
        d = self.knob.get()
        return d.generation if d is not None else 0

    def add_listener(self, fn: Callable[[FeedDirectory], None]) -> None:
        """Run ``fn(directory)`` after each accepted manifest. A listener must never call
        ``set_manifest`` and may read the directory only through ``current``/``generation``."""
        self._listeners.append(fn)

    def set_manifest(self, payload: dict) -> dict:
        """Accept a newer generation. An older or equal one is a no-op that returns the
        current generation; a sender port seen before raises ValueError."""
        d = validate_manifest(payload)
        with self.lock:
            current = self.knob.get()
            if current is not None and d.generation <= current.generation:
                return {"generation": current.generation}
            if d.sender_port in self.used_sender_ports:
                raise ValueError("SBE replay sender port must be new for each generation")
            self.used_sender_ports.add(d.sender_port)
            self.knob.set(d)
            # Hand over hand: listeners of consecutive generations run in publication order,
            # yet no listener ever runs while a data-plane reader could wait on `lock`.
            self._listener_lock.acquire()
        try:
            for fn in self._listeners:
                fn(d)
        finally:
            self._listener_lock.release()
        return {"generation": d.generation}


class ReplayGenerations:
    """``GpuSbeReplaySourceOp.on_loop``: every replay pass after the first is a new
    generation from a fresh sender port, as ``tools.replay_sender`` registers each loop."""

    def __init__(self, directory: ReplayDirectory, base_manifest: dict, first_port: int) -> None:
        self._directory = directory
        self._base = dict(base_manifest)
        self._first_port = first_port

    def __call__(self, pass_index: int) -> int | None:
        """Register pass ``pass_index`` and return its sender port (None for pass 0, which
        the application context registered)."""
        if pass_index == 0:
            return None
        while True:
            port = self._fresh_port(pass_index)
            generation = max(self._directory.generation + 1, time.time_ns())
            manifest = {**self._base, "sender_port": port, "generation": generation}
            if self._directory.set_manifest(manifest)["generation"] == generation:
                return port
            # A concurrent manifest (POST /manifest) won with a newer generation and ours
            # was a no-op: retry above that generation.

    def _fresh_port(self, pass_index: int) -> int:
        span = _PORT_HI - _PORT_LO + 1
        used = self._directory.used_sender_ports
        for step in range(span):
            port = _PORT_LO + (self._first_port + pass_index + step - _PORT_LO) % span
            if port not in used:
                return port
        raise RuntimeError("no fresh UDP source port for the replay generation")
