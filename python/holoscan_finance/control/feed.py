# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pacing controls for the external replay sender (``/feedrate``, ``/feedstatus``).

The sender (``tools.replay_sender``) polls ``GET /feedrate`` every 0.2 s and reports its
cursor through ``POST /feedstatus``; nothing here reaches an operator. The rules are 4.3
``EquityState.feed_control`` and the ``/feedstatus`` and ``/replay/resume`` handlers,
verbatim. The NIC-free replay source ignores these controls (its pace is a CLI setting).
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .directory import ReplayDirectory


class FeedControl:
    """The replay sender's pace, pause and seek cursor, and its last status report."""

    sender: dict | None  # the sender's last accepted /feedstatus report

    def __init__(self, directory: ReplayDirectory) -> None:
        self.directory = directory
        self._lock = threading.Lock()  # a leaf: nothing is called while it is held
        self._control = dict(speed=10.0, rate=50000.0, paused=False, seek=48600.0, seek_seq=1)
        self.sender = None

    def control(self, payload: dict | None = None) -> dict:
        """Apply any subset of ``{speed, rate, paused, seek}``; a seek bumps ``seek_seq``.
        Returns a copy of the controls; ValueError or TypeError on an invalid value."""
        with self._lock:
            if payload is not None:
                if set(payload) - {"speed", "rate", "paused", "seek"}:
                    raise ValueError("unsupported replay control")
                c = {**self._control, **payload}
                if (
                    not 0 <= float(c["speed"]) <= 100
                    or not 1 <= float(c["rate"]) <= 200000
                    or type(c["paused"]) is not bool
                    or not 0 <= float(c["seek"]) < 86400
                ):
                    raise ValueError("invalid replay control")
                if "seek" in payload:
                    c["seek_seq"] += 1
                self._control = c
            return dict(self._control)

    def sender_status(self, payload: dict) -> dict:
        """Keep the sender's report if it is for the current generation and a seek already
        issued; reports of an older generation or a future seek are ignored."""
        with self._lock:
            if (
                payload.get("generation") == self.directory.generation
                and type(payload.get("seek_seq")) is int
                and 0 <= payload["seek_seq"] <= self._control["seek_seq"]
            ):
                self.sender = payload
        return {"status": "received"}

    def resume(self) -> dict:
        """Back to the paced sender: unpause and restart it at the session open (the sender
        then registers a new generation)."""
        d = self.directory.current
        start = d.start_ns if d is not None else None
        seek = start // 10**9 % 86400 if start else self.control()["seek"]
        return self.control({"paused": False, "seek": float(seek)})
