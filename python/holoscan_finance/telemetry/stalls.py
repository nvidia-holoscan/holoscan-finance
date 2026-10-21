# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Process-wide stall monitor: interpreter-lock stalls and garbage-collection pauses.

A 5 ms ticker thread (``stall-heartbeat``) measures how late it wakes: a late tick means
the whole process lost the GIL or the CPU, the full-day burst failure mode (a stall longer
than the NIC buffer pool drops packets). Ticks later than ``threshold_ms`` (50 ms) are
recorded as ``heartbeat`` stall events on the ingest stats. A ``gc.callbacks`` hook times
collections; it runs in whichever thread collects, so it takes no lock.
"""

from __future__ import annotations

import gc
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .stats import IngestStats


class StallMonitor:
    """The ``stall-heartbeat`` ticker and the GC pause timer of one process."""

    heartbeat: dict  # {max_ms, at_s}: the latest tick since reset()
    gc_pause: dict  # {max_ms, generation, started}: the longest collection since reset()

    def __init__(
        self, ingest: IngestStats, *, period_s: float = 0.005, threshold_ms: float = 50.0
    ) -> None:
        self.ingest = ingest
        self.period_s, self.threshold_ms = period_s, threshold_ms
        self.heartbeat = dict(max_ms=0.0, at_s=None)
        self.gc_pause = dict(max_ms=0.0, generation=None, started=0)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Install the GC hook and start the ticker; once per run."""
        if self._thread is not None:
            return
        self._stop.clear()
        gc.callbacks.append(self._gc_timer)
        self._thread = threading.Thread(target=self._tick, daemon=True, name="stall-heartbeat")
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join()
        self._thread = None
        gc.callbacks.remove(self._gc_timer)

    def reset(self) -> None:
        """Start a new measurement (each full-day burst)."""
        self.gc_pause.update(max_ms=0.0, generation=None)
        self.heartbeat.update(max_ms=0.0, at_s=None)

    def stalls(self, analytics_max_ms: float) -> dict:
        """The ``fullday.stalls`` dict: the source's stall record plus these monitors and
        the slowest analytics window."""
        return dict(
            self.ingest.stalls_view(),
            heartbeat=dict(self.heartbeat),
            gc_max_ms=round(self.gc_pause["max_ms"], 1),
            gc_generation=self.gc_pause["generation"],
            analytics_max_ms=round(analytics_max_ms, 1),
        )

    def _tick(self) -> None:
        period_ms = self.period_s * 1e3
        while True:
            t = time.perf_counter_ns()
            if self._stop.wait(self.period_s):
                return
            late = (time.perf_counter_ns() - t) / 1e6 - period_ms
            if late > self.threshold_ms:
                self.ingest.stall_event("heartbeat", late)
            if late > self.heartbeat["max_ms"]:
                self.heartbeat.update(max_ms=round(late, 1), at_s=round(time.monotonic(), 1))

    def _gc_timer(self, phase: str, info: dict) -> None:
        g = self.gc_pause
        if phase == "start":
            g["started"] = time.perf_counter_ns()
        elif g["started"]:
            ms = (time.perf_counter_ns() - g["started"]) / 1e6
            if ms > g["max_ms"]:
                g.update(max_ms=ms, generation=info.get("generation"))
