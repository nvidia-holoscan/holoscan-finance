# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Full-day bursts: tcpreplay of a captured day at line rate, timed at the receiver.

The dashboard's button pauses the paced sender and asks the replay controller
(``deploy/igx_thor/gpu_ingest_bench/replay_control.py``) to start; the burst's manifest
moves the run from ``starting`` to ``running`` and sets the base counters, and the status
reports first-to-last packet arrival from the exact per-batch ``BarCounters``. The CPU
cross-check is paused during a burst: it competes with ingest for cores and memory
bandwidth.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.request
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..nyq.manifest import FeedDirectory
    from ..telemetry.stalls import StallMonitor
    from ..telemetry.stats import BarsStats, IngestStats
    from .feed import FeedControl
    from .results import ResultsStore

logger = logging.getLogger(__name__)


class FullDayController:
    """One full-day burst at a time: ``starting`` -> ``running`` -> ``done`` (or ``error``)."""

    def __init__(
        self,
        *,
        replay_control: str | None,
        feed: FeedControl,
        ingest: IngestStats,
        bars: BarsStats,
        results: ResultsStore,
        stalls: StallMonitor,
        backend: str,
    ) -> None:
        self.replay_control = replay_control
        self.feed, self.ingest, self.bars = feed, ingest, bars
        self.results, self.stalls, self.backend = results, stalls, backend
        self._lock = threading.Lock()  # after the directory's locks, before ResultsStore.lock
        self._fullday: dict | None = None

    def available(self) -> bool:
        return bool(self.replay_control)

    def validation_allowed(self) -> bool:
        """``CrossCheckGate.allowed``: no CPU validation while a burst is in flight."""
        f = self._fullday  # one reference read; the risk lane never waits on our lock
        return self.backend != "compare" or not (f and f["status"] in ("starting", "running"))

    def start(self, pps: int) -> dict:
        """Pause the paced sender and start the burst at ``pps``. Blocks for the controller's
        reply (up to 60 s), so HTTP calls it from a worker thread."""
        if not self.replay_control:
            raise ValueError("no replay controller configured (--replay-control)")
        # The controller's own range, checked here so a bad request never pauses the feed.
        if type(pps) is not int or not 1 <= pps <= 250_000:
            raise ValueError("pps must be an integer in 1..250000")
        with self._lock:
            if self._fullday and self._fullday["status"] in ("starting", "running"):
                raise ValueError("a full-day replay is already running")
            self._fullday = dict(status="starting", pps=pps, requested=time.time())
        self.feed.control({"paused": True})
        reply = replay_request(self.replay_control, dict(pps=pps), timeout=60)
        with self._lock:
            if reply.get("error") and self._fullday["status"] == "starting":
                self._fullday.update(status="error", error=reply["error"])
            self._fullday["controller"] = reply
            return dict(self._fullday)

    def on_generation(self, d: FeedDirectory) -> None:
        """Directory listener: the burst's manifest arrived; measure from here."""
        with self._lock:
            f = self._fullday
            if not f or f["status"] != "starting":
                return
            i = self.ingest
            f.update(
                status="running",
                generation=d.generation,
                sender_port=d.sender_port,
                tape_seconds=(d.end_ns - d.start_ns) / 1e9,
                base=(i.packets, i.dropped, i.ring_dropped),
            )
            i.reset_stalls()
            self.stalls.reset()
            with self.results.lock:
                self.results.analytics_max_ms = 0.0

    def status(self) -> dict | None:
        """A copy of the burst's state, refreshed while it runs. Times are first to last
        packet arrival of the burst generation; ``done`` after 2 s without packets."""
        with self._lock:
            f = self._fullday
            if not f:
                return None
            if f["status"] == "running" and self.feed.directory.generation == f["generation"]:
                self._refresh(f)
            return dict(f)

    def _refresh(self, f: dict) -> None:
        i = self.ingest
        p0, g0, r0 = f["base"]
        c = self.bars.counters
        if c is not None and c.generation != f["generation"]:
            c = None
        first, last = (c.first_batch_ns, c.last_batch_ns) if c is not None else (None, None)
        seconds = (last - first) / 1e9 if first and last and last > first else None
        f.update(
            packets=i.packets - p0,
            total_packets=f["controller"].get("packets") if f.get("controller") else None,
            sequence_gaps=i.dropped - g0,
            ring_dropped=i.ring_dropped - r0,
            seconds=seconds,
            events=c.events if c is not None else None,
            completed_bars=c.completed_bars if c is not None else None,
            first_s=round(first / 1e9, 3) if first else None,
            stalls=self.stalls.stalls(self.results.analytics_max_ms),
        )
        if last is not None and time.monotonic_ns() - last > 2e9 and not i.pending_packets():
            f["status"] = "done"
            f["x_realtime"] = f["tape_seconds"] / seconds if seconds else None
            # NIC queue overflow appears in DAQIRI's stats-thread log ("Rx: Dropped N
            # packets"); daqiri.print_stats() is not safe while the engine runs.
            worst = {k: v for k, v in f["stalls"].items() if k != "events"}
            logger.info(
                "full-day replay done: %s packets, gaps %s, ring drops %s, stalls %s",
                f["packets"],
                f["sequence_gaps"],
                f["ring_dropped"],
                worst,
            )
        elif last is None and time.time() - f["requested"] > 120:
            f.update(status="error", error="no packets arrived from the replay controller")


def replay_request(url: str, payload: dict, timeout: float = 5) -> dict:
    """Ask the replay controller (deploy/igx_thor/gpu_ingest_bench/replay_control.py) to start."""
    try:
        req = urllib.request.Request(
            url.rstrip("/") + "/start",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception as exc:  # the dashboard shows it; ingest is unaffected
        return dict(error=str(exc))
