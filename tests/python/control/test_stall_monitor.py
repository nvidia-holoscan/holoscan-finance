# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""StallMonitor: GIL stalls seen by the 5 ms heartbeat, GC pauses, the 4.3 stalls shape."""

from __future__ import annotations

import ctypes
import gc
import threading
import time

from holoscan_finance.telemetry.stalls import StallMonitor
from holoscan_finance.telemetry.stats import IngestStats


def hold_gil(seconds: float) -> None:
    # As a Python thread blocked in the kernel does: ctypes.PyDLL keeps the GIL.
    ctypes.PyDLL(None).usleep(int(seconds * 1e6))


def wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.005)
    return predicate()


def test_heartbeat_records_a_gil_stall_as_an_ingest_stall_event():
    ingest = IngestStats()
    monitor = StallMonitor(ingest)
    monitor.start()
    try:
        assert wait_for(lambda: any(t.name == "stall-heartbeat" for t in threading.enumerate()))
        time.sleep(0.02)
        hold_gil(0.15)
        assert wait_for(lambda: monitor.heartbeat["max_ms"] > 50)
    finally:
        monitor.stop()
    events = [e for e in ingest.stalls["events"] if e["kind"] == "heartbeat"]
    assert events and events[0]["ms"] > 50
    assert monitor.heartbeat["at_s"] is not None
    assert not any(t.name == "stall-heartbeat" for t in threading.enumerate())


def test_short_oversleeps_are_tracked_but_not_reported():
    ingest = IngestStats()
    monitor = StallMonitor(ingest, threshold_ms=10_000)
    monitor.start()
    try:
        time.sleep(0.02)
        hold_gil(0.03)
        assert wait_for(lambda: monitor.heartbeat["max_ms"] > 10)
    finally:
        monitor.stop()
    assert not ingest.stalls["events"]


def test_gc_pauses_are_timed_until_stop():
    monitor = StallMonitor(IngestStats())
    monitor.start()
    try:
        gc.collect()
        assert monitor.gc_pause["max_ms"] > 0 and monitor.gc_pause["generation"] is not None
    finally:
        monitor.stop()
    assert monitor._gc_timer not in gc.callbacks
    monitor.reset()
    gc.collect()
    assert monitor.gc_pause["max_ms"] == 0.0 and monitor.gc_pause["generation"] is None


def test_start_and_stop_are_idempotent():
    monitor = StallMonitor(IngestStats())
    monitor.stop()
    monitor.start()
    monitor.start()
    assert gc.callbacks.count(monitor._gc_timer) == 1
    monitor.stop()
    monitor.stop()
    assert monitor._gc_timer not in gc.callbacks


def test_stalls_has_the_4_3_shape():
    ingest = IngestStats()
    ingest.stall_event("compute", 75.0, decode=70.0)
    ingest.burst_gap(0)
    ingest.burst_gap(30_000_000)
    ingest.stalls["native"] = {"poll_gap_max_ms": 1.0}
    monitor = StallMonitor(ingest)
    monitor.heartbeat.update(max_ms=12.3, at_s=4.0)
    monitor.gc_pause.update(max_ms=1.26, generation=2)
    stalls = monitor.stalls(analytics_max_ms=3.14159)
    assert set(stalls) == {
        "burst_gaps_ms", "worst_burst", "worst_compute", "events", "native", "heartbeat",
        "gc_max_ms", "gc_generation", "analytics_max_ms",
    }  # fmt: skip
    assert stalls["burst_gaps_ms"] == [30.0] and stalls["events"][0]["kind"] == "compute"
    assert stalls["heartbeat"] == {"max_ms": 12.3, "at_s": 4.0}
    assert (stalls["gc_max_ms"], stalls["gc_generation"], stalls["analytics_max_ms"]) == (
        1.3,
        2,
        3.1,
    )
    stalls["events"].clear()
    stalls["heartbeat"]["max_ms"] = 0  # copies: the snapshot never aliases live state
    assert ingest.stalls["events"] and monitor.heartbeat["max_ms"] == 12.3


def test_reset_starts_a_new_measurement():
    monitor = StallMonitor(IngestStats())
    monitor.heartbeat.update(max_ms=80.0, at_s=1.0)
    monitor.gc_pause.update(max_ms=9.0, generation=2)
    monitor.reset()
    assert monitor.heartbeat == {"max_ms": 0.0, "at_s": None}
    assert (monitor.gc_pause["max_ms"], monitor.gc_pause["generation"]) == (0.0, None)
