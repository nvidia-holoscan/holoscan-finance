# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capture membership, complete distributions and explicit invalidation (from 4.3
``test_equity_capture`` and ``test_mitch``), and the ``CaptureTap`` that holds the active
capture for the decode sites, the results store and HTTP."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from holoscan_finance.marketdata.covariance import CovarianceEngine
from holoscan_finance.marketdata.gpu_bars import GpuReplayBatch
from holoscan_finance.marketdata.records import EVENT_DTYPE, ReplayBatch
from holoscan_finance.telemetry.capture import CaptureTap, LatencyCapture, percentiles


def packet(seq, received, generation=1, events=17, wire_count=None):
    return ReplayBatch(
        np.zeros(events, dtype=EVENT_DTYPE), generation, received, seq, wire_count=wire_count
    )


def calculation(capture, *, received=1_000_000_000, generation=1, valid=True):
    work = dict(recv_ns=received, times=np.array([100, 200]), generation=generation)
    gpu = dict(
        eligible=10,
        method="incremental",
        receive_to_result_ms=7,
        milliseconds=3,
    )
    cpu = dict(
        eligible=10,
        method="incremental",
        finished_wall_ns=received + 20_000_000,
        milliseconds=12,
        threads=2,
    )
    capture.record_gpu(work, gpu)
    capture.record_cpu(work, cpu, gpu, valid)


def test_percentiles():
    assert percentiles([]) == {"samples": 0, "p50": None, "p99": None, "max": None}
    assert percentiles(range(101)) == {"samples": 101, "p50": 50.0, "p99": 99.0, "max": 100.0}


def test_capture_uses_every_packet_and_calculation_in_the_interval():
    capture = LatencyCapture(1, 1, start_ns=1_000_000_000)
    capture.record_packet(packet(1, capture.start_ns - 1))
    for i in range(100):
        capture.record_packet(packet(1 + 17 * i, capture.start_ns + i))
    capture.record_packet(packet(1701, capture.end_ns))
    calculation(capture)
    result = capture.finish(now_ns=capture.end_ns + 1)
    assert result["valid"]
    assert result["observations"] == 1700 and result["packets"] == 100
    assert result["gpu_receive_to_result_ms"]["p99"] == 7
    assert result["cpu_receive_to_result_ms"]["p99"] == 20
    assert result["matched_analytics_speedup"] == 4
    calculation(capture)
    assert capture.status()["calculations"] == 1


@pytest.mark.parametrize("failure", ["overflow", "gap", "generation", "comparison", "pending"])
def test_incomplete_or_corrupted_captures_cannot_supply_speedup(failure):
    capture = LatencyCapture(
        1, 1, start_ns=1_000_000_000, max_packets=1 if failure == "overflow" else 100
    )
    capture.record_packet(packet(1, capture.start_ns))
    capture.record_packet(
        packet(
            50 if failure == "gap" else 18,
            capture.start_ns + 100,
            generation=2 if failure == "generation" else 1,
        )
    )
    calculation(capture, valid=failure != "comparison")
    if failure == "pending":
        capture.cpu.clear()
    with pytest.raises(ValueError, match="still running"):
        capture.finish(now_ns=capture.start_ns)
    result = capture.finish(now_ns=capture.end_ns)
    assert not result["valid"] and result["matched_analytics_speedup"] is None


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_single_backend_capture_records_inputs_without_claiming_comparison(backend):
    capture = LatencyCapture(1, 1, start_ns=1_000_000_000, backend=backend)
    capture.record_packet(packet(1, capture.start_ns))
    rows = np.random.default_rng(91).normal(size=(10, 5))
    work = dict(recv_ns=capture.start_ns, times=np.arange(10), rows=rows)
    result = CovarianceEngine().update(rows, work["times"])
    result["backend"] = backend
    published = {**result, "receive_to_result_ms": 8}
    capture.record_single(work, published, result)
    report = capture.finish(now_ns=capture.end_ns)
    assert report["valid"] and report["analytics_backend"] == backend
    assert "matched_analytics_speedup" not in report
    assert report["analytics_ms"]["samples"] == 1
    sample = report["single_calculations"][0]
    assert len(sample["input_sha256"]) == 64
    assert sample["validation"]["annual_vol"] == result["annual_vol"]


def test_capture_counts_wire_messages_separately_from_observations():
    cap = LatencyCapture(1, 1, start_ns=100)
    for sequence in (1, 3):
        cap.record_packet(packet(sequence, 200, events=1, wire_count=2))
    result = cap.finish(now_ns=cap.end_ns)
    assert result["observations"] == 2 and result["wire_messages"] == 4
    assert result["packets"] == 2 and result["sequence_gaps"] == 0


def test_invalid_capture_settings():
    for seconds in (0, 601):
        with pytest.raises(ValueError, match="duration"):
            LatencyCapture(seconds, 1)
    with pytest.raises(ValueError, match="backend"):
        LatencyCapture(1, 1, backend="tpu")


def test_tap_lifecycle():
    tap = CaptureTap()
    assert tap.capture is None and not tap.active and tap.status() is None
    with pytest.raises(ValueError, match="no capture"):
        tap.finish()
    status = tap.start(60, generation=4, backend="gpu")
    assert status["generation"] == 4 and status["analytics_backend"] == "gpu"
    assert tap.active and tap.status()["packets"] == 0
    with pytest.raises(ValueError, match="finish the previous capture"):
        tap.start(60, generation=4, backend="gpu")
    with pytest.raises(ValueError, match="still running"):
        tap.finish()
    tap.capture.end_ns = time.time_ns() - 1  # let the interval end now
    assert tap.finish()["generation"] == 4
    assert not tap.active and tap.status()["closed"]
    assert tap.start(60, generation=5, backend="compare")["generation"] == 5


def test_tap_records_host_and_device_batches():
    tap = CaptureTap()
    tap.record_host_batches([packet(1, time.time_ns())])  # no capture: ignored
    tap.start(5, generation=1, backend="compare")
    now = tap.capture.start_ns
    tap.record_host_batches([packet(1, now, events=3, wire_count=5), packet(6, now + 1, events=2)])
    # A device batch of three packets; capture only reads its host metadata.
    columns = {"timestamp_ns": np.zeros(9, np.int64)}
    gpu = GpuReplayBatch(
        columns,
        1,
        np.array([8, 12, 20], np.uint32),
        np.array([4, 4, 4], np.int32),
        np.array([0, 4, 6], np.int64),
        np.array([now + 2, now + 3, now + 4], np.int64),
        now + 4,
    )
    tap.record_gpu_batch(gpu)
    status = tap.capture.status()
    assert status["packets"] == 5
    capture = tap.capture
    assert capture.events == 3 + 2 + 4 + 2 + 3 and capture.wire_messages == 5 + 2 + 12
    assert capture.sequence_gaps == 4 and capture.expected_sequence == 24  # 16..19 missing


def test_a_new_generation_invalidates_only_a_running_capture():
    d = object()  # the listener ignores the directory itself
    tap = CaptureTap()
    tap.on_generation(d)
    tap.start(60, generation=1, backend="compare")
    tap.on_generation(d)
    assert tap.capture.generation_changed
    tap.capture.end_ns = time.time_ns() - 1
    result = tap.finish()
    assert result["generation_changed"] and not result["valid"]
    later = CaptureTap()
    later.start(60, generation=1, backend="compare")
    later.capture.end_ns = time.time_ns() - 1
    later.on_generation(d)  # after the interval: no effect
    assert not later.capture.generation_changed


def test_tap_is_thread_safe():
    tap = CaptureTap()
    started, errors = [], []

    def start():
        try:
            started.append(tap.start(5, generation=1, backend="compare"))
        except ValueError:
            errors.append(1)

    threads = [threading.Thread(target=start) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(started) == 1 and len(errors) == 7
    now = tap.capture.start_ns

    def record(k):
        for i in range(500):
            seq = 1 + (k * 500 + i) * 2
            tap.record_host_batches([packet(seq, now + i, events=1, wire_count=2)])

    threads = [threading.Thread(target=record, args=(k,)) for k in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert tap.capture.packet_count == 2000 and tap.capture.wire_messages == 4000
    assert tap.capture.events == 2000 and tap.capture.expected_sequence == 4001
