# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capture membership, complete distributions and explicit invalidation."""

# ruff: noqa: E402
import pytest

np = pytest.importorskip("numpy")
from holoscan.equities.capture import LatencyCapture
from holoscan.marketdata.wire import EVENT_DTYPE, ReplayBatch


def packet(seq, received, generation=1):
    return ReplayBatch(np.zeros(17, dtype=EVENT_DTYPE), generation, received, seq)


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
    from holoscan.marketdata.covariance import CovarianceEngine

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
