# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``BarBuilder(backend="gpu")`` equals ``BarBuilder(backend="cpu")`` on the same stream
(windows, counters, symbol snapshots, sequence state), and ``host_batches_from_gpu`` turns
device batches into exactly the host decoder's batches. Also the bars warm-up."""

from __future__ import annotations

import dataclasses
import logging

import numpy as np
import pytest
from library.test_bar_builder import directory
from test_gpu_bars import device_batch, host_batches, stream_groups, ulps

from holoscan_finance.marketdata import warmup
from holoscan_finance.marketdata.bar_builder import BarBuilder, host_batches_from_gpu
from holoscan_finance.marketdata.records import EVENT_DTYPE
from holoscan_finance.payloads import BarCounters
from holoscan_finance.settings import BarSettings

pytestmark = pytest.mark.gpu
cp = pytest.importorskip("cupy")

SETTINGS = BarSettings(bar_seconds=5, window_seconds=60, stale_seconds=30, min_bars=4)


def comparable(counters: BarCounters) -> BarCounters:
    """Counters without the wall-clock arrival marks."""
    return dataclasses.replace(counters, **dict.fromkeys(BarCounters.TIMING_FIELDS))


def assert_windows_equal(a, b):
    assert (a.generation, a.completed_bars, a.recv_ns) == (
        b.generation,
        b.completed_bars,
        b.recv_ns,
    )
    np.testing.assert_array_equal(a.times, b.times)
    rows_a = a.rows.get() if a.device else a.rows
    rows_b = b.rows.get() if b.device else b.rows
    np.testing.assert_array_equal(np.isnan(rows_a), np.isnan(rows_b))
    finite = np.isfinite(rows_a)
    assert (ulps(rows_a[finite], rows_b[finite]) <= 1).all()


def test_gpu_and_cpu_builders_agree_on_the_same_stream():
    gpu, cpu = BarBuilder(SETTINGS, backend="gpu"), BarBuilder(SETTINGS, backend="cpu")
    for b in (gpu, cpu):
        b.reset(directory())
    windows = 0
    for k, (part, recv) in enumerate(stream_groups()):
        device, host = [device_batch(part, recv)], host_batches(part, recv)
        if k % 3 == 0:  # two packets of an older generation in front: stale
            device.insert(0, device_batch(part[:2], recv[:2], generation=7))
            host = [dataclasses.replace(h, generation=7) for h in host[:2]] + host
        g, c = gpu.observe(device), cpu.observe(host)
        assert (g is None) == (c is None)
        if g is not None:
            windows += 1
            assert g.device and all(isinstance(p, cp.ndarray) for p in g.parts)
            assert not c.device and isinstance(c.rows, np.ndarray)
            assert_windows_equal(g, c)
        assert comparable(gpu.counters()) == comparable(cpu.counters())
    assert windows >= 5
    assert (gpu.expected, gpu.stale_packets, gpu.sequence_gaps) == (
        cpu.expected,
        cpu.stale_packets,
        cpu.sequence_gaps,
    )
    assert gpu.stale_packets > 0 and gpu.sequence_gaps > 0
    gs, cs = gpu.symbol_snapshot(), cpu.symbol_snapshot()
    assert (gs.generation, gs.symbols, gs.last_event_ns) == (
        cs.generation,
        cs.symbols,
        cs.last_event_ns,
    )
    for name in ("mid", "trade", "volume", "quote_time"):
        a, b = getattr(gs, name), getattr(cs, name)
        assert isinstance(a, np.ndarray) and not a.flags.writeable
        np.testing.assert_array_equal(a.view(np.int64), b.view(np.int64), err_msg=name)


def test_host_batches_from_gpu_equal_the_host_decode():
    (part, recv), *_ = stream_groups(packets=60, group=60)
    batch = device_batch(part, recv, generation=4)
    converted = host_batches_from_gpu(batch)
    expected = [dataclasses.replace(h, generation=4) for h in host_batches(part, recv)]
    assert len(converted) == len(expected) == batch.n_packets
    for got, want in zip(converted, expected):
        assert (got.generation, got.recv_wall_ns, got.sequence, got.sequence_count) == (
            want.generation,
            want.recv_wall_ns,
            want.sequence,
            want.sequence_count,
        )
        assert got.events.dtype == EVENT_DTYPE
        for name in EVENT_DTYPE.names:
            a, b = got.events[name], want.events[name]
            if a.dtype.kind == "f":
                a, b = a.view(np.int64), b.view(np.int64)
            np.testing.assert_array_equal(a, b, err_msg=name)


def test_the_cpu_backend_takes_device_batches_as_validation_input():
    from_device, from_host = (
        BarBuilder(SETTINGS, backend="cpu"),
        BarBuilder(SETTINGS, backend="cpu"),
    )
    for b in (from_device, from_host):
        b.reset(directory())
    for part, recv in stream_groups():
        a = from_device.observe([device_batch(part, recv)])
        b = from_host.observe(host_batches(part, recv))
        assert (a is None) == (b is None)
        if a is not None:
            assert_windows_equal(a, b)
    assert comparable(from_device.counters()) == comparable(from_host.counters())


def test_the_gpu_backend_rejects_host_batches():
    builder = BarBuilder(SETTINGS, backend="gpu")
    builder.reset(directory())
    (part, recv), *_ = stream_groups(packets=5, group=5)
    with pytest.raises(TypeError, match="gpu bars take"):
        builder.observe(host_batches(part, recv))


def test_bars_warm_up_runs_once_per_device(monkeypatch, caplog):
    monkeypatch.setattr(warmup, "_warm", set())
    with caplog.at_level(logging.INFO, logger=warmup.__name__):
        first = warmup.warm_bars_kernels(SETTINGS.as_params(), 0)
        assert warmup.warm_bars_kernels(SETTINGS.as_params(), 0) == 0.0
    assert first > 0
    assert [r.getMessage().split(" in ")[0] for r in caplog.records] == ["GPU bars kernels ready"]
