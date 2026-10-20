# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The GPU ingest path must be bit-compatible with the host path it replaces.

Covers the MITCH decode kernel (every field, every rejection rule, via golden
bytes, randomized streams and mutations), the Ethernet/IPv4/UDP gather kernel
(against ``strip_udp_datagram``) and the device-resident quote bars (against
``QuoteBars``, including a sequence gap). The full-day version of these checks
is ``python -m holoscan.equities.gpu_validate``.
"""

import struct

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
try:
    if cp.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("CUDA GPU required", allow_module_level=True)
except cp.cuda.runtime.CUDARuntimeError:
    pytest.skip("CUDA unavailable", allow_module_level=True)

from holoscan.equities import gpu_validate as gv  # noqa: E402
from holoscan.equities import mitch as replay  # noqa: E402
from holoscan.marketdata import mitch as unit  # noqa: E402
from holoscan.marketdata.bars import QuoteBars  # noqa: E402
from holoscan.marketdata.daqiri import strip_udp_datagram  # noqa: E402
from holoscan.marketdata.gpu_bars import GpuQuoteBars, GpuReplayBatch, apply_sequence  # noqa: E402
from holoscan.marketdata.gpu_mitch import (  # noqa: E402
    SLOT_BYTES,
    decode_packets,
    receive_frames,
)

MIDNIGHT = replay.day_start_ns("2025-04-01")
UNIVERSE = 50
from test_mitch import GOLDEN  # noqa: E402


def _stream(rng, packets=400, start_s=34200):
    return gv.synthetic_stream(rng, packets, UNIVERSE, MIDNIGHT, start_s)


def _decode_both(payloads):
    gv._CTX.update(midnight_ns=MIDNIGHT, universe=UNIVERSE)
    host = gv._host_decode(payloads)
    slots, lens = gv.to_slots(payloads)
    gpu = decode_packets(slots, lens, midnight_ns=MIDNIGHT, universe=UNIVERSE)
    return host, gpu


def _assert_identical(report):
    bad = {k: v for k, v in report.items() if k.endswith("mismatch") and v}
    assert not bad, report
    assert report["events_host"] == report["events_gpu"]


def test_golden_unit_decodes_identically():
    host, gpu = _decode_both([GOLDEN])
    report = gv.compare(host, gpu)
    _assert_identical(report)
    assert report["events_gpu"] == 1
    assert float(gpu.columns["bid"][0]) == -1.0


def test_random_streams_and_mutations_are_bit_identical():
    rng = np.random.default_rng(20260930)
    payloads = _stream(rng)
    report = gv.compare(*_decode_both(payloads))
    _assert_identical(report)
    assert report["status_counts_host"] == {0: len(payloads)}
    mutated = gv.fuzz(payloads, 6000, seed=11)
    report = gv.compare(*_decode_both(mutated))
    _assert_identical(report)
    # Every rejection stage is exercised, not just the happy path.
    assert set(report["status_counts_host"]) == {0, 1, 2}


def test_huge_prices_round_like_python_integer_division():
    unit = bytearray(GOLDEN)
    for magnitude in (2**53 + 1, 2**60 + 12345, 2**63 - 1, 123456789012345678):
        struct.pack_into("<Q", unit, 24, magnitude | (1 << 63))
        struct.pack_into("<Q", unit, 36, magnitude)
        host, gpu = _decode_both([bytes(unit)])
        _assert_identical(gv.compare(host, gpu))


def _frame(payload, *, sport=40001, proto=17, frag=0, ethertype=b"\x08\x00", trim=0):
    ip_len = 20 + 8 + len(payload)
    ip = bytearray(20)
    ip[0], ip[9] = 0x45, proto
    struct.pack_into(">HH", ip, 2, ip_len, 0)
    struct.pack_into(">H", ip, 6, frag)
    ip[12:16], ip[16:20] = bytes([192, 168, 100, 1]), bytes([192, 168, 100, 2])
    udp = struct.pack(">HHHH", sport, 26400, 8 + len(payload), 0)
    frame = b"\xaa" * 6 + b"\xbb" * 6 + ethertype + bytes(ip) + udp + payload + b"\x00" * 4
    return frame[: len(frame) - trim] if trim else frame


def test_gather_matches_host_strip():
    rng = np.random.default_rng(3)
    good = _stream(rng, packets=20)
    frames = [_frame(p, sport=40000 + i) for i, p in enumerate(good)]
    frames += [
        _frame(good[0], proto=6),  # TCP
        _frame(good[1], frag=0x2000),  # IP fragment
        _frame(good[2], ethertype=b"\x86\xdd"),  # IPv6
        _frame(good[3], trim=len(good[3]) + 10),  # truncated
        _frame(b"\x00" * 4),  # payload shorter than a MITCH header
    ]
    stride = 2048
    host_buf = np.zeros((len(frames), stride), np.uint8)
    for i, f in enumerate(frames):
        host_buf[i, : len(f)] = np.frombuffer(f, np.uint8)
    dev = cp.asarray(host_buf)
    ptrs = cp.asarray(dev.data.ptr + np.arange(len(frames), dtype=np.uint64) * stride)
    packed = cp.asarray(np.r_[cp.asnumpy(ptrs), np.asarray([len(f) for f in frames], np.uint64)])
    fused_slots = cp.zeros((len(frames), SLOT_BYTES), cp.uint8)
    header = receive_frames(packed, len(frames), fused_slots).get()
    fused_slots = fused_slots.get()
    for i, f in enumerate(frames):
        expected = strip_udp_datagram(f, minimum_payload=8)
        if expected is None:
            assert header[i, 0] < 0, i
            continue
        payload, peer = expected
        assert len(payload) == header[i, 0], i
        assert peer[1] == header[i, 1], i
        assert fused_slots[i, : len(payload)].tobytes() == payload, i
        try:
            group, sequence, count = unit.inspect_unit(payload)
            assert header[i, 3] == 0 and (header[i, 4], header[i, 5], header[i, 6]) == (
                count,
                sequence,
                group,
            ), i
        except ValueError:
            assert header[i, 3] == 1, i


def test_covariance_engine_consumes_a_device_window_in_place():
    """GPU ingest feeds analytics a CuPy window; results must match the host-window
    path, including incremental updates, a supplied variance and a completed-bar check."""
    from holoscan.marketdata.covariance import CovarianceEngine

    rng = np.random.default_rng(5)
    rows = rng.normal(0, 0.001, (90, 40))
    rows[3:9, 2] = np.nan
    rows[:, 7] = 0
    host_engine, dev_engine = CovarianceEngine("gpu"), CovarianceEngine("gpu")
    for end in (60, 61, 62, 70, 90):
        window, times = rows[end - 60 : end], np.arange(end - 60, end)
        h = host_engine.update(window, times, return_matrix=True)
        d = dev_engine.update(cp.asarray(window), times, return_matrix=True)
        assert (h["method"], h["added_bars"], h["expired_bars"]) == (
            d["method"],
            d["added_bars"],
            d["expired_bars"],
        )
        assert h["eligible_indices"] == d["eligible_indices"]
        assert h["heatmap_indices"] == d["heatmap_indices"]
        np.testing.assert_allclose(h["matrix"], d["matrix"], rtol=1e-10, atol=1e-16)
        np.testing.assert_allclose(h["annual_vol"], d["annual_vol"], rtol=1e-10)
    supplied = dev_engine.update(
        cp.asarray(rows[30:90]),
        np.arange(30, 90),
        observed_variance=np.var(rows[30:90], axis=0, ddof=1),
    )
    assert supplied["eligible_indices"] == d["eligible_indices"]
    changed = rows[30:90].copy()
    changed[0, 0] += 0.01
    with pytest.raises(ValueError, match="completed bars changed"):
        dev_engine.update(cp.asarray(changed), np.arange(30, 90))


@pytest.mark.parametrize("path", ["fused", "general"])
def test_gpu_bars_match_host_bars_through_a_sequence_gap(path, monkeypatch):
    if path == "general":
        monkeypatch.setattr(GpuQuoteBars, "_observe_fused", lambda self, batch: False)
    rng = np.random.default_rng(99)
    payloads = _stream(rng, packets=900)
    del payloads[400:403]  # wire loss: the gap path must reset marks identically
    symbols = [f"S{i}" for i in range(UNIVERSE)]
    settings = dict(
        bar_seconds=5, window_seconds=60, stale_seconds=30, max_spread_bps=100.0, min_bars=4
    )
    hb, gb = QuoteBars(symbols, **settings), GpuQuoteBars(symbols, **settings)
    hst = {"expected": 1, "stale": 0, "gaps": 0}
    g_expected, g_gaps = 1, 0
    for lo in range(0, len(payloads), 97):
        part = payloads[lo : lo + 97]
        recv = np.arange(lo, lo + len(part), dtype=np.int64) + 1
        batches = [
            replay.decode_packet(d, int(r), generation=1, midnight_ns=MIDNIGHT, universe=UNIVERSE)[
                0
            ][0]
            for d, r in zip(part, recv)
        ]
        gv._observe_host(hb, batches, hst)
        slots, lens = gv.to_slots(part)
        dec = decode_packets(slots, lens, midnight_ns=MIDNIGHT, universe=UNIVERSE)
        batch = GpuReplayBatch(
            dec.columns, 1, dec.sequence, dec.wire_count, dec.offsets, recv, int(recv[-1])
        )
        g_expected, _, gaps = apply_sequence(gb, batch, g_expected)
        g_gaps += gaps
    assert hst["gaps"] > 0 and hst["gaps"] == g_gaps and hst["expected"] == g_expected
    assert hb.completed_bars == gb.completed_bars > 10
    for name in ("events", "quotes", "trades", "invalid_quotes", "late_events", "bar_recv_ns"):
        assert getattr(hb, name) == getattr(gb, name), name
    for name in ("mid", "trade", "volume", "previous"):
        np.testing.assert_array_equal(
            getattr(hb, name).view(np.int64), getattr(gb, name).view(np.int64)
        )
    np.testing.assert_array_equal(hb.quote_time, gb.quote_time)
    assert list(hb.times) == list(gb.times)
    for a, b in zip(hb.marks, gb.marks):
        assert isinstance(b, cp.ndarray)  # bars stay in GPU memory
        np.testing.assert_array_equal(a.view(np.int64), b.get().view(np.int64))
    for a, b in zip(hb.rows, gb.rows):
        b = b.get()
        np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
        f = np.isfinite(a)
        assert (gv._ulps(a[f], b[f]) <= 1).all()  # CUDA vs NumPy log: <= 1 ULP
    wa, wb = hb.window(), gb.window()
    assert isinstance(wb[0], cp.ndarray)
    np.testing.assert_array_equal(wa[1], wb[1])
    np.testing.assert_array_equal(np.isnan(wa[0]), np.isnan(wb[0].get()))
