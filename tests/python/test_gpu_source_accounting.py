# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU source receive accounting must match the host sources' `_enqueue_packet`.

The GPU source sees only a per-packet header (payload length, source port/IP,
framing status, message count, sequence, group) and does its accounting per
burst, vectorized for the contiguous case. Feed the same packet stream --
gaps, reordering, duplicates, session changes, framing rejects, non-UDP frames and
ring overflow -- through both and require identical counters. No GPU needed.
"""

import threading
import types
from collections import deque

import pytest

np = pytest.importorskip("numpy")
from holoscan.equities import mitch as replay  # noqa: E402
from holoscan.marketdata import mitch as unit  # noqa: E402
from holoscan.marketdata.wire import EVENT_DTYPE, QUOTE  # noqa: E402

gpu_source = pytest.importorskip("holoscan.marketdata.gpu_mitch_source_op")
host_source = pytest.importorskip("holoscan.equities.host_sources")

MIDNIGHT = replay.day_start_ns("2025-04-01")
COUNTERS = (
    "packets",
    "received_messages",
    "dropped",
    "out_of_order_packets",
    "ring_dropped",
    "malformed_packets",
    "last_packet_wall_ns",
)


def _unit(seq, n=3):
    ev = np.zeros(n, EVENT_DTYPE)
    ev["timestamp_ns"] = MIDNIGHT + 36000 * 10**9 + np.arange(n)
    ev["kind"] = QUOTE
    ev["bid"], ev["ask"], ev["bid_size"], ev["ask_size"] = 10.0, 10.01, 1, 1
    for f in ("price", "volume"):
        ev[f] = np.nan
    return replay.pack_events(ev, seq, MIDNIGHT)[0]


def _host(ring_size):
    op = types.SimpleNamespace()
    op.packet_header = host_source.packet_header
    op._session = op._expected_seq = None
    op.dropped = op.out_of_order_packets = op.packets = op.received_messages = 0
    op.ring_dropped = op.malformed_packets = op.last_packet_wall_ns = 0
    op._lock, op._buckets, op._bucket_ns = threading.Lock(), deque(maxlen=600), 100_000_000
    op._ring, op.ring_size, op._ring_msgs = deque(), ring_size, 0
    return op


def _gpu(ring_size):
    op = types.SimpleNamespace()
    cls = gpu_source.GpuMitchSourceOp
    for name in ("_account_slow", "reset_stalls", "stall_event"):
        setattr(op, name, types.MethodType(getattr(cls, name), op))
    op._lock, op._buckets, op._bucket_ns = threading.Lock(), deque(maxlen=600), 100_000_000
    op._ring, op.ring_size, op._ring_packets, op._ring_msgs = deque(), ring_size, 0, 0
    cls._reset_counters(op)
    return op


def _header(packets):
    """What the gather kernel reports for each (payload bytes | None, port, ip)."""
    rows = []
    for data, port, ip in packets:
        if data is None:
            rows.append([-1, -1, 0, 1, 0, 0, -1])
            continue
        try:
            group, seq, count = unit.inspect_unit(data)
            rows.append([len(data), port, ip, 0, count, seq, group])
        except ValueError:
            count, group, seq = (
                (data[2], data[3], int.from_bytes(data[4:8], "little"))
                if len(data) >= 8
                else (0, -1, 0)
            )
            rows.append([len(data), port, ip, 1, count, seq, group])
    return np.asarray(rows, np.int64).T


def _drive(bursts, ring_size, drain_after=()):
    host, gpu = _host(ring_size), _gpu(ring_size)
    for b, (burst, recv_ns) in enumerate(bursts):
        for data, port, ip in burst:
            if data is None:
                continue  # strip_udp_datagram -> None: skipped before accounting
            host_source.AsyncUdpSourceOp._enqueue_packet(
                host, data, recv_ns, (f"192.168.100.{ip}", port)
            )
        gpu_source.GpuMitchSourceOp._account(gpu, None, None, _header(burst), recv_ns)
        if b in drain_after:  # consumer empties both rings
            host._ring.clear()
            host._ring_msgs = 0
            gpu._ring.clear()
            gpu._ring_packets = gpu._ring_msgs = 0
    return host, gpu


def _assert_same(host, gpu):
    for name in COUNTERS:
        assert getattr(host, name) == getattr(gpu, name), name
    assert host._expected_seq == gpu._expected_seq
    assert len(host._ring) == sum(len(b.index) for b in gpu._ring)
    assert host._ring_msgs == gpu._ring_msgs
    assert list(host._buckets) == list(gpu._buckets)


def _stream(start, packets, port=40001, ip=1):
    out, seq = [], start
    for _ in range(packets):
        data = _unit(seq)
        out.append((data, port, ip))
        seq += data[2]
    return out, seq


def test_contiguous_bursts_use_the_fast_path_and_match():
    a, nxt = _stream(1, 40)
    b, _ = _stream(nxt, 40)
    _assert_same(*_drive([(a, 10**9), (b, 2 * 10**9)], ring_size=1000))


def test_gaps_reordering_duplicates_sessions_rejects_and_overflow_match():
    a, nxt = _stream(1, 30)
    b, _ = _stream(nxt + 12, 30)  # gap between bursts
    c = b[5:8] + a[3:5] + b[20:22]  # reordered / duplicate packets inside a burst
    d, _ = _stream(1, 25, port=40002)  # new sender session restarts at 1
    junk = [(b"\x01\x02", 40001, 1), (bytes(8), 40001, 1), (None, 0, 0)]  # framing rejects, non-UDP
    bursts = [(a, 10**9), (b, 10**9 + 5), (c + junk, 2 * 10**9), (d, 3 * 10**9)]
    _assert_same(*_drive(bursts, ring_size=10**6))
    # Ring overflow mid-burst: sequence consumed, remainder counted as ring drops.
    _assert_same(*_drive(bursts, ring_size=50))
    _assert_same(*_drive(bursts, ring_size=50, drain_after=(1,)))


class _FakeDrain:
    """Stands in for the native drain: hands out burst records, logs ring releases."""

    def __init__(self, records):
        self.records, self.released = np.asarray(records, np.int64), []

    def take(self):
        out, self.records = self.records, np.zeros((0, 9), np.int64)
        return out

    def release(self, end):
        self.released.append(end)

    def stats(self):
        return dict(host_ns=0, error="", last_rx_ns=0, running=True, poll_gap_max_ms=0.0,
                    poll_gap_at=0, ring_waits=0, ring_wait_ms=0.0)  # fmt: skip


def test_native_drain_records_account_like_bursts_and_release_ring_space():
    a, nxt = _stream(1, 4)
    b, _ = _stream(nxt, 3)
    junk = [(b"\x01\x02", 40001, 1)]  # framing reject: nothing kept for decode
    bursts = [(junk, 10**9), (a, 10**9 + 1), (junk, 10**9 + 2), (b, 2 * 10**9)]
    ring_size, pos, records = 16, 0, []
    header_host = np.zeros((ring_size, 7), np.int64)
    for burst, recv_ns in bursts:
        header_host[pos : pos + len(burst)] = _header(burst).T
        records.append([pos, len(burst), recv_ns, 0, 1, 2, 3, 4, 0])
        pos += len(burst)
    gpu = _gpu(ring_size)
    cls = gpu_source.GpuMitchSourceOp
    for name in ("_account", "_pull_native", "_burst_gap", "_timed"):
        setattr(gpu, name, types.MethodType(getattr(cls, name), gpu))
    gpu.ring_size, gpu.keep_alive, gpu.error, gpu.recv_timeout = ring_size, True, None, 5.0
    gpu._eos, gpu._native_checked_ns = threading.Event(), 0
    gpu._native = _FakeDrain(records)
    gpu._native_ring = dict(
        slots=np.zeros((ring_size, 4), np.uint8),
        header_dev=header_host.copy(),
        header_host=header_host,
    )
    gpu._pull_native()
    host, _ = _drive(bursts, ring_size=10**6)
    for name in COUNTERS:
        assert getattr(host, name) == getattr(gpu, name), name
    # The leading reject is released at once; the later one rides on the burst before it.
    assert gpu._native.released == [1]
    assert [(x.start, x.end) for x in gpu._ring] == [(1, 6), (6, 9)]
    assert [len(x.index) for x in gpu._ring] == [len(a), len(b)]
