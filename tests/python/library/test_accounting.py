# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Receive accounting: per datagram (host receivers) and per burst (GPU path) agree.

The GPU path sees only the 7-column receive header of each packet and accounts a burst at a
time, vectorized for the contiguous case; the host receivers account every datagram. The
same stream (gaps, reordering, duplicates, session changes, framing rejects, non-UDP frames
and ring overflow) must give identical counters on both (from 4.3
``test_gpu_source_accounting``). No GPU needed.
"""

from __future__ import annotations

import numpy as np
from fakes.fake_drain import receive_header

from holoscan_finance.ingest.accounting import ReceiveAccounting, mitch_packet_header
from holoscan_finance.ingest.gpu_ingest import GpuIngest
from holoscan_finance.ingest.host_receivers import UdpReceiver
from holoscan_finance.marketdata.records import EVENT_DTYPE, QUOTE
from holoscan_finance.nyq.mapping import day_start_ns, pack_events
from holoscan_finance.telemetry.stats import IngestStats

MIDNIGHT = day_start_ns("2025-04-01")
COUNTERS = (
    "packets",
    "received_messages",
    "dropped",
    "out_of_order_packets",
    "ring_dropped",
    "malformed_packets",
    "last_packet_wall_ns",
    "ring_packets",
    "ring_msgs",
)


def unit(seq: int, n: int = 3) -> bytes:
    ev = np.zeros(n, EVENT_DTYPE)
    ev["timestamp_ns"] = MIDNIGHT + 36000 * 10**9 + np.arange(n)
    ev["kind"] = QUOTE
    ev["bid"], ev["ask"], ev["bid_size"], ev["ask_size"] = 10.0, 10.01, 1, 1
    for f in ("price", "volume"):
        ev[f] = np.nan
    return pack_events(ev, seq, MIDNIGHT)[0]


def stream(start: int, packets: int, port: int = 40001, ip: int = 1) -> tuple[list, int]:
    out, seq = [], start
    for _ in range(packets):
        data = unit(seq)
        out.append((data, port, ip))
        seq += data[2]
    return out, seq


def peer(port: int, ip: int) -> tuple[str, int]:
    return (f"192.168.100.{ip}", port)


def scenario() -> list[tuple[list, int]]:
    """Bursts of ``(payload | None, port, ip)`` with their receive times."""
    a, nxt = stream(1, 30)
    b, _ = stream(nxt + 12, 30)  # gap between bursts
    c = b[5:8] + a[3:5] + b[20:22]  # reordered and duplicate packets inside a burst
    d, _ = stream(1, 25, port=40002)  # a new sender session restarts at 1
    junk = [(b"\x01\x02", 40001, 1), (bytes(8), 40001, 1), (None, 0, 0)]  # rejects, non-UDP
    return [(a, 10**9), (b, 10**9 + 5), (c + junk, 2 * 10**9), (d, 3 * 10**9)]


def assert_same(left: IngestStats, right: IngestStats) -> None:
    for name in COUNTERS:
        assert getattr(left, name) == getattr(right, name), name
    assert left.feed_buckets() == right.feed_buckets()


def contiguous() -> list[tuple[list, int]]:
    a, nxt = stream(1, 40)
    return [(a, 10**9), (stream(nxt, 40)[0], 2 * 10**9)]


def framed(data: bytes | None) -> bool:
    try:
        return data is not None and bool(mitch_packet_header(data, None))
    except ValueError:
        return False


def test_packet_and_burst_observation_agree():
    for bursts in (scenario(), contiguous()):
        host, gpu = IngestStats(), IngestStats()
        per_packet, per_burst = ReceiveAccounting(host), ReceiveAccounting(gpu)
        for burst, recv_ns in bursts:
            for data, port, ip in burst:
                if data is None:
                    continue  # strip_udp_datagram -> None: never framed
                try:
                    per_packet.observe_packet(*mitch_packet_header(data, peer(port, ip)), recv_ns)
                except ValueError:
                    host.malformed_packets += 1
            header = receive_header(burst)
            kept = per_burst.observe_burst(header, recv_ns)
            assert list(kept) == [i for i, (d, *_) in enumerate(burst) if framed(d)]
        assert_same(host, gpu)
        assert per_packet._expected == per_burst._expected
        assert gpu.bursts == len(bursts)


def test_contiguous_burst_records_one_gap_event():
    stats = IngestStats()
    acct = ReceiveAccounting(stats)
    a, nxt = stream(1, 4)
    acct.observe_burst(receive_header(a), 1)
    acct.observe_burst(receive_header(stream(nxt + 9, 4)[0]), 2)
    assert stats.dropped == 9
    assert [e["messages"] for e in stats.stalls["events"] if e["kind"] == "gap"] == [9]


def drive(bursts, ring_size, drain_after=()):
    """The host receiver's admission against ``GpuIngest.account_burst``, burst by burst."""
    host_stats, gpu_stats = IngestStats(), IngestStats()
    rx = UdpReceiver(
        host="127.0.0.1",
        port=1,
        recv_buf=65536,
        ring_size=ring_size,
        keep_alive=True,
        recv_timeout_s=1.0,
        stats=host_stats,
    )
    ingest = GpuIngest(
        stats=gpu_stats,
        decoder=None,
        ring_size=ring_size,
        batch_size=1 << 20,
        keep_alive=True,
        recv_timeout_s=1.0,
    )
    for b, (burst, recv_ns) in enumerate(bursts):
        for data, port, ip in burst:
            if data is not None:
                rx._enqueue(data, recv_ns, peer(port, ip))
        header = receive_header(burst)
        ingest.account_burst(None, None, header, recv_ns)
        if b in drain_after:  # the consumer empties both queues
            rx.poll(1 << 30)
            with gpu_stats.lock:  # what next_batch's staging does, without the decode
                ingest._bursts.clear()
                gpu_stats.ring_packets = gpu_stats.ring_msgs = 0
    return host_stats, gpu_stats, rx, ingest


def test_ring_admission_agrees_through_overflow():
    for ring_size, drain_after in ((10**6, ()), (50, ()), (50, (1,))):
        host, gpu, rx, ingest = drive(scenario(), ring_size, drain_after)
        assert_same(host, gpu)
        assert len(rx._ring) == sum(len(b.index) for b in ingest._bursts)
    assert host.ring_dropped > 0  # the overflow case consumed sequences at receipt
