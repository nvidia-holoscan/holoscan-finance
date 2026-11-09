# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host receivers: the UDP socket over loopback and DAQIRI host memory through the fake
``daqiri`` (frame stripping, sequence accounting, the ring, failures, thread masks).

Socket and DAQIRI receive identical units. Mask changes run in helper threads, so the test
process keeps its own mask.
"""

from __future__ import annotations

import contextlib
import os
import socket
import sys
import threading
import time

import numpy as np
import pytest
from fakes.fake_daqiri import make_fake_daqiri

from holoscan_finance.ingest import host_receivers
from holoscan_finance.ingest.accounting import ReceiveAccounting, sbe_packet_header
from holoscan_finance.ingest.affinity import process_mask
from holoscan_finance.ingest.host_receivers import DaqiriHostReceiver, UdpReceiver
from holoscan_finance.marketdata.framing import udp_frame
from holoscan_finance.marketdata.records import CONDITION, EVENT_DTYPE, QUOTE, TRADE
from holoscan_finance.nyq.mapping import day_start_ns, decode_packet, pack_events
from holoscan_finance.telemetry.stats import IngestStats

MIDNIGHT = day_start_ns("2025-04-01")
UNIVERSE = 2886
SENDER = ("10.0.0.1", 55555)
COUNTERS = ("packets", "received_messages", "dropped", "out_of_order_packets", "malformed_packets")


def unit(sequence: int, bids: list[tuple[int, float]]) -> tuple[bytes, int]:
    """One SBE packet of quotes and its entry count (one entry per quote)."""
    rows = np.zeros(len(bids), EVENT_DTYPE)
    rows["kind"] = QUOTE
    rows["timestamp_ns"] = MIDNIGHT + 34200 * 10**9 + np.arange(len(rows))
    rows["symbol_id"] = [s for s, _ in bids]
    rows["bid"] = [b for _, b in bids]
    rows["ask"] = rows["bid"] + 0.02
    rows["bid_size"] = rows["ask_size"] = 100
    for f in ("price", "volume"):
        rows[f] = np.nan
    return pack_events(rows, sequence, MIDNIGHT)


def units(count: int, start: int = 1, gap_after: int | None = None, gap: int = 0) -> list[bytes]:
    out, seq = [], start
    for i in range(count):
        data, n = unit(seq, [(i % 7, 100.0 + i)])
        out.append(data)
        seq += n + (gap if i == gap_after else 0)
    return out


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def udp(stats: IngestStats, **kw) -> UdpReceiver:
    params = dict(host="127.0.0.1", port=free_port(), recv_buf=1 << 20, ring_size=65536)
    params.update(keep_alive=True, recv_timeout_s=5.0, stats=stats)
    params.update(kw)
    return UdpReceiver(**params)


def daqiri(fake, stats: IngestStats, **kw) -> DaqiriHostReceiver:
    params = dict(config={"fake": True}, interface_name="rx_port", queue_id=0, ring_size=65536)
    params.update(keep_alive=True, recv_timeout_s=5.0, stats=stats)
    params.update(kw)
    return DaqiriHostReceiver(**params)


def test_udp_receiver_counts_like_receive_accounting():
    stats = IngestStats()
    rx = udp(stats)
    payloads = units(20, gap_after=9, gap=5) + [b"\x01\x02"]  # a sequence gap, a malformed unit
    rx.start()
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
            for p in payloads:
                sender.sendto(p, rx._address)
            src = ("127.0.0.1", sender.getsockname()[1])
            assert wait(lambda: stats.packets + stats.malformed_packets == len(payloads))
    finally:
        rx.stop()
    reference = IngestStats()
    acct = ReceiveAccounting(reference)
    for p in payloads[:-1]:
        acct.observe_packet(*sbe_packet_header(p, src), stats.last_packet_wall_ns)
    reference.malformed_packets = 1
    for name in COUNTERS:
        assert getattr(stats, name) == getattr(reference, name), name
    assert stats.dropped == 5 and stats.ring_packets == 20

    batch = rx.poll(1 << 20)
    assert [item[0] for item in batch.items] == payloads[:-1]
    assert all(item[3] == src for item in batch.items)
    assert batch.messages == sum(item[2] for item in batch.items) == stats.received_messages
    assert batch.index == 0 and rx.poll(1 << 20) is None
    assert stats.ring_packets == stats.ring_msgs == 0


def test_poll_takes_at_least_one_packet_up_to_max_messages():
    stats = IngestStats()
    rx = udp(stats)
    for i, p in enumerate(units(5)):
        rx._enqueue(p, i, SENDER)
    count = stats.received_messages // 5
    first, second = rx.poll(1), rx.poll(2 * count)
    assert (len(first.items), first.messages, first.index) == (1, count, 0)
    assert (len(second.items), second.index) == (2, 1)
    assert stats.ring_packets == 2 and rx.poll(10**9).index == 2


def test_ring_overflow_counts_ring_dropped_messages():
    stats = IngestStats()
    rx = udp(stats, ring_size=3)
    for i, p in enumerate(units(5)):
        rx._enqueue(p, i, SENDER)
    assert stats.packets == 5 and stats.ring_packets == 3
    assert stats.ring_dropped == 2 * (stats.received_messages // 5)
    assert stats.ring_peak() == 3


def test_udp_idle_feed_ends_input_once_the_ring_is_empty():
    stats = IngestStats()
    rx = udp(stats, keep_alive=False, recv_timeout_s=0.3)
    rx.start()
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
            sender.sendto(units(1)[0], rx._address)
        assert wait(lambda: rx._ended)
        assert not rx.eof  # one packet still queued
        assert rx.poll(100) is not None and rx.eof
    finally:
        rx.stop()


def test_udp_bind_error_raises_in_start():
    rx = udp(IngestStats(), host="203.0.113.1")  # TEST-NET-3: not a local address
    with pytest.raises(OSError):
        rx.start()
    assert rx._thread is None


def test_socket_and_daqiri_receive_identical_units(monkeypatch):
    rows = np.zeros(3, EVENT_DTYPE)
    rows["timestamp_ns"] = MIDNIGHT + 34200 * 10**9 + np.arange(3)
    rows["kind"] = [QUOTE, TRADE, CONDITION]
    rows["symbol_id"] = [0, 1, 2]
    for f in ("bid", "ask", "bid_size", "ask_size", "price", "volume"):
        rows[f] = np.nan
    rows["bid"][0], rows["ask"][0], rows["bid_size"][0], rows["ask_size"][0] = 1.5, 1.6, 10, 20
    rows["price"][1], rows["volume"][1] = 412345.12, 300
    data, _ = pack_events(rows, 1, MIDNIGHT)
    received = []

    stats = IngestStats()
    rx = udp(stats)
    rx.start()
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
            sender.sendto(data, rx._address)
        assert wait(lambda: stats.packets == 1)
        received.append(rx.poll(1024).items[0])
    finally:
        rx.stop()

    fake = make_fake_daqiri([[udp_frame(data, src_port=SENDER[1], src_ip=SENDER[0])]])
    monkeypatch.setitem(sys.modules, "daqiri", fake)
    stats = IngestStats()
    rx = daqiri(fake, stats)
    rx.start()
    try:
        assert wait(lambda: stats.packets == 1)
        received.append(rx.poll(1024).items[0])
    finally:
        rx.stop()
    assert received[1][3] == SENDER and received[0][0] == received[1][0] == data
    events = [
        decode_packet(item[0], item[1], generation=1, midnight_ns=MIDNIGHT, universe=UNIVERSE)
        for item in received
    ]
    assert events[0][1] == events[1][1] == received[0][2] == received[1][2]
    for field in EVENT_DTYPE.names:
        np.testing.assert_equal(events[0][0][0].events[field], events[1][0][0].events[field])


def test_daqiri_receiver_strips_frames_counts_gaps_and_frees_bursts(monkeypatch):
    first, n1 = unit(1, [(0, 108.42), (1, 155.01)])
    later, _ = unit(n1 + 4, [(0, 108.50)])  # sequences n1+1 .. n1+3 never arrive
    arp = b"\xff" * 12 + b"\x08\x06" + b"\x00" * 60
    frames = [arp, udp_frame(first, src_port=55555), udp_frame(later, src_port=55555, ihl=6)]
    fake = make_fake_daqiri([frames[:2], frames[2:]])
    monkeypatch.setitem(sys.modules, "daqiri", fake)
    stats = IngestStats()
    rx = daqiri(fake, stats)
    rx.start()
    try:
        assert wait(lambda: stats.packets == 2)
        assert stats.dropped == 3 and stats.malformed_packets == 0
        assert stats.messages == 0  # decode is SbeDecodeOp's job, not the drain's
        assert stats.ring_fill() == stats.received_messages
        assert wait(lambda: len(fake.freed) == 2)  # every burst returned to the pool
    finally:
        rx.stop()
    assert fake.shutdowns == 1 and fake.configs == [{"fake": True}]


def test_daqiri_read_error_still_frees_the_burst(monkeypatch):
    fake = make_fake_daqiri([[udp_frame(units(1)[0], src_port=55555)]])
    fake.get_packet_bytes = lambda *_: ("INTERNAL_ERROR", b"")
    monkeypatch.setitem(sys.modules, "daqiri", fake)
    stats = IngestStats()
    rx = daqiri(fake, stats)
    rx.start()
    try:
        assert wait(lambda: stats.error is not None)
        assert stats.error == "DAQIRI RX failed: get_packet_bytes returned INTERNAL_ERROR"
        assert len(fake.freed) == 1 and wait(lambda: rx._ended)
        assert not rx.probe().alive()
    finally:
        rx.stop()


def test_daqiri_fatal_rx_error_is_reported_instead_of_idle(monkeypatch):
    fake = make_fake_daqiri()
    fake.get_rx_burst = lambda *_: ("INTERNAL_ERROR", None)
    monkeypatch.setitem(sys.modules, "daqiri", fake)
    stats = IngestStats()
    rx = daqiri(fake, stats)
    rx.start()
    try:
        assert wait(lambda: stats.error is not None)
        assert "get_rx_burst returned INTERNAL_ERROR" in stats.error
    finally:
        rx.stop()
    assert fake.shutdowns == 1


def test_daqiri_start_failures_shut_down(monkeypatch):
    fake = make_fake_daqiri()
    fake.daqiri_init = lambda cfg: "BAD_CONFIG"
    monkeypatch.setitem(sys.modules, "daqiri", fake)
    with pytest.raises(RuntimeError, match="daqiri_init failed: BAD_CONFIG"):
        daqiri(fake, IngestStats()).start()
    assert fake.shutdowns == 0  # nothing was initialized

    fake = make_fake_daqiri()
    fake.get_port_id = lambda name: -1
    monkeypatch.setitem(sys.modules, "daqiri", fake)
    with pytest.raises(RuntimeError, match="interface 'rx_port' not found"):
        daqiri(fake, IngestStats()).start()
    assert fake.shutdowns == 1

    # The helper mask cannot be restored after the drain started: it stops before shutdown.
    real, uses = host_receivers.thread_mask, []

    @contextlib.contextmanager
    def unrestorable(cpus=None):
        uses.append(cpus)
        with real(cpus):
            yield
        if len(uses) == 2:
            raise RuntimeError("thread mask not restored")

    fake = make_fake_daqiri()
    shutdown = fake.shutdown

    def shutdown_after_the_drain():  # counted only once the drain thread has stopped
        if not rx.probe().alive():
            shutdown()

    fake.shutdown = shutdown_after_the_drain
    monkeypatch.setitem(sys.modules, "daqiri", fake)
    monkeypatch.setattr(host_receivers, "thread_mask", unrestorable)
    rx = daqiri(fake, IngestStats())
    with pytest.raises(RuntimeError, match="not restored"):
        rx.start()
    assert fake.shutdowns == 1 and not rx.probe().alive()


def in_thread(cpus: set[int], fn):
    """Run ``fn`` in a helper thread masked to ``cpus``; returns its result and the thread's
    mask afterwards."""
    out = {}

    def body():
        os.sched_setaffinity(0, cpus)
        try:
            out["result"] = fn()
        except BaseException as exc:  # re-raised in the test thread
            out["error"] = exc
        out["mask"] = os.sched_getaffinity(0)

    t = threading.Thread(target=body)
    t.start()
    t.join()
    if "error" in out:
        raise out["error"]
    return out["result"], out["mask"]


@pytest.mark.skipif(len(os.sched_getaffinity(0)) < 2, reason="needs two CPUs")
def test_daqiri_init_pins_are_undone_and_the_drain_uses_the_process_mask(monkeypatch):
    lane, other = sorted(os.sched_getaffinity(0))[:2]
    fake = make_fake_daqiri(pin={other})  # EAL pins its caller to a "DPDK lcore"
    monkeypatch.setitem(sys.modules, "daqiri", fake)
    swept = []
    monkeypatch.setattr(host_receivers, "keep_off_cores", lambda r: swept.append(set(r)))
    stats = IngestStats()
    rx = daqiri(fake, stats, reserved=(3, 9), isolate_cores=True)  # default init/helper cpus

    def start():
        rx.start()
        drain = next(t for t in threading.enumerate() if t.name == "daqiri-host-drain")
        return os.sched_getaffinity(drain.native_id)

    try:
        drain_mask, lane_mask = in_thread({lane}, start)
    finally:
        rx.stop()
    assert lane_mask == {lane}  # the narrowed "lane" mask is restored after daqiri_init
    assert drain_mask == process_mask()  # never the narrowed lane mask
    assert swept == [{3, 9}]


@pytest.mark.skipif(len(os.sched_getaffinity(0)) < 2, reason="needs two CPUs")
def test_daqiri_init_cpus_widen_the_caller_during_init(monkeypatch):
    lane = min(os.sched_getaffinity(0))
    seen = []
    fake = make_fake_daqiri()
    fake.daqiri_init = lambda cfg: seen.append(os.sched_getaffinity(0)) or "SUCCESS"
    monkeypatch.setitem(sys.modules, "daqiri", fake)
    every = sorted(process_mask())
    rx = daqiri(fake, IngestStats(), init_cpus=every, helper_cpus=[lane])

    def start():
        rx.start()
        drain = next(t for t in threading.enumerate() if t.name == "daqiri-host-drain")
        return os.sched_getaffinity(drain.native_id)

    try:
        drain_mask, lane_mask = in_thread({lane}, start)
    finally:
        rx.stop()
    assert seen == [set(every)] and lane_mask == {lane} and drain_mask == {lane}
