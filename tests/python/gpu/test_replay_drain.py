# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``DeviceReplayDrain``: the native ``Drain`` protocol over pcap frames in GPU memory.

The ring wrap rule, the gather (same kernel, so the same header as the host strip path),
hold-when-full on a condition variable (no CPU while held, woken by ``release``), the
DAQIRI burst timeout, the NIC-buffer drop model, the loop boundary (``on_loop`` only before
pass >= 1, after everything was released, with the port rewritten) and the statistics.
"""

from __future__ import annotations

import socket
import time

import numpy as np
import pytest
from fakes.fake_drain import receive_header

from holoscan_finance.ingest.accounting import ReceiveAccounting
from holoscan_finance.ingest.drains import (
    DRAIN_FIELDS,
    DRAIN_STATS_KEYS,
    DeviceFrames,
    DeviceReplayDrain,
    DeviceRing,
)
from holoscan_finance.marketdata import gpu_mitch
from holoscan_finance.marketdata.framing import PcapFrames, udp_frame, write_pcap
from holoscan_finance.nyq.mapping import day_start_ns
from holoscan_finance.nyq.synthetic import synthetic_stream
from holoscan_finance.telemetry.stats import IngestStats

pytestmark = pytest.mark.gpu

MIDNIGHT = day_start_ns("2025-04-01")
SRC_IP = "10.0.0.1"
SRC_IP_INT = int.from_bytes(socket.inet_aton(SRC_IP), "big")


def payloads(n: int, seed: int = 3) -> list[bytes]:
    return synthetic_stream(np.random.default_rng(seed), n, 50, MIDNIGHT)


def device_frames(tmp_path, units: list[bytes], src_port: int = 40000) -> DeviceFrames:
    path = tmp_path / "replay.pcap"
    write_pcap(path, [udp_frame(u, src_port=src_port, src_ip=SRC_IP) for u in units])
    return DeviceFrames.upload(PcapFrames.read(path), 0)


def make_drain(tmp_path, units, *, capacity=64, burst=16, pps=0, loops=1, **kw):
    ring = DeviceRing.allocate(capacity, 0)
    drain = DeviceReplayDrain(
        device_frames(tmp_path, units),
        ring,
        gather_fn=kw.pop("gather_fn", None) or gpu_mitch._kernels().get_function("gather_frame"),
        pps=pps,
        loops=loops,
        burst_packets=burst,
        rx_timeout_us=kw.pop("rx_timeout_us", 20000),
        **kw,
    )
    return drain, ring


class Consumer:
    """Takes records like ``GpuIngest.pull()``; copies each burst's header and payloads out
    of the ring before releasing it."""

    def __init__(self, drain: DeviceReplayDrain, ring: DeviceRing) -> None:
        self.drain, self.ring = drain, ring
        self.records: list[np.ndarray] = []
        self.headers: list[np.ndarray] = []
        self.slots: list[np.ndarray] = []

    def poll(self, release: bool = True) -> int:
        taken = self.drain.take()
        for record in taken:
            start, n = int(record[0]), int(record[1])
            slot = start % self.ring.capacity
            self.records.append(record)
            self.headers.append(self.ring.header_host[slot : slot + n].copy())
            self.slots.append(self.ring.slots[slot : slot + n].get())
            if release:
                self.drain.release(start + n)
        return len(taken)

    def run(self, until, timeout: float = 10.0, release: bool = True) -> None:
        if release and self.records:  # what earlier runs held back
            self.drain.release(int(self.records[-1][0] + self.records[-1][1]))
        deadline = time.monotonic() + timeout
        while not until():
            assert time.monotonic() < deadline, "timed out"
            self.poll(release)
            time.sleep(0.0005)
        self.poll(release)

    @property
    def header(self) -> np.ndarray:
        return np.concatenate(self.headers)


def test_bursts_never_wrap_and_gather_like_the_host_strip(tmp_path):
    units = payloads(200)
    drain, ring = make_drain(tmp_path, units, capacity=100, burst=30)
    consumer = Consumer(drain, ring)
    drain.start()
    try:
        consumer.run(lambda: drain.stats()["eof"])
    finally:
        drain.stop()
    records = np.array(consumer.records)
    assert records.dtype == np.int64 and records.shape[1] == len(DRAIN_FIELDS)
    assert records[:, 0].tolist() == [0, 30, 60, 100, 130, 160, 200]  # padded to the wrap
    assert records[:, 1].tolist() == [30] * 6 + [20]
    expected = receive_header([(u, 40000, SRC_IP_INT) for u in units]).T
    np.testing.assert_array_equal(consumer.header, expected)
    gathered = np.concatenate(consumer.slots)
    for i, u in enumerate(units):
        assert gathered[i, : len(u)].tobytes() == u
    st = drain.stats()
    assert set(DRAIN_STATS_KEYS) <= set(st) and st["error"] == "" and not st["running"]
    assert (st["bursts"], st["packets"], st["fill"]) == (7, 200, 0)
    assert st["replay"]["sent"] == 200 and st["replay"]["eof"] and st["replay"]["pass_index"] == 0


def test_a_full_ring_holds_the_burst_without_cpu_until_release(tmp_path):
    drain, ring = make_drain(tmp_path, payloads(200), capacity=64, burst=16)
    consumer = Consumer(drain, ring)
    drain.start()
    try:
        consumer.run(lambda: len(consumer.records) == 4, release=False)
        time.sleep(0.05)
        assert drain.stats()["fill"] == 64 and drain.take().shape == (0, 9)
        cpu0 = drain.stats()["replay"]["thread_cpu_s"]
        time.sleep(0.2)
        cpu1 = drain.stats()["replay"]["thread_cpu_s"]
        assert cpu1 - cpu0 < 0.010  # < 5% of the 200 ms held: a condition wait, not a poll
        released = time.monotonic_ns()
        drain.release(16)
        consumer.run(lambda: len(consumer.records) == 5, release=False)
        woke = consumer.records[-1][DRAIN_FIELDS.index("t_frames")] - released
        assert woke < 2_000_000  # release() signals the drain
        assert consumer.records[-1][DRAIN_FIELDS.index("wait_ns")] > 200_000_000
        consumer.run(lambda: drain.stats()["eof"])
    finally:
        drain.stop()
    st = drain.stats()
    assert st["ring_waits"] >= 1 and st["ring_wait_ms"] > 200 and st["ring_peak"] == 64
    assert st["replay"]["lag_max_ms"] > 200  # gathered long after it was due


def test_take_and_release_protocol():
    ring = DeviceRing.allocate(64, 0)
    frames = DeviceFrames(None, np.zeros(1, np.uint64), np.zeros(1, np.uint64), None)
    drain = DeviceReplayDrain(frames, ring, gather_fn=None, pps=0, loops=1, burst_packets=16,
                              rx_timeout_us=0)  # fmt: skip
    assert drain.take().shape == (0, 9) and drain.take().dtype == np.int64
    drain._written = 40
    drain.release(30)
    drain.release(10)  # positions only move forward
    assert drain.stats()["fill"] == 10
    with pytest.raises(ValueError, match="burst_packets"):
        DeviceReplayDrain(frames, ring, gather_fn=None, pps=0, loops=1, burst_packets=65,
                          rx_timeout_us=0)  # fmt: skip


def test_a_paced_burst_closes_at_the_rx_timeout(tmp_path):
    # 1,000 packets/s and 20 ms: a burst holds the frames due 20 ms after its first one.
    drain, ring = make_drain(tmp_path, payloads(60), burst=4096, capacity=4096, pps=1000)
    consumer = Consumer(drain, ring)
    drain.start()
    try:
        consumer.run(lambda: drain.stats()["eof"])
    finally:
        drain.stop()
    assert [int(r[1]) for r in consumer.records] == [21, 21, 18]
    assert (
        consumer.header[:, 5].tolist()
        == receive_header([(u, 40000, SRC_IP_INT) for u in payloads(60)])[5].tolist()
    )


def test_nic_buffers_drop_the_oldest_due_frames_as_sequence_gaps(tmp_path):
    units = payloads(400)
    drain, ring = make_drain(tmp_path, units, capacity=32, burst=16, pps=20000, nic_bufs=40)
    consumer = Consumer(drain, ring)
    drain.start()
    try:
        consumer.run(lambda: len(consumer.records) == 2, release=False)  # the ring is full
        time.sleep(0.05)  # every frame comes due meanwhile
        consumer.run(lambda: drain.stats()["eof"])
    finally:
        drain.stop()
    replay = drain.stats()["replay"]
    assert replay["nic_dropped"] > 0 and replay["sent"] + replay["nic_dropped"] == 400
    stats = IngestStats()
    acct = ReceiveAccounting(stats)
    for header in consumer.headers:
        acct.observe_burst(header.T, 1)
    seqs = consumer.header[:, 5]
    kept = {int(s) for s in seqs}
    full = receive_header([(u, 40000, SRC_IP_INT) for u in units])
    lost = sum(int(c) for s, c in zip(full[5], full[4]) if int(s) not in kept)
    assert stats.packets == replay["sent"] and stats.dropped == lost > 0


def test_loop_boundary_waits_for_every_release_and_rewrites_the_port(tmp_path):
    calls = []
    drain = None

    def on_loop(pass_index: int) -> int:
        calls.append((pass_index, drain.stats()["fill"], time.monotonic_ns()))
        return 41000

    units = payloads(50)
    drain, ring = make_drain(tmp_path, units, burst=16, loops=2, on_loop=on_loop)
    consumer = Consumer(drain, ring)
    drain.start()
    try:
        consumer.run(lambda: len(consumer.records) == 4, release=False)  # all of pass 0
        time.sleep(0.05)
        assert calls == []  # pass 0 not released yet
        released = time.monotonic_ns()
        consumer.run(lambda: drain.stats()["eof"])
    finally:
        drain.stop()
    assert [(c[0], c[1]) for c in calls] == [(1, 0)] and calls[0][2] >= released
    ports = consumer.header[:, 1]
    assert ports[:50].tolist() == [40000] * 50 and ports[50:].tolist() == [41000] * 50
    replay = drain.stats()["replay"]
    assert replay["pass_index"] == 1 and replay["sent"] == 100 and replay["loop_wait_ms"] > 0

    calls.clear()
    drain, ring = make_drain(tmp_path, units, burst=16, loops=1, on_loop=on_loop)
    consumer = Consumer(drain, ring)
    drain.start()
    try:
        consumer.run(lambda: drain.stats()["eof"])
    finally:
        drain.stop()
    assert calls == []  # never before pass 0, never after the last pass


def test_a_drain_failure_is_reported_in_stats(tmp_path):
    def broken(*args, **kwargs):
        raise RuntimeError("gather_frame launch failed")

    drain, _ = make_drain(tmp_path, payloads(10), gather_fn=broken)
    drain.start()
    deadline = time.monotonic() + 5
    while drain.stats()["running"] and time.monotonic() < deadline:
        time.sleep(0.005)
    drain.stop()
    st = drain.stats()
    assert st["error"] == "gather_frame launch failed" and not st["running"] and not st["eof"]


@pytest.mark.slow
@pytest.mark.timing
def test_paced_replay_keeps_the_requested_rate(tmp_path):
    drain, ring = make_drain(tmp_path, payloads(4000), capacity=8192, burst=4096, pps=20000)
    consumer = Consumer(drain, ring)
    started = time.monotonic()
    drain.start()
    try:
        consumer.run(lambda: drain.stats()["eof"])
    finally:
        drain.stop()
    assert abs(time.monotonic() - started - 0.2) < 0.01  # 4,000 frames at 20,000/s, +-5%
    assert drain.stats()["replay"]["lag_p99_ms"] < 5
