# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``GpuIngest.pull()`` on host arrays with ``FakeNativeDrain``: the records path accounts
like the host receivers, releases ring space for bursts with nothing to decode, copies the
drain's stall statistics (every ``poll_gaps_*`` key, 43bf4e1) and replay statistics, and
reports drain errors and the end of input. CPU only, as 4.3's native-records test was.
"""

from __future__ import annotations

import time

import numpy as np
from fakes.fake_drain import FakeNativeDrain, receive_header

from holoscan_finance.ingest.drains import DeviceRing
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
)
JUNK = [(b"\x01\x02", 40001, 1)]  # a framing reject: nothing to decode


def unit(seq: int) -> bytes:
    ev = np.zeros(3, EVENT_DTYPE)
    ev["timestamp_ns"] = MIDNIGHT + 36000 * 10**9 + np.arange(3)
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


def host_ring(capacity: int) -> DeviceRing:
    """A ring of host arrays: pull() only slices it."""
    return DeviceRing(
        slots=np.zeros((capacity, 4), np.uint8),
        header_dev=np.zeros((capacity, 7), np.int64),
        header_host=np.zeros((capacity, 7), np.int64),
        packed_dev=None,
        packed_host=None,
        capacity=capacity,
        max_burst=capacity,
    )


def records_ingest(bursts, *, capacity=16, drain=None, keep_alive=True):
    """A ``GpuIngest`` whose drain published ``bursts`` into consecutive ring rows."""
    ring = host_ring(capacity)
    drain = drain or FakeNativeDrain(capacity=capacity)
    pos = 0
    for burst, recv_ns in bursts:
        header = receive_header(burst).T
        ring.header_host[pos : pos + len(burst)] = header
        ring.header_dev[pos : pos + len(burst)] = header
        drain.publish(pos, len(burst), recv_ns)
        pos += len(burst)
    stats = IngestStats()
    ingest = GpuIngest(
        stats=stats,
        decoder=None,
        ring_size=capacity,
        batch_size=1 << 17,
        keep_alive=keep_alive,
        recv_timeout_s=5.0,
    )
    ingest.attach_drain(drain, ring)
    return ingest, drain, stats


def host_reference(bursts) -> IngestStats:
    stats = IngestStats()
    rx = UdpReceiver(
        host="127.0.0.1",
        port=1,
        recv_buf=65536,
        ring_size=10**6,
        keep_alive=True,
        recv_timeout_s=5.0,
        stats=stats,
    )
    for burst, recv_ns in bursts:
        for data, port, ip in burst:
            rx._enqueue(data, recv_ns, (f"192.168.100.{ip}", port))
    return stats


def test_records_account_like_the_host_path_and_release_ring_space():
    a, nxt = stream(1, 4)
    b, _ = stream(nxt, 3)
    bursts = [(JUNK, 10**9), (a, 10**9 + 1), (JUNK, 10**9 + 2), (b, 2 * 10**9)]
    ingest, drain, stats = records_ingest(bursts)
    ingest.pull()
    reference = host_reference(bursts)
    for name in COUNTERS:
        assert getattr(stats, name) == getattr(reference, name), name
    assert stats.bursts == 4
    # The leading reject is released at once; the later one rides on the burst before it.
    assert drain.released == [1]
    assert [(x.start, x.end) for x in ingest._bursts] == [(1, 6), (6, 9)]
    assert [len(x.index) for x in ingest._bursts] == [len(a), len(b)]
    assert stats.ring_packets == len(a) + len(b)
    assert stats.ring_fill() == reference.ring_fill()
    assert drain.take().shape == (0, 9)  # every record consumed


def test_stall_statistics_copy_every_poll_gap_key():
    drain = FakeNativeDrain(capacity=16)
    drain.poll_gaps = {"poll_gaps_10ms": 3, "poll_gaps_50ms": 2, "poll_gaps_100ms": 1}
    ingest, _, stats = records_ingest([(stream(1, 2)[0], 10**9)], drain=drain)
    ingest.pull()
    native = stats.stalls["native"]
    assert {k: native[k] for k in drain.poll_gaps} == drain.poll_gaps
    assert {"poll_gap_max_ms", "poll_gap_at", "ring_waits", "ring_wait_ms"} <= set(native)
    assert stats.stalls["worst_burst"]["packets"] == 2  # per-record stall timing
    assert stats.replay is None  # the native drain has no replay statistics


def test_drain_error_reaches_the_stats():
    ingest, drain, stats = records_ingest([])
    drain.fail("gather_frame launch failed")
    ingest.pull()
    assert stats.error == "GPU MITCH RX failed: gather_frame launch failed"
    assert not ingest.probe().alive()


class ReplayLikeDrain(FakeNativeDrain):
    """A records drain that knows where its input ends, as the replay drain does."""

    eof = False

    def stats(self) -> dict:
        return super().stats() | {"eof": self.eof, "replay": {"sent": self._packets}}


def test_end_of_input_needs_keep_alive_off_and_an_empty_queue():
    silent = FakeNativeDrain(capacity=16)
    ingest, _, _ = records_ingest([(stream(1, 2)[0], 1)], drain=silent, keep_alive=False)
    silent._last_rx_ns = time.monotonic_ns() - 6 * 10**9  # longer than recv_timeout_s
    ingest.pull()
    assert ingest._input_ended and not ingest.eof  # two packets still queued for decode

    ingest, _, _ = records_ingest([(JUNK, 1)], drain=silent, keep_alive=False)
    silent._last_rx_ns = time.monotonic_ns() - 6 * 10**9
    ingest.pull()
    assert ingest.eof

    replay = ReplayLikeDrain(capacity=16)
    ingest, _, stats = records_ingest([(JUNK, 1)], drain=replay, keep_alive=False)
    ingest.pull()
    assert not ingest.eof and stats.replay == {"sent": 1}
    replay.eof = True
    ingest._checked_ns = 0  # the statistics are copied every 100 ms
    ingest.pull()
    assert ingest.eof

    replay = ReplayLikeDrain(capacity=16)
    replay.eof = True
    ingest, _, _ = records_ingest([(JUNK, 1)], drain=replay, keep_alive=True)
    ingest.pull()
    assert not ingest.eof  # keep_alive: the stream never ends


def test_probe_reads_the_drain():
    ingest, drain, stats = records_ingest([(stream(1, 3)[0], 1)])
    probe = ingest.probe()
    assert probe.alive() is False  # never started
    drain.running = True
    assert probe.alive() and probe.pending() == 3 and probe.ring_peak() == 3
    drain.release(3)
    probe.reset_peak()
    assert probe.pending() == 0 and probe.ring_peak() == 0
    stats.attach(probe)
    assert stats.pending_packets() == 0 and stats.drain_alive()


def test_a_pushing_drain_ends_input_and_reports_its_error():
    stats = IngestStats()
    ingest = GpuIngest(
        stats=stats, decoder=None, ring_size=16, batch_size=64, keep_alive=False, recv_timeout_s=1
    )
    probe = ingest.probe()
    ingest.account_burst(None, None, receive_header(stream(1, 2)[0]), 5)
    assert probe.alive() and probe.pending() == 2
    ingest.pull()  # nothing to pull: the Python drain accounts from its own thread
    ingest.end_input("GPU MITCH RX failed: get_rx_burst returned INTERNAL_ERROR")
    assert not probe.alive() and stats.error.endswith("INTERNAL_ERROR")
    assert not ingest.eof  # two packets still queued
