# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``GpuIngest`` on device ring rows published through ``FakeNativeDrain``: staging (one
zero-copy view of consecutive ring rows, a copy otherwise, including across the ring end),
decode equal to the host decoder, ring release only after the legacy stream synchronized,
the batch size, stale packets and the capture tap.
"""

from __future__ import annotations

import cupy as cp
import numpy as np
import pytest
from fakes.fake_drain import FakeNativeDrain

from holoscan_finance.ingest.drains import DeviceFrames, DeviceRing
from holoscan_finance.ingest.gpu_ingest import GpuIngest
from holoscan_finance.marketdata.framing import PcapFrames, udp_frame, write_pcap
from holoscan_finance.marketdata.gpu_decoder import GpuMitchDecoder
from holoscan_finance.marketdata.gpu_mitch import SLOT_BYTES, receive_frames
from holoscan_finance.marketdata.records import EVENT_DTYPE
from holoscan_finance.nyq.manifest import validate_manifest
from holoscan_finance.nyq.mapping import PROTOCOL, TIMEZONE, day_start_ns, decode_packet
from holoscan_finance.nyq.synthetic import synthetic_stream
from holoscan_finance.telemetry.stats import IngestStats

pytestmark = pytest.mark.gpu

DATE, PORT, UNIVERSE = "2025-04-01", 40000, 50
MIDNIGHT = day_start_ns(DATE)
DIRECTORY = validate_manifest(
    dict(
        format="nyq-equity-replay-v1",
        wire_protocol=PROTOCOL,
        market_timezone=TIMEZONE,
        sender_port=PORT,
        symbols=[f"S{i}" for i in range(UNIVERSE)],
        generation=7,
        date=DATE,
        start_ns=MIDNIGHT,
        end_ns=MIDNIGHT + 86_400 * 10**9,
        events=1,
        profile=[],
    )
)


class RecordingDecoder(GpuMitchDecoder):
    """The real decoder; remembers the slot buffer it was given."""

    def decode(self, slots, lens_i32, src_port, recv_ns, directory):
        self.slots = slots
        return super().decode(slots, lens_i32, src_port, recv_ns, directory)


class SyncCheckingDrain(FakeNativeDrain):
    """Records whether the legacy stream was idle when ring space was released."""

    def release(self, end: int) -> None:
        self.idle_at_release = getattr(self, "idle_at_release", []) + [cp.cuda.Stream.null.done]
        super().release(end)


class Rig:
    """A device ring that a test fills burst by burst, as the native drain would."""

    def __init__(
        self, tmp_path, frames: list[bytes], capacity: int = 64, batch_size: int = 1 << 17
    ):
        path = tmp_path / "frames.pcap"
        write_pcap(path, frames)
        self.frames = DeviceFrames.upload(PcapFrames.read(path), 0)
        self.ring = DeviceRing.allocate(capacity, 0)
        self.drain = SyncCheckingDrain(capacity=capacity)
        self.stats = IngestStats()
        self.decoder = RecordingDecoder(0)
        self.ingest = GpuIngest(
            stats=self.stats,
            decoder=self.decoder,
            ring_size=capacity,
            batch_size=batch_size,
            keep_alive=True,
            recv_timeout_s=5.0,
        )
        self.ingest.attach_drain(self.drain, self.ring)

    def gather(self, lo: int, hi: int, pos: int) -> None:
        """Gather frames ``[lo, hi)`` into the ring at position ``pos`` and publish them."""
        n, slot = hi - lo, pos % self.ring.capacity
        packed = cp.asarray(self.frames.addresses(lo, hi))
        header = receive_frames(packed, n, self.ring.slots[slot : slot + n])
        self.ring.header_dev[slot : slot + n] = header
        self.ring.header_host[slot : slot + n] = header.get()
        self.drain.publish(pos, n, 1_000 + lo)


def units(n: int, seed: int = 5) -> list[bytes]:
    return synthetic_stream(np.random.default_rng(seed), n, UNIVERSE, MIDNIGHT)


def frames(payloads: list[bytes], port: int = PORT) -> list[bytes]:
    return [udp_frame(p, src_port=port) for p in payloads]


def host_events(payloads: list[bytes]) -> np.ndarray:
    out = [
        decode_packet(p, 0, generation=7, midnight_ns=MIDNIGHT, universe=UNIVERSE)[0][0].events
        for p in payloads
    ]
    return np.concatenate(out)


def assert_decoded(batches, payloads: list[bytes]) -> None:
    (batch,) = batches
    expected = host_events(payloads)
    assert batch.generation == 7 and batch.n_packets == len(payloads)
    for name in EVENT_DTYPE.names:
        if name in batch.columns:
            got = cp.asnumpy(batch.columns[name])
            np.testing.assert_array_equal(got, expected[name].astype(got.dtype), err_msg=name)


def test_consecutive_bursts_decode_from_one_zero_copy_view(tmp_path):
    payloads = units(48)
    rig = Rig(tmp_path, frames(payloads))
    for lo in (0, 16, 32):
        rig.gather(lo, lo + 16, lo)
    rig.ingest.pull()
    batches = rig.ingest.next_batch(DIRECTORY)
    assert rig.decoder.slots.data.ptr == rig.ring.slots.data.ptr  # a view of the ring rows
    assert rig.decoder.slots.shape == (48, SLOT_BYTES)
    assert_decoded(batches, payloads)
    assert rig.drain.released == [48] and rig.drain.idle_at_release == [True]
    assert rig.stats.messages == sum(p[2] for p in payloads) and rig.stats.ring_packets == 0
    assert batches[0].packet_recv_ns.tolist() == [1000] * 16 + [1016] * 16 + [1032] * 16


def test_rows_across_the_ring_end_are_copied_not_viewed(tmp_path):
    # Bursts at positions 16 and 32 are consecutive but sit at ring rows 16..31 and 0..15:
    # one view of rows 16..47 would run past the ring.
    payloads = units(48)
    rig = Rig(tmp_path, frames(payloads), capacity=32)
    rig.gather(0, 16, 0)
    rig.ingest.pull()
    assert_decoded(rig.ingest.next_batch(DIRECTORY), payloads[:16])
    rig.gather(16, 32, 16)
    rig.gather(32, 48, 32)
    rig.ingest.pull()
    batches = rig.ingest.next_batch(DIRECTORY)
    assert rig.decoder.slots.shape == (32, SLOT_BYTES)
    assert rig.decoder.slots.data.ptr != rig.ring.slots.data.ptr
    assert_decoded(batches, payloads[16:])
    assert rig.drain.released == [16, 48]


def test_rejected_rows_are_staged_by_copy_and_counted(tmp_path):
    payloads = units(8)
    mixed = frames(payloads[:4]) + [b"\x00" * 60] + frames([b"\x01\x02" * 8]) + frames(payloads[4:])
    rig = Rig(tmp_path, mixed)
    rig.gather(0, len(mixed), 0)
    rig.ingest.pull()
    batches = rig.ingest.next_batch(DIRECTORY)
    assert rig.decoder.slots.shape == (8, SLOT_BYTES)  # only the accepted rows
    assert_decoded(batches, payloads)
    assert (rig.stats.non_udp_frames, rig.stats.malformed_packets) == (1, 1)
    assert rig.drain.released == [len(mixed)]


def test_batch_size_bounds_the_staging_and_stale_packets_are_released(tmp_path):
    payloads = units(32)
    rig = Rig(tmp_path, frames(payloads[:16]) + frames(payloads[16:], port=PORT + 1))
    rig.ingest._batch_size = 1  # one burst per batch
    rig.gather(0, 16, 0)
    rig.gather(16, 32, 16)
    rig.ingest.pull()
    assert_decoded(rig.ingest.next_batch(DIRECTORY), payloads[:16])
    assert rig.ingest.next_batch(DIRECTORY) is None  # another sender: stale
    assert rig.stats.stale_packets == 16 and rig.drain.released == [16, 32]
    assert rig.ingest.next_batch(DIRECTORY) is None and rig.drain.released == [16, 32]

    rig.gather(0, 16, 32)  # no directory: every packet is stale
    rig.ingest.pull()
    assert rig.ingest.next_batch(None) is None and rig.stats.stale_packets == 32


def test_an_active_capture_tap_sees_every_batch(tmp_path):
    class Tap:
        active = True

        def __init__(self) -> None:
            self.batches = []

        def record_gpu_batch(self, batch) -> None:
            self.batches.append(batch)

    payloads = units(16)
    rig = Rig(tmp_path, frames(payloads))
    rig.ingest._tap = tap = Tap()
    rig.gather(0, 16, 0)
    rig.ingest.pull()
    assert tap.batches == rig.ingest.next_batch(DIRECTORY)
