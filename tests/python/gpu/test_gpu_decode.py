# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The GPU MITCH decode kernel is bit-compatible with the host decoder: every field and every
rejection rule, via golden bytes, randomized streams and mutations; the Ethernet/IPv4/UDP
gather kernel against ``strip_udp_datagram`` (from 4.3 ``test_equities_gpu``). The full-day
version of these checks is ``python -m holoscan_finance.tools.gpu_validate``.

``to_slots``, ``host_decode`` and ``compare`` are the test oracle the other GPU tests share.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest
from library.test_mitch_codec import GOLDEN

from holoscan_finance.marketdata import mitch_codec as unit
from holoscan_finance.marketdata.framing import strip_udp_datagram, udp_frame
from holoscan_finance.marketdata.gpu_mitch import FIELDS, SLOT_BYTES, decode_packets, receive_frames
from holoscan_finance.marketdata.records import EVENT_DTYPE
from holoscan_finance.nyq import mapping as replay
from holoscan_finance.nyq.synthetic import fuzz, synthetic_stream

pytestmark = pytest.mark.gpu
cp = pytest.importorskip("cupy")

MIDNIGHT = replay.day_start_ns("2025-04-01")
UNIVERSE = 50


def to_slots(payloads):
    """A device slot buffer (one payload per row) and its int32 lengths."""
    host = np.zeros((len(payloads), SLOT_BYTES), np.uint8)
    for i, p in enumerate(payloads):
        host[i, : len(p)] = np.frombuffer(p, np.uint8)
    return cp.asarray(host), cp.asarray([len(p) for p in payloads], cp.int32)


def host_decode(payloads):
    """Per packet: status (0 decoded, 1 framing, 2 content), observations, wire messages and
    sequence; plus every observation, all from the host decoder."""
    n = len(payloads)
    status, nev = np.zeros(n, np.int32), np.zeros(n, np.int32)
    wire, seq = np.zeros(n, np.int32), np.zeros(n, np.uint32)
    rows = []
    for i, data in enumerate(payloads):
        try:
            _, seq[i], wire[i] = unit.inspect_unit(data)
        except (ValueError, struct.error):
            status[i] = 1
            continue
        try:
            (batch,), _ = replay.decode_packet(
                data, 0, generation=1, midnight_ns=MIDNIGHT, universe=UNIVERSE
            )
        except (ValueError, struct.error):
            status[i] = 2
            continue
        nev[i] = len(batch.events)
        rows.append(batch.events)
    events = np.concatenate(rows) if rows else np.zeros(0, EVENT_DTYPE)
    return status, nev, wire, seq, events


def compare(host, gpu):
    """Mismatch counts between ``host_decode`` and ``decode_packets`` (all zero = identical);
    floats are compared as bit patterns, so NaN placement and signed zeros count."""
    status, nev, wire, seq, events = host
    out = {"status_mismatch": int((status != gpu.status).sum())}
    ok, framed = status == 0, status != 1
    out["events_per_packet_mismatch"] = int((nev[ok] != gpu.events_per_packet[ok]).sum())
    out["wire_count_mismatch"] = int((wire[framed] != gpu.wire_count[framed]).sum())
    out["sequence_mismatch"] = int((seq[framed] != gpu.sequence[framed]).sum())
    out["event_count_mismatch"] = int(len(events) != gpu.total_events)
    if out["event_count_mismatch"]:
        return out
    native = events.astype(events.dtype.newbyteorder("="))
    for name, dt in FIELDS:
        h, g = native[name], cp.asnumpy(gpu.columns[name])
        if np.dtype(dt).kind == "f":
            bad = h.view(np.int64) != g.view(np.int64)
        else:
            bad = h.astype(np.int64) != g.astype(np.int64)
        out[f"field_{name}_mismatch"] = int(bad.sum())
    return out


def decode_both(payloads):
    slots, lens = to_slots(payloads)
    gpu = decode_packets(slots, lens, midnight_ns=MIDNIGHT, universe=UNIVERSE)
    return host_decode(payloads), gpu


def assert_identical(host, gpu):
    bad = {k: v for k, v in compare(host, gpu).items() if v}
    assert not bad, bad


def test_golden_unit_decodes_identically():
    host, gpu = decode_both([GOLDEN])
    assert_identical(host, gpu)
    assert gpu.total_events == 1
    assert float(gpu.columns["bid"][0]) == -1.0


def test_random_streams_and_mutations_are_bit_identical():
    payloads = synthetic_stream(np.random.default_rng(20260930), 400, UNIVERSE, MIDNIGHT)
    host, gpu = decode_both(payloads)
    assert_identical(host, gpu)
    assert (host[0] == 0).all()
    host, gpu = decode_both(fuzz(payloads, 6000, seed=11))
    assert_identical(host, gpu)
    # Every rejection stage is exercised, not just the happy path.
    assert set(np.unique(host[0]).tolist()) == {0, 1, 2}


def test_huge_prices_round_like_python_integer_division():
    data = bytearray(GOLDEN)
    for magnitude in (2**53 + 1, 2**60 + 12345, 2**63 - 1, 123456789012345678):
        struct.pack_into("<Q", data, 24, magnitude | (1 << 63))
        struct.pack_into("<Q", data, 36, magnitude)
        assert_identical(*decode_both([bytes(data)]))


def test_gather_matches_host_strip():
    good = synthetic_stream(np.random.default_rng(3), 20, UNIVERSE, MIDNIGHT)
    frames = [udp_frame(p, src_port=40000 + i) for i, p in enumerate(good)]
    tcp, fragment, ipv6 = (bytearray(udp_frame(good[k], src_port=1)) for k in range(3))
    tcp[23] = 6
    fragment[20:22] = b"\x20\x00"
    ipv6[12:14] = b"\x86\xdd"
    frames += [
        bytes(tcp),
        bytes(fragment),
        bytes(ipv6),
        udp_frame(good[3], src_port=1)[: 14 + 20 + 8 + len(good[3]) - 10],  # truncated
        udp_frame(b"\x00" * 4, src_port=1),  # payload shorter than a MITCH header
        udp_frame(good[4], src_port=2, ihl=6),  # IPv4 options
        udp_frame(good[5], src_port=3) + b"\x00" * 4,  # Ethernet padding / FCS
        udp_frame(GOLDEN[:-1], src_port=4),  # UDP fine, MITCH framing not
    ]
    stride = 2048
    host_buf = np.zeros((len(frames), stride), np.uint8)
    for i, f in enumerate(frames):
        host_buf[i, : len(f)] = np.frombuffer(f, np.uint8)
    dev = cp.asarray(host_buf)
    addresses = dev.data.ptr + np.arange(len(frames), dtype=np.uint64) * stride
    packed = cp.asarray(np.concatenate([addresses, np.array([len(f) for f in frames], np.uint64)]))
    slots = cp.zeros((len(frames), SLOT_BYTES), cp.uint8)
    header = receive_frames(packed, len(frames), slots).get()
    slots = slots.get()
    for i, f in enumerate(frames):
        expected = strip_udp_datagram(f, minimum_payload=8)
        if expected is None:
            assert header[i, 0] < 0, i
            continue
        payload, peer = expected
        assert len(payload) == header[i, 0], i
        assert peer[1] == header[i, 1], i
        assert slots[i, : len(payload)].tobytes() == payload, i
        try:
            group, sequence, count = unit.inspect_unit(payload)
            assert header[i, 3] == 0, i
            assert (header[i, 4], header[i, 5], header[i, 6]) == (count, sequence, group), i
        except ValueError:
            assert header[i, 3] == 1, i
