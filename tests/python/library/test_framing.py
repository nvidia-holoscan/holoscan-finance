# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ethernet/IPv4/UDP framing and pcap files: the host strip path, synthetic frames, pcap
round trips and IHL-aware source ports."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from holoscan_finance.marketdata.framing import (
    PcapFrames,
    strip_udp_datagram,
    udp_frame,
    write_pcap,
)
from holoscan_finance.marketdata.records import EVENT_DTYPE, QUOTE
from holoscan_finance.nyq import mapping as replay

MIDNIGHT = replay.day_start_ns("2025-04-01")
PEER = ("192.168.100.1", 55555)


def unit(sequence=1):
    rows = np.zeros(1, dtype=EVENT_DTYPE)
    rows["kind"] = QUOTE
    rows["timestamp_ns"] = MIDNIGHT + 34200 * 10**9
    rows["bid"], rows["ask"], rows["bid_size"], rows["ask_size"] = 108.42, 108.44, 100, 100
    return replay.pack_events(rows, sequence, MIDNIGHT)[0]


def frame(payload, ihl=5):
    return udp_frame(payload, src_port=PEER[1], src_ip=PEER[0], dst_ip="192.168.100.2", ihl=ihl)


def test_strip_udp_datagram_follows_ihl():
    payload = unit()
    assert strip_udp_datagram(frame(payload)) == (payload, PEER)
    # IPv4 options (ihl > 5) shift the payload
    assert strip_udp_datagram(frame(payload, ihl=6)) == (payload, PEER)


def test_strip_udp_datagram_rejects_junk():
    assert strip_udp_datagram(b"") is None
    assert strip_udp_datagram(b"\x00" * 60) is None  # wrong ethertype
    arp = b"\xff" * 12 + b"\x08\x06" + b"\x00" * 60
    assert strip_udp_datagram(arp) is None
    ipv6 = bytearray(frame(unit()))
    ipv6[12:14] = b"\x86\xdd"
    assert strip_udp_datagram(bytes(ipv6)) is None


def test_strip_respects_datagram_lengths_and_rejects_fragments():
    payload = unit()
    f = frame(payload)
    strip = lambda f: strip_udp_datagram(f, minimum_payload=8)  # noqa: E731
    assert strip(f + b"\x00" * 24) == (payload, PEER)  # Ethernet padding / FCS
    assert strip(f[:-1]) is None
    for offset, value in (
        (14, b"\x65"),  # IP version 6
        (20, b"\x20\x00"),  # more fragments
        (20, b"\x00\x01"),  # fragment offset
        (23, b"\x06"),  # TCP
        (38, b"\x00\x08"),  # UDP length without payload
        (38, b"\xff\xff"),  # UDP length beyond the datagram
        (16, b"\x00\x14"),  # IP total length without UDP
    ):
        broken = bytearray(f)
        broken[offset : offset + len(value)] = value
        assert strip(bytes(broken)) is None
    assert strip_udp_datagram(frame(b"\x00" * 4), minimum_payload=8) is None


def test_udp_frame_layout_and_valid_ip_checksum():
    payload = unit()
    for ihl in (5, 6, 15):
        f = udp_frame(payload, src_port=40001, dst_port=26401, ihl=ihl)
        ip = f[14 : 14 + ihl * 4]
        assert len(f) == 14 + ihl * 4 + 8 + len(payload)
        assert (
            ip[0] == 0x40 | ihl and ip[9] == 17 and ip[12:20] == bytes([10, 0, 0, 1, 10, 0, 0, 2])
        )
        total = sum(struct.unpack(f">{len(ip) // 2}H", ip))
        while total > 0xFFFF:
            total = (total & 0xFFFF) + (total >> 16)
        assert total == 0xFFFF  # the one's-complement sum of a valid header
        assert struct.unpack_from(">HH", f, 14 + ihl * 4) == (40001, 26401)
        assert strip_udp_datagram(f, minimum_payload=8) == (payload, ("10.0.0.1", 40001))
    for ihl in (4, 16):
        with pytest.raises(ValueError, match="IHL"):
            udp_frame(payload, src_port=1, ihl=ihl)
    with pytest.raises(ValueError, match="datagram"):
        udp_frame(b"\x00" * 65508, src_port=1)


def test_pcap_round_trip_with_limit(tmp_path):
    frames = [frame(unit(i + 1)) for i in range(5)] + [b"\x00" * 60]
    stamps = [1_700_000_000_123_456_789 + i for i in range(len(frames))]
    path = tmp_path / "x.pcap"
    assert write_pcap(path, frames, ts_ns=stamps) == len(frames)
    got = PcapFrames.read(path)
    assert len(got) == len(frames)
    assert [got.frame(i) for i in range(len(got))] == frames
    assert got.ts_ns.tolist() == stamps
    assert got.lengths.tolist() == [len(f) for f in frames]
    assert got.data.dtype == np.uint8 and got.offsets.dtype == np.int64
    assert got.payloads() == [unit(i + 1) for i in range(5)]
    head = PcapFrames.read(path, limit=2)
    assert len(head) == 2 and head.frame(1) == frames[1]
    assert len(head.data) == head.offsets[-1] + head.lengths[-1]
    with pytest.raises(ValueError):
        write_pcap(tmp_path / "y.pcap", frames, ts_ns=stamps[:-1])
    assert write_pcap(tmp_path / "z.pcap", frames[:1]) == 1
    assert PcapFrames.read(tmp_path / "z.pcap").ts_ns.tolist() == [0]


def test_microsecond_big_endian_pcap_and_a_truncated_tail(tmp_path):
    frames = [frame(unit(1)), frame(unit(2))]
    data = struct.pack(">IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1)
    for i, f in enumerate(frames):
        data += struct.pack(">IIII", 1_700_000_000 + i, 250_000, len(f), len(f)) + f
    path = tmp_path / "us.pcap"
    path.write_bytes(data + struct.pack(">IIII", 1, 1, 100, 100) + b"\x00" * 10)
    got = PcapFrames.read(path)
    assert [got.frame(i) for i in range(len(got))] == frames  # the cut-off record is dropped
    assert got.ts_ns.tolist() == [1_700_000_000_250_000_000, 1_700_000_001_250_000_000]


def test_files_that_are_not_ethernet_pcaps_are_rejected(tmp_path):
    (tmp_path / "short").write_bytes(b"\xd4\xc3\xb2\xa1")
    (tmp_path / "magic").write_bytes(b"\x00" * 24)
    (tmp_path / "cooked").write_bytes(struct.pack("<IHHiIII", 0xA1B23C4D, 2, 4, 0, 0, 65535, 113))
    for name in ("short", "magic", "cooked"):
        with pytest.raises(ValueError):
            PcapFrames.read(tmp_path / name)


def test_source_port_offsets_follow_ihl_and_skip_non_udp(tmp_path):
    tcp = bytearray(frame(unit(), ihl=5))
    tcp[23] = 6
    fragment = bytearray(frame(unit()))
    fragment[20:22] = b"\x00\x10"  # a non-first fragment carries no UDP header
    frames = [
        udp_frame(unit(), src_port=4000, ihl=6),
        udp_frame(unit(), src_port=5000),
        bytes(tcp),
        bytes(fragment),
        b"\x00" * 30,
        udp_frame(unit(), src_port=5000, ihl=7),
        udp_frame(unit(), src_port=4000),
    ]
    write_pcap(tmp_path / "p.pcap", frames)
    got = PcapFrames.read(tmp_path / "p.pcap")
    rel = got.udp_src_port_offsets()
    assert rel.tolist() == [38, 34, -1, -1, -1, 42, 34]
    for i in np.flatnonzero(rel >= 0):
        f = got.frame(i)
        assert struct.unpack_from(">H", f, rel[i])[0] == strip_udp_datagram(f)[1][1]
    assert got.dominant_src_port() == 4000  # a 2-2 tie goes to the lower port
    write_pcap(tmp_path / "q.pcap", [bytes(tcp)])
    assert PcapFrames.read(tmp_path / "q.pcap").dominant_src_port() is None
