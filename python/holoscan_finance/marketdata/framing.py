# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ethernet II / IPv4 / UDP framing and pcap files, on the host.

``strip_udp_datagram`` is the host strip path; the ``gather_frame`` kernel applies the same
checks on the GPU. ``udp_frame`` and ``write_pcap`` build synthetic captures, and
``PcapFrames`` holds a capture as one contiguous buffer, as the NIC-free replay source
uploads it to the GPU. Source-port offsets are IHL-aware, so frames with IP options parse
correctly.
"""

from __future__ import annotations

import socket
import struct
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import repeat
from pathlib import Path

import numpy as np

# Ethernet header without VLAN; DAQIRI raw RX delivers full L2 frames.
_ETH_HLEN = 14
_ETHERTYPE_IPV4 = b"\x08\x00"
_IPPROTO_UDP = 17
_UDP_HLEN = 8

# Locally administered MACs for synthetic frames; receivers match on IP/UDP only.
_ETH_HEADER = bytes.fromhex("020000000002020000000001") + _ETHERTYPE_IPV4
_LINKTYPE_ETHERNET = 1
# Magic -> (byte order, ns per sub-second unit): microsecond and nanosecond pcap.
_PCAP_MAGIC = {
    b"\xd4\xc3\xb2\xa1": ("<", 1000),
    b"\xa1\xb2\xc3\xd4": (">", 1000),
    b"\x4d\x3c\xb2\xa1": ("<", 1),
    b"\xa1\xb2\x3c\x4d": (">", 1),
}
_PCAP_HEADER = struct.Struct("<IHHiIII")  # written as little-endian nanosecond pcap
_PCAP_HLEN = _PCAP_HEADER.size
_RECORD = struct.Struct("<IIII")  # ts_sec, ts_frac, incl_len, orig_len
_RECORD_DTYPE = np.dtype([("sec", "<u4"), ("frac", "<u4"), ("incl", "<u4"), ("orig", "<u4")])
RECORD_HLEN = _RECORD.size
FRAME_PREFIX = _ETH_HLEN + 20 + _UDP_HLEN  # headers of a udp_frame without IP options


def strip_udp_datagram(
    frame: bytes, minimum_payload: int = 0
) -> tuple[bytes, tuple[str, int]] | None:
    """Return (UDP payload, source peer), respecting IPv4/UDP boundaries."""
    if len(frame) < _ETH_HLEN + 20 + _UDP_HLEN + minimum_payload:
        return None
    if frame[12:14] != _ETHERTYPE_IPV4:
        return None
    ihl = (frame[_ETH_HLEN] & 0x0F) * 4
    ip_length = int.from_bytes(frame[16:18], "big")
    fragment = int.from_bytes(frame[20:22], "big")
    if (
        frame[_ETH_HLEN] >> 4 != 4
        or ihl < 20
        or frame[_ETH_HLEN + 9] != _IPPROTO_UDP
        or fragment & 0x3FFF  # MF or fragment offset: no IP reassembly here
        or ip_length < ihl + _UDP_HLEN + minimum_payload
        or len(frame) < _ETH_HLEN + ip_length
    ):
        return None
    udp = _ETH_HLEN + ihl
    udp_length = int.from_bytes(frame[udp + 4 : udp + 6], "big")
    if udp_length < _UDP_HLEN + minimum_payload or udp_length != ip_length - ihl:
        return None
    # Respect the datagram boundary: Ethernet padding/FCS are not payload bytes.
    peer = (socket.inet_ntoa(frame[26:30]), int.from_bytes(frame[udp : udp + 2], "big"))
    return frame[udp + _UDP_HLEN : udp + udp_length], peer


def _ip_checksum(header: bytes) -> int:
    total = sum(struct.unpack(f">{len(header) // 2}H", header))
    while total > 0xFFFF:
        total = (total & 0xFFFF) + (total >> 16)
    return ~total & 0xFFFF


def udp_frame(
    payload: bytes,
    *,
    src_port: int,
    dst_port: int = 26400,
    src_ip: str = "10.0.0.1",
    dst_ip: str = "10.0.0.2",
    ihl: int = 5,
) -> bytes:
    """One Ethernet II + IPv4 (``ihl`` 32-bit words, options zero-filled) + UDP frame."""
    if not 5 <= ihl <= 15:
        raise ValueError("IPv4 IHL must be 5..15 words")
    ip_hlen = ihl * 4
    total = ip_hlen + _UDP_HLEN + len(payload)
    if total > 0xFFFF:
        raise ValueError("UDP payload exceeds the IPv4 datagram size")
    ip = bytearray(
        struct.pack(
            ">BBHHHBBH4s4s",
            0x40 | ihl,
            0,
            total,
            0,
            0,
            64,
            _IPPROTO_UDP,
            0,
            socket.inet_aton(src_ip),
            socket.inet_aton(dst_ip),
        )
        + bytes(ip_hlen - 20)
    )
    # A valid header checksum: a kernel UDP receiver drops frames without one.
    struct.pack_into(">H", ip, 10, _ip_checksum(ip))
    # UDP checksum 0: none, which IPv4 allows.
    udp = struct.pack(">HHHH", src_port, dst_port, _UDP_HLEN + len(payload), 0)
    return _ETH_HEADER + bytes(ip) + udp + payload


@dataclass
class PcapFrames:
    """Raw Ethernet frames of a capture in one contiguous host buffer.

    ``data`` is the file content up to the last frame read, so frame ``i`` is
    ``data[offsets[i] : offsets[i] + lengths[i]]`` and uploading ``data`` once makes every
    frame addressable on the device.
    """

    data: np.ndarray  # uint8
    offsets: np.ndarray  # int64 per frame
    lengths: np.ndarray  # int32 per frame
    ts_ns: np.ndarray  # int64 per frame (record timestamps)

    @classmethod
    def read(cls, path: str | Path, limit: int = 0) -> PcapFrames:
        """Read a pcap or nanosecond pcap (endianness from the magic); ``limit`` 0 = all."""
        path = Path(path)
        if path.stat().st_size < _PCAP_HLEN:
            raise ValueError(f"{path}: not a pcap file")
        # Mapped, so a small ``limit`` touches only the start of a multi-GB capture.
        mapped = np.memmap(path, dtype=np.uint8, mode="r")
        view = memoryview(mapped)
        magic = bytes(view[:4])
        if magic not in _PCAP_MAGIC:
            raise ValueError(f"{path}: not a pcap file (magic {magic.hex()})")
        order, unit_ns = _PCAP_MAGIC[magic]
        if struct.unpack_from(order + "I", view, 20)[0] & 0xFFFF != _LINKTYPE_ETHERNET:
            raise ValueError(f"{path}: link type is not Ethernet")
        record = struct.Struct(order + "IIII")
        offsets, lengths, ts_ns = [], [], []
        pos, size = _PCAP_HLEN, len(view)
        while pos + record.size <= size and (not limit or len(offsets) < limit):
            sec, frac, incl, _ = record.unpack_from(view, pos)
            pos += record.size
            if pos + incl > size:
                break  # a capture cut off mid-record ends at its last whole frame
            offsets.append(pos)
            lengths.append(incl)
            ts_ns.append(sec * 1_000_000_000 + frac * unit_ns)
            pos += incl
        end = offsets[-1] + lengths[-1] if offsets else 0
        data = np.array(mapped[:end])  # owned copy: the file may change after reading
        return cls(
            data,
            np.asarray(offsets, np.int64),
            np.asarray(lengths, np.int32),
            np.asarray(ts_ns, np.int64),
        )

    def __len__(self) -> int:
        return len(self.offsets)

    def frame(self, i: int) -> bytes:
        start = int(self.offsets[i])
        return self.data[start : start + int(self.lengths[i])].tobytes()

    def payloads(self, minimum_payload: int = 8) -> list[bytes]:
        """UDP payloads of every IPv4/UDP frame, via the host strip path."""
        out = []
        for i in range(len(self)):
            got = strip_udp_datagram(self.frame(i), minimum_payload)
            if got is not None:
                out.append(got[0])
        return out

    def udp_src_port_offsets(self) -> np.ndarray:
        """IHL-aware byte offset of each frame's UDP source port within the frame (-1 if not
        IPv4/UDP or a non-first fragment)."""
        out = np.full(len(self), -1, np.int64)
        lengths = self.lengths.astype(np.int64)
        candidates = np.flatnonzero(lengths >= _ETH_HLEN + 20 + _UDP_HLEN)
        at = self.offsets[candidates]
        d = self.data
        version_ihl = d[at + _ETH_HLEN]
        ihl = (version_ihl & 0x0F).astype(np.int64) * 4
        fragment_offset = (d[at + 20].astype(np.int64) << 8 | d[at + 21]) & 0x1FFF
        udp = (
            (d[at + 12] == 0x08)
            & (d[at + 13] == 0x00)
            & (version_ihl >> 4 == 4)
            & (ihl >= 20)
            & (d[at + _ETH_HLEN + 9] == _IPPROTO_UDP)
            & (fragment_offset == 0)
            & (lengths[candidates] >= _ETH_HLEN + ihl + _UDP_HLEN)
        )
        out[candidates[udp]] = _ETH_HLEN + ihl[udp]
        return out

    def dominant_src_port(self) -> int | None:
        """The most common UDP source port (the lowest on a tie), or None without UDP frames."""
        rel = self.udp_src_port_offsets()
        udp = rel >= 0
        if not udp.any():
            return None
        at = self.offsets[udp] + rel[udp]
        ports = self.data[at].astype(np.int64) << 8 | self.data[at + 1]
        values, counts = np.unique(ports, return_counts=True)
        return int(values[np.argmax(counts)])


def pcap_header() -> bytes:
    """The file header of a little-endian nanosecond pcap of Ethernet frames."""
    return _PCAP_HEADER.pack(0xA1B23C4D, 2, 4, 0, 0, 65535, _LINKTYPE_ETHERNET)


def pcap_record(ts_ns: int, length: int) -> bytes:
    """The record header of one whole frame of ``length`` bytes captured at ``ts_ns``."""
    sec, ns = divmod(int(ts_ns), 1_000_000_000)
    return _RECORD.pack(sec, ns, length, length)


def pcap_records(ts_ns: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """``pcap_record`` over arrays: a ``(len(ts_ns), 16)`` uint8 array of record headers."""
    records = np.zeros(len(ts_ns), _RECORD_DTYPE)
    records["sec"], records["frac"] = np.divmod(np.asarray(ts_ns, np.int64), 1_000_000_000)
    records["incl"] = records["orig"] = lengths
    return records.view(np.uint8).reshape(len(ts_ns), RECORD_HLEN)


def write_pcap(
    path: str | Path, frames: Iterable[bytes], *, ts_ns: Iterable[int] | None = None
) -> int:
    """Write a nanosecond pcap of ``frames`` (record times ``ts_ns``, else 0); returns the
    frame count."""
    pairs = zip(frames, repeat(0)) if ts_ns is None else zip(frames, ts_ns, strict=True)
    count = 0
    with open(path, "wb", buffering=1 << 20) as f:
        f.write(pcap_header())
        for frame, ts in pairs:
            f.write(pcap_record(ts, len(frame)))
            f.write(frame)
            count += 1
    return count
