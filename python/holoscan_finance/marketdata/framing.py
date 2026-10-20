# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ethernet II / IPv4 / UDP framing and pcap files, on the host.

``strip_udp_datagram`` is the host strip path (verbatim from 4.3 ``marketdata.daqiri``); the
``gather_frame`` kernel applies the same checks on the GPU. ``udp_frame`` and
``write_pcap`` build synthetic captures, and ``PcapFrames`` holds a capture as one
contiguous buffer, as the NIC-free replay source uploads it to the GPU (was
``gpu_validate.read_pcap``). Source-port offsets are IHL-aware, unlike 4.3
``bench_replay``.
"""

from __future__ import annotations

import socket
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Ethernet header without VLAN; DAQIRI raw RX delivers full L2 frames.
_ETH_HLEN = 14
_ETHERTYPE_IPV4 = b"\x08\x00"
_IPPROTO_UDP = 17
_UDP_HLEN = 8


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
    raise NotImplementedError


@dataclass
class PcapFrames:
    """Raw Ethernet frames of a capture in one contiguous host buffer."""

    data: np.ndarray  # uint8
    offsets: np.ndarray  # int64 per frame
    lengths: np.ndarray  # int32 per frame
    ts_ns: np.ndarray  # int64 per frame (record timestamps)

    @classmethod
    def read(cls, path: str | Path, limit: int = 0) -> PcapFrames:
        """Read a pcap or nanosecond pcap (endianness from the magic); ``limit`` 0 = all."""
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def frame(self, i: int) -> bytes:
        raise NotImplementedError

    def payloads(self, minimum_payload: int = 8) -> list[bytes]:
        """UDP payloads of every IPv4/UDP frame, via the host strip path."""
        raise NotImplementedError

    def udp_src_port_offsets(self) -> np.ndarray:
        """IHL-aware byte offset of each frame's UDP source port (-1 if not IPv4/UDP)."""
        raise NotImplementedError

    def dominant_src_port(self) -> int | None:
        """The most common UDP source port, or None without UDP frames."""
        raise NotImplementedError


def write_pcap(
    path: str | Path, frames: Iterable[bytes], *, ts_ns: Iterable[int] | None = None
) -> int:
    """Write a nanosecond pcap of ``frames``; returns the frame count."""
    raise NotImplementedError
