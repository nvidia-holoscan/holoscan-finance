# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MIT303 issue 11.9 unit framing and T/q/P/H message layouts.

Specification: https://docs.londonstockexchange.com/sites/default/files/documents/mit303issue119.pdf
Prices use SIGN-MAGNITUDE, including the sign bit, per Appendix B. They are
not two's-complement integers despite the specification's "signed" shorthand.
This codec implements a subset, not an exchange connection/recovery service.
Verbatim from 4.3 ``holoscan.marketdata.mitch``; the ``gpu_mitch`` kernels reproduce its
framing and parsing rules.
"""

from __future__ import annotations

import math
import struct

HEADER = struct.Struct("<HBBI")
TIME = struct.Struct("<BBI")
TOP = struct.Struct("<BBIIQIQI")
TRADE = struct.Struct("<BBIII2xQQBBB")
STATUS = struct.Struct("<BBII2xcB4sB8sB")
SCALE = 100_000_000
SIGN = 1 << 63
LENGTHS = {ord("T"): TIME.size, ord("q"): TOP.size, ord("P"): TRADE.size, ord("H"): STATUS.size}


def price_to_wire(value):
    """Encode a finite price at eight decimals; reject range overflow."""
    if not math.isfinite(value):
        raise ValueError("MITCH price must be finite")
    magnitude = round(abs(value) * SCALE)
    if magnitude >= SIGN:
        raise ValueError("MITCH price exceeds sign-magnitude range")
    return magnitude | (SIGN if value < 0 else 0)


def price_from_wire(value):
    return (-1 if value & SIGN else 1) * (value & (SIGN - 1)) / SCALE


def pack_unit(messages, sequence, group=ord("A")):
    if not 0 <= sequence <= 0xFFFFFFFF or not 0 <= group <= 255:
        raise ValueError("invalid MITCH sequence/group")
    if len(messages) > 255:
        raise ValueError("too many MITCH messages")
    payload = b"".join(messages)
    if len(payload) + HEADER.size > 1472:
        raise ValueError("MITCH replay unit exceeds the unfragmented IPv4 UDP budget")
    return HEADER.pack(HEADER.size + len(payload), len(messages), group, sequence) + payload


def inspect_unit(data):
    if len(data) < HEADER.size:
        raise ValueError("short MITCH unit")
    length, count, group, sequence = HEADER.unpack_from(data)
    if length != len(data) or length < HEADER.size + count * 2:
        raise ValueError("invalid MITCH unit length/count")
    if not count and length != HEADER.size:
        raise ValueError("MITCH heartbeat has a payload")
    if count and (sequence == 0 or sequence + count > 0x100000000):
        raise ValueError("invalid sequenced MITCH replay unit")
    return group, sequence, count


def parse_unit(data):
    group, sequence, count = inspect_unit(data)
    messages, offset = [], HEADER.size
    for _ in range(count):
        if offset + 2 > len(data):
            raise ValueError("truncated MITCH message header")
        length, kind = data[offset : offset + 2]
        if kind not in LENGTHS or length != LENGTHS[kind] or offset + length > len(data):
            raise ValueError("unsupported, malformed or truncated MITCH message")
        messages.append(memoryview(data)[offset : offset + length])
        offset += length
    if offset != len(data):
        raise ValueError("trailing MITCH bytes")
    return group, sequence, messages
