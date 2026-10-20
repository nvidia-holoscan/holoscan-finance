# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic synthetic NYQ data: replay units, fuzzed units, tapes and pcap captures.

``synthetic_stream`` and ``fuzz`` are verbatim from 4.3 ``equities.gpu_validate``: random
but valid units covering every observation kind (one-sided, negative and halted quotes,
trades, conditions, several Time seconds per packet) and mutations aimed at each rejection
rule. ``synthetic_tape`` writes a tape in the ``open_tape`` format and ``synthetic_capture``
packetizes a tape as 4.3 ``replay.py`` sends it, into a pcap; with them every test and the
NIC-free validation run without the licensed data. NumPy only.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from ..marketdata.records import CONDITION, EVENT_DTYPE, HALTED, QUOTE, TRADE
from .mapping import MAX_PACKET_EVENTS as MAX_EVENTS
from .mapping import pack_events


def synthetic_stream(
    rng: np.random.Generator, packets: int, universe: int, midnight_ns: int, start_s: int = 34200
) -> list[bytes]:
    """Random but valid replay units: quotes (incl. one-sided, halted, negative),
    trades, halt/resume conditions, several Time seconds per packet."""
    out, seq, t = [], 1, midnight_ns + start_s * 10**9
    for _ in range(packets):
        n = int(rng.integers(1, MAX_EVENTS + 1))
        ev = np.zeros(n, EVENT_DTYPE)
        t += int(rng.integers(0, 3 * 10**8))
        ev["timestamp_ns"] = t + np.sort(rng.integers(0, 2 * 10**9, n))
        t = int(ev["timestamp_ns"][-1])
        ev["symbol_id"] = rng.integers(0, universe, n)
        ev["kind"] = rng.choice([QUOTE, QUOTE, QUOTE, TRADE, TRADE, CONDITION], n)
        for f in ("bid", "ask", "bid_size", "ask_size", "price", "volume"):
            ev[f] = np.nan
        q = ev["kind"] == QUOTE
        mid = rng.uniform(1, 500, n)
        ev["bid"][q] = np.round(mid[q] * rng.uniform(0.97, 1.0, q.sum()), 4)
        ev["ask"][q] = np.round(mid[q] * rng.uniform(1.0, 1.03, q.sum()), 4)
        ev["bid"][q & (rng.random(n) < 0.05)] = np.nan
        ev["bid"][q & (rng.random(n) < 0.03)] *= -1
        ev["bid_size"][q] = rng.integers(0, 2000, q.sum())
        ev["ask_size"][q] = rng.integers(0, 2000, q.sum())
        tr = ev["kind"] == TRADE
        ev["price"][tr] = np.round(rng.uniform(1, 900_000, tr.sum()), 6)
        ev["volume"][tr] = rng.integers(0, 10_000, tr.sum())
        ev["flags"][(ev["kind"] == CONDITION) & (rng.random(n) < 0.5)] = HALTED
        ev["flags"][q & (rng.random(n) < 0.02)] = HALTED
        packed, count = pack_events(ev, seq, midnight_ns)
        out.append(packed)
        seq += count
    return out


def fuzz(payloads: list[bytes], count: int, seed: int = 7) -> list[bytes]:
    """Mutations aimed at each rejection rule, applied to real packets."""
    rng = np.random.default_rng(seed)
    out = []
    picks = rng.integers(0, len(payloads), count)
    for j, i in enumerate(picks):
        b = bytearray(payloads[int(i)])
        mode = j % 12
        if mode == 0:  # random byte flip anywhere
            b[rng.integers(0, len(b))] ^= 1 << int(rng.integers(0, 8))
        elif mode == 1:  # truncate
            b = b[: int(rng.integers(0, len(b)))]
        elif mode == 2:  # wrong unit length
            struct.pack_into("<H", b, 0, (len(b) + int(rng.integers(1, 5))) & 0xFFFF)
        elif mode == 3:  # wrong group
            b[3] = int(rng.choice([ord("B"), 0, 255]))
        elif mode == 4:  # sequence 0 / overflow
            struct.pack_into("<I", b, 4, int(rng.choice([0, 0xFFFFFFFF])))
        elif mode == 5:  # message kind / length byte corruption inside the unit
            off = 8
            k = int(rng.integers(0, max(1, b[2])))
            for _ in range(k):
                if off + 1 >= len(b):
                    break
                off += b[off]
            if off + 1 < len(b):
                b[off + int(rng.integers(0, 2))] = int(rng.integers(0, 256))
        elif mode == 6:  # Time out of range
            if len(b) > 13 and b[9] == ord("T"):
                struct.pack_into("<I", b, 10, 86400 + int(rng.integers(0, 5)))
        elif mode == 7:  # nanoseconds >= 1e9 on the first observation
            if len(b) > 21 and b[9] == ord("T") and b[14] in (34, 35, 28):
                struct.pack_into("<I", b, 16, 10**9 + int(rng.integers(0, 100)))
        elif mode == 8:  # symbol 0 / beyond universe on the first observation
            if len(b) > 25 and b[9] == ord("T") and b[15] == ord("q"):
                struct.pack_into("<I", b, 20, int(rng.choice([0, 0xFFFFFFFF, 99999])))
        elif mode == 9:  # unmutated control packet
            pass
        elif mode == 10:  # sign bit on a price (negative prices decode, not reject)
            if len(b) > 30 and b[9] == ord("T") and b[15] == ord("q"):
                b[31] |= 0x80
        elif mode == 11:  # message count mismatch
            b[2] = (b[2] + int(rng.integers(1, 4))) & 0xFF
        out.append(bytes(b))
    return out


def synthetic_tape(
    out_dir: str | Path,
    *,
    date: str = "2025-04-01",
    symbols: int = 500,
    packets: int = 20000,
    seconds: int = 600,
    max_events_per_packet: int = 32,
    seed: int = 7,
) -> dict:
    """Write ``events.bin`` (time ordered within 09:30 New York + [0, ``seconds``)) and
    ``manifest.json`` in the ``open_tape`` format; returns the manifest."""
    raise NotImplementedError


def synthetic_capture(
    tape_dir: str | Path,
    pcap_path: str | Path,
    *,
    sender_port: int = 40000,
    dst_port: int = 26400,
    max_events_per_packet: int = 32,
    start_seq: int = 1,
    drop_every: int = 0,
) -> dict:
    """Packetize a tape as 4.3 ``replay.py`` does (``pack_events`` units plus the final Status
    marker) into UDP frames in a pcap; ``drop_every`` > 0 leaves out every n-th packet
    (sequence gaps). Returns the counts."""
    raise NotImplementedError
