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

import json
import logging
import struct
from pathlib import Path

import numpy as np

from ..marketdata.framing import udp_frame, write_pcap
from ..marketdata.records import CONDITION, EVENT_DTYPE, HALTED, QUOTE, TRADE
from .manifest import MANIFEST_FORMAT
from .mapping import MAX_PACKET_EVENTS as MAX_EVENTS
from .mapping import day_start_ns, pack_events
from .tape import open_tape

logger = logging.getLogger(__name__)

_SESSION_START_S = 34200  # 09:30 New York
_CHUNK = 1 << 20  # observations generated per step: bounded memory for full-day tapes
_PROGRESS_PACKETS = 1 << 18
# Tape kinds (quotes, trades, conditions) and anomaly rates: synthetic_stream's kinds, rare
# enough that about 4 in 5 symbols keep every mark of a 60-bar window (risk eligibility).
_KIND_P = (0.75, 0.248, 0.002)
_ONE_SIDED, _NEGATIVE, _HALTED_QUOTE = 0.001, 0.0005, 0.0005  # per quote
_HALT = 0.5  # of conditions; the rest resume trading


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


def _check_unit_size(max_events_per_packet: int) -> None:
    if not 1 <= max_events_per_packet <= MAX_EVENTS:
        raise ValueError(f"max_events_per_packet must be 1..{MAX_EVENTS}")


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
    ``manifest.json`` in the ``open_tape`` format; returns the manifest.

    The tape holds ``packets * max_events_per_packet`` observations, so ``synthetic_capture``
    with the same ``max_events_per_packet`` sends ``packets`` full units. Kinds and anomalies
    are those of ``synthetic_stream``, but quote mids follow per-symbol random walks around
    a common market factor, with spreads inside the bar rules, and anomalies are rare: most
    symbols keep a complete return window, so risk windows have eligible instruments.
    """
    if not 1 <= symbols <= 10000:
        raise ValueError("symbols must be 1..10000 (the manifest limit)")
    if packets < 1:
        raise ValueError("packets must be positive")
    if not 1 <= seconds < 86400 - _SESSION_START_S:
        raise ValueError("seconds must keep the session and its end marker inside the day")
    _check_unit_size(max_events_per_packet)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    start_ns = day_start_ns(date) + _SESSION_START_S * 10**9
    span = seconds * 10**9
    events = packets * max_events_per_packet
    model = _SymbolModel(rng, symbols, seconds)
    profile = np.zeros(-(-seconds // 60), np.int64)
    kinds = np.zeros(4, np.int64)
    partial = out / "events.partial"
    with partial.open("wb") as target:
        for lo in range(0, events, _CHUNK):
            hi = min(lo + _CHUNK, events)
            # Equal event counts per equal time slice keep the tape time ordered chunk by chunk.
            offset = np.sort(rng.integers(span * lo // events, span * hi // events, hi - lo))
            rows = model.observations(rng, offset)
            rows["timestamp_ns"] = start_ns + offset
            rows.tofile(target)
            profile += np.bincount(offset // (60 * 10**9), minlength=len(profile))
            kinds += np.bincount(rows["kind"], minlength=4)
    partial.replace(out / "events.bin")
    manifest = {
        "format": MANIFEST_FORMAT,
        "date": date,
        "symbols": [f"SYN{i:05d}.N" for i in range(symbols)],
        "events": events,
        "kinds": {
            "quotes": int(kinds[QUOTE]),
            "trades": int(kinds[TRADE]),
            "conditions": int(kinds[CONDITION]),
        },
        "start_ns": start_ns,
        "end_ns": start_ns + span,
        "record_bytes": EVENT_DTYPE.itemsize,
        # As tape.prepare: [UTC second of day, observations] per minute.
        "profile": [
            [int(start_ns // 10**9 % 86400 + i * 60), int(c)] for i, c in enumerate(profile)
        ],
        "synthetic": {"seed": seed},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    logger.info("synthetic tape: %d observations, %d symbols at %s", events, symbols, out)
    return manifest


class _SymbolModel:
    """Per-symbol price level, spread, market beta and volatility; the idiosyncratic walk
    carries over between chunks."""

    def __init__(self, rng: np.random.Generator, symbols: int, seconds: int) -> None:
        self.symbols = symbols
        self.log_base = np.log(rng.uniform(5, 500, symbols))
        self.half_spread = rng.uniform(0.5e-4, 10e-4, symbols)  # 1-20 bps wide
        self.beta = rng.uniform(0.3, 1.2, symbols)
        self.step = rng.uniform(0.5e-4, 2e-4, symbols)  # idiosyncratic log move per observation
        self.market = np.cumsum(rng.normal(0, 2e-4, seconds + 1))  # log level per second
        self.walk = np.zeros(symbols)

    def _mids(self, rng: np.random.Generator, sym: np.ndarray, offset: np.ndarray) -> np.ndarray:
        # Each symbol's walk advances once per observation of it, in time order: a cumulative
        # sum over the observations grouped by symbol (stable sort keeps time order).
        step = rng.normal(0, 1, len(sym)) * self.step[sym]
        order = np.argsort(sym, kind="stable")
        grouped, moves = sym[order], step[order]
        level = np.cumsum(moves)
        first = np.flatnonzero(np.r_[True, grouped[1:] != grouped[:-1]])
        sizes = np.diff(np.r_[first, len(sym)])
        level += np.repeat(self.walk[grouped[first]] - level[first] + moves[first], sizes)
        self.walk[grouped[first]] = level[first + sizes - 1]
        idio = np.empty(len(sym))
        idio[order] = level
        market = self.market[offset // 10**9] * self.beta[sym]
        return np.exp(self.log_base[sym] + market + idio)

    def observations(self, rng: np.random.Generator, offset: np.ndarray) -> np.ndarray:
        n = len(offset)
        rows = np.zeros(n, EVENT_DTYPE)
        sym = rng.integers(0, self.symbols, n)
        kind = rng.choice(np.array([QUOTE, TRADE, CONDITION], np.uint8), n, p=_KIND_P)
        mid = self._mids(rng, sym, offset)
        half = self.half_spread[sym]
        for f in ("bid", "ask", "bid_size", "ask_size", "price", "volume"):
            rows[f] = np.nan
        q, tr, cond = kind == QUOTE, kind == TRADE, kind == CONDITION
        rows["symbol_id"], rows["kind"] = sym, kind
        rows["bid"][q] = np.round(mid[q] * (1 - half[q]), 4)
        rows["ask"][q] = np.round(mid[q] * (1 + half[q]), 4)
        rows["bid_size"][q] = rng.integers(1, 2000, q.sum())
        rows["ask_size"][q] = rng.integers(1, 2000, q.sum())
        u = rng.random(n)
        rows["bid"][q & (u < _ONE_SIDED)] = np.nan
        negative = q & (u >= _ONE_SIDED) & (u < _ONE_SIDED + _NEGATIVE)
        rows["bid"][negative] *= -1
        rows["flags"][q & (u >= 1 - _HALTED_QUOTE)] = HALTED
        rows["price"][tr] = np.round(mid[tr] * (1 + rng.uniform(-1, 1, tr.sum()) * half[tr]), 4)
        rows["volume"][tr] = rng.integers(1, 5000, tr.sum())
        rows["flags"][cond & (u < _HALT)] = HALTED
        return rows


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
    marker) into UDP frames in a pcap; ``drop_every`` > 0 leaves out every n-th data packet
    (sequence gaps; the marker is always sent). Returns the counts.

    The counts are what a lossless receiver of the pcap sees: ``packets``, ``events``
    (observations, the marker's included) and ``messages`` (MITCH messages, Time included),
    with ``dropped_messages`` sequence gaps; ``next_sequence`` follows the marker. Record
    times are the event time of each unit's last observation.
    """
    _check_unit_size(max_events_per_packet)
    if not (1 <= sender_port <= 65535 and 1 <= dst_port <= 65535):
        raise ValueError("UDP ports must be 1..65535")
    if start_seq < 1:
        raise ValueError("start_seq must be >= 1 (MITCH sequence 0 is unsequenced)")
    if drop_every < 0:
        raise ValueError("drop_every must be >= 0")
    manifest, rows = open_tape(tape_dir)
    midnight_ns = day_start_ns(manifest["date"])
    starts = range(0, len(rows), max_events_per_packet)
    sent = np.ones(len(starts), bool)
    if drop_every:
        sent[drop_every - 1 :: drop_every] = False
    last = np.minimum(np.asarray(starts) + max_events_per_packet, len(rows)) - 1
    stamps = np.r_[rows["timestamp_ns"][last[sent]].astype(np.int64), manifest["end_ns"]]
    # Filled in while write_pcap consumes frames(): complete once the pcap is written.
    counts = dict.fromkeys(
        ("packets", "events", "messages", "dropped_packets", "dropped_events", "dropped_messages"),
        0,
    )

    def frames():
        sequence = start_seq
        for k, lo in enumerate(starts):
            chunk = rows[lo : lo + max_events_per_packet]
            unit, count = pack_events(chunk, sequence, midnight_ns)
            sequence += count  # a dropped unit still consumes its sequence numbers
            prefix = "" if sent[k] else "dropped_"
            counts[prefix + "packets"] += 1
            counts[prefix + "events"] += len(chunk)
            counts[prefix + "messages"] += count
            if sent[k]:
                yield udp_frame(unit, src_port=sender_port, dst_port=dst_port)
            if k and k % _PROGRESS_PACKETS == 0:
                logger.info("synthetic capture: %d of %d units", k, len(starts))
        # Event-time watermark closes the final regular-session bar.
        marker = np.zeros(1, dtype=EVENT_DTYPE)
        marker["timestamp_ns"] = manifest["end_ns"]
        marker["kind"] = CONDITION
        unit, count = pack_events(marker, sequence, midnight_ns)
        counts["packets"] += 1
        counts["events"] += 1
        counts["messages"] += count
        counts.update(first_sequence=start_seq, next_sequence=sequence + count)
        yield udp_frame(unit, src_port=sender_port, dst_port=dst_port)

    write_pcap(pcap_path, frames(), ts_ns=stamps)
    counts.update(sender_port=sender_port, dst_port=dst_port)
    logger.info("synthetic capture: %d packets to %s", counts["packets"], pcap_path)
    return counts
