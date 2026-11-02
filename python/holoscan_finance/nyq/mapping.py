# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""equity observations mapped to MIT303 11.9 T/q/P/H messages for replay.

The q/P layouts are reused for consolidated quotes/trades; their original LSE
auction/hidden-trade semantics are not asserted. See docs/METHODOLOGY.md.
Dates, instrument directory and sender-session identity come from the HTTP manifest.
There are no custom trailers or timestamps hidden in exchange identifier fields.
``decode_packet`` is the host oracle the GPU decoder matches bit for bit.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np

from ..marketdata import mitch_codec as mitch
from ..marketdata.records import CONDITION, EVENT_DTYPE, HALTED, QUOTE, TRADE, ReplayBatch

PROTOCOL = "mit303-11.9-nyq-replay-v1"
TIMEZONE = "America/New_York"
MAX_PACKET_EVENTS = 32  # worst case: 32 * (35-byte Trade + 6-byte Time) + 8


def day_start_ns(day):
    return int(datetime.fromisoformat(day).replace(tzinfo=ZoneInfo(TIMEZONE)).timestamp()) * 10**9


def _price(value):
    # MITCH has no IEEE NaN: zero denotes unavailable/empty in this replay.
    return mitch.price_to_wire(float(value)) if np.isfinite(value) else 0


def _quantity(value):
    if not np.isfinite(value) or value <= 0:
        return 0
    if value != int(value) or value > 0xFFFFFFFF:
        raise ValueError("source quantity cannot be represented by MITCH UInt32")
    return int(value)


def pack_events(events, sequence, midnight_ns):
    if not 1 <= len(events) <= MAX_PACKET_EVENTS:
        raise ValueError(f"packets must contain 1..{MAX_PACKET_EVENTS} observations")
    messages, second = [], None
    previous = None
    for row in events:
        ts = int(row["timestamp_ns"])
        seconds, ns = divmod(ts - midnight_ns, 10**9)
        if not 0 <= seconds < 86400 or (previous is not None and ts < previous):
            raise ValueError("MITCH observations must be ordered within the configured market day")
        previous = ts
        # Repeat Time at each unit boundary so a lost packet cannot corrupt later times.
        if seconds != second:
            messages.append(mitch.TIME.pack(6, ord("T"), seconds))
            second = seconds
        symbol = int(row["symbol_id"]) + 1  # replay IDs; zero is not an instrument
        kind = int(row["kind"])
        if kind == QUOTE:
            halted = bool(row["flags"] & HALTED)
            messages.append(
                mitch.TOP.pack(
                    34,
                    ord("q"),
                    ns,
                    symbol,
                    0 if halted else _price(row["bid"]),
                    _quantity(row["bid_size"]),
                    0 if halted else _price(row["ask"]),
                    _quantity(row["ask_size"]),
                )
            )
        elif kind == TRADE:
            # Unique replay trade ID; never a disguised wall-clock timestamp.
            messages.append(
                mitch.TRADE.pack(
                    35,
                    ord("P"),
                    ns,
                    _quantity(row["volume"]),
                    symbol,
                    _price(row["price"]),
                    sequence + len(messages),
                    0,
                    0,
                    0,
                )
            )
        elif kind == CONDITION:
            messages.append(
                mitch.STATUS.pack(
                    28,
                    ord("H"),
                    ns,
                    symbol,
                    b"H" if row["flags"] & HALTED else b"T",
                    0,
                    b"    ",
                    0,
                    b"        ",
                    1,
                )
            )
        else:
            raise ValueError("unsupported source observation kind")
    return mitch.pack_unit(messages, sequence), len(messages)


_UNIT_HDR = np.dtype([("length", "<u2"), ("count", "u1"), ("group", "u1"), ("sequence", "<u4")])
_TIME = np.dtype([("length", "u1"), ("kind", "u1"), ("seconds", "<u4")])
_TOP = np.dtype(
    [
        ("length", "u1"),
        ("kind", "u1"),
        ("ns", "<u4"),
        ("symbol", "<u4"),
        ("bid", "<u8"),
        ("bid_size", "<u4"),
        ("ask", "<u8"),
        ("ask_size", "<u4"),
    ]
)
_TRADE = np.dtype(
    [
        ("length", "u1"),
        ("kind", "u1"),
        ("ns", "<u4"),
        ("volume", "<u4"),
        ("symbol", "<u4"),
        ("pad", "u1", (2,)),
        ("price", "<u8"),
        ("trade_id", "<u8"),
        ("cross", "u1"),
        ("subbook", "u1"),
        ("flags", "u1"),
    ]
)
_STATUS = np.dtype(
    [
        ("length", "u1"),
        ("kind", "u1"),
        ("ns", "<u4"),
        ("symbol", "<u4"),
        ("pad", "u1", (2,)),
        ("status", "u1"),
        ("flags", "u1"),
        ("reason", "u1", (4,)),
        ("change", "u1"),
        ("end", "u1", (8,)),
        ("book", "u1"),
    ]
)
assert (_UNIT_HDR.itemsize, _TIME.itemsize) == (mitch.HEADER.size, mitch.TIME.size)
assert (_TOP.itemsize, _TRADE.itemsize, _STATUS.itemsize) == (
    mitch.TOP.size,
    mitch.TRADE.size,
    mitch.STATUS.size,
)
_MESSAGE_LENGTHS = np.zeros(256, np.int64)
_MESSAGE_LENGTHS[[QUOTE, TRADE, CONDITION]] = (mitch.TOP.size, mitch.TRADE.size, mitch.STATUS.size)
_ROW_BYTES = mitch.TIME.size + mitch.TRADE.size  # a row's Time message and its widest message


def _prices_to_wire(values):
    """``_price`` over an array: sign-magnitude at eight decimals, 0 where not finite."""
    v = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(v)
    magnitude = np.rint(np.abs(v) * mitch.SCALE)  # round(): ties to even, as np.rint
    if (magnitude[finite] >= float(mitch.SIGN)).any():
        raise ValueError("MITCH price exceeds sign-magnitude range")
    wire = np.where(finite, magnitude, 0.0).astype(np.uint64)
    wire[finite & (v < 0)] |= np.uint64(mitch.SIGN)
    return wire


def _quantities(values):
    """``_quantity`` over an array: whole positive counts, 0 where missing or non-positive."""
    v = np.asarray(values, dtype=np.float64)
    positive = np.isfinite(v) & (v > 0)
    given = v[positive]
    if (given != np.floor(given)).any() or (given > 0xFFFFFFFF).any():
        raise ValueError("source quantity cannot be represented by MITCH UInt32")
    return np.where(positive, v, 0.0).astype(np.uint32)


def pack_units(events, sequence, midnight_ns, unit_events=MAX_PACKET_EVENTS):
    """``pack_events`` over consecutive units of ``unit_events`` observations, vectorized.

    Returns ``(data, offsets, lengths, counts)``: the units' bytes concatenated (uint8), and
    per unit its offset into ``data``, its length and its MITCH message count. The first
    unit carries ``sequence`` and each unit consumes its messages' sequence numbers, so unit
    ``k`` is byte for byte ``pack_events(events[k * unit_events:(k + 1) * unit_events],
    sequence + sum(counts[:k]), midnight_ns)[0]``; the same ValueErrors reject the same
    input. ``synthetic_capture`` and the replay sender pack in blocks with it.
    """
    if not 1 <= unit_events <= MAX_PACKET_EVENTS:
        raise ValueError(f"packets must contain 1..{MAX_PACKET_EVENTS} observations")
    ev = np.asarray(events)
    n = len(ev)
    if n < 1:
        raise ValueError(f"packets must contain 1..{MAX_PACKET_EVENTS} observations")
    ts = ev["timestamp_ns"].astype(np.int64)
    seconds, ns = np.divmod(ts - np.int64(midnight_ns), np.int64(10**9))
    unit_starts = np.arange(0, n, unit_events)
    first = np.zeros(n, bool)
    first[unit_starts] = True
    if (seconds < 0).any() or (seconds >= 86400).any() or (~first[1:] & (ts[1:] < ts[:-1])).any():
        raise ValueError("MITCH observations must be ordered within the configured market day")
    kind = ev["kind"]
    message_bytes = _MESSAGE_LENGTHS[kind]
    if not message_bytes.all():
        raise ValueError("unsupported source observation kind")
    symbol = ev["symbol_id"].astype(np.uint64) + np.uint64(
        1
    )  # replay IDs; zero is not an instrument
    if (symbol > 0xFFFFFFFF).any():
        raise ValueError("source instrument exceeds the MITCH UInt32 range")
    # One Time message before the first observation of a unit and at every second change.
    timed = first.copy()
    timed[1:] |= seconds[1:] != seconds[:-1]
    row_messages = 1 + timed.astype(np.int64)
    row_bytes = message_bytes + mitch.TIME.size * timed
    unit_counts = np.add.reduceat(row_messages, unit_starts)
    unit_lengths = mitch.HEADER.size + np.add.reduceat(row_bytes, unit_starts)
    unit_sequences = sequence + np.r_[0, np.cumsum(unit_counts[:-1])]
    if sequence < 0 or unit_sequences[-1] > 0xFFFFFFFF:
        raise ValueError("invalid MITCH sequence/group")
    n_units = len(unit_starts)
    unit_rows = np.diff(np.r_[unit_starts, n])
    unit_of_row = np.repeat(np.arange(n_units), unit_rows)
    # Each row's message index within its unit (its Time message, if any, comes first).
    exclusive = np.cumsum(row_messages) - row_messages
    message_index = exclusive - np.repeat(exclusive[unit_starts], unit_rows) + timed

    # Every row in a fixed-width buffer: Time at [0:6], the message after it; the mask keeps
    # the bytes each row really emits, in order.
    rows = np.zeros((n, _ROW_BYTES), np.uint8)
    time_rec = np.zeros(n, _TIME)
    time_rec["length"], time_rec["kind"], time_rec["seconds"] = mitch.TIME.size, ord("T"), seconds
    rows[:, : mitch.TIME.size] = time_rec.view(np.uint8).reshape(n, mitch.TIME.size)
    body = mitch.TIME.size

    q = np.flatnonzero(kind == QUOTE)
    if len(q):
        rec = np.zeros(len(q), _TOP)
        rec["length"], rec["kind"] = mitch.TOP.size, ord("q")
        rec["ns"], rec["symbol"] = ns[q], symbol[q]
        halted = (ev["flags"][q] & HALTED) != 0
        bid, ask = _prices_to_wire(ev["bid"][q]), _prices_to_wire(ev["ask"][q])
        bid[halted] = ask[halted] = 0
        rec["bid"], rec["ask"] = bid, ask
        rec["bid_size"], rec["ask_size"] = (
            _quantities(ev["bid_size"][q]),
            _quantities(ev["ask_size"][q]),
        )
        rows[q, body : body + mitch.TOP.size] = rec.view(np.uint8).reshape(-1, mitch.TOP.size)
    t = np.flatnonzero(kind == TRADE)
    if len(t):
        rec = np.zeros(len(t), _TRADE)
        rec["length"], rec["kind"] = mitch.TRADE.size, ord("P")
        rec["ns"], rec["symbol"] = ns[t], symbol[t]
        rec["volume"], rec["price"] = _quantities(ev["volume"][t]), _prices_to_wire(ev["price"][t])
        # Unique replay trade ID: the message's sequence number.
        rec["trade_id"] = (unit_sequences[unit_of_row[t]] + message_index[t]).astype(np.uint64)
        rows[t, body : body + mitch.TRADE.size] = rec.view(np.uint8).reshape(-1, mitch.TRADE.size)
    c = np.flatnonzero(kind == CONDITION)
    if len(c):
        rec = np.zeros(len(c), _STATUS)
        rec["length"], rec["kind"] = mitch.STATUS.size, ord("H")
        rec["ns"], rec["symbol"] = ns[c], symbol[c]
        rec["status"] = np.where((ev["flags"][c] & HALTED) != 0, ord("H"), ord("T"))
        rec["reason"], rec["end"], rec["book"] = ord(" "), ord(" "), 1
        rows[c, body : body + mitch.STATUS.size] = rec.view(np.uint8).reshape(-1, mitch.STATUS.size)

    column = np.arange(_ROW_BYTES)
    keep = ((column < body) & timed[:, None]) | (
        (column >= body) & (column < body + message_bytes[:, None])
    )
    payload = rows[keep]
    headers = np.zeros(n_units, _UNIT_HDR)
    headers["length"], headers["count"] = unit_lengths, unit_counts
    headers["group"], headers["sequence"] = ord("A"), unit_sequences
    offsets = np.r_[0, np.cumsum(unit_lengths[:-1])]
    data = np.empty(int(offsets[-1] + unit_lengths[-1]), np.uint8)
    is_header = np.zeros(len(data), bool)
    is_header[(offsets[:, None] + np.arange(mitch.HEADER.size)).ravel()] = True
    data[is_header] = headers.view(np.uint8)
    data[~is_header] = payload
    return data, offsets, unit_lengths, unit_counts


def decode_packet(data, recv_ns, *, generation, midnight_ns, universe):
    group, sequence, messages = mitch.parse_unit(data)
    if group != ord("A"):
        raise ValueError("unexpected MITCH replay market-data group")
    rows = np.zeros(len(messages), dtype=EVENT_DTYPE)
    for field in ("bid", "ask", "bid_size", "ask_size", "price", "volume"):
        rows[field] = np.nan
    count, seconds, previous = 0, None, None
    for message in messages:
        kind = message[1]
        if kind == ord("T"):
            seconds = mitch.TIME.unpack(message)[2]
            if seconds >= 86400:
                raise ValueError("MITCH Time is outside the configured day")
            continue
        if seconds is None:
            raise ValueError("MITCH replay unit must supply Time before observations")
        if kind == ord("q"):
            _, _, ns, symbol, bid, bid_size, ask, ask_size = mitch.TOP.unpack(message)
            rows[count]["kind"] = QUOTE
            rows[count]["bid"], rows[count]["ask"] = (
                mitch.price_from_wire(bid),
                mitch.price_from_wire(ask),
            )
            rows[count]["bid_size"], rows[count]["ask_size"] = bid_size, ask_size
        elif kind == ord("P"):
            _, _, ns, volume, symbol, price, _, cross, subbook, flags = mitch.TRADE.unpack(message)
            if bytes(message[14:16]) != b"\0\0" or cross or subbook or flags:
                raise ValueError("unsupported MITCH trade semantics for equity replay")
            rows[count]["kind"] = TRADE
            rows[count]["price"], rows[count]["volume"] = mitch.price_from_wire(price), volume
        else:
            _, _, ns, symbol, status, flags, reason, change, end, book = mitch.STATUS.unpack(
                message
            )
            if (
                bytes(message[10:12]) != b"\0\0"
                or status not in (b"H", b"T")
                or flags
                or reason != b"    "
                or change
                or end != b"        "
                or book != 1
            ):
                raise ValueError("unsupported MITCH status semantics for equity replay")
            rows[count]["kind"] = CONDITION
            rows[count]["flags"] = HALTED if status == b"H" else 0
        timestamp = midnight_ns + seconds * 10**9 + ns
        if (
            ns >= 10**9
            or not 1 <= symbol <= universe
            or (previous is not None and timestamp < previous)
        ):
            raise ValueError("invalid MITCH timestamp or instrument")
        previous = timestamp
        rows[count]["timestamp_ns"], rows[count]["symbol_id"] = timestamp, symbol - 1
        count += 1
    batch = ReplayBatch(rows[:count], generation, recv_ns, sequence, wire_count=len(messages))
    return [batch], len(messages)
