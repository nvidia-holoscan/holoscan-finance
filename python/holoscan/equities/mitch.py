# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NYQ observations mapped to MIT303 11.9 T/q/P/H messages for replay.

The q/P layouts are reused for consolidated quotes/trades; their original LSE
auction/hidden-trade semantics are not asserted. See docs/METHODOLOGY.md.
Dates, RIC directory and sender-session identity come from the HTTP manifest.
There are no custom trailers or timestamps hidden in exchange identifier fields.
"""

from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np

from holoscan.marketdata import mitch
from holoscan.marketdata.wire import CONDITION, EVENT_DTYPE, HALTED, QUOTE, TRADE, ReplayBatch

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
                raise ValueError("unsupported MITCH trade semantics for NYQ replay")
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
                raise ValueError("unsupported MITCH status semantics for NYQ replay")
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
