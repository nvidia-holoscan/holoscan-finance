# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Equity observations mapped to SBE ``MarketDataIncrementalRefresh`` entries.

One observation is one 48-octet entry of the ``NoMDEntries`` group: a quote carries
``BidPx``/``OfferPx``/``BidSize``/``OfferSize``, a trade ``LastPx``/``LastQty``, a halt or
resume only ``SecurityTradingStatus``; the other fields are null. ``EntryType`` is the
observation kind (the ``records`` values), ``SecurityID`` the zero-based directory index,
``TransactTime`` the event time in epoch nanoseconds and ``SecurityTradingStatus`` TradingHalt
for a halted observation (a halted quote keeps its prices). Missing source values are null on
the wire and NaN after decode: nothing is coerced. See docs/METHODOLOGY.md. Dates, the
instrument directory and the sender-session identity come from the HTTP manifest.
``decode_packet`` is the host oracle the GPU and numba decoders match bit for bit;
``entries_of`` is the oracle of the GPU encoder.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np

from ..marketdata import sbe_codec as sbe
from ..marketdata.records import CONDITION, EVENT_DTYPE, HALTED, QUOTE, TRADE, ReplayBatch

PROTOCOL = "fix-sbe-1.0-nyq-replay-v1"
TIMEZONE = "America/New_York"
MAX_PACKET_EVENTS = sbe.MAX_ENTRIES  # 30 entries: 24 + 30 x 48 = 1464 <= 1472
_ORDERED_ERROR = "SBE observations must be ordered within the configured market day"
# The observation kinds are the EntryType values: the kind byte is the wire byte.
assert (QUOTE, TRADE, CONDITION) == (sbe.QUOTE, sbe.TRADE, sbe.TRADING_STATUS)
_KINDS = np.zeros(256, bool)
_KINDS[[QUOTE, TRADE, CONDITION]] = True
_ENTRY = sbe.ENTRY_DTYPE
_U8 = np.uint8
# The 24-octet prefix of a packet (message header, root block, group dimensions) as one record.
_PREFIX = np.dtype(
    {
        "names": ["blockLength", "templateId", "schemaId", "version", "SeqNum", "entryLength", "count"],
        "formats": [sbe.ORDER + c for c in ("u2", "u2", "u2", "u2", "u8", "u2", "u2")],
        "offsets": [
            *(f.offset for f in sbe.SCHEMA.header.fields),
            sbe.HEADER.size + sbe.SCHEMA.root["SeqNum"].offset,
            *(sbe.HEADER.size + sbe.ROOT_BLOCK_LENGTH + f.offset for f in sbe.SCHEMA.group_size.fields),
        ],
        "itemsize": sbe.PACKET_HEADER,
    }
)  # fmt: skip


def day_start_ns(day):
    return int(datetime.fromisoformat(day).replace(tzinfo=ZoneInfo(TIMEZONE)).timestamp()) * 10**9


def _prices(values):
    """``sbe.price_to_wire`` over an array."""
    v = np.asarray(values, dtype=np.float64)
    missing = np.isnan(v)
    if not np.isfinite(v[~missing]).all():
        raise ValueError("SBE price must be finite or NaN")
    mantissa = np.rint(v * sbe.PRICE_SCALE)  # round(): ties to even, as np.rint
    given = mantissa[~missing]
    if ((given <= float(sbe.PRICE_NULL)) | (given >= 2.0**63)).any():
        raise ValueError("price exceeds the SBE PriceNULL4 range")
    out = np.full(len(v), sbe.PRICE_NULL, np.int64)
    out[~missing] = given.astype(np.int64)
    return out


def _quantities(values):
    """``sbe.qty_to_wire`` over an array."""
    v = np.asarray(values, dtype=np.float64)
    missing = np.isnan(v)
    given = v[~missing]
    if (
        not np.isfinite(given).all()
        or (given != np.floor(given)).any()
        or (given < 0).any()
        or (given >= sbe.QTY_NULL).any()
    ):
        raise ValueError("source quantity cannot be represented by SBE QtyNULL")
    out = np.full(len(v), sbe.QTY_NULL, np.uint32)
    out[~missing] = given.astype(np.uint32)
    return out


def entries_of(events, midnight_ns, unit_events=None):
    """The entries of ``events`` (an ``EVENT_DTYPE`` array), one per row, vectorized: the
    ``ENTRY_DTYPE`` record the GPU encoder reproduces byte for byte.

    ``unit_events`` rows form one packet (all of them without it): timestamps must lie in the
    market day and be non-decreasing within a packet. Rejected: another kind, a negative,
    fractional or overflowing quantity, an infinite or out-of-range price, an instrument
    beyond the SecurityID range."""
    ev = np.asarray(events)
    n = len(ev)
    ts = ev["timestamp_ns"].astype(np.int64)
    rel = ts - np.int64(midnight_ns)
    first = np.zeros(n, bool)
    first[:: unit_events or max(n, 1)] = True
    if (rel < 0).any() or (rel >= sbe.DAY_NS).any() or (~first[1:] & (ts[1:] < ts[:-1])).any():
        raise ValueError(_ORDERED_ERROR)
    kind = ev["kind"]
    if not _KINDS[kind].all():
        raise ValueError("unsupported source observation kind")
    symbol_id = ev["symbol_id"].astype(np.int64)
    if (symbol_id > sbe.MAX_SECURITY_ID).any():
        raise ValueError("source instrument exceeds the SBE SecurityID range")
    q, t, c = kind == QUOTE, kind == TRADE, kind == CONDITION
    halted = (ev["flags"] & HALTED) != 0
    out = np.zeros(n, _ENTRY)
    out["TransactTime"], out["SecurityID"], out["EntryType"] = ts, symbol_id, kind
    out["SecurityTradingStatus"] = np.where(
        halted, sbe.TRADING_HALT, np.where(c, sbe.READY_TO_TRADE, sbe.STATUS_NULL)
    )
    for name in ("BidPx", "OfferPx", "LastPx"):
        out[name] = sbe.PRICE_NULL
    for name in ("BidSize", "OfferSize", "LastQty"):
        out[name] = sbe.QTY_NULL
    out["BidPx"][q], out["OfferPx"][q] = _prices(ev["bid"][q]), _prices(ev["ask"][q])
    out["BidSize"][q], out["OfferSize"][q] = (
        _quantities(ev["bid_size"][q]),
        _quantities(ev["ask_size"][q]),
    )
    out["LastPx"][t], out["LastQty"][t] = _prices(ev["price"][t]), _quantities(ev["volume"][t])
    return out


def pack_events(events, sequence, midnight_ns):
    """One packet of ``events`` carrying ``sequence`` onward; returns ``(packet, entry
    count)``, the count being the number of observations (every entry consumes one
    sequence number)."""
    if not 1 <= len(events) <= MAX_PACKET_EVENTS:
        raise ValueError(f"packets must contain 1..{MAX_PACKET_EVENTS} observations")
    entries = entries_of(events, midnight_ns)
    return sbe.pack_packet(entries, sequence), len(entries)


def pack_units(events, sequence, midnight_ns, unit_events=MAX_PACKET_EVENTS):
    """``pack_events`` over consecutive packets of ``unit_events`` observations, vectorized.

    Returns ``(data, offsets, lengths, counts)``: the packets' bytes concatenated (uint8), and
    per packet its offset into ``data``, its length and its entry count. The first packet
    carries ``sequence`` and each packet consumes one sequence number per entry, so packet
    ``k`` is byte for byte ``pack_events(events[k * unit_events:(k + 1) * unit_events],
    sequence + k * unit_events, midnight_ns)[0]``; the same ValueErrors reject the same input.
    ``synthetic_capture`` and the replay sender pack in blocks with it; ``gpu_sbe.encode_units``
    is its GPU twin.
    """
    if not 1 <= unit_events <= MAX_PACKET_EVENTS:
        raise ValueError(f"packets must contain 1..{MAX_PACKET_EVENTS} observations")
    ev = np.asarray(events)
    n = len(ev)
    if n < 1:
        raise ValueError(f"packets must contain 1..{MAX_PACKET_EVENTS} observations")
    if sequence < 1 or sequence + n > sbe.MAX_SEQUENCE:
        raise ValueError("invalid SBE sequence number")
    entries = entries_of(ev, midnight_ns, unit_events)
    unit_starts = np.arange(0, n, unit_events)
    counts = np.diff(np.r_[unit_starts, n])
    lengths = sbe.PACKET_HEADER + sbe.ENTRY_BLOCK_LENGTH * counts
    offsets = np.r_[0, np.cumsum(lengths[:-1])]
    prefix = np.zeros(len(unit_starts), _PREFIX)
    prefix["blockLength"], prefix["templateId"] = sbe.ROOT_BLOCK_LENGTH, sbe.TEMPLATE_ID
    prefix["schemaId"], prefix["version"] = sbe.SCHEMA_ID, sbe.VERSION
    prefix["SeqNum"], prefix["entryLength"], prefix["count"] = (
        sequence + unit_starts,
        sbe.ENTRY_BLOCK_LENGTH,
        counts,
    )
    data = np.empty(int(offsets[-1] + lengths[-1]), _U8)
    is_prefix = np.zeros(len(data), bool)
    is_prefix[(offsets[:, None] + np.arange(sbe.PACKET_HEADER)).ravel()] = True
    data[is_prefix] = prefix.view(_U8)
    data[~is_prefix] = entries.view(_U8)
    return data, offsets, lengths, counts


def _px(mantissa):
    """``sbe.price_from_wire`` over an array: ``float64(mantissa) / 10000.0``, NaN for null."""
    return np.where(
        mantissa == sbe.PRICE_NULL, np.nan, mantissa.astype(np.float64) / sbe.PRICE_SCALE
    )


def _qty(value):
    return np.where(value == sbe.QTY_NULL, np.nan, value.astype(np.float64))


def decode_packet(data, recv_ns, *, generation, midnight_ns, universe):
    """The host reference decode of one packet: ``([ReplayBatch], entry count)``, or a
    ValueError under any rejection rule (framing, exact length, entry type, trading status,
    timestamp, instrument, ordering)."""
    sequence, e = sbe.parse_packet(data)
    kind, status, t = e["EntryType"], e["SecurityTradingStatus"], e["TransactTime"]
    if not _KINDS[kind].all():
        raise ValueError("unsupported SBE entry type")
    known = (status == sbe.TRADING_HALT) | (status == sbe.READY_TO_TRADE)
    if not (known | (status == sbe.STATUS_NULL)).all() or ((kind == CONDITION) & ~known).any():
        raise ValueError("unsupported SBE trading status")
    symbol_id = e["SecurityID"]
    midnight = np.uint64(midnight_ns)
    if (
        (t < midnight).any()
        or (t >= midnight + np.uint64(sbe.DAY_NS)).any()
        or (t[1:] < t[:-1]).any()
        or (symbol_id >= universe).any()
    ):
        raise ValueError("invalid SBE timestamp or instrument")
    n = len(e)
    rows = np.zeros(n, dtype=EVENT_DTYPE)
    rows["timestamp_ns"], rows["symbol_id"], rows["kind"] = t, symbol_id, kind
    rows["flags"] = np.where(status == sbe.TRADING_HALT, HALTED, 0)
    q, tr = kind == QUOTE, kind == TRADE
    nan = np.nan
    rows["bid"], rows["ask"] = (
        np.where(q, _px(e["BidPx"]), nan),
        np.where(q, _px(e["OfferPx"]), nan),
    )
    rows["bid_size"] = np.where(q, _qty(e["BidSize"]), nan)
    rows["ask_size"] = np.where(q, _qty(e["OfferSize"]), nan)
    rows["price"], rows["volume"] = (
        np.where(tr, _px(e["LastPx"]), nan),
        np.where(tr, _qty(e["LastQty"]), nan),
    )
    batch = ReplayBatch(rows, generation, recv_ns, sequence, wire_count=n)
    return [batch], n
