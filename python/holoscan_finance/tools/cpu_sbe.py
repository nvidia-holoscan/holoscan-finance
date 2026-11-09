# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compiled CPU SBE decode (numba), the CPU counterpart of ``marketdata.gpu_sbe``.

Same algorithm and output as the CUDA decoder -- and therefore as the Python reference
``nyq.mapping.decode_packet`` -- but compiled to machine code and run over a whole batch of
packets at once: pass 1 frames and validates every packet, pass 2 writes the observations
of accepted packets into structure-of-arrays columns. Both passes are ``prange`` loops over
packets, so the thread count is set with ``numba.set_num_threads``. It exists to separate
"faster than interpreted Python" from "faster than an optimized CPU" in the ingest benchmark.

Prices decode as ``float64(mantissa) / 10000.0``: the int64 converts to the nearest double,
then one IEEE division, as on the GPU and in the Python reference.

Benchmark only (``decode_bench``, ``gpu_validate --decoder cpu``): no graph uses it.
Needs numba>=0.68.
"""

from __future__ import annotations

import numpy as np

from ..marketdata import sbe_codec as sbe
from ..marketdata.gpu_sbe import FIELDS, DecodedPackets

# An optional dependency: absent, or too old for this NumPy (numba 0.64, in the 5.0 image,
# refuses NumPy 2.5 at import), it is reported as missing, so callers and tests skip cleanly.
try:
    import numba
    from numba import njit, prange
except ImportError as exc:
    raise ModuleNotFoundError(f"cpu_sbe needs numba>=0.68 ({exc})", name="numba") from exc
if tuple(int(x) for x in numba.__version__.split(".")[:2]) < (0, 68):
    raise ModuleNotFoundError(f"cpu_sbe needs numba>=0.68, found {numba.__version__}", name="numba")

_NAN = np.nan
_DAY_NS = np.int64(sbe.DAY_NS)
_MAX_SEQUENCE = np.int64(sbe.MAX_SEQUENCE)
_SCALE = float(sbe.PRICE_SCALE)
_PRICE_NULL = np.int64(sbe.PRICE_NULL)
_QTY_NULL = np.int64(sbe.QTY_NULL)
_STATUS_NULL, _HALT, _READY = sbe.STATUS_NULL, sbe.TRADING_HALT, sbe.READY_TO_TRADE
_QUOTE, _TRADE, _STATUS = sbe.QUOTE, sbe.TRADE, sbe.TRADING_STATUS
_SCHEMA_ID, _TEMPLATE_ID = sbe.SCHEMA_ID, sbe.TEMPLATE_ID
_HEADER, _GROUP_SIZE = sbe.HEADER.size, sbe.GROUP_SIZE.size
_ROOT_BLOCK, _ENTRY_BLOCK = sbe.ROOT_BLOCK_LENGTH, sbe.ENTRY_BLOCK_LENGTH
_E = {f.name: f.offset for f in sbe.SCHEMA.entries.fields}
_T, _BIDPX, _OFFERPX, _LASTPX = _E["TransactTime"], _E["BidPx"], _E["OfferPx"], _E["LastPx"]
_BIDSIZE, _OFFERSIZE, _LASTQTY = _E["BidSize"], _E["OfferSize"], _E["LastQty"]
_SECURITYID, _ENTRYTYPE, _TRADINGSTATUS = (
    _E["SecurityID"],
    _E["EntryType"],
    _E["SecurityTradingStatus"],
)
_SEQNUM = _HEADER + sbe.SCHEMA.root["SeqNum"].offset


@njit(inline="always")
def _u16(d, o):
    return np.int64(d[o]) | (np.int64(d[o + 1]) << 8)


@njit(inline="always")
def _u32(d, o):
    return _u16(d, o) | (_u16(d, o + 2) << 16)


@njit(inline="always")
def _i64(d, o):
    """Little-endian 64 bits as a (wrapping) int64, as the receive header carries them."""
    return _u32(d, o) | (_u32(d, o + 4) << 32)


@njit(inline="always")
def _frame(d, L):
    """(ok, count, sequence, entries offset, entry length): packet_frame_ok()."""
    if L < _HEADER:
        return False, 0, np.int64(0), 0, 0
    block = _u16(d, 0)
    if _u16(d, 2) != _TEMPLATE_ID or _u16(d, 4) != _SCHEMA_ID:
        return False, 0, np.int64(0), 0, 0
    if block < _ROOT_BLOCK or (block & 7) != (_ROOT_BLOCK & 7):
        return False, 0, np.int64(0), 0, 0
    at = _HEADER + block
    if L < at + _GROUP_SIZE:
        return False, 0, np.int64(0), 0, 0
    elen = _u16(d, at)
    count = _u16(d, at + 2)
    if elen < _ENTRY_BLOCK or (elen & 7) != 0:
        return False, 0, np.int64(0), 0, 0
    seq = _i64(d, _SEQNUM)
    eoff = at + _GROUP_SIZE
    if count == 0:
        return L == eoff, 0, seq, eoff, elen
    # A u64 sequence above int64 wraps negative: rejected with the other framing rules.
    ok = not (L < eoff + count * elen or seq <= 0 or seq > _MAX_SEQUENCE - count)
    return ok, count, seq, eoff, elen


@njit(inline="always")
def _price(m):
    return _NAN if m == _PRICE_NULL else np.float64(m) / _SCALE


@njit(inline="always")
def _qty(v):
    return _NAN if v == _QTY_NULL else np.float64(v)


@njit(inline="always")
def _packet(d, L, midnight, universe, write, base, cols):
    """Returns (status, events, count, sequence) for one packet."""
    ts, sym, kind, flags, bid, ask, bsz, asz, price, vol = cols
    framed, count, seq, eoff, elen = _frame(d, L)
    if not framed:
        return 1, 0, count, seq
    if L != eoff + count * elen:  # parse_packet: trailing octets
        return 2, 0, count, seq
    end = midnight + _DAY_NS
    prev = np.int64(0)
    for j in range(count):  # decode_packet: every entry's rules
        e = eoff + j * elen
        t = _i64(d, e + _T)
        k = d[e + _ENTRYTYPE]
        s = d[e + _TRADINGSTATUS]
        known = s == _HALT or s == _READY
        if (
            k < _QUOTE
            or k > _STATUS
            or not (known or s == _STATUS_NULL)
            or (k == _STATUS and not known)
            or np.uint64(t) < np.uint64(midnight)
            or np.uint64(t) >= np.uint64(end)
            or _u16(d, e + _SECURITYID) >= universe
            or (j > 0 and t < prev)
        ):
            return 2, 0, count, seq
        prev = t
    if write:
        for j in range(count):
            e = eoff + j * elen
            o = base + j
            k = d[e + _ENTRYTYPE]
            q = k == _QUOTE
            tr = k == _TRADE
            ts[o] = _i64(d, e + _T)
            sym[o] = _u16(d, e + _SECURITYID)
            kind[o] = k
            flags[o] = 1 if d[e + _TRADINGSTATUS] == _HALT else 0
            bid[o] = _price(_i64(d, e + _BIDPX)) if q else _NAN
            ask[o] = _price(_i64(d, e + _OFFERPX)) if q else _NAN
            bsz[o] = _qty(_u32(d, e + _BIDSIZE)) if q else _NAN
            asz[o] = _qty(_u32(d, e + _OFFERSIZE)) if q else _NAN
            price[o] = _price(_i64(d, e + _LASTPX)) if tr else _NAN
            vol[o] = _qty(_u32(d, e + _LASTQTY)) if tr else _NAN
    return 0, count, count, seq


@njit(parallel=True, cache=True)
def _inspect(slots, lens, midnight, universe, status, nev, wire, seq):
    dummy_cols = (
        np.empty(0, np.int64), np.empty(0, np.uint32), np.empty(0, np.uint8), np.empty(0, np.uint8),
        np.empty(0, np.float64), np.empty(0, np.float64), np.empty(0, np.float64),
        np.empty(0, np.float64), np.empty(0, np.float64), np.empty(0, np.float64),
    )  # fmt: skip
    for i in prange(lens.shape[0]):
        status[i], nev[i], wire[i], seq[i] = _packet(
            slots[i], lens[i], midnight, universe, False, 0, dummy_cols
        )


@njit(parallel=True, cache=True)
def _write(slots, lens, emit, midnight, universe, offsets, cols):
    for i in prange(lens.shape[0]):
        if emit[i]:
            _packet(slots[i], lens[i], midnight, universe, True, offsets[i], cols)


def decode_packets(slots, lens, *, midnight_ns, universe, keep=None):
    """Same contract as ``gpu_sbe.decode_packets``, with NumPy inputs and outputs."""
    n = int(lens.shape[0])
    status = np.empty(n, np.int64)
    nev = np.empty(n, np.int64)
    wire = np.empty(n, np.int64)
    seq = np.empty(n, np.int64)
    midnight, uni = np.int64(midnight_ns), np.int64(universe)
    if n:
        _inspect(slots, lens, midnight, uni, status, nev, wire, seq)
    emit = status == 0 if keep is None else (status == 0) & keep
    counts = np.where(emit, nev, 0)
    offsets = np.zeros(n, np.int64)
    if n > 1:
        np.cumsum(counts[:-1], out=offsets[1:])
    total = int(offsets[-1] + counts[-1]) if n else 0
    columns = {name: np.empty(total, dt) for name, dt in FIELDS}
    if total:
        _write(
            slots, lens, emit, midnight, uni, offsets, tuple(columns[name] for name, _ in FIELDS)
        )
    return DecodedPackets(
        status.astype(np.int32), np.where(emit, nev, 0).astype(np.int32), wire.astype(np.int32),
        seq.view(np.uint64), offsets, columns,
    )  # fmt: skip
