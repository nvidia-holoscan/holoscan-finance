# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compiled CPU MITCH decode (numba), the CPU counterpart of ``marketdata.gpu_mitch``.

Same algorithm and output as the CUDA decoder -- and therefore as the Python
reference ``nyq.mapping.decode_packet`` -- but compiled to machine code and run over a
whole batch of packets at once: pass 1 validates and counts every packet, pass 2
writes the observations of accepted packets into structure-of-arrays columns.
Both passes are ``prange`` loops over packets, so the thread count is set with
``numba.set_num_threads``. It exists to separate "faster than interpreted Python"
from "faster than an optimized CPU" in the ingest benchmark.

Prices: for magnitudes up to 2**53 one IEEE division is correctly rounded and
equals Python's integer true division. Larger magnitudes (absent from real data,
present in fuzzed packets) are written as placeholders, recorded, and patched
with the Python reference conversion, so results stay bit-identical.

Benchmark only (``decode_bench``, ``gpu_validate --decoder cpu``): no graph uses it. From 4.3
``equities.cpu_mitch``, verbatim; needs numba>=0.68.
"""

from __future__ import annotations

import numpy as np

from ..marketdata import mitch_codec as unit
from ..marketdata.gpu_mitch import FIELDS, DecodedPackets

# An optional dependency: absent, or too old for this NumPy (numba 0.64, in the 5.0 image,
# refuses NumPy 2.5 at import), it is reported as missing, so callers and tests skip cleanly.
try:
    import numba
    from numba import njit, prange
except ImportError as exc:
    raise ModuleNotFoundError(f"cpu_mitch needs numba>=0.68 ({exc})", name="numba") from exc
if tuple(int(x) for x in numba.__version__.split(".")[:2]) < (0, 68):
    raise ModuleNotFoundError(
        f"cpu_mitch needs numba>=0.68, found {numba.__version__}", name="numba"
    )

_TWO53 = np.uint64(1 << 53)
_MAG = np.uint64((1 << 63) - 1)
_NAN = np.nan


@njit(inline="always")
def _u32(d, o):
    return (
        np.uint64(d[o])
        | (np.uint64(d[o + 1]) << np.uint64(8))
        | (np.uint64(d[o + 2]) << np.uint64(16))
        | (np.uint64(d[o + 3]) << np.uint64(24))
    )


@njit(inline="always")
def _u64(d, o):
    return _u32(d, o) | (_u32(d, o + 4) << np.uint64(32))


@njit(inline="always")
def _price(v):
    """(value, is_big): correctly rounded v / 1e8 with sign-magnitude, or a flag."""
    mag = v & _MAG
    if mag > _TWO53:
        return _NAN, True
    r = np.float64(mag) / 100000000.0
    if (v >> np.uint64(63)) != np.uint64(0) and mag != np.uint64(0):
        return -r, False
    return r, False


@njit(inline="always")
def _packet(d, L, midnight, universe, write, base, bbase, cols, big_idx, big_field, big_raw):
    """Returns (status, events, count, group, sequence, big_prices) for one unit."""
    ts, sym, kind, flags, bid, ask, bsz, asz, price, vol = cols
    if L < 8:
        return 1, 0, 0, -1, 0, 0
    length = np.int64(d[0]) | (np.int64(d[1]) << 8)
    count = np.int64(d[2])
    group = np.int64(d[3])
    seq = np.int64(_u32(d, 4))
    if (
        length != L
        or length < 8 + count * 2
        or (count == 0 and length != 8)
        or (count > 0 and (seq == 0 or seq + count > 4294967296))
    ):
        return 1, 0, count, group, seq, 0
    st = 0
    off = 8
    for _ in range(count):  # parse_unit: message kinds and sizes
        if off + 2 > L:
            st = 2
            break
        ml = np.int64(d[off])
        k = d[off + 1]
        need = 6 if k == 84 else 34 if k == 113 else 35 if k == 80 else 28 if k == 72 else 0
        if need == 0 or ml != need or off + ml > L:
            st = 2
            break
        off += ml
    if st == 0 and off != L:
        st = 2
    if st == 0 and group != 65:
        st = 2
    ev = 0
    nbig = 0
    if st == 0:  # decode_packet
        have_sec = False
        have_prev = False
        seconds = np.int64(0)
        prev = np.int64(0)
        off = 8
        for _ in range(count):
            p = off
            k = d[p + 1]
            off += np.int64(d[p])
            if k == 84:
                seconds = np.int64(_u32(d, p + 2))
                if seconds >= 86400:
                    st = 2
                    break
                have_sec = True
                continue
            if not have_sec:
                st = 2
                break
            b = _NAN
            a = _NAN
            bs = _NAN
            as_ = _NAN
            pr = _NAN
            vo = _NAN
            fl = 0
            raw0 = np.uint64(0)
            raw1 = np.uint64(0)
            big0 = False
            big1 = False
            if k == 113:
                ns = np.int64(_u32(d, p + 2))
                s = np.int64(_u32(d, p + 6))
                raw0 = _u64(d, p + 10)
                raw1 = _u64(d, p + 22)
                b, big0 = _price(raw0)
                a, big1 = _price(raw1)
                bs = np.float64(_u32(d, p + 18))
                as_ = np.float64(_u32(d, p + 30))
                kd = 1
            elif k == 80:
                ns = np.int64(_u32(d, p + 2))
                v = _u32(d, p + 6)
                s = np.int64(_u32(d, p + 10))
                if d[p + 14] or d[p + 15] or d[p + 32] or d[p + 33] or d[p + 34]:
                    st = 2
                    break
                raw0 = _u64(d, p + 16)
                pr, big0 = _price(raw0)
                vo = np.float64(v)
                kd = 2
            else:
                ns = np.int64(_u32(d, p + 2))
                s = np.int64(_u32(d, p + 6))
                c = d[p + 12]
                ok = d[p + 10] == 0 and d[p + 11] == 0 and (c == 72 or c == 84)
                ok = ok and d[p + 13] == 0 and d[p + 18] == 0 and d[p + 27] == 1
                for j in range(14, 18):
                    ok = ok and d[p + j] == 32
                for j in range(19, 27):
                    ok = ok and d[p + j] == 32
                if not ok:
                    st = 2
                    break
                kd = 3
                fl = 1 if c == 72 else 0
            t = midnight + seconds * 1000000000 + ns
            if ns >= 1000000000 or s < 1 or s > universe or (have_prev and t < prev):
                st = 2
                break
            have_prev = True
            prev = t
            if write:
                o = base + ev
                ts[o] = t
                sym[o] = s - 1
                kind[o] = kd
                flags[o] = fl
                bid[o] = b
                ask[o] = a
                bsz[o] = bs
                asz[o] = as_
                price[o] = pr
                vol[o] = vo
                if big0:
                    q = bbase + nbig
                    big_idx[q] = o
                    big_field[q] = 0 if kd == 1 else 2
                    big_raw[q] = raw0
                if big1:
                    q = bbase + nbig + (1 if big0 else 0)
                    big_idx[q] = o
                    big_field[q] = 1
                    big_raw[q] = raw1
            nbig += (1 if big0 else 0) + (1 if big1 else 0)
            ev += 1
    if st != 0:
        return st, 0, count, group, seq, 0
    return 0, ev, count, group, seq, nbig


@njit(parallel=True, cache=True)
def _inspect(slots, lens, midnight, universe, status, nev, wire, seq, group, nbig):
    dummy_i = np.empty(0, np.int64)
    dummy_u = np.empty(0, np.uint64)
    dummy_cols = (
        np.empty(0, np.int64), np.empty(0, np.uint32), np.empty(0, np.uint8), np.empty(0, np.uint8),
        np.empty(0, np.float64), np.empty(0, np.float64), np.empty(0, np.float64),
        np.empty(0, np.float64), np.empty(0, np.float64), np.empty(0, np.float64),
    )  # fmt: skip
    for i in prange(lens.shape[0]):
        r = _packet(slots[i], lens[i], midnight, universe, False, 0, 0, dummy_cols,
                    dummy_i, dummy_i, dummy_u)  # fmt: skip
        status[i], nev[i], wire[i], group[i], seq[i], nbig[i] = r


@njit(parallel=True, cache=True)
def _write(
    slots, lens, emit, midnight, universe, offsets, boffsets, cols, big_idx, big_field, big_raw
):
    for i in prange(lens.shape[0]):
        if emit[i]:
            _packet(slots[i], lens[i], midnight, universe, True, offsets[i], boffsets[i], cols,
                    big_idx, big_field, big_raw)  # fmt: skip


def decode_packets(slots, lens, *, midnight_ns, universe, keep=None):
    """Same contract as ``gpu_mitch.decode_packets``, with NumPy inputs and outputs."""
    n = int(lens.shape[0])
    status = np.empty(n, np.int64)
    nev = np.empty(n, np.int64)
    wire = np.empty(n, np.int64)
    seq = np.empty(n, np.int64)
    group = np.empty(n, np.int64)
    nbig = np.empty(n, np.int64)
    midnight, uni = np.int64(midnight_ns), np.int64(universe)
    if n:
        _inspect(slots, lens, midnight, uni, status, nev, wire, seq, group, nbig)
    emit = status == 0 if keep is None else (status == 0) & keep
    counts = np.where(emit, nev, 0)
    offsets = np.zeros(n, np.int64)
    boffsets = np.zeros(n, np.int64)
    if n > 1:
        np.cumsum(counts[:-1], out=offsets[1:])
        np.cumsum(np.where(emit, nbig, 0)[:-1], out=boffsets[1:])
    total = int(offsets[-1] + counts[-1]) if n else 0
    nb = int(boffsets[-1] + (nbig[-1] if emit[-1] else 0)) if n else 0
    columns = {name: np.empty(total, dt) for name, dt in FIELDS}
    if total:
        big_idx, big_field, big_raw = (
            np.empty(nb, np.int64),
            np.empty(nb, np.int64),
            np.empty(nb, np.uint64),
        )
        _write(slots, lens, emit, midnight, uni, offsets, boffsets,
               tuple(columns[name] for name, _ in FIELDS), big_idx, big_field, big_raw)  # fmt: skip
        names = ("bid", "ask", "price")
        for k in range(nb):  # magnitudes above 2**53: exact Python conversion
            columns[names[big_field[k]]][big_idx[k]] = unit.price_from_wire(int(big_raw[k]))
    return DecodedPackets(
        status.astype(np.int32), np.where(emit, nev, 0).astype(np.int32), wire.astype(np.int32),
        seq.astype(np.uint32), offsets, columns,
    )  # fmt: skip
