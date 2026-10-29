# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU MITCH (MIT303 11.9) decode, bit-compatible with ``nyq.mapping.decode_packet``.

Packets arrive as a device-resident slot buffer (one UDP payload per row). One
kernel, launched twice, mirrors the host decoder exactly: pass 1 validates every packet and counts
its observations (one thread per packet: MITCH units carry their own Time
messages, so packets decode independently); pass 2 writes the observations of
accepted packets, in wire order, into compact structure-of-arrays columns.

Status per packet: 0 = decoded, 1 = rejected by ``inspect_unit`` (the receiver's
framing check), 2 = rejected by ``parse_unit``/``decode_packet``. Every rejection
rule of the host decoder is reproduced; ``gpu_validate`` checks field-for-field
equality over generated replay captures and mutated packets.

``gather_frame`` (via :func:`receive_frames`, one launch per burst) copies UDP payloads
out of the raw Ethernet frames DAQIRI DMA'd into GPU memory, applying
``strip_udp_datagram``'s IPv4/UDP checks and then the ``inspect_unit`` framing check.

The module compiles with ``--device-int128`` (the exact 128-bit error in the correctly
rounded price division) and ``--fmad=false`` (no FMA contraction), so the decode stays
bit-identical to the host decoder.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SLOT_BYTES = 2048  # >= DAQIRI buf_size, so any received payload fits one slot
UNIT_HEADER = 8

_SOURCE = r"""
typedef unsigned char u8;
typedef unsigned int u32;
typedef unsigned long long u64;
typedef long long i64;

__device__ __forceinline__ u32 rd16(const u8* p) { return (u32)p[0] | ((u32)p[1] << 8); }
__device__ __forceinline__ u32 rd32(const u8* p) {
    return (u32)p[0] | ((u32)p[1] << 8) | ((u32)p[2] << 16) | ((u32)p[3] << 24);
}
__device__ __forceinline__ u64 rd64(const u8* p) { return (u64)rd32(p) | ((u64)rd32(p + 4) << 32); }

// Correctly rounded m / 1e8 for 0 <= m < 2^63 (Python int/int true division).
// Below 2^53 the numerator converts exactly, so one IEEE division is already
// correctly rounded. Above it, (double)m rounds first; pick the nearest of the
// neighbouring doubles using exact 128-bit integer error (ties to even).
__device__ double div_1e8_rn(u64 m) {
    if (m <= (1ULL << 53)) return (double)m / 100000000.0;
    const double y0 = (double)m / 100000000.0;
    double c = nextafter(nextafter(y0, 0.0), 0.0), best = y0;
    unsigned __int128 best_d = ~(unsigned __int128)0;
    for (int t = 0; t < 5; ++t) {
        int e;
        const double fr = frexp(c, &e);
        const u64 M = (u64)ldexp(fr, 53);  // c = M * 2^(e - 53), e - 53 in [-26, -16]
        const int shift = 40 + (e - 53);
        const unsigned __int128 a = (unsigned __int128)m << 40;
        const unsigned __int128 b = ((unsigned __int128)M * 100000000ULL) << shift;
        const unsigned __int128 d = a > b ? a - b : b - a;
        if (d < best_d || (d == best_d && (M & 1ULL) == 0)) { best_d = d; best = c; }
        c = nextafter(c, 1.0e308);
    }
    return best;
}

// Python: (-1 if v & SIGN else 1) * (v & (SIGN - 1)) / SCALE. The numerator is an
// exact integer, so a signed zero collapses to +0.0 exactly as in Python.
__device__ __forceinline__ double price_from_wire(u64 v) {
    const u64 mag = v & 0x7FFFFFFFFFFFFFFFULL;
    const double r = div_1e8_rn(mag);
    return ((v >> 63) && mag) ? -r : r;
}

// inspect_unit(): the receiver's framing check. Shared by decode and gather so the
// two kernels cannot drift apart.
__device__ __forceinline__ bool unit_frame_ok(const u8* d, int L, u32* count, u32* group, u32* seq) {
    if (L < 8) { *count = 0; *group = 0xFFFFFFFFu; *seq = 0; return false; }
    const u32 length = rd16(d);
    *count = d[2]; *group = d[3]; *seq = rd32(d + 4);
    return !(length != (u32)L || length < 8 + *count * 2 || (*count == 0 && length != 8) ||
             (*count > 0 && (*seq == 0 || (u64)*seq + *count > 0x100000000ULL)));
}

extern "C" __global__ void mitch_decode(
    const u8* __restrict__ slots, const int stride, const int* __restrict__ lens, const int n,
    const i64 midnight_ns, const u32 universe, const i64* __restrict__ offsets,
    int* __restrict__ status, int* __restrict__ nev, int* __restrict__ wire,
    u32* __restrict__ seq_out, int* __restrict__ group_out,
    i64* __restrict__ ts, u32* __restrict__ sym, u8* __restrict__ kind, u8* __restrict__ flags,
    double* __restrict__ bid, double* __restrict__ ask, double* __restrict__ bid_size,
    double* __restrict__ ask_size, double* __restrict__ price, double* __restrict__ volume)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const bool write = offsets != nullptr;
    if (write && status[i] != 0) return;
    const u8* d = slots + (size_t)i * stride;
    const int L = lens[i];

    // ---- inspect_unit (receiver framing check) -------------------------------------
    u32 count, group, seq;
    const bool framed = unit_frame_ok(d, L, &count, &group, &seq);
    if (!write) { seq_out[i] = seq; group_out[i] = (int)group; wire[i] = (int)count; }
    if (!framed) { if (!write) { status[i] = 1; nev[i] = 0; } return; }

    // ---- parse_unit: walk and size-check every message ------------------------------
    int st = 0, off = 8;
    for (u32 m = 0; m < count; ++m) {
        if (off + 2 > L) { st = 2; break; }
        const u32 ml = d[off], k = d[off + 1];
        const u32 need = k == 'T' ? 6u : k == 'q' ? 34u : k == 'P' ? 35u : k == 'H' ? 28u : 0u;
        if (need == 0 || ml != need || off + (int)ml > L) { st = 2; break; }
        off += (int)ml;
    }
    if (st == 0 && off != L) st = 2;
    if (st == 0 && group != 'A') st = 2;

    // ---- decode_packet --------------------------------------------------------------
    int ev = 0;
    if (st == 0) {
        const double NaN = __longlong_as_double(0x7FF8000000000000LL);
        const i64 base = write ? offsets[i] : 0;
        bool have_sec = false, have_prev = false;
        u32 seconds = 0;
        i64 prev = 0;
        off = 8;
        for (u32 m = 0; m < count; ++m) {
            const u8* p = d + off;
            const u32 k = p[1];
            off += p[0];
            if (k == 'T') {
                seconds = rd32(p + 2);
                if (seconds >= 86400u) { st = 2; break; }
                have_sec = true;
                continue;
            }
            if (!have_sec) { st = 2; break; }
            u32 ns, s;
            u8 kd, fl = 0;
            double b = NaN, a = NaN, bs = NaN, as = NaN, pr = NaN, vo = NaN;
            if (k == 'q') {
                ns = rd32(p + 2); s = rd32(p + 6);
                b = price_from_wire(rd64(p + 10)); bs = (double)rd32(p + 18);
                a = price_from_wire(rd64(p + 22)); as = (double)rd32(p + 30);
                kd = 1;
            } else if (k == 'P') {
                ns = rd32(p + 2); const u32 v = rd32(p + 6); s = rd32(p + 10);
                if (p[14] || p[15] || p[32] || p[33] || p[34]) { st = 2; break; }
                pr = price_from_wire(rd64(p + 16)); vo = (double)v;
                kd = 2;
            } else {  // 'H'
                ns = rd32(p + 2); s = rd32(p + 6);
                const u8 c = p[12];
                bool ok = p[10] == 0 && p[11] == 0 && (c == 'H' || c == 'T') && p[13] == 0 &&
                          p[18] == 0 && p[27] == 1;
                for (int j = 14; j < 18; ++j) ok = ok && p[j] == ' ';
                for (int j = 19; j < 27; ++j) ok = ok && p[j] == ' ';
                if (!ok) { st = 2; break; }
                kd = 3; fl = c == 'H' ? 1 : 0;
            }
            const i64 t = midnight_ns + (i64)seconds * 1000000000LL + (i64)ns;
            if (ns >= 1000000000u || s < 1u || s > universe || (have_prev && t < prev)) { st = 2; break; }
            have_prev = true; prev = t;
            if (write) {
                const i64 o = base + ev;
                ts[o] = t; sym[o] = s - 1; kind[o] = kd; flags[o] = fl;
                bid[o] = b; ask[o] = a; bid_size[o] = bs; ask_size[o] = as; price[o] = pr; volume[o] = vo;
            }
            ++ev;
        }
    }
    if (!write) { status[i] = st; nev[i] = st == 0 ? ev : 0; }
}

// Receive path, one launch per burst: gather each frame's UDP payload into its
// slot, then run the framing check on it. Header row (int64 x 7):
// payload_len (-1: not IPv4/UDP), src_port, src_ip, framing_status (0/1),
// message_count, sequence, group.
extern "C" __global__ void gather_frame(
    const u64* __restrict__ packed, const int n, const int min_payload,
    u8* __restrict__ slots, const int stride, i64* __restrict__ header)
{
    const int i = blockIdx.x;
    if (i >= n) return;
    const u8* f = (const u8*)packed[i];
    const int flen = (int)packed[n + i];
    __shared__ int s_off, s_len, s_port;
    __shared__ u32 s_ip;
    if (threadIdx.x == 0) {
        int off = -1, len = -1, port = -1;
        u32 ip = 0;
        if (flen >= 14 + 20 + 8 + min_payload && f[12] == 0x08 && f[13] == 0x00) {
            const int ihl = (f[14] & 0x0F) * 4;
            const int ip_length = ((int)f[16] << 8) | f[17];
            const int fragment = ((int)f[20] << 8) | f[21];
            if ((f[14] >> 4) == 4 && ihl >= 20 && f[14 + 9] == 17 && (fragment & 0x3FFF) == 0 &&
                ip_length >= ihl + 8 + min_payload && flen >= 14 + ip_length) {
                const int udp = 14 + ihl;
                const int udp_length = ((int)f[udp + 4] << 8) | f[udp + 5];
                if (udp_length >= 8 + min_payload && udp_length == ip_length - ihl &&
                    udp_length - 8 <= stride) {
                    off = udp + 8; len = udp_length - 8;
                    port = ((int)f[udp] << 8) | f[udp + 1];
                    ip = ((u32)f[26] << 24) | ((u32)f[27] << 16) | ((u32)f[28] << 8) | f[29];
                }
            }
        }
        s_off = off; s_len = len; s_port = port; s_ip = ip;
    }
    __syncthreads();
    u8* dst = slots + (size_t)i * stride;
    for (int j = threadIdx.x; j < s_len; j += blockDim.x) dst[j] = f[s_off + j];
    __syncthreads();
    if (threadIdx.x == 0) {
        i64* h = header + (size_t)i * 7;
        h[0] = s_len; h[1] = s_port; h[2] = (i64)s_ip;
        if (s_len < 0) { h[3] = 1; h[4] = 0; h[5] = 0; h[6] = -1; return; }
        u32 count, group, seq;
        const bool ok = unit_frame_ok(dst, s_len, &count, &group, &seq);
        h[3] = ok ? 0 : 1; h[4] = count; h[5] = seq; h[6] = (i64)(int)group;
    }
}
"""

_module = None


def _kernels():
    global _module
    if _module is None:
        import cupy as cp

        # --fmad=false is a guard: no current expression contracts into an FMA, but the
        # decoder must stay bit-compatible with the host path if the code changes.
        _module = cp.RawModule(
            code=_SOURCE, options=("-std=c++17", "--device-int128", "--fmad=false")
        )
    return _module


@dataclass
class DecodedPackets:
    """Per-packet header results (host) plus device-resident observation columns."""

    status: np.ndarray  # int32 per packet: 0 ok, 1 framing reject, 2 decode reject
    events_per_packet: np.ndarray  # int32
    wire_count: np.ndarray  # int32, MITCH messages in the unit (incl. Time)
    sequence: np.ndarray  # uint32
    offsets: np.ndarray  # int64 exclusive prefix of kept events
    columns: dict  # device arrays, one entry per observation field

    @property
    def total_events(self):
        return int(self.offsets[-1] + self.events_per_packet[-1]) if len(self.offsets) else 0


FIELDS = (
    ("timestamp_ns", "int64"),
    ("symbol_id", "uint32"),
    ("kind", "uint8"),
    ("flags", "uint8"),
    ("bid", "float64"),
    ("ask", "float64"),
    ("bid_size", "float64"),
    ("ask_size", "float64"),
    ("price", "float64"),
    ("volume", "float64"),
)


def inspect_packets(slots, lens, *, midnight_ns, universe):
    """Pass 1 on device; returns device per-packet status, events, wire count, sequence, group."""
    import cupy as cp

    n = int(lens.shape[0])
    status = cp.empty(n, cp.int32)
    nev = cp.empty(n, cp.int32)
    wire = cp.empty(n, cp.int32)
    seq = cp.empty(n, cp.uint32)
    group = cp.empty(n, cp.int32)
    if n:
        null = np.uint64(0)
        _kernels().get_function("mitch_decode")(
            ((n + 127) // 128,),
            (128,),
            (
                slots,
                np.int32(slots.shape[1]),
                lens,
                np.int32(n),
                np.int64(midnight_ns),
                np.uint32(universe),
                null,
                status,
                nev,
                wire,
                seq,
                group,
                null,
                null,
                null,
                null,
                null,
                null,
                null,
                null,
                null,
                null,
            ),
        )
    return status, nev, wire, seq, group


def decode_packets(slots, lens, *, midnight_ns, universe, keep=None):
    """Decode a device slot buffer. ``keep`` (host bool per packet) drops e.g. stale
    senders before pass 2; dropped and rejected packets emit no observations."""
    import cupy as cp

    n = int(lens.shape[0])
    status_d, nev_d, wire_d, seq_d, group_d = inspect_packets(
        slots, lens, midnight_ns=midnight_ns, universe=universe
    )
    status, nev, wire, seq = (cp.asnumpy(x) for x in (status_d, nev_d, wire_d, seq_d))
    emit = status == 0 if keep is None else (status == 0) & keep
    counts = np.where(emit, nev, 0).astype(np.int64)
    offsets = np.zeros(n, np.int64)
    if n > 1:
        np.cumsum(counts[:-1], out=offsets[1:])
    total = int(offsets[-1] + counts[-1]) if n else 0
    columns = {name: cp.empty(total, dt) for name, dt in FIELDS}
    if total:
        # Pass 2 writes only packets whose status is 0 on device; hide the dropped ones.
        status_w = status_d if keep is None else cp.where(cp.asarray(emit), status_d, 3)
        _kernels().get_function("mitch_decode")(
            ((n + 127) // 128,),
            (128,),
            (
                slots,
                np.int32(slots.shape[1]),
                lens,
                np.int32(n),
                np.int64(midnight_ns),
                np.uint32(universe),
                cp.asarray(offsets),
                status_w.astype(cp.int32, copy=False),
                nev_d,
                wire_d,
                seq_d,
                group_d,
                *(columns[name] for name, _ in FIELDS),
            ),
        )
    return DecodedPackets(
        status, np.where(emit, nev, 0).astype(np.int32), wire, seq, offsets, columns
    )


def receive_frames(packed, n, slots, *, min_payload=UNIT_HEADER, stream=None):
    """Fused receive step: ``packed`` = device uint64[2n] (frame device addresses,
    then frame lengths). Fills ``slots`` and returns a device int64 (n, 7) header:
    payload_len, src_port, src_ip, framing_status, message_count, sequence, group."""
    import cupy as cp

    header = cp.empty((n, 7), cp.int64)
    if n:
        _kernels().get_function("gather_frame")(
            (n,),
            (128,),
            (packed, np.int32(n), np.int32(min_payload), slots, np.int32(slots.shape[1]), header),
            stream=stream,
        )
    return header
