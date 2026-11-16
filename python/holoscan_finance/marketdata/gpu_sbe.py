# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU SBE (FIX Simple Binary Encoding) decode and encode, bit-compatible with ``equity.mapping``.

Decode. Packets arrive as a device-resident slot buffer (one UDP payload per 2 KiB row).
``sbe_decode`` runs one warp per packet and one lane per entry: the entries are fixed 48-octet
blocks on 8-octet boundaries, so every field is one aligned little-endian load, the lanes
validate their entries in parallel (``decode_packet``'s rules), one warp vote rejects the
packet if any entry fails, and the lanes write the observations straight into the final
structure-of-arrays columns, in wire order. Each packet's output position is known before the
launch (the receive header carries the framing status and entry count of every packet, or
``sbe_inspect`` reads them: 24 octets per packet), so a batch decodes in one launch and one
device-to-host copy of the per-packet results. Only a framed packet that a content rule
rejects (malformed input; never the replay's own packets) leaves a hole, which ``decode_packets``
compacts out afterwards.

Status per packet: 0 = decoded, 1 = rejected by ``inspect_packet`` (the receiver's framing
check), 2 = rejected by ``parse_packet``/``decode_packet``. ``gpu_validate`` checks field-for-field
equality with the host decoder over generated captures and mutated packets.

Encode. ``sbe_encode`` is ``equity.mapping.pack_units`` on the device: one thread per observation
writes its entry as six aligned 8-octet stores, the first thread of a packet its 24-octet
prefix; ``sbe_frame`` wraps the packets in pcap records and Ethernet/IPv4/UDP headers as
``marketdata.framing`` does. The encoder reports the first violated packing rule and the wrapper
raises the same ``ValueError`` as the host packer.

``gather_frame`` (via :func:`receive_frames`, one launch per burst) copies UDP payloads out of
the raw Ethernet frames DAQIRI DMA'd into GPU memory, applying ``strip_udp_datagram``'s
IPv4/UDP checks and then the ``inspect_packet`` framing check, and reports the
``HEADER_COLS``-column receive header per frame.

Prices decode as ``(double)mantissa / 10000.0``: the int64 converts to the nearest double
(ties to even), then one IEEE division, exactly ``sbe_codec.price_from_wire``. The module
compiles with ``--fmad=false`` so no expression contracts into an FMA.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import sbe_codec as sbe
from .framing import FRAME_PREFIX, RECORD_HLEN, udp_frame

SLOT_BYTES = 2048  # >= DAQIRI buf_size, so any received payload fits one slot
PACKET_HEADER = sbe.PACKET_HEADER  # message header, root block and group dimensions: 24
HEADER_COLS = 6  # the receive header: see receive_frames()
ENTRY_BYTES = sbe.ENTRY_BLOCK_LENGTH
_WARP = 32
_PACKETS_PER_BLOCK = 4
_ENCODE_BLOCK = 256

_SOURCE = (
    sbe.c_defines()
    + r"""
typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long long u64;
typedef long long i64;

// sbe_encode writes the 24-octet prefix as three words and packs neighbouring entry fields
// into one store; those stores spell out this layout.
#if SBE_HEADER != 8 || SBE_ROOT_BLOCK != 12 || SBE_GROUP_SIZE != 4 || SBE_R_SEQNUM != 0 || \
    SBE_ENTRY_BLOCK != 48 || SBE_E_OFFERSIZE != SBE_E_BIDSIZE + 4 || \
    SBE_E_SECURITYID != SBE_E_LASTQTY + 4 || SBE_E_ENTRYTYPE != SBE_E_LASTQTY + 6 || \
    SBE_E_SECURITYTRADINGSTATUS != SBE_E_LASTQTY + 7
#error "sbe_schema.xml changed its layout: update the stores of sbe_encode"
#endif

#define HEADER_COLS 6
#define HALT SBE_SECURITYTRADINGSTATUS_TRADINGHALT
#define READY SBE_SECURITYTRADINGSTATUS_READYTOTRADE
#define KIND_QUOTE SBE_ENTRYTYPE_QUOTE
#define KIND_TRADE SBE_ENTRYTYPE_TRADE
#define KIND_STATUS SBE_ENTRYTYPE_TRADINGSTATUS
#define HALTED_FLAG 1u

// SBE is little-endian, as the GPU is. Slot rows are 2 KiB apart from a 256-byte aligned
// allocation and the framing check keeps every entry on an 8-octet boundary, so each field
// is one aligned load.
__device__ __forceinline__ u32 ld16(const u8* p) { return *(const u16*)p; }
__device__ __forceinline__ u32 ld32(const u8* p) { return *(const u32*)p; }
__device__ __forceinline__ u64 ld64(const u8* p) { return *(const u64*)p; }
__device__ __forceinline__ void st64(u8* p, u64 v) { *(u64*)p = v; }

// sbe_codec.price_from_wire: (double)int64 rounds to nearest even, then one IEEE division.
__device__ __forceinline__ double price_from_wire(i64 m) { return (double)m / SBE_PRICE_SCALE; }
__device__ __forceinline__ double nan_value() { return __longlong_as_double(0x7FF8000000000000LL); }

// inspect_packet(): the receiver's framing check, shared by gather, inspect and decode so
// the kernels cannot drift apart. Reports count and sequence as soon as they are read.
__device__ __forceinline__ bool packet_frame_ok(const u8* d, int L, u32* count, i64* seq,
                                                int* eoff, int* elen) {
    *count = 0; *seq = 0; *eoff = 0; *elen = 0;
    if (L < SBE_HEADER) return false;
    const u32 block = ld16(d), tid = ld16(d + 2), sid = ld16(d + 4);
    if (tid != SBE_TEMPLATE_ID || sid != SBE_SCHEMA_ID) return false;
    if (block < SBE_ROOT_BLOCK || (block & 7u) != (SBE_ROOT_BLOCK & 7u)) return false;
    const int at = SBE_HEADER + (int)block;
    if (L < at + SBE_GROUP_SIZE) return false;
    const u32 el = ld16(d + at), n = ld16(d + at + 2);
    if (el < SBE_ENTRY_BLOCK || (el & 7u)) return false;
    const u64 s = ld64(d + SBE_HEADER + SBE_R_SEQNUM);
    *seq = (i64)s; *count = n; *eoff = at + SBE_GROUP_SIZE; *elen = (int)el;
    if (n == 0) return L == at + SBE_GROUP_SIZE;
    return !(L < *eoff + (int)n * (int)el || s == 0ULL || s > SBE_MAX_SEQUENCE - (u64)n);
}

// The receive header: one int64 row of HEADER_COLS per packet, as gather_frame writes it
// (payload_len, src_port, src_ip, framing_status, entry_count, sequence). Decode is driven by
// it, so a batch needs no host-to-device copy. A packet is emitted (decoded into the columns)
// when it passed framing, comes from sender_port (if >= 0) and keep[i] (if given) holds.
#define H_LEN 0
#define H_PORT 1
#define H_STATUS 3
#define H_COUNT 4
#define H_SEQ 5

__device__ __forceinline__ bool emitted(const i64* h, const u8* keep, const i64 sender_port,
                                        const int i) {
    return h[H_STATUS] == 0 && (sender_port < 0 || h[H_PORT] == sender_port) &&
           (keep == nullptr || keep[i] != 0);
}

// Framing only, one thread per packet: the receive header of a batch that did not come
// through gather_frame (benchmarks, validation), from the payload lengths.
extern "C" __global__ void sbe_inspect(
    const u8* __restrict__ slots, const int stride, const int* __restrict__ lens, const int n,
    i64* __restrict__ header)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    u32 count; i64 seq; int eoff, elen;
    const int L = lens[i];
    const bool ok = packet_frame_ok(slots + (size_t)i * stride, L, &count, &seq, &eoff, &elen);
    i64* h = header + (size_t)i * HEADER_COLS;
    h[H_LEN] = L; h[H_PORT] = -1; h[2] = 0; h[H_STATUS] = ok ? 0 : 1; h[H_COUNT] = count; h[H_SEQ] = seq;
}

// Exclusive scan of the emitted packets' entry counts: offsets[i] is where packet i's
// observations go, offsets[n] the total. One block, looping over the packets in chunks.
extern "C" __global__ void sbe_offsets(
    const i64* __restrict__ header, const u8* __restrict__ keep, const i64 sender_port,
    const int n, i64* __restrict__ offsets)
{
    __shared__ i64 s[1024];
    __shared__ i64 carry;
    if (threadIdx.x == 0) carry = 0;
    __syncthreads();
    for (int base = 0; base < n; base += blockDim.x) {
        const int i = base + threadIdx.x;
        const i64* h = header + (size_t)i * HEADER_COLS;
        const i64 v = i < n && emitted(h, keep, sender_port, i) ? h[H_COUNT] : 0;
        s[threadIdx.x] = v;
        __syncthreads();
        for (int d = 1; d < blockDim.x; d <<= 1) {
            const i64 t = threadIdx.x >= d ? s[threadIdx.x - d] : 0;
            __syncthreads();
            s[threadIdx.x] += t;
            __syncthreads();
        }
        if (i < n) offsets[i] = carry + s[threadIdx.x] - v;
        __syncthreads();
        if (threadIdx.x == blockDim.x - 1) carry += s[threadIdx.x];
        __syncthreads();
    }
    if (threadIdx.x == 0) offsets[n] = carry;
}

// One warp per packet, one lane per entry (lanes loop past 32 entries). Emitted packets
// write their observations at offsets[i] (offsets nullptr: validate only). results is
// int64 x 4 rows of n: status, observations, entry count, sequence.
extern "C" __global__ void sbe_decode(
    const u8* __restrict__ slots, const int stride, const i64* __restrict__ header,
    const u8* __restrict__ keep, const i64 sender_port, const int n,
    const u64 midnight_ns, const u32 universe, const i64* __restrict__ offsets,
    i64* __restrict__ results,
    i64* __restrict__ ts, u32* __restrict__ sym, u8* __restrict__ kind, u8* __restrict__ flags,
    double* __restrict__ bid, double* __restrict__ ask, double* __restrict__ bid_size,
    double* __restrict__ ask_size, double* __restrict__ price, double* __restrict__ volume)
{
    const int i = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const u32 lane = threadIdx.x & 31u;
    if (i >= n) return;  // warp-uniform: every lane of a live warp reaches the votes below
    const u8* d = slots + (size_t)i * stride;
    const i64* h = header + (size_t)i * HEADER_COLS;
    const int L = (int)h[H_LEN];
    u32 count; i64 seq; int eoff, elen;
    const bool framed = packet_frame_ok(d, L, &count, &seq, &eoff, &elen);
    int st = framed ? 0 : 1;
    if (framed && L != eoff + (int)count * elen) st = 2;  // parse_packet: trailing octets
    if (st == 0) {
        // decode_packet: every entry's type, status, time, instrument and order.
        bool bad = false;
        for (u32 j = lane; j < count; j += 32u) {
            const u8* e = d + eoff + j * elen;
            const u64 t = ld64(e + SBE_E_TRANSACTTIME);
            const u32 k = e[SBE_E_ENTRYTYPE], s = e[SBE_E_SECURITYTRADINGSTATUS];
            const u32 id = ld16(e + SBE_E_SECURITYID);
            const bool known = s == HALT || s == READY;
            bad |= k < KIND_QUOTE || k > KIND_STATUS || !(known || s == SBE_STATUS_NULL) ||
                   (k == KIND_STATUS && !known) || t < midnight_ns ||
                   t >= midnight_ns + SBE_DAY_NS || id >= universe ||
                   (j > 0u && t < ld64(e - elen + SBE_E_TRANSACTTIME));
        }
        if (__any_sync(0xffffffffu, bad)) st = 2;
    }
    const bool write = offsets != nullptr && emitted(h, keep, sender_port, i);
    if (st == 0 && write) {
        const i64 base = offsets[i];
        const double NaN = nan_value();
        for (u32 j = lane; j < count; j += 32u) {
            const u8* e = d + eoff + j * elen;
            const i64 o = base + (i64)j;
            const u32 k = e[SBE_E_ENTRYTYPE];
            const bool q = k == KIND_QUOTE, tr = k == KIND_TRADE;
            const i64 bm = (i64)ld64(e + SBE_E_BIDPX), am = (i64)ld64(e + SBE_E_OFFERPX);
            const i64 lm = (i64)ld64(e + SBE_E_LASTPX);
            const u32 bs = ld32(e + SBE_E_BIDSIZE), as = ld32(e + SBE_E_OFFERSIZE);
            const u32 lq = ld32(e + SBE_E_LASTQTY);
            ts[o] = (i64)ld64(e + SBE_E_TRANSACTTIME);
            sym[o] = ld16(e + SBE_E_SECURITYID);
            kind[o] = (u8)k;
            flags[o] = e[SBE_E_SECURITYTRADINGSTATUS] == HALT ? (u8)HALTED_FLAG : (u8)0;
            bid[o] = q && bm != SBE_PRICE_NULL ? price_from_wire(bm) : NaN;
            ask[o] = q && am != SBE_PRICE_NULL ? price_from_wire(am) : NaN;
            bid_size[o] = q && bs != SBE_QTY_NULL ? (double)bs : NaN;
            ask_size[o] = q && as != SBE_QTY_NULL ? (double)as : NaN;
            price[o] = tr && lm != SBE_PRICE_NULL ? price_from_wire(lm) : NaN;
            volume[o] = tr && lq != SBE_QTY_NULL ? (double)lq : NaN;
        }
    }
    if (lane == 0u) {
        results[i] = st; results[n + i] = st == 0 && write ? (i64)count : 0;
        results[2 * n + i] = (i64)count; results[3 * n + i] = seq;
    }
}

// Receive path, one launch per burst: gather each frame's UDP payload into its slot, then
// run the framing check on it. Header row (int64 x 6): payload_len (-1: not IPv4/UDP),
// src_port, src_ip, framing_status (0/1), entry_count, sequence.
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
        i64* h = header + (size_t)i * HEADER_COLS;
        h[0] = s_len; h[1] = s_port; h[2] = (i64)s_ip;
        if (s_len < 0) { h[3] = 1; h[4] = 0; h[5] = 0; return; }
        u32 count; i64 seq; int eoff, elen;
        const bool ok = packet_frame_ok(dst, s_len, &count, &seq, &eoff, &elen);
        h[3] = ok ? 0 : 1; h[4] = count; h[5] = seq;
    }
}

// ---- encode: equity.mapping.pack_units on the device --------------------------------------
// Packing-rule violations, reported as bits so the wrapper raises the host packer's error.
#define ERR_ORDER 1u
#define ERR_KIND 2u
#define ERR_SYMBOL 4u
#define ERR_PRICE 8u
#define ERR_QTY 16u

__device__ __forceinline__ i64 price_to_wire(double v, u32* err) {
    if (isnan(v)) return SBE_PRICE_NULL;
    if (isinf(v)) { *err |= ERR_PRICE; return 0; }
    const double m = rint(v * SBE_PRICE_SCALE);  // round(): ties to even
    if (m <= (double)SBE_PRICE_NULL || m >= 9223372036854775808.0) { *err |= ERR_PRICE; return 0; }
    return (i64)m;
}

__device__ __forceinline__ u32 qty_to_wire(double v, u32* err) {
    if (isnan(v)) return SBE_QTY_NULL;
    if (isinf(v) || v != floor(v) || v < 0.0 || v >= (double)SBE_QTY_NULL) { *err |= ERR_QTY; return 0; }
    return (u32)v;
}

// One thread per observation: its 48-octet entry, and for the first of a packet the 24-octet
// prefix. Packet k of unit_events observations starts at k * (24 + 48 * unit_events).
extern "C" __global__ void sbe_encode(
    const i64* __restrict__ ts, const u32* __restrict__ sym, const u8* __restrict__ kind,
    const u8* __restrict__ flags, const double* __restrict__ bid, const double* __restrict__ ask,
    const double* __restrict__ bid_size, const double* __restrict__ ask_size,
    const double* __restrict__ price, const double* __restrict__ volume, const int n,
    const int unit_events, const u64 sequence, const i64 midnight_ns,
    u8* __restrict__ data, u32* __restrict__ error)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n) return;
    const int k = r / unit_events, j = r - k * unit_events;
    const size_t unit_bytes = SBE_HEADER + SBE_ROOT_BLOCK + SBE_GROUP_SIZE + (size_t)SBE_ENTRY_BLOCK * unit_events;
    u8* p = data + (size_t)k * unit_bytes;
    u32 err = 0;
    const i64 t = ts[r], rel = t - midnight_ns;
    if (rel < 0 || rel >= (i64)SBE_DAY_NS || (j > 0 && t < ts[r - 1])) err |= ERR_ORDER;
    const u32 kd = kind[r];
    if (kd < KIND_QUOTE || kd > KIND_STATUS) err |= ERR_KIND;
    const u32 id = sym[r];
    if (id > 0xFFFFu) err |= ERR_SYMBOL;
    const bool q = kd == KIND_QUOTE, tr = kd == KIND_TRADE;
    const i64 bm = q ? price_to_wire(bid[r], &err) : SBE_PRICE_NULL;
    const i64 am = q ? price_to_wire(ask[r], &err) : SBE_PRICE_NULL;
    const i64 lm = tr ? price_to_wire(price[r], &err) : SBE_PRICE_NULL;
    const u32 bs = q ? qty_to_wire(bid_size[r], &err) : SBE_QTY_NULL;
    const u32 as = q ? qty_to_wire(ask_size[r], &err) : SBE_QTY_NULL;
    const u32 lq = tr ? qty_to_wire(volume[r], &err) : SBE_QTY_NULL;
    const u32 st = (flags[r] & HALTED_FLAG) ? HALT : (kd == KIND_STATUS ? READY : SBE_STATUS_NULL);
    if (err) { atomicOr(error, err); return; }
    if (j == 0) {
        const u32 count = min(unit_events, n - k * unit_events);
        st64(p, (u64)SBE_ROOT_BLOCK | ((u64)SBE_TEMPLATE_ID << 16) | ((u64)SBE_SCHEMA_ID << 32) |
                    ((u64)SBE_VERSION << 48));
        st64(p + 8, sequence + (u64)(k * unit_events));
        st64(p + 16, ((u64)SBE_ENTRY_BLOCK << 32) | ((u64)count << 48));  // 4 octets of root padding
    }
    u8* e = p + SBE_HEADER + SBE_ROOT_BLOCK + SBE_GROUP_SIZE + (size_t)j * SBE_ENTRY_BLOCK;
    st64(e + SBE_E_TRANSACTTIME, (u64)t);
    st64(e + SBE_E_BIDPX, (u64)bm);
    st64(e + SBE_E_OFFERPX, (u64)am);
    st64(e + SBE_E_LASTPX, (u64)lm);
    st64(e + SBE_E_BIDSIZE, (u64)bs | ((u64)as << 32));
    st64(e + SBE_E_LASTQTY, (u64)lq | ((u64)id << 32) | ((u64)kd << 48) | ((u64)st << 56));
}

// One block per packet: the nanosecond pcap record header, the Ethernet/IPv4/UDP prefix
// (as framing.udp_frame: IPv4 total length and header checksum follow the payload length)
// and the payload. Packets are consecutive in data at offsets[k] (lengths[k] octets); records
// are consecutive in out at out_offsets[k].
extern "C" __global__ void sbe_frame(
    const u8* __restrict__ data, const i64* __restrict__ offsets, const i64* __restrict__ lengths,
    const i64* __restrict__ stamps, const i64* __restrict__ out_offsets, const int n,
    const u8* __restrict__ prefix, u8* __restrict__ out)
{
    const int k = blockIdx.x;
    if (k >= n) return;
    const int L = (int)lengths[k];
    const u8* src = data + offsets[k];
    u8* dst = out + out_offsets[k];
    if (threadIdx.x == 0) {
        const i64 stamp = stamps[k];
        const u32 sec = (u32)(stamp / 1000000000LL), frac = (u32)(stamp - (i64)sec * 1000000000LL);
        const u32 incl = 42u + (u32)L;
        // Little-endian pcap record: ts_sec, ts_frac, incl_len, orig_len. Records are not
        // word aligned (58 + payload octets each), so octet stores.
        const u32 rec[4] = {sec, frac, incl, incl};
        for (int w = 0; w < 4; ++w)
            for (int b = 0; b < 4; ++b) dst[4 * w + b] = (u8)(rec[w] >> (8 * b));
        u8* fr = dst + 16;
        for (int b = 0; b < 42; ++b) fr[b] = prefix[b];
        const u32 total = 28u + (u32)L;
        fr[16] = (u8)(total >> 8); fr[17] = (u8)total;
        fr[24] = 0; fr[25] = 0;
        u32 sum = 0;
        for (int b = 14; b < 34; b += 2) sum += ((u32)fr[b] << 8) | fr[b + 1];
        while (sum > 0xFFFFu) sum = (sum & 0xFFFFu) + (sum >> 16);
        const u32 check = ~sum & 0xFFFFu;
        fr[24] = (u8)(check >> 8); fr[25] = (u8)check;
        const u32 ulen = 8u + (u32)L;
        fr[38] = (u8)(ulen >> 8); fr[39] = (u8)ulen;
    }
    __syncthreads();
    u8* payload = dst + 16 + 42;
    for (int b = threadIdx.x; b < L; b += blockDim.x) payload[b] = src[b];
}
"""
)

_module = None


def _kernels():
    global _module
    if _module is None:
        import cupy as cp

        # --fmad=false is a guard: no current expression contracts into an FMA, but the
        # decoder must stay bit-compatible with the host path if the code changes.
        _module = cp.RawModule(code=_SOURCE, options=("-std=c++17", "--fmad=false"))
    return _module


@dataclass
class DecodedPackets:
    """Per-packet header results (host) plus device-resident observation columns."""

    status: np.ndarray  # int32 per packet: 0 ok, 1 framing reject, 2 decode reject
    events_per_packet: np.ndarray  # int32
    wire_count: np.ndarray  # int32, entries in the packet (one per observation)
    sequence: np.ndarray  # uint64, sequence number of the first entry
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
_ENCODE_ERRORS = (  # the GPU encoder's error bits, in the host packer's order of checks
    (1, "SBE observations must be ordered within the configured market day"),
    (2, "unsupported source observation kind"),
    (4, "source instrument exceeds the SBE SecurityID range"),
    (8, "price exceeds the SBE PriceNULL4 range"),
    (16, "source quantity cannot be represented by SBE QtyNULL"),
)


_COLUMN_BYTES = sum(np.dtype(dt).itemsize for _, dt in FIELDS)  # 62 per observation


def inspect_packets(slots, lens):
    """Framing only (one thread per packet): a device ``(n, HEADER_COLS)`` receive header of
    every packet (length, no port or address, framing status, entry count, sequence), as
    ``receive_frames`` reports it for frames."""
    import cupy as cp

    n = int(lens.shape[0])
    header = cp.empty((n, HEADER_COLS), cp.int64)
    if n:
        _kernels().get_function("sbe_inspect")(
            ((n + 127) // 128,),
            (128,),
            (slots, np.int32(slots.shape[1]), lens, np.int32(n), header),
        )
    return header


def _columns(total):
    """The observation columns as views of one allocation, each on an 8-byte boundary."""
    import cupy as cp

    sizes = [(name, np.dtype(dt)) for name, dt in FIELDS]
    at, offsets = 0, {}
    for name, dt in sizes:
        offsets[name] = at
        at += -(-(total * dt.itemsize) // 8) * 8
    buffer = cp.empty(max(at, 8), cp.uint8)
    return {
        name: buffer[offsets[name] : offsets[name] + total * dt.itemsize].view(dt)
        for name, dt in sizes
    }


def _offsets(counts):
    offsets = np.zeros(len(counts), np.int64)
    if len(counts) > 1:
        np.cumsum(counts[:-1], out=offsets[1:])
    return offsets


def decode_packets(
    slots, lens=None, *, midnight_ns, universe, keep=None, header=None, sender_port=-1,
    max_events=None
):  # fmt: skip
    """Decode a device slot buffer (one packet per row).

    ``header`` is the device receive header of the rows (``receive_frames``); without it,
    ``lens`` (device int32 per packet) is framed by ``sbe_inspect`` first. A packet is emitted
    when it passed framing, its source port is ``sender_port`` (-1: any) and ``keep`` (bool per
    packet, host or device) holds; the others, and the packets a content rule rejects, emit
    no observations. ``max_events`` (host) bounds the emitted entries, as the sources know it
    from the header, and saves the device-to-host read of the exact total. One scan launch,
    one decode launch and one device-to-host copy of the per-packet results per call."""
    import cupy as cp

    if header is None:
        header = inspect_packets(slots, lens)
    n = int(header.shape[0])
    if keep is not None and not isinstance(keep, cp.ndarray):
        keep = cp.asarray(np.asarray(keep, bool))
    keep_d = keep.view(cp.uint8) if keep is not None else np.uint64(0)
    port = np.int64(sender_port)
    offsets_d = cp.empty(n + 1, cp.int64)
    results = cp.empty((4, n), cp.int64)
    if not n:
        return DecodedPackets(
            *(np.zeros(0, dt) for dt in (np.int32,) * 3 + (np.uint64, np.int64)), _columns(0)
        )
    k = _kernels()
    k.get_function("sbe_offsets")((1,), (1024,), (header, keep_d, port, np.int32(n), offsets_d))
    total = int(offsets_d[n].get()) if max_events is None else int(max_events)
    columns = _columns(total)
    k.get_function("sbe_decode")(
        ((n + _PACKETS_PER_BLOCK - 1) // _PACKETS_PER_BLOCK,),
        (_WARP * _PACKETS_PER_BLOCK,),
        (
            slots,
            np.int32(slots.shape[1]),
            header,
            keep_d,
            port,
            np.int32(n),
            np.uint64(midnight_ns),
            np.uint32(universe),
            offsets_d if total else np.uint64(0),
            results,
            *(columns[name] for name, _ in FIELDS),
        ),
    )
    status, nev, wire, seq = results.get()  # the one synchronizing copy
    status, nev, wire = (x.astype(np.int32) for x in (status, nev, wire))
    emitted_total = int(nev.sum())
    dev_offsets = None
    if ((status == 2) & (wire > 0)).any():
        # A content reject among the emitted packets leaves a hole: squeeze it out (malformed
        # input only; the host side of the emission rule is needed just here).
        dev_offsets = offsets_d.get()
        holes = np.flatnonzero((status == 2) & (dev_offsets[1:] > dev_offsets[:-1]))
        if len(holes):
            keep_ev = np.ones(int(dev_offsets[n]), bool)
            for i in holes:
                keep_ev[dev_offsets[i] : dev_offsets[i + 1]] = False
            index = cp.asarray(np.flatnonzero(keep_ev))
            columns = {name: col[index] for name, col in columns.items()}
    if dev_offsets is None and emitted_total < total:
        columns = {name: col[:emitted_total] for name, col in columns.items()}
    return DecodedPackets(
        status, nev, wire, seq.view(np.uint64), _offsets(nev.astype(np.int64)), columns
    )


def receive_frames(packed, n, slots, *, min_payload=PACKET_HEADER, stream=None):
    """Fused receive step: ``packed`` = device uint64[2n] (frame device addresses, then frame
    lengths). Fills ``slots`` and returns a device int64 ``(n, HEADER_COLS)`` header:
    payload_len, src_port, src_ip, framing_status, entry_count, sequence."""
    import cupy as cp

    header = cp.empty((n, HEADER_COLS), cp.int64)
    if n:
        _kernels().get_function("gather_frame")(
            (n,),
            (128,),
            (packed, np.int32(n), np.int32(min_payload), slots, np.int32(slots.shape[1]), header),
            stream=stream,
        )
    return header


def columns_from_events(events):
    """Device structure-of-arrays columns (the ``FIELDS`` layout, native byte order) of a
    host ``EVENT_DTYPE`` array: the encoder's input, and what ``decode_packets`` returns."""
    import cupy as cp

    ev = np.asarray(events)
    return {name: cp.asarray(np.ascontiguousarray(ev[name].astype(dt))) for name, dt in FIELDS}


def encode_units(columns, sequence, midnight_ns, unit_events=sbe.MAX_ENTRIES):
    """``equity.mapping.pack_units`` on the device: ``columns`` (``FIELDS`` device arrays of
    ``n`` observations) packed into consecutive packets of ``unit_events`` entries, the first
    carrying ``sequence``. Returns ``(data, offsets, lengths, counts)`` with ``data`` a device
    uint8 array holding the packets byte for byte as the host packer writes them; the same
    ``ValueError`` rejects the same input."""
    import cupy as cp

    if not 1 <= unit_events <= sbe.MAX_ENTRIES:
        raise ValueError(f"packets must contain 1..{sbe.MAX_ENTRIES} observations")
    n = int(columns["timestamp_ns"].shape[0])
    if n < 1:
        raise ValueError(f"packets must contain 1..{sbe.MAX_ENTRIES} observations")
    if sequence < 1 or sequence + n > sbe.MAX_SEQUENCE:
        raise ValueError("invalid SBE sequence number")
    unit_starts = np.arange(0, n, unit_events)
    counts = np.diff(np.r_[unit_starts, n])
    lengths = PACKET_HEADER + ENTRY_BYTES * counts
    offsets = np.r_[0, np.cumsum(lengths[:-1])]
    data = cp.empty(int(offsets[-1] + lengths[-1]), cp.uint8)
    error = cp.zeros(1, cp.uint32)
    _kernels().get_function("sbe_encode")(
        ((n + _ENCODE_BLOCK - 1) // _ENCODE_BLOCK,),
        (_ENCODE_BLOCK,),
        (
            *(columns[name] for name, _ in FIELDS),
            np.int32(n),
            np.int32(unit_events),
            np.uint64(sequence),
            np.int64(midnight_ns),
            data,
            error,
        ),
    )
    err = int(error.get()[0])
    for bit, message in _ENCODE_ERRORS:
        if err & bit:
            raise ValueError(message)
    return data, offsets, lengths, counts


def frame_units(data, offsets, lengths, stamps, *, src_port, dst_port):
    """The nanosecond pcap records of the packets at ``offsets``/``lengths`` of device
    ``data``, each an Ethernet/IPv4/UDP frame from ``src_port`` to ``dst_port`` stamped from
    ``stamps`` (host int64): a device uint8 array, byte for byte what ``synthetic._FrameWriter``
    writes."""
    import cupy as cp

    k = len(offsets)
    if not k:
        return cp.empty(0, cp.uint8)
    lengths = np.asarray(lengths, np.int64)
    out_lengths = RECORD_HLEN + FRAME_PREFIX + lengths
    out_offsets = np.r_[0, np.cumsum(out_lengths[:-1])]
    prefix = np.frombuffer(udp_frame(b"", src_port=src_port, dst_port=dst_port), np.uint8)
    out = cp.empty(int(out_offsets[-1] + out_lengths[-1]), cp.uint8)
    _kernels().get_function("sbe_frame")(
        (k,),
        (128,),
        (
            data,
            cp.asarray(np.asarray(offsets, np.int64)),
            cp.asarray(lengths),
            cp.asarray(np.asarray(stamps, np.int64)),
            cp.asarray(out_offsets),
            np.int32(k),
            cp.asarray(prefix[:FRAME_PREFIX]),
            out,
        ),
    )
    return out
