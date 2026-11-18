# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Device-resident event-time quote bars: a GPU mirror of :class:`bars.QuoteBars`.

Observations stay in GPU memory (straight from :mod:`gpu_sbe`). Per-symbol
state -- last valid quote midpoint, quote time, last trade, cumulative volume --
is updated with scatter-max ("last observation in tape order wins") and
scatter-add, segment by segment exactly as the host implementation does. Host
control flow only touches tiny arrays (segment boundaries, counters). Closed bars
(marks and return rows) stay in GPU memory; ``window()`` returns a device array
that the GPU covariance engine consumes in place.

``gpu_validate --bars`` validates it against ``QuoteBars`` over a full day: marks, quote
times, trades, volumes and counters must be identical; returns use the CUDA ``log`` and are
compared in ULPs.

``GpuQuoteBars.counts()`` and ``symbol_arrays()`` are the batched host mirrors the bars lane
publishes, so the HTTP thread never copies from the GPU.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .bars import QuoteBars
from .records import CONDITION, HALTED, QUOTE, TRADE


@dataclass
class GpuReplayBatch:
    """Observations of consecutive accepted packets, columns resident on the GPU."""

    columns: dict
    generation: int
    sequence: np.ndarray  # per packet (host)
    wire_count: np.ndarray  # per packet (host), SBE entries (one per observation)
    packet_offsets: np.ndarray  # per packet: index of its first observation
    packet_recv_ns: np.ndarray  # per packet receive wall clock
    recv_wall_ns: int

    @property
    def n_events(self):
        return int(self.columns["timestamp_ns"].shape[0])

    @property
    def n_packets(self):
        return len(self.sequence)


_BARS_SOURCE = r"""
typedef long long i64;
typedef unsigned int u32;
typedef unsigned char u8;

// Identical arithmetic to QuoteBars.observe's validity mask (compiled with
// --fmad=false so nothing is contracted into an FMA).
__device__ __forceinline__ bool valid_quote(u8 k, u8 f, double b, double a, double bs, double as,
                                            double msb, double* mid) {
    *mid = (b + a) * 0.5;
    return k == 1 && isfinite(*mid) && b > 0 && a >= b && bs > 0 && as > 0 &&
           (a - b) <= *mid * msb / 10000.0 && (f & 1) == 0;
}

// Phase 1, events [left, right) of one bar segment: last quote/halt and last valid
// trade position per symbol, trade volume (whole numbers: exact in any order), counters.
extern "C" __global__ void bars_scatter(
    const int left, const int right, const u32* __restrict__ sym, const u8* __restrict__ kind,
    const u8* __restrict__ flags, const double* __restrict__ bid, const double* __restrict__ ask,
    const double* __restrict__ bid_size, const double* __restrict__ ask_size,
    const double* __restrict__ price, const double* __restrict__ volume, const double msb,
    int* __restrict__ last_q, int* __restrict__ last_t, double* __restrict__ vol_total,
    unsigned long long* __restrict__ counts)  // quotes, trades, invalid_quotes
{
    __shared__ unsigned int s_q, s_t, s_inv;
    if (threadIdx.x == 0) { s_q = 0; s_t = 0; s_inv = 0; }
    __syncthreads();
    const int i = left + blockIdx.x * blockDim.x + threadIdx.x;
    if (i < right) {
        const u8 k = kind[i], f = flags[i];
        const u32 s = sym[i];
        if (k == 1 || (k == 3 && (f & 1))) atomicMax(&last_q[s], i);
        if (k == 1) {
            atomicAdd(&s_q, 1u);
            double mid;
            if (!valid_quote(k, f, bid[i], ask[i], bid_size[i], ask_size[i], msb, &mid)) atomicAdd(&s_inv, 1u);
        } else if (k == 2) {
            const double p = price[i];
            if (isfinite(p) && p > 0) {
                atomicAdd(&s_t, 1u);
                atomicMax(&last_t[s], i);
                const double v = volume[i];
                if (isfinite(v) && v > 0) atomicAdd(&vol_total[s], v);
            }
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        if (s_q) atomicAdd(&counts[0], (unsigned long long)s_q);
        if (s_t) atomicAdd(&counts[1], (unsigned long long)s_t);
        if (s_inv) atomicAdd(&counts[2], (unsigned long long)s_inv);
    }
}

// Phase 2, one thread per symbol: apply the last observations, reset scratch.
extern "C" __global__ void bars_apply(
    const int n, int* __restrict__ last_q, int* __restrict__ last_t, const i64* __restrict__ ts,
    const u8* __restrict__ kind, const u8* __restrict__ flags, const double* __restrict__ bid,
    const double* __restrict__ ask, const double* __restrict__ bid_size,
    const double* __restrict__ ask_size, const double* __restrict__ price, const double msb,
    double* __restrict__ mid_state, i64* __restrict__ quote_time, double* __restrict__ trade_state)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n) return;
    const int p = last_q[s];
    if (p >= 0) {
        double mid;
        const bool ok = valid_quote(kind[p], flags[p], bid[p], ask[p], bid_size[p], ask_size[p], msb, &mid);
        mid_state[s] = ok ? mid : __longlong_as_double(0x7FF8000000000000LL);
        quote_time[s] = ts[p];
        last_q[s] = -1;
    }
    const int t = last_t[s];
    if (t >= 0) { trade_state[s] = price[t]; last_t[s] = -1; }
}

// Bar close: marks = fresh mid else NaN; returns = log(marks / previous); previous = marks.
extern "C" __global__ void bars_close(
    const int n, const i64 boundary, const i64 stale_ns, const double* __restrict__ mid,
    const i64* __restrict__ quote_time, double* __restrict__ previous,
    double* __restrict__ marks_out, double* __restrict__ returns_out)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n) return;
    const i64 qt = quote_time[s];
    const double m = (qt > 0 && boundary - qt <= stale_ns) ? mid[s] : __longlong_as_double(0x7FF8000000000000LL);
    returns_out[s] = log(m / previous[s]);
    marks_out[s] = m;
    previous[s] = m;
}
"""

_bars_module = None


def _bars_kernels(cp):
    global _bars_module
    if _bars_module is None:
        _bars_module = cp.RawModule(code=_BARS_SOURCE, options=("-std=c++17", "--fmad=false"))
    return _bars_module


def _scatter(cp, op, target, indices, values):
    """``cp.<op>.at`` where available (CuPy >= 13), else the older cupyx scatter."""
    ufunc = getattr(cp, op)
    if hasattr(ufunc, "at"):
        ufunc.at(target, indices, values)
    else:
        import cupyx

        getattr(cupyx, "scatter_max" if op == "maximum" else "scatter_add")(target, indices, values)


class GpuQuoteBars:
    """Same constructor, counters, window() and attribute names as ``QuoteBars``."""

    def __init__(self, symbols, **settings):
        import cupy as cp

        QuoteBars([], **settings)  # identical validation
        self.cp = cp
        bar_seconds = settings.get("bar_seconds", 5)
        window_seconds = settings.get("window_seconds", 300)
        self.symbols = list(symbols)
        self.bar_ns = int(bar_seconds * 10**9)
        self.stale_ns = int(settings.get("stale_seconds", 30) * 10**9)
        self.window_bars = window_seconds // bar_seconds
        self.min_bars = settings.get("min_bars", 20)
        self.max_spread_bps = settings.get("max_spread_bps", 100)
        n = len(self.symbols)
        self.d_mid = cp.full(n, cp.nan)
        self.d_quote_time = cp.zeros(n, cp.int64)
        self.d_trade = cp.full(n, cp.nan)
        self.d_volume = cp.zeros(n, cp.float64)
        self.d_previous = cp.full(n, cp.nan)
        self._last = cp.empty(n, cp.int32)
        self._last_q = cp.full(n, -1, cp.int32)  # fused-path scratch, reset by bars_apply
        self._last_t = cp.full(n, -1, cp.int32)
        self._pending = []  # closed bars awaiting one batched device->host copy
        # device counters: events, quotes, trades, invalid_quotes, late_events
        self.d_counts = cp.zeros(5, cp.uint64)
        self.rows = deque(maxlen=self.window_bars)
        self.marks = deque(maxlen=self.window_bars)
        self.times = deque(maxlen=self.window_bars)
        self.boundary = None
        self.first_boundary = None
        self.last_event_ns = 0
        self.completed_bars = 0
        self.bar_recv_ns = 0

    # --- host views used by the dashboard snapshot (small, on demand) -------------
    def _counter(self, i):
        return int(self.d_counts[i].get())

    events = property(lambda self: self._counter(0))
    quotes = property(lambda self: self._counter(1))
    trades = property(lambda self: self._counter(2))
    invalid_quotes = property(lambda self: self._counter(3))
    late_events = property(lambda self: self._counter(4))
    mid = property(lambda self: self.cp.asnumpy(self.d_mid))
    trade = property(lambda self: self.cp.asnumpy(self.d_trade))
    volume = property(lambda self: self.cp.asnumpy(self.d_volume))
    quote_time = property(lambda self: self.cp.asnumpy(self.d_quote_time))
    previous = property(lambda self: self.cp.asnumpy(self.d_previous))

    def reset_marks(self):
        """Unknown missing updates (sequence gap): require refreshed marks."""
        self.d_mid.fill(self.cp.nan)
        self.d_previous.fill(self.cp.nan)

    def _close(self):
        cp = self.cp
        fresh = (self.d_quote_time > 0) & (self.boundary - self.d_quote_time <= self.stale_ns)
        marks = cp.where(fresh, self.d_mid, cp.nan)
        returns = cp.log(marks / self.d_previous)
        self.rows.append(returns)
        self.marks.append(marks)
        self.times.append(self.boundary)
        # Own buffer: the fused path and reset_marks() update previous in place,
        # which must never alter a stored bar.
        self.d_previous = marks.copy()
        self.completed_bars += 1

    def _last_per_symbol(self, ids, positions, mask):
        """Position of the last masked observation per symbol (-1 if none)."""
        last = self._last
        last.fill(-1)
        _scatter(self.cp, "maximum", last, ids, self.cp.where(mask, positions, -1))
        return last

    def observe(self, batch):
        """Fused fast path for in-order batches; exact general path otherwise."""
        if not batch.n_events:
            return
        try:
            if self._observe_fused(batch):
                return
        finally:
            self._flush()
        self._observe_general(batch)

    def _flush(self):
        """Publish bars closed during this batch; they stay in GPU memory."""
        for marks, returns, boundary in self._pending:
            self.marks.append(marks)
            self.rows.append(returns)
            self.times.append(boundary)
        self._pending.clear()

    def _close_fused(self):
        cp = self.cp
        n = len(self.symbols)
        marks, returns = cp.empty(n, cp.float64), cp.empty(n, cp.float64)
        _bars_kernels(cp).get_function("bars_close")(
            ((n + 255) // 256,),
            (256,),
            (
                np.int32(n),
                np.int64(self.boundary),
                np.int64(self.stale_ns),
                self.d_mid,
                self.d_quote_time,
                self.d_previous,
                marks,
                returns,
            ),
        )
        self._pending.append((marks, returns, self.boundary))
        self.completed_bars += 1

    def _observe_fused(self, batch):
        cp = self.cp
        c = batch.columns
        m = batch.n_events
        ts = c["timestamp_ns"]
        buckets = ts // self.bar_ns
        starts = cp.concatenate(
            [cp.zeros(1, cp.int64), cp.flatnonzero(buckets[1:] != buckets[:-1]) + 1]
        )
        k = int(starts.shape[0])
        disorder = (ts[1:] < ts[:-1]).any() if m > 1 else cp.zeros((), bool)
        host = cp.asnumpy(
            cp.concatenate([ts[starts], starts, ts[-1:], disorder.astype(cp.int64)[None]])
        )
        seg_ts, seg_start, last_ts, unordered = (
            host[:k],
            host[k : 2 * k],
            int(host[-2]),
            bool(host[-1]),
        )
        if unordered or int(seg_ts[0]) < self.last_event_ns:
            return False  # late or unordered observations: exact general path
        if self.boundary is None:
            self.boundary = (int(seg_ts[0]) // self.bar_ns + 1) * self.bar_ns
            self.first_boundary = self.boundary
        kern = _bars_kernels(cp)
        scatter, apply = kern.get_function("bars_scatter"), kern.get_function("bars_apply")
        n = len(self.symbols)
        msb = np.float64(self.max_spread_bps)
        splits = np.r_[seg_start, m]
        cols = (c["kind"], c["flags"], c["bid"], c["ask"], c["bid_size"], c["ask_size"])
        for s in range(k):
            left, right = int(splits[s]), int(splits[s + 1])
            ts0 = int(seg_ts[s])
            if ts0 - self.boundary > (self.window_bars + 1) * self.bar_ns:
                self._pending.clear()
                self.rows.clear()
                self.times.clear()
                self.marks.clear()
                self.d_previous.fill(cp.nan)
                self.boundary = (ts0 // self.bar_ns - self.window_bars) * self.bar_ns
            while ts0 >= self.boundary:
                self._close_fused()
                self.bar_recv_ns = batch.recv_wall_ns
                if batch.n_packets > 1:
                    packet = np.searchsorted(batch.packet_offsets, left, side="right") - 1
                    self.bar_recv_ns = int(batch.packet_recv_ns[packet])
                self.boundary += self.bar_ns
            scatter(
                ((right - left + 255) // 256,),
                (256,),
                (
                    np.int32(left),
                    np.int32(right),
                    c["symbol_id"],
                    *cols,
                    c["price"],
                    c["volume"],
                    msb,
                    self._last_q,
                    self._last_t,
                    self.d_volume,
                    self.d_counts[1:],
                ),
            )
            apply(
                ((n + 255) // 256,),
                (256,),
                (
                    np.int32(n),
                    self._last_q,
                    self._last_t,
                    ts,
                    *cols,
                    c["price"],
                    msb,
                    self.d_mid,
                    self.d_quote_time,
                    self.d_trade,
                ),
            )
        self.d_counts[0] += m
        self.last_event_ns = last_ts
        return True

    def _observe_general(self, batch):
        cp = self.cp
        c = batch.columns
        total = batch.n_events
        if not total:
            return
        ts_all = c["timestamp_ns"]
        good = (ts_all >= self.last_event_ns) & (c["symbol_id"] < len(self.symbols))
        gidx = cp.flatnonzero(good)
        m = int(gidx.shape[0])
        self.d_counts[4] += total - m
        if not m:
            return
        if m != total:
            c = {k: v[gidx] for k, v in c.items()}
        ts, sym = c["timestamp_ns"], c["symbol_id"].astype(cp.int64)
        kind, flags = c["kind"], c["flags"]
        if m > 1 and bool((ts[1:] < ts[:-1]).any()):
            raise ValueError("events within a packet must be time ordered")
        buckets = ts // self.bar_ns
        cuts = cp.flatnonzero(buckets[1:] != buckets[:-1]) + 1
        starts = cp.concatenate([cp.zeros(1, cp.int64), cuts])
        # One small D2H: segment first timestamps, their original indices, last timestamp.
        k = int(starts.shape[0])
        host = cp.asnumpy(cp.concatenate([ts[starts], gidx[starts], ts[-1:]]))
        seg_ts, seg_orig, last_ts = host[:k], host[k : 2 * k], int(host[-1])
        splits = np.r_[cp.asnumpy(starts), m]
        if self.boundary is None:
            self.boundary = (int(seg_ts[0]) // self.bar_ns + 1) * self.bar_ns
            self.first_boundary = self.boundary
        # Whole-batch elementwise terms (identical arithmetic to the host path).
        mid_all = (c["bid"] + c["ask"]) * 0.5
        is_quote = kind == QUOTE
        halted = (flags & HALTED) != 0
        qmask = is_quote | ((kind == CONDITION) & halted)
        valid_q = (
            is_quote
            & cp.isfinite(mid_all)
            & (c["bid"] > 0)
            & (c["ask"] >= c["bid"])
            & (c["bid_size"] > 0)
            & (c["ask_size"] > 0)
            & ((c["ask"] - c["bid"]) <= mid_all * self.max_spread_bps / 10000)
            & ~halted
        )
        price = c["price"]
        tmask = (kind == TRADE) & cp.isfinite(price) & (price > 0)
        vol = c["volume"]
        add_vol = cp.where(tmask & cp.isfinite(vol) & (vol > 0), vol, 0.0)
        positions = cp.arange(m, dtype=cp.int32)
        for s in range(len(splits) - 1):
            left, right = int(splits[s]), int(splits[s + 1])
            ts0 = int(seg_ts[s])
            if ts0 - self.boundary > (self.window_bars + 1) * self.bar_ns:
                self.rows.clear()
                self.times.clear()
                self.marks.clear()
                self.d_previous = cp.full(len(self.symbols), cp.nan)
                self.boundary = (ts0 // self.bar_ns - self.window_bars) * self.bar_ns
            while ts0 >= self.boundary:
                self._close()
                self.bar_recv_ns = batch.recv_wall_ns
                if batch.n_packets > 1:
                    packet = (
                        np.searchsorted(batch.packet_offsets, int(seg_orig[s]), side="right") - 1
                    )
                    self.bar_recv_ns = int(batch.packet_recv_ns[packet])
                self.boundary += self.bar_ns
            seg = slice(left, right)
            sid = sym[seg]
            # Quotes and halts: last one per symbol in tape order sets the mark.
            last = self._last_per_symbol(sid, positions[seg], qmask[seg])
            upd = last >= 0
            p = cp.maximum(last, 0)
            self.d_mid = cp.where(upd, cp.where(valid_q[p], mid_all[p], cp.nan), self.d_mid)
            self.d_quote_time = cp.where(upd, ts[p], self.d_quote_time)
            # Trades: last valid price per symbol; volume summed (integers: order-exact).
            last = self._last_per_symbol(sid, positions[seg], tmask[seg])
            upd = last >= 0
            p = cp.maximum(last, 0)
            self.d_trade = cp.where(upd, price[p], self.d_trade)
            _scatter(cp, "add", self.d_volume, sid, add_vol[seg])
            self.d_counts[:4] += cp.stack(
                [
                    cp.asarray(right - left, cp.int64),
                    is_quote[seg].sum(dtype=cp.int64),
                    tmask[seg].sum(dtype=cp.int64),
                    (is_quote[seg] & ~valid_q[seg]).sum(dtype=cp.int64),
                ]
            ).astype(cp.uint64)
        self.last_event_ns = last_ts

    def window_parts(self):
        """``(rows, times)`` without stacking: rows is a list of per-bar device return
        vectors, which are never modified after their bar closes; times is a host copy."""
        if len(self.rows) < self.min_bars:
            return None
        rows, times = list(self.rows), np.asarray(self.times, dtype=np.int64)
        if times[0] == self.first_boundary:
            rows, times = rows[1:], times[1:]
        return (rows, times.copy()) if len(rows) >= self.min_bars else None

    def window(self):
        """``(rows, times)``: rows is a device (bars, instruments) array, times host."""
        parts = self.window_parts()
        return None if parts is None else (self.cp.stack(parts[0]), parts[1])

    def counts(self) -> np.ndarray:
        """``uint64[5]`` (events, quotes, trades, invalid_quotes, late_events): one 40-byte D2H."""
        return self.d_counts.get()

    def symbol_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Host copies of (mid, trade, volume, quote_time) in one batched D2H.

        The rows travel as int64 bit patterns: a float64 cast would round quote_time (epoch
        ns, about 2^61) to 256 ns, and the bitwise CPU/GPU state check needs exact values."""
        cp = self.cp
        floats = (self.d_mid, self.d_trade, self.d_volume)
        host = cp.asnumpy(cp.stack([a.view(cp.int64) for a in floats] + [self.d_quote_time]))
        mid, trade, volume = host[:3].view(np.float64)
        return mid, trade, volume, host[3]


def select_packets(batch, indices):
    """Sub-batch of arbitrary packets (rare path: stale/out-of-order packets removed)."""
    import cupy as cp

    indices = np.asarray(indices, np.int64)
    ends = np.r_[batch.packet_offsets[1:], batch.n_events]
    starts, stops = batch.packet_offsets[indices], ends[indices]
    lengths = stops - starts
    event_index = (
        np.concatenate([np.arange(a, b) for a, b in zip(starts, stops)])
        if len(indices)
        else np.zeros(0, np.int64)
    )
    ev = cp.asarray(event_index)
    offsets = np.zeros(len(indices), np.int64)
    if len(indices) > 1:
        np.cumsum(lengths[:-1], out=offsets[1:])
    return GpuReplayBatch(
        {k: v[ev] for k, v in batch.columns.items()},
        batch.generation,
        batch.sequence[indices],
        batch.wire_count[indices],
        offsets,
        batch.packet_recv_ns[indices],
        int(batch.packet_recv_ns[indices[-1]]) if len(indices) else batch.recv_wall_ns,
    )


def apply_sequence(bars, batch, expected):
    """Mirror ``SequenceTracker.observe`` for one GPU batch of same-generation packets.

    Returns ``(expected, stale, gaps)``. The common case -- one contiguous run of
    sequence numbers -- is fully vectorized; anomalies fall back to the exact
    packet-by-packet rules (stale packets skipped, gaps flush pending packets and
    invalidate marks).
    """
    seq = batch.sequence.astype(np.int64)
    wire = batch.wire_count.astype(np.int64)
    n = len(seq)
    if not n:
        return expected, 0, 0
    if seq[0] >= expected and (n == 1 or np.array_equal(seq[1:], seq[:-1] + wire[:-1])):
        gap = int(seq[0] - expected) if expected and seq[0] > expected else 0
        if gap:
            bars.reset_marks()
        bars.observe(batch)
        return int(seq[-1] + wire[-1]), 0, gap
    stale = gaps = 0
    pending = []
    for i in range(n):
        if seq[i] < expected:
            stale += 1
            continue
        if expected and seq[i] > expected:
            if pending:
                bars.observe(select_packets(batch, pending))
                pending = []
            gaps += int(seq[i] - expected)
            bars.reset_marks()
        expected = int(seq[i] + wire[i])
        pending.append(i)
    if pending:
        bars.observe(select_packets(batch, pending))
    return expected, stale, gaps
