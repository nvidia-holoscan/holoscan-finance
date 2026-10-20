# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Event-time quote midpoint bars; no trades mixed into the return estimator."""

from collections import deque

import numpy as np

from .wire import CONDITION, HALTED, QUOTE, TRADE


class QuoteBars:
    def __init__(
        self,
        symbols,
        *,
        bar_seconds=5,
        window_seconds=300,
        stale_seconds=30,
        max_spread_bps=100,
        min_bars=20,
    ):
        if (
            bar_seconds < 1
            or window_seconds % bar_seconds
            or window_seconds < 2 * bar_seconds
            or stale_seconds < bar_seconds
            or not np.isfinite(max_spread_bps)
            or max_spread_bps <= 0
            or not 2 <= min_bars <= window_seconds // bar_seconds
        ):
            raise ValueError("invalid bar, window, staleness or minimum-observation settings")
        self.symbols = list(symbols)
        self.bar_ns = int(bar_seconds * 10**9)
        self.stale_ns = int(stale_seconds * 10**9)
        self.window_bars = window_seconds // bar_seconds
        self.min_bars = min_bars
        self.max_spread_bps = max_spread_bps
        n = len(symbols)
        self.mid = np.full(n, np.nan)
        self.quote_time = np.zeros(n, dtype=np.int64)
        self.trade = np.full(n, np.nan)
        self.volume = np.zeros(n, dtype=np.float64)
        self.previous = np.full(n, np.nan)
        self.rows = deque(maxlen=self.window_bars)
        self.marks = deque(maxlen=self.window_bars)
        self.times = deque(maxlen=self.window_bars)
        self.boundary = None
        self.first_boundary = None
        self.last_event_ns = 0
        self.events = self.quotes = self.trades = self.invalid_quotes = self.late_events = 0
        self.completed_bars = 0
        self.bar_recv_ns = 0

    def _close(self):
        fresh = (self.quote_time > 0) & (self.boundary - self.quote_time <= self.stale_ns)
        marks = np.where(fresh, self.mid, np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            returns = np.log(marks / self.previous)
        self.rows.append(returns)
        self.marks.append(marks.copy())
        self.times.append(self.boundary)
        self.previous = marks.copy()
        self.completed_bars += 1

    def observe(self, batch):
        a = batch.events
        if not len(a):
            return
        # Replay is sorted. A reordered UDP packet cannot rewrite completed bars.
        good = (a["timestamp_ns"] >= self.last_event_ns) & (a["symbol_id"] < len(self.symbols))
        self.late_events += int((~good).sum())
        original_indices = np.flatnonzero(good)
        a = a[good]
        if not len(a):
            return
        if np.any(np.diff(a["timestamp_ns"].astype(np.int64)) < 0):
            raise ValueError("events within a packet must be time ordered")
        if self.boundary is None:
            self.boundary = (int(a["timestamp_ns"][0]) // self.bar_ns + 1) * self.bar_ns
            self.first_boundary = self.boundary
        # Work once per bar segment; last observation per symbol wins in tape order.
        buckets = a["timestamp_ns"] // self.bar_ns
        splits = np.r_[0, np.flatnonzero(np.diff(buckets)) + 1, len(a)]
        for left, right in zip(splits[:-1], splits[1:]):
            segment = a[left:right]
            ts = int(segment["timestamp_ns"][0])
            # Long gaps yield missing bars; cap processing to the maintained horizon.
            if ts - self.boundary > (self.window_bars + 1) * self.bar_ns:
                self.rows.clear()
                self.times.clear()
                self.marks.clear()
                self.previous[:] = np.nan
                self.boundary = (ts // self.bar_ns - self.window_bars) * self.bar_ns
            while ts >= self.boundary:
                self._close()
                self.bar_recv_ns = batch.recv_wall_ns
                if batch.packet_offsets is not None:
                    packet = (
                        np.searchsorted(batch.packet_offsets, original_indices[left], side="right")
                        - 1
                    )
                    self.bar_recv_ns = int(batch.packet_recv_ns[packet])
                self.boundary += self.bar_ns
            is_quote = segment["kind"] == QUOTE
            is_halt = (segment["kind"] == CONDITION) & ((segment["flags"] & HALTED) != 0)
            quote = segment[is_quote | is_halt]
            if len(quote):
                ids = quote["symbol_id"].astype(np.int64)
                mid = (quote["bid"] + quote["ask"]) * 0.5
                valid = (
                    (quote["kind"] == QUOTE)
                    & np.isfinite(mid)
                    & (quote["bid"] > 0)
                    & (quote["ask"] >= quote["bid"])
                    & (quote["bid_size"] > 0)
                    & (quote["ask_size"] > 0)
                    & ((quote["ask"] - quote["bid"]) <= mid * self.max_spread_bps / 10000)
                    & ((quote["flags"] & HALTED) == 0)
                )
                self.invalid_quotes += int(((quote["kind"] == QUOTE) & ~valid).sum())
                _, reverse = np.unique(ids[::-1], return_index=True)
                last = len(ids) - 1 - reverse
                self.mid[ids[last]] = np.where(valid[last], mid[last], np.nan)
                self.quote_time[ids[last]] = quote["timestamp_ns"][last]
                self.quotes += int(is_quote.sum())
            trade = segment[segment["kind"] == TRADE]
            if len(trade):
                valid = np.isfinite(trade["price"]) & (trade["price"] > 0)
                trade = trade[valid]
                ids = trade["symbol_id"].astype(np.int64)
                _, reverse = np.unique(ids[::-1], return_index=True)
                last = len(ids) - 1 - reverse
                self.trade[ids[last]] = trade["price"][last]
                volume = np.asarray(trade["volume"], dtype=float)
                np.add.at(self.volume, ids, np.where(np.isfinite(volume) & (volume > 0), volume, 0))
                self.trades += len(trade)
            self.events += len(segment)
        self.last_event_ns = int(a["timestamp_ns"][-1])

    def window(self):
        if len(self.rows) < self.min_bars:
            return None
        # The first bar has no previous midpoint: wait until enough post-start returns.
        rows, times = np.asarray(self.rows), np.asarray(self.times, dtype=np.int64)
        if times[0] == self.first_boundary:
            rows, times = rows[1:], times[1:]
        return (rows.copy(), times.copy()) if len(rows) >= self.min_bars else None
