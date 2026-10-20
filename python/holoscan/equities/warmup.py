# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compile the GPU decode and bar kernels on synthetic NYQ MITCH units before live data."""

from __future__ import annotations

import numpy as np


def warmup(settings, device=0):
    """Compile (or load from CUPY_CACHE_DIR) every decode/bars kernel before live data.

    Without this the first market-data batches stall behind CUDA JIT compilation.
    Decodes a few synthetic MITCH units spanning several bar boundaries.
    """
    import cupy as cp

    from holoscan.marketdata.gpu_bars import GpuQuoteBars, GpuReplayBatch, apply_sequence
    from holoscan.marketdata.gpu_mitch import SLOT_BYTES, decode_packets
    from holoscan.marketdata.wire import EVENT_DTYPE

    from .mitch import pack_events

    cp.cuda.Device(device).use()
    midnight = 1_700_000_000 * 10**9 // 86400 // 10**9 * 86400 * 10**9
    symbols = ["A", "B", "C"]
    bars = GpuQuoteBars(symbols, **settings)
    bar_ns = int(settings.get("bar_seconds", 5) * 10**9)
    expected, seq = 1, 1
    for step in range(4):
        rows = np.zeros(6, EVENT_DTYPE)
        rows["timestamp_ns"] = midnight + 36000 * 10**9 + step * bar_ns + np.arange(6) * 10**6
        rows["symbol_id"] = [0, 1, 2, 0, 1, 2]
        rows["kind"] = [1, 1, 2, 1, 3, 2]
        for f in ("bid", "ask", "bid_size", "ask_size", "price", "volume"):
            rows[f] = np.nan
        rows["bid"][[0, 1, 3]], rows["ask"][[0, 1, 3]] = 10.0, 10.02
        rows["bid_size"][[0, 1, 3]] = rows["ask_size"][[0, 1, 3]] = 100
        rows["price"][[2, 5]], rows["volume"][[2, 5]] = 10.01, 5
        rows["flags"][4] = 1
        unit, count = pack_events(rows, seq, midnight)
        slots = cp.zeros((1, SLOT_BYTES), cp.uint8)
        slots[0, : len(unit)] = cp.asarray(np.frombuffer(unit, np.uint8))
        dec = decode_packets(
            slots, cp.asarray([len(unit)], cp.int32), midnight_ns=midnight, universe=3
        )
        batch = GpuReplayBatch(
            dec.columns, 1, dec.sequence, dec.wire_count, dec.offsets, np.zeros(1, np.int64), 0
        )
        expected, _, _ = apply_sequence(bars, batch, expected)
        seq += count
    bars.window()
    cp.cuda.Device(device).synchronize()
