# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compile (or load from ``CUPY_CACHE_DIR``) the GPU kernels before live data.

Without this the first market-data batches stall behind CUDA JIT compilation. Each
operator warms only its own kernels in ``start()``: the sources the gather and decode
RawModule, the bars operator both bar paths and ``apply_sequence``. Both functions run once
per process and device and log "GPU <x> kernels ready in N s".
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Mapping

import numpy as np

from ..nyq.mapping import pack_events
from .framing import udp_frame
from .gpu_bars import GpuQuoteBars, GpuReplayBatch, apply_sequence
from .gpu_mitch import SLOT_BYTES, UNIT_HEADER, decode_packets, receive_frames
from .records import EVENT_DTYPE

logger = logging.getLogger(__name__)

_lock = threading.Lock()  # one warm-up at a time; later callers find the kernels ready
_warm: set[tuple[str, int]] = set()  # (kernels, device) warmed in this process

_MIDNIGHT = 1_700_000_000 * 10**9 // 86400 // 10**9 * 86400 * 10**9  # the warm-up day
_SYMBOLS = ["A", "B", "C"]


def _unit(step: int, sequence: int, bar_ns: int, *, late_ns: int = 0) -> tuple[bytes, int]:
    """One MITCH unit of quotes, trades and a halt in bar ``step``; ``late_ns`` moves the
    first observation that far back."""
    rows = np.zeros(6, EVENT_DTYPE)
    rows["timestamp_ns"] = _MIDNIGHT + 36000 * 10**9 + step * bar_ns + np.arange(6) * 10**6
    rows["timestamp_ns"][0] -= late_ns
    rows["symbol_id"] = [0, 1, 2, 0, 1, 2]
    rows["kind"] = [1, 1, 2, 1, 3, 2]
    for f in ("bid", "ask", "bid_size", "ask_size", "price", "volume"):
        rows[f] = np.nan
    rows["bid"][[0, 1, 3]], rows["ask"][[0, 1, 3]] = 10.0, 10.02
    rows["bid_size"][[0, 1, 3]] = rows["ask_size"][[0, 1, 3]] = 100
    rows["price"][[2, 5]], rows["volume"][[2, 5]] = 10.01, 5
    rows["flags"][4] = 1
    return pack_events(rows, sequence, _MIDNIGHT)


def _to_slots(cp, units: list[bytes]):
    host = np.zeros((len(units), SLOT_BYTES), np.uint8)
    for i, u in enumerate(units):
        host[i, : len(u)] = np.frombuffer(u, np.uint8)
    return cp.asarray(host), cp.asarray([len(u) for u in units], cp.int32)


def _once(kernels: str, gpu_device: int, warm) -> float:
    with _lock:
        if (kernels, gpu_device) in _warm:
            return 0.0
        import cupy as cp

        started = time.perf_counter()
        with cp.cuda.Device(gpu_device) as device:
            warm(cp)
            device.synchronize()
        seconds = time.perf_counter() - started
        _warm.add((kernels, gpu_device))
    logger.info("GPU %s kernels ready in %.1f s", kernels, seconds)
    return seconds


def warm_decode_kernels(gpu_device: int = 0) -> float:
    """Compile and run ``gather_frame`` and ``mitch_decode`` on synthetic units; returns seconds
    (0.0 when this device is already warm)."""

    def warm(cp) -> None:
        frames = [udp_frame(_unit(step, 1, 5 * 10**9)[0], src_port=40000) for step in range(2)]
        pool, _ = _to_slots(cp, frames)  # one frame per 2 KiB buffer, as DAQIRI leaves them
        addresses = pool.data.ptr + np.arange(len(frames), dtype=np.uint64) * SLOT_BYTES
        lengths = np.array([len(f) for f in frames], np.uint64)
        packed = cp.asarray(np.concatenate([addresses, lengths]))
        slots = cp.zeros_like(pool)
        header = receive_frames(packed, len(frames), slots, min_payload=UNIT_HEADER)
        # The sources decode with a stale-port mask, which takes the masked pass-2 path.
        keep = np.array([True, False])
        decode_packets(
            slots, header[:, 0].astype(cp.int32), midnight_ns=_MIDNIGHT, universe=3, keep=keep
        )

    return _once("decode", gpu_device, warm)


def warm_bars_kernels(settings: Mapping[str, float], gpu_device: int = 0) -> float:
    """The bars warm-up: bars across several bar boundaries (the fused path), then a
    sequence gap and a late observation (``select_packets`` and the general path); returns
    seconds (0.0 when this device is already warm)."""

    def warm(cp) -> None:
        bars = GpuQuoteBars(_SYMBOLS, **settings)
        bar_ns = int(settings.get("bar_seconds", 5) * 10**9)

        def observe(units: list[bytes], expected: int) -> int:
            slots, lens = _to_slots(cp, units)
            dec = decode_packets(slots, lens, midnight_ns=_MIDNIGHT, universe=len(_SYMBOLS))
            batch = GpuReplayBatch(
                dec.columns,
                1,
                dec.sequence,
                dec.wire_count,
                dec.offsets,
                np.zeros(len(units), np.int64),
                0,
            )
            return apply_sequence(bars, batch, expected)[0]

        expected = sequence = 1
        for step in range(4):
            unit, count = _unit(step, sequence, bar_ns)
            expected = observe([unit], expected)
            sequence += count
        late, count = _unit(4, sequence + 1, bar_ns, late_ns=bar_ns)
        gap, _ = _unit(5, sequence + 1 + count + 1, bar_ns)
        observe([late, gap], expected)
        bars.window()

    return _once("bars", gpu_device, warm)
