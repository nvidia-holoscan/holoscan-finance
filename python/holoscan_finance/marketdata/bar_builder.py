# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Quote bars for one feed generation on the GPU or with NumPy: the processor of ``QuoteBarsOp``.

Merges 4.3 ``GpuQuoteBarsOp.compute`` (gpu: ``GpuQuoteBars`` + ``apply_sequence``) and the
host ``EquityState.observe`` (cpu: ``QuoteBars`` + ``SequenceTracker``). Batches of other
generations are stale, a sequence gap resets the marks, and a window is produced only when a
bar closed. ``counters()`` and ``symbol_snapshot()`` give the immutable host mirrors that
the bars lane publishes, so the HTTP thread never touches CUDA (gpu: one 40-byte D2H per
batch; one batched D2H per snapshot).
"""

from __future__ import annotations

import contextlib
import time
from typing import TYPE_CHECKING

import numpy as np

from ..payloads import BarCounters, BarWindow, SymbolSnapshot
from ..settings import BarSettings
from .bars import QuoteBars
from .gpu_bars import GpuQuoteBars, GpuReplayBatch, apply_sequence
from .gpu_mitch import FIELDS
from .records import EVENT_DTYPE, ReplayBatch
from .sequence import SequenceTracker

if TYPE_CHECKING:
    from ..nyq.manifest import FeedDirectory


def host_batches_from_gpu(batch: GpuReplayBatch) -> list[ReplayBatch]:
    """Validation only: one D2H of the columns into ``EVENT_DTYPE`` rows, one ``ReplayBatch``
    per packet, so the cpu backend can consume the GPU sources' batches."""
    import cupy as cp

    n = batch.n_events
    # One copy of every column as raw bytes, split and reinterpreted on the host.
    raw = cp.asnumpy(cp.concatenate([batch.columns[name].view(cp.uint8) for name, _ in FIELDS]))
    rows = np.zeros(n, EVENT_DTYPE)  # internal_sequence stays 0, as in a host decode
    at = 0
    for name, dtype in FIELDS:
        size = n * np.dtype(dtype).itemsize
        rows[name] = raw[at : at + size].view(dtype)
        at += size
    ends = np.r_[batch.packet_offsets[1:], n]
    return [
        ReplayBatch(
            rows[int(batch.packet_offsets[k]) : int(ends[k])],
            batch.generation,
            int(batch.packet_recv_ns[k]),
            int(batch.sequence[k]),
            wire_count=int(batch.wire_count[k]),
        )
        for k in range(batch.n_packets)
    ]


class BarBuilder:
    """One bars engine (``QuoteBars`` or ``GpuQuoteBars``) plus the generation and sequence
    state in front of it. ``stale_packets`` and ``sequence_gaps`` are run totals."""

    backend: str
    generation: int | None
    expected: int
    stale_packets: int
    sequence_gaps: int
    engine: QuoteBars | GpuQuoteBars | None

    def __init__(self, settings: BarSettings, *, backend: str = "gpu", gpu_device: int = 0) -> None:
        if backend not in ("gpu", "cpu"):
            raise ValueError(f"bars backend must be gpu or cpu, got {backend!r}")
        self.settings, self.backend, self.gpu_device = settings, backend, gpu_device
        self._gpu = backend == "gpu"
        self.generation, self.expected = None, 1
        self.stale_packets = self.sequence_gaps = 0
        self.engine = None
        self._symbols: tuple[str, ...] = ()  # the directory's tuple, shared by every snapshot
        self._first_ns: int | None = None  # monotonic arrival of the generation's first batch
        self._last_ns: int | None = None
        self._emitted = 0  # completed_bars when the last window was considered

    def _device(self, active: bool = True) -> contextlib.AbstractContextManager:
        """The builder's CUDA device as the current one (a no-op unless ``active``)."""
        if not active:
            return contextlib.nullcontext()
        import cupy as cp

        return cp.cuda.Device(self.gpu_device)

    def needs_reset(self, directory: FeedDirectory | None) -> bool:
        """True when a directory is published whose generation differs from the engine's."""
        return directory is not None and directory.generation != self.generation

    def reset(self, directory: FeedDirectory) -> None:
        """A new engine for ``directory.symbols``; the next expected sequence is 1."""
        engine = GpuQuoteBars if self._gpu else QuoteBars
        with self._device(self._gpu):
            self.engine = engine(directory.symbols, **self.settings.as_params())
        self._symbols = directory.symbols
        self.generation, self.expected = directory.generation, 1
        self._first_ns = self._last_ns = None
        self._emitted = 0

    def observe(self, batches: list) -> BarWindow | None:
        """Apply one input message (``list[GpuReplayBatch]`` or ``list[ReplayBatch]``); the new
        window when a bar closed and enough bars exist, else None."""
        engine = self.engine
        if engine is None or not batches:
            return None
        if self._gpu:
            with self._device():
                self._observe_gpu(batches)
        else:
            self._observe_cpu(batches)
        if engine.completed_bars == self._emitted:
            return None
        self._emitted = engine.completed_bars
        parts = engine.window_parts()
        if parts is None:
            return None
        rows, times = parts
        times.flags.writeable = False
        if not self._gpu:
            # The engine never modifies a closed bar; read-only rows hold consumers to it too.
            for row in rows:
                row.flags.writeable = False
        return BarWindow(
            tuple(rows),
            times,
            self.generation,
            engine.completed_bars,
            engine.bar_recv_ns,
            device=self._gpu,
        )

    def _arrived(self) -> None:
        # First and last arrival of the current generation (burst timing, 4.3 _arrived).
        now = time.monotonic_ns()
        self._first_ns = self._first_ns or now
        self._last_ns = now

    def _observe_gpu(self, batches: list) -> None:
        for batch in batches:
            if not isinstance(batch, GpuReplayBatch):
                raise TypeError(f"gpu bars take GpuReplayBatch, not {type(batch).__name__}")
            if batch.generation != self.generation:
                self.stale_packets += batch.n_packets
                continue
            self._arrived()
            self.expected, stale, gaps = apply_sequence(self.engine, batch, self.expected)
            self.stale_packets += stale
            self.sequence_gaps += gaps

    def _observe_cpu(self, batches: list) -> None:
        host: list[ReplayBatch] = []
        for batch in batches:
            if isinstance(batch, ReplayBatch):
                host.append(batch)
            elif isinstance(batch, GpuReplayBatch):
                with self._device():
                    host.extend(host_batches_from_gpu(batch))
            else:
                raise TypeError(
                    f"cpu bars take ReplayBatch or GpuReplayBatch, not {type(batch).__name__}"
                )
        if any(b.generation == self.generation for b in host):
            self._arrived()
        tracker = SequenceTracker(self.generation, self.expected)
        tracker.observe(self.engine, host)
        self.expected = tracker.expected
        self.stale_packets += tracker.stale_packets
        self.sequence_gaps += tracker.sequence_gaps

    def counters(self) -> BarCounters | None:
        """The engine's exact counters (gpu: one 40-byte D2H); None before the first reset."""
        engine = self.engine
        if engine is None:
            return None
        with self._device(self._gpu):
            counts = engine.counts()
        events, quotes, trades, invalid_quotes, late_events = (int(c) for c in counts)
        return BarCounters(
            generation=self.generation,
            events=events,
            quotes=quotes,
            trades=trades,
            invalid_quotes=invalid_quotes,
            late_events=late_events,
            completed_bars=engine.completed_bars,
            last_event_ns=int(engine.last_event_ns),
            first_batch_ns=self._first_ns,
            last_batch_ns=self._last_ns,
        )

    def symbol_snapshot(self) -> SymbolSnapshot | None:
        """Read-only host copies of the per-symbol state (gpu: one batched D2H); None before
        the first reset."""
        engine = self.engine
        if engine is None:
            return None
        with self._device(self._gpu):
            arrays = engine.symbol_arrays()
        for a in arrays:
            a.flags.writeable = False
        mid, trade, volume, quote_time = arrays
        return SymbolSnapshot(
            generation=self.generation,
            symbols=self._symbols,
            mid=mid,
            trade=trade,
            volume=volume,
            quote_time=quote_time,
            last_event_ns=int(engine.last_event_ns),
            taken_ns=time.time_ns(),
        )
