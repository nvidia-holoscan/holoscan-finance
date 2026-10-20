# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU quote bars: device-resident observations in, rolling windows of bar returns out.

Each ``events`` message is a list of :class:`holoscan.marketdata.gpu_bars.GpuReplayBatch`
(observation columns in GPU memory plus per-packet sequence numbers), as
:class:`holoscan.marketdata.GpuMitchSourceOp`'s decoder produces. Within the current
generation, stale packets are skipped and a sequence gap invalidates the marks, so no
return spans missing updates. Event-time quote-midpoint bars close on the GPU
(:class:`holoscan.marketdata.gpu_bars.GpuQuoteBars`) and stay in GPU memory.

After a bar closes, ``window`` carries a :class:`BarWindow`. The port never blocks:
connect it to an input that keeps only the latest message, as
:class:`holoscan.marketdata.GpuRollingRiskOp`'s does, so slow analytics skip windows
instead of stalling ingest.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np

from holoscan.core import ConditionType, ExecutionContext, InputContext, Operator, OutputContext

# Absolute imports: in the source tree this operator is a package under operators/, in the
# built package a flat holoscan.marketdata module.
from holoscan.marketdata.bars import QuoteBars
from holoscan.marketdata.gpu_bars import GpuQuoteBars, apply_sequence


@dataclass
class BarWindow:
    """The rolling window after a bar close. ``rows`` (bars x instruments, CuPy) is
    stacked from the per-bar return vectors on first access, so a window that the
    consumer skips costs no GPU work on the ingest path."""

    parts: list  # per-bar device return vectors, oldest first
    times: np.ndarray  # bar close times, ns (host)
    generation: int
    completed_bars: int
    recv_ns: int  # receive time of the packet that closed the newest bar
    _rows: object = field(default=None, repr=False)

    @property
    def rows(self):
        if self._rows is None:
            import cupy as cp

            self._rows = cp.stack(self.parts)
        return self._rows


class GpuQuoteBarsOp(Operator):
    """Settings are those of :class:`holoscan.marketdata.bars.QuoteBars`. ``lock`` guards
    the bar state; pass the application's lock (an ``RLock`` if it calls :meth:`reset`
    while holding it) to read ``bars`` consistently from other threads."""

    def __init__(
        self,
        fragment,
        *args,
        bar_seconds=5,
        window_seconds=300,
        stale_seconds=30,
        min_bars=20,
        max_spread_bps=100,
        lock=None,
        name="gpu_quote_bars",
        **kwargs,
    ):
        self.settings = dict(
            bar_seconds=bar_seconds,
            window_seconds=window_seconds,
            stale_seconds=stale_seconds,
            min_bars=min_bars,
            max_spread_bps=max_spread_bps,
        )
        QuoteBars([], **self.settings)  # validate before the graph starts
        self.lock = lock or threading.RLock()
        self.bars = None
        self.generation = self.expected_sequence = 0
        self.stale_packets = self.sequence_gaps = 0
        self.first_batch_ns = self.last_batch_ns = None
        self._emitted = 0
        super().__init__(fragment, *args, name=name, **kwargs)

    def setup(self, spec):
        spec.input("events")
        spec.output("window").condition(ConditionType.NONE)

    def reset(self, symbols, generation, expected_sequence=1):
        """Start a new instrument universe and feed generation. Batches of any other
        generation are then counted in ``stale_packets`` and skipped."""
        bars = GpuQuoteBars(symbols, **self.settings)
        with self.lock:
            self.bars, self.generation = bars, generation
            self.expected_sequence = expected_sequence
            self.first_batch_ns = self.last_batch_ns = None
            self._emitted = 0

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):
        del context
        batches = op_input.receive("events")
        window = None
        with self.lock:
            bars = self.bars
            if bars is None or not batches:
                return
            now = time.monotonic_ns()
            for batch in batches:
                if batch.generation != self.generation:
                    self.stale_packets += batch.n_packets
                    continue
                # First and last arrival of the current generation (burst timing).
                self.first_batch_ns = self.first_batch_ns or now
                self.last_batch_ns = now
                self.expected_sequence, stale, gaps = apply_sequence(
                    bars, batch, self.expected_sequence
                )
                self.stale_packets += stale
                self.sequence_gaps += gaps
            if bars.completed_bars != self._emitted:
                self._emitted = bars.completed_bars
                parts = bars.window_parts()
                if parts is not None:
                    window = BarWindow(
                        parts[0], parts[1], self.generation, bars.completed_bars, bars.bar_recv_ns
                    )
        if window is not None:
            op_output.emit(window, "window")


__all__ = ["BarWindow", "GpuQuoteBarsOp"]
