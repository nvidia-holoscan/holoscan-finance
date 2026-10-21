# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``QuoteBarsOp``: event-time quote bars on the GPU (device-resident) or with NumPy.

Ports: ``events`` (in: ``list[GpuReplayBatch]`` from the GPU sources, ``list[ReplayBatch]``
from ``MitchDecodeOp``) -> ``window`` (out, ``BarWindow``; only with ``emit_windows``). Edges:
in, ``lossless(4)`` from a GPU source or ``lossless(16)`` from decode, same lane; out,
``LATEST`` to risk in another lane (bars never block on analytics) or ``lossless(64)`` to a
same-lane sink when every window matters. Lane: the source's (``ingest``). A new directory
generation resets the bars at the first batch decoded under it (4.3: at the HTTP call), so
batches still in flight under the old one are built, not stale. Per batch the exact
counters go to ``BarsStats.counters`` (GPU: one 40-byte D2H), and a ``SymbolSnapshot`` at most
every ``snapshot_period_ms``, so the dashboard never touches CUDA. The cpu backend also takes
GPU batches (one D2H each): the bench's reference only. Processor: ``BarBuilder`` (4.3
``GpuQuoteBarsOp``, the host ``BarsOp`` and ``EquityState.observe``).
"""

from __future__ import annotations

import time
from contextlib import nullcontext

import holoscan as hs

from ..marketdata.bar_builder import BarBuilder
from ..marketdata.gpu_bars import GpuReplayBatch
from ..marketdata.warmup import warm_bars_kernels
from ..ref import Ref
from ..settings import BarSettings
from ..telemetry.stats import BarsStats
from ._common import device_scope, lane_cpus, ref_obj


class QuoteBarsOp(hs.Operator):
    """Bars for the directory's current generation; other generations' batches are stale."""

    events = hs.Input("events", hs.pyobject)
    window = hs.Output("window", hs.pyobject)

    backend = hs.Param(
        "gpu", allowed=["gpu", "cpu"], description="gpu: GpuQuoteBars; cpu: QuoteBars"
    )
    bar_seconds = hs.Param(5, description="bar length; the five bar Params follow BarSettings")
    window_seconds = hs.Param(300)
    stale_seconds = hs.Param(30)
    min_bars = hs.Param(20)
    max_spread_bps = hs.Param(100.0)
    gpu_device = hs.Param(0, range=(0, 63), description="CUDA device; ignored by the cpu backend")
    warmup = hs.Param(True, description="gpu: compile and run the bar kernels in start()")
    emit_windows = hs.Param(True, description="declare the window output")
    snapshot_period_ms = hs.Param(250, range=(0, 60_000), description="0: no symbol snapshots")
    directory = hs.Param(Ref(), description="Ref(Knob[FeedDirectory]); empty: input is ignored")
    stats = hs.Param(Ref(), description="Ref(BarsStats); private if empty")

    def __init__(self) -> None:
        self._builder = self._knob = self._stats = None
        self._gpu = False
        self._snapshot_ns = self._snapshot_due = 0

    def setup(self, spec: hs.OperatorSpec) -> None:
        self._settings()  # the 4.3 QuoteBars rules: ValueError at graph.op
        spec.input(self.events)
        if self.emit_windows:
            spec.output(self.window)

    def contract(self) -> hs.Contract:
        return hs.Contract().trigger(hs.OnEach(self.events))

    def start(self) -> None:
        settings = self._settings()
        self._knob = self.directory.obj
        self._stats = ref_obj(self.stats, BarsStats)
        self._stats.lane_cpus = lane_cpus()
        self._gpu = self.backend == "gpu"
        if self._gpu:
            import cupy as cp

            cp.cuda.Device(self.gpu_device).use()
            if self.warmup:
                warm_bars_kernels(settings.as_params(), self.gpu_device)
        self._builder = BarBuilder(settings, backend=self.backend, gpu_device=self.gpu_device)
        self._snapshot_ns = self.snapshot_period_ms * 1_000_000
        self._snapshot_due = 0

    def compute(self, ctx: hs.ExecutionContext) -> None:
        batches = self.events.receive_data()
        # The lane thread's current device is 0; a cpu backend fed GPU batches converts them.
        on_device = self._gpu or (batches and isinstance(batches[0], GpuReplayBatch))
        with device_scope(self.gpu_device) if on_device else nullcontext():
            self._observe(batches)

    def stop(self) -> None:
        # On the lane thread after the last compute: finite runs read the end state from
        # these two objects, so they must not wait for the snapshot cadence.
        b = self._builder
        if b is None:
            return
        if b.engine is not None:
            with device_scope(self.gpu_device) if self._gpu else nullcontext():
                self._stats.counters = b.counters()
                self._stats.symbol_snapshot = b.symbol_snapshot()
        self._builder = None

    def _observe(self, batches: list) -> None:
        st, b = self._stats, self._builder
        directory = self._knob.get() if self._knob is not None else None
        # Every batch of a message is decoded under one directory. A message still stamped
        # with the builder's generation was in flight when the knob moved (a replay loop
        # releases the ring before the bars ran): build it, reset at the next generation's.
        if b.needs_reset(directory) and not (batches and batches[0].generation == b.generation):
            b.reset(directory)
            st.reset(directory.generation, directory.symbols, device=self._gpu)
            self._snapshot_due = 0
        window = b.observe(batches)
        st.stale_packets, st.sequence_gaps = b.stale_packets, b.sequence_gaps
        st.expected_sequence = b.expected
        if b.engine is not None:  # no directory yet: input is ignored and nothing to mirror
            st.counters = b.counters()
            now = time.monotonic_ns()
            if self._snapshot_ns and now >= self._snapshot_due:
                st.symbol_snapshot = b.symbol_snapshot()
                self._snapshot_due = now + self._snapshot_ns
        if window is not None and self.emit_windows:
            # The key first: a drained check must never match risk to the previous window
            # while this one is in flight.
            st.last_window_key = (window.generation, window.completed_bars)
            st.last_window_bars = window.completed_bars
            self.window.emit(window)
            st.windows_emitted += 1
        st.batches_in += 1  # last: "batches_in == event_batches" means fully processed

    def _settings(self) -> BarSettings:
        return BarSettings(
            bar_seconds=self.bar_seconds,
            window_seconds=self.window_seconds,
            stale_seconds=self.stale_seconds,
            min_bars=self.min_bars,
            max_spread_bps=self.max_spread_bps,
        )
