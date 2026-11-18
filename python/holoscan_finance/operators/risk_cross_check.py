# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``RiskCrossCheckOp``: the CPU twin of every validated GPU risk window.

Ports: ``risk`` (in, validated ``RiskUpdate`` with ``rows_host``) -> ``comparison`` (out,
``RiskComparison``). Edges: in, ``lossless(2)`` from ``RollingRiskOp.validate`` (another
lane; the gate keeps one in flight); out, ``lossless(2)`` to the ``CallbackSinkOp`` that
publishes comparisons and releases the gate, same lane. Lane: the entry of ``crosscheck``.
It polls every ``poll_period_us`` (5 ms) for the spawned ``cpu_window`` worker, and with a
``directory`` has the worker tune its BLAS threads for each new universe as its manifest
arrives, or in ``start`` for a universe registered before the graph runs (replay, whose
drain starts with its source), so no GPU result waits for the tuning. Processor:
``analytics.crosscheck.CpuCrossCheck``.
"""

from __future__ import annotations

import holoscan as hs

from ..analytics.crosscheck import CpuCrossCheck
from ..ingest.affinity import normalize_cpu_list
from ..ref import Ref
from ..telemetry.stats import RiskStats
from ._common import PendingEmits, ref_obj


class RiskCrossCheckOp(hs.Operator):
    """Recompute validated windows on the CPU and emit the comparison with its GPU twin."""

    risk = hs.Input("risk", hs.pyobject)
    comparison = hs.Output("comparison", hs.pyobject)

    bar_seconds = hs.Param(5, range=(1, 3600))
    cpu_threads = hs.Param(0, range=(0, 1024), description="BLAS threads; 0 tunes")
    full_window_bars = hs.Param(
        60,
        range=(1, 100_000),
        description="the bars' full window (window_seconds // bar_seconds): engine capacity,"
        " and the window the BLAS tuning times",
    )
    poll_period_us = hs.Param(5000, range=(1000, 100_000))
    worker_cpus = hs.Param([], description="pin_worker set of the worker; [] inherits")
    directory = hs.Param(
        Ref(),
        description="Ref(Knob[FeedDirectory]): tune the worker for each new universe when its"
        " manifest arrives (in start() if it is known then); empty: at its first window",
    )
    stats = hs.Param(Ref(), description="Ref(RiskStats): crosschecks_* only; private if empty")

    def __init__(self) -> None:
        self._xc = self._pending = self._knob = None

    def setup(self, spec: hs.OperatorSpec) -> None:
        normalize_cpu_list(self.worker_cpus, name="worker_cpus")
        spec.input(self.risk)
        spec.output(self.comparison)

    def contract(self) -> hs.Contract:
        return (
            hs.Contract()
            .trigger(hs.OnClock(period=hs.time.us(self.poll_period_us)))
            .input(self.risk, hs.DrainAll())
        )

    def start(self) -> None:
        self._pending = PendingEmits()
        self._xc = CpuCrossCheck(
            bar_seconds=self.bar_seconds,
            cpu_threads=self.cpu_threads,
            full_window_bars=self.full_window_bars,
            worker_cpus=normalize_cpu_list(self.worker_cpus, name="worker_cpus"),
            stats=ref_obj(self.stats, RiskStats),
        )
        self._xc.start()
        self._knob = self.directory.obj
        if self._knob is not None and (d := self._knob.get()) is not None:
            # Replay registers its universe before the graph runs and starts its drain with
            # the source: tune now, before any window, as the GPU warm-ups run in start().
            self._xc.prepare(d.universe, wait=True)

    def compute(self, ctx: hs.ExecutionContext) -> None:
        for sample in self.risk.selected_batch():
            self._xc.submit(sample.data)
        if self._knob is not None and (d := self._knob.get()) is not None:
            self._xc.prepare(d.universe)  # once per universe, while the worker is idle
        if self._pending:
            self._pending.flush()
            return
        comparison = self._xc.poll()
        if comparison is not None:
            self._pending.add(self.comparison, comparison)
            self._pending.flush()

    def stop(self) -> None:
        if self._xc is not None:
            self._xc.stop()
            self._xc = None
