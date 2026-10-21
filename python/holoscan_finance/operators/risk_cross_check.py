# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``RiskCrossCheckOp``: the CPU twin of every validated GPU risk window.

Ports: ``risk`` (in, validated ``RiskUpdate`` with ``rows_host``) -> ``comparison`` (out,
``RiskComparison``). Edges: in, ``lossless(2)`` from ``RollingRiskOp.validate`` (another
lane; the gate keeps one in flight); out, ``lossless(2)`` to the ``CallbackSinkOp`` that
publishes comparisons and releases the gate, same lane. Lane: the entry of ``crosscheck``.
It polls every ``poll_period_us`` (5 ms, 4.3 ``PublishOp``) for the spawned ``cpu_window``
worker. Processor: ``analytics.crosscheck.CpuCrossCheck``.
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
    full_window_bars = hs.Param(60, range=(1, 100_000), description="cpu_window re-tunes here")
    poll_period_us = hs.Param(5000, range=(1000, 100_000))
    worker_cpus = hs.Param([], description="pin_worker set of the worker; [] inherits")
    stats = hs.Param(Ref(), description="Ref(RiskStats): crosschecks_* only; private if empty")

    def __init__(self) -> None:
        self._xc = self._pending = None

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

    def compute(self, ctx: hs.ExecutionContext) -> None:
        for sample in self.risk.selected_batch():
            self._xc.submit(sample.data)
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
