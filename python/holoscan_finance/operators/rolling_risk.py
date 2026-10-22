# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``RollingRiskOp``: rolling volatility, covariance and correlation of the newest bar window.

Ports: ``window`` (in, ``BarWindow``) -> ``risk`` (out, ``RiskUpdate``) and, only with
``cross_check``, ``validate`` (out: the validated updates, with ``rows_host``). Edges: in,
``LATEST`` from ``QuoteBarsOp`` in another lane; ``risk`` -> ``CallbackSinkOp``
``lossless(8)``, same lane; ``validate`` -> ``RiskCrossCheckOp`` ``lossless(2)``. Lane: the
entry of ``analytics``. It polls every ``poll_period_us`` (250 µs) and keeps the newest
window in a slot, analyzing it as soon as ``min_interval_us`` has passed: run-on-arrival
timing at a bounded rate. ``executor="process"`` runs the analysis in
a spawned worker (host front ends: off the decoder's GIL). Processor:
``analytics.rolling.RollingRisk``.
"""

from __future__ import annotations

import holoscan as hs

from ..analytics.rolling import RollingRisk
from ..ingest.affinity import normalize_cpu_list
from ..ref import Ref
from ..telemetry.stats import RiskStats
from ._common import PendingEmits, lane_cpus, ref_obj


class RollingRiskOp(hs.Operator):
    """Risk of the newest window at a bounded rate; outputs held and retried on backpressure."""

    window = hs.Input("window", hs.pyobject)
    risk = hs.Output("risk", hs.pyobject)
    validate = hs.Output("validate", hs.pyobject)

    backend = hs.Param("gpu", allowed=["gpu", "cpu"], description="covariance engine backend")
    executor = hs.Param(
        "inline",
        allowed=["inline", "process"],
        description="process: one spawned worker; cpu+inline is for tests (SciPy holds the GIL)",
    )
    bar_seconds = hs.Param(5, range=(1, 3600), description="annualizes the volatility")
    gpu_device = hs.Param(0, range=(0, 63))
    min_interval_us = hs.Param(
        50_000,
        range=(1000, 60_000_000),
        description="--interval: at most one analysis per interval",
    )
    poll_period_us = hs.Param(
        250, range=(100, 100_000), description="--risk-poll-us: how often to look for a window"
    )
    host_variance = hs.Param(False, description="inline: rank by host NumPy variance")
    cross_check = hs.Param(False, description="claim the gate and emit validated updates")
    gate = hs.Param(Ref(), description="Ref(CrossCheckGate); required with cross_check")
    cpu_threads = hs.Param(0, range=(0, 1024), description="cpu_window BLAS threads; 0 tunes")
    full_window_bars = hs.Param(
        60,
        range=(1, 100_000),
        description="the bars' full window (window_seconds // bar_seconds): engine capacity,"
        " and the window the CPU worker's BLAS tuning times",
    )
    worker_cpus = hs.Param([], description="pin_worker set of the process executor; [] inherits")
    directory = hs.Param(
        Ref(),
        description="Ref(Knob[FeedDirectory]); process + cpu: tune the worker for each new"
        " universe when its manifest arrives (in start() if it is known then); empty: at its"
        " first window",
    )
    deadline_us = hs.Param(
        0, range=(0, 10_000_000), description=">0: count poll ticks that started this late"
    )
    stats = hs.Param(Ref(), description="Ref(RiskStats); private if empty")

    def __init__(self) -> None:
        self._risk = self._pending = self._stats = self._knob = None
        self._count_late = False

    def setup(self, spec: hs.OperatorSpec) -> None:
        normalize_cpu_list(self.worker_cpus, name="worker_cpus")
        if self.cross_check and (not self.gate or self.backend != "gpu"):
            raise ValueError("cross_check requires a gate Ref(CrossCheckGate) and backend='gpu'")
        if self.host_variance and self.executor != "inline":
            raise ValueError("host_variance requires executor='inline'")
        spec.input(self.window)
        spec.output(self.risk)
        if self.cross_check:
            spec.output(self.validate)

    def contract(self) -> hs.Contract:
        contract = (
            hs.Contract()
            .trigger(hs.OnClock(period=hs.time.us(self.poll_period_us)))
            .input(self.window, hs.DrainAll())
        )
        if self.deadline_us > 0:
            contract = contract.max_start_delay(hs.time.us(self.deadline_us)).on_violation(
                hs.OnViolation.EMIT_WITNESS
            )
        return contract

    def start(self) -> None:
        self._stats = ref_obj(self.stats, RiskStats)
        self._stats.lane_cpus = lane_cpus()
        self._pending = PendingEmits()
        self._count_late = self.deadline_us > 0
        self._risk = RollingRisk(
            backend=self.backend,
            executor=self.executor,
            bar_seconds=self.bar_seconds,
            gpu_device=self.gpu_device,
            min_interval_ns=self.min_interval_us * 1000,
            cross_check=self.cross_check,
            gate=self.gate.obj,
            host_variance=self.host_variance,
            cpu_threads=self.cpu_threads,
            full_window_bars=self.full_window_bars,
            worker_cpus=normalize_cpu_list(self.worker_cpus, name="worker_cpus"),
            stats=self._stats,
        )
        self._risk.start()  # process: the worker is running before the first window
        cpu_worker = self.backend == "cpu" and self.executor == "process"
        self._knob = self.directory.obj if cpu_worker else None
        if self._knob is not None and (d := self._knob.get()) is not None:
            # Replay registers its universe before the graph runs: tune before any window.
            self._risk.prepare(d.universe, wait=True)

    def compute(self, ctx: hs.ExecutionContext) -> None:
        # Into the slot first: DrainAll retires samples this activation does not read.
        for sample in self.window.selected_batch():
            self._risk.offer(sample.data)
        if self._knob is not None and (d := self._knob.get()) is not None:
            self._risk.prepare(d.universe)  # once per universe, while the worker is idle
        if (
            self._count_late
            and ctx.activation.reason == hs.ActivationReason.MAX_START_DELAY_EXCEEDED
        ):
            self._stats.late_activations += 1
        if self._pending:
            self._pending.flush()
            return
        update = self._risk.poll()
        if update is not None:
            self._pending.add(self.risk, update)
            if update.validated:
                self._pending.add(self.validate, update)
            self._pending.flush()

    def stop(self) -> None:
        if self._risk is not None:
            self._risk.stop()
            self._risk = None
