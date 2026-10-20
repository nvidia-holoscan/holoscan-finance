# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU rolling risk: volatility, covariance and correlation over a window of bar returns.

The ``window`` input takes a :class:`holoscan.marketdata.BarWindow` (device-resident
returns, bars x instruments). It holds one message and replaces it when a newer one
arrives, and the operator runs at most once per ``min_interval_s``, so it always
analyzes the latest window and never slows the producer. Each run updates a
:class:`holoscan.marketdata.covariance.CovarianceEngine` with cuBLAS products for the
added and expired bars and emits a :class:`RiskUpdate` on ``risk``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from holoscan.conditions import PeriodicCondition
from holoscan.core import (
    ExecutionContext,
    InputContext,
    IOSpec,
    Operator,
    OutputContext,
)

# Absolute import: in the source tree this operator is a package under operators/, in the
# built package a flat holoscan.marketdata module.
from holoscan.marketdata.covariance import CovarianceEngine, gpu_name


@dataclass
class RiskUpdate:
    window: object  # the analyzed BarWindow
    result: dict  # CovarianceEngine.update() output plus gpu_name and finished_wall_ns
    rows_host: np.ndarray | None = None  # host copy of the window when host_variance applied


class GpuRollingRiskOp(Operator):
    """``ready`` (callable) returning False leaves the latest window queued, so a
    consumer that must see every result, such as a CPU cross-check, can hold the
    operator back. ``host_variance`` (bool or callable) ranks and selects instruments
    by the NumPy variance of a host copy of the window, so a NumPy reference makes
    identical choices; the copy is attached to the update as ``rows_host``."""

    def __init__(
        self,
        fragment,
        *args,
        bar_seconds=5,
        gpu_device=0,
        min_interval_s=0.05,
        ready=None,
        host_variance=False,
        name="gpu_rolling_risk",
        **kwargs,
    ):
        if min_interval_s <= 0:
            raise ValueError("min_interval_s must be positive")
        self.bar_seconds, self.gpu_device = bar_seconds, int(gpu_device)
        self.ready, self.host_variance = ready, host_variance
        self._engine = self._generation = None
        self._interval = PeriodicCondition(
            fragment, recess_period=int(min_interval_s * 1e9), name=name + "_interval"
        )
        super().__init__(fragment, self._interval, *args, name=name, **kwargs)

    def setup(self, spec):
        spec.input("window", size=IOSpec.IOSize(1), policy=IOSpec.QueuePolicy.POP)
        spec.output("risk")

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):
        del context
        if self.ready is not None and not self.ready():
            return  # the latest window stays queued
        window = op_input.receive("window")
        if window is None:
            return
        import cupy as cp

        host = self.host_variance() if callable(self.host_variance) else self.host_variance
        with cp.cuda.Device(self.gpu_device):
            rows = window.rows
            rows_host = observed = None
            if host:
                rows_host = rows.get()
                observed = np.var(rows_host, axis=0, ddof=1)
            if self._engine is None or window.generation != self._generation:
                # First window of a generation: a throwaway engine compiles the kernels and
                # allocates workspaces, so the kept engine's timings exclude that cost.
                CovarianceEngine(
                    "gpu", bar_seconds=self.bar_seconds, gpu_device=self.gpu_device
                ).update(rows, window.times, observed_variance=observed)
                self._engine = CovarianceEngine(
                    "gpu", bar_seconds=self.bar_seconds, gpu_device=self.gpu_device
                )
                self._generation = window.generation
            result = self._engine.update(rows, window.times, observed_variance=observed)
            result["gpu_name"] = gpu_name(cp, self.gpu_device)
            result["finished_wall_ns"] = time.time_ns()
        op_output.emit(RiskUpdate(window, result, rows_host), "risk")


__all__ = ["GpuRollingRiskOp", "RiskUpdate"]
