# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``RiskSummaryOp``: a compact JSON risk summary for other processes over holomq (``--topics``).

Ports: ``risk`` (in, ``RiskUpdate``) -> ``summary`` (out, ``holoscan.schema.String``, at most
``max_bytes``: a ``String`` over 512 B makes ``emit`` raise, sc-F7). Edges: in, ``LATEST``
from ``RollingRiskOp.risk`` (a fan-out beside the publish edge); out, ``LATEST`` into
``export_topic("holomq://finance.risk", ..., **edges.topic_export_kwargs())``. Lane:
``topics``, of which it is the entry. Off by default and absent from measured runs.
The one operator that drops instead of holding: a summary is latest-value and its reader is
an external process the graph does not control, so an emit error is counted in
``RiskStats.summary_drops`` and never fails the run (a ``BLOCKING`` export failed a run with
``HOLOMQ_BACKPRESSURED`` after 17 emits once a reader stopped: c5).
"""

from __future__ import annotations

import json
import time

import holoscan as hs
import numpy as np
from holoscan.schema import String

from ..payloads import RiskUpdate
from ..ref import Ref
from ..telemetry.stats import RiskStats
from ._common import ref_obj

_ALWAYS_KEPT = 2  # "g" and "t" are never dropped


def risk_summary(update: RiskUpdate, max_bytes: int = 480) -> str:
    """Compact JSON: generation, bar time, eligible names, median volatility, mean
    correlation, engine ms, completed bars and receive-to-result ms; trailing keys are
    dropped until the encoding fits ``max_bytes``."""
    window, result = update.window, update.result
    vol = result["annual_vol"]
    summary = {
        "g": window.generation,
        "t": int(window.times[-1]),
        "n": result["eligible"],
        "vol": float(np.median(vol)) if vol else None,
        "corr": result["mean_correlation"],
        "ms": result["milliseconds"],
        "bars": window.completed_bars,
        "r2r": (time.time_ns() - window.recv_ns) / 1e6,
    }
    keys = list(summary)
    text = json.dumps(summary, separators=(",", ":"))
    while len(text.encode()) > max_bytes and len(keys) > _ALWAYS_KEPT:
        del summary[keys.pop()]
        text = json.dumps(summary, separators=(",", ":"))
    return text


class RiskSummaryOp(hs.Operator):
    """One summary per ``RiskUpdate``; dropped and counted when the export cannot take it."""

    risk = hs.Input("risk", hs.pyobject)
    summary = hs.Output("summary", hs.payload.schema.String)

    max_bytes = hs.Param(480, range=(64, 512), description="encoded size bound (String: 512 B)")
    stats = hs.Param(
        Ref(), description="Ref(RiskStats): summaries, summary_drops; private if empty"
    )

    def __init__(self) -> None:
        self._stats = None

    def contract(self) -> hs.Contract:
        return hs.Contract().trigger(hs.OnEach(self.risk))

    def start(self) -> None:
        self._stats = ref_obj(self.stats, RiskStats)

    def compute(self, ctx: hs.ExecutionContext) -> None:
        self._send(self.summary, risk_summary(self.risk.receive_data(), self.max_bytes))

    def _send(self, port: hs.OperatorPort, text: str) -> None:
        # The documented exception to hold-and-retry: at <= 20 Hz the 17 us of a caught emit
        # is irrelevant, and a newer summary replaces a dropped one anyway.
        try:
            port.emit(String(data=text))
        except (hs.errors.BackpressuredError, hs.errors.NotReadyError):
            self._stats.summary_drops += 1
        else:
            self._stats.summaries += 1
