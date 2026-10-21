# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``RiskSummaryOp``: compact JSON within ``max_bytes``, counted summaries, and emit errors
(a stalled export, an unconnected output) dropped and counted instead of failing the run."""

from __future__ import annotations

import json
import time

import holoscan as hs
import numpy as np
import pytest
from holoscan.testing import OperatorTestHarness

from holoscan_finance.marketdata.covariance import CovarianceEngine
from holoscan_finance.operators import RiskSummaryOp
from holoscan_finance.operators.risk_summary import risk_summary
from holoscan_finance.payloads import BarWindow, RiskUpdate
from holoscan_finance.ref import Ref
from holoscan_finance.telemetry.stats import RiskStats

pytestmark = pytest.mark.sdk
pytest.importorskip("scipy")

KEYS = ["g", "t", "n", "vol", "corr", "ms", "bars", "r2r"]


def update(generation: int = 1_760_000_000_000_000_000, bars: int = 61) -> RiskUpdate:
    rows = np.random.default_rng(bars).normal(0, 0.001, (60, 30))
    times = np.arange(60, dtype=np.int64) * 5_000_000_000 + 1_760_000_000_000_000_000
    window = BarWindow(tuple(rows), times, generation, bars, time.time_ns() - 3_000_000)
    return RiskUpdate(window, CovarianceEngine().update(rows, times), backend="gpu")


def test_summaries_are_compact_json_of_the_update():
    stats, updates = RiskStats(), [update(bars=b) for b in (61, 62, 63)]
    with OperatorTestHarness(RiskSummaryOp, params={"stats": Ref(stats)}) as h:
        h.push_many("risk", updates)
        result = h.run()
    texts = [s.data for s in result.values("summary")]
    assert (stats.summaries, stats.summary_drops) == (3, 0)
    for text, u in zip(texts, updates):
        s = json.loads(text)
        assert list(s) == KEYS and len(text.encode()) <= 480
        assert (s["g"], s["t"], s["bars"]) == (
            u.window.generation,
            int(u.window.times[-1]),
            u.window.completed_bars,
        )
        assert s["n"] == u.result["eligible"] and s["corr"] == u.result["mean_correlation"]
        assert (
            s["vol"] == float(np.median(u.result["annual_vol"]))
            and s["ms"] == u.result["milliseconds"]
        )
        assert 3.0 <= s["r2r"] < 60_000


def test_trailing_keys_are_dropped_to_fit_max_bytes():
    text = risk_summary(update(), max_bytes=64)
    s = json.loads(text)
    assert len(text.encode()) <= 64 and list(s) == KEYS[: len(s)] and {"g", "t"} <= set(s)
    with OperatorTestHarness(RiskSummaryOp, params={"max_bytes": 64}) as h:
        h.push("risk", update())
        assert len(h.run().value("summary").data.encode()) <= 64


def test_an_unconnected_export_drops_and_counts():
    stats = RiskStats()
    with OperatorTestHarness(RiskSummaryOp, params={"stats": Ref(stats)}, outputs=[]) as h:
        h.push_many("risk", [update(), update()])
        result = h.run()
    assert result.ok and (stats.summaries, stats.summary_drops) == (0, 2)


class _StalledExport:
    """Stands in for the export port: backpressured, then not ready, then accepting."""

    def __init__(self) -> None:
        self.errors = [hs.errors.BackpressuredError("held"), hs.errors.NotReadyError("detached")]
        self.sent: list = []

    def emit(self, item: object) -> None:
        if self.errors:
            raise self.errors.pop(0)
        self.sent.append(item)


class StalledSummaryOp(RiskSummaryOp):
    """The real drop-and-count path, against a port that refuses twice."""

    export = hs.Param(Ref())

    def compute(self, ctx: hs.ExecutionContext) -> None:
        self._send(self.export.obj, risk_summary(self.risk.receive_data(), self.max_bytes))


def test_emit_errors_are_counted_and_never_fail_the_run():
    stats, export = RiskStats(), _StalledExport()
    params = {"stats": Ref(stats), "export": Ref(export)}
    with OperatorTestHarness(StalledSummaryOp, params=params, outputs=[]) as h:
        h.push_many("risk", [update(), update(), update()])
        result = h.run()
    assert result.ok and result.termination is hs.RunTermination.COMPLETED
    assert (stats.summary_drops, stats.summaries, len(export.sent)) == (2, 1, 1)
