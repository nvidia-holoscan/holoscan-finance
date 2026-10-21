# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``--topics``: an external reader gets the risk summaries, and a stalled one costs the graph
nothing (the c5 regression, with the production pieces).

A 20 Hz test-local ``RiskUpdate`` source feeds the real ``RiskSummaryOp`` (lane ``topics``),
which exports to ``holomq://finance.risk`` over ``LATEST`` with ``topic_export_kwargs()``,
exactly as ``equity_analytics`` wires it; ``runner.compile_plan`` writes the endpoint
catalog into a directory that does not exist yet. A child process subscribes with a 10 s
timeout, first reading, then attaching and not reading for 5 s. With the former ``BLOCKING``
export edge, the stalled reader failed the run with ``HOLOMQ_BACKPRESSURED`` (c5).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import holoscan as hs
import numpy as np
import pytest

from holoscan_finance.app import plan as rules
from holoscan_finance.app import runner
from holoscan_finance.app.placement import Lanes
from holoscan_finance.ingest.affinity import CpuSets
from holoscan_finance.operators import RiskSummaryOp
from holoscan_finance.operators.edges import LATEST, topic_export_kwargs
from holoscan_finance.payloads import BarWindow, RiskUpdate
from holoscan_finance.ref import Ref
from holoscan_finance.telemetry.stats import RiskStats

pytestmark = [pytest.mark.sdk, pytest.mark.slow]

SUBSCRIBER = textwrap.dedent(
    """
    import json, sys, time

    import holoscan as hs
    from holoscan import channel

    catalog, mode, seconds = sys.argv[1], sys.argv[2], float(sys.argv[3])
    got, last = 0, None
    with channel.Runtime() as runtime:
        sub = runtime.subscriber(
            hs.topics.EndpointCatalog.load(catalog), "finance.risk", timeout=hs.time.ms(10_000)
        )
        end = time.time() + seconds
        while time.time() < end:
            if mode == "read":
                try:
                    sample = sub.try_receive()
                except hs.errors.ClosedError:
                    break
                if sample is not None:
                    got, last = got + 1, sample.data.data
            time.sleep(0.01 if mode == "read" else 0.2)  # "stall": attached, never reading
    print(json.dumps({"got": got, "last": last}))
    """
)


class Risk(hs.Operator):
    """A ``RollingRiskOp`` stand-in: one small ``RiskUpdate`` every 50 ms."""

    risk = hs.Output("risk", hs.pyobject)

    def contract(self) -> hs.Contract:
        return hs.Contract().trigger(hs.OnClock(period=hs.time.ms(50)))

    def start(self) -> None:
        self.bars = 0

    def compute(self, ctx: hs.ExecutionContext) -> None:
        self.bars += 1
        window = BarWindow(
            parts=(np.zeros(2),),
            times=np.array([self.bars * 5_000_000_000], dtype=np.int64),
            generation=1,
            completed_bars=self.bars,
            recv_ns=time.time_ns(),
        )
        result = {
            "annual_vol": [0.2, 0.3],
            "eligible": 2,
            "mean_correlation": 0.5,
            "milliseconds": 1.0,
        }
        self.risk.emit(RiskUpdate(window=window, result=result))


def subscribe(catalog: Path, mode: str, seconds: float) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-c", SUBSCRIBER, str(catalog), mode, str(seconds)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=dict(os.environ),
    )


def result_of(child: subprocess.Popen) -> dict:
    out, err = child.communicate(timeout=60)
    lines = [line for line in out.splitlines() if line.startswith("{")]
    assert child.returncode == 0 and lines, err[-3000:]
    return json.loads(lines[-1])


def test_a_stalled_reader_never_blocks_the_graph(tmp_path: Path) -> None:
    stats = RiskStats()
    g = hs.Graph("topics", default_clock=hs.RealtimeClock)
    risk = g.op(Risk, "risk")
    summary = g.op(RiskSummaryOp, "risk_summary", stats=Ref(stats))
    g.add_flow(risk.risk, summary.risk, **LATEST)
    topic = g.export_topic(
        "holomq://finance.risk",
        hs.payload.schema.String,
        endpoint_name="finance.risk",
        **topic_export_kwargs(),
    )
    g.add_flow(summary.summary, topic.input, **LATEST)
    lanes = Lanes(CpuSets.compute()).add("analytics", risk).add("topics", summary)
    lanes.apply(g)
    catalog = tmp_path / "missing" / "dir" / "endpoints.json"
    plan = runner.compile_plan(
        g,
        lanes,
        rules=[rules.edge("risk_summary.summary", "*", policy="DROP_OLDEST", depth=1)],
        topics_catalog=str(catalog),
    )
    assert catalog.is_file()

    with hs.run_async(plan) as session:
        reader = result_of(subscribe(catalog, "read", 3))
        assert reader["got"] > 10  # 20 Hz for ~3 s
        assert set(json.loads(reader["last"])) >= {"g", "t", "n", "vol", "corr", "ms", "bars"}

        staller = subscribe(catalog, "stall", 5)
        assert not session.wait(6), f"the graph ended while a reader stalled: {session.summary()}"
        result_of(staller)
        session.request_stop()
        assert session.wait(10)

    assert session.termination == hs.RunTermination.STOPPED
    assert runner.report_diagnostics(session) == []
    assert stats.summaries > 100  # it kept summarizing through the stall
