# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The operator catalog's full example compositions wire every port they declare.

An operator README's ``## Example`` is what a user copies. An output that it declares but
leaves unwired still compiles cleanly, and then the first emit fails the run with
``NotReadyError``: for ``QuoteBarsOp`` that is the first window, about 100 s of market time
in. So each full composition is compiled here and checked with the apps' own wiring rules.
The other READMEs show fragments that extend an existing graph.
"""

from __future__ import annotations

import os
import re
from types import SimpleNamespace

import holoscan as hs
import pytest
from apphelpers import REPO

from holoscan_finance.app.drained import DrainedCondition
from holoscan_finance.app.plan import check_plan, inputs_connected, outputs_connected
from holoscan_finance.ref import Knob
from holoscan_finance.telemetry.stats import BarsStats, IngestStats

pytestmark = pytest.mark.sdk

CATALOG = REPO / "operators"
COMPOSITIONS = (
    "daqiri_packet_source_op",
    "gpu_sbe_replay_source_op",
    "gpu_sbe_source_op",
    "sbe_decode_op",
    "quote_bars_op",
    "udp_packet_source_op",
)


def example(name: str) -> str:
    """The Python block under ``## Example`` in ``operators/<name>/README.md``."""
    section = (CATALOG / name / "README.md").read_text().split("## Example", 1)[1]
    return re.search(r"```python\n(.*?)```", section, re.S).group(1)


@pytest.mark.parametrize("name", COMPOSITIONS)
def test_a_catalog_example_wires_every_port(name):
    mask = frozenset(os.sched_getaffinity(0))
    ingest, bars = IngestStats(), BarsStats()
    g = hs.Graph(name, default_clock=hs.RealtimeClock)
    # The objects an application context provides (EquityContext, BenchContext).
    context = dict(
        g=g,
        cpus=SimpleNamespace(all=mask, helper=mask),
        knob=Knob(),
        directory_knob=Knob(),
        ingest_stats=ingest,
        bars_stats=bars,
        replay_generations=None,
        drained=DrainedCondition(ingest=ingest, bars=[bars]),
        collected=[],
    )
    exec(compile(example(name), f"operators/{name}/README.md", "exec"), context)
    plan = hs.compile(g)
    assert plan.ok, plan.summary()
    assert check_plan(plan, [outputs_connected(), inputs_connected()]) == []
