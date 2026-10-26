# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Application test helpers: load an app by file path, parse its flags, build dry contexts.

The example applications are scripts, not package modules, so ``load_app`` imports
``examples/<name>/<name>.py`` by path. ``dry_equity_context`` holds the objects that
``equity_analytics.compose()`` and ``rules()`` read, with no I/O, no GPU and no control
plane, so the plan tests compile every mode and option on the CPU-only tier; their CPU
sets come from ``runner.cpu_sets``, which leaves the test process as it is. ``child_env``
is the environment of a subprocess that imports the checkout.
"""

from __future__ import annotations

import argparse
import functools
import importlib.util
import os
from pathlib import Path
from types import ModuleType, SimpleNamespace

from holoscan_finance.app import cli
from holoscan_finance.app.drained import DrainedCondition
from holoscan_finance.app.runner import cpu_sets
from holoscan_finance.control.gate import CrossCheckGate
from holoscan_finance.ref import Knob
from holoscan_finance.telemetry.capture import CaptureTap
from holoscan_finance.telemetry.stats import BarsStats, IngestStats, RiskStats

REPO = Path(__file__).resolve().parents[2]


@functools.cache
def load_app(name: str) -> ModuleType:
    """``examples/<name>/<name>.py`` as a module named ``app_<name>`` (loaded once)."""
    path = REPO / "examples" / name / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"app_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def child_env(*paths: str | Path) -> dict[str, str]:
    """This environment, with the checkout (and ``paths``) appended to ``PYTHONPATH``: the
    image's own entry, which holds the SDK, stays first."""
    entries = [os.environ.get("PYTHONPATH", ""), REPO / "python", REPO / "tests" / "python", *paths]
    return dict(os.environ, PYTHONPATH=os.pathsep.join(str(e) for e in entries if e))


def equity_args(*argv: str) -> argparse.Namespace:
    """Parse and normalize ``equity_analytics`` flags as its ``main()`` does."""
    return cli.normalize(cli.equity_analytics_parser().parse_args(list(argv)))


def dry_equity_context(args: argparse.Namespace) -> SimpleNamespace:
    """What ``compose()`` and ``rules()`` read from an ``EquityContext``; the results store
    is two no-op callbacks and there is no replay registration."""
    ingest, bars = IngestStats(), BarsStats()
    return SimpleNamespace(
        args=args,
        cpus=cpu_sets(args),
        settings=cli.bar_settings(args),
        directory_knob=Knob(),
        ingest_stats=ingest,
        bars_stats=bars,
        risk_stats=RiskStats(),
        tap=CaptureTap(),
        gate=CrossCheckGate(),
        results=SimpleNamespace(on_risk=_ignore, on_comparison=_ignore),
        replay_generations=None,
        drained=DrainedCondition(ingest=ingest, bars=[bars]),
    )


def _ignore(item: object) -> None:
    pass
