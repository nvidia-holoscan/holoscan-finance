# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``holoscan.finance`` is the canonical ``holoscan_finance``, not a copy of it.

With the two-file dist-info of ``devpath`` on ``PYTHONPATH`` (what the wheel and the CMake
staging install), Holoscan's module discovery aliases the package: the same module objects,
the same operator classes, and plan type names that keep the canonical module path. It runs
in a subprocess because ``import holoscan`` reads the entry points once per process.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from apphelpers import REPO, child_env

from holoscan_finance.devpath import write_dist_info

pytestmark = pytest.mark.sdk

CHECK = textwrap.dedent(
    """
    import json

    import holoscan as hs
    import holoscan.finance as f
    import holoscan_finance as g
    from holoscan.finance.operators import QuoteBarsOp

    import holoscan_finance.operators

    modules = {name: getattr(f.operators, name).__module__ for name in f.operators.__all__}
    graph = hs.Graph("alias", default_clock=hs.RealtimeClock)
    bars = graph.op(QuoteBarsOp, "bars", backend="cpu")
    print(json.dumps(dict(
        same=f is g and f.operators is holoscan_finance.operators,
        same_class=QuoteBarsOp is holoscan_finance.operators.QuoteBarsOp,
        file=g.__file__, modules=modules, type_name=bars.type_name, version=f.__version__,
    )))
    """
)


def test_holoscan_finance_alias_is_the_canonical_package(tmp_path: Path) -> None:
    env = child_env(write_dist_info(tmp_path / "dist"))
    done = subprocess.run(
        [sys.executable, "-c", CHECK], capture_output=True, text=True, timeout=120, env=env
    )
    lines = [line for line in done.stdout.splitlines() if line.startswith("{")]
    assert done.returncode == 0 and lines, done.stderr[-3000:]
    out = json.loads(lines[-1])
    assert out["same"] and out["same_class"]
    assert Path(out["file"]).resolve() == REPO / "python" / "holoscan_finance" / "__init__.py"
    assert out["version"] == "0.2.0"
    assert out["modules"] == {
        "CallbackSinkOp": "holoscan_finance.operators.callback_sink",
        "DaqiriPacketSourceOp": "holoscan_finance.operators.packet_sources",
        "GpuSbeReplaySourceOp": "holoscan_finance.operators.gpu_sbe_source",
        "GpuSbeSourceOp": "holoscan_finance.operators.gpu_sbe_source",
        "SbeDecodeOp": "holoscan_finance.operators.sbe_decode",
        "QuoteBarsOp": "holoscan_finance.operators.quote_bars",
        "RiskCrossCheckOp": "holoscan_finance.operators.risk_cross_check",
        "RiskSummaryOp": "holoscan_finance.operators.risk_summary",
        "RollingRiskOp": "holoscan_finance.operators.rolling_risk",
        "UdpPacketSourceOp": "holoscan_finance.operators.packet_sources",
    }
    assert out["type_name"] == "holoscan_finance.operators.quote_bars.QuoteBarsOp"
