# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU market-data operators and the modules they use.

Operators: :class:`GpuMitchSourceOp` (NIC to GPU memory, MITCH decode),
:class:`GpuQuoteBarsOp` (event-time quote bars on the GPU) and
:class:`GpuRollingRiskOp` (rolling volatility, covariance and correlation on the GPU).
Modules: the MITCH codec (``mitch``), GPU MITCH kernels (``gpu_mitch``), DAQIRI helpers
(``daqiri``), observation records (``wire``), CPU and GPU quote bars (``bars``,
``gpu_bars``) and the covariance engine (``covariance``).

Each operator lives under ``operators/<name>/`` at the repository root (HoloHub
per-operator layout) and resolves as a submodule of this package; CMake copies it
flat into the built package. Operators are re-exported lazily so the modules import
without the Holoscan SDK.
"""

from __future__ import annotations

import os as _os
from importlib import import_module
from typing import TYPE_CHECKING, Any

# Source tree: <repo>/operators holds the per-operator packages. In the built package
# the operator modules sit in this directory and the path does not exist.
_ops_dir = _os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..", "operators"
)
if _os.path.isdir(_ops_dir) and _os.path.abspath(_ops_dir) not in __path__:
    __path__.append(_os.path.abspath(_ops_dir))

if TYPE_CHECKING:
    from .gpu_mitch_source_op import GpuMitchSourceOp
    from .gpu_quote_bars_op import BarWindow, GpuQuoteBarsOp
    from .gpu_rolling_risk_op import GpuRollingRiskOp, RiskUpdate

_LAZY_OPERATOR_EXPORTS = {
    "GpuMitchSourceOp": (".gpu_mitch_source_op", "GpuMitchSourceOp"),
    "GpuQuoteBarsOp": (".gpu_quote_bars_op", "GpuQuoteBarsOp"),
    "BarWindow": (".gpu_quote_bars_op", "BarWindow"),
    "GpuRollingRiskOp": (".gpu_rolling_risk_op", "GpuRollingRiskOp"),
    "RiskUpdate": (".gpu_rolling_risk_op", "RiskUpdate"),
}


def __getattr__(name: str) -> Any:
    """Load Holoscan-dependent operator classes only when requested."""
    target = _LAZY_OPERATOR_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "BarWindow",
    "GpuMitchSourceOp",
    "GpuQuoteBarsOp",
    "GpuRollingRiskOp",
    "RiskUpdate",
]
