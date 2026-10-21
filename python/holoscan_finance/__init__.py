# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Streaming NYQ equity analytics on Holoscan SDK 5, exposed as ``holoscan.finance``.

Importing the package imports nothing else: ``Ref``/``Knob``, the payload classes and the
ten operator classes resolve on first access, so the CPU tools and the library modules run
without the Holoscan SDK or CuPy. Inside the package, modules import each other relatively:
the ``holoscan.finance`` alias must not be imported while this package initializes.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__version__ = "0.2.0"

_PAYLOADS = (
    "PacketBatch",
    "BarWindow",
    "RiskUpdate",
    "RiskComparison",
    "BarCounters",
    "SymbolSnapshot",
    "ReplayBatch",
    "GpuReplayBatch",
)
_OPERATORS = (
    "GpuMitchSourceOp",
    "GpuMitchReplaySourceOp",
    "UdpPacketSourceOp",
    "DaqiriPacketSourceOp",
    "MitchDecodeOp",
    "QuoteBarsOp",
    "RollingRiskOp",
    "RiskCrossCheckOp",
    "CallbackSinkOp",
    "RiskSummaryOp",
)
_LAZY = {
    "Ref": ".ref",
    "Knob": ".ref",
    **dict.fromkeys(_PAYLOADS, ".payloads"),
    **dict.fromkeys(_OPERATORS, ".operators"),
}


def __getattr__(name: str) -> Any:
    """Import the defining module on first access; operator classes import Holoscan."""
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))


__all__ = sorted(_LAZY)
