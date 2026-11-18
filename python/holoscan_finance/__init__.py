# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Holoscan Module for Financial Services Industry, exposed as ``holoscan.finance``.

A self-contained Holoscan SDK 5 module for building streaming financial-services
applications: reusable operators for market-data ingest, decoding, quote bars and rolling
risk, the libraries behind them, and CPU-only tools. The example applications composed from
the operators live under ``examples/`` in the repository.

Importing the package imports nothing else: ``Ref``/``Knob``, the payload classes and the
ten operator classes resolve on first access, so the CPU tools and the library modules run
without the Holoscan SDK or CuPy. Inside the package, modules import each other relatively:
the ``holoscan.finance`` alias must not be imported while this package initializes.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__version__ = "1.0.0"

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
    "GpuSbeSourceOp",
    "GpuSbeReplaySourceOp",
    "UdpPacketSourceOp",
    "DaqiriPacketSourceOp",
    "SbeDecodeOp",
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
