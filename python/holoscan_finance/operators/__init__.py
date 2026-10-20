# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ten Holoscan 5 operators, each a thin adapter around a library processor.

Sources: ``GpuMitchSourceOp``, ``GpuMitchReplaySourceOp``, ``UdpPacketSourceOp``,
``DaqiriPacketSourceOp``. Processing: ``MitchDecodeOp``, ``QuoteBarsOp``, ``RollingRiskOp``,
``RiskCrossCheckOp``. Sink: ``CallbackSinkOp``. holomq: ``RiskSummaryOp``. Their catalog
entries are ``operators/<snake>_op/`` at the repository root. The classes load on first
access, so importing this package imports no Holoscan; ``edges`` holds the edge presets.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_MODULES = {
    "GpuMitchSourceOp": ".gpu_mitch_source",
    "GpuMitchReplaySourceOp": ".gpu_mitch_source",
    "UdpPacketSourceOp": ".packet_sources",
    "DaqiriPacketSourceOp": ".packet_sources",
    "MitchDecodeOp": ".mitch_decode",
    "QuoteBarsOp": ".quote_bars",
    "RollingRiskOp": ".rolling_risk",
    "RiskCrossCheckOp": ".risk_cross_check",
    "CallbackSinkOp": ".callback_sink",
    "RiskSummaryOp": ".risk_summary",
}

__all__ = sorted(_MODULES)


def __getattr__(name: str) -> Any:
    module_name = _MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_MODULES))
