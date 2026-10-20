# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bar settings, ingest defaults and ingest mode names: one copy of each 4.3 default.

``BarSettings`` replaces the 4.3 copies of the bar defaults (``EquityState``,
``GpuQuoteBarsOp``, ``decode_bench``, ``gpu_validate``) and validates with the
``QuoteBars`` rules. ``IngestDefaults`` are the measured 4.3 GPUDirect settings: lossless at
200k pps with 131072-message batches, a ``1 << 20`` ring and 114688 NIC buffers
(GPU_INGEST). The canonical ingest names replace ``async|ano|gpu``, which stay accepted as
aliases; ``/snapshot`` keeps reporting the legacy tokens for the 4.3 tools.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from .marketdata.bars import QuoteBars


@dataclass(frozen=True)
class BarSettings:
    """The single source of the 4.3 bar defaults."""

    bar_seconds: int = 5
    window_seconds: int = 300
    stale_seconds: int = 30
    min_bars: int = 20
    max_spread_bps: float = 100.0

    def __post_init__(self) -> None:
        QuoteBars([], **self.as_params())  # the 4.3 rules: ValueError on bad values

    def as_params(self) -> dict[str, int | float]:
        """Keyword arguments of ``QuoteBars``/``GpuQuoteBars`` (and ``/snapshot.settings``)."""
        return dataclasses.asdict(self)

    @property
    def window_bars(self) -> int:
        return self.window_seconds // self.bar_seconds


@dataclass(frozen=True)
class IngestDefaults:
    gpu_batch_messages: int = 131072
    host_batch_messages: int = 16384
    gpu_ring_size: int = 1 << 20
    host_ring_size: int = 65536
    gpu_num_bufs: int = 114688
    interval_s: float = 0.05
    burst_packets: int = 4096
    rx_timeout_us: int = 20000


INGEST_MODES = ("udp", "daqiri-host", "gpudirect", "replay")
INGEST_ALIASES = {"async": "udp", "ano": "daqiri-host", "gpu": "gpudirect"}
# /snapshot network.ingest: measure.py, ano_runtime.wait and index.html read these tokens.
LEGACY_INGEST_TOKEN = {"udp": "async", "daqiri-host": "ano", "gpudirect": "gpu", "replay": "replay"}
GPU_INGEST_MODES = ("gpudirect", "replay")
DAQIRI_INGEST_MODES = ("daqiri-host", "gpudirect")


def canonical_ingest(name: str) -> tuple[str, bool]:
    """``(canonical name, was_alias)``; ValueError for an unknown mode."""
    if name in INGEST_MODES:
        return name, False
    if name in INGEST_ALIASES:
        return INGEST_ALIASES[name], True
    known = ", ".join(INGEST_MODES + tuple(INGEST_ALIASES))
    raise ValueError(f"unknown ingest mode {name!r}; expected one of {known}")
