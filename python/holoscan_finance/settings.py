# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bar settings, ingest defaults and ingest mode names: one copy of each.

``BarSettings`` is the single copy of the bar defaults (the CLI, the quote-bar operators,
``decode_bench`` and ``gpu_validate`` read it) and validates with the ``QuoteBars`` rules.
``IngestDefaults`` are the GPUDirect sizes the paced 200k pps gate runs with: 131072-message
batches, a ``1 << 20`` ring and 114688 NIC buffers. ``INGEST_MODES`` are the ingest names
the CLI, ``/snapshot`` and the tools use.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from .marketdata.bars import QuoteBars


@dataclass(frozen=True)
class BarSettings:
    """The single source of the bar defaults."""

    bar_seconds: int = 5
    window_seconds: int = 300
    stale_seconds: int = 30
    min_bars: int = 20
    max_spread_bps: float = 100.0

    def __post_init__(self) -> None:
        QuoteBars([], **self.as_params())  # the QuoteBars rules: ValueError on bad values

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
GPU_INGEST_MODES = ("gpudirect", "replay")
DAQIRI_INGEST_MODES = ("daqiri-host", "gpudirect")
