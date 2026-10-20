# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU MITCH decode of one staged device slot buffer (was 4.3 ``EquityState.decode_gpu``).

Packets from a sender port other than the directory's are stale and never reach pass 2;
accepted packets of one call become one ``GpuReplayBatch`` whose columns stay on the GPU.
Both GPU sources own one decoder each (through ``ingest.gpu_ingest.GpuIngest``), which reads
the directory immediately before decoding, as 4.3 did. ``PacketView`` is the per-packet
record a latency capture counts (4.3 ``_PacketView``).
"""

from __future__ import annotations

from collections.abc import Iterator, Sized
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..nyq.manifest import FeedDirectory
    from .gpu_bars import GpuReplayBatch


@dataclass(frozen=True)
class DecodeCounts:
    """Per-call decode counters: MITCH messages decoded, stale and malformed packets."""

    messages: int
    stale: int
    malformed: int


class PacketView:
    """Minimal ReplayBatch-shaped record for LatencyCapture.record_packet."""

    __slots__ = (
        "events",
        "generation",
        "sequence",
        "sequence_count",
        "recv_wall_ns",
    )

    def __init__(
        self,
        events: Sized,
        generation: int,
        sequence: int,
        sequence_count: int,
        recv_wall_ns: int,
    ) -> None:
        self.events, self.generation, self.sequence = events, generation, sequence
        self.sequence_count, self.recv_wall_ns = sequence_count, recv_wall_ns


class GpuMitchDecoder:
    """Stale-port mask, two-pass GPU decode and batch assembly for one source instance."""

    def __init__(self, gpu_device: int = 0) -> None:
        raise NotImplementedError

    def decode(
        self,
        slots,
        lens_i32,
        src_port: np.ndarray,
        recv_ns: np.ndarray,
        directory: FeedDirectory | None,
    ) -> tuple[list[GpuReplayBatch], DecodeCounts]:
        """Decode device ``slots``/``lens_i32`` (one packet per row; host ``src_port`` and
        ``recv_ns`` per packet). Every packet is stale without a directory."""
        raise NotImplementedError

    def capture_views(self, batch: GpuReplayBatch) -> Iterator[PacketView]:
        """One ``PacketView`` per packet of ``batch``, for ``CaptureTap``."""
        raise NotImplementedError
