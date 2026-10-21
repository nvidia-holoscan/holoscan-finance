# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU MITCH decode of one staged device slot buffer.

Packets from a sender port other than the directory's are stale and never reach pass 2;
accepted packets of one call become one ``GpuReplayBatch`` whose columns stay on the GPU.
Both GPU sources own one decoder each (through ``ingest.gpu_ingest.GpuIngest``), which reads
the directory immediately before decoding. ``PacketView`` is the per-packet record a
latency capture counts.
"""

from __future__ import annotations

from collections.abc import Iterator, Sized
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .gpu_bars import GpuReplayBatch
from .gpu_mitch import decode_packets

if TYPE_CHECKING:
    import cupy as cp

    from ..nyq.manifest import FeedDirectory


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
        self.gpu_device = gpu_device

    def decode(
        self,
        slots: cp.ndarray,
        lens_i32: cp.ndarray,
        src_port: np.ndarray,
        recv_ns: np.ndarray,
        directory: FeedDirectory | None,
    ) -> tuple[list[GpuReplayBatch], DecodeCounts]:
        """Decode device ``slots``/``lens_i32`` (one packet per row; host ``src_port`` and
        ``recv_ns`` per packet). Every packet is stale without a directory."""
        if directory is None:
            return [], DecodeCounts(0, len(src_port), 0)
        keep = src_port == directory.sender_port
        stale = len(keep) - int(keep.sum())
        if not keep.any():
            return [], DecodeCounts(0, stale, 0)
        import cupy as cp

        with cp.cuda.Device(self.gpu_device):
            dec = decode_packets(
                slots,
                lens_i32,
                midnight_ns=directory.midnight_ns,
                universe=directory.universe,
                keep=keep,
            )
        ok = (dec.status == 0) & keep
        malformed = int(((dec.status != 0) & keep).sum())
        idx = np.flatnonzero(ok)
        if not len(idx):
            return [], DecodeCounts(0, stale, malformed)
        batch = GpuReplayBatch(
            dec.columns,
            directory.generation,
            dec.sequence[idx],
            dec.wire_count[idx],
            dec.offsets[idx],
            recv_ns[idx],
            int(recv_ns[idx[-1]]),
        )
        return [batch], DecodeCounts(int(dec.wire_count[idx].sum()), stale, malformed)

    @staticmethod
    def capture_views(batch: GpuReplayBatch) -> Iterator[PacketView]:
        """One ``PacketView`` per packet of ``batch``, for ``CaptureTap``."""
        ends = np.r_[batch.packet_offsets[1:], batch.n_events]
        for k in range(batch.n_packets):
            yield PacketView(
                range(int(ends[k] - batch.packet_offsets[k])),
                batch.generation,
                int(batch.sequence[k]),
                int(batch.wire_count[k]),
                int(batch.packet_recv_ns[k]),
            )
