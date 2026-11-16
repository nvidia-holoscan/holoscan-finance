# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU SBE decode of one staged device slot buffer.

Packets from a sender port other than the directory's are stale and never written;
accepted packets of one call become one ``GpuReplayBatch`` whose columns stay on the GPU.
The receive header already carries every packet's entry count, so the decode is one
launch and one device-to-host copy of the per-packet results. Both GPU sources own one
decoder each (through ``ingest.gpu_ingest.GpuIngest``), which reads the directory immediately
before decoding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .gpu_bars import GpuReplayBatch
from .gpu_sbe import decode_packets

if TYPE_CHECKING:
    import cupy as cp

    from ..equity.manifest import FeedDirectory


@dataclass(frozen=True)
class DecodeCounts:
    """Per-call decode counters: entries (observations) decoded, stale and malformed packets."""

    messages: int
    stale: int
    malformed: int


class GpuSbeDecoder:
    """Stale-port mask, one-pass GPU decode and batch assembly for one source instance."""

    def __init__(self, gpu_device: int = 0) -> None:
        self.gpu_device = gpu_device

    def decode(
        self,
        slots: cp.ndarray,
        header: cp.ndarray,
        src_port: np.ndarray,
        recv_ns: np.ndarray,
        directory: FeedDirectory | None,
        max_events: int | None = None,
    ) -> tuple[list[GpuReplayBatch], DecodeCounts]:
        """Decode device ``slots`` with their device receive ``header`` rows (one framed packet
        per row; host ``src_port`` and ``recv_ns`` per packet; ``max_events`` the header's
        entry total). The directory's sender port selects the packets on the device; every
        packet is stale without a directory."""
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
                header=header,
                midnight_ns=directory.midnight_ns,
                universe=directory.universe,
                sender_port=directory.sender_port,
                max_events=max_events,
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
