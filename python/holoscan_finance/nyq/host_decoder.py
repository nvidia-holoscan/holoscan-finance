# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Python reference MITCH decode of host packet batches: the processor of ``MitchDecodeOp``.

The 4.3 rules of ``EquityState.decode`` and the ``AsyncUdpSourceOp.compute`` decode loop: a
packet is stale without a directory or from a sender port other than the directory's;
otherwise ``mapping.decode_packet`` runs under the directory's generation, midnight and
universe, and a ``ValueError`` or ``struct.error`` counts the packet as malformed. This is
the 0.28M msgs/s host baseline, so it stays a plain per-packet loop.
"""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING

from ..marketdata.gpu_decoder import DecodeCounts
from .mapping import decode_packet

if TYPE_CHECKING:
    from ..marketdata.records import ReplayBatch
    from ..payloads import PacketBatch
    from .manifest import FeedDirectory


class HostMitchDecoder:
    """Stateless per-batch decoder; counts are returned, never stored."""

    def decode(
        self, batch: PacketBatch, directory: FeedDirectory | None
    ) -> tuple[list[ReplayBatch], DecodeCounts]:
        """One ``ReplayBatch`` per accepted packet, in receive order, and the call's counts."""
        if directory is None:
            return [], DecodeCounts(0, len(batch.items), 0)
        out: list[ReplayBatch] = []
        messages = stale = malformed = 0
        for data, recv_ns, _wire_count, peer in batch.items:
            if peer is None or peer[1] != directory.sender_port:
                stale += 1
                continue
            try:
                batches, count = decode_packet(
                    data,
                    recv_ns,
                    generation=directory.generation,
                    midnight_ns=directory.midnight_ns,
                    universe=directory.universe,
                )
            except (ValueError, struct.error):
                malformed += 1
                continue
            messages += count
            out.extend(batches)
        return out, DecodeCounts(messages, stale, malformed)
