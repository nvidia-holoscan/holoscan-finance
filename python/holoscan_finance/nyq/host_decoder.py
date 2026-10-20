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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..marketdata.gpu_decoder import DecodeCounts
    from ..marketdata.records import ReplayBatch
    from ..payloads import PacketBatch
    from .manifest import FeedDirectory


class HostMitchDecoder:
    """Stateless per-batch decoder; counts are returned, never stored."""

    def decode(
        self, batch: PacketBatch, directory: FeedDirectory | None
    ) -> tuple[list[ReplayBatch], DecodeCounts]:
        """One ``ReplayBatch`` per accepted packet, in receive order, and the call's counts."""
        raise NotImplementedError
