# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decoded observation records and the batches the receivers hand to the graph.

The numeric symbol ID indexes the HTTP-delivered directory. NaN denotes a missing
source value. A generation counter resets analytics on seek, loop or restart.
Verbatim from 4.3 ``holoscan.marketdata.wire``: ``ReplayBatch`` and ``EVENT_DTYPE`` are the
host side of the bit-identity checks, so neither layout changes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

QUOTE, TRADE, CONDITION = 1, 2, 3
HALTED = 1
EVENT_DTYPE = np.dtype(
    [
        ("timestamp_ns", ">i8"),
        ("symbol_id", ">u4"),
        ("internal_sequence", ">i4"),
        ("kind", "u1"),
        ("flags", "u1"),
        ("bid", ">f8"),
        ("ask", ">f8"),
        ("bid_size", ">f8"),
        ("ask_size", ">f8"),
        ("price", ">f8"),
        ("volume", ">f8"),
    ]
)


@dataclass
class ReplayBatch:
    events: np.ndarray
    generation: int
    recv_wall_ns: int
    sequence: int
    packet_offsets: np.ndarray | None = None
    packet_recv_ns: np.ndarray | None = None
    wire_count: int | None = None

    @property
    def sequence_count(self):
        return len(self.events) if self.wire_count is None else self.wire_count


def combine_batches(batches):
    """Coalesce graph input while retaining each packet's receive timestamp."""
    if len(batches) == 1:
        return batches[0]
    return ReplayBatch(
        np.concatenate([b.events for b in batches]),
        batches[0].generation,
        batches[-1].recv_wall_ns,
        batches[0].sequence,
        np.r_[0, np.cumsum([len(b.events) for b in batches[:-1]])],
        np.asarray([b.recv_wall_ns for b in batches], dtype=np.int64),
    )
