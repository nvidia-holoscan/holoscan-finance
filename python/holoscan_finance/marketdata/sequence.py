# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host packet-sequence rules in front of ``QuoteBars``: the one host copy of 4.3
``EquityState.observe`` (it also replaces ``gpu_validate._observe_host``).

Per generation, a packet whose sequence is below the expected one is stale and skipped,
and a gap flushes the pending packets and resets the marks, so no bar return spans
missing updates. ``gpu_bars.apply_sequence`` is the device-batch mirror of these rules.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .records import combine_batches

if TYPE_CHECKING:
    from .bars import QuoteBars
    from .records import ReplayBatch


class SequenceTracker:
    """Expected-sequence state of one generation; counters accumulate across ``observe`` calls."""

    generation: int
    expected: int
    stale_packets: int
    sequence_gaps: int

    def __init__(self, generation: int, expected: int = 1) -> None:
        self.generation = generation
        self.expected = expected
        self.stale_packets = self.sequence_gaps = 0

    def observe(self, bars: QuoteBars, batches: list[ReplayBatch]) -> None:
        """Feed ``batches`` (one per packet) to ``bars`` under the 4.3 stale and gap rules."""
        pending: list[ReplayBatch] = []
        for batch in batches:
            if batch.generation != self.generation or batch.sequence < self.expected:
                self.stale_packets += 1
                continue
            if self.expected and batch.sequence > self.expected:
                if pending:
                    bars.observe(combine_batches(pending))
                    pending.clear()
                self.sequence_gaps += batch.sequence - self.expected
                # Unknown missing updates: require refreshed marks and a new return endpoint.
                bars.reset_marks()
            self.expected = batch.sequence + batch.sequence_count
            pending.append(batch)
        if pending:
            bars.observe(combine_batches(pending))
