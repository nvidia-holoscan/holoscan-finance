# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Quote bars for one feed generation on the GPU or with NumPy: the processor of ``QuoteBarsOp``.

Merges 4.3 ``GpuQuoteBarsOp.compute`` (gpu: ``GpuQuoteBars`` + ``apply_sequence``) and the
host ``EquityState.observe`` (cpu: ``QuoteBars`` + ``SequenceTracker``). Batches of other
generations are stale, a sequence gap resets the marks, and a window is produced only when a
bar closed. ``counters()`` and ``symbol_snapshot()`` give the immutable host mirrors that
the bars lane publishes, so the HTTP thread never touches CUDA (gpu: one 40-byte D2H per
batch; one batched D2H per snapshot).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..payloads import BarCounters, BarWindow, SymbolSnapshot
from ..settings import BarSettings

if TYPE_CHECKING:
    from ..nyq.manifest import FeedDirectory
    from .bars import QuoteBars
    from .gpu_bars import GpuQuoteBars, GpuReplayBatch
    from .records import ReplayBatch


def host_batches_from_gpu(batch: GpuReplayBatch) -> list[ReplayBatch]:
    """Validation only: one D2H of the columns into ``EVENT_DTYPE`` rows, one ``ReplayBatch``
    per packet, so the cpu backend can consume the GPU sources' batches."""
    raise NotImplementedError


class BarBuilder:
    """One bars engine (``QuoteBars`` or ``GpuQuoteBars``) plus the generation and sequence
    state in front of it."""

    backend: str
    generation: int | None
    expected: int
    stale_packets: int
    sequence_gaps: int
    engine: QuoteBars | GpuQuoteBars | None

    def __init__(self, settings: BarSettings, *, backend: str = "gpu", gpu_device: int = 0) -> None:
        raise NotImplementedError

    def needs_reset(self, directory: FeedDirectory | None) -> bool:
        """True when a directory is published whose generation differs from the engine's."""
        raise NotImplementedError

    def reset(self, directory: FeedDirectory) -> None:
        """A new engine for ``directory.symbols``; the next expected sequence is 1."""
        raise NotImplementedError

    def observe(self, batches: list) -> BarWindow | None:
        """Apply one input message (``list[GpuReplayBatch]`` or ``list[ReplayBatch]``); the new
        window when a bar closed and enough bars exist, else None."""
        raise NotImplementedError

    def counters(self) -> BarCounters:
        raise NotImplementedError

    def symbol_snapshot(self) -> SymbolSnapshot:
        raise NotImplementedError
