# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Payloads that cross operator ports, and the two immutable bar telemetry snapshots.

Every port is ``hs.pyobject``: zero-copy, identity-preserving, shared across a fan-out.
So the producer never touches a payload after ``emit()`` and consumers copy before
modifying; the dataclasses are frozen. The batch classes' layouts are fixed (bit-identity
oracle): ``ReplayBatch`` is re-exported from ``marketdata.records`` and
``GpuReplayBatch`` lazily from ``marketdata.gpu_bars``. ``BarCounters`` and
``SymbolSnapshot`` live here because ``marketdata.bar_builder`` produces them and
``telemetry.stats`` stores them: this NumPy-only leaf is the one place both import without
a cycle.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from .marketdata.records import ReplayBatch

if TYPE_CHECKING:
    from .marketdata.gpu_bars import GpuReplayBatch


@dataclass(frozen=True)
class PacketBatch:
    """Datagrams of the host front ends, in receive order. Each item is
    ``(datagram: bytes, recv_wall_ns: int, wire_count: int, peer: (ip, port) | None)``."""

    items: tuple
    messages: int  # sum of wire_count
    index: int = 0  # batch number since the source's start()


# Array-carrying payloads compare and hash by identity (eq=False): field-wise equality is
# ambiguous for arrays, and a generated hash would fail on them.
@dataclass(frozen=True, eq=False)
class BarWindow:
    """The rolling window after a bar close; ``rows`` is stacked only by the consumer, so a
    window that latest-only delivery skips costs no GPU work on the ingest lane."""

    parts: tuple  # per-bar return vectors (NumPy or CuPy), oldest first; never mutated
    times: np.ndarray  # int64 bar close times, host, read-only
    generation: int
    completed_bars: int
    recv_ns: int  # receive stamp of the packet that closed the newest bar
    device: bool = False

    # cached_property writes the instance __dict__ directly, which a frozen dataclass allows.
    @functools.cached_property
    def rows(self) -> Any:
        """The ``(bars, N)`` window; a device window is stacked on its parts' device,
        whichever device is current in the consumer's thread."""
        if self.device:
            import cupy as cp

            with self.parts[0].device:
                return cp.stack(self.parts)
        return np.stack(self.parts)


@dataclass(frozen=True, eq=False)
class RiskUpdate:
    window: BarWindow
    result: dict  # CovarianceEngine.update() output + gpu_name + finished_wall_ns (host values)
    rows_host: np.ndarray | None = None  # owned host copy, set iff validated or host_variance
    backend: str = "gpu"
    validated: bool = False


@dataclass(frozen=True, eq=False)
class RiskComparison:
    update: RiskUpdate  # the GPU side
    cpu: dict | None  # cpu_window() output
    agreement: bool
    matched_work: bool
    usable: bool
    error: str | None = None


@dataclass(frozen=True)
class BarCounters:
    """Exact bar counters after one batch, published with one attribute assignment."""

    generation: int
    events: int
    quotes: int
    trades: int
    invalid_quotes: int
    late_events: int
    completed_bars: int
    last_event_ns: int
    first_batch_ns: int | None
    last_batch_ns: int | None
    # Wall-clock arrival marks: excluded from cross-backend equality.
    TIMING_FIELDS: ClassVar[tuple[str, ...]] = ("first_batch_ns", "last_batch_ns")


@dataclass(frozen=True, eq=False)
class SymbolSnapshot:
    """Per-symbol dashboard state; the arrays are read-only host copies."""

    generation: int
    symbols: tuple[str, ...]
    mid: np.ndarray
    trade: np.ndarray
    volume: np.ndarray
    quote_time: np.ndarray
    last_event_ns: int
    taken_ns: int


def __getattr__(name: str) -> Any:
    # Lazy, so that importing the payloads never imports gpu_bars (which itself imports
    # CuPy only inside functions).
    if name == "GpuReplayBatch":
        from .marketdata.gpu_bars import GpuReplayBatch

        return GpuReplayBatch
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "PacketBatch",
    "BarWindow",
    "RiskUpdate",
    "RiskComparison",
    "BarCounters",
    "SymbolSnapshot",
    "ReplayBatch",
    "GpuReplayBatch",
]
