# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stats objects that operators write and the dashboard, reports and finite runs read.

Each field has one writer thread: the source lane (and the drain thread, under ``lock``,
for the ring and bucket fields), the bars lane, the risk lane. Readers rely on GIL-atomic
int reads, on immutable snapshots (``BarCounters``, ``SymbolSnapshot``) and on copies taken
under ``IngestStats.lock``, which covers O(1) updates only; so the HTTP thread never touches
CUDA or waits behind data-plane work. Counter names: ``dropped`` = sequence gaps,
``ring_dropped`` = messages.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import NamedTuple

from ..payloads import BarCounters, SymbolSnapshot


class SourceProbe(NamedTuple):
    """Views of a running receiver that only the receiver can answer (native drain state)."""

    alive: Callable[[], bool]
    pending: Callable[[], int]
    ring_peak: Callable[[], int]
    reset_peak: Callable[[], None]


class IngestStats:
    """Receive and decode counters of one source and its decode operator."""

    BUCKET_NS = 100_000_000  # arrival buckets for the dashboard's message-rate plot

    lock: threading.Lock
    front_end: str  # "socket" | "daqiri-host" | "gpudirect" | "replay"; not network.transport
    opened: bool
    eof: bool
    error: str | None
    frame_fetch: str | None
    packets: int
    received_messages: int
    messages: int
    dropped: int
    ring_dropped: int
    malformed_packets: int
    decode_malformed: int
    stale_packets: int
    out_of_order_packets: int
    non_udp_frames: int
    bursts: int
    last_packet_wall_ns: int
    drain_host_ns: int
    packet_batches: int
    packet_batches_decoded: int
    event_batches: int
    decoded_batches: int  # GPU sources: batches to emit, counted before their ring release
    retries: int
    held: int
    ring_packets: int  # under lock
    ring_msgs: int  # under lock
    ring_peak_local: int  # under lock
    buckets: deque  # under lock: [bucket, msgs], newest last
    stalls: dict  # burst_gaps_ms, worst_burst, worst_compute, events, native
    replay: dict | None
    lane_cpus: list[int]

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.reset(front_end="")

    def reset(self, *, front_end: str) -> None:
        """Zero every counter for a new run of the source ``front_end`` (in its ``start()``)."""
        self.front_end = front_end
        self.opened = self.eof = False
        self.error = self.frame_fetch = None
        self.packets = self.received_messages = self.messages = 0
        self.dropped = self.ring_dropped = 0
        self.malformed_packets = self.decode_malformed = self.stale_packets = 0
        self.out_of_order_packets = self.non_udp_frames = self.bursts = 0
        self.last_packet_wall_ns = self.drain_host_ns = 0
        self.packet_batches = self.packet_batches_decoded = self.event_batches = 0
        self.decoded_batches = 0
        self.retries = self.held = 0
        self.replay = None
        self.lane_cpus = []
        self._probe: SourceProbe | None = None
        self._prev_burst_ns: int | None = None
        with self.lock:
            self.ring_packets = self.ring_msgs = self.ring_peak_local = 0
            self.buckets = deque(maxlen=600)
        self.reset_stalls()

    # ---- receive ring and arrival buckets ---------------------------------------------
    def add_bucket(self, recv_ns: int, msgs: int) -> None:
        """Count ``msgs`` into the arrival bucket of ``recv_ns``. The caller holds ``lock``."""
        b = recv_ns // self.BUCKET_NS
        if self.buckets and self.buckets[-1][0] == b:
            self.buckets[-1][1] += msgs
        else:
            self.buckets.append([b, msgs])

    def feed_buckets(self, last: int = 120) -> list[list[int]]:
        """The last ``last`` buckets as ``[epoch_ms, msgs]``."""
        with self.lock:
            out = list(self.buckets)[-last:]
        ms = self.BUCKET_NS // 1_000_000
        return [[b * ms, n] for b, n in out]

    def ring_fill(self) -> int:
        """Messages received but not yet staged for decode."""
        with self.lock:
            return self.ring_msgs

    def ring_peak(self) -> int:
        """Most packets held since the last ``reset_stalls()`` (the native drain's own peak
        when it runs)."""
        probe = self._probe
        return probe.ring_peak() if probe is not None else self.ring_peak_local

    def pending_packets(self) -> int:
        """Packets received but not yet decoded."""
        probe = self._probe
        if probe is not None:
            return probe.pending()
        with self.lock:
            return self.ring_packets

    def drain_alive(self) -> bool:
        probe = self._probe
        return probe is not None and probe.alive()

    def ready(self) -> bool:
        return self.opened and self.drain_alive() and not self.error

    def drained(self) -> bool:
        """Input ended and everything received was decoded and emitted (host front ends: every
        packet batch reached the decoder; GPU front ends emit none, so both counts are 0)."""
        return (
            self.eof
            and self.pending_packets() == 0
            and self.held == 0
            and self.packet_batches == self.packet_batches_decoded
        )

    def attach(self, probe: SourceProbe) -> None:
        self._probe = probe

    def detach(self) -> None:
        self._probe = None

    def count_packet_batch(self) -> None:
        """``on_sent`` callback of a host source's ``PendingEmits``."""
        self.packet_batches += 1

    # ---- stall diagnosis -------------------------------------------------------------
    def reset_stalls(self) -> None:
        """Loss diagnosis since the last reset: the longest gaps between NIC bursts, the
        slowest drain and decode passes with per-step times, and every >50 ms stall.

        A drain stall longer than the NIC buffer pool (``num_bufs`` packets) overflows the
        receive queue; DAQIRI counts those packets as missed."""
        self.stalls = dict(burst_gaps_ms=[], worst_burst=None, worst_compute=None, events=[])
        with self.lock:
            self.ring_peak_local = 0
        probe = self._probe
        if probe is not None:
            probe.reset_peak()

    def stall_event(self, kind: str, ms: float, **detail: object) -> None:
        if (ms > 50 or kind == "gap") and len(self.stalls["events"]) < 64:
            self.stalls["events"].append(
                dict(kind=kind, ms=round(ms, 1), at=round(time.monotonic(), 3), **detail)
            )

    def burst_gap(self, t0_ns: int) -> None:
        """Record the gap before a NIC burst that arrived at ``t0_ns`` if over 20 ms."""
        if self._prev_burst_ns is not None and t0_ns - self._prev_burst_ns > 20_000_000:
            gaps = self.stalls["burst_gaps_ms"]
            gaps.append(round((t0_ns - self._prev_burst_ns) / 1e6, 1))
            if len(gaps) > 8:
                gaps.remove(min(gaps))
        self._prev_burst_ns = t0_ns

    def timed(self, kind: str, marks: list[int], steps: tuple, **detail: object) -> None:
        """Record one drain (``burst``) or decode (``compute``) pass from its step marks."""
        ms = (marks[-1] - marks[0]) / 1e6
        worst = self.stalls["worst_" + kind]
        if ms > 50 or worst is None or ms > worst["ms"]:
            split = {k: round((b - a) / 1e6, 1) for k, a, b in zip(steps, marks, marks[1:])}
            self.stall_event(kind, ms, **split, **detail)
            if worst is None or ms > worst["ms"]:
                self.stalls["worst_" + kind] = dict(ms=round(ms, 1), **split, **detail)

    def stalls_view(self) -> dict:
        """Copies: the drain thread keeps appending while the snapshot serializes."""
        st = self.stalls
        return dict(st, burst_gaps_ms=sorted(st["burst_gaps_ms"]), events=list(st["events"]))


@dataclass(eq=False)
class BarsStats:
    """Bars-lane telemetry; the bars lane is the single writer."""

    generation: int | None = None
    symbols: tuple[str, ...] = ()
    device: bool = False
    batches_in: int = 0
    stale_packets: int = 0
    sequence_gaps: int = 0
    expected_sequence: int = 0
    windows_emitted: int = 0
    last_window_bars: int = 0
    last_window_key: tuple[int, int] | None = None  # (generation, completed_bars) last emitted
    counters: BarCounters | None = None  # replaced atomically per batch
    symbol_snapshot: SymbolSnapshot | None = None
    lane_cpus: list[int] = field(default_factory=list)

    def reset(self, generation: int, symbols: tuple[str, ...], *, device: bool) -> None:
        """A new feed generation. The run totals and the last emitted window key carry over:
        finite runs match the risk lane against the last window ever emitted."""
        self.generation, self.symbols, self.device = generation, tuple(symbols), device
        self.expected_sequence = 1
        self.counters = self.symbol_snapshot = None

    def observations(self) -> dict | None:
        """The ``/snapshot.observations`` dict, or None before the first batch."""
        c = self.counters
        if c is None:
            return None
        return dict(
            events=c.events,
            quotes=c.quotes,
            trades=c.trades,
            invalid_quotes=c.invalid_quotes,
            late_events=c.late_events,
            completed_bars=c.completed_bars,
        )


@dataclass(eq=False)
class RiskStats:
    """Risk-lane telemetry. ``RollingRiskOp``, ``RiskCrossCheckOp`` (``crosschecks_*``) and
    ``RiskSummaryOp`` (``summaries``, ``summary_drops``) write disjoint fields."""

    windows_received: int = 0
    windows_superseded: int = 0
    analyzed: int = 0
    gate_waits: int = 0
    late_activations: int = 0
    last_completed_bars: int = 0
    last_generation: int | None = None
    busy: bool = False
    last_ms: float = 0.0
    max_ms: float = 0.0
    last_window_key: tuple[int, int] | None = None  # (generation, completed_bars) last analyzed
    crosschecks_submitted: int = 0
    crosschecks_done: int = 0
    error: str | None = None
    lane_cpus: list[int] = field(default_factory=list)
    summaries: int = 0
    summary_drops: int = 0


__all__ = [
    "BarCounters",
    "BarsStats",
    "IngestStats",
    "RiskStats",
    "SourceProbe",
    "SymbolSnapshot",
]
