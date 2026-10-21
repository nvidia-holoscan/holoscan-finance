# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host decode oracle and bitwise comparisons for the GPU and numba MITCH decoders.

``host_decode`` decodes captured payloads with the reference ``nyq.mapping.decode_packet`` in
a fork pool; ``compare_decoded`` requires another decoder's output to match it bit for bit:
per-packet accept/reject stage, counts, sequences and every observation field, floats as raw
bit patterns so NaN placement and signed zeros count; ``ulps`` measures float distances.

``HostReference`` is the post-run oracle of ``market_data_bench --check``: the capture through
the host sequence rules and ``QuoteBars``, recording every bar as it closes, so each GPU
window of the graph is checked against an independent decode (docs/VALIDATION.md step 2,
2026-10-20, full-size synthetic day: 41 of 13,414,033 finite returns differ by 1 ULP, CUDA
vs NumPy ``log``, none by more).
"""

from __future__ import annotations

import multiprocessing as mp
import struct
from typing import TYPE_CHECKING

import numpy as np

from ..marketdata import mitch_codec as unit
from ..marketdata.bars import QuoteBars
from ..marketdata.gpu_mitch import FIELDS
from ..marketdata.records import EVENT_DTYPE, ReplayBatch
from ..marketdata.sequence import SequenceTracker
from ..nyq.mapping import decode_packet

if TYPE_CHECKING:
    import cupy as cp

    from ..marketdata.framing import PcapFrames
    from ..marketdata.gpu_mitch import DecodedPackets
    from ..nyq.manifest import FeedDirectory
    from ..settings import BarSettings

# Packets HostReference decodes at once (as gpu_validate --chunk). Decoding holds about 4 KB
# per packet (events and worker results): about 12 GB for a full day in one piece.
_CHUNK_PACKETS = 250_000
_CTX: dict = {}  # per worker process: midnight_ns, universe


def _init(midnight_ns: int, universe: int) -> None:
    _CTX.update(midnight_ns=midnight_ns, universe=universe)


def _host_decode(chunk: list[bytes]) -> tuple:
    midnight_ns, universe = _CTX["midnight_ns"], _CTX["universe"]
    n = len(chunk)
    status = np.zeros(n, np.int32)
    nev = np.zeros(n, np.int32)
    wire = np.zeros(n, np.int32)
    seq = np.zeros(n, np.uint32)
    rows = []
    for i, data in enumerate(chunk):
        try:
            _, s, c = unit.inspect_unit(data)
        except (ValueError, struct.error):
            status[i] = 1
            continue
        seq[i], wire[i] = s, c
        try:
            batches, count = decode_packet(
                data, 0, generation=1, midnight_ns=midnight_ns, universe=universe
            )
        except (ValueError, struct.error):
            status[i] = 2
            continue
        ev = batches[0].events
        nev[i] = len(ev)
        rows.append(ev)
    events = np.concatenate(rows) if rows else np.zeros(0, dtype=EVENT_DTYPE)
    return status, nev, wire, seq, events


def host_decode(payloads: list[bytes], midnight_ns: int, universe: int, workers: int = 6) -> tuple:
    """``(status, events_per_packet, wire_count, sequence, events)`` of every payload; status
    is 0 decoded, 1 rejected by ``inspect_unit``, 2 rejected by ``decode_packet``.

    ``workers`` > 1 decodes in a fork pool (about 0.3M messages/s per worker); 1 decodes in
    this process."""
    if workers <= 1 or not payloads:
        _init(midnight_ns, universe)
        return _host_decode(payloads)
    parts = max(1, workers * 4)
    size = (len(payloads) + parts - 1) // parts
    chunks = [payloads[i : i + size] for i in range(0, len(payloads), size)]
    with mp.get_context("fork").Pool(
        workers, initializer=_init, initargs=(midnight_ns, universe)
    ) as pool:
        results = pool.map(_host_decode, chunks)
    return tuple(np.concatenate([r[k] for r in results]) for k in range(5))


def to_host(array: np.ndarray | cp.ndarray) -> np.ndarray:
    """A host copy of a NumPy or CuPy array (no copy for NumPy)."""
    return (
        array.get()
        if hasattr(array, "get") and not isinstance(array, np.ndarray)
        else np.asarray(array)
    )


def compare_decoded(host: tuple, gpu: DecodedPackets) -> dict:
    """Mismatch counts (all zero == bit-identical) between ``host_decode`` output and a
    ``DecodedPackets`` holding NumPy (CPU decoder) or CuPy (GPU decoder) columns."""
    status, nev, wire, seq, events = host
    out = {
        "packets": int(len(status)),
        "status_mismatch": int((status != gpu.status).sum()),
        "status_counts_host": {
            int(k): int(v) for k, v in zip(*np.unique(status, return_counts=True))
        },
    }
    ok = status == 0
    out["events_per_packet_mismatch"] = int((nev[ok] != gpu.events_per_packet[ok]).sum())
    fr = status != 1  # framing fields are defined whenever inspect_unit passed
    out["wire_count_mismatch"] = int((wire[fr] != gpu.wire_count[fr]).sum())
    out["sequence_mismatch"] = int((seq[fr] != gpu.sequence[fr]).sum())
    out["events_host"] = int(len(events))
    out["events_gpu"] = int(gpu.total_events)
    if len(events) != gpu.total_events:
        out["event_count_mismatch"] = True
        return out
    native = events.astype(events.dtype.newbyteorder("="))
    for name, dt in FIELDS:
        h = native[name]
        g = to_host(gpu.columns[name])
        if np.dtype(dt).kind == "f":
            bad = h.view(np.int64) != g.view(np.int64)  # bitwise: NaN and signed zero exact
        else:
            bad = h.astype(np.int64) != g.astype(np.int64)
        out[f"field_{name}_mismatch"] = int(bad.sum())
    out["max_price_magnitude"] = float(
        np.nanmax(np.abs(np.r_[native["bid"], native["ask"], native["price"], 0.0]))
    )
    return out


def ulps(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance in units in the last place between two float64 arrays of finite values."""
    ai, bi = a.view(np.int64), b.view(np.int64)
    ai = np.where(ai < 0, np.int64(-(2**63)) - ai, ai)
    bi = np.where(bi < 0, np.int64(-(2**63)) - bi, bi)
    return np.abs(ai - bi)


class HostReference:
    """CPU oracle for one capture of one generation: host decode, the host sequence rules
    and ``QuoteBars``, with every closed bar recorded."""

    directory: FeedDirectory
    bars: QuoteBars
    tracker: SequenceTracker

    def __init__(self, directory: FeedDirectory, settings: BarSettings) -> None:
        self.directory = directory
        self.bars = QuoteBars(directory.symbols, **settings.as_params())
        self.tracker = SequenceTracker(directory.generation)
        self._decode_counts: dict[str, int] = {}
        self._times: list[int] = []
        self._returns: list[np.ndarray] = []

    def feed(self, frames: PcapFrames, workers: int = 6) -> None:
        """Decode every IPv4/UDP frame of ``frames`` (``workers`` as in ``host_decode``) and
        feed the bars in capture order."""
        d = self.directory
        payloads = frames.payloads()
        counts = dict.fromkeys(("packets", "received_messages", "messages", "rejected_packets"), 0)
        bar_ns = self.bars.bar_ns
        pending: list[ReplayBatch] = []
        bucket = None
        for base in range(0, len(payloads), _CHUNK_PACKETS):
            part = payloads[base : base + _CHUNK_PACKETS]
            status, nev, wire, seq, events = host_decode(part, d.midnight_ns, d.universe, workers)
            framed = status != 1
            counts["packets"] += int(framed.sum())
            counts["received_messages"] += int(wire[framed].sum())
            counts["messages"] += int(wire[status == 0].sum())
            counts["rejected_packets"] += int((status != 0).sum())
            offsets = np.r_[0, np.cumsum(nev, dtype=np.int64)]
            stamps = events["timestamp_ns"]
            for i in np.flatnonzero(status == 0):
                lo, hi = int(offsets[i]), int(offsets[i + 1])
                # Feed one bar at a time: QuoteBars keeps only the newest window of rows, so
                # bars are recorded as they close, never after more than a window closed at once.
                if lo < hi and int(stamps[lo]) // bar_ns != bucket:
                    self._observe(pending)
                    pending, bucket = [], int(stamps[lo]) // bar_ns
                # recv_wall_ns is the packet index: receive stamps never change a bar's returns.
                pending.append(
                    ReplayBatch(
                        events[lo:hi],
                        d.generation,
                        base + int(i),
                        int(seq[i]),
                        wire_count=int(wire[i]),
                    )
                )
        self._observe(pending)
        self._decode_counts = counts

    def _observe(self, batches: list[ReplayBatch]) -> None:
        if not batches:
            return
        before = self.bars.completed_bars
        self.tracker.observe(self.bars, batches)
        # A jump past the window (the long-gap rule) closes more bars than the deque holds;
        # the bars it dropped reach no window on either side either.
        closed = min(self.bars.completed_bars - before, len(self.bars.rows))
        for r in range(closed, 0, -1):
            self._times.append(int(self.bars.times[-r]))
            self._returns.append(self.bars.rows[-r])  # QuoteBars never mutates a closed bar

    def closed_bars(self) -> tuple[np.ndarray, np.ndarray]:
        """``(times int64[k], returns float64[k, universe])`` of every recorded bar, in order."""
        times = np.asarray(self._times, dtype=np.int64)
        if not self._returns:
            return times, np.zeros((0, len(self.bars.symbols)))
        return times, np.stack(self._returns)

    def counters(self) -> dict[str, int]:
        """The ``BarCounters`` fields that both backends must reproduce, plus the sequence
        counters (``BarsStats.sequence_gaps`` and ``stale_packets``)."""
        b, t = self.bars, self.tracker
        return dict(
            events=b.events,
            quotes=b.quotes,
            trades=b.trades,
            invalid_quotes=b.invalid_quotes,
            late_events=b.late_events,
            completed_bars=b.completed_bars,
            sequence_gaps=t.sequence_gaps,
            stale_packets=t.stale_packets,
        )

    def decode_counts(self) -> dict[str, int]:
        """What the GPU receive and decode must count for the same capture: ``packets`` and
        ``received_messages`` pass the framing check, ``messages`` decode, and
        ``rejected_packets`` fail either check (``malformed_packets + decode_malformed``)."""
        return dict(self._decode_counts)
