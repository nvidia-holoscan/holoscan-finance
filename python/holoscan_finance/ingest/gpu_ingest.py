# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``GpuIngest``: the processor of both GPU sources.

Per burst it accounts the ``HEADER_COLS``-column receive header (``ReceiveAccounting.observe_burst``) and
admits the accepted rows into a queue of ``ring_size`` packets. Per batch it stages up to
``batch_size`` entries (a zero-copy view when the bursts are consecutive ring rows), decodes
SBE on the GPU under the directory read just before -- one launch, since the header's entry
counts fix every packet's output position -- synchronizes the legacy stream and only then
releases the ring rows to the drain: decode is the single consumer of each lease, in FIFO
order. No payload byte reaches the host. Records drains (native, replay) are pulled from the
source's lane; the Python drain pushes bursts from its thread. Sized for 200k packets/s (6M
entries/s) with 131072-entry batches (the paced NIC-free gate, docs/VALIDATION.md).
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ..telemetry.stats import IngestStats, SourceProbe
from .accounting import ReceiveAccounting

if TYPE_CHECKING:
    from ..marketdata.gpu_bars import GpuReplayBatch
    from ..marketdata.gpu_decoder import GpuSbeDecoder
    from ..nyq.manifest import FeedDirectory
    from ..telemetry.capture import CaptureTap
    from .drains import DeviceRing, DrainLike

_NATIVE_STEPS = ("frames", "launch", "sync", "free")  # "frames" includes any ring wait
_NATIVE_STALL_KEYS = ("poll_gap_max_ms", "poll_gap_at", "ring_waits", "ring_wait_ms")
_DRAIN_STATS_PERIOD_NS = 100_000_000


@dataclass
class _Burst:
    slots: Any  # device (n, SLOT_BYTES) uint8
    header: Any  # device (n, HEADER_COLS) int64 receive header (column 0: payload length)
    index: np.ndarray  # accepted rows (passed framing, fit in the queue)
    src_port: np.ndarray  # host, per accepted row
    recv_ns: int
    messages: int
    start: int | None = None  # records drains: ring position of row 0
    end: int | None = None  # records drains: ring position to release once decoded


class GpuIngest:
    """Receive accounting, staging, GPU decode and ring release of one GPU source."""

    def __init__(
        self,
        *,
        stats: IngestStats,
        decoder: GpuSbeDecoder,
        ring_size: int,
        batch_size: int,
        keep_alive: bool,
        recv_timeout_s: float,
        tap: CaptureTap | None = None,
    ) -> None:
        self.stats = stats
        self._accounting = ReceiveAccounting(stats)
        self._decoder, self._tap = decoder, tap
        self._ring_size, self._batch_size = ring_size, batch_size
        self._keep_alive, self._timeout_ns = keep_alive, int(recv_timeout_s * 1e9)
        self._bursts: deque[_Burst] = deque()  # under stats.lock
        self._drain: DrainLike | None = None
        self._ring: DeviceRing | None = None
        self._checked_ns = 0
        self._input_ended = False

    def attach_drain(self, drain: DrainLike, ring: DeviceRing) -> None:
        """Pull records from ``drain`` (native or replay), which fills ``ring``."""
        self._drain, self._ring = drain, ring

    def end_input(self, error: str | None = None) -> None:
        """A pushing drain (``PythonGpuDrain``) stopped receiving, with ``error`` if it failed."""
        if error:
            self.stats.error = error
        self._input_ended = True

    def account_burst(
        self,
        slots: Any,
        header_dev: Any,
        header_t: np.ndarray,
        recv_ns: int,
        start: int | None = None,
        end: int | None = None,
    ) -> None:
        """Account one burst and queue its accepted rows for decode.

        Rows beyond ``ring_size`` queued packets are counted as ``ring_dropped`` messages
        (their sequences were consumed at receipt). A records burst with nothing kept is
        released together with the last queued burst, or at once when none is queued."""
        stats = self.stats
        cand = self._accounting.observe_burst(header_t, recv_ns)
        sport, count = header_t[1], header_t[4]
        release = None
        with stats.lock:
            room = max(0, self._ring_size - stats.ring_packets)
            take = cand[:room]
            if len(take) < len(cand):
                stats.ring_dropped += int(count[cand[room:]].sum())
            if len(take):
                msgs = int(count[take].sum())
                self._bursts.append(
                    _Burst(slots, header_dev, take, sport[take], recv_ns, msgs, start, end)
                )
                stats.ring_packets += len(take)
                stats.ring_msgs += msgs
                stats.ring_peak_local = max(stats.ring_peak_local, stats.ring_packets)
            elif end is not None:
                if self._bursts:
                    self._bursts[-1].end = end
                else:
                    release = end
        if release is not None:
            self._drain.release(release)

    def pull(self) -> None:
        """Account every burst the records drain published since the last call; every
        100 ms also copy its stall and replay statistics."""
        drain, stats = self._drain, self.stats
        if drain is None:  # the Python drain accounts from its own thread
            return
        now = time.monotonic_ns()
        if now - self._checked_ns > _DRAIN_STATS_PERIOD_NS:
            self._checked_ns = now
            st = drain.stats()
            stats.drain_host_ns = st["host_ns"]
            stats.stalls["native"] = {k: st[k] for k in _NATIVE_STALL_KEYS} | {
                k: v for k, v in st.items() if k.startswith("poll_gaps_") or k in ("cpu", "spin")
            }
            if "replay" in st:
                stats.replay = st["replay"]
            if st["error"]:
                stats.error = f"GPU SBE RX failed: {st['error']}"
            elif not self._keep_alive and (
                # The replay drain knows where its input ends; a NIC goes silent.
                st["eof"] if "eof" in st else now - st["last_rx_ns"] > self._timeout_ns
            ):
                self._input_ended = True
        ring = self._ring
        for start, n, recv_ns, *marks, wait_ns in drain.take().tolist():
            stats.burst_gap(marks[0])
            stats.timed("burst", marks, _NATIVE_STEPS, packets=n, ring_wait_ms=wait_ns / 1e6)
            slot = start % ring.capacity
            self.account_burst(
                ring.slots[slot : slot + n],
                ring.header_dev[slot : slot + n],
                ring.header_host[slot : slot + n].T,
                recv_ns,
                start,
                start + n,
            )

    def next_batch(self, directory: FeedDirectory | None) -> list[GpuReplayBatch] | None:
        """Stage, decode and release up to ``batch_size`` queued entries; None if nothing
        was queued or nothing decoded."""
        stats = self.stats
        staged = []
        with stats.lock:
            msgs = 0
            while self._bursts and msgs < self._batch_size:
                b = self._bursts.popleft()
                stats.ring_packets -= len(b.index)
                stats.ring_msgs -= b.messages
                msgs += b.messages
                staged.append(b)
        if not staged:
            return None
        import cupy as cp

        marks = [time.perf_counter_ns()]
        slots, header = self._stage(staged)
        sport = np.concatenate([b.src_port for b in staged])
        recv = np.concatenate([np.full(len(b.index), b.recv_ns, np.int64) for b in staged])
        marks.append(time.perf_counter_ns())
        batches, counts = self._decoder.decode(slots, header, sport, recv, directory, msgs)
        marks.append(time.perf_counter_ns())
        # The drains gather on their own non-blocking streams: finish the decode before
        # these ring rows can be gathered into again.
        cp.cuda.get_current_stream().synchronize()
        if batches:  # before the release, which can open a replay loop boundary
            stats.decoded_batches += 1
        if self._drain is not None:
            self._drain.release(staged[-1].end)
        marks.append(time.perf_counter_ns())
        stats.timed("compute", marks, ("gather", "decode", "sync"), bursts=len(staged))
        stats.messages += counts.messages
        stats.decode_malformed += counts.malformed
        stats.stale_packets += counts.stale
        tap = self._tap
        if tap is not None and tap.active:
            for batch in batches:
                tap.record_gpu_batch(batch)
        return batches or None

    def _stage(self, staged: list[_Burst]) -> tuple[Any, Any]:
        """One device slot buffer and its device receive header rows for the staged bursts."""
        import cupy as cp

        first = staged[0]
        whole = all(len(b.index) == b.slots.shape[0] for b in staged)
        if whole and len(staged) == 1:
            return first.slots, first.header
        if (
            whole
            and first.start is not None
            and all(b.start == a.start + a.slots.shape[0] for a, b in zip(staged, staged[1:]))
        ):
            lo = first.start % self._ring.capacity
            hi = lo + sum(b.slots.shape[0] for b in staged)
            # Consecutive positions can still straddle the ring end when a burst ends
            # exactly there; only rows inside one ring pass form one view.
            if hi <= self._ring.capacity:  # consecutive ring rows: one view, no copy
                return self._ring.slots[lo:hi], self._ring.header_dev[lo:hi]
        picks = [cp.asarray(b.index) for b in staged]
        slots = cp.concatenate([b.slots[i] for b, i in zip(staged, picks)])
        header = cp.concatenate([b.header[i] for b, i in zip(staged, picks)])
        return slots, header

    def count_event_batch(self) -> None:
        """``on_sent`` callback of the source's ``PendingEmits``."""
        self.stats.event_batches += 1

    @property
    def eof(self) -> bool:
        """The drain's input ended (``keep_alive`` off) and every queued burst was decoded."""
        return self._input_ended and not self._bursts

    def probe(self) -> SourceProbe:
        """Liveness, pending packets and peak fill, from the records drain when there is one."""
        drain, stats = self._drain, self.stats
        if drain is None:  # the Python drain: its packets wait in this ingest's queue

            def queued() -> int:
                with stats.lock:
                    return stats.ring_packets

            return SourceProbe(
                alive=lambda: not self._input_ended,
                pending=queued,
                ring_peak=lambda: stats.ring_peak_local,
                reset_peak=lambda: None,  # reset_stalls() zeroes ring_peak_local itself
            )

        def alive() -> bool:
            st = drain.stats()
            return bool(st["running"]) and not st["error"]

        return SourceProbe(
            alive=alive,
            pending=lambda: int(drain.stats()["fill"]),
            ring_peak=lambda: int(drain.stats()["ring_peak"]),
            reset_peak=drain.reset_peak,
        )


__all__ = ["GpuIngest"]
