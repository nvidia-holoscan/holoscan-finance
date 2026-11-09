# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The device slot ring and the drains that fill it: the native protocol, the software NIC
and the Python fallback.

A drain gathers each burst's UDP payloads into consecutive rows of a preallocated device
ring (never wrapping a burst), copies the ``HEADER_COLS``-column receive header to pinned host
memory and
publishes one record per burst; the GPU source releases ring space once decode has
synchronized. ``daqiri_drain.cpp`` implements this protocol without the GIL;
``DeviceReplayDrain`` implements it over pcap frames resident in device memory, with the
same gather kernel, ring and DAQIRI burst cadence, so everything after the drain is the
production code; ``PythonGpuDrain`` is the per-packet-binding fallback drain (and the switch
for drain comparisons). CuPy is imported inside functions.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from ..marketdata.gpu_sbe import HEADER_COLS, PACKET_HEADER, SLOT_BYTES, receive_frames

if TYPE_CHECKING:
    from types import ModuleType

    from ..marketdata.framing import PcapFrames
    from .gpu_ingest import GpuIngest

logger = logging.getLogger(__name__)

DRAIN_FIELDS = (
    "start",
    "n",
    "recv_ns",
    "t0",
    "t_frames",
    "t_launch",
    "t_sync",
    "t_free",
    "wait_ns",
)
# The native Drain.stats() keys.
DRAIN_STATS_KEYS = (
    "running",
    "bursts",
    "packets",
    "ring_peak",
    "fill",
    "poll_gap_max_ms",
    "poll_gap_at",
    "poll_gaps_10ms",
    "poll_gaps_50ms",
    "poll_gaps_100ms",
    "ring_wait_ms",
    "ring_waits",
    "last_rx_ns",
    "host_ns",
    "error",
)
_GAP_MS = (10, 50, 100)  # poll-gap thresholds, as kGapMs in daqiri_drain.cpp


class DrainLike(Protocol):
    """The native ``Drain`` protocol that ``GpuIngest.pull()`` reads."""

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def take(self) -> np.ndarray: ...  # int64 (k, 9), DRAIN_FIELDS order
    def release(self, end: int) -> None: ...  # absolute ring position, monotonic
    def reset_peak(self) -> None: ...
    def stats(self) -> dict: ...  # DRAIN_STATS_KEYS (+ "eof", "replay" for the replay drain)


@dataclass
class DeviceRing:
    """Device slots and receive headers, pinned host headers and the frame-address buffers."""

    slots: Any  # cupy (capacity, SLOT_BYTES) uint8
    header_dev: Any  # cupy (capacity, HEADER_COLS) int64
    header_host: np.ndarray  # pinned (capacity, HEADER_COLS) int64
    packed_dev: Any  # cupy (2 * max_burst,) uint64: frame addresses, then lengths
    packed_host: np.ndarray  # pinned, the same
    capacity: int
    max_burst: int

    @classmethod
    def allocate(cls, capacity: int, gpu_device: int, max_burst: int = 65536) -> DeviceRing:
        """Allocate once, before data (a new CUDA allocation can wake a suspended iGPU)."""
        import cupy as cp
        import cupyx

        max_burst = min(capacity, max_burst)  # DAQIRI bursts are at most rx_batch_size
        with cp.cuda.Device(gpu_device):
            ring = cls(
                slots=cp.empty((capacity, SLOT_BYTES), cp.uint8),
                header_dev=cp.empty((capacity, HEADER_COLS), cp.int64),
                header_host=cupyx.empty_pinned((capacity, HEADER_COLS), np.int64),
                packed_dev=cp.empty(2 * max_burst, cp.uint64),
                packed_host=cupyx.empty_pinned(2 * max_burst, np.uint64),
                capacity=capacity,
                max_burst=max_burst,
            )
            cp.cuda.Device(gpu_device).synchronize()
        return ring

    def native_kwargs(self) -> dict:
        """The ring arguments of the native ``Drain`` constructor, as addresses."""
        return dict(
            slots=self.slots.data.ptr,
            header_dev=self.header_dev.data.ptr,
            header_host=self.header_host.ctypes.data,
            capacity=self.capacity,
            packed_dev=self.packed_dev.data.ptr,
            packed_host=self.packed_host.ctypes.data,
            max_burst=self.max_burst,
        )


class DeviceFrames:
    """The frames of a capture resident in device memory, as GPUDirect leaves received frames:
    a software NIC's buffer pool that the replay drain gathers from by address."""

    def __init__(self, data: Any, ptrs: np.ndarray, lengths: np.ndarray, port_at: Any) -> None:
        self._data = data  # device uint8: the capture buffer
        self._ptrs = ptrs  # host uint64: frame device addresses
        self._lengths = lengths  # host uint64
        self._port_at = port_at  # device int64: byte index of each feed frame's source port
        self.n = len(ptrs)

    @classmethod
    def upload(
        cls, frames: PcapFrames, gpu_device: int, sender_port: int | None = None
    ) -> DeviceFrames:
        """One host-to-device copy of the whole capture, its feed sent from ``sender_port``.

        The feed is the frames from ``sender_port`` when the capture carries that port, else
        those from its most common UDP source port, rewritten to ``sender_port`` here.
        ``set_src_port`` re-sources the feed only, like ``tcprewrite --portmap feed:new``:
        frames of another sender stay stale in every pass."""
        import cupy as cp

        rel = frames.udp_src_port_offsets()
        udp = rel >= 0
        at = frames.offsets[udp] + rel[udp]
        ports = frames.data[at].astype(np.int64) << 8 | frames.data[at + 1]
        carried = sender_port is not None and bool((ports == sender_port).any())
        feed = sender_port if carried else frames.dominant_src_port()
        with cp.cuda.Device(gpu_device):
            data = cp.asarray(frames.data)
            port_at = cp.asarray(at[ports == feed] if feed is not None else at[:0])
        ptrs = np.uint64(data.data.ptr) + frames.offsets.astype(np.uint64)
        device_frames = cls(data, ptrs, frames.lengths.astype(np.uint64), port_at)
        if sender_port is not None and not carried:
            device_frames.set_src_port(sender_port)  # pass 0 comes from the published port
        cp.cuda.Device(gpu_device).synchronize()  # before a drain's non-blocking stream reads
        return device_frames

    def addresses(self, lo: int, hi: int) -> np.ndarray:
        """Frames ``[lo, hi)`` in the gather kernel's layout: addresses, then lengths."""
        return np.concatenate((self._ptrs[lo:hi], self._lengths[lo:hi]))

    def set_src_port(self, port: int) -> None:
        """Rewrite the feed frames' UDP source port (a new sender, as a replay loop's); other
        senders' frames keep theirs."""
        with self._data.device:
            self._data[self._port_at] = port >> 8
            self._data[self._port_at + 1] = port & 0xFF


class DeviceReplayDrain:
    """A software NIC over ``DeviceFrames`` that implements the native ``Drain`` protocol
    (thread ``hf-replay``); the loop mirrors ``daqiri_drain.cpp``.

    Frame ``i`` of a pass is due at ``t0 + i / pps`` (``pps`` 0: when the thread reaches it).
    A burst is closed at ``burst_packets`` frames or ``rx_timeout_us`` after its first frame
    was due, DAQIRI's rule. With ``nic_bufs`` (paced only), due frames beyond that many are
    dropped oldest first: they never reach the ring, so the loss shows up as sequence gaps,
    as on a NIC. A forming burst holds NIC buffers too, so ``nic_bufs`` must cover one burst
    (``pps * rx_timeout``) or every burst loses frames. A full ring holds the burst, and the
    thread sleeps on a condition that ``release()`` signals: in unpaced runs the ring is full
    most of the time, and a polling Python thread would tax the ingest lane's GIL (one cut a
    1 kHz pipeline to 38-43 Hz).

    Before pass ``k >= 1`` the thread waits until the consumer released everything gathered
    and ``settled()`` holds (every decoded batch consumed downstream), calls ``on_loop(k)``
    (a returned port is written into the feed frames' UDP source port) and re-bases the
    schedule, so no packet of pass ``k - 1`` is decoded or built into bars under pass ``k``'s
    directory and the wait never counts as lag. ``loops`` 0 repeats forever.

    ``poll_gap_*`` measure how late the thread woke for a burst's deadline (its NIC poll),
    ``lag`` how late the burst was gathered: ``min(due(first) + rx_timeout, due(last))``.
    It holds the GIL for about 100-200 us per burst, so it is no GIL-stall resilience test.
    """

    _LAG_WINDOW = 16384  # bursts in the lag percentiles
    _WAIT_S = 0.05  # condition-wait bound; release(), take() and stop() notify
    _SETTLE_S = 0.001  # settled() poll period at a loop boundary

    def __init__(
        self,
        frames: DeviceFrames,
        ring: DeviceRing,
        *,
        gather_fn: Any,
        pps: int,
        loops: int,
        burst_packets: int,
        rx_timeout_us: int,
        nic_bufs: int = 0,
        start_delay_ms: int = 0,
        on_loop: Callable[[int], int | None] | None = None,
        settled: Callable[[], bool] | None = None,
        min_payload: int = PACKET_HEADER,
        max_records: int = 65536,
        gpu_device: int = 0,
    ) -> None:
        if frames.n == 0:
            raise ValueError("no frames to replay")
        if not 1 <= burst_packets <= ring.max_burst:
            raise ValueError(f"burst_packets must be in 1..{ring.max_burst} (the ring's max_burst)")
        self._frames, self._ring, self._gather = frames, ring, gather_fn
        self._pps, self._loops, self._burst = pps, loops, burst_packets
        self._timeout_ns = rx_timeout_us * 1000
        self._nic_bufs, self._delay_s = nic_bufs, start_delay_ms / 1e3
        self._on_loop, self._settled, self._min_payload = on_loop, settled, min_payload
        self._max_records, self._gpu_device = max_records, gpu_device
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._cond = threading.Condition()
        # Under _cond: ring positions, the record queue and the lag window.
        self._written = self._released = 0
        self._records: list[tuple[int, ...]] = []
        self._lags: deque[int] = deque(maxlen=self._LAG_WINDOW)
        self._lag_count = 0
        self._lag_cache = (-1, 0.0, 0.0, 0.0)
        # Written by the drain thread only (GIL-atomic reads).
        self._running = self._eof = False
        self._error = ""
        self._bursts = self._packets = self._peak = self._host_ns = self._last_rx_ns = 0
        self._ring_waits = self._ring_wait_ns = 0
        self._gap_max_ns = self._gap_at_ns = 0
        self._gaps = [0] * len(_GAP_MS)
        self._pass = self._sent = self._nic_dropped = self._loop_wait_ns = 0
        self._cpu_s = 0.0

    # ---- the native Drain protocol -----------------------------------------------------
    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("drain already started")
        self._running = True
        self._last_rx_ns = time.monotonic_ns()
        self._thread = threading.Thread(target=self._run, name="hf-replay", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                logger.warning("hf-replay did not stop within 5 s")

    def take(self) -> np.ndarray:
        """Every record published since the last call: int64 ``(k, 9)``."""
        with self._cond:
            records, self._records = self._records, []
            if len(records) >= self._max_records:
                self._cond.notify_all()  # the thread may hold a burst for record space
        return np.array(records, np.int64).reshape(-1, len(DRAIN_FIELDS))

    def release(self, end: int) -> None:
        """The consumer is done with every ring row before position ``end``."""
        with self._cond:
            if end > self._released:
                self._released = end
                self._cond.notify_all()

    def reset_peak(self) -> None:
        self._peak = self._gap_max_ns = 0
        self._gaps = [0] * len(_GAP_MS)

    def stats(self) -> dict:
        with self._cond:
            fill = self._written - self._released
            count, cached = self._lag_count, self._lag_cache
            lags = None if cached[0] == count else np.array(self._lags, np.float64)
        if lags is not None:  # only when bursts were added: stats() runs every 100 ms
            p50, p99 = np.percentile(lags, (50, 99)) / 1e6 if len(lags) else (0.0, 0.0)
            cached = (count, float(p50), float(p99), float(lags.max()) / 1e6 if len(lags) else 0.0)
            self._lag_cache = cached
        return {
            "running": self._running,
            "bursts": self._bursts,
            "packets": self._packets,
            "ring_peak": self._peak,
            "fill": fill,
            "poll_gap_max_ms": self._gap_max_ns / 1e6,
            "poll_gap_at": self._gap_at_ns,
            **{f"poll_gaps_{ms}ms": n for ms, n in zip(_GAP_MS, self._gaps)},
            "ring_wait_ms": self._ring_wait_ns / 1e6,
            "ring_waits": self._ring_waits,
            "last_rx_ns": self._last_rx_ns,
            "host_ns": self._host_ns,
            "error": self._error,
            "eof": self._eof,
            "replay": {
                "pps": self._pps,
                "pass_index": self._pass,
                "sent": self._sent,
                "nic_dropped": self._nic_dropped,
                "lag_p50_ms": cached[1],
                "lag_p99_ms": cached[2],
                "lag_max_ms": cached[3],
                "loop_wait_ms": self._loop_wait_ns / 1e6,
                "thread_cpu_s": self._thread_cpu_s(),
                "eof": self._eof,
            },
        }

    # ---- the thread ----------------------------------------------------------------------
    def _thread_cpu_s(self) -> float:
        """CPU time of hf-replay: the evidence that a full ring costs the thread nothing."""
        thread = self._thread
        if thread is not None and thread.is_alive():
            try:
                return time.clock_gettime(time.pthread_getcpuclockid(thread.ident))
            except OSError:  # it exited in between
                pass
        return self._cpu_s

    def _run(self) -> None:
        try:
            self._loop()
        except Exception as exc:  # the source's next compute raises it (stats()["error"])
            logger.exception("replay drain failed")
            self._error = str(exc) or type(exc).__name__
        finally:
            self._cpu_s = time.thread_time()
            self._running = False

    def _loop(self) -> None:
        import cupy as cp

        cp.cuda.Device(self._gpu_device).use()
        stream = cp.cuda.Stream(non_blocking=True)
        n_frames = self._frames.n
        if self._delay_s and self._stop.wait(self._delay_s):
            return
        while True:
            t0, f = time.monotonic_ns(), 0  # pass schedule base
            while f < n_frames:
                end = min(f + self._burst, n_frames)
                deadline = self._deadline(t0, f, end)
                early_s = (deadline - time.monotonic_ns()) / 1e9
                stopped = self._stop.wait(early_s) if early_s > 0 else self._stop.is_set()
                if stopped:
                    return
                now = time.monotonic_ns()
                if self._pps:
                    excess = self._due(t0, now) - f - self._nic_bufs
                    if self._nic_bufs and excess > 0:  # the NIC pool overflowed: the oldest
                        f += excess  # frames never reach the ring
                        self._nic_dropped += excess
                        continue
                    end = min(end, self._due(t0, deadline))  # timed out: what was due
                self._poll_gap(now, now - deadline)
                if not self._gather_burst(stream, f, end, now, deadline):
                    return
                f = end
            if self._loops and self._pass + 1 >= self._loops:
                break
            if not self._next_pass(stream):
                return
        self._eof = True

    def _deadline(self, t0: int, f: int, end: int) -> int:
        """When burst ``[f, end)`` closes: full, or ``rx_timeout`` after its first frame."""
        if not self._pps:
            return time.monotonic_ns()
        due_first = t0 + f * 1_000_000_000 // self._pps
        return min(due_first + self._timeout_ns, t0 + (end - 1) * 1_000_000_000 // self._pps)

    def _due(self, t0: int, t: int) -> int:
        """Frames of the pass due by ``t`` (paced): frame ``i`` is due at
        ``t0 + i * 1e9 // pps``."""
        if t < t0:
            return 0
        return min(self._frames.n, ((t - t0 + 1) * self._pps + 999_999_999) // 1_000_000_000)

    def _poll_gap(self, now: int, gap: int) -> None:
        if gap > self._gap_max_ns:
            self._gap_max_ns, self._gap_at_ns = gap, now
        for i, ms in enumerate(_GAP_MS):
            if gap > ms * 1_000_000:
                self._gaps[i] += 1

    def _gather_burst(self, stream: Any, f: int, end: int, t_burst: int, deadline: int) -> bool:
        """Gather frames ``[f, end)`` into the ring and publish the record; False if stopped."""
        ring, n = self._ring, end - f
        recv_ns = time.time_ns()
        pos = self._written
        if pos % ring.capacity + n > ring.capacity:  # a burst never wraps
            pos += ring.capacity - pos % ring.capacity
        wait_ns = self._wait_for_room(pos, n)
        if wait_ns is None:
            return False
        slot = pos % ring.capacity
        packed_host = ring.packed_host[: 2 * n]
        packed_host[:] = self._frames.addresses(f, end)
        t_frames = time.monotonic_ns()
        packed_dev = ring.packed_dev[: 2 * n]
        packed_dev.set(packed_host, stream=stream)
        header = ring.header_dev[slot : slot + n]
        self._gather(
            (n,),
            (128,),
            (
                packed_dev,
                np.int32(n),
                np.int32(self._min_payload),
                ring.slots[slot : slot + n],
                np.int32(SLOT_BYTES),
                header,
            ),
            stream=stream,
        )
        header.get(stream=stream, out=ring.header_host[slot : slot + n], blocking=False)
        t_launch = time.monotonic_ns()
        stream.synchronize()  # payloads and header complete before the record is visible
        t_sync = time.monotonic_ns()
        record = (pos, n, recv_ns, t_burst, t_frames, t_launch, t_sync, t_sync, wait_ns)
        with self._cond:
            self._records.append(record)
            self._written = pos + n
            fill = self._written - self._released
            self._lags.append(t_frames - deadline)
            self._lag_count += 1
        self._peak = max(self._peak, fill)
        self._bursts += 1
        self._packets += n
        self._sent += n
        self._host_ns += t_sync - t_burst - wait_ns
        self._last_rx_ns = t_burst
        return True

    def _wait_for_room(self, pos: int, n: int) -> int | None:
        """Hold the burst until ring rows ``[pos, pos + n)`` and a record slot are free;
        the time waited, or None if stopped."""
        with self._cond:
            if self._room(pos, n):
                return 0
            w0 = time.monotonic_ns()
            while not self._room(pos, n):
                if self._stop.is_set():
                    return None
                self._cond.wait(self._WAIT_S)
        waited = time.monotonic_ns() - w0
        self._ring_waits += 1
        self._ring_wait_ns += waited
        return waited

    def _room(self, pos: int, n: int) -> bool:  # caller holds _cond
        return (
            pos + n - self._released <= self._ring.capacity
            and len(self._records) < self._max_records
        )

    def _next_pass(self, stream: Any) -> bool:
        """The loop boundary; False if stopped."""
        w0 = time.monotonic_ns()
        with self._cond:
            while self._released < self._written:  # ring and staging empty
                if self._stop.is_set():
                    return False
                self._cond.wait(self._WAIT_S)
        # Released means decoded, not consumed: the last batches may still be on their way
        # to the bars, which would count them stale under the next generation.
        while self._settled is not None and not self._settled():
            if self._stop.wait(self._SETTLE_S):
                return False
        self._loop_wait_ns += time.monotonic_ns() - w0
        port = self._on_loop(self._pass + 1) if self._on_loop is not None else None
        if port is not None:
            with stream:
                self._frames.set_src_port(port)
            stream.synchronize()
        self._pass += 1
        return True


class PythonGpuDrain:
    """The Python drain (thread ``gpu-sbe-drain``): per-packet ``get_packet_ptr`` and
    ``get_packet_length`` calls, the gather into a slot buffer per burst on its own stream,
    and the accounting from this thread. The fallback when the native helper cannot be
    built and the switch for drain comparisons; unlike the native drain it needs the GIL for
    every burst, so a stalled Python thread delays NIC buffer recycling.
    """

    _BURST_STEPS = ("frames", "alloc", "h2d_kernel", "d2h", "sync", "free", "account")

    def __init__(
        self,
        ingest: GpuIngest,
        daq: ModuleType,
        port_id: int,
        queue_id: int,
        *,
        gpu_device: int,
        keep_alive: bool,
        recv_timeout_s: float,
    ) -> None:
        self._ingest, self._daq = ingest, daq
        self._port_id, self._queue_id, self._gpu_device = port_id, queue_id, gpu_device
        self._keep_alive, self._recv_timeout_s = keep_alive, recv_timeout_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="gpu-sbe-drain", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        error = None
        try:
            self._drain()
        except Exception as exc:
            error = f"GPU SBE RX failed: {exc}"
            logger.exception(error)
        finally:
            self._ingest.end_input(error)

    def _drain(self) -> None:
        import cupy as cp

        cp.cuda.Device(self._gpu_device).use()
        stream = cp.cuda.Stream(non_blocking=True)
        daq, stats = self._daq, self._ingest.stats
        ok = daq.Status.SUCCESS
        idle = {daq.Status.NULL_PTR, getattr(daq.Status, "NOT_READY", None)}
        get_ptr, get_len = daq.get_packet_ptr, daq.get_packet_length
        packed_buf = np.empty(2 * 4096, np.uint64)
        idle_limit_ns = int(self._recv_timeout_s * 1e9)
        last_rx_ns = time.monotonic_ns()
        while not self._stop.is_set():
            status, burst = daq.get_rx_burst(self._port_id, self._queue_id)
            if status != ok or burst is None:
                if status not in idle:
                    raise RuntimeError(f"get_rx_burst returned {status}")
                if not self._keep_alive and time.monotonic_ns() - last_rx_ns > idle_limit_ns:
                    return
                time.sleep(0.001)  # each poll retakes the GIL: keep idle polling cheap
                continue
            t0 = time.perf_counter_ns()
            stats.burst_gap(t0)
            last_rx_ns = time.monotonic_ns()
            recv_ns = time.time_ns()
            marks = [t0]  # per-step times for the stall diagnosis
            try:
                n = daq.get_num_packets(burst)
                # Device addresses then lengths of every frame: one host->device copy.
                if 2 * n > packed_buf.shape[0]:
                    packed_buf = np.empty(4 * n, np.uint64)
                packed = packed_buf[: 2 * n]
                packed[:n] = [get_ptr(burst, i) for i in range(n)]
                packed[n:] = [get_len(burst, i) for i in range(n)]
                marks.append(time.perf_counter_ns())
                with stream:
                    slots = cp.empty((n, SLOT_BYTES), cp.uint8)
                    marks.append(time.perf_counter_ns())
                    header_d = receive_frames(
                        cp.asarray(packed), n, slots, min_payload=PACKET_HEADER, stream=stream
                    )
                    marks.append(time.perf_counter_ns())
                    header = cp.asnumpy(header_d, stream=stream).T
                marks.append(time.perf_counter_ns())
                stream.synchronize()  # payloads copied: NIC buffers may be recycled
                marks.append(time.perf_counter_ns())
            finally:
                daq.free_all_packets_and_burst_rx(burst)
            marks.append(time.perf_counter_ns())
            self._ingest.account_burst(slots, header_d, header, recv_ns)
            marks.append(time.perf_counter_ns())
            stats.drain_host_ns += marks[-1] - t0
            stats.timed("burst", marks, self._BURST_STEPS, packets=n)


__all__ = [
    "DRAIN_FIELDS",
    "DRAIN_STATS_KEYS",
    "DeviceFrames",
    "DeviceReplayDrain",
    "DeviceRing",
    "DrainLike",
    "PythonGpuDrain",
]
