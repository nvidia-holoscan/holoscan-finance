# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPUDirect MITCH source: NIC -> GPU memory -> GPU payload gather -> GPU decode.

DAQIRI receives flow-matched frames straight into a ``device`` memory region
(GPUDirect RDMA via dma-buf). Per burst, the drain reads each frame's device
address and length from DAQIRI, a kernel copies the UDP payloads into a slot
buffer and applies the ``inspect_unit`` framing check on the GPU, and the burst
is returned to the NIC. Only per-packet header fields (source port, sequence,
message count, status) come back to the host for receive accounting (sequence
gaps, reordering, ring capacity), vectorized for the normal contiguous case. No
packet payload is ever copied to or parsed on the CPU.

The drain is a native thread (``native/daqiri_drain.cpp``) that never takes the
Python interpreter lock: it gathers bursts into a preallocated device ring of
``ring_size`` slots, and ``compute()`` accounts and decodes them. A Python thread
that stalls while holding the lock therefore cannot stop NIC buffers from being
recycled; the ring absorbs the stall. If the helper cannot be built, a Python
drain thread does the same work with per-burst allocations.

``compute()`` stages buffered bursts up to ``batch_size`` messages and calls
``batch_decoder(slots, lengths, src_port, recv_ns)`` with one device slot buffer;
it runs the GPU MITCH decode (:func:`holoscan.marketdata.gpu_mitch.decode_packets`)
and returns ``(batches, decoded, malformed)``. ``batches`` is emitted on ``tick``.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from holoscan.conditions import BooleanCondition, PeriodicCondition
from holoscan.core import ExecutionContext, InputContext, Operator, OperatorSpec, OutputContext

# Absolute imports: in the source tree this operator is a package under operators/, in the
# built package a flat holoscan.marketdata module.
from holoscan.marketdata.daqiri import build_daqiri_config, require_daqiri
from holoscan.marketdata.gpu_mitch import SLOT_BYTES, UNIT_HEADER, receive_frames

logger = logging.getLogger(__name__)


def _isolate_cores(dpdk_cores):
    """Move this process's threads off the DPDK lcores, leaving the lcore threads (pinned
    inside ``dpdk_cores``) alone; threads created later inherit the mask. A Python or CUDA
    thread scheduled onto the RX core takes CPU time from the NIC poll loop."""
    if not dpdk_cores or not hasattr(os, "sched_setaffinity"):
        return
    reserved = set(dpdk_cores)
    for tid in os.listdir("/proc/self/task"):
        try:
            current = os.sched_getaffinity(int(tid))
            if not current <= reserved and current - reserved:
                os.sched_setaffinity(int(tid), current - reserved)
        except OSError:
            pass


_BURST_STEPS = ("frames", "alloc", "h2d_kernel", "d2h", "sync", "free", "account")
_NATIVE_STEPS = ("frames", "launch", "sync", "free")  # "frames" includes any ring wait


@dataclass
class _Burst:
    slots: object  # device (n, SLOT_BYTES) uint8
    header: object  # device (n, 7) int64 receive header (column 0: payload length)
    index: np.ndarray  # accepted packet rows (passed framing, fit in the ring)
    src_port: np.ndarray  # host, per accepted packet
    recv_ns: int
    messages: int
    start: int | None = None  # native drain: ring position of row 0
    end: int | None = None  # native drain: ring space to release once decoded


class GpuMitchSourceOp(Operator):
    """Counters and helpers match the application's host-memory sources
    (``holoscan.equities.host_sources``), so the network panel, captures and
    benchmarks read the same fields for every ingest path."""

    def __init__(
        self,
        fragment,
        *args,
        port=26400,
        pcie_address="0004:03:00.0",
        daqiri_config=None,
        interface_name="rx_port",
        queue_id=0,
        master_core=3,
        rx_core=9,
        num_bufs=51200,
        buf_size=2048,
        # Few, large bursts: every burst costs the drain thread several GIL round trips,
        # which other Python threads can stretch to ~1 ms each. At 200k pkt/s a 5 ms
        # timeout yields ~200 bursts/s of ~1,000 packets; 20 ms yields ~50 of ~4,000.
        rx_batch_size=4096,
        rx_timeout_us=20000,
        daqiri_log_level="warn",
        ring_size=65536,
        batch_size=16384,
        batch_decoder=None,
        keep_alive=True,
        recv_timeout=5.0,
        poll_period_ns=500_000,
        gpu_device=0,
        native_drain=True,
        name="gpu_mitch_source",
        **kwargs,
    ):
        if batch_decoder is None:
            raise ValueError("GpuMitchSourceOp needs a batch_decoder")
        if buf_size > SLOT_BYTES:
            raise ValueError(f"buf_size must be <= {SLOT_BYTES}")
        self.port = int(port)
        self.pcie_address, self.interface_name, self.queue_id = (
            pcie_address,
            interface_name,
            int(queue_id),
        )
        self._daqiri_config = daqiri_config or build_daqiri_config(
            pcie_address=pcie_address,
            udp_dst_port=int(port),
            interface_name=interface_name,
            queue_id=int(queue_id),
            master_core=int(master_core),
            rx_core=int(rx_core),
            memory_kind="device",
            num_bufs=int(num_bufs),
            buf_size=int(buf_size),
            rx_batch_size=int(rx_batch_size),
            rx_timeout_us=int(rx_timeout_us),
            log_level=daqiri_log_level,
        )
        # DAQIRI's DPDK lcores; every other thread of this process is kept off them.
        self.dpdk_cores = () if daqiri_config else (int(master_core), int(rx_core))
        self.ring_size, self.batch_size = int(ring_size), int(batch_size)
        self.native_drain = bool(native_drain)
        self._native = None
        self._native_ring = None
        self.batch_decoder = batch_decoder
        self.keep_alive, self.recv_timeout = bool(keep_alive), float(recv_timeout)
        self.gpu_device = int(gpu_device)
        self._daq = None
        self._port_id = -1
        self.error = None
        self._lock = threading.Lock()
        self._ring: deque = deque()
        self._ring_packets = self._ring_msgs = 0
        self._thread = None
        self._stop = threading.Event()
        self._eos = threading.Event()
        self._bucket_ns = 100_000_000
        self._buckets: deque = deque(maxlen=600)
        self._reset_counters()
        self._alive = BooleanCondition(fragment, name=name + "_alive", enable_tick=True)
        self._tick = PeriodicCondition(
            fragment, recess_period=int(poll_period_ns), name=name + "_tick"
        )
        super().__init__(fragment, self._alive, self._tick, *args, name=name, **kwargs)

    def _reset_counters(self):
        self.packets = self.messages = self.dropped = self.ring_dropped = 0
        self.received_messages = self.malformed_packets = self.out_of_order_packets = 0
        self.non_udp_frames = self.bursts = 0
        self.frame_fetch = None
        self._ring_peak = 0
        self._prev_burst_ns = None
        self.last_packet_wall_ns = 0
        self._expected_seq = None
        self._session = None
        # Host time spent per burst outside the NIC poll (for the benchmark report).
        self.drain_host_ns = 0
        self.reset_stalls()

    def reset_stalls(self):
        """Loss diagnosis since the last reset: the longest gaps between NIC bursts, the
        slowest drain and decode passes with per-step times, and every >50 ms stall.

        A drain stall longer than the NIC buffer pool (``num_bufs`` packets) overflows the
        receive queue; DAQIRI counts those packets as missed."""
        self.stalls = dict(burst_gaps_ms=[], worst_burst=None, worst_compute=None, events=[])
        self._ring_peak = 0
        if getattr(self, "_native", None) is not None:
            self._native.reset_peak()

    def stall_event(self, kind, ms, **detail):
        if (ms > 50 or kind == "gap") and len(self.stalls["events"]) < 64:
            self.stalls["events"].append(
                dict(kind=kind, ms=round(ms, 1), at=round(time.monotonic(), 3), **detail)
            )

    def _burst_gap(self, t0):
        """Record the gap before a NIC burst that arrived at ``t0`` (ns) if over 20 ms."""
        if self._prev_burst_ns is not None and t0 - self._prev_burst_ns > 20_000_000:
            gaps = self.stalls["burst_gaps_ms"]
            gaps.append(round((t0 - self._prev_burst_ns) / 1e6, 1))
            if len(gaps) > 8:
                gaps.remove(min(gaps))
        self._prev_burst_ns = t0

    def _timed(self, kind, marks, steps, **detail):
        """Record one drain (``burst``) or decode (``compute``) pass from its step marks."""
        ms = (marks[-1] - marks[0]) / 1e6
        worst = self.stalls["worst_" + kind]
        if ms > 50 or worst is None or ms > worst["ms"]:
            split = {k: round((b - a) / 1e6, 1) for k, a, b in zip(steps, marks, marks[1:])}
            self.stall_event(kind, ms, **split, **detail)
            if worst is None or ms > worst["ms"]:
                self.stalls["worst_" + kind] = dict(ms=round(ms, 1), **split, **detail)

    def setup(self, spec: OperatorSpec):
        spec.output("tick")

    # ---- lifecycle -------------------------------------------------------------------
    def start(self):
        daqiri = require_daqiri()
        for name in ("get_packet_ptr", "get_packet_length"):
            if not hasattr(daqiri, name):
                raise RuntimeError(
                    f"DAQIRI bindings lack {name}; GPU ingest needs device packet pointers"
                )
        self.error = None
        with self._lock:
            self._ring.clear()
            self._ring_packets = self._ring_msgs = 0
            self._buckets.clear()
        self._reset_counters()
        self._stop.clear()
        self._eos.clear()
        status = daqiri.daqiri_init(self._daqiri_config)
        if status != daqiri.Status.SUCCESS:
            raise RuntimeError(f"daqiri_init failed: {status}")
        self._daq = daqiri
        _isolate_cores(self.dpdk_cores)
        self._port_id = daqiri.get_port_id(self.interface_name)
        if self._port_id < 0:
            daqiri.shutdown()
            self._daq = None
            raise RuntimeError(f"DAQIRI interface {self.interface_name!r} not found")
        try:
            native = self._start_native()
        except Exception:  # any failure: the Python drain below still works
            logger.exception("native drain failed to start; using the Python drain")
            self._native = self._native_ring = None
            native = False
        if not native:
            self._thread = threading.Thread(
                target=self._drain_checked, daemon=True, name="gpu-mitch-drain"
            )
            self._thread.start()
        logger.info(
            "GPUDirect MITCH source draining %s (%s) q%d into GPU %d, udp_dst=%d (%s)",
            self.interface_name,
            self.pcie_address,
            self.queue_id,
            self.gpu_device,
            self.port,
            self.frame_fetch or "python drain",
        )

    def _start_native(self):
        """Start the GIL-free drain over a preallocated device ring; False if unavailable."""
        from holoscan.marketdata.native import load_daqiri_drain

        drain_cls = load_daqiri_drain() if self.native_drain else None
        if drain_cls is None:
            return False
        import cupy as cp
        import cupyx

        from holoscan.marketdata.gpu_mitch import _kernels

        cp.cuda.Device(self.gpu_device).use()
        max_burst = min(self.ring_size, 65536)  # DAQIRI bursts are at most rx_batch_size
        ring = dict(
            slots=cp.empty((self.ring_size, SLOT_BYTES), cp.uint8),
            header_dev=cp.empty((self.ring_size, 7), cp.int64),
            header_host=cupyx.empty_pinned((self.ring_size, 7), np.int64),
            packed_dev=cp.empty(2 * max_burst, cp.uint64),
            packed_host=cupyx.empty_pinned(2 * max_burst, np.uint64),
        )
        # Compiled here, not in the drain thread; the launch handle is the CUfunction.
        gather = _kernels().get_function("gather_frame").kernel
        cp.cuda.Device(self.gpu_device).synchronize()
        native = drain_cls(
            port=self._port_id,
            queue=self.queue_id,
            gather_fn=gather.ptr,
            device=self.gpu_device,
            slots=ring["slots"].data.ptr,
            header_dev=ring["header_dev"].data.ptr,
            header_host=ring["header_host"].ctypes.data,
            capacity=self.ring_size,
            stride=SLOT_BYTES,
            packed_dev=ring["packed_dev"].data.ptr,
            packed_host=ring["packed_host"].ctypes.data,
            max_burst=max_burst,
            min_payload=UNIT_HEADER,
            max_records=65536,
        )
        self._native_ring, self._native = ring, native
        self._native_checked_ns = 0
        native.start()
        self.frame_fetch = "native drain"
        return True

    def drain_alive(self):
        if self._native is not None:
            st = self._native.stats()
            return bool(st["running"]) and not st["error"]
        return self._thread is not None and self._thread.is_alive()

    def pending_packets(self):
        """Packets received but not yet decoded (in the ring or the native drain)."""
        if self._native is not None:
            return int(self._native.stats()["fill"])
        with self._lock:
            return self._ring_packets

    def ring_peak(self):
        """Most packets held in the ring since the last ``reset_stalls()``."""
        if self._native is not None:
            return int(self._native.stats()["ring_peak"])
        return self._ring_peak

    def stop(self):
        self._stop.set()
        if self._native is not None:
            self._native.stop()  # joins the thread before DAQIRI shuts down
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._daq is not None:
            self._daq.print_stats()
            self._daq.shutdown()
            self._daq = None

    def _drain_checked(self):
        try:
            self._drain()
        except Exception as exc:
            self.error = f"GPU MITCH RX failed: {exc}"
            logger.exception(self.error)
            self._eos.set()

    # ---- drain thread: NIC burst -> GPU slots -> host header accounting ---------------
    def _drain(self):
        import cupy as cp

        cp.cuda.Device(self.gpu_device).use()
        stream = cp.cuda.Stream(non_blocking=True)
        daq = self._daq
        ok = daq.Status.SUCCESS
        idle = {daq.Status.NULL_PTR, getattr(daq.Status, "NOT_READY", None)}
        get_ptr, get_len = daq.get_packet_ptr, daq.get_packet_length
        from holoscan.marketdata.native import load_daqiri_frames

        burst_frames = load_daqiri_frames()  # None -> per-packet binding calls
        self.frame_fetch = "native batch" if burst_frames else "per-packet binding"
        packed_buf = np.empty(2 * 4096, np.uint64)
        idle_limit_ns = int(self.recv_timeout * 1e9)
        last_rx_ns = time.monotonic_ns()
        while not self._stop.is_set():
            status, burst = daq.get_rx_burst(self._port_id, self.queue_id)
            if status != ok or burst is None:
                if status not in idle:
                    raise RuntimeError(f"get_rx_burst returned {status}")
                if not self.keep_alive and time.monotonic_ns() - last_rx_ns > idle_limit_ns:
                    self._eos.set()
                    return
                time.sleep(0.001)  # each poll retakes the GIL: keep idle polling cheap
                continue
            t0 = time.perf_counter_ns()
            self._burst_gap(t0)
            last_rx_ns = time.monotonic_ns()
            recv_ns = time.time_ns()
            marks = [t0]  # per-step times for the stall diagnosis
            try:
                n = daq.get_num_packets(burst)
                # Device addresses then lengths of every frame: one host->device copy.
                if 2 * n > packed_buf.shape[0]:
                    packed_buf = np.empty(4 * n, np.uint64)
                packed = packed_buf[: 2 * n]
                if burst_frames is not None:
                    burst_frames(burst, packed)
                else:
                    packed[:n] = [get_ptr(burst, i) for i in range(n)]
                    packed[n:] = [get_len(burst, i) for i in range(n)]
                marks.append(time.perf_counter_ns())
                with stream:
                    slots = cp.empty((n, SLOT_BYTES), cp.uint8)
                    marks.append(time.perf_counter_ns())
                    header_d = receive_frames(
                        cp.asarray(packed), n, slots, min_payload=UNIT_HEADER, stream=stream
                    )
                    marks.append(time.perf_counter_ns())
                    header = cp.asnumpy(header_d, stream=stream).T
                marks.append(time.perf_counter_ns())
                stream.synchronize()  # payloads copied: NIC buffers may be recycled
                marks.append(time.perf_counter_ns())
            finally:
                daq.free_all_packets_and_burst_rx(burst)
            marks.append(time.perf_counter_ns())
            self._account(slots, header_d, header, recv_ns)
            marks.append(time.perf_counter_ns())
            self.drain_host_ns += marks[-1] - t0
            self._timed("burst", marks, _BURST_STEPS, packets=n)

    def _account(self, slots, header_d, header, recv_ns):
        """Vectorized mirror of AsyncUdpSourceOp._enqueue_packet for one burst."""
        length, sport, sip, status, count, seq, group = header
        self.bursts += 1
        udp = length >= 0
        self.non_udp_frames += int((~udp).sum())  # strip_udp_datagram -> None: skipped silently
        framing_bad = udp & (status == 1)
        self.malformed_packets += int(framing_bad.sum())
        cand = np.flatnonzero(udp & (status != 1))
        if not len(cand):
            return
        seq_c, cnt_c = seq[cand], count[cand]
        sess = np.stack([sip[cand], sport[cand], group[cand]], axis=1)
        single = len(cand) == 1 or bool((sess == sess[0]).all())
        contiguous = len(cand) == 1 or bool(np.array_equal(seq_c[1:], seq_c[:-1] + cnt_c[:-1]))
        session0 = tuple(int(x) for x in sess[0])
        if single and contiguous:
            if session0 != self._session:
                self._session, self._expected_seq = session0, None
            first = int(seq_c[0])
            if self._expected_seq is None or first >= self._expected_seq:
                if self._expected_seq is not None and first > self._expected_seq:
                    self.dropped += first - self._expected_seq
                    self.stall_event("gap", 0, messages=first - self._expected_seq)
                self._expected_seq = int(seq_c[-1] + cnt_c[-1])
            else:
                self._account_slow(sess, seq_c, cnt_c)
        else:
            self._account_slow(sess, seq_c, cnt_c)
        self.packets += len(cand)
        msgs = int(cnt_c.sum())
        self.received_messages += msgs
        self.last_packet_wall_ns = recv_ns
        b = recv_ns // self._bucket_ns
        with self._lock:
            if self._buckets and self._buckets[-1][0] == b:
                self._buckets[-1][1] += msgs
            else:
                self._buckets.append([b, msgs])
            room = max(0, self.ring_size - self._ring_packets)
            take = cand[:room]
            if len(take) < len(cand):
                self.ring_dropped += int(count[cand[room:]].sum())
            if len(take):
                taken_msgs = int(count[take].sum())
                self._ring.append(_Burst(slots, header_d, take, sport[take], recv_ns, taken_msgs))
                self._ring_packets += len(take)
                self._ring_msgs += taken_msgs
                self._ring_peak = max(self._ring_peak, self._ring_packets)

    def _account_slow(self, sess, seq_c, cnt_c):
        for s, q, c in zip(map(tuple, sess.tolist()), seq_c.tolist(), cnt_c.tolist()):
            if s != self._session:
                self._session, self._expected_seq = s, None
            if self._expected_seq is not None and q > self._expected_seq:
                self.dropped += q - self._expected_seq
            if self._expected_seq is not None and q < self._expected_seq:
                self.out_of_order_packets += 1
            self._expected_seq = max(self._expected_seq or q, q + c)

    def _pull_native(self):
        """Account every burst the native drain has gathered since the last call."""
        native, ring = self._native, self._native_ring
        now = time.monotonic_ns()
        if now - self._native_checked_ns > 100_000_000:
            self._native_checked_ns = now
            st = native.stats()
            self.drain_host_ns = st["host_ns"]
            self.stalls["native"] = {
                k: st[k] for k in ("poll_gap_max_ms", "poll_gap_at", "ring_waits", "ring_wait_ms")
            } | {k: v for k, v in st.items() if k.startswith("poll_gaps_")}
            if st["error"]:
                self.error = f"GPU MITCH RX failed: {st['error']}"
            elif not self.keep_alive and now - st["last_rx_ns"] > self.recv_timeout * 1e9:
                self._eos.set()
        for start, n, recv_ns, *marks, wait_ns in native.take().tolist():
            self._burst_gap(marks[0])
            self._timed("burst", marks, _NATIVE_STEPS, packets=n, ring_wait_ms=wait_ns / 1e6)
            slot, end = start % self.ring_size, start + n
            before = len(self._ring)
            self._account(
                ring["slots"][slot : slot + n],
                ring["header_dev"][slot : slot + n],
                ring["header_host"][slot : slot + n].T,
                recv_ns,
            )
            if len(self._ring) > before:
                self._ring[-1].start, self._ring[-1].end = start, end
            elif self._ring:
                self._ring[-1].end = end  # nothing kept: release together with the last kept
            else:
                native.release(end)

    # ---- helpers read by the app -----------------------------------------------------
    def feed_buckets(self, last=120):
        with self._lock:
            out = list(self._buckets)[-last:]
        return [[b * 100, n] for b, n in out]

    def ring_fill(self):
        with self._lock:
            return self._ring_msgs

    # ---- compute: stage bursts, GPU decode, emit device-resident batches --------------
    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):
        del op_input, context
        if self._native is not None:
            self._pull_native()
        if self.error:
            raise RuntimeError(self.error)
        staged = []
        with self._lock:
            msgs = 0
            while self._ring and msgs < self.batch_size:
                b = self._ring.popleft()
                self._ring_packets -= len(b.index)
                self._ring_msgs -= b.messages
                msgs += b.messages
                staged.append(b)
        if staged:
            import cupy as cp

            marks = [time.perf_counter_ns()]
            whole = all(len(b.index) == b.slots.shape[0] for b in staged)
            if whole and len(staged) == 1:
                slots = staged[0].slots
                lens = staged[0].header[:, 0].astype(cp.int32)
            elif (
                whole
                and staged[0].start is not None
                and all(b.start == a.start + a.slots.shape[0] for a, b in zip(staged, staged[1:]))
            ):  # consecutive native ring rows: one view, no copy
                lo = staged[0].start % self.ring_size
                hi = lo + sum(b.slots.shape[0] for b in staged)
                slots = self._native_ring["slots"][lo:hi]
                lens = self._native_ring["header_dev"][lo:hi, 0].astype(cp.int32)
            else:
                picks = [cp.asarray(b.index) for b in staged]
                slots = cp.concatenate([b.slots[i] for b, i in zip(staged, picks)])
                lens = cp.concatenate([b.header[i, 0] for b, i in zip(staged, picks)]).astype(
                    cp.int32
                )
            sport = np.concatenate([b.src_port for b in staged])
            recv = np.concatenate([np.full(len(b.index), b.recv_ns, np.int64) for b in staged])
            marks.append(time.perf_counter_ns())
            batches, decoded, malformed = self.batch_decoder(slots, lens, sport, recv)
            marks.append(time.perf_counter_ns())
            # The drain thread allocates on its own non-blocking stream: finish the
            # decode before these slot buffers can be recycled for new bursts.
            cp.cuda.get_current_stream().synchronize()
            if self._native is not None:
                self._native.release(staged[-1].end)  # decoded: the drain may reuse these slots
            marks.append(time.perf_counter_ns())
            self._timed("compute", marks, ("gather", "decode", "sync"), bursts=len(staged))
            self.messages += decoded
            self.malformed_packets += malformed
            if batches:
                op_output.emit(batches, "tick")
                return
        if self._eos.is_set():
            logger.info(
                "%s: done (pkts=%d msgs=%d sequence_gaps=%d ring_drops=%d)",
                type(self).__name__,
                self.packets,
                self.messages,
                self.dropped,
                self.ring_dropped,
            )
            self._alive.disable_tick()
        elif not staged:
            time.sleep(0.0005)


__all__ = ["GpuMitchSourceOp"]
