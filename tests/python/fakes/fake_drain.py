# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``FakeNativeDrain``: the native ``Drain`` of ``daqiri_drain.cpp`` in Python.

Same constructor keywords, methods, record fields and ``stats()`` keys, ``poll_gaps_*``
included. Used by hand (no thread), a test fills the ring itself and calls ``publish`` for
each burst, as the native thread does after a gather. Started by an operator (patch
``load_daqiri_drain`` to return this class), a thread ``hf-drain`` polls the (fake)
``daqiri`` module, gathers every burst into the ring through the raw addresses it was
given with the real ``gather_frame`` kernel, holds a burst while the ring is full and
publishes one record per burst. ``fail`` reports a drain error; ``released`` logs every
``release``. ``receive_header`` builds the receive header the gather kernel reports.
"""

from __future__ import annotations

import ctypes
import importlib
import threading
import time
from typing import ClassVar

import numpy as np

from holoscan_finance.ingest.drains import DRAIN_FIELDS
from holoscan_finance.marketdata.gpu_sbe import HEADER_COLS, PACKET_HEADER
from holoscan_finance.marketdata.sbe_codec import frame_fields


def receive_header(packets: list[tuple[bytes | None, int, int]]) -> np.ndarray:
    """What ``gather_frame`` reports for each ``(UDP payload | None, source port, source
    ip)``, as a ``HEADER_COLS x n`` header: length, port, ip, framing status, entry count,
    sequence."""
    rows = []
    for data, port, ip in packets:
        if data is None:  # not IPv4/UDP
            rows.append([-1, -1, 0, 1, 0, 0])
            continue
        ok, seq, count = frame_fields(data)
        seq = int(np.array(seq, np.uint64).view(np.int64))  # as the kernel reports it
        rows.append([len(data), port, ip, 0 if ok else 1, count, seq])
    return np.asarray(rows, np.int64).reshape(-1, HEADER_COLS).T


class FakeNativeDrain:
    """Stands in for the native ``Drain``; see the module docstring."""

    instances: ClassVar[list[FakeNativeDrain]] = []  # every drain an operator created

    def __init__(
        self,
        *,
        capacity: int,
        port: int = 0,
        queue: int = 0,
        gather_fn: int = 0,
        device: int = 0,
        slots: int = 0,
        header_dev: int = 0,
        header_host: int = 0,
        stride: int = 2048,
        packed_dev: int = 0,
        packed_host: int = 0,
        max_burst: int = 65536,
        min_payload: int = PACKET_HEADER,
        max_records: int = 65536,
        cpu: int = -1,
    ) -> None:
        self.cpu = cpu
        self.port, self.queue, self.device = port, queue, device
        self.capacity, self.stride, self.min_payload = capacity, stride, min_payload
        self._addresses = dict(
            slots=slots, header_dev=header_dev, header_host=header_host, packed_dev=packed_dev
        )
        self.max_burst = min(max_burst, capacity)
        self.gather_fn = gather_fn
        self.released: list[int] = []
        self.poll_gaps = {"poll_gaps_10ms": 0, "poll_gaps_50ms": 0, "poll_gaps_100ms": 0}
        self.running = False
        self.error = ""
        self._lock = threading.Lock()
        self._records: list[tuple[int, ...]] = []
        self._head = self._tail = self._peak = self._bursts = self._packets = 0
        self._last_rx_ns = time.monotonic_ns()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        FakeNativeDrain.instances.append(self)

    # ---- the native Drain API ------------------------------------------------------------
    def start(self) -> None:
        self.running = True
        self._thread = threading.Thread(target=self._run, name="hf-drain", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def take(self) -> np.ndarray:
        with self._lock:
            records, self._records = self._records, []
        return np.array(records, np.int64).reshape(-1, len(DRAIN_FIELDS))

    def release(self, end: int) -> None:
        self.released.append(end)
        self._tail = max(self._tail, end)

    def reset_peak(self) -> None:
        self._peak = 0

    def stats(self) -> dict:
        return {
            "running": self.running,
            "bursts": self._bursts,
            "packets": self._packets,
            "ring_peak": self._peak,
            "fill": self._head - self._tail,
            "poll_gap_max_ms": 0.0,
            "poll_gap_at": 0,
            **self.poll_gaps,
            "ring_wait_ms": 0.0,
            "ring_waits": 0,
            "last_rx_ns": self._last_rx_ns,
            "host_ns": 0,
            "error": self.error,
        }

    # ---- test controls ---------------------------------------------------------------------
    def publish(self, start: int, n: int, recv_ns: int) -> None:
        """One gathered burst at ring position ``start``, as the native thread reports it."""
        t = time.monotonic_ns()
        with self._lock:
            self._records.append((start, n, recv_ns, t, t, t, t, t, 0))
        self._head = start + n
        self._peak = max(self._peak, self._head - self._tail)
        self._bursts += 1
        self._packets += n
        self._last_rx_ns = t

    def fail(self, message: str) -> None:
        self.error = message

    # ---- the thread: daqiri_drain.cpp's loop ----------------------------------------------
    def _run(self) -> None:
        try:
            self._loop()
        except Exception as exc:  # as the native drain: an error the source reports
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            self.running = False

    def _loop(self) -> None:
        import cupy as cp

        from holoscan_finance.marketdata.gpu_sbe import _kernels

        daq = importlib.import_module("daqiri")
        cp.cuda.Device(self.device).use()
        stream = cp.cuda.Stream(non_blocking=True)
        slots = self._device_view("slots", (self.capacity, self.stride), cp.uint8)
        header_dev = self._device_view("header_dev", (self.capacity, HEADER_COLS), cp.int64)
        packed_dev = self._device_view("packed_dev", (2 * self.max_burst,), cp.uint64)
        header_host = np.ctypeslib.as_array(
            (ctypes.c_int64 * (self.capacity * HEADER_COLS)).from_address(
                self._addresses["header_host"]
            )
        ).reshape(self.capacity, HEADER_COLS)
        gather = _kernels().get_function("gather_frame")
        while not self._stop.is_set():
            status, burst = daq.get_rx_burst(self.port, self.queue)
            if status != daq.Status.SUCCESS or burst is None:
                time.sleep(0.0005)
                continue
            recv_ns = time.time_ns()
            n = daq.get_num_packets(burst)
            pos = self._head
            if pos % self.capacity + n > self.capacity:  # a burst never wraps
                pos += self.capacity - pos % self.capacity
            while pos + n - self._tail > self.capacity:  # hold the burst while the ring is full
                if self._stop.is_set():
                    daq.free_all_packets_and_burst_rx(burst)
                    return
                time.sleep(0.0005)
            slot = pos % self.capacity
            packed = [daq.get_packet_ptr(burst, i) for i in range(n)]
            packed += [daq.get_packet_length(burst, i) for i in range(n)]
            with stream:
                packed_dev[: 2 * n].set(np.asarray(packed, np.uint64), stream=stream)
                rows = (
                    packed_dev[: 2 * n],
                    np.int32(n),
                    np.int32(self.min_payload),
                    slots[slot : slot + n],
                    np.int32(self.stride),
                    header_dev[slot : slot + n],
                )
                gather((n,), (128,), rows, stream=stream)
                header_host[slot : slot + n] = header_dev[slot : slot + n].get(stream=stream)
            stream.synchronize()
            daq.free_all_packets_and_burst_rx(burst)
            self.publish(pos, n, recv_ns)

    def _device_view(self, name: str, shape: tuple[int, ...], dtype: type):
        import cupy as cp

        nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        memory = cp.cuda.UnownedMemory(self._addresses[name], nbytes, owner=self)
        return cp.ndarray(shape, dtype, cp.cuda.MemoryPointer(memory, 0))


__all__ = ["FakeNativeDrain", "receive_header"]
