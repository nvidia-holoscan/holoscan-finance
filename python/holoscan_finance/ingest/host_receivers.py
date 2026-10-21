# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-memory receivers behind ``UdpPacketSourceOp`` and ``DaqiriPacketSourceOp``.

A drain thread copies every datagram into a ring of ``ring_size`` packets after only a
header peek (``ReceiveAccounting.observe_packet``), so decode load can never make the
kernel or the NIC drop packets; overload is counted as ``ring_dropped`` instead. The
source's lane takes ``PacketBatch``es of up to ``batch_size`` messages out of the ring, and
``MitchDecodeOp`` decodes them. These are the front ends of the host (CPU) path.
"""

from __future__ import annotations

import logging
import socket
import struct
import threading
import time
from collections import deque
from collections.abc import Iterable, Sequence

from ..marketdata.framing import strip_udp_datagram
from ..payloads import PacketBatch
from ..telemetry.stats import IngestStats, SourceProbe
from .accounting import ReceiveAccounting, mitch_packet_header
from .affinity import helper_mask, keep_off_cores, thread_mask
from .daqiri import require_daqiri

logger = logging.getLogger(__name__)

_POLL_S = 0.2  # socket timeout: stop() and the idle rule are checked between receives


class _HostReceiver:
    """The receive ring shared by both receivers; one drain thread fills it."""

    _thread_name = ""
    _error_prefix = ""

    def __init__(
        self, *, ring_size: int, keep_alive: bool, recv_timeout_s: float, stats: IngestStats
    ) -> None:
        self._stats = stats
        self._accounting = ReceiveAccounting(stats)
        self._ring: deque[tuple] = deque()  # (datagram, recv_ns, count, peer), under stats.lock
        self._ring_size = ring_size
        self._keep_alive, self._idle_ns = keep_alive, int(recv_timeout_s * 1e9)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._ended = False
        self._batches = 0

    @property
    def eof(self) -> bool:
        """The drain ended (``keep_alive`` off: ``recv_timeout_s`` of silence) and the ring is
        empty."""
        return self._ended and not self._ring

    def poll(self, max_messages: int) -> PacketBatch | None:
        """The oldest packets in the ring, up to ``max_messages`` messages (at least one
        packet); None when the ring is empty."""
        stats = self._stats
        items = []
        msgs = 0
        with stats.lock:
            while self._ring and msgs < max_messages:
                item = self._ring.popleft()
                msgs += item[2]
                items.append(item)
            stats.ring_packets -= len(items)
            stats.ring_msgs -= msgs
        if not items:
            return None
        self._batches += 1
        return PacketBatch(tuple(items), msgs, self._batches - 1)

    def probe(self) -> SourceProbe:
        stats = self._stats

        def queued() -> int:
            with stats.lock:
                return stats.ring_packets

        return SourceProbe(
            alive=lambda: self._thread is not None and self._thread.is_alive(),
            pending=queued,
            ring_peak=lambda: stats.ring_peak_local,
            reset_peak=lambda: None,  # reset_stalls() zeroes ring_peak_local itself
        )

    def _start_thread(self) -> None:
        self._stop.clear()
        self._ended = False
        self._thread = threading.Thread(target=self._run, name=self._thread_name, daemon=True)
        self._thread.start()

    def _join(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        try:
            self._receive()
        except Exception as exc:
            if not self._stop.is_set():
                self._stats.error = f"{self._error_prefix}: {exc}"
                logger.exception(self._stats.error)
        finally:
            self._ended = True

    def _receive(self) -> None:
        raise NotImplementedError

    def _enqueue(self, data: bytes, recv_ns: int, peer: tuple[str, int] | None) -> None:
        """Account one datagram and admit it to the ring."""
        stats = self._stats
        try:
            session, seq, count = mitch_packet_header(data, peer)
        except (ValueError, struct.error):
            stats.malformed_packets += 1
            return
        self._accounting.observe_packet(session, seq, count, recv_ns)
        with stats.lock:
            if len(self._ring) < self._ring_size:
                self._ring.append((data, recv_ns, count, peer))
                stats.ring_packets += 1
                stats.ring_msgs += count
                stats.ring_peak_local = max(stats.ring_peak_local, stats.ring_packets)
            else:
                stats.ring_dropped += count


class UdpReceiver(_HostReceiver):
    """Kernel UDP socket receive (thread ``udp-drain``; ``recvfrom`` releases the GIL)."""

    _thread_name = "udp-drain"
    _error_prefix = "UDP RX failed"

    def __init__(
        self,
        *,
        host: str,
        port: int,
        recv_buf: int,
        ring_size: int,
        keep_alive: bool,
        recv_timeout_s: float,
        stats: IngestStats,
    ) -> None:
        super().__init__(
            ring_size=ring_size, keep_alive=keep_alive, recv_timeout_s=recv_timeout_s, stats=stats
        )
        self._address, self._recv_buf = (host, port), recv_buf
        self._sock: socket.socket | None = None

    def start(self) -> None:
        """Bind the socket (raises on a bind error) and start the drain thread, which
        inherits the caller's thread mask."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self._recv_buf)
            except OSError:
                pass  # capped by net.core.rmem_max; the default buffer still works
            sock.bind(self._address)
            sock.settimeout(_POLL_S)
        except BaseException:
            sock.close()
            raise
        self._sock = sock
        self._start_thread()

    def stop(self) -> None:
        self._join()
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def _receive(self) -> None:
        sock = self._sock
        last_rx = time.monotonic_ns()
        while not self._stop.is_set():
            try:
                data, peer = sock.recvfrom(65535)
            except TimeoutError:
                if not self._keep_alive and time.monotonic_ns() - last_rx > self._idle_ns:
                    return  # feed idle: end of input
                continue
            last_rx = time.monotonic_ns()
            self._enqueue(data, time.time_ns(), peer)


class DaqiriHostReceiver(_HostReceiver):
    """DAQIRI receive into pinned host memory (thread ``daqiri-host-drain``): per packet one
    copy out of the NIC buffer and a header peek; every burst goes back to DAQIRI at once,
    also when a read fails. One DAQIRI engine may be active per process."""

    _thread_name = "daqiri-host-drain"
    _error_prefix = "DAQIRI RX failed"

    def __init__(
        self,
        *,
        config: dict | str,
        interface_name: str,
        queue_id: int,
        ring_size: int,
        keep_alive: bool,
        recv_timeout_s: float,
        stats: IngestStats,
        reserved: Iterable[int] = (),
        isolate_cores: bool = False,
        init_cpus: Sequence[int] = (),
        helper_cpus: Sequence[int] = (),
    ) -> None:
        super().__init__(
            ring_size=ring_size, keep_alive=keep_alive, recv_timeout_s=recv_timeout_s, stats=stats
        )
        self._config, self._interface_name, self._queue_id = config, interface_name, queue_id
        self._reserved, self._isolate_cores = tuple(reserved), isolate_cores
        self._init_cpus, self._helper_cpus = list(init_cpus), list(helper_cpus)
        self._daq = None
        self._port_id = -1

    def start(self) -> None:
        """Initialize DAQIRI and start the drain thread; a successful init is shut down again
        if a later step fails."""
        daq = require_daqiri(gpu=False)
        status = None
        try:
            # EAL pins the calling thread, and the runtime never re-applies a lane's placement:
            # thread_mask restores (and checks) the lane mask even when nothing is widened.
            with thread_mask(self._init_cpus or None):
                status = daq.daqiri_init(self._config)
            if status != daq.Status.SUCCESS:
                raise RuntimeError(f"daqiri_init failed: {status}")
            if self._isolate_cores:
                keep_off_cores(self._reserved)
            port_id = daq.get_port_id(self._interface_name)
            if port_id < 0:
                raise RuntimeError(f"DAQIRI interface {self._interface_name!r} not found")
            self._daq, self._port_id = daq, port_id
            # Never under the lane mask: it may be narrowed or busy-polled.
            with thread_mask(helper_mask(self._helper_cpus)):
                self._start_thread()
        except BaseException:
            # Undo a successful init (also when the mask check failed after it); a started
            # drain polls DAQIRI, so it stops first.
            self._join()
            self._daq = None
            if status == daq.Status.SUCCESS:
                daq.shutdown()
            raise

    def stop(self) -> None:
        self._join()
        if self._daq is not None:
            self._daq.print_stats()
            self._daq.shutdown()
            self._daq = None

    def _receive(self) -> None:
        daq = self._daq
        ok = daq.Status.SUCCESS
        idle_statuses = {daq.Status.NULL_PTR, getattr(daq.Status, "NOT_READY", None)}
        last_rx = time.monotonic_ns()
        while not self._stop.is_set():
            status, burst = daq.get_rx_burst(self._port_id, self._queue_id)
            if status != ok or burst is None:
                if status not in idle_statuses:
                    raise RuntimeError(
                        f"get_rx_burst returned {status} (burst={burst is not None})"
                    )
                if not self._keep_alive and time.monotonic_ns() - last_rx > self._idle_ns:
                    return  # feed idle: end of input
                time.sleep(0.0002)
                continue
            last_rx = time.monotonic_ns()
            recv_ns = time.time_ns()
            try:
                for idx in range(daq.get_num_packets(burst)):
                    st, frame = daq.get_packet_bytes(burst, idx)
                    if st != ok:
                        raise RuntimeError(f"get_packet_bytes returned {st}")
                    datagram = strip_udp_datagram(frame, minimum_payload=8)
                    if datagram is not None:
                        self._enqueue(datagram[0], recv_ns, datagram[1])
            finally:
                daq.free_all_packets_and_burst_rx(burst)


__all__ = ["DaqiriHostReceiver", "UdpReceiver"]
