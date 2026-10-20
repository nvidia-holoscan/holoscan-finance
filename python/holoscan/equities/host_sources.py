# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-memory receive (``--ingest async`` and ``--ingest ano``): the baselines for
:class:`holoscan.marketdata.GpuMitchSourceOp`.

Both copy every packet into a ring on the CPU and decode MITCH in Python.
:class:`AsyncUdpSourceOp` uses a kernel UDP socket and runs anywhere;
:class:`AnoSourceOp` receives through DAQIRI into pinned host memory.
"""

from __future__ import annotations

import logging
import socket
import struct
import threading
import time
from collections import deque

from holoscan.conditions import BooleanCondition, PeriodicCondition
from holoscan.core import (
    ExecutionContext,
    Fragment,
    InputContext,
    Operator,
    OperatorSpec,
    OutputContext,
)
from holoscan.marketdata.daqiri import build_daqiri_config, require_daqiri, strip_udp_datagram
from holoscan.marketdata.mitch import inspect_unit

logger = logging.getLogger(__name__)


class AsyncUdpSourceOp(Operator):
    """A drain thread moves datagrams from the socket into a ring; ``compute()``
    decodes them and emits a list of batches on port ``tick``.

    Draining does only a header peek, so decoding load cannot make the kernel drop
    packets; overload is counted as ``ring_dropped`` instead.

    ``packet_header(data, peer) -> (session, sequence, count)`` frames a datagram for
    the receive accounting. ``datagram_decoder(data, recv_ns, peer) -> (batches,
    count)`` decodes it in ``compute()``, which takes up to ``batch_size`` messages
    from the ring per call. Counters: ``dropped`` (sequence gaps), ``ring_dropped``,
    ``malformed_packets``, ``out_of_order_packets``, ``packets``,
    ``received_messages`` and ``messages`` (decoded). With ``keep_alive`` the source
    keeps listening through idle periods; otherwise ``recv_timeout`` of silence ends it.
    """

    def __init__(
        self,
        fragment: Fragment,
        *args,
        packet_header,
        datagram_decoder,
        host: str = "127.0.0.1",
        port: int = 26400,
        recv_buf: int = 1 << 26,
        recv_timeout: float = 5.0,
        ring_size: int = 2_000_000,
        poll_period_ns: int = 500_000,
        batch_size: int = 16384,
        keep_alive: bool = False,
        name: str = "async_udp_source",
        **kwargs,
    ) -> None:
        self.host, self.port = host, int(port)
        self.recv_buf, self.recv_timeout = int(recv_buf), float(recv_timeout)
        self.ring_size, self.batch_size = int(ring_size), int(batch_size)
        self.keep_alive = bool(keep_alive)
        self.packet_header = packet_header
        self.datagram_decoder = datagram_decoder
        # 100 ms arrival buckets (message counts), measured on the drain thread.
        self._bucket_ns = 100_000_000
        self._buckets: deque = deque(maxlen=600)
        self._ring: deque = deque()  # (data, recv_ns, count, peer)
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._eos = threading.Event()
        self._sock: socket.socket | None = None
        self._reset()
        self._alive = BooleanCondition(fragment, name=name + "_alive", enable_tick=True)
        # Keeps the scheduler ticking this source while the ring is empty; an
        # all-idle graph would otherwise be treated as a deadlock.
        self._tick = PeriodicCondition(
            fragment, recess_period=int(poll_period_ns), name=name + "_tick"
        )
        super().__init__(fragment, self._alive, self._tick, *args, name=name, **kwargs)

    def _reset(self) -> None:
        self._ring.clear()
        self._ring_msgs = 0
        self._buckets.clear()
        self._expected_seq: int | None = None
        self._session = None
        self.packets = self.messages = self.dropped = self.ring_dropped = 0
        self.received_messages = self.malformed_packets = self.out_of_order_packets = 0
        self.last_packet_wall_ns = 0
        self._stop.clear()
        self._eos.clear()

    def setup(self, spec: OperatorSpec) -> None:
        spec.output("tick")

    def start(self) -> None:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.recv_buf)
        except OSError:
            pass
        s.bind((self.host, self.port))
        s.settimeout(min(self.recv_timeout, 0.5) if self.keep_alive else self.recv_timeout)
        self._sock = s
        self._reset()
        self._thread = threading.Thread(target=self._drain, daemon=True)
        self._thread.start()
        logger.info(
            "async UDP source draining %s:%d (ring=%d)", self.host, self.port, self.ring_size
        )

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def _drain(self) -> None:
        while not self._stop.is_set():
            try:
                data, peer = self._sock.recvfrom(65535)
            except TimeoutError:
                if self.keep_alive:
                    continue
                self._eos.set()  # feed idle -> treat as end
                return
            except OSError:
                return
            self._enqueue_packet(data, time.time_ns(), peer)

    def _enqueue_packet(self, data: bytes, recv_ns: int, peer=None) -> None:
        """Receive accounting shared by the socket and DAQIRI drains."""
        try:
            session, seq, count = self.packet_header(data, peer)
        except (ValueError, struct.error):
            self.malformed_packets += 1
            return
        if session != self._session:
            self._session = session
            self._expected_seq = None
        self.last_packet_wall_ns = recv_ns
        if self._expected_seq is not None and seq > self._expected_seq:
            self.dropped += seq - self._expected_seq
        if self._expected_seq is not None and seq < self._expected_seq:
            self.out_of_order_packets += 1
        self._expected_seq = max(self._expected_seq or seq, seq + count)
        self.packets += 1
        self.received_messages += count
        b = recv_ns // self._bucket_ns
        with self._lock:
            if self._buckets and self._buckets[-1][0] == b:
                self._buckets[-1][1] += count
            else:
                self._buckets.append([b, count])
            if len(self._ring) < self.ring_size:
                self._ring.append((data, recv_ns, count, peer))
                self._ring_msgs += count
            else:
                self.ring_dropped += count

    def feed_buckets(self, last: int = 120) -> list[list[int]]:
        """Last ``last`` 100 ms arrival buckets as ``[epoch_ms, count]``."""
        with self._lock:
            out = list(self._buckets)[-last:]
        return [[b * 100, n] for b, n in out]

    def ring_fill(self) -> int:
        with self._lock:
            return self._ring_msgs

    def compute(
        self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext
    ) -> None:
        del op_input, context
        raws: list = []
        with self._lock:
            staged = 0
            while self._ring and staged < self.batch_size:
                pkt = self._ring.popleft()
                self._ring_msgs -= pkt[2]
                staged += pkt[2]
                raws.append(pkt)
        batch: list = []
        for data, recv_ns, _count, peer in raws:
            try:
                out, count = self.datagram_decoder(data, recv_ns, peer)
            except (ValueError, struct.error):
                self.malformed_packets += 1
                continue
            self.messages += count
            batch.extend(out)
        if batch:
            op_output.emit(batch, "tick")
        elif self._eos.is_set():
            if self.packets == 0:
                logger.warning(
                    "no market data arrived on %s:%d within %.0fs; start the sender "
                    "(python -m holoscan.equities.replay) against this port",
                    self.host,
                    self.port,
                    self.recv_timeout,
                )
            logger.info(
                "%s: done (pkts=%d msgs=%d sequence_gaps=%d ring_drops=%d)",
                type(self).__name__,
                self.packets,
                self.messages,
                self.dropped,
                self.ring_dropped,
            )
            self._alive.disable_tick()
        else:
            time.sleep(0.0005)  # idle; the drain thread is filling the ring


class AnoSourceOp(AsyncUdpSourceOp):
    """DAQIRI-backed receive (the ``--ingest ano`` path).

    The drain thread polls DAQIRI RX bursts instead of a socket; the NIC DMAs
    flow-matched packets into pinned host memory. Port, counters and
    ``feed_buckets()``/``ring_fill()`` are inherited unchanged. ``daqiri`` is
    imported in ``start()`` so CPU-only tests do not need it; one DAQIRI engine
    may be active per process.
    """

    def __init__(
        self,
        fragment,
        *args,
        port: int = 26400,
        pcie_address: str = "0004:03:00.0",
        daqiri_config: str | dict | None = None,
        interface_name: str = "rx_port",
        queue_id: int = 0,
        master_core: int = 3,
        rx_core: int = 9,
        num_bufs: int = 51200,
        buf_size: int = 2048,
        rx_batch_size: int = 1024,
        rx_timeout_us: int = 5000,
        daqiri_log_level: str = "warn",
        name: str = "ano_source",
        **kwargs,
    ) -> None:
        self.pcie_address = pcie_address
        self.interface_name = interface_name
        self.queue_id = int(queue_id)
        self._daqiri_config = daqiri_config or build_daqiri_config(
            pcie_address=pcie_address,
            udp_dst_port=int(port),
            interface_name=interface_name,
            queue_id=int(queue_id),
            master_core=int(master_core),
            rx_core=int(rx_core),
            num_bufs=int(num_bufs),
            buf_size=int(buf_size),
            rx_batch_size=int(rx_batch_size),
            rx_timeout_us=int(rx_timeout_us),
            log_level=daqiri_log_level,
        )
        self._daq = None  # the daqiri module once the engine is live
        self.error = None
        self._port_id = -1
        super().__init__(fragment, *args, port=port, name=name, **kwargs)

    def start(self) -> None:
        daqiri = require_daqiri()
        self.error = None
        self._reset()
        status = daqiri.daqiri_init(self._daqiri_config)
        if status != daqiri.Status.SUCCESS:
            raise RuntimeError(f"daqiri_init failed: {status}")
        self._daq = daqiri
        self._port_id = daqiri.get_port_id(self.interface_name)
        if self._port_id < 0:
            daqiri.shutdown()
            self._daq = None
            raise RuntimeError(f"DAQIRI interface {self.interface_name!r} not found")
        self._thread = threading.Thread(target=self._drain_checked, daemon=True)
        self._thread.start()
        logger.info(
            "ANO UDP source draining %s (%s) q%d, flow udp_dst=%d (ring=%d)",
            self.interface_name,
            self.pcie_address,
            self.queue_id,
            self.port,
            self.ring_size,
        )

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._daq is not None:
            self._daq.print_stats()
            self._daq.shutdown()
            self._daq = None

    def compute(self, op_input, op_output, context):
        if self.error:
            raise RuntimeError(self.error)
        super().compute(op_input, op_output, context)

    def _drain_checked(self):
        try:
            self._drain()
        except Exception as exc:
            self.error = f"DAQIRI RX failed: {exc}"
            logger.exception(self.error)
            self._eos.set()

    def _drain(self) -> None:
        """Poll DAQIRI RX bursts into the ring: per packet, one copy out of the NIC
        buffer and a header peek. Bursts go back to DAQIRI immediately
        (``get_packet_bytes`` copies); a slow drain can still exhaust the NIC pool.
        """
        daq = self._daq
        ok = daq.Status.SUCCESS
        idle_statuses = {daq.Status.NULL_PTR, getattr(daq.Status, "NOT_READY", None)}
        idle_limit_ns = int(self.recv_timeout * 1e9)
        last_rx_ns = time.monotonic_ns()
        while not self._stop.is_set():
            status, burst = daq.get_rx_burst(self._port_id, self.queue_id)
            if status != ok or burst is None:
                if status not in idle_statuses:
                    raise RuntimeError(
                        f"get_rx_burst returned {status} (burst={burst is not None})"
                    )
                if not self.keep_alive and time.monotonic_ns() - last_rx_ns > idle_limit_ns:
                    self._eos.set()  # feed idle -> treat as end
                    return
                time.sleep(0.0002)
                continue
            last_rx_ns = time.monotonic_ns()
            recv_ns = time.time_ns()
            try:
                for idx in range(daq.get_num_packets(burst)):
                    st, frame = daq.get_packet_bytes(burst, idx)
                    if st != ok:
                        raise RuntimeError(f"get_packet_bytes returned {st}")
                    datagram = strip_udp_datagram(frame, minimum_payload=8)
                    if datagram is not None:
                        data, peer = datagram
                        self._enqueue_packet(data, recv_ns, peer)
            finally:
                daq.free_all_packets_and_burst_rx(burst)


def packet_header(data, peer):
    """MITCH framing for receive accounting. Sessions are keyed by sender ``(ip, port)``
    and market-data group, so packets from an earlier replay generation are not read as
    the current one; sequences count wire messages, Time messages included."""
    group, sequence, count = inspect_unit(data)
    return (peer, group), sequence, count


class AsyncUdpMitchSourceOp(AsyncUdpSourceOp):
    def __init__(self, fragment, *args, datagram_decoder, name="async_udp_mitch_source", **kwargs):
        super().__init__(
            fragment,
            *args,
            packet_header=packet_header,
            datagram_decoder=datagram_decoder,
            name=name,
            **kwargs,
        )


class AnoMitchSourceOp(AnoSourceOp):
    def __init__(self, fragment, *args, datagram_decoder, name="ano_mitch_source", **kwargs):
        super().__init__(
            fragment,
            *args,
            packet_header=packet_header,
            datagram_decoder=datagram_decoder,
            name=name,
            **kwargs,
        )


__all__ = [
    "AnoMitchSourceOp",
    "AnoSourceOp",
    "AsyncUdpMitchSourceOp",
    "AsyncUdpSourceOp",
    "packet_header",
]
