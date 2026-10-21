# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``UdpPacketSourceOp`` and ``DaqiriPacketSourceOp``: the host-memory baselines.

Port: ``packets`` (out, ``PacketBatch`` of up to ``batch_size`` messages). Contract:
``OnClock(poll_period_us)``. Edge: ``packets -> MitchDecodeOp.packets`` with ``lossless(16)``
in the same lane; the source is the entry of ``ingest``. A drain thread copies datagrams
into a host ring after a header peek, so decode load never drops packets at the socket or
NIC (overload is ``ring_dropped``); decode is the separate ``MitchDecodeOp``. These carry
the published host baselines (0.28M msgs/s Python decode; 4,500 packets/s through DAQIRI
host memory). Processors: ``ingest.host_receivers``. From 4.3 ``AsyncUdpSourceOp`` and
``AnoSourceOp``.
"""

from __future__ import annotations

import logging

import holoscan as hs

from ..ingest.affinity import helper_mask, normalize_cpu_list, thread_mask
from ..ingest.daqiri import LOG_LEVELS, build_daqiri_config
from ..ingest.host_receivers import DaqiriHostReceiver, UdpReceiver
from ..ref import Ref
from ..telemetry.stats import IngestStats
from ._common import PendingEmits, lane_cpus, ref_obj

logger = logging.getLogger(__name__)


class _HostPacketSourceBase(hs.Operator):
    """Ports, Params and the compute shared by both host sources."""

    packets = hs.Output("packets", hs.pyobject)

    port = hs.Param(26400, range=(1, 65535), description="UDP destination port")
    ring_size = hs.Param(65536, range=(16, 10_000_000), description="host ring, in packets")
    batch_size = hs.Param(16384, range=(1, 4_194_304), description="max messages per PacketBatch")
    poll_period_us = hs.Param(
        500, range=(20, 100_000), description="OnClock period; below 100 needs a busy-polled lane"
    )
    keep_alive = hs.Param(True, description="False: recv_timeout_s of silence ends the stream")
    recv_timeout_s = hs.Param(5.0, range=(0.1, 3600.0))
    helper_cpus = hs.Param(
        [], description="mask the receiver thread starts under; [] = the process (main thread) mask"
    )
    stats = hs.Param(Ref(), description="Ref(IngestStats); private if empty")

    def __init__(self) -> None:
        self._rx = self._stats = self._pending = None

    def setup(self, spec: hs.OperatorSpec) -> None:
        normalize_cpu_list(self.helper_cpus, name="helper_cpus")
        spec.output(self.packets)

    def contract(self) -> hs.Contract:
        return hs.Contract().trigger(hs.OnClock(period=hs.time.us(self.poll_period_us)))

    def compute(self, ctx: hs.ExecutionContext) -> None:
        pending, stats = self._pending, self._stats
        if pending:  # backlog: retry it and produce nothing new
            pending.flush()
            self._sync_stats()
            return
        if stats.error:
            raise RuntimeError(stats.error)  # FAILED run; the message reaches /snapshot
        batch = self._rx.poll(self.batch_size)
        if batch is not None:
            pending.add(self.packets, batch, on_sent=stats.count_packet_batch)
            pending.flush()
        self._sync_stats()
        if self._rx.eof and not pending and not stats.eof:
            stats.eof = True
            self._log_done(stats)

    def stop(self) -> None:
        if self._rx is not None:
            self._rx.stop()
            self._rx = None
        if self._stats is not None:
            self._stats.detach()

    def _begin(self, front_end: str) -> IngestStats:
        stats = self._stats = ref_obj(self.stats, IngestStats)
        stats.reset(front_end=front_end)
        stats.lane_cpus = lane_cpus()
        self._pending = PendingEmits()
        return stats

    def _open(self, rx: UdpReceiver | DaqiriHostReceiver) -> None:
        self._rx = rx
        self._stats.attach(rx.probe())
        self._stats.opened = True

    def _helper_cpus(self) -> list[int]:
        return normalize_cpu_list(self.helper_cpus, name="helper_cpus")

    def _sync_stats(self) -> None:
        self._stats.retries = self._pending.retries
        self._stats.held = len(self._pending)

    def _log_done(self, stats: IngestStats) -> None:
        logger.info(
            "%s: done (pkts=%d sequence_gaps=%d ring_drops=%d)",
            type(self).__name__,
            stats.packets,
            stats.dropped,
            stats.ring_dropped,
        )


class UdpPacketSourceOp(_HostPacketSourceBase):
    """Kernel UDP socket receive (thread ``udp-drain``); runs anywhere."""

    host = hs.Param("127.0.0.1", description="address to bind")
    recv_buf = hs.Param(1 << 26, range=(65536, 1 << 30), description="SO_RCVBUF bytes")

    def start(self) -> None:
        stats = self._begin("socket")
        rx = UdpReceiver(
            host=self.host,
            port=self.port,
            recv_buf=self.recv_buf,
            ring_size=self.ring_size,
            keep_alive=self.keep_alive,
            recv_timeout_s=self.recv_timeout_s,
            stats=stats,
        )
        # Never under the lane mask, which may be narrowed or busy-polled (c3).
        with thread_mask(helper_mask(self._helper_cpus())):
            rx.start()
        self._open(rx)
        logger.info("UDP source draining %s:%d (ring=%d)", self.host, self.port, self.ring_size)

    def _log_done(self, stats: IngestStats) -> None:
        if stats.packets == 0:
            logger.warning(
                "no market data arrived on %s:%d within %.0fs; start the sender "
                "(python -m holoscan_finance.tools.replay_sender) against this port",
                self.host,
                self.port,
                self.recv_timeout_s,
            )
        super()._log_done(stats)


class DaqiriPacketSourceOp(_HostPacketSourceBase):
    """DAQIRI receive into pinned host memory (thread ``daqiri-host-drain``): the NIC DMAs
    flow-matched frames to the host and the drain copies each one out (4.3 ``ano``)."""

    pcie_address = hs.Param("0004:03:00.0", description="NIC PCIe address")
    daqiri_config = hs.Param(
        "", description="DAQIRI YAML path; overrides the built config and disables isolate_cores"
    )
    interface_name = hs.Param("rx_port")
    queue_id = hs.Param(0, range=(0, 63))
    master_core = hs.Param(3, range=(0, 1023), description="DPDK master lcore")
    rx_core = hs.Param(9, range=(0, 1023), description="DPDK RX lcore")
    num_bufs = hs.Param(51200, range=(1024, 1_048_576), description="host receive buffers")
    buf_size = hs.Param(2048, range=(64, 2048))
    rx_batch_size = hs.Param(1024, range=(1, 65536), description="DAQIRI burst size")
    rx_timeout_us = hs.Param(5000, range=(0, 1_000_000), description="DAQIRI partial-burst timeout")
    daqiri_log_level = hs.Param("warn", allowed=LOG_LEVELS)
    isolate_cores = hs.Param(
        False, description="move this process's threads off the lcores (4.3 ano never did)"
    )
    init_cpus = hs.Param(
        [], description="mask to widen to around daqiri_init; the lane mask is always restored"
    )

    def setup(self, spec: hs.OperatorSpec) -> None:
        if not self.pcie_address:
            raise ValueError("pcie_address must not be empty")
        normalize_cpu_list(self.init_cpus, name="init_cpus")
        super().setup(spec)

    def start(self) -> None:
        stats = self._begin("daqiri-host")
        config = self.daqiri_config or build_daqiri_config(
            pcie_address=self.pcie_address,
            udp_port=self.port,
            master_core=self.master_core,
            rx_core=self.rx_core,
            num_bufs=self.num_bufs,
            buf_size=self.buf_size,
            rx_batch_size=self.rx_batch_size,
            rx_timeout_us=self.rx_timeout_us,
            memory_kind="host_pinned",
            interface_name=self.interface_name,
            queue_id=self.queue_id,
            log_level=self.daqiri_log_level,
        )
        rx = DaqiriHostReceiver(
            config=config,
            interface_name=self.interface_name,
            queue_id=self.queue_id,
            ring_size=self.ring_size,
            keep_alive=self.keep_alive,
            recv_timeout_s=self.recv_timeout_s,
            stats=stats,
            # A custom config names its own lcores; only the built one's are known here.
            reserved=() if self.daqiri_config else (self.master_core, self.rx_core),
            isolate_cores=self.isolate_cores,
            init_cpus=normalize_cpu_list(self.init_cpus, name="init_cpus"),
            helper_cpus=self._helper_cpus(),
        )
        rx.start()
        self._open(rx)
        logger.info(
            "DAQIRI host source draining %s (%s) q%d, flow udp_dst=%d (ring=%d)",
            self.interface_name,
            self.pcie_address,
            self.queue_id,
            self.port,
            self.ring_size,
        )


__all__ = ["DaqiriPacketSourceOp", "UdpPacketSourceOp"]
