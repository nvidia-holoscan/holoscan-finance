# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``GpuMitchSourceOp`` and ``GpuMitchReplaySourceOp``: GPU MITCH sources with fused decode.

Port: ``events`` (out, ``list[GpuReplayBatch]``, always one batch with device-resident
columns). Contract: ``OnClock(poll_period_us)``. Edge: ``events -> QuoteBarsOp(backend="gpu")``
with ``lossless(4)`` in the same lane: the source is the entry of ``ingest`` and shares it
with the bars, so no per-packet work crosses a lane. Decode is fused because ring rows stay
leased until decode synchronizes; the lease has one consumer by construction.
``GpuMitchSourceOp`` receives through DAQIRI GPUDirect and the GIL-free native drain
(GPU_INGEST: lossless at 200k packets/s, 6.6M msgs/s); ``GpuMitchReplaySourceOp`` replays a
pcap from GPU memory through the same gather, ring, accounting and decode, with no NIC.
Processor: ``ingest.gpu_ingest.GpuIngest``. From 4.3 ``gpu_mitch_source_op``.
"""

from __future__ import annotations

import logging
from typing import Any

import holoscan as hs

from ..ingest.affinity import helper_mask, keep_off_cores, normalize_cpu_list, thread_mask
from ..ingest.daqiri import LOG_LEVELS, build_daqiri_config, require_daqiri
from ..ingest.drains import DeviceFrames, DeviceReplayDrain, DeviceRing, PythonGpuDrain
from ..ingest.gpu_ingest import GpuIngest
from ..ingest.native import load_daqiri_drain
from ..marketdata import gpu_mitch
from ..marketdata.framing import PcapFrames
from ..marketdata.gpu_decoder import GpuMitchDecoder
from ..marketdata.warmup import warm_decode_kernels
from ..ref import Ref
from ..telemetry.stats import IngestStats
from ._common import PendingEmits, device_scope, lane_cpus, ref_obj

logger = logging.getLogger(__name__)


def _gather_kernel() -> Any:
    """The ``gather_frame`` RawKernel, compiled on the calling (lane) thread, not in a drain."""
    return gpu_mitch._kernels().get_function("gather_frame")


class _GpuMitchSourceBase(hs.Operator):
    """Ports, Params and the compute shared by both GPU sources."""

    events = hs.Output("events", hs.pyobject)

    batch_size = hs.Param(131072, range=(1, 4_194_304), description="max MITCH messages per batch")
    ring_size = hs.Param(1 << 20, range=(4096, 4_194_304), description="device slots (2 KiB each)")
    poll_period_us = hs.Param(
        500, range=(20, 100_000), description="OnClock period; below 100 needs a busy-polled lane"
    )
    gpu_device = hs.Param(0, range=(0, 63))
    recv_timeout_s = hs.Param(5.0, range=(0.1, 3600.0), description="idle time that ends input")
    keep_alive = hs.Param(True, description="False: the end of input ends the stream (stats.eof)")
    warmup = hs.Param(True, description="compile and run the gather and decode kernels in start()")
    helper_cpus = hs.Param(
        [], description="mask the drain thread starts under; [] = the process (main thread) mask"
    )
    directory = hs.Param(
        Ref(), description="Ref(Knob[FeedDirectory]); empty: every packet is stale"
    )
    stats = hs.Param(Ref(), description="Ref(IngestStats); private if empty")
    tap = hs.Param(Ref(), description="Ref(CaptureTap); no capture recording if empty")

    def __init__(self) -> None:
        self._ingest = self._drain = self._ring = None
        self._knob = self._stats = self._tap = self._pending = None

    def setup(self, spec: hs.OperatorSpec) -> None:
        normalize_cpu_list(self.helper_cpus, name="helper_cpus")
        spec.output(self.events)

    def contract(self) -> hs.Contract:
        return hs.Contract().trigger(hs.OnClock(period=hs.time.us(self.poll_period_us)))

    def compute(self, ctx: hs.ExecutionContext) -> None:
        pending, stats, ingest = self._pending, self._stats, self._ingest
        if pending:  # backlog: retry it and produce nothing new
            pending.flush()
            self._sync_stats()
            return
        with device_scope(self.gpu_device):
            ingest.pull()
            if stats.error:
                raise RuntimeError(stats.error)  # FAILED run; the message reaches /snapshot
            batches = ingest.next_batch(self._knob.get() if self._knob is not None else None)
        if batches:
            pending.add(self.events, batches, on_sent=ingest.count_event_batch)
            pending.flush()
        self._sync_stats()
        if ingest.eof and not pending and not stats.eof:
            stats.eof = True
            logger.info(
                "%s: done (pkts=%d msgs=%d sequence_gaps=%d ring_drops=%d)",
                type(self).__name__,
                stats.packets,
                stats.messages,
                stats.dropped,
                stats.ring_dropped,
            )

    def stop(self) -> None:
        if self._drain is not None:
            self._drain.stop()  # joins the drain thread (the native one without the GIL)
        if self._stats is not None:
            self._stats.detach()
        self._ingest = self._drain = self._ring = None

    def _begin(self, front_end: str) -> IngestStats:
        """Resolve the refs, reset the stats, activate the device and warm the kernels."""
        import cupy as cp

        self._knob, self._tap = self.directory.obj, self.tap.obj
        stats = self._stats = ref_obj(self.stats, IngestStats)
        stats.reset(front_end=front_end)
        stats.lane_cpus = lane_cpus()
        self._pending = PendingEmits()
        cp.cuda.Device(self.gpu_device).use()  # persists on the lane thread
        if self.warmup:
            warm_decode_kernels(self.gpu_device)
        return stats

    def _new_ingest(self) -> GpuIngest:
        return GpuIngest(
            stats=self._stats,
            decoder=GpuMitchDecoder(self.gpu_device),
            ring_size=self.ring_size,
            batch_size=self.batch_size,
            keep_alive=self.keep_alive,
            recv_timeout_s=self.recv_timeout_s,
            tap=self._tap,
        )

    def _helper_mask(self) -> list[int]:
        """Never the lane mask, which may be narrowed or busy-polled."""
        return helper_mask(normalize_cpu_list(self.helper_cpus, name="helper_cpus"))

    def _open(self, ingest: GpuIngest, frame_fetch: str) -> None:
        stats = self._stats
        stats.frame_fetch = frame_fetch
        stats.attach(ingest.probe())
        stats.opened = True

    def _sync_stats(self) -> None:
        self._stats.retries = self._pending.retries
        self._stats.held = len(self._pending)


class GpuMitchSourceOp(_GpuMitchSourceBase):
    """DAQIRI GPUDirect receive: the NIC writes frames into GPU memory and the GIL-free
    native drain (``hf-drain``) gathers their UDP payloads into the device ring, so a Python
    thread that stalls holding the GIL cannot stop NIC buffers from being recycled. Without
    the native helper (``native_drain`` False, ``HOLOSCAN_FINANCE_NATIVE_DRAIN=0`` or a
    failed build) the Python drain ``gpu-mitch-drain`` does the same work with the GIL.
    """

    port = hs.Param(26400, range=(1, 65535), description="UDP destination port of the flow rule")
    pcie_address = hs.Param("0004:03:00.0", description="NIC PCIe address")
    daqiri_config = hs.Param(
        "", description="DAQIRI YAML path; overrides the built config and disables isolate_cores"
    )
    interface_name = hs.Param("rx_port")
    queue_id = hs.Param(0, range=(0, 63))
    master_core = hs.Param(3, range=(0, 1023), description="DPDK master lcore")
    rx_core = hs.Param(9, range=(0, 1023), description="DPDK RX lcore")
    num_bufs = hs.Param(114688, range=(1024, 1_048_576), description="GPU receive buffers (BAR1)")
    buf_size = hs.Param(2048, range=(64, 2048))
    rx_batch_size = hs.Param(4096, range=(1, 65536), description="DAQIRI burst size")
    rx_timeout_us = hs.Param(
        20000, range=(0, 1_000_000), description="DAQIRI partial-burst timeout"
    )
    daqiri_log_level = hs.Param("warn", allowed=LOG_LEVELS)
    native_drain = hs.Param(True, description="False: the Python drain (A/B switch)")
    isolate_cores = hs.Param(True, description="move this process's threads off the lcores")
    init_cpus = hs.Param(
        [], description="mask to widen to around daqiri_init; the lane mask is always restored"
    )

    def __init__(self) -> None:
        super().__init__()
        self._daq = None

    def setup(self, spec: hs.OperatorSpec) -> None:
        if not self.pcie_address:
            raise ValueError("pcie_address must not be empty")
        normalize_cpu_list(self.init_cpus, name="init_cpus")
        super().setup(spec)

    def start(self) -> None:
        stats = self._begin("gpudirect")
        daq = require_daqiri(gpu=True)
        # Built before daqiri_init: a cold build takes about 5 s, which must not pass with the
        # NIC up and nobody polling it.
        drain_cls = load_daqiri_drain() if self.native_drain else None
        ring = (
            DeviceRing.allocate(self.ring_size, self.gpu_device) if drain_cls is not None else None
        )
        status = drain = None
        try:
            # EAL pins the calling thread, and the runtime never re-applies a lane's placement:
            # thread_mask restores (and checks) the lane mask even when nothing is widened.
            with thread_mask(normalize_cpu_list(self.init_cpus, name="init_cpus") or None):
                status = daq.daqiri_init(self._daqiri_config())
            if status != daq.Status.SUCCESS:
                raise RuntimeError(f"daqiri_init failed: {status}")
            if self.isolate_cores and not self.daqiri_config:
                keep_off_cores({self.master_core, self.rx_core})
            port_id = daq.get_port_id(self.interface_name)
            if port_id < 0:
                raise RuntimeError(f"DAQIRI interface {self.interface_name!r} not found")
            ingest = self._new_ingest()
            with thread_mask(self._helper_mask()):
                if drain_cls is not None:
                    drain = drain_cls(
                        port=port_id,
                        queue=self.queue_id,
                        gather_fn=_gather_kernel().kernel.ptr,  # the CUfunction
                        device=self.gpu_device,
                        **ring.native_kwargs(),
                        stride=gpu_mitch.SLOT_BYTES,
                        min_payload=gpu_mitch.UNIT_HEADER,
                        max_records=65536,
                    )
                    ingest.attach_drain(drain, ring)  # pull() reads its records
                else:  # pushes accounted bursts into the ingest from its own thread
                    drain = PythonGpuDrain(
                        ingest,
                        daq,
                        port_id,
                        self.queue_id,
                        gpu_device=self.gpu_device,
                        keep_alive=self.keep_alive,
                        recv_timeout_s=self.recv_timeout_s,
                    )
                drain.start()
        except BaseException:
            # Undo a successful init (also when the mask check failed after it); a started
            # drain polls DAQIRI, so it stops first.
            if drain is not None:
                drain.stop()
            if status == daq.Status.SUCCESS:
                daq.shutdown()
            raise
        self._daq, self._ingest, self._drain, self._ring = daq, ingest, drain, ring
        self._open(ingest, "native drain" if drain_cls is not None else "per-packet binding")
        logger.info(
            "GPUDirect MITCH source draining %s (%s) q%d into GPU %d, udp_dst=%d (%s)",
            self.interface_name,
            self.pcie_address,
            self.queue_id,
            self.gpu_device,
            self.port,
            stats.frame_fetch,
        )

    def stop(self) -> None:
        super().stop()  # the drain is joined before DAQIRI shuts down
        if self._daq is not None:
            self._daq.print_stats()
            self._daq.shutdown()
            self._daq = None

    def _daqiri_config(self) -> dict | str:
        return self.daqiri_config or build_daqiri_config(
            pcie_address=self.pcie_address,
            udp_port=self.port,
            master_core=self.master_core,
            rx_core=self.rx_core,
            num_bufs=self.num_bufs,
            buf_size=self.buf_size,
            rx_batch_size=self.rx_batch_size,
            rx_timeout_us=self.rx_timeout_us,
            memory_kind="device",
            interface_name=self.interface_name,
            queue_id=self.queue_id,
            log_level=self.daqiri_log_level,
        )


class GpuMitchReplaySourceOp(_GpuMitchSourceBase):
    """NIC-free GPU source: the Ethernet frames of a pcap, uploaded once into GPU memory, are
    replayed by a software NIC (``DeviceReplayDrain``, thread ``hf-replay``) at ``pps`` in
    DAQIRI-sized bursts; everything after the drain is the production code. The drain is
    Python, so this validates function, bit identity, headroom and paced losslessness, not
    resilience to GIL stalls.
    """

    pcap_path = hs.Param("", description="the capture to replay (required)")
    pps = hs.Param(200000, range=(0, 10_000_000), description="packets/s; 0 = unpaced")
    loops = hs.Param(1, range=(0, 1_000_000), description="passes over the capture; 0 = forever")
    burst_packets = hs.Param(4096, range=(1, 65536), description="DAQIRI rx_batch_size")
    rx_timeout_us = hs.Param(
        20000, range=(0, 1_000_000), description="DAQIRI partial-burst timeout"
    )
    nic_bufs = hs.Param(
        0, range=(0, 4_194_304), description="paced NIC buffer model: drop due frames beyond this"
    )
    start_delay_ms = hs.Param(0, range=(0, 600_000))
    on_loop = hs.Param(
        Ref(), description="Ref(callable(pass_index) -> sender port | None), before pass >= 1"
    )
    settled = hs.Param(
        Ref(),
        description="Ref(callable() -> bool): before pass >= 1, wait until true (every batch "
        "consumed downstream, e.g. DrainedCondition.batches_consumed)",
    )
    keep_alive = hs.Param(False, description="False: the run's input ends after the last loop")

    def setup(self, spec: hs.OperatorSpec) -> None:
        if not self.pcap_path:
            raise ValueError("pcap_path is required")
        if self.burst_packets > self.ring_size:
            raise ValueError("burst_packets must not exceed ring_size")
        super().setup(spec)

    def start(self) -> None:
        self._begin("replay")
        frames = PcapFrames.read(self.pcap_path)
        # Pass 0 is sent from the published first generation's port, like every later pass.
        first = self._knob.get() if self._knob is not None else None
        device_frames = DeviceFrames.upload(
            frames, self.gpu_device, sender_port=first.sender_port if first is not None else None
        )
        n_frames = len(frames)
        # Free the host copy before the drain's schedule starts: for a full day (3.4 GB) that
        # takes about 240 ms with the GIL held, and the first bursts would be gathered that late.
        del frames
        ring = DeviceRing.allocate(self.ring_size, self.gpu_device)
        ingest = self._new_ingest()
        with thread_mask(self._helper_mask()):
            drain = DeviceReplayDrain(
                device_frames,
                ring,
                gather_fn=_gather_kernel(),
                pps=self.pps,
                loops=self.loops,
                burst_packets=self.burst_packets,
                rx_timeout_us=self.rx_timeout_us,
                nic_bufs=self.nic_bufs,
                start_delay_ms=self.start_delay_ms,
                on_loop=self.on_loop.obj,
                settled=self.settled.obj,
                gpu_device=self.gpu_device,
            )
            drain.start()
        ingest.attach_drain(drain, ring)
        self._ingest, self._drain, self._ring = ingest, drain, ring
        self._open(ingest, "device frame pool")
        logger.info(
            "GPU replay source: %d frames of %s on GPU %d (pps=%d, loops=%d, bursts of %d)",
            n_frames,
            self.pcap_path,
            self.gpu_device,
            self.pps,
            self.loops,
            self.burst_packets,
        )


__all__ = ["GpuMitchReplaySourceOp", "GpuMitchSourceOp"]
