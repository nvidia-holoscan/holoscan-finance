# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DAQIRI binding checks and the single-queue RX configuration of the DAQIRI sources.

``require_daqiri`` fails explicitly on missing or incompatible bindings and never falls back
to sockets. ``build_daqiri_config`` is the 4.3 config (one flow-isolated queue steering every
datagram for the feed's UDP port) plus ``"engine": "dpdk"``: DAQIRI >= #253 defaults raw
streams to ibverbs, where the Thor steering patch would silently stop applying. From 4.3
``marketdata.daqiri``; ``strip_udp_datagram`` moved to ``marketdata.framing``.
"""

from __future__ import annotations

from types import ModuleType

_REQUIRED = (
    "Status",
    "daqiri_init",
    "get_port_id",
    "get_rx_burst",
    "get_num_packets",
    "free_all_packets_and_burst_rx",
    "print_stats",
    "shutdown",
)
_HOST_BINDINGS = ("get_packet_bytes",)  # host_pinned: one copy of each frame
_GPU_BINDINGS = ("get_packet_ptr", "get_packet_length")  # device: frame addresses only
LOG_LEVELS = ["trace", "debug", "info", "warn", "error", "critical", "off"]  # config log_level


def require_daqiri(*, gpu: bool = False) -> ModuleType:
    """The ``daqiri`` module with every binding the host (or, with ``gpu``, the GPUDirect)
    source calls; RuntimeError otherwise."""
    try:
        import daqiri
    except ImportError as exc:
        raise RuntimeError(
            "DAQIRI ingest requires DAQIRI with Python bindings and the DPDK runtime; "
            "see deploy/igx_thor/README.md"
        ) from exc
    required = _REQUIRED + (_GPU_BINDINGS if gpu else _HOST_BINDINGS)
    missing = [name for name in required if not hasattr(daqiri, name)]
    if missing:
        raise RuntimeError(f"Incompatible DAQIRI Python bindings: missing {', '.join(missing)}")
    return daqiri


def build_daqiri_config(
    *,
    pcie_address: str,
    udp_port: int,
    master_core: int,
    rx_core: int,
    num_bufs: int,
    buf_size: int,
    rx_batch_size: int,
    rx_timeout_us: int,
    memory_kind: str,
    gpu_device: int = 0,
    interface_name: str = "rx_port",
    queue_id: int = 0,
    log_level: str = "warn",
) -> dict:
    """Single-port RX-only DAQIRI config: flow-isolated, one queue, steering every UDP
    datagram for ``udp_port`` into it. ``host_pinned`` receives into CPU-accessible memory;
    ``device`` (GPUDirect) into GPU memory. ``gpu_device`` is the region's ``affinity``, the
    process-local CUDA ordinal the buffers belong to: the device the decode runs on.
    """
    return {
        "version": 1,
        "stream_type": "raw",
        "engine": "dpdk",
        "master_core": master_core,
        "log_level": log_level,
        "memory_regions": [
            {
                "name": "rx_pool",
                "kind": memory_kind,
                "affinity": gpu_device,
                "num_bufs": num_bufs,
                "buf_size": buf_size,
            }
        ],
        "interfaces": [
            {
                "name": interface_name,
                "address": pcie_address,
                "rx": {
                    "flow_isolation": True,
                    "queues": [
                        {
                            "name": "rx_queue",
                            "id": queue_id,
                            "cpu_core": rx_core,
                            "batch_size": rx_batch_size,
                            "timeout_us": rx_timeout_us,
                            "memory_regions": ["rx_pool"],
                        }
                    ],
                    "flows": [
                        {
                            "name": "udp_flow",
                            "id": 1,
                            "action": {"type": "queue", "id": queue_id},
                            "match": {"udp_dst": int(udp_port)},
                        }
                    ],
                },
            }
        ],
    }


__all__ = ["LOG_LEVELS", "build_daqiri_config", "require_daqiri"]
