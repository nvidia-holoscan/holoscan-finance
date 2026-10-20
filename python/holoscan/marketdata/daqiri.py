# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DAQIRI helpers: binding check, single-queue RX config, Ethernet/IPv4/UDP stripping."""

from __future__ import annotations

import socket

# Ethernet header without VLAN; DAQIRI raw RX delivers full L2 frames.
_ETH_HLEN = 14
_ETHERTYPE_IPV4 = b"\x08\x00"
_IPPROTO_UDP = 17
_UDP_HLEN = 8


def require_daqiri():
    """Fail explicitly on missing/incompatible bindings; never switch to sockets."""
    try:
        import daqiri
    except ImportError as exc:
        raise RuntimeError(
            "ANO ingest requires DAQIRI with Python bindings and the DPDK runtime; "
            "see deploy/igx_thor/README.md"
        ) from exc
    required = (
        "Status",
        "daqiri_init",
        "get_port_id",
        "get_rx_burst",
        "get_num_packets",
        "get_packet_bytes",
        "free_all_packets_and_burst_rx",
        "print_stats",
        "shutdown",
    )
    missing = [name for name in required if not hasattr(daqiri, name)]
    if missing:
        raise RuntimeError(f"Incompatible DAQIRI Python bindings: missing {', '.join(missing)}")
    return daqiri


def strip_udp_datagram(frame: bytes, minimum_payload=0):
    """Return (UDP payload, source peer), respecting IPv4/UDP boundaries."""
    if len(frame) < _ETH_HLEN + 20 + _UDP_HLEN + minimum_payload:
        return None
    if frame[12:14] != _ETHERTYPE_IPV4:
        return None
    ihl = (frame[_ETH_HLEN] & 0x0F) * 4
    ip_length = int.from_bytes(frame[16:18], "big")
    fragment = int.from_bytes(frame[20:22], "big")
    if (
        frame[_ETH_HLEN] >> 4 != 4
        or ihl < 20
        or frame[_ETH_HLEN + 9] != _IPPROTO_UDP
        or fragment & 0x3FFF  # MF or fragment offset: no IP reassembly here
        or ip_length < ihl + _UDP_HLEN + minimum_payload
        or len(frame) < _ETH_HLEN + ip_length
    ):
        return None
    udp = _ETH_HLEN + ihl
    udp_length = int.from_bytes(frame[udp + 4 : udp + 6], "big")
    if udp_length < _UDP_HLEN + minimum_payload or udp_length != ip_length - ihl:
        return None
    # Respect the datagram boundary: Ethernet padding/FCS are not payload bytes.
    peer = (socket.inet_ntoa(frame[26:30]), int.from_bytes(frame[udp : udp + 2], "big"))
    return frame[udp + _UDP_HLEN : udp + udp_length], peer


def build_daqiri_config(
    *,
    pcie_address: str,
    udp_dst_port: int,
    interface_name: str = "rx_port",
    queue_id: int = 0,
    master_core: int = 3,
    rx_core: int = 9,
    memory_kind: str = "host_pinned",
    num_bufs: int = 51200,
    buf_size: int = 2048,
    rx_batch_size: int = 1024,
    rx_timeout_us: int = 5000,
    log_level: str = "warn",
) -> dict:
    """Single-port RX-only DAQIRI config: flow-isolated, one queue, steering every
    UDP datagram for ``udp_dst_port`` into it. ``host_pinned`` receives into
    CPU-accessible memory; ``device`` (the GPU source) into GPU memory.
    """
    return {
        "version": 1,
        "stream_type": "raw",
        "master_core": master_core,
        "log_level": log_level,
        "memory_regions": [
            {
                "name": "rx_pool",
                "kind": memory_kind,
                "affinity": 0,
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
                            "match": {"udp_dst": int(udp_dst_port)},
                        }
                    ],
                },
            }
        ],
    }


__all__ = ["build_daqiri_config", "require_daqiri", "strip_udp_datagram"]
