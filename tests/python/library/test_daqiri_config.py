# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DAQIRI helpers: the RX config shape (4.3 plus the DPDK engine) and the binding checks of
the host and GPUDirect sources (from 4.3 ``test_ano_source_op``)."""

from __future__ import annotations

import sys
import types

import pytest
from fakes.fake_daqiri import make_fake_daqiri

from holoscan_finance.ingest.daqiri import build_daqiri_config, require_daqiri

NIC = dict(
    pcie_address="0004:03:00.0",
    udp_port=26400,
    master_core=3,
    rx_core=9,
    num_bufs=51200,
    buf_size=2048,
    rx_batch_size=1024,
    rx_timeout_us=5000,
)


def test_build_daqiri_config_shape():
    cfg = build_daqiri_config(**NIC, memory_kind="host_pinned")
    assert cfg["stream_type"] == "raw"
    # DAQIRI >= #253 defaults raw streams to ibverbs, where the Thor steering patch is inert.
    assert cfg["engine"] == "dpdk"
    assert cfg["master_core"] == 3 and cfg["log_level"] == "warn"
    (region,) = cfg["memory_regions"]
    assert (region["kind"], region["num_bufs"], region["buf_size"]) == ("host_pinned", 51200, 2048)
    (iface,) = cfg["interfaces"]
    assert iface["name"] == "rx_port" and iface["address"] == "0004:03:00.0"
    assert iface["rx"]["flow_isolation"] is True
    (flow,) = iface["rx"]["flows"]
    assert flow["match"]["udp_dst"] == 26400
    assert flow["action"] == {"type": "queue", "id": 0}
    (queue,) = iface["rx"]["queues"]
    assert queue["memory_regions"] == [region["name"]]
    assert (queue["cpu_core"], queue["batch_size"], queue["timeout_us"]) == (9, 1024, 5000)


def test_build_daqiri_config_device_queue_and_names():
    cfg = build_daqiri_config(
        **NIC, memory_kind="device", interface_name="nic0", queue_id=2, log_level="info"
    )
    assert cfg["memory_regions"][0]["kind"] == "device"
    iface = cfg["interfaces"][0]
    assert iface["name"] == "nic0" and cfg["log_level"] == "info"
    assert iface["rx"]["queues"][0]["id"] == 2
    assert iface["rx"]["flows"][0]["action"]["id"] == 2


def test_the_buffer_pool_lives_on_the_decode_gpu():
    # A region's affinity is the CUDA ordinal of its buffers; GPUDirect decodes them in place.
    assert build_daqiri_config(**NIC, memory_kind="device")["memory_regions"][0]["affinity"] == 0
    cfg = build_daqiri_config(**NIC, memory_kind="device", gpu_device=1)
    assert cfg["memory_regions"][0]["affinity"] == 1


def test_missing_or_incompatible_bindings_fail_explicitly(monkeypatch):
    monkeypatch.setitem(sys.modules, "daqiri", None)
    with pytest.raises(RuntimeError, match="requires DAQIRI"):
        require_daqiri()
    monkeypatch.setitem(sys.modules, "daqiri", types.ModuleType("daqiri"))
    with pytest.raises(RuntimeError, match="Incompatible DAQIRI"):
        require_daqiri()


def test_host_and_gpu_need_their_own_bindings(monkeypatch):
    host = make_fake_daqiri()
    monkeypatch.setitem(sys.modules, "daqiri", host)
    assert require_daqiri() is host
    with pytest.raises(RuntimeError, match="get_packet_ptr, get_packet_length"):
        require_daqiri(gpu=True)

    gpu = make_fake_daqiri()
    del gpu.get_packet_bytes  # GPUDirect reads frame addresses only
    gpu.get_packet_ptr = gpu.get_packet_length = lambda burst, i: 0
    monkeypatch.setitem(sys.modules, "daqiri", gpu)
    assert require_daqiri(gpu=True) is gpu
    with pytest.raises(RuntimeError, match="missing get_packet_bytes"):
        require_daqiri()
