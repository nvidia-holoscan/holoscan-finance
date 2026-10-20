# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for the DAQIRI-backed MITCH source.

The real DAQIRI engine needs a ConnectX NIC, hugepages and root. A fake ``daqiri``
module delivering raw Ethernet frames through the real burst-polling API exercises
everything else: frame stripping, sequence-gap accounting, the ring and decoding.
"""

from __future__ import annotations

import struct
import sys
import time
import types

import numpy as np
import pytest

from holoscan.equities import mitch as replay
from holoscan.marketdata.wire import EVENT_DTYPE, QUOTE

MIDNIGHT = replay.day_start_ns("2025-04-01")
UNIVERSE = 2886
PEER = ("192.168.100.1", 55555)


def eth_ip_udp_frame(payload: bytes, *, dst_port: int = 26400, ihl_words: int = 5) -> bytes:
    """Wrap a UDP payload the way the wire would: Ethernet II + IPv4 + UDP."""
    eth = b"\xac\x3a\xe2\x86\x76\x06" + b"\xac\x3a\xe2\x86\x76\x06" + b"\x08\x00"
    ip_len = ihl_words * 4
    total = ip_len + 8 + len(payload)
    ip = struct.pack(
        ">BBHHHBBH4s4s",
        (4 << 4) | ihl_words,
        0,
        total,
        0,
        0,
        64,
        17,  # UDP
        0,
        b"\xc0\xa8\x64\x01",  # 192.168.100.1
        b"\xc0\xa8\x64\x02",  # 192.168.100.2
    ) + b"\x00" * (ip_len - 20)
    udp = struct.pack(">HHHH", PEER[1], dst_port, 8 + len(payload), 0)
    return eth + ip + udp + payload


def quotes(symbols_bids):
    rows = np.zeros(len(symbols_bids), dtype=EVENT_DTYPE)
    rows["kind"] = QUOTE
    rows["timestamp_ns"] = MIDNIGHT + 34200 * 10**9 + np.arange(len(rows))
    rows["symbol_id"] = [s for s, _ in symbols_bids]
    rows["bid"] = [b for _, b in symbols_bids]
    rows["ask"] = rows["bid"] + 0.02
    rows["bid_size"] = rows["ask_size"] = 100
    return rows


def unit(sequence, symbols_bids):
    """One MITCH unit and its wire-message count (Time messages included)."""
    return replay.pack_events(quotes(symbols_bids), sequence, MIDNIGHT)


def decode(data, recv_ns, peer):
    return replay.decode_packet(
        data, recv_ns, generation=1, midnight_ns=MIDNIGHT, universe=UNIVERSE
    )


class _Status:
    SUCCESS = "SUCCESS"
    NULL_PTR = "NULL_PTR"


def make_fake_daqiri(bursts: list[list[bytes]]) -> types.ModuleType:
    """A ``daqiri`` double: each entry of ``bursts`` is one RX burst of raw frames."""
    mod = types.ModuleType("daqiri")
    mod.Status = _Status
    mod.freed = []
    mod.shutdowns = 0
    queue = list(bursts)

    def get_rx_burst(port, queue_id):
        if queue:
            return _Status.SUCCESS, queue.pop(0)
        return _Status.NULL_PTR, None

    def shutdown():
        mod.shutdowns += 1

    mod.daqiri_init = lambda cfg: _Status.SUCCESS
    mod.get_port_id = lambda name: 0
    mod.get_rx_burst = get_rx_burst
    mod.get_num_packets = lambda burst: len(burst)
    mod.get_packet_bytes = lambda burst, idx: (_Status.SUCCESS, burst[idx])
    mod.free_all_packets_and_burst_rx = lambda burst: mod.freed.append(burst)
    mod.print_stats = lambda: None
    mod.shutdown = shutdown
    return mod


@pytest.fixture
def ano_op_factory(fragment, monkeypatch):
    def make(bursts):
        fake = make_fake_daqiri(bursts)
        monkeypatch.setitem(sys.modules, "daqiri", fake)
        from holoscan.equities.host_sources import AnoMitchSourceOp

        op = AnoMitchSourceOp(fragment, name="ano_test", datagram_decoder=decode, batch_size=1024)
        return op, fake

    return make


def _wait(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def test_strip_udp_datagram_follows_ihl():
    from holoscan.marketdata.daqiri import strip_udp_datagram

    payload, _ = unit(1, [(0, 108.42)])
    assert strip_udp_datagram(eth_ip_udp_frame(payload)) == (payload, PEER)
    # IPv4 options (ihl > 5) shift the payload
    assert strip_udp_datagram(eth_ip_udp_frame(payload, ihl_words=6)) == (payload, PEER)


def test_strip_udp_datagram_rejects_junk():
    from holoscan.marketdata.daqiri import strip_udp_datagram

    assert strip_udp_datagram(b"") is None
    assert strip_udp_datagram(b"\x00" * 60) is None  # wrong ethertype
    arp = b"\xff" * 12 + b"\x08\x06" + b"\x00" * 60
    assert strip_udp_datagram(arp) is None


def test_strip_respects_datagram_lengths_and_rejects_fragments():
    from holoscan.marketdata.daqiri import strip_udp_datagram

    payload, _ = unit(1, [(0, 108.42)])
    frame = eth_ip_udp_frame(payload)
    strip = lambda f: strip_udp_datagram(f, minimum_payload=8)  # noqa: E731
    assert strip(frame + b"\x00" * 24) == (payload, PEER)  # Ethernet padding / FCS
    assert strip(frame[:-1]) is None
    for offset, value in (
        (14, b"\x65"),  # IP version 6
        (20, b"\x20\x00"),  # more fragments
        (20, b"\x00\x01"),  # fragment offset
        (38, b"\x00\x08"),  # UDP length without payload
        (38, b"\xff\xff"),  # UDP length beyond the datagram
        (16, b"\x00\x14"),  # IP total length without UDP
    ):
        broken = bytearray(frame)
        broken[offset : offset + len(value)] = value
        assert strip(bytes(broken)) is None


def test_build_daqiri_config_shape():
    from holoscan.marketdata.daqiri import build_daqiri_config

    cfg = build_daqiri_config(pcie_address="0004:03:00.0", udp_dst_port=26400)
    assert cfg["stream_type"] == "raw"
    iface = cfg["interfaces"][0]
    assert iface["address"] == "0004:03:00.0"
    assert iface["rx"]["flow_isolation"] is True
    assert iface["rx"]["flows"][0]["match"]["udp_dst"] == 26400
    (queue,) = iface["rx"]["queues"]
    assert queue["memory_regions"] == [cfg["memory_regions"][0]["name"]]


def test_ano_op_decodes_units_and_frees_bursts(ano_op_factory, op_output_factory):
    first, n1 = unit(1, [(0, 108.42), (1, 155.01)])
    second, n2 = unit(1 + n1, [(0, 108.44)])
    op, fake = ano_op_factory([[eth_ip_udp_frame(first)], [eth_ip_udp_frame(second)]])
    op.start()
    try:
        assert _wait(lambda: op.packets == 2)
        out = op_output_factory()
        op.compute(None, out, None)
        batches, port = out.emitted
        assert port == "tick"
        events = np.concatenate([b.events for b in batches])
        assert list(events["symbol_id"]) == [0, 1, 0]
        assert list(np.round(events["bid"], 2)) == [108.42, 155.01, 108.44]
        assert op.messages == op.received_messages == n1 + n2
        assert op.dropped == 0
        assert len(fake.freed) == 2  # every burst returned to the pool
    finally:
        op.stop()
    assert fake.shutdowns == 1


def test_ano_op_counts_wire_gaps(ano_op_factory):
    first, n1 = unit(1, [(0, 108.42)])
    # sequences n1+1 .. n1+3 never arrive
    later, _ = unit(n1 + 4, [(0, 108.50)])
    op, _ = ano_op_factory([[eth_ip_udp_frame(first), eth_ip_udp_frame(later)]])
    op.start()
    try:
        assert _wait(lambda: op.packets == 2)
        assert op.dropped == 3
    finally:
        op.stop()


def test_ano_preserves_quote_payload(ano_op_factory, op_output_factory):
    rows = quotes([(2885, 812345.12)])
    payload, _ = replay.pack_events(rows, 1, MIDNIGHT)
    op, fake = ano_op_factory([[eth_ip_udp_frame(payload)]])
    op.start()
    try:
        assert _wait(lambda: op.packets == 1)
        out = op_output_factory()
        op.compute(None, out, None)
        batches, _ = out.emitted
        (got,) = batches[0].events
        for field in ("timestamp_ns", "symbol_id", "kind", "bid", "ask", "bid_size", "ask_size"):
            assert got[field] == rows[0][field]
        assert batches[0].generation == 1 and len(fake.freed) == 1
    finally:
        op.stop()


def test_ano_op_ignores_non_ipv4_frames(ano_op_factory):
    arp = b"\xff" * 12 + b"\x08\x06" + b"\x00" * 60
    payload, count = unit(1, [(0, 108.42)])
    op, _ = ano_op_factory([[arp, eth_ip_udp_frame(payload)]])
    op.start()
    try:
        assert _wait(lambda: op.packets == 1)
        assert op.messages == 0  # decoding happens in compute(), not the drain
        assert op.ring_fill() == count
    finally:
        op.stop()


def test_ano_missing_or_incompatible_bindings_fail_explicitly(monkeypatch):
    from holoscan.marketdata.daqiri import require_daqiri

    monkeypatch.setitem(sys.modules, "daqiri", None)
    with pytest.raises(RuntimeError, match="requires DAQIRI"):
        require_daqiri()
    monkeypatch.setitem(sys.modules, "daqiri", types.ModuleType("daqiri"))
    with pytest.raises(RuntimeError, match="Incompatible DAQIRI"):
        require_daqiri()


def test_ano_fatal_rx_error_is_reported_instead_of_idle(ano_op_factory, op_output_factory):
    op, fake = ano_op_factory([])
    fake.get_rx_burst = lambda *_: ("INTERNAL_ERROR", None)
    op.start()
    try:
        assert _wait(lambda: op.error is not None)
        assert "INTERNAL_ERROR" in op.error
        with pytest.raises(RuntimeError, match="DAQIRI RX failed"):
            op.compute(None, op_output_factory(), None)
    finally:
        op.stop()
    assert fake.shutdowns == 1


def test_ano_packet_read_error_still_frees_burst(ano_op_factory):
    payload, _ = unit(1, [(0, 108.42)])
    op, fake = ano_op_factory([[eth_ip_udp_frame(payload)]])
    fake.get_packet_bytes = lambda *_: ("INTERNAL_ERROR", b"")
    op.start()
    try:
        assert _wait(lambda: op.error is not None)
        assert len(fake.freed) == 1
    finally:
        op.stop()
