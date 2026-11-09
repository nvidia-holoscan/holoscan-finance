# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``HostSbeDecoder``: the host decode rules per packet batch (stale sender ports, no
directory, malformed packets) and outputs equal to ``decode_packet``."""

from __future__ import annotations

import struct

import numpy as np

from holoscan_finance.marketdata.gpu_decoder import DecodeCounts
from holoscan_finance.marketdata.records import EVENT_DTYPE
from holoscan_finance.marketdata.sbe_codec import packet_count
from holoscan_finance.nyq.host_decoder import HostSbeDecoder
from holoscan_finance.nyq.manifest import MANIFEST_FORMAT, validate_manifest
from holoscan_finance.nyq.mapping import PROTOCOL, TIMEZONE, day_start_ns, decode_packet
from holoscan_finance.nyq.synthetic import fuzz, synthetic_stream
from holoscan_finance.payloads import PacketBatch

MIDNIGHT = day_start_ns("2025-04-01")
UNIVERSE = 50
PORT = 50001


def directory(generation=5, port=PORT):
    return validate_manifest(
        dict(
            format=MANIFEST_FORMAT,
            wire_protocol=PROTOCOL,
            market_timezone=TIMEZONE,
            symbols=[f"S{i}" for i in range(UNIVERSE)],
            generation=generation,
            sender_port=port,
            date="2025-04-01",
            start_ns=MIDNIGHT,
            end_ns=MIDNIGHT + 86400 * 10**9,
            events=100,
            profile=[],
        )
    )


def packets(payloads, port=PORT):
    wire = [packet_count(p) for p in payloads]  # the packet header's message count
    return PacketBatch(
        tuple((p, 1000 + i, w, ("10.0.0.1", port)) for i, (p, w) in enumerate(zip(payloads, wire))),
        sum(wire),
    )


def test_outputs_and_counts_equal_decode_packet():
    payloads = synthetic_stream(np.random.default_rng(1), 60, UNIVERSE, MIDNIGHT)
    out, counts = HostSbeDecoder().decode(packets(payloads), directory())
    expected, messages = [], 0
    for i, p in enumerate(payloads):
        batches, count = decode_packet(
            p, 1000 + i, generation=5, midnight_ns=MIDNIGHT, universe=UNIVERSE
        )
        expected += batches
        messages += count
    assert counts == DecodeCounts(messages, 0, 0)
    assert len(out) == len(expected) == len(payloads)
    for got, want in zip(out, expected):
        assert (got.generation, got.recv_wall_ns, got.sequence, got.sequence_count) == (
            want.generation,
            want.recv_wall_ns,
            want.sequence,
            want.sequence_count,
        )
        for name in EVENT_DTYPE.names:
            np.testing.assert_array_equal(got.events[name], want.events[name])


def test_without_a_directory_every_packet_is_stale():
    payloads = synthetic_stream(np.random.default_rng(2), 5, UNIVERSE, MIDNIGHT)
    assert HostSbeDecoder().decode(packets(payloads), None) == ([], DecodeCounts(0, 5, 0))


def test_other_sender_ports_and_unknown_peers_are_stale():
    payloads = synthetic_stream(np.random.default_rng(3), 4, UNIVERSE, MIDNIGHT)
    batch = packets(payloads)
    items = list(batch.items)
    items[1] = items[1][:3] + (("10.0.0.1", PORT + 1),)  # the previous generation's port
    items[2] = items[2][:3] + (None,)
    out, counts = HostSbeDecoder().decode(PacketBatch(tuple(items), batch.messages), directory())
    assert counts == DecodeCounts(packet_count(payloads[0]) + packet_count(payloads[3]), 2, 0)
    assert [b.recv_wall_ns for b in out] == [1000, 1003]


def test_malformed_units_are_counted_and_skipped():
    good = synthetic_stream(np.random.default_rng(4), 200, UNIVERSE, MIDNIGHT)
    mutated = fuzz(good, 600, seed=11)
    out, counts = HostSbeDecoder().decode(packets(mutated), directory())
    accepted = 0
    for i, p in enumerate(mutated):
        try:
            decode_packet(p, i, generation=5, midnight_ns=MIDNIGHT, universe=UNIVERSE)
            accepted += 1
        except (ValueError, struct.error):
            pass
    assert counts.malformed == len(mutated) - accepted > 0
    assert len(out) == accepted and counts.stale == 0
