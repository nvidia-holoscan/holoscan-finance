# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``MitchDecodeOp`` under the operator harness: outputs equal ``decode_packet``, packets from
another sender port (or with no directory) are stale, malformed units are counted, the
counters it owns are exact, and an active capture tap sees every decoded list."""

from __future__ import annotations

import numpy as np
import pytest
from holoscan.testing import OperatorTestHarness

from holoscan_finance.nyq.manifest import validate_manifest
from holoscan_finance.nyq.mapping import PROTOCOL, TIMEZONE, day_start_ns, decode_packet
from holoscan_finance.nyq.synthetic import synthetic_stream
from holoscan_finance.operators import MitchDecodeOp
from holoscan_finance.payloads import PacketBatch
from holoscan_finance.ref import Knob, Ref
from holoscan_finance.telemetry.stats import IngestStats

pytestmark = pytest.mark.sdk

DATE, PORT, UNIVERSE = "2025-04-01", 40000, 20
MIDNIGHT = day_start_ns(DATE)


def directory(generation: int = 1, port: int = PORT) -> Knob:
    return Knob(
        validate_manifest(
            dict(
                format="nyq-equity-replay-v1",
                wire_protocol=PROTOCOL,
                market_timezone=TIMEZONE,
                sender_port=port,
                symbols=[f"S{i}" for i in range(UNIVERSE)],
                generation=generation,
                date=DATE,
                start_ns=MIDNIGHT,
                end_ns=MIDNIGHT + 86_400 * 10**9,
                events=1,
                profile=[],
            )
        )
    )


class Tap:
    """CaptureTap protocol: an active capture recording what the decode site hands it."""

    active = True

    def __init__(self) -> None:
        self.recorded: list = []

    def record_host_batches(self, batches: list) -> None:
        self.recorded.append(batches)


def packet_batch(
    payloads: list[bytes], *, port: int = PORT, first: int = 0, index: int = 0
) -> PacketBatch:
    items = tuple((p, first + k, 0, ("10.0.0.1", port)) for k, p in enumerate(payloads))
    return PacketBatch(items, 0, index)


def run(batches: list[PacketBatch], **params) -> tuple[list, IngestStats]:
    stats = IngestStats()
    with OperatorTestHarness(MitchDecodeOp, params={"stats": Ref(stats), **params}) as h:
        h.push_many("packets", batches)
        result = h.run()
    return result.values("events"), stats


def test_outputs_equal_decode_packet_and_the_counters_are_exact():
    payloads = synthetic_stream(np.random.default_rng(4), 60, UNIVERSE, MIDNIGHT)
    batches = [packet_batch(payloads[:25]), packet_batch(payloads[25:], first=25, index=1)]
    knob, tap = directory(generation=3), Tap()
    outputs, stats = run(batches, directory=Ref(knob), tap=Ref(tap))
    want, messages = [], 0
    for k, p in enumerate(payloads):
        decoded, count = decode_packet(p, k, generation=3, midnight_ns=MIDNIGHT, universe=UNIVERSE)
        want += decoded
        messages += count

    def fields(b) -> tuple:
        return b.generation, b.recv_wall_ns, b.sequence, b.wire_count, b.events.tobytes()

    assert [len(out) for out in outputs] == [25, 35]  # one ReplayBatch per packet, per input batch
    assert [fields(b) for out in outputs for b in out] == [fields(b) for b in want]
    assert (stats.messages, stats.stale_packets, stats.decode_malformed) == (messages, 0, 0)
    assert (stats.event_batches, stats.packet_batches_decoded) == (2, 2)
    assert tap.recorded == outputs


def test_another_sender_port_is_stale_and_malformed_units_are_counted():
    good = synthetic_stream(np.random.default_rng(5), 4, UNIVERSE, MIDNIGHT)
    truncated = good[0][:-3]
    bad_group = good[1][:3] + b"B" + good[1][4:]  # unit header: length, count, group, sequence
    batch = packet_batch([good[2], truncated, bad_group])
    stale = packet_batch([good[3]], port=PORT + 1, index=1)
    outputs, stats = run([batch, stale], directory=Ref(directory()))
    (decoded,) = decode_packet(good[2], 0, generation=1, midnight_ns=MIDNIGHT, universe=UNIVERSE)[0]
    assert [[b.sequence for b in out] for out in outputs] == [[decoded.sequence]]
    assert (stats.stale_packets, stats.decode_malformed) == (1, 2)
    # The stale batch decoded to nothing, so it emitted no event batch.
    assert (stats.event_batches, stats.packet_batches_decoded) == (1, 2)


def test_without_a_directory_every_packet_is_stale():
    payloads = synthetic_stream(np.random.default_rng(6), 5, UNIVERSE, MIDNIGHT)
    outputs, stats = run([packet_batch(payloads)])
    assert outputs == [] and stats.stale_packets == 5 and stats.messages == 0
    assert stats.packet_batches_decoded == 1 and stats.event_batches == 0
