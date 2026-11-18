# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The host ingest lane loses nothing: 50,000 packets through a test ``PacketBatch`` source
-> ``SbeDecodeOp`` -> ``QuoteBarsOp(cpu)`` on ``lossless(16)`` edges in one lane. Every
message is decoded and every observation reaches the bars, with no backpressure error, and
the finite run ends by the drained rule (the ``DrainedCondition`` terms for this chain)."""

from __future__ import annotations

import time

import holoscan as hs
import numpy as np
import pytest

from holoscan_finance.equity.manifest import validate_manifest
from holoscan_finance.equity.mapping import PROTOCOL, TIMEZONE, day_start_ns, pack_events
from holoscan_finance.marketdata.records import EVENT_DTYPE, QUOTE, TRADE
from holoscan_finance.marketdata.sbe_codec import packet_count
from holoscan_finance.operators import QuoteBarsOp, SbeDecodeOp
from holoscan_finance.operators._common import PendingEmits
from holoscan_finance.operators.edges import lossless
from holoscan_finance.payloads import PacketBatch
from holoscan_finance.ref import Knob, Ref
from holoscan_finance.telemetry.stats import BarsStats, IngestStats

pytestmark = pytest.mark.sdk

PACKETS, EVENTS, UNIVERSE, PORT = 50_000, 4, 200, 40000
MIDNIGHT = day_start_ns("2026-06-01")


def replay_units(count: int) -> list[bytes]:
    """``count`` sequenced units of quotes and trades, 10 ms apart from 09:30 (a 50k-packet
    ``synthetic_stream`` would run past the market day)."""
    rng = np.random.default_rng(17)
    units, sequence, t0 = [], 1, MIDNIGHT + 34_200 * 10**9
    for i in range(count):
        ev = np.zeros(EVENTS, EVENT_DTYPE)
        ev["timestamp_ns"] = t0 + i * 10_000_000 + np.arange(EVENTS) * 1000
        ev["symbol_id"] = rng.integers(0, UNIVERSE, EVENTS)
        ev["kind"] = rng.choice([QUOTE, QUOTE, TRADE], EVENTS)
        mid = np.round(rng.uniform(10, 500, EVENTS), 2)
        ev["bid"], ev["ask"], ev["bid_size"], ev["ask_size"] = mid - 0.01, mid + 0.01, 100, 100
        ev["price"], ev["volume"] = mid, 10
        unit, messages = pack_events(ev, sequence, MIDNIGHT)
        units.append(unit)
        sequence += messages
    return units


def directory() -> Knob:
    return Knob(
        validate_manifest(
            dict(
                format="equity-replay-v1",
                wire_protocol=PROTOCOL,
                market_timezone=TIMEZONE,
                sender_port=PORT,
                symbols=[f"S{i}" for i in range(UNIVERSE)],
                generation=1,
                date="2026-06-01",
                start_ns=MIDNIGHT,
                end_ns=MIDNIGHT + 86_400 * 10**9,
                events=1,
                profile=[],
            )
        )
    )


class PacketFeed(hs.Operator):
    """A host source without a socket: the staged datagrams in ``PacketBatch``es, with the
    sources' hold-and-retry and counters, then ``eof`` and idle ticks (it never ends)."""

    packets = hs.Output("packets", hs.pyobject)
    payloads = hs.Param(Ref())
    stats = hs.Param(Ref())
    batch_packets = hs.Param(256)

    def __init__(self) -> None:
        self._next = self._index = 0
        self._pending = None

    def contract(self) -> hs.Contract:
        return hs.Contract().trigger(hs.OnClock(period=hs.time.us(500)))

    def start(self) -> None:
        self._pending = PendingEmits()
        self.stats.obj.reset(front_end="socket")
        self.stats.obj.opened = True

    def compute(self, ctx: hs.ExecutionContext) -> None:
        st, payloads = self.stats.obj, self.payloads.obj
        if self._pending:
            self._pending.flush()
        elif self._next < len(payloads):
            chunk = payloads[self._next : self._next + self.batch_packets]
            now = time.time_ns()
            items = tuple((p, now, packet_count(p), ("127.0.0.1", PORT)) for p in chunk)
            batch = PacketBatch(items, sum(packet_count(p) for p in chunk), self._index)
            self._next, self._index = self._next + len(chunk), self._index + 1
            self._pending.add(self.packets, batch, on_sent=st.count_packet_batch)
            self._pending.flush()
        st.held, st.retries = len(self._pending), self._pending.retries
        st.eof = self._next == len(payloads) and not self._pending


def test_every_message_reaches_the_bars_and_the_run_drains():
    payloads = replay_units(PACKETS)
    messages = sum(packet_count(p) for p in payloads)  # header count: every SBE entry
    ingest, bars_stats, knob = IngestStats(), BarsStats(), directory()
    g = hs.Graph("host_chain", default_clock=hs.RealtimeClock)
    source = g.op(PacketFeed, "source", payloads=Ref(payloads), stats=Ref(ingest))
    decode = g.op(SbeDecodeOp, "decode", directory=Ref(knob), stats=Ref(ingest))
    bars = g.op(
        QuoteBarsOp,
        "bars",
        backend="cpu",
        emit_windows=False,
        directory=Ref(knob),
        stats=Ref(bars_stats),
    )
    g.add_flow(source.packets, decode.packets, **lossless(16))
    g.add_flow(decode.events, bars.events, **lossless(16))
    g.partition("ingest").add(source, decode, bars)
    plan = hs.compile(g)
    assert plan.ok, plan.summary()

    def drained() -> bool:  # DrainedCondition's ingest and bars terms
        return ingest.drained() and bars_stats.batches_in == ingest.event_batches

    with hs.run_async(plan) as session:
        deadline = time.monotonic() + 240
        while not drained() and time.monotonic() < deadline and not session.wait(0.05):
            pass
        assert drained(), "the chain did not drain"
        session.request_stop()
        session.wait(10)
    assert session.termination is hs.RunTermination.STOPPED, session.summary()
    assert [d.token for d in session.diagnostics] == []
    assert ingest.messages == messages and ingest.stale_packets == ingest.decode_malformed == 0
    assert ingest.packet_batches == ingest.packet_batches_decoded == -(-PACKETS // 256)
    assert bars_stats.batches_in == ingest.event_batches == ingest.packet_batches
    c = bars_stats.counters
    assert c.events == PACKETS * EVENTS and c.late_events == 0 and c.completed_bars > 0
    assert bars_stats.sequence_gaps == bars_stats.stale_packets == 0
