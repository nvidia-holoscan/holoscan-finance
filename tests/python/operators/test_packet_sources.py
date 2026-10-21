# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``UdpPacketSourceOp`` and ``DaqiriPacketSourceOp`` under the harness and in tiny placed
graphs: the emitted ``PacketBatch``es decode (through ``HostMitchDecoder``) to what was
sent, DAQIRI bursts are freed and gaps counted, failures fail the run with their message,
and threads keep their masks (from 4.3 ``test_ano_source_op``).

The receiver and drain threads start under the process mask, never a narrowed lane mask,
and the lane mask survives a ``daqiri_init`` that re-pins its caller. CPU only.
"""

from __future__ import annotations

import os
import socket
import sys
import threading
import time

import holoscan as hs
import numpy as np
import pytest
from fakes.fake_daqiri import make_fake_daqiri
from holoscan.testing import OperatorTestHarness, TestRunOptions

from holoscan_finance.ingest.affinity import process_mask
from holoscan_finance.marketdata.framing import udp_frame
from holoscan_finance.marketdata.records import EVENT_DTYPE, QUOTE
from holoscan_finance.nyq.host_decoder import HostMitchDecoder
from holoscan_finance.nyq.manifest import validate_manifest
from holoscan_finance.nyq.mapping import PROTOCOL, TIMEZONE, day_start_ns, pack_events
from holoscan_finance.operators import DaqiriPacketSourceOp, UdpPacketSourceOp
from holoscan_finance.ref import Ref
from holoscan_finance.telemetry.stats import IngestStats

pytestmark = pytest.mark.sdk

DATE, UNIVERSE = "2025-04-01", 2886
MIDNIGHT = day_start_ns(DATE)
PEER = ("192.168.100.1", 55555)


def directory(port: int):
    return validate_manifest(
        dict(
            format="nyq-equity-replay-v1",
            wire_protocol=PROTOCOL,
            market_timezone=TIMEZONE,
            sender_port=port,
            symbols=[f"S{i}" for i in range(UNIVERSE)],
            generation=1,
            date=DATE,
            start_ns=MIDNIGHT,
            end_ns=MIDNIGHT + 86_400 * 10**9,
            events=1,
            profile=[],
        )
    )


def quotes(bids: list[tuple[int, float]]) -> np.ndarray:
    rows = np.zeros(len(bids), EVENT_DTYPE)
    rows["kind"] = QUOTE
    rows["timestamp_ns"] = MIDNIGHT + 34200 * 10**9 + np.arange(len(rows))
    rows["symbol_id"] = [s for s, _ in bids]
    rows["bid"] = [b for _, b in bids]
    rows["ask"] = rows["bid"] + 0.02
    rows["bid_size"] = rows["ask_size"] = 100
    for f in ("price", "volume"):
        rows[f] = np.nan
    return rows


def unit(sequence: int, bids: list[tuple[int, float]]) -> tuple[bytes, int]:
    """One MITCH unit and its wire-message count (Time messages included)."""
    return pack_events(quotes(bids), sequence, MIDNIGHT)


def frame(payload: bytes, **kw) -> bytes:
    return udp_frame(payload, src_port=PEER[1], src_ip=PEER[0], **kw)


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def decoded(batches, port: int = PEER[1]) -> tuple[np.ndarray, int]:
    """Every emitted packet through the host reference decoder."""
    events, messages = [], 0
    for batch in batches:
        out, counts = HostMitchDecoder().decode(batch, directory(port))
        assert counts.stale == counts.malformed == 0
        events += [b.events for b in out]
        messages += counts.messages
    return np.concatenate(events), messages


def emitted(packets: int):
    """Every one of ``packets`` received packets went out in a ``PacketBatch``."""
    return lambda st: st.packets == packets and st.ring_fill() == 0 and st.held == 0


def run_source(cls, params: dict, until, *, before=None) -> list:
    """Run a source under the harness until ``until(stats)``; returns its batches."""
    stats = params.setdefault("stats", Ref(IngestStats())).obj
    with OperatorTestHarness(cls, params=params) as h:
        h.capture("packets", capacity=10_000)
        with h.run_async(TestRunOptions(expect_stop=True, timeout_ms=20_000)) as running:
            if before is not None:
                before()
            assert wait(lambda: until(stats), timeout=10.0)
            time.sleep(0.02)  # a packet may be counted a moment before it is queued
            assert wait(lambda: until(stats), timeout=10.0)
            # A stop drops what the collector has not taken yet: wait for every batch sent.
            assert running.wait_for_output("packets", stats.packet_batches, timeout_ms=10_000)
            running.request_stop()
            result = running.finish()
    assert result.ok, result.describe_failure()
    return result.values("packets")


def test_udp_source_emits_what_the_socket_received():
    stats = IngestStats()
    port = free_port()
    payloads, seq = [], 1
    for i in range(30):
        data, n = unit(seq, [(i % 9, 100.0 + i), (i % 7, 50.0)])
        payloads.append(data)
        seq += n + (4 if i == 14 else 0)  # four messages lost on the wire
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send():
        assert wait(lambda: stats.opened)
        for p in payloads:
            sender.sendto(p, ("127.0.0.1", port))

    with sender:
        params = {"port": port, "stats": Ref(stats), "batch_size": 16}
        batches = run_source(UdpPacketSourceOp, params, emitted(30), before=send)
        src_port = sender.getsockname()[1]
    items = [item for b in batches for item in b.items]
    assert [i[0] for i in items] == payloads
    assert all(b.messages - b.items[-1][2] < 16 for b in batches)  # batch_size messages at most
    assert [b.index for b in batches] == list(range(len(batches))) and len(batches) > 1
    events, messages = decoded(batches, src_port)
    assert messages == stats.received_messages and len(events) == 2 * len(items)
    assert stats.dropped == 4 and stats.front_end == "socket" and stats.lane_cpus
    assert stats.packet_batches == len(batches) and stats.retries == stats.held == 0


def test_cpu_list_params_are_validated_at_setup():
    g = hs.Graph("bad", default_clock=hs.RealtimeClock)
    with pytest.raises(hs.errors.HoloscanError, match="helper_cpus"):
        g.op(UdpPacketSourceOp, "udp", helper_cpus=["a"])
    with pytest.raises(hs.errors.HoloscanError, match="init_cpus"):
        g.op(DaqiriPacketSourceOp, "daqiri", init_cpus=[-1])
    with pytest.raises(hs.errors.HoloscanError, match="pcie_address"):
        g.op(DaqiriPacketSourceOp, "daqiri2", pcie_address="")
    g.op(UdpPacketSourceOp, "udp_ok", helper_cpus=(2, 1, 2))  # a tuple is a CPU list too


def test_daqiri_source_decodes_units_and_frees_bursts(monkeypatch):
    first, n1 = unit(1, [(0, 108.42), (1, 155.01)])
    second, n2 = unit(1 + n1, [(0, 108.44)])
    fake = make_fake_daqiri([[frame(first)], [frame(second)]])
    monkeypatch.setitem(sys.modules, "daqiri", fake)
    stats = IngestStats()
    batches = run_source(DaqiriPacketSourceOp, {"stats": Ref(stats)}, emitted(2))
    events, messages = decoded(batches)
    assert list(events["symbol_id"]) == [0, 1, 0]
    assert list(np.round(events["bid"], 2)) == [108.42, 155.01, 108.44]
    assert messages == stats.received_messages == n1 + n2 and stats.dropped == 0
    assert len(fake.freed) == 2 and fake.shutdowns == 1  # every burst returned; DAQIRI down
    assert fake.configs[0]["memory_regions"][0]["kind"] == "host_pinned"
    assert fake.configs[0]["engine"] == "dpdk"


def test_daqiri_source_preserves_the_quote_payload(monkeypatch):
    rows = quotes([(2885, 812345.12)])
    payload, _ = pack_events(rows, 1, MIDNIGHT)
    fake = make_fake_daqiri([[frame(payload)]])
    monkeypatch.setitem(sys.modules, "daqiri", fake)
    (batch,) = run_source(DaqiriPacketSourceOp, {}, emitted(1))
    (out,), counts = HostMitchDecoder().decode(batch, directory(PEER[1]))
    (got,) = out.events
    for field in ("timestamp_ns", "symbol_id", "kind", "bid", "ask", "bid_size", "ask_size"):
        assert got[field] == rows[0][field]
    assert out.generation == 1 and counts.messages == batch.messages and len(fake.freed) == 1


def test_daqiri_source_counts_gaps_and_ignores_non_ipv4(monkeypatch):
    first, n1 = unit(1, [(0, 108.42)])
    later, _ = unit(n1 + 4, [(0, 108.50)])  # sequences n1+1 .. n1+3 never arrive
    arp = b"\xff" * 12 + b"\x08\x06" + b"\x00" * 60
    fake = make_fake_daqiri([[arp, frame(first), frame(later, ihl=6)]])
    monkeypatch.setitem(sys.modules, "daqiri", fake)
    stats = IngestStats()
    batches = run_source(DaqiriPacketSourceOp, {"stats": Ref(stats)}, emitted(2))
    assert stats.dropped == 3 and stats.front_end == "daqiri-host"
    assert [i[0] for b in batches for i in b.items] == [first, later]


@pytest.mark.parametrize(
    "fault, message",
    [
        ("rx", "DAQIRI RX failed: get_rx_burst returned INTERNAL_ERROR"),
        ("read", "DAQIRI RX failed: get_packet_bytes returned INTERNAL_ERROR"),
        ("bindings", "Incompatible DAQIRI Python bindings: missing get_packet_bytes"),
    ],
)
def test_daqiri_failures_fail_the_run_with_their_message(monkeypatch, fault, message):
    fake = make_fake_daqiri([[frame(unit(1, [(0, 1.0)])[0])]])
    if fault == "rx":
        fake.get_rx_burst = lambda *_: ("INTERNAL_ERROR", None)
    elif fault == "read":
        fake.get_packet_bytes = lambda *_: ("INTERNAL_ERROR", b"")
    else:
        del fake.get_packet_bytes
    monkeypatch.setitem(sys.modules, "daqiri", fake)
    with OperatorTestHarness(DaqiriPacketSourceOp, params={}) as h:
        result = h.run(TestRunOptions(timeout_ms=10_000), check=False)
    assert not result.ok and message in result.describe_failure()
    if fault == "read":
        assert len(fake.freed) == 1  # the burst went back although the read failed
    assert fake.shutdowns == (0 if fault == "bindings" else 1)


class Collect(hs.Operator):
    """Test sink: records each item with the mask of the thread that received it."""

    items = hs.Input("items", hs.pyobject)
    sink = hs.Param(Ref())

    def contract(self) -> hs.Contract:
        return hs.Contract().trigger(hs.OnEach(self.items))

    def compute(self, ctx: hs.ExecutionContext) -> None:
        self.sink.obj.append((self.items.receive_data(), os.sched_getaffinity(0)))


def run_placed(source, params: dict, lane: list[int], until) -> list:
    """``source -> Collect`` in one ``ingest`` lane placed on ``lane``; returns the sink."""
    sink: list = []
    g = hs.Graph("placed", default_clock=hs.RealtimeClock)
    src = g.op(source, "source", **params)
    collect = g.op(Collect, "collect", sink=Ref(sink))
    g.add_flow(src.packets, collect.items, queue_policy=hs.QueuePolicy.BLOCKING, queue_depth=16)
    g.partition("ingest").add(src, collect)
    deployment = hs.execution.DeploymentBinding()
    deployment.bind_thread_placement(
        "ingest", hs.execution.ThreadPlacement(affinity=hs.execution.CPUAffinity(lane))
    )
    plan = hs.compile(g, hs.CompileOptions(deployment=deployment))
    assert plan.ok, plan.summary()
    with hs.run_async(plan) as session:
        deadline = time.monotonic() + 10
        while not until(sink) and time.monotonic() < deadline and not session.wait(0.01):
            pass
        session.request_stop()
        session.wait(10)
    assert session.termination is hs.RunTermination.STOPPED, session.summary()
    return sink


def thread_mask_of(name: str) -> set[int] | None:
    thread = next((t for t in threading.enumerate() if t.name == name), None)
    return None if thread is None else os.sched_getaffinity(thread.native_id)


@pytest.mark.skipif(len(os.sched_getaffinity(0)) < 2, reason="needs two CPUs")
def test_udp_receiver_thread_never_inherits_a_narrowed_lane():
    lane = [min(os.sched_getaffinity(0))]
    port, seen = free_port(), {}

    def until(sink):
        if "mask" not in seen and (mask := thread_mask_of("udp-drain")) is not None:
            seen["mask"] = mask
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.sendto(unit(1, [(0, 1.0)])[0], ("127.0.0.1", port))
        return bool(sink)

    sink = run_placed(UdpPacketSourceOp, {"port": port}, lane, until)
    assert sink and sink[0][1] == set(lane)  # the lane runs where it was placed
    assert seen["mask"] == process_mask()  # the receiver does not


@pytest.mark.skipif(len(os.sched_getaffinity(0)) < 3, reason="needs three CPUs")
def test_daqiri_init_re_pinning_never_leaks_into_the_lane(monkeypatch):
    cpus = sorted(os.sched_getaffinity(0))
    lane, lcore = cpus[:2], cpus[2]
    fake = make_fake_daqiri(pin={lcore})  # EAL pins the calling (lane) thread
    monkeypatch.setitem(sys.modules, "daqiri", fake)
    seen = {}

    def until(sink):
        if "mask" not in seen and (mask := thread_mask_of("daqiri-host-drain")) is not None:
            seen["mask"] = mask
            fake.push([frame(unit(1, [(0, 1.0)])[0])])
        return bool(sink)

    sink = run_placed(DaqiriPacketSourceOp, {}, lane, until)  # default init_cpus, helper_cpus
    assert sink and sink[0][1] == set(lane)  # restored after daqiri_init, seen by later computes
    assert seen["mask"] == process_mask()
