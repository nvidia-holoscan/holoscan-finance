# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``GpuSbeSourceOp`` with the fake ``daqiri`` (frames in GPU memory) and
``FakeNativeDrain`` standing in for the native helper.

The emitted device batches decode like the host decoder; ring space is released in order;
the drain's ``poll_gaps_*`` reach ``stalls["native"]``; receive counters equal
``ReceiveAccounting``; a full downstream edge is held and retried; a drain error fails the
run with its message; the helper is built before ``daqiri_init``; a failed ``start()`` stops
a started drain and undoes a successful ``daqiri_init``; the lane mask survives a
``daqiri_init`` that re-pins its caller, and the drain never runs on the lane mask; and
``native_drain=False`` takes the Python drain.
"""

from __future__ import annotations

import contextlib
import os
import sys
import threading
import time

import cupy as cp
import holoscan as hs
import numpy as np
import pytest
from fakes.fake_daqiri import make_fake_daqiri
from fakes.fake_drain import FakeNativeDrain
from holoscan.testing import OperatorTestHarness, TestRunOptions

from holoscan_finance.equity.manifest import validate_manifest
from holoscan_finance.equity.mapping import PROTOCOL, TIMEZONE, day_start_ns, decode_packet
from holoscan_finance.equity.synthetic import synthetic_stream
from holoscan_finance.ingest.accounting import ReceiveAccounting, sbe_packet_header
from holoscan_finance.ingest.affinity import process_mask
from holoscan_finance.marketdata.framing import udp_frame
from holoscan_finance.operators import GpuSbeSourceOp, gpu_sbe_source
from holoscan_finance.ref import Knob, Ref
from holoscan_finance.telemetry.stats import IngestStats

pytestmark = [pytest.mark.sdk, pytest.mark.gpu]

DATE, PORT, UNIVERSE = "2026-06-01", 40000, 50
MIDNIGHT = day_start_ns(DATE)
SENDER = ("10.0.0.1", PORT)
COUNTERS = ("packets", "received_messages", "dropped", "out_of_order_packets", "malformed_packets")


def directory() -> Knob:
    return Knob(
        validate_manifest(
            dict(
                format="equity-replay-v1",
                wire_protocol=PROTOCOL,
                market_timezone=TIMEZONE,
                sender_port=PORT,
                symbols=[f"S{i}" for i in range(UNIVERSE)],
                generation=3,
                date=DATE,
                start_ns=MIDNIGHT,
                end_ns=MIDNIGHT + 86_400 * 10**9,
                events=1,
                profile=[],
            )
        )
    )


def bursts(n_bursts: int, size: int, *, gap_after: int | None = None) -> list[list[bytes]]:
    """Bursts of synthetic units as Ethernet frames; ``gap_after`` drops the next unit."""
    units = synthetic_stream(np.random.default_rng(11), n_bursts * size + 1, UNIVERSE, MIDNIGHT)
    if gap_after is not None:
        units.pop(gap_after + 1)
    frames = [udp_frame(u, src_port=PORT, src_ip=SENDER[0]) for u in units]
    return [frames[i * size : (i + 1) * size] for i in range(n_bursts)]


def payloads_of(burst_list: list[list[bytes]]) -> list[bytes]:
    return [f[42:] for b in burst_list for f in b]  # IHL 5: 14 + 20 + 8 header bytes


@pytest.fixture
def nic(monkeypatch):
    """Installs a fake ``daqiri`` and the fake native helper; returns a factory."""
    calls: list[str] = []
    FakeNativeDrain.instances.clear()

    def loader():
        calls.append("load_daqiri_drain")
        return FakeNativeDrain

    monkeypatch.setattr(gpu_sbe_source, "load_daqiri_drain", loader)

    def install(burst_list=(), **kw):
        fake = make_fake_daqiri(burst_list, device=True, calls=calls, **kw)
        monkeypatch.setitem(sys.modules, "daqiri", fake)
        return fake

    return install


def wait(predicate, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def run_source(params: dict, packets: int) -> list:
    """Run the source until its input ended (``recv_timeout_s`` of silence) and every one of
    ``packets`` packets was emitted; returns the emitted lists."""
    stats = params.setdefault("stats", Ref(IngestStats())).obj
    params.setdefault("directory", Ref(directory()))
    params.setdefault("warmup", False)
    params.update(keep_alive=False, recv_timeout_s=0.2)
    with OperatorTestHarness(GpuSbeSourceOp, params=params) as h:
        h.capture("events", capacity=10_000)
        with h.run_async(TestRunOptions(expect_stop=True, timeout_ms=30_000)) as running:
            assert wait(lambda: stats.eof)  # set after the last emit
            # A stop drops what the collector has not taken yet: wait for every batch sent.
            assert running.wait_for_output("events", stats.event_batches, timeout_ms=10_000)
            running.request_stop()
            result = running.finish()
    assert result.ok, result.describe_failure()
    assert stats.packets == packets
    return result.values("events")


def host_events(payloads: list[bytes]) -> np.ndarray:
    return np.concatenate(
        [
            decode_packet(p, 0, generation=3, midnight_ns=MIDNIGHT, universe=UNIVERSE)[0][0].events
            for p in payloads
        ]
    )


def assert_decoded(outputs: list, payloads: list[bytes]) -> None:
    batches = [b for out in outputs for b in out]
    assert all(len(out) == 1 for out in outputs)  # always a one-element list
    expected = host_events(payloads)
    for name in ("timestamp_ns", "symbol_id", "kind", "flags", "bid", "ask", "price", "volume"):
        got = np.concatenate([cp.asnumpy(b.columns[name]) for b in batches])
        np.testing.assert_array_equal(got, expected[name].astype(got.dtype), err_msg=name)
    assert sum(b.n_packets for b in batches) == len(payloads)
    assert {b.generation for b in batches} == {3}


def reference(payloads: list[bytes]) -> IngestStats:
    stats = IngestStats()
    acct = ReceiveAccounting(stats)
    for p in payloads:
        acct.observe_packet(*sbe_packet_header(p, SENDER), 0)
    return stats


def test_native_drain_path_decodes_accounts_and_releases_in_order(nic):
    burst_list = bursts(6, 40, gap_after=100)
    fake = nic(burst_list)
    stats = IngestStats()
    outputs = run_source({"stats": Ref(stats), "batch_size": 2000}, 240)
    payloads = payloads_of(burst_list)
    assert_decoded(outputs, payloads)
    ref = reference(payloads)
    for name in COUNTERS:
        assert getattr(stats, name) == getattr(ref, name), name
    assert stats.dropped > 0 and stats.messages == stats.received_messages
    assert stats.event_batches == len(outputs) and stats.retries == 0
    assert stats.frame_fetch == "native drain" and stats.front_end == "gpudirect"
    (drain,) = FakeNativeDrain.instances
    assert drain.released == sorted(drain.released) and drain.released[-1] == 240
    assert isinstance(drain.gather_fn, int) and drain.gather_fn  # the CUfunction address
    assert {"poll_gaps_10ms", "poll_gaps_50ms", "poll_gaps_100ms"} <= set(stats.stalls["native"])
    assert fake.calls == ["load_daqiri_drain", "daqiri_init", "shutdown"]  # built before init
    assert fake.configs[0]["memory_regions"][0]["kind"] == "device"
    assert len(fake.freed) == 6


def test_drain_core_reaches_the_native_drain(nic):
    nic(bursts(1, 40))
    run_source({"drain_core": 5, "isolate_cores": False}, 40)  # no sweep of the test process
    (drain,) = FakeNativeDrain.instances
    assert drain.cpu == 5


def test_warm_up_runs_in_start(nic, monkeypatch):
    nic(bursts(1, 4))
    warmed = []
    monkeypatch.setattr(gpu_sbe_source, "warm_decode_kernels", warmed.append)
    run_source({"warmup": True}, 4)
    assert warmed == [0]


def test_python_drain_path(nic):
    burst_list = bursts(3, 20)
    fake = nic(burst_list)
    stats = IngestStats()
    outputs = run_source({"stats": Ref(stats), "native_drain": False}, 60)
    assert_decoded(outputs, payloads_of(burst_list))
    assert stats.frame_fetch == "per-packet binding" and FakeNativeDrain.instances == []
    assert fake.calls == ["daqiri_init", "shutdown"] and len(fake.freed) == 3


def test_drain_error_fails_the_run_with_its_message(nic):
    nic(bursts(1, 4))
    stats = IngestStats()
    with OperatorTestHarness(GpuSbeSourceOp, params={"stats": Ref(stats), "warmup": False}) as h:
        with h.run_async(TestRunOptions(expect_stop=True, timeout_ms=30_000)) as running:
            assert wait(lambda: FakeNativeDrain.instances and stats.packets == 4)
            FakeNativeDrain.instances[0].fail("gather_frame launch failed")
            assert wait(lambda: stats.error is not None)
            result = running.finish(check=False)
    assert not result.ok
    assert "GPU SBE RX failed: gather_frame launch failed" in result.describe_failure()


@pytest.mark.parametrize(
    "fault, message, shutdowns",
    [
        ("init", "daqiri_init failed: BAD_CONFIG", 0),
        ("port", "DAQIRI interface 'rx_port' not found", 1),
        ("mask", "thread mask not restored", 1),
    ],
)
def test_start_failures_undo_daqiri_init(nic, monkeypatch, fault, message, shutdowns):
    fake = nic()
    shutdown = fake.shutdown

    def checked_shutdown():
        assert not any(d.running for d in FakeNativeDrain.instances)  # drains stop first
        shutdown()

    fake.shutdown = checked_shutdown
    if fault == "init":
        fake.daqiri_init = lambda cfg: "BAD_CONFIG"
    elif fault == "port":
        fake.get_port_id = lambda name: -1
    else:  # the helper mask cannot be restored after the drain started
        real, uses = gpu_sbe_source.thread_mask, []

        @contextlib.contextmanager
        def unrestorable(cpus=None):
            uses.append(cpus)
            with real(cpus):
                yield
            if len(uses) == 2:
                raise RuntimeError("thread mask not restored")

        monkeypatch.setattr(gpu_sbe_source, "thread_mask", unrestorable)
    with OperatorTestHarness(GpuSbeSourceOp, params={"warmup": False}) as h:
        result = h.run(TestRunOptions(timeout_ms=10_000), check=False)
    assert not result.ok and message in result.describe_failure()
    assert fake.shutdowns == shutdowns
    assert len(FakeNativeDrain.instances) == (1 if fault == "mask" else 0)


class SlowSink(hs.Operator):
    """Test sink in another lane: slow enough to fill a depth-1 BLOCKING edge."""

    items = hs.Input("items", hs.pyobject)
    sink = hs.Param(Ref())

    def contract(self) -> hs.Contract:
        return hs.Contract().trigger(hs.OnEach(self.items))

    def compute(self, ctx: hs.ExecutionContext) -> None:
        self.sink.obj.append((self.items.receive_data(), os.sched_getaffinity(0)))
        time.sleep(0.03)


def run_graph(params: dict, until, *, lanes: dict[str, list[int]], split: bool) -> tuple:
    """``source -> SlowSink``; the sink in its own lane when ``split``."""
    sink: list = []
    stats = params.setdefault("stats", Ref(IngestStats())).obj
    params.setdefault("directory", Ref(directory()))
    params.setdefault("warmup", False)
    g = hs.Graph("gpu_source", default_clock=hs.RealtimeClock)
    src = g.op(GpuSbeSourceOp, "source", **params)
    out = g.op(SlowSink, "sink", sink=Ref(sink))
    g.add_flow(src.events, out.items, queue_policy=hs.QueuePolicy.BLOCKING, queue_depth=1)
    if split:
        g.partition("ingest").add(src)
        g.partition("other").add(out)
    else:
        g.partition("ingest").add(src, out)
    deployment = hs.execution.DeploymentBinding()
    for name, cpus in lanes.items():
        deployment.bind_thread_placement(
            name, hs.execution.ThreadPlacement(affinity=hs.execution.CPUAffinity(cpus))
        )
    plan = hs.compile(
        g,
        hs.CompileOptions(
            deployment=deployment,
            serialization_domain_splits=hs.execution.SerializationDomainSplitPolicy.ALLOW,
        ),
    )
    assert plan.ok, plan.summary()
    seen = {}
    with hs.run_async(plan) as session:
        deadline = time.monotonic() + 20
        while not until(stats, sink, seen) and time.monotonic() < deadline:
            if session.wait(0.01):
                break
        session.request_stop()
        session.wait(10)
    assert session.termination is hs.RunTermination.STOPPED, session.summary()
    return sink, stats, seen


def test_a_full_downstream_edge_is_held_and_retried(nic):
    burst_list = bursts(8, 5)
    nic(burst_list)
    cpus = sorted(os.sched_getaffinity(0))

    def until(stats, sink, seen):
        return len(sink) == 8

    sink, stats, _ = run_graph(
        {"batch_size": 1}, until, lanes={"ingest": cpus, "other": cpus}, split=True
    )
    assert stats.retries > 0 and stats.held == 0 and stats.event_batches == 8
    batches = [item[0][0] for item in sink]
    sequences = np.concatenate([b.sequence for b in batches])
    assert np.all(np.diff(sequences.astype(np.int64)) > 0)  # every batch, in order


@pytest.mark.skipif(len(os.sched_getaffinity(0)) < 3, reason="needs three CPUs")
def test_lane_mask_survives_daqiri_init_and_the_drain_uses_the_process_mask(nic):
    cpus = sorted(os.sched_getaffinity(0))
    lane, lcore = cpus[:2], cpus[2]
    fake = nic(pin={lcore})  # EAL pins its caller, the lane thread

    def until(stats, sink, seen):
        if "drain" not in seen:
            drain = next((t for t in threading.enumerate() if t.name == "hf-drain"), None)
            if drain is not None:
                seen["drain"] = os.sched_getaffinity(drain.native_id)
                fake.push(bursts(1, 4)[0])
        return bool(sink)

    sink, stats, seen = run_graph({}, until, lanes={"ingest": lane}, split=False)  # defaults
    assert sink and sink[0][1] == set(lane)  # a compute after start() still runs on the lane
    assert stats.lane_cpus == lane
    assert seen["drain"] == process_mask()  # never the narrowed lane mask
