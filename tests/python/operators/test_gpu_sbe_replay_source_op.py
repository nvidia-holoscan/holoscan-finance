# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``GpuSbeReplaySourceOp``: a synthetic pcap replayed from GPU memory through the
production gather, ring, accounting and decode.

Every packet is accounted and decoded like the host decoder, the run ends (``stats.eof``);
one pass registers no second generation and nothing is stale; two passes call ``on_loop``
exactly once, between them, so pass 1 decodes under its own generation with nothing stale;
a first-generation port the capture does not carry re-sources pass 0, and another sender's
frames stay stale in every pass; looped passes reach the bars under their own generation;
the NIC-buffer model's drops show up as sequence gaps. Pacing is a separate timing case.
"""

from __future__ import annotations

import ctypes
import time
import weakref

import cupy as cp
import holoscan as hs
import numpy as np
import pytest
from holoscan.testing import OperatorTestHarness, TestRunOptions

from holoscan_finance.app.drained import DrainedCondition
from holoscan_finance.marketdata.framing import udp_frame, write_pcap
from holoscan_finance.marketdata.sbe_codec import packet_count
from holoscan_finance.nyq.manifest import validate_manifest
from holoscan_finance.nyq.mapping import PROTOCOL, TIMEZONE, day_start_ns, decode_packet
from holoscan_finance.nyq.synthetic import synthetic_stream
from holoscan_finance.operators import GpuSbeReplaySourceOp, QuoteBarsOp
from holoscan_finance.operators.edges import lossless
from holoscan_finance.ref import Knob, Ref
from holoscan_finance.telemetry.stats import BarsStats, IngestStats

pytestmark = [pytest.mark.sdk, pytest.mark.gpu]

DATE, PORT, UNIVERSE = "2025-04-01", 40000, 100
MIDNIGHT = day_start_ns(DATE)


def manifest(generation: int, port: int) -> dict:
    return dict(
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


@pytest.fixture(scope="module")
def capture(tmp_path_factory):
    """2,000 synthetic units in a pcap, sent from PORT."""
    units = synthetic_stream(np.random.default_rng(17), 2000, UNIVERSE, MIDNIGHT)
    path = tmp_path_factory.mktemp("replay") / "day.pcap"
    write_pcap(path, [udp_frame(u, src_port=PORT) for u in units])
    return str(path), units


class Generations:
    """``ReplayGenerations``-like ``on_loop``: a new generation on a fresh sender port."""

    def __init__(self, knob: Knob) -> None:
        self.knob, self.calls = knob, []

    def __call__(self, pass_index: int) -> int:
        port = PORT + pass_index
        self.calls.append(pass_index)
        self.knob.set(validate_manifest(manifest(1 + pass_index, port)))
        return port


def replay(path: str, *, during=None, **params):
    """Run the replay source until its input ended; returns (batches, stats). ``during(stats)``
    runs on the test thread while the replay is live."""
    stats = IngestStats()
    params = dict(pcap_path=path, pps=0, warmup=False, stats=Ref(stats)) | params
    params.setdefault("directory", Ref(Knob(validate_manifest(manifest(1, PORT)))))
    with OperatorTestHarness(GpuSbeReplaySourceOp, params=params) as h:
        h.capture("events", capacity=100_000)
        with h.run_async(TestRunOptions(expect_stop=True, timeout_ms=60_000)) as running:
            if during is not None:
                during(stats)
            deadline = time.monotonic() + 30
            while not stats.eof and time.monotonic() < deadline:
                time.sleep(0.005)
            # A stop drops what the collector has not taken yet: wait for every batch sent.
            assert running.wait_for_output("events", stats.event_batches, timeout_ms=10_000)
            running.request_stop()
            result = running.finish()
    assert result.ok, result.describe_failure()
    assert stats.eof, "the replay never ended"
    return [b for out in result.values("events") for b in out], stats


def columns(batches) -> dict[str, np.ndarray]:
    names = ("timestamp_ns", "symbol_id", "kind", "flags", "bid", "ask", "price", "volume")
    return {n: np.concatenate([cp.asnumpy(b.columns[n]) for b in batches]) for n in names}


def host_events(units: list[bytes]) -> np.ndarray:
    return np.concatenate(
        [
            decode_packet(u, 0, generation=1, midnight_ns=MIDNIGHT, universe=UNIVERSE)[0][0].events
            for u in units
        ]
    )


def test_one_pass_is_accounted_decoded_and_ends(capture):
    path, units = capture
    calls = []
    batches, stats = replay(path, on_loop=Ref(calls.append), batch_size=8192)
    assert stats.packets == 2000
    assert stats.dropped == stats.ring_dropped == stats.stale_packets == 0
    assert stats.messages == stats.received_messages == sum(packet_count(u) for u in units)
    assert stats.malformed_packets == stats.decode_malformed == 0
    got, expected = columns(batches), host_events(units)
    for name, values in got.items():
        np.testing.assert_array_equal(values, expected[name].astype(values.dtype), err_msg=name)
    assert {b.generation for b in batches} == {1} and calls == []  # no second generation
    assert stats.front_end == "replay" and stats.frame_fetch == "device frame pool"
    assert stats.replay["sent"] == 2000 and stats.replay["eof"] and stats.replay["pps"] == 0
    assert {"poll_gaps_10ms", "poll_gaps_50ms", "poll_gaps_100ms"} <= set(stats.stalls["native"])
    assert stats.pending_packets() == 0 and stats.drained()


def test_two_passes_get_two_generations_and_nothing_stale(capture):
    path, units = capture
    knob = Knob(validate_manifest(manifest(1, PORT)))
    generations = Generations(knob)
    batches, stats = replay(path, loops=2, on_loop=Ref(generations), directory=Ref(knob))
    assert generations.calls == [1]  # before pass 1 only
    assert stats.packets == 4000 and stats.stale_packets == 0 and stats.dropped == 0
    assert stats.messages == 2 * sum(packet_count(u) for u in units)
    per_generation = {g: sum(b.n_packets for b in batches if b.generation == g) for g in (1, 2)}
    assert per_generation == {1: 2000, 2: 2000}
    assert stats.replay["pass_index"] == 1 and stats.replay["sent"] == 4000


def test_a_first_generation_port_the_capture_does_not_carry_sends_pass_0_from_it(capture):
    """``--replay-sender-port``: pass 0 is re-sourced to the published port, not stale."""
    path, units = capture
    knob = Knob(validate_manifest(manifest(1, PORT + 500)))
    batches, stats = replay(path, directory=Ref(knob), batch_size=8192)
    assert stats.packets == 2000 and stats.stale_packets == 0 and stats.dropped == 0
    assert stats.messages == sum(packet_count(u) for u in units)
    got, expected = columns(batches), host_events(units)
    for name, values in got.items():
        np.testing.assert_array_equal(values, expected[name].astype(values.dtype), err_msg=name)


@pytest.fixture(scope="module")
def mixed_capture(capture, tmp_path_factory):
    """The capture behind 35 frames of another sender (PORT - 1) that repeat mid-day units,
    as a raw capture holds a previous generation's tail."""
    _, units = capture
    path = tmp_path_factory.mktemp("replay") / "mixed.pcap"
    foreign = [udp_frame(u, src_port=PORT - 1) for u in units[1000:1035]]
    write_pcap(path, foreign + [udp_frame(u, src_port=PORT) for u in units])
    return str(path), units


def test_another_senders_frames_stay_stale_in_every_pass(mixed_capture):
    """Only the feed is re-sourced per pass (like ``tcprewrite --portmap``): the foreign
    frames never join the feed's session, so every pass decodes the same."""
    path, units = mixed_capture
    knob = Knob(validate_manifest(manifest(1, PORT)))
    batches, stats = replay(path, loops=2, on_loop=Ref(Generations(knob)), directory=Ref(knob))
    assert stats.packets == 2 * 2035 and stats.stale_packets == 2 * 35
    assert stats.out_of_order_packets == stats.dropped == 0
    assert stats.messages == 2 * sum(packet_count(u) for u in units)
    per_generation = {g: sum(b.n_packets for b in batches if b.generation == g) for g in (1, 2)}
    assert per_generation == {1: 2000, 2: 2000}


def test_looped_passes_reach_the_bars_under_their_own_generation(capture):
    """Unpaced, three passes: the loop boundary waits until the bars took every batch of the
    pass (``settled``), so no pass's tail (or, unpaced, whole pass) is stale at the bars."""
    path, units = capture
    knob = Knob(validate_manifest(manifest(1, PORT)))
    generations = Generations(knob)
    ingest, bars_stats = IngestStats(), BarsStats()
    drained = DrainedCondition(ingest=ingest, bars=[bars_stats])
    g = hs.Graph("replay_loops", default_clock=hs.RealtimeClock)
    source = g.op(
        GpuSbeReplaySourceOp,
        "source",
        pcap_path=path,
        pps=0,
        loops=3,
        batch_size=8192,  # one burst and one batch per pass: the race is deterministic
        warmup=False,
        on_loop=Ref(generations),
        settled=Ref(drained.batches_consumed),
        directory=Ref(knob),
        stats=Ref(ingest),
    )
    bars = g.op(
        QuoteBarsOp,
        "bars",
        backend="gpu",
        emit_windows=False,
        warmup=False,
        directory=Ref(knob),
        stats=Ref(bars_stats),
    )
    g.add_flow(source.events, bars.events, **lossless(4))
    g.partition("ingest").add(source, bars)
    plan = hs.compile(g)
    assert plan.ok, plan.summary()
    with hs.run_async(plan) as session:
        deadline = time.monotonic() + 120
        while not drained() and time.monotonic() < deadline and not session.wait(0.05):
            pass
        assert drained(), drained.why()
        session.request_stop()
        session.wait(10)
    assert session.termination is hs.RunTermination.STOPPED, session.summary()
    assert generations.calls == [1, 2] and ingest.stale_packets == 0
    assert ingest.decoded_batches == ingest.event_batches == bars_stats.batches_in
    assert bars_stats.stale_packets == 0 and bars_stats.sequence_gaps == 0
    assert bars_stats.generation == knob.get().generation == 3
    assert bars_stats.counters.events == len(host_events(units))  # the last pass only


def test_nic_buffer_drops_show_up_as_sequence_gaps(capture):
    path, units = capture

    def stall(stats: IngestStats) -> None:
        """Once the first burst is in, hold the GIL for 40 ms (800 packets at 20k/s): the
        drain cannot take the next burst, so 600 NIC buffers (a 20 ms burst needs 401)
        overflow mid-stream."""
        deadline = time.monotonic() + 10
        while not stats.packets and time.monotonic() < deadline:
            time.sleep(0.001)
        ctypes.PyDLL(None).usleep(40_000)

    batches, stats = replay(path, pps=20000, nic_bufs=600, during=stall)
    dropped = stats.replay["nic_dropped"]
    assert dropped > 0 and stats.packets + dropped == 2000 and stats.ring_dropped == 0
    sequences = {int(s) for b in batches for s in b.sequence}
    received = [i for i, u in enumerate(units) if int.from_bytes(u[8:16], "little") in sequences]
    lost = [u for i, u in enumerate(units) if received[0] < i < received[-1] and i not in received]
    assert stats.dropped == sum(packet_count(u) for u in lost) > 0  # the drops are sequence gaps


def test_setup_rejects_a_missing_capture_and_oversized_bursts():
    g = hs.Graph("bad", default_clock=hs.RealtimeClock)
    with pytest.raises(hs.errors.HoloscanError, match="pcap_path is required"):
        g.op(GpuSbeReplaySourceOp, "replay")
    with pytest.raises(hs.errors.HoloscanError, match="burst_packets"):
        g.op(GpuSbeReplaySourceOp, "replay2", pcap_path="x", ring_size=4096, burst_packets=8192)


def test_the_host_copy_of_the_capture_is_freed_before_the_drain_starts(capture, monkeypatch):
    """Freeing a full day's host copy (3.4 GB) holds the GIL for about 240 ms. Freed after
    ``drain.start()``, it made the first bursts that late: paced lag p99 86 ms against the
    5 ms gate of ``market_data_bench paced``."""
    from holoscan_finance.ingest import drains
    from holoscan_finance.operators import gpu_sbe_source

    read, start, seen = gpu_sbe_source.PcapFrames.read, drains.DeviceReplayDrain.start, {}

    def tracked_read(path, limit=0):
        frames = read(path, limit)
        seen["data"] = weakref.ref(frames.data)
        return frames

    def checked_start(self):
        seen["host_copy_alive"] = seen["data"]() is not None
        start(self)

    monkeypatch.setattr(gpu_sbe_source.PcapFrames, "read", staticmethod(tracked_read))
    monkeypatch.setattr(drains.DeviceReplayDrain, "start", checked_start)
    _, stats = replay(capture[0])
    assert stats.packets == 2000
    assert seen["host_copy_alive"] is False


def test_an_unreadable_capture_fails_start(tmp_path):
    bad = tmp_path / "bad.pcap"
    bad.write_bytes(b"not a pcap at all, just text....")
    with OperatorTestHarness(GpuSbeReplaySourceOp, params={"pcap_path": str(bad)}) as h:
        result = h.run(TestRunOptions(timeout_ms=10_000), check=False)
    assert not result.ok and "not a pcap file" in result.describe_failure()


@pytest.mark.slow
@pytest.mark.timing
def test_paced_replay_keeps_the_requested_rate(capture):
    path, _ = capture
    batches, stats = replay(path, pps=20000)  # bursts close at the 20 ms timeout
    recv = np.concatenate([b.packet_recv_ns for b in batches])
    first = int(np.count_nonzero(recv == recv[0]))  # the first burst's packets
    rate = (len(recv) - first) / ((recv[-1] - recv[0]) / 1e9)
    assert abs(rate / 20000 - 1) < 0.05
    assert stats.replay["lag_p99_ms"] < 5
