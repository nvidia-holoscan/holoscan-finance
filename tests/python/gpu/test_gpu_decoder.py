# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``GpuSbeDecoder`` keeps its contract: stale sender ports never reach pass 2, accepted
packets become one device batch whose per-packet metadata and columns equal the host decode,
and malformed packets are counted. Also the decode warm-up."""

from __future__ import annotations

import logging

import numpy as np
import pytest
from test_gpu_decode import MIDNIGHT, UNIVERSE, host_decode, to_slots

from holoscan_finance.marketdata import warmup
from holoscan_finance.marketdata.gpu_decoder import DecodeCounts, GpuSbeDecoder
from holoscan_finance.marketdata.gpu_sbe import FIELDS, inspect_packets
from holoscan_finance.marketdata.records import EVENT_DTYPE, ReplayBatch
from holoscan_finance.nyq.manifest import MANIFEST_FORMAT, validate_manifest
from holoscan_finance.nyq.mapping import PROTOCOL, TIMEZONE
from holoscan_finance.nyq.synthetic import fuzz, synthetic_stream
from holoscan_finance.telemetry.capture import CaptureTap, LatencyCapture

pytestmark = pytest.mark.gpu
cp = pytest.importorskip("cupy")

PORT = 40000


def directory(generation=9):
    return validate_manifest(
        dict(
            format=MANIFEST_FORMAT,
            wire_protocol=PROTOCOL,
            market_timezone=TIMEZONE,
            symbols=[f"S{i}" for i in range(UNIVERSE)],
            generation=generation,
            sender_port=PORT,
            date="2025-04-01",
            start_ns=MIDNIGHT,
            end_ns=MIDNIGHT + 86400 * 10**9,
            events=100,
            profile=[],
        )
    )


def staged(payloads, ports):
    """Device slots and the receive header rows the sources stage: ``inspect_packets`` has
    no frame to take the source port from, so the test fills that column in."""
    slots, lens = to_slots(payloads)
    header = inspect_packets(slots, lens)
    header[:, 1] = cp.asarray(np.asarray(ports, np.int64))
    return slots, header


def decode(payloads, ports, decoder=None):
    slots, header = staged(payloads, ports)
    recv = np.arange(len(payloads), dtype=np.int64) * 1000 + 7
    out = (decoder or GpuSbeDecoder()).decode(slots, header, np.asarray(ports), recv, directory())
    return out, recv


def test_without_a_directory_or_with_other_ports_everything_is_stale():
    payloads = synthetic_stream(np.random.default_rng(1), 5, UNIVERSE, MIDNIGHT)
    ports, recv = np.full(5, PORT), np.arange(5, dtype=np.int64)
    slots, header = staged(payloads, ports)
    assert GpuSbeDecoder().decode(slots, header, ports, recv, None) == ([], DecodeCounts(0, 5, 0))
    (batches, counts), _ = decode(payloads, np.full(5, PORT + 1))
    assert batches == [] and counts == DecodeCounts(0, 5, 0)


def test_mixed_packets_match_the_host_decode_of_the_accepted_ones():
    good = synthetic_stream(np.random.default_rng(2), 120, UNIVERSE, MIDNIGHT)
    payloads = fuzz(good, 360, seed=5)
    rng = np.random.default_rng(3)
    ports = np.where(rng.random(len(payloads)) < 0.2, PORT + 1, PORT)
    (batches, counts), recv = decode(payloads, ports)
    status, nev, wire, seq, _ = host_decode(payloads)
    fresh = ports == PORT
    accepted = np.flatnonzero(fresh & (status == 0))
    assert counts == DecodeCounts(
        messages=int(wire[accepted].sum()),
        stale=int((~fresh).sum()),
        malformed=int((fresh & (status != 0)).sum()),
    )
    assert counts.malformed > 0 and counts.stale > 0
    (batch,) = batches
    assert batch.generation == 9 and batch.n_packets == len(accepted)
    np.testing.assert_array_equal(batch.sequence, seq[accepted])
    np.testing.assert_array_equal(batch.wire_count, wire[accepted])
    np.testing.assert_array_equal(batch.packet_recv_ns, recv[accepted])
    assert batch.recv_wall_ns == recv[accepted[-1]]
    np.testing.assert_array_equal(batch.packet_offsets, np.r_[0, np.cumsum(nev[accepted])[:-1]])
    # Stale and rejected packets emit nothing: the columns hold exactly the accepted ones.
    _, _, _, _, events = host_decode([payloads[i] for i in accepted])
    native = events.astype(events.dtype.newbyteorder("="))
    assert batch.n_events == len(events)
    for name, dt in FIELDS:
        got = cp.asnumpy(batch.columns[name])
        want = native[name].astype(dt)
        if np.dtype(dt).kind == "f":
            got, want = got.view(np.int64), want.view(np.int64)
        np.testing.assert_array_equal(got, want, err_msg=name)


def test_only_malformed_packets_give_no_batch():
    good = synthetic_stream(np.random.default_rng(4), 3, UNIVERSE, MIDNIGHT)
    broken = [p[:-1] for p in good]
    (batches, counts), _ = decode(broken, np.full(3, PORT))
    assert batches == [] and counts == DecodeCounts(0, 0, 3)


def test_a_latency_capture_counts_every_packet_of_a_decoded_batch():
    """``CaptureTap.record_gpu_batch`` on a decoded batch counts exactly what recording the
    host decode's packets one at a time counts."""
    payloads = synthetic_stream(np.random.default_rng(6), 30, UNIVERSE, MIDNIGHT)
    (batches, _), recv = decode(payloads, np.full(30, PORT))
    _, nev, wire, seq, _ = host_decode(payloads)
    tap, want = CaptureTap(), LatencyCapture(60, 9, start_ns=int(recv[3]))
    tap.capture = LatencyCapture(60, 9, start_ns=int(recv[3]))  # the first 3 fall outside
    tap.record_gpu_batch(batches[0])
    for k in range(30):
        events = np.zeros(int(nev[k]), EVENT_DTYPE)
        want.record_packet(
            ReplayBatch(events, 9, int(recv[k]), int(seq[k]), wire_count=int(wire[k]))
        )
    got = tap.capture
    assert want.packet_count == 27 and want.events == int(nev[3:].sum())
    for key in ("packet_count", "events", "wire_messages", "sequence_gaps", "reordered"):
        assert getattr(got, key) == getattr(want, key), key
    assert got.expected_sequence == want.expected_sequence and not got.generation_changed


def test_decode_warm_up_runs_once_per_device(monkeypatch, caplog):
    monkeypatch.setattr(warmup, "_warm", set())
    with caplog.at_level(logging.INFO, logger=warmup.__name__):
        first = warmup.warm_decode_kernels(0)
        assert warmup.warm_decode_kernels(0) == 0.0
    assert first > 0
    assert [r.getMessage().split(" in ")[0] for r in caplog.records] == ["GPU decode kernels ready"]
