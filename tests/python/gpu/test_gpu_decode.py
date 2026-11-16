# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The GPU SBE kernels are bit-compatible with the host code: the decoder with every field
and every rejection rule of ``decode_packet`` (golden bytes, randomized streams, mutations,
value edges, schema extensions, the receive-header fast path), the encoder byte for byte
with ``pack_units`` (and its errors), the frame builder with the pcap writer, and the
Ethernet/IPv4/UDP gather kernel with ``strip_udp_datagram``. The full-day version of these
checks is ``python -m holoscan_finance.tools.gpu_validate``.

``to_slots``, ``host_decode`` and ``compare`` are the test oracle the other GPU tests share.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest
from fakes.fake_drain import receive_header
from library.test_equity_mapping import mixed_rows
from library.test_sbe_codec import GOLDEN, extended

from holoscan_finance.equity import mapping as replay
from holoscan_finance.equity.synthetic import _FrameWriter, fuzz, synthetic_stream
from holoscan_finance.marketdata import sbe_codec as sbe
from holoscan_finance.marketdata.framing import strip_udp_datagram, udp_frame
from holoscan_finance.marketdata.gpu_sbe import (
    FIELDS,
    HEADER_COLS,
    PACKET_HEADER,
    SLOT_BYTES,
    columns_from_events,
    decode_packets,
    encode_units,
    frame_units,
    inspect_packets,
    receive_frames,
)
from holoscan_finance.marketdata.records import EVENT_DTYPE

pytestmark = pytest.mark.gpu
cp = pytest.importorskip("cupy")

MIDNIGHT = replay.day_start_ns("2026-06-01")
UNIVERSE = 50


def to_slots(payloads):
    """A device slot buffer (one payload per row) and its int32 lengths."""
    host = np.zeros((len(payloads), SLOT_BYTES), np.uint8)
    for i, p in enumerate(payloads):
        host[i, : len(p)] = np.frombuffer(p, np.uint8)
    return cp.asarray(host), cp.asarray([len(p) for p in payloads], cp.int32)


def host_decode(payloads):
    """Per packet: status (0 decoded, 1 framing, 2 content), observations, wire entries and
    sequence; plus every observation, all from the host decoder."""
    n = len(payloads)
    status, nev = np.zeros(n, np.int32), np.zeros(n, np.int32)
    wire, seq = np.zeros(n, np.int32), np.zeros(n, np.uint64)
    rows = []
    for i, data in enumerate(payloads):
        try:
            seq[i], wire[i] = sbe.inspect_packet(data)
        except (ValueError, struct.error):
            status[i] = 1
            continue
        try:
            (batch,), _ = replay.decode_packet(
                data, 0, generation=1, midnight_ns=MIDNIGHT, universe=UNIVERSE
            )
        except (ValueError, struct.error):
            status[i] = 2
            continue
        nev[i] = len(batch.events)
        rows.append(batch.events)
    events = np.concatenate(rows) if rows else np.zeros(0, EVENT_DTYPE)
    return status, nev, wire, seq, events


def compare(host, gpu):
    """Mismatch counts between ``host_decode`` and ``decode_packets`` (all zero = identical);
    floats are compared as bit patterns, so NaN placement and signed zeros count."""
    status, nev, wire, seq, events = host
    out = {"status_mismatch": int((status != gpu.status).sum())}
    ok, framed = status == 0, status != 1
    out["events_per_packet_mismatch"] = int((nev[ok] != gpu.events_per_packet[ok]).sum())
    out["wire_count_mismatch"] = int((wire[framed] != gpu.wire_count[framed]).sum())
    out["sequence_mismatch"] = int((seq[framed] != gpu.sequence[framed]).sum())
    out["event_count_mismatch"] = int(len(events) != gpu.total_events)
    if out["event_count_mismatch"]:
        return out
    native = events.astype(events.dtype.newbyteorder("="))
    for name, dt in FIELDS:
        h, g = native[name], cp.asnumpy(gpu.columns[name])
        if np.dtype(dt).kind == "f":
            bad = h.view(np.int64) != g.view(np.int64)
        else:
            bad = h.astype(np.int64) != g.astype(np.int64)
        out[f"field_{name}_mismatch"] = int(bad.sum())
    return out


def decode_both(payloads, **kw):
    slots, lens = to_slots(payloads)
    gpu = decode_packets(slots, lens, midnight_ns=MIDNIGHT, universe=UNIVERSE, **kw)
    return host_decode(payloads), gpu


def assert_identical(host, gpu):
    bad = {k: v for k, v in compare(host, gpu).items() if v}
    assert not bad, bad


def test_golden_packet_decodes_identically():
    host, gpu = decode_both([GOLDEN])
    assert_identical(host, gpu)
    assert gpu.total_events == 1
    assert float(gpu.columns["bid"][0]) == 1.0 and float(gpu.columns["ask"][0]) == 2.0
    assert np.isnan(float(gpu.columns["price"][0])) and int(gpu.columns["flags"][0]) == 0
    assert int(gpu.sequence[0]) == 7 and gpu.sequence.dtype == np.uint64


def test_random_streams_and_mutations_are_bit_identical():
    payloads = synthetic_stream(np.random.default_rng(20260930), 400, UNIVERSE, MIDNIGHT)
    host, gpu = decode_both(payloads)
    assert_identical(host, gpu)
    assert (host[0] == 0).all()
    host, gpu = decode_both(fuzz(payloads, 6000, seed=11))
    assert_identical(host, gpu)
    # Every rejection stage is exercised, not just the happy path.
    assert set(np.unique(host[0]).tolist()) == {0, 1, 2}


def test_receive_header_fast_path_matches_and_compacts_rejects():
    """With the header's entry counts (the sources' path) the decode is one launch; packets
    a content rule rejects leave no hole in the columns."""
    payloads = synthetic_stream(np.random.default_rng(8), 60, UNIVERSE, MIDNIGHT)
    mutated = fuzz(payloads, 300, seed=5)
    slots, lens = to_slots(mutated)
    header = inspect_packets(slots, lens).get()
    framed, counts, seqs = header[:, 3] == 0, header[:, 4], header[:, 5].view(np.uint64)
    host = host_decode(mutated)
    np.testing.assert_array_equal(framed, host[0] != 1)
    np.testing.assert_array_equal(counts[framed], host[2][framed])
    np.testing.assert_array_equal(seqs[framed], host[3][framed])
    np.testing.assert_array_equal(header[:, 0], lens.get())
    # The sources pass only framed packets with their header rows and the header's entry
    # total; the content rejects among them are squeezed out.
    keep_framed = [p for p, f in zip(mutated, framed) if f]
    slots, lens = to_slots(keep_framed)
    header_d = inspect_packets(slots, lens)
    gpu = decode_packets(
        slots,
        header=header_d,
        midnight_ns=MIDNIGHT,
        universe=UNIVERSE,
        max_events=int(counts[framed].sum()),
    )
    assert_identical(host_decode(keep_framed), gpu)
    assert (gpu.status == 2).any()
    # A stale-sender mask and a sender port on the fast path: masked packets emit nothing.
    keep = np.arange(len(keep_framed)) % 3 != 0
    h = host_decode(keep_framed)
    for kw in ({"keep": keep}, {"keep": cp.asarray(keep)}):
        gpu = decode_packets(slots, header=header_d, midnight_ns=MIDNIGHT, universe=UNIVERSE, **kw)
        assert gpu.total_events == int(h[1][keep & (h[0] == 0)].sum())
        assert (gpu.events_per_packet[~keep] == 0).all()
        np.testing.assert_array_equal(gpu.status, h[0])  # validated even when not emitted
    header_d[:, 1] = 7  # the inspect header has no port; the gather kernel's does
    header_d[::2, 1] = 8
    gpu = decode_packets(
        slots, header=header_d, midnight_ns=MIDNIGHT, universe=UNIVERSE, sender_port=7
    )
    odd = np.arange(len(keep_framed)) % 2 == 1
    assert gpu.total_events == int(h[1][odd & (h[0] == 0)].sum())
    assert (gpu.events_per_packet[~odd] == 0).all()


def test_value_edges_and_schema_extensions_decode_identically():
    data = bytearray(GOLDEN)
    for bid, ask in (
        (2**63 - 1, sbe.PRICE_NULL + 1),
        (2**53 + 1, -(2**53) - 1),
        (12345678, -87654321),
        (sbe.PRICE_NULL, 0),
    ):
        struct.pack_into("<q", data, 24 + 8, bid)
        struct.pack_into("<q", data, 24 + 16, ask)
        host, gpu = decode_both([bytes(data)])
        assert_identical(host, gpu)
        if bid != sbe.PRICE_NULL:
            assert float(gpu.columns["bid"][0]) == float(bid) / 10000.0
    for qty in (sbe.QTY_NULL, sbe.QTY_NULL - 1, 0):
        struct.pack_into("<I", data, 24 + 32, qty)
        assert_identical(*decode_both([bytes(data)]))
    for sequence in (1, 2**63 - 3, 2**63 - 2, 2**63 - 1, 2**63, 2**64 - 1):
        struct.pack_into("<Q", data, 8, sequence)
        assert_identical(*decode_both([bytes(data)]))
    # A later schema version: longer root block and entries, read through the wire lengths.
    extensions = [extended(), extended(8, 0), extended(0, 16), extended(16, 8)]
    host, gpu = decode_both(extensions)
    assert_identical(host, gpu)
    assert (gpu.status == 0).all() and gpu.total_events == 4
    host, gpu = decode_both([extended(4, 0), extended(1, 0), extended(0, 4), extended(0, 1)])
    assert_identical(host, gpu)
    assert (gpu.status == 1).all()
    # More entries than a warp has lanes (a jumbo frame, beyond pack_packet's MTU budget):
    # the lanes loop.
    rows = mixed_rows(np.random.default_rng(1), 40, 40)
    rows["symbol_id"] %= UNIVERSE
    many = replay.entries_of(rows, MIDNIGHT)
    body = sbe.HEADER.pack(12, 1, sbe.SCHEMA_ID, 0) + (1).to_bytes(8, "little") + bytes(4)
    body += sbe.GROUP_SIZE.pack(48, 40) + many.tobytes()
    host, gpu = decode_both([body])
    assert_identical(host, gpu)
    assert gpu.total_events == 40


def test_gpu_encoder_writes_pack_units_bytes_and_raises_its_errors():
    rng = np.random.default_rng(3)
    for unit_events in (replay.MAX_PACKET_EVENTS, 7, 1):
        for n in (1, unit_events + 1, 1000):
            rows = mixed_rows(rng, n, unit_events)
            sequence = int(rng.integers(1, 10**6))
            data, offsets, lengths, counts = replay.pack_units(
                rows, sequence, MIDNIGHT, unit_events
            )
            d_data, d_offsets, d_lengths, d_counts = encode_units(
                columns_from_events(rows), sequence, MIDNIGHT, unit_events
            )
            assert d_data.get().tobytes() == data.tobytes(), (unit_events, n)
            np.testing.assert_array_equal(d_offsets, offsets)
            np.testing.assert_array_equal(d_lengths, lengths)
            np.testing.assert_array_equal(d_counts, counts)
    # Encode on the GPU, decode on the GPU: the columns come back equal (NaN for NaN; a
    # -0.0 price is the mantissa 0, so it comes back as +0.0).
    rows = mixed_rows(rng, 300, 30)
    columns = columns_from_events(rows)
    data, offsets, lengths, _ = encode_units(columns, 1, MIDNIGHT, 30)
    payloads = [data[o : o + ln].get().tobytes() for o, ln in zip(offsets, lengths)]
    gpu = decode_packets(*to_slots(payloads), midnight_ns=MIDNIGHT, universe=3000)
    assert gpu.total_events == 300
    for name, _ in FIELDS:
        a, b = cp.asnumpy(columns[name]), cp.asnumpy(gpu.columns[name])
        np.testing.assert_array_equal(a, b, err_msg=name)
    # The same ValueError for the same input.
    base = mixed_rows(rng, 5, 30)
    cases = []
    bad = base.copy()
    bad["timestamp_ns"][2] = bad["timestamp_ns"][1] - 1
    cases.append((bad, "ordered"))
    bad = base.copy()
    bad["timestamp_ns"][0] = MIDNIGHT - 1
    cases.append((bad, "ordered"))
    bad = base.copy()
    bad["kind"][1] = 7
    cases.append((bad, "kind"))
    bad = base.copy()
    bad["symbol_id"][3] = 0x10000
    cases.append((bad, "SecurityID"))
    bad = base.copy()
    bad["kind"][:] = 1
    bad["bid"][4] = 1e15
    cases.append((bad, "Price"))
    bad = base.copy()
    bad["kind"][:] = 2
    bad["price"][0] = np.inf
    cases.append((bad, "Price|finite"))
    bad = base.copy()
    bad["kind"][:] = 1
    bad["bid_size"][2] = 1.5
    cases.append((bad, "QtyNULL"))
    bad = base.copy()
    bad["kind"][:] = 2
    bad["volume"][2] = -1
    cases.append((bad, "QtyNULL"))
    for rows, match in cases:
        with pytest.raises(ValueError, match=match):
            replay.pack_units(rows, 1, MIDNIGHT, 30)
        with pytest.raises(ValueError, match=match):
            encode_units(columns_from_events(rows), 1, MIDNIGHT, 30)
    with pytest.raises(ValueError, match="sequence"):
        encode_units(columns_from_events(base), 0, MIDNIGHT, 30)
    with pytest.raises(ValueError, match="1..30"):
        encode_units(columns_from_events(base), 1, MIDNIGHT, 31)


def test_gpu_frames_are_the_pcap_writer_bytes():
    rng = np.random.default_rng(12)
    rows = mixed_rows(rng, 95, 30)
    data, offsets, lengths, _ = replay.pack_units(rows, 11, MIDNIGHT, 30)
    stamps = rows["timestamp_ns"][np.minimum(np.arange(1, len(offsets) + 1) * 30, 95) - 1]
    stamps = stamps.astype(np.int64)
    kept = np.array([True, False, True, True])
    reference = _FrameWriter(40123, 26400, MIDNIGHT, 30).frames(
        data, offsets[kept], lengths[kept], stamps[kept]
    )
    d_data = encode_units(columns_from_events(rows), 11, MIDNIGHT, 30)[0]
    frames = frame_units(
        d_data, offsets[kept], lengths[kept], stamps[kept], src_port=40123, dst_port=26400
    )
    assert frames.get().tobytes() == reference
    assert (
        len(frame_units(d_data, offsets[:0], lengths[:0], stamps[:0], src_port=1, dst_port=2)) == 0
    )


def test_gather_matches_host_strip():
    good = synthetic_stream(np.random.default_rng(3), 20, UNIVERSE, MIDNIGHT)
    frames = [udp_frame(p, src_port=40000 + i) for i, p in enumerate(good)]
    tcp, fragment, ipv6 = (bytearray(udp_frame(good[k], src_port=1)) for k in range(3))
    tcp[23] = 6
    fragment[20:22] = b"\x20\x00"
    ipv6[12:14] = b"\x86\xdd"
    frames += [
        bytes(tcp),
        bytes(fragment),
        bytes(ipv6),
        udp_frame(good[3], src_port=1)[: 14 + 20 + 8 + len(good[3]) - 10],  # truncated
        udp_frame(b"\x00" * 4, src_port=1),  # payload shorter than an SBE message prefix
        udp_frame(good[4], src_port=2, ihl=6),  # IPv4 options
        udp_frame(good[5], src_port=3) + b"\x00" * 4,  # Ethernet padding / FCS
        udp_frame(GOLDEN[:-1], src_port=4),  # UDP fine, but the entry is cut short
        udp_frame(GOLDEN[:20] + b"\x00" * 4, src_port=5),  # UDP fine, SBE framing not
        udp_frame(extended(), src_port=6),  # a later schema version
    ]
    stride = 2048
    host_buf = np.zeros((len(frames), stride), np.uint8)
    for i, f in enumerate(frames):
        host_buf[i, : len(f)] = np.frombuffer(f, np.uint8)
    dev = cp.asarray(host_buf)
    addresses = dev.data.ptr + np.arange(len(frames), dtype=np.uint64) * stride
    packed = cp.asarray(np.concatenate([addresses, np.array([len(f) for f in frames], np.uint64)]))
    slots = cp.zeros((len(frames), SLOT_BYTES), cp.uint8)
    header = receive_frames(packed, len(frames), slots).get()
    assert header.shape == (len(frames), HEADER_COLS)
    slots = slots.get()
    for i, f in enumerate(frames):
        expected = strip_udp_datagram(f, minimum_payload=PACKET_HEADER)
        if expected is None:
            assert header[i, 0] < 0 and tuple(header[i, 3:]) == (1, 0, 0), i
            continue
        payload, peer = expected
        assert len(payload) == header[i, 0], i
        assert peer[1] == header[i, 1], i
        assert slots[i, : len(payload)].tobytes() == payload, i
        ok, sequence, count = sbe.frame_fields(payload)
        assert header[i, 3] == (0 if ok else 1), i
        assert tuple(header[i, 4:]) == (count, sequence), i
    # The fake drain's receive_header reproduces the kernel's rows, framing rejects included.
    rows = []
    for f in frames:
        got = strip_udp_datagram(f, PACKET_HEADER)
        rows.append((None, -1, 0) if got is None else (got[0], got[1][1], 0))
    fake = receive_header(rows).T
    np.testing.assert_array_equal(fake[:, 3:], header[:, 3:])
    np.testing.assert_array_equal(fake[:, 0], header[:, 0])
