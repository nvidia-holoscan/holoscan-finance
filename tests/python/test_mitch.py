# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MIT303 byte fixtures, lossy-field policy, session isolation and receive paths."""

import socket
import time

import numpy as np
import pytest

from holoscan.equities import mitch as replay
from holoscan.equities.capture import LatencyCapture
from holoscan.equities.stream import EquityState
from holoscan.marketdata import mitch
from holoscan.marketdata.bars import QuoteBars
from holoscan.marketdata.wire import CONDITION, EVENT_DTYPE, HALTED, QUOTE, TRADE, ReplayBatch

# Independently transcribed from MIT303 11.9 sections 4.6, 4.9.1, 4.9.16 and
# Appendix B: sequence 7, Time 09:30:00, q offset 123000ns, instrument 1,
# bid -1 x 3, ask 2 x 4. Price negatives use the high sign bit, not two's complement.
GOLDEN = bytes.fromhex(
    "30 00 02 41 07 00 00 00"
    "06 54 98 85 00 00"
    "22 71 78 e0 01 00 01 00 00 00"
    "00 e1 f5 05 00 00 00 80 03 00 00 00"
    "00 c2 eb 0b 00 00 00 00 04 00 00 00"
)
MIDNIGHT = replay.day_start_ns("2025-04-01")


def rows(kinds=(QUOTE, TRADE, CONDITION)):
    out = np.zeros(len(kinds), dtype=EVENT_DTYPE)
    out["kind"] = kinds
    out["timestamp_ns"] = MIDNIGHT + 34200 * 10**9 + np.arange(len(kinds)) * 10**9
    out["bid"], out["ask"], out["price"] = 100, 101, 100.5
    out["bid_size"], out["ask_size"], out["volume"] = 3, 4, 5
    return out


def decode(data):
    return replay.decode_packet(data, 1000, generation=1, midnight_ns=MIDNIGHT, universe=1)


def manifest(generation=1, port=50001):
    return dict(
        format="nyq-equity-replay-v1",
        wire_protocol=replay.PROTOCOL,
        market_timezone=replay.TIMEZONE,
        symbols=["LONG_RIC"],
        generation=generation,
        sender_port=port,
        date="2025-04-01",
        start_ns=MIDNIGHT,
        end_ns=MIDNIGHT + 86400 * 10**9,
        events=100,
        profile=[],
    )


def test_specification_golden_bytes_and_sign_magnitude():
    result, count = decode(GOLDEN)
    b = result[0]
    assert count == 2 and b.sequence == 7
    assert b.events["timestamp_ns"].tolist() == [MIDNIGHT + 34200 * 10**9 + 123000]
    assert b.events["bid"].tolist() == [-1]
    assert b.events["ask_size"].tolist() == [4]
    source = rows((QUOTE,))
    source["timestamp_ns"] += 123000
    source["bid"], source["ask"] = -1, 2
    encoded, count = replay.pack_events(source, 7, MIDNIGHT)
    assert encoded == GOLDEN and count == 2
    assert mitch.price_to_wire(-1).to_bytes(8, "little") == bytes.fromhex("00e1f50500000080")


def test_messages_preserve_analytics_inputs():
    source = rows()
    source["flags"][-1] = HALTED
    data, n = replay.pack_events(source, 100, MIDNIGHT)
    group, seq, messages = mitch.parse_unit(data)
    assert (group, seq, n) == (ord("A"), 100, 6)
    assert [bytes(m[:2]) for m in messages] == [
        b"\x06T",
        b"\x22q",
        b"\x06T",
        b"\x23P",
        b"\x06T",
        b"\x1cH",
    ]
    assert int.from_bytes(messages[3][24:32], "little") == 103  # replay trade ID
    assert messages[3][14:16] == b"\0\0"  # reserved fields
    batches, count = decode(data)
    assert count == 6 and batches[0].sequence_count == 6
    a, b = QuoteBars(["A"]), QuoteBars(["A"])
    a.observe(ReplayBatch(source, 1, 1000, 100))
    b.observe(batches[0])
    np.testing.assert_equal(a.mid, b.mid)
    np.testing.assert_equal(a.trade, b.trade)
    np.testing.assert_equal(a.volume, b.volume)


def test_missing_values_and_unrepresentable_quantities():
    source = rows((QUOTE, TRADE))
    source["bid"][0] = np.nan
    source["volume"][1] = np.nan
    data, _ = replay.pack_events(source, 1, MIDNIGHT)
    events = decode(data)[0][0].events
    assert events["bid"][0] == 0 and events["volume"][1] == 0
    for invalid in (1.5, 2**32):
        source["ask_size"][0] = invalid
        with pytest.raises(ValueError, match="UInt32"):
            replay.pack_events(source, 1, MIDNIGHT)
    for invalid in (float("nan"), 1e12):
        with pytest.raises(ValueError):
            mitch.price_to_wire(invalid)


@pytest.mark.parametrize(
    "offset,value", [(0, 49), (2, 3), (3, 66), (8, 5), (9, 88), (15, 88), (20, 0)]
)
def test_malformed_or_unsupported_units_fail_closed(offset, value):
    bad = bytearray(GOLDEN)
    bad[offset] = value
    with pytest.raises(ValueError):
        decode(bytes(bad))


def test_missing_time_trailing_bytes_and_sequence_overflow():
    for data in (GOLDEN[:-1], GOLDEN + b"x", GOLDEN[:7], mitch.pack_unit([GOLDEN[14:]], 1)):
        with pytest.raises(ValueError):
            decode(data)
    for sequence in (0, 2**32 - 1):
        with pytest.raises(ValueError):
            decode(mitch.pack_unit([GOLDEN[8:14], GOLDEN[14:]], sequence))


def test_seek_rejects_queued_old_sender_and_counts_all_wire_messages():
    state = EquityState()
    state.set_manifest(manifest())
    p, count = replay.pack_events(rows((QUOTE,)), 1, MIDNIGHT)
    state.observe(state.decode(p, 1000, ("127.0.0.1", 50001))[0])
    assert state.expected_sequence == count + 1 and state.bars.events == 1
    state.set_manifest(manifest(2, 50002))
    assert state.decode(p, 1100, ("127.0.0.1", 50001)) == ([], 0)
    state.observe(state.decode(p, 1200, ("127.0.0.1", 50002))[0])
    assert state.bars.events == 1 and state.sequence_gaps == 0
    with pytest.raises(ValueError, match="port must be new"):
        state.set_manifest(manifest(3, 50001))


def test_capture_counts_wire_messages_separately_from_observations():
    cap = LatencyCapture(1, 1, start_ns=100)
    for sequence in (1, 3):
        cap.record_packet(ReplayBatch(rows((QUOTE,)), 1, 200, sequence, wire_count=2))
    result = cap.finish(now_ns=cap.end_ns)
    assert result["observations"] == 2 and result["wire_messages"] == 4
    assert result["packets"] == 2 and result["sequence_gaps"] == 0


def test_mitch_socket_and_daqiri_receive_identical_units(fragment, monkeypatch, op_output_factory):
    import sys

    from test_ano_source_op import eth_ip_udp_frame, make_fake_daqiri

    from holoscan.equities.host_sources import AnoMitchSourceOp, AsyncUdpMitchSourceOp

    data, _ = replay.pack_events(rows(), 1, MIDNIGHT)
    outputs = []
    for backend in ("async", "ano"):
        state = EquityState(ingest=backend)
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sender.bind(("127.0.0.1", 0))
        port = sender.getsockname()[1] if backend == "async" else 55555
        state.set_manifest(manifest(port=port))
        if backend == "ano":
            fake = make_fake_daqiri([[eth_ip_udp_frame(data)]])
            monkeypatch.setitem(sys.modules, "daqiri", fake)
            cls = AnoMitchSourceOp
        else:
            cls = AsyncUdpMitchSourceOp
        op = cls(fragment, port=0, datagram_decoder=state.decode, batch_size=1024, keep_alive=True)
        op.start()
        try:
            if backend == "async":
                sender.sendto(data, op._sock.getsockname())
            deadline = time.monotonic() + 5
            while not op.packets and time.monotonic() < deadline:
                time.sleep(0.005)
            assert op.packets == 1
            out = op_output_factory()
            op.compute(None, out, None)
            batches, _ = out.emitted
            outputs.append(batches[0].events)
            assert op.messages == 6 and op.malformed_packets == 0
        finally:
            op.stop()
            sender.close()
    for field in EVENT_DTYPE.names:
        np.testing.assert_equal(outputs[0][field], outputs[1][field])
