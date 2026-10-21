# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MIT303 byte fixtures, sign-magnitude prices, message layouts and fail-closed framing."""

from __future__ import annotations

import numpy as np
import pytest

from holoscan_finance.marketdata import mitch_codec as mitch
from holoscan_finance.marketdata.records import CONDITION, EVENT_DTYPE, HALTED, QUOTE, TRADE
from holoscan_finance.nyq import mapping as replay

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
    assert mitch.price_from_wire(mitch.price_to_wire(-1)) == -1.0


def test_message_layouts_and_reserved_fields():
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
    assert mitch.inspect_unit(data) == (ord("A"), 100, 6)
    assert mitch.LENGTHS == {ord("T"): 6, ord("q"): 34, ord("P"): 35, ord("H"): 28}


def test_price_range_and_unit_limits():
    for invalid in (float("nan"), float("inf"), 1e12):
        with pytest.raises(ValueError):
            mitch.price_to_wire(invalid)
    for sequence, group in ((-1, ord("A")), (2**32, ord("A")), (1, 256)):
        with pytest.raises(ValueError):
            mitch.pack_unit([], sequence, group)
    with pytest.raises(ValueError, match="too many"):
        mitch.pack_unit([GOLDEN[8:14]] * 256, 1)
    with pytest.raises(ValueError, match="budget"):
        mitch.pack_unit([GOLDEN[14:]] * 44, 1)
    assert mitch.parse_unit(mitch.pack_unit([], 0)) == (ord("A"), 0, [])
    with pytest.raises(ValueError, match="heartbeat"):
        mitch.inspect_unit(mitch.HEADER.pack(9, 0, ord("A"), 0) + b"x")


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
