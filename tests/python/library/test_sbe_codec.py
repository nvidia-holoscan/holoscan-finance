# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FIX SBE 1.0 byte fixtures, the schema layout, PriceNULL4/QtyNULL conversion, message
framing and the schema extension rules, fail-closed."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from holoscan_finance.marketdata import sbe_codec as sbe
from holoscan_finance.marketdata.records import CONDITION, EVENT_DTYPE, HALTED, QUOTE, TRADE
from holoscan_finance.nyq import mapping as replay

# Transcribed by hand from the SBE 1.0 specification (message header, root block, group
# dimensions, little-endian fields in schema order) and sbe_schema.xml: a message of sequence
# 7 with one Quote entry for SecurityID 0 at 2025-04-01 09:30:00.000123000 New York
# (1,743,514,200,000,123,000 ns since the epoch): bid 1.0000 x 3, offer 2.0000 x 4, no last
# price or quantity, no trading status.
GOLDEN = bytes.fromhex(
    "0c 00"  # messageHeader.blockLength 12 (SeqNum and 4 octets of padding)
    "01 00"  # templateId 1, MarketDataIncrementalRefresh
    "46 48"  # schemaId 18502
    "00 00"  # version 0
    "07 00 00 00 00 00 00 00"  # SeqNum 7
    "00 00 00 00"  # root block padding
    "30 00"  # groupSizeEncoding.blockLength 48
    "01 00"  # numInGroup 1
    "78 50 db 67 ed 34 32 18"  # TransactTime
    "10 27 00 00 00 00 00 00"  # BidPx mantissa 10000 (1.0000)
    "20 4e 00 00 00 00 00 00"  # OfferPx mantissa 20000 (2.0000)
    "00 00 00 00 00 00 00 80"  # LastPx null (int64 min)
    "03 00 00 00"  # BidSize 3
    "04 00 00 00"  # OfferSize 4
    "ff ff ff ff"  # LastQty null
    "00 00"  # SecurityID 0
    "01"  # EntryType Quote
    "ff"  # SecurityTradingStatus null
)
MIDNIGHT = replay.day_start_ns("2025-04-01")
OPEN = MIDNIGHT + 34200 * 10**9
ENTRY_AT = 24  # the first entry's offset


def rows(kinds=(QUOTE, TRADE, CONDITION)):
    out = np.zeros(len(kinds), dtype=EVENT_DTYPE)
    out["kind"] = kinds
    out["timestamp_ns"] = OPEN + np.arange(len(kinds)) * 10**9
    out["bid"], out["ask"], out["price"] = 100, 101, 100.5
    out["bid_size"], out["ask_size"], out["volume"] = 3, 4, 5
    return out


def decode(data, universe=1):
    return replay.decode_packet(data, 1000, generation=1, midnight_ns=MIDNIGHT, universe=universe)


def test_specification_golden_bytes_and_price_null4():
    result, count = decode(GOLDEN)
    b = result[0]
    assert count == 1 and b.sequence == 7 and b.sequence_count == 1
    assert b.events["timestamp_ns"].tolist() == [OPEN + 123000]
    assert b.events["kind"].tolist() == [QUOTE] and b.events["symbol_id"].tolist() == [0]
    assert b.events["flags"].tolist() == [0]
    assert b.events["bid"].tolist() == [1.0] and b.events["ask"].tolist() == [2.0]
    assert b.events["bid_size"].tolist() == [3] and b.events["ask_size"].tolist() == [4]
    assert np.isnan(b.events["price"][0]) and np.isnan(b.events["volume"][0])
    source = rows((QUOTE,))
    source["timestamp_ns"] += 123000
    source["bid"], source["ask"] = 1, 2
    source["price"] = source["volume"] = np.nan
    encoded, count = replay.pack_events(source, 7, MIDNIGHT)
    assert encoded == GOLDEN and count == 1
    assert sbe.price_to_wire(1.0) == 10000 and sbe.price_to_wire(-1.5) == -15000
    assert sbe.price_from_wire(sbe.price_to_wire(123.4567)) == 123.4567
    assert sbe.price_to_wire(0.00025) == 2 and sbe.price_to_wire(0.00035) == 4  # ties to even
    assert sbe.price_to_wire(float("nan")) == sbe.PRICE_NULL == -(2**63)
    assert np.isnan(sbe.price_from_wire(sbe.PRICE_NULL))
    # float(mantissa) / 10000.0: two roundings, the same on every decoder.
    big = 2**53 + 1
    assert sbe.price_from_wire(big) == float(big) / 10000.0
    assert sbe.qty_to_wire(float("nan")) == sbe.QTY_NULL == 2**32 - 1
    assert sbe.qty_to_wire(0) == 0 and np.isnan(sbe.qty_from_wire(sbe.QTY_NULL))


def test_schema_layout_is_the_documented_one():
    """The codec derives its layout from sbe_schema.xml; these are the numbers the GPU and
    numba decoders, the docs and the golden bytes assume."""
    assert (sbe.SCHEMA_ID, sbe.TEMPLATE_ID, sbe.VERSION) == (18502, 1, 0)
    assert sbe.ORDER == "<" and sbe.HEADER.size == 8 and sbe.GROUP_SIZE.size == 4
    assert (sbe.ROOT_BLOCK_LENGTH, sbe.ENTRY_BLOCK_LENGTH, sbe.PACKET_HEADER) == (12, 48, 24)
    assert sbe.MAX_ENTRIES == 30 and 24 + 30 * 48 <= sbe.MAX_PAYLOAD < 24 + 31 * 48
    offsets = {f.name: (f.offset, f.size) for f in sbe.SCHEMA.entries.fields}
    assert offsets == {
        "TransactTime": (0, 8),
        "BidPx": (8, 8),
        "OfferPx": (16, 8),
        "LastPx": (24, 8),
        "BidSize": (32, 4),
        "OfferSize": (36, 4),
        "LastQty": (40, 4),
        "SecurityID": (44, 2),
        "EntryType": (46, 1),
        "SecurityTradingStatus": (47, 1),
    }
    ids = {f.name: f.id for f in sbe.SCHEMA.entries.fields}
    assert ids["TransactTime"] == 60 and ids["SecurityID"] == 48 and ids["LastPx"] == 31
    assert (ids["BidPx"], ids["OfferPx"], ids["BidSize"], ids["OfferSize"]) == (132, 133, 134, 135)
    assert ids["LastQty"] == 32 and ids["SecurityTradingStatus"] == 326
    assert sbe.SCHEMA.group_id == 268 and sbe.SCHEMA.root["SeqNum"].offset == 0
    assert sbe.SCHEMA.price_exponent == -4 and sbe.PRICE_SCALE == 10000
    assert (sbe.QUOTE, sbe.TRADE, sbe.TRADING_STATUS) == (QUOTE, TRADE, CONDITION) == (1, 2, 3)
    assert (sbe.TRADING_HALT, sbe.READY_TO_TRADE, sbe.STATUS_NULL) == (2, 17, 255)
    nulls = {f.name: f.null for f in sbe.SCHEMA.entries.fields}
    assert nulls["BidPx"] == sbe.PRICE_NULL and nulls["BidSize"] == sbe.QTY_NULL
    assert nulls["TransactTime"] is None and nulls["SecurityID"] is None
    assert nulls["EntryType"] is None and nulls["SecurityTradingStatus"] == 255
    assert "#define SBE_E_SECURITYTRADINGSTATUS 47" in sbe.c_defines()
    assert sbe.ENTRY_DTYPE.itemsize == 48 and sbe.entry_dtype(56).itemsize == 56


def test_message_layout_and_sequence_numbers():
    source = rows()
    source["flags"][-1] = HALTED
    data, n = replay.pack_events(source, 100, MIDNIGHT)
    assert n == 3 and len(data) == 24 + 3 * 48
    seq, entries = sbe.parse_packet(data)
    assert seq == 100 and entries.dtype == sbe.ENTRY_DTYPE
    assert entries["EntryType"].tolist() == [QUOTE, TRADE, CONDITION]
    assert entries["SecurityTradingStatus"].tolist() == [255, 255, sbe.TRADING_HALT]
    assert entries["BidPx"].tolist() == [1_000_000, sbe.PRICE_NULL, sbe.PRICE_NULL]
    assert entries["OfferPx"].tolist() == [1_010_000, sbe.PRICE_NULL, sbe.PRICE_NULL]
    assert entries["LastPx"].tolist() == [sbe.PRICE_NULL, 1_005_000, sbe.PRICE_NULL]
    assert entries["BidSize"].tolist() == [3, sbe.QTY_NULL, sbe.QTY_NULL]
    assert entries["OfferSize"].tolist() == [4, sbe.QTY_NULL, sbe.QTY_NULL]
    assert entries["LastQty"].tolist() == [sbe.QTY_NULL, 5, sbe.QTY_NULL]
    assert (entries["TransactTime"] == source["timestamp_ns"]).all()
    assert sbe.inspect_packet(data) == (100, 3) and sbe.packet_count(data) == 3
    assert struct.unpack_from("<HHHH", data) == (12, 1, 18502, 0)
    assert struct.unpack_from("<HH", data, 20) == (48, 3)
    # A resume is ReadyToTrade; the sequence of each entry is implied by its position.
    resumed, _ = replay.pack_events(rows((CONDITION,)), 1, MIDNIGHT)
    assert sbe.parse_packet(resumed)[1]["SecurityTradingStatus"].tolist() == [sbe.READY_TO_TRADE]


def test_empty_messages_and_packing_limits():
    for sequence in (-1, 0, 2**63 - 1):
        with pytest.raises(ValueError, match="sequence"):
            sbe.pack_packet(GOLDEN[ENTRY_AT:], sequence)
    with pytest.raises(ValueError, match="too many"):
        sbe.pack_packet(np.zeros(0x10000, sbe.ENTRY_DTYPE), 1)
    with pytest.raises(ValueError, match="budget"):
        sbe.pack_packet(GOLDEN[ENTRY_AT:] * 31, 1)
    with pytest.raises(ValueError, match="whole blocks"):
        sbe.pack_packet(GOLDEN[ENTRY_AT:-1], 1)
    heartbeat = sbe.pack_packet([], 0)
    assert len(heartbeat) == 24 and sbe.inspect_packet(heartbeat) == (0, 0)
    seq, entries = sbe.parse_packet(sbe.pack_packet(np.zeros(0, sbe.ENTRY_DTYPE), 5))
    assert (seq, len(entries)) == (5, 0) and sbe.packet_count(heartbeat) == 0
    assert sbe.packet_count(b"") == 0 and decode(heartbeat)[1] == 0
    with pytest.raises(ValueError, match="empty"):
        sbe.inspect_packet(heartbeat + b"x")
    with pytest.raises(ValueError, match="entry count"):
        sbe.inspect_packet(GOLDEN[:-1])
    with pytest.raises(ValueError, match="short"):
        sbe.inspect_packet(GOLDEN[:7])
    with pytest.raises(ValueError, match="short"):
        sbe.inspect_packet(GOLDEN[:23])


@pytest.mark.parametrize(
    "offset,value",
    [
        (0, 8),  # root block too short for SeqNum
        (0, 16),  # root block that misaligns the entries
        (2, 2),  # another template
        (4, 0),  # another schema
        (20, 40),  # entry block shorter than the schema's
        (20, 52),  # entry block that misaligns the next entry
        (22, 0),  # count 0 with an entry present
        (22, 2),  # count beyond the entries present
        (ENTRY_AT + 7, 0xFF),  # timestamp outside the day
        (ENTRY_AT + 44, 1),  # SecurityID beyond the universe
        (ENTRY_AT + 46, 0),  # unknown EntryType
        (ENTRY_AT + 46, 3),  # a TradingStatus entry without a status
        (ENTRY_AT + 47, 1),  # unknown SecurityTradingStatus
    ],
)
def test_malformed_or_unsupported_packets_fail_closed(offset, value):
    bad = bytearray(GOLDEN)
    bad[offset] = value
    with pytest.raises(ValueError):
        decode(bytes(bad))


def test_truncation_trailing_bytes_and_sequence_overflow():
    for data in (GOLDEN[:-1], GOLDEN + b"x", GOLDEN[:7], GOLDEN[:8], GOLDEN[:23], GOLDEN[:24]):
        with pytest.raises(ValueError):
            decode(data)
    for sequence in (0, 2**63 - 1, 2**63, 2**64 - 1):
        bad = bytearray(GOLDEN)
        bad[8:16] = sequence.to_bytes(8, "little")
        with pytest.raises(ValueError, match="sequence"):
            decode(bytes(bad))
    back = bytearray(replay.pack_events(rows((QUOTE, QUOTE)), 1, MIDNIGHT)[0])
    struct.pack_into("<Q", back, ENTRY_AT + 48, OPEN - 1)  # the second entry steps back
    with pytest.raises(ValueError, match="timestamp"):
        decode(bytes(back))


def extended(root_extra=8, entry_extra=8):
    """GOLDEN re-encoded by a later schema version: longer root block and entries."""
    out = bytearray()
    out += struct.pack("<HHHH", 12 + root_extra, 1, 18502, 1)
    out += GOLDEN[8:20] + bytes(root_extra)
    out += struct.pack("<HH", 48 + entry_extra, 1)
    out += GOLDEN[ENTRY_AT:] + bytes(entry_extra)
    return bytes(out)


def test_schema_extension_decodes_through_the_wire_block_lengths():
    """SBE: decoders skip the root block and stride the entries by the lengths on the wire,
    so appended fields and a newer version decode; misaligned extensions are refused."""
    for data in (extended(), extended(8, 0), extended(0, 16), extended(16, 8)):
        (batch,), count = decode(data)
        reference = decode(GOLDEN)[0][0]
        assert count == 1 and batch.sequence == 7
        assert batch.events.tobytes() == reference.events.tobytes()
        assert sbe.inspect_packet(data) == (7, 1)
    for root_extra, entry_extra in ((4, 0), (1, 0), (0, 4), (0, 1)):
        with pytest.raises(ValueError, match="block length"):
            decode(extended(root_extra, entry_extra))


def test_frame_fields_report_as_the_gather_kernel():
    assert sbe.frame_fields(GOLDEN) == (True, 7, 1)
    assert sbe.frame_fields(GOLDEN[:-1]) == (False, 7, 1)  # dimensions read, entries short
    assert sbe.frame_fields(GOLDEN[:20]) == (False, 0, 0)  # no dimensions
    other = bytearray(GOLDEN)
    other[4] = 0
    assert sbe.frame_fields(bytes(other)) == (False, 0, 0)  # another schema: nothing read
    huge = bytearray(GOLDEN)
    huge[8:16] = (2**64 - 1).to_bytes(8, "little")
    assert sbe.frame_fields(bytes(huge)) == (False, 2**64 - 1, 1)
    # The informational octets are not validated: the version and the root padding.
    loose = bytearray(GOLDEN)
    loose[6:8] = b"\x05\x00"
    loose[16:20] = b"\xde\xad\xbe\xef"
    assert decode(bytes(loose))[1] == 1
