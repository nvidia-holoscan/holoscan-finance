# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FIX Simple Binary Encoding (SBE 1.0) of the equity replay: ``sbe_schema.xml`` read into
layouts, and the host-side framing, packing and parsing of one datagram.

Specification (public): FIX Simple Binary Encoding 1.0, https://www.fixtrading.org/standards/sbe/.
A datagram is exactly one ``MarketDataIncrementalRefresh`` message, little-endian: the 8-octet
message header (``blockLength``, ``templateId``, ``schemaId``, ``version``), the root block
(``SeqNum``, the sequence number of the first entry, padded to 12 octets) and the
``NoMDEntries`` repeating group (``blockLength``, ``numInGroup``, then fixed 48-octet entries,
one per observation). Every entry starts on an 8-octet boundary. As SBE requires, decoders take
the root and entry lengths from the wire, so a later schema version that appends fields still
decodes; the replay receivers additionally require the lengths to keep the entries 8-octet
aligned (the GPU reads each field with one aligned load) and reject the message otherwise.

Prices are ``PriceNULL4``: an int64 mantissa with four implied decimals, ``PRICE_NULL`` when
absent; a price decodes as ``float(mantissa) / 10000.0`` (the int64 converts to the nearest
double, then one IEEE division), the same two roundings on every decoder. Quantities are
``QtyNULL`` (uint32, ``QTY_NULL`` when absent). Timestamps are nanoseconds since the Unix
epoch. The layout constants below (and the ``#define`` block the CUDA module compiles with) come
from the schema, so changing the schema changes every codec, and the golden-bytes test pins the
wire format. This is a replay codec, not a feed handler: no retransmission, no order book.
"""

from __future__ import annotations

import math
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCHEMA_PATH = Path(__file__).with_name("sbe_schema.xml")
_NS = "{http://fixprotocol.io/2016/sbe}"
# SBE primitive -> (NumPy code, octets, default null value).
_PRIMITIVES = {
    "char": ("S1", 1, 0),
    "int8": ("i1", 1, -(1 << 7)),
    "uint8": ("u1", 1, (1 << 8) - 1),
    "int16": ("i2", 2, -(1 << 15)),
    "uint16": ("u2", 2, (1 << 16) - 1),
    "int32": ("i4", 4, -(1 << 31)),
    "uint32": ("u4", 4, (1 << 32) - 1),
    "int64": ("i8", 8, -(1 << 63)),
    "uint64": ("u8", 8, (1 << 64) - 1),
}


@dataclass(frozen=True)
class Field:
    """One wire field of a block: a simple type, the mantissa of a decimal composite or an
    enum, at ``offset`` octets into the block."""

    name: str
    id: int
    offset: int
    size: int
    code: str  # NumPy type code (without byte order)
    null: int | None  # the encoding's null value, None for a required field
    values: dict[str, int] | None = None  # enum valid values


@dataclass(frozen=True)
class Block:
    """The fixed-size fields of a message root, a group entry or a composite."""

    name: str
    fields: tuple[Field, ...]
    size: int  # the block length (declared padding included)

    def __getitem__(self, name: str) -> Field:
        for f in self.fields:
            if f.name == name:
                return f
        raise KeyError(name)

    def dtype(self, byte_order: str, itemsize: int | None = None) -> np.dtype:
        """A structured dtype that views the block in place (``itemsize`` >= size for the
        entries of an extended schema)."""
        return np.dtype(
            {
                "names": [f.name for f in self.fields],
                "formats": [byte_order + f.code if f.size > 1 else f.code for f in self.fields],
                "offsets": [f.offset for f in self.fields],
                "itemsize": self.size if itemsize is None else itemsize,
            }
        )


@dataclass(frozen=True)
class Schema:
    schema_id: int
    version: int
    byte_order: str  # "<" or ">"
    header: Block  # messageHeader
    group_size: Block  # groupSizeEncoding
    template_id: int
    root: Block  # the message root block
    entries: Block  # the one repeating group's entry
    group_id: int
    price_exponent: int
    constants: dict[str, int]  # enum valid values, as NAME_VALUE


class _Types:
    """The ``<types>`` of a schema: size and null of every encoding, composite members."""

    def __init__(self, element: ET.Element) -> None:
        self.simple: dict[str, tuple[str, int, int | None, bool]] = {}  # code, size, null, const
        self.enums: dict[str, tuple[str, dict[str, int]]] = {}  # encoding type, values
        self.composites: dict[str, list[tuple[str, str]]] = {}  # (member name, type name)
        for types in element.findall("types"):
            for node in types:
                self._add(node, "")

    def _add(self, node: ET.Element, prefix: str) -> None:
        name = prefix + node.get("name")
        if node.tag == "type":
            code, size, default = _PRIMITIVES[node.get("primitiveType")]
            if int(node.get("length", "1")) != 1:
                raise ValueError(f"{name}: array types are not used by this codec")
            presence = node.get("presence", "required")
            if presence == "constant":  # takes no space; the value is the text
                self.simple[name] = (code, 0, int(node.text), True)
            else:
                null = int(node.get("nullValue", default)) if presence == "optional" else None
                self.simple[name] = (code, size, null, False)
        elif node.tag == "enum":
            self.enums[name] = (
                node.get("encodingType"),
                {v.get("name"): int(v.text) for v in node.findall("validValue")},
            )
        elif node.tag == "composite":
            members = []
            for child in node:
                self._add(child, name + ".")
                members.append((child.get("name"), name + "." + child.get("name")))
            self.composites[name] = members
        else:
            raise ValueError(f"unsupported schema element <{node.tag}> ({name})")

    def encoding(self, type_name: str) -> tuple[str, int, int | None, dict[str, int] | None]:
        """``(code, size, null, enum values)`` of a field type; a decimal composite is read
        through its mantissa (constant members take no space)."""
        if type_name in self.enums:
            encoding, values = self.enums[type_name]
            code, size, null, _ = self._simple(encoding)
            return code, size, null, values
        if type_name in self.composites:
            variable = [
                (member, self.simple[tn]) for member, tn in self.composites[type_name]
                if tn in self.simple and not self.simple[tn][3]
            ]  # fmt: skip
            if [m for m, _ in variable] != ["mantissa"]:
                raise ValueError(f"{type_name}: only decimal composites are fields here")
            code, size, null, _ = variable[0][1]
            return code, size, null, None
        return self._simple(type_name)[:3] + (None,)

    def _simple(self, type_name: str) -> tuple[str, int, int | None, bool]:
        if type_name in _PRIMITIVES:
            code, size, _ = _PRIMITIVES[type_name]
            return code, size, None, False
        return self.simple[type_name]

    def composite_block(self, name: str) -> Block:
        fields, offset = [], 0
        for i, (member, tn) in enumerate(self.composites[name]):
            code, size, null, const = self.simple[tn]
            if const:
                continue
            fields.append(Field(member, i, offset, size, code, null))
            offset += size
        return Block(name, tuple(fields), offset)

    def constant(self, composite: str, member: str) -> int:
        return self.simple[f"{composite}.{member}"][2]


def _block(types: _Types, element: ET.Element, name: str) -> Block:
    """The fields of a message root or group entry, laid out as SBE does: in order, each at
    its declared ``offset`` or right after the previous one; ``blockLength`` may pad."""
    fields, offset = [], 0
    for node in element.findall("field"):
        code, size, null, values = types.encoding(node.get("type"))
        presence = node.get("presence")  # absent: the field inherits its type's presence
        if presence == "optional" and null is None:
            raise ValueError(f"{node.get('name')}: optional field without a null value")
        if presence == "required":
            null = None
        at = int(node.get("offset", offset))
        if at < offset:
            raise ValueError(f"{node.get('name')}: fields overlap")
        fields.append(Field(node.get("name"), int(node.get("id")), at, size, code, null, values))
        offset = at + size
    declared = int(element.get("blockLength", offset))
    if declared < offset:
        raise ValueError(f"{name}: blockLength {declared} is shorter than its fields ({offset})")
    return Block(name, tuple(fields), declared)


def load_schema(path: str | Path = SCHEMA_PATH) -> Schema:
    """Read a message schema with one message of fixed fields and one repeating group of
    fixed-size entries (the subset this codec implements)."""
    root = ET.parse(path).getroot()
    if root.tag != _NS + "messageSchema":
        raise ValueError(f"{path}: not an SBE message schema")
    types = _Types(root)
    messages = root.findall(_NS + "message")
    if len(messages) != 1:
        raise ValueError("this codec implements a schema of exactly one message")
    message = messages[0]
    groups = message.findall("group")
    if len(groups) != 1 or message.findall("data") or groups[0].findall("group"):
        raise ValueError("this codec implements one flat repeating group and no var data")
    group = groups[0]
    if group.get("dimensionType", "groupSizeEncoding") != "groupSizeEncoding":
        raise ValueError("the group must use groupSizeEncoding")
    constants = {}
    for enum, (_, values) in types.enums.items():
        constants.update({f"{enum}.{k}": v for k, v in values.items()})
    exponents = {
        name: types.constant(name, "exponent")
        for name, members in types.composites.items()
        if any(m == "exponent" for m, _ in members)
    }
    if len(exponents) != 1:
        raise ValueError("expected one decimal composite")
    return Schema(
        schema_id=int(root.get("id")),
        version=int(root.get("version")),
        byte_order="<" if root.get("byteOrder", "littleEndian") == "littleEndian" else ">",
        header=types.composite_block("messageHeader"),
        group_size=types.composite_block("groupSizeEncoding"),
        template_id=int(message.get("id")),
        root=_block(types, message, message.get("name")),
        entries=_block(types, group, group.get("name")),
        group_id=int(group.get("id")),
        price_exponent=next(iter(exponents.values())),
        constants=constants,
    )


SCHEMA = load_schema()
ORDER = SCHEMA.byte_order
SCHEMA_ID, VERSION, TEMPLATE_ID = SCHEMA.schema_id, SCHEMA.version, SCHEMA.template_id
HEADER = struct.Struct(ORDER + "HHHH")  # blockLength, templateId, schemaId, version
GROUP_SIZE = struct.Struct(ORDER + "HH")  # blockLength, numInGroup
ROOT_BLOCK_LENGTH = SCHEMA.root.size  # 12
ENTRY_BLOCK_LENGTH = SCHEMA.entries.size  # 48
PACKET_HEADER = HEADER.size + ROOT_BLOCK_LENGTH + GROUP_SIZE.size  # 24: header, root, dimensions
HEADER_DTYPE = SCHEMA.header.dtype(ORDER)
ROOT_DTYPE = SCHEMA.root.dtype(ORDER)
ENTRY_DTYPE = SCHEMA.entries.dtype(ORDER)
_SEQ = SCHEMA.root["SeqNum"]
assert HEADER.size == SCHEMA.header.size and GROUP_SIZE.size == SCHEMA.group_size.size
assert (_SEQ.offset, _SEQ.size) == (0, 8) and ENTRY_DTYPE.itemsize == ENTRY_BLOCK_LENGTH
assert PACKET_HEADER % 8 == 0 and ENTRY_BLOCK_LENGTH % 8 == 0
MAX_PAYLOAD = 1472  # unfragmented IPv4 UDP payload at a 1500-byte MTU
MAX_ENTRIES = (MAX_PAYLOAD - PACKET_HEADER) // ENTRY_BLOCK_LENGTH  # 30
MAX_SEQUENCE = (1 << 63) - 1  # receive headers and accounting hold sequences as int64
DAY_NS = 86_400 * 10**9
PRICE_SCALE = 10**-SCHEMA.price_exponent  # 10000
PRICE_NULL = SCHEMA.entries["BidPx"].null  # int64 min
QTY_NULL = SCHEMA.entries["BidSize"].null  # uint32 max
STATUS_NULL = SCHEMA.entries["SecurityTradingStatus"].null  # 255
MAX_SECURITY_ID = (1 << (8 * SCHEMA.entries["SecurityID"].size)) - 1
QUOTE, TRADE, TRADING_STATUS = (
    SCHEMA.constants[f"EntryType.{k}"] for k in ("Quote", "Trade", "TradingStatus")
)
TRADING_HALT, READY_TO_TRADE = (
    SCHEMA.constants[f"SecurityTradingStatus.{k}"] for k in ("TradingHalt", "ReadyToTrade")
)
_SEQUENCE_ERROR = "invalid SBE sequence number"
_ENTRY_DTYPES: dict[int, np.dtype] = {ENTRY_BLOCK_LENGTH: ENTRY_DTYPE}


def entry_dtype(block_length: int) -> np.dtype:
    """The entry view for a wire ``blockLength`` (the schema's, or a later version's)."""
    dt = _ENTRY_DTYPES.get(block_length)
    if dt is None:
        dt = _ENTRY_DTYPES[block_length] = SCHEMA.entries.dtype(ORDER, block_length)
    return dt


def c_defines() -> str:
    """The schema as ``#define`` lines for the CUDA module: identifiers, block lengths, entry
    field offsets (``SBE_E_<FIELD>``), null values and enum values."""
    lines = [
        f"#define SBE_SCHEMA_ID {SCHEMA_ID}u",
        f"#define SBE_TEMPLATE_ID {TEMPLATE_ID}u",
        f"#define SBE_VERSION {VERSION}u",
        f"#define SBE_HEADER {HEADER.size}",
        f"#define SBE_GROUP_SIZE {GROUP_SIZE.size}",
        f"#define SBE_ROOT_BLOCK {ROOT_BLOCK_LENGTH}",
        f"#define SBE_ENTRY_BLOCK {ENTRY_BLOCK_LENGTH}",
        f"#define SBE_R_SEQNUM {_SEQ.offset}",
        f"#define SBE_PRICE_SCALE {PRICE_SCALE}.0",
        f"#define SBE_PRICE_NULL ({PRICE_NULL + 1}LL - 1LL)",
        f"#define SBE_QTY_NULL {QTY_NULL}u",
        f"#define SBE_STATUS_NULL {STATUS_NULL}u",
        f"#define SBE_MAX_SEQUENCE {MAX_SEQUENCE}ULL",
        f"#define SBE_DAY_NS {DAY_NS}ULL",
    ]
    lines += [f"#define SBE_E_{f.name.upper()} {f.offset}" for f in SCHEMA.entries.fields]
    lines += [
        f"#define SBE_{name.replace('.', '_').upper()} {value}u"
        for name, value in SCHEMA.constants.items()
    ]
    return "\n".join(lines) + "\n"


def price_to_wire(value: float) -> int:
    """The ``PriceNULL4`` mantissa of a price: null for NaN, else the nearest integer of four
    decimals (ties to even); infinite or out-of-range prices are rejected."""
    if math.isnan(value):
        return PRICE_NULL
    if not math.isfinite(value):
        raise ValueError("SBE price must be finite or NaN")
    mantissa = round(value * PRICE_SCALE)
    if not PRICE_NULL < mantissa <= MAX_SEQUENCE:
        raise ValueError("price exceeds the SBE PriceNULL4 range")
    return mantissa


def price_from_wire(mantissa: int) -> float:
    """``float(mantissa) / 10000.0`` (two roundings, as every decoder), NaN for null."""
    return math.nan if mantissa == PRICE_NULL else float(mantissa) / PRICE_SCALE


def qty_to_wire(value: float) -> int:
    """The ``QtyNULL`` of a quantity: null for NaN, else a whole non-negative count."""
    if math.isnan(value):
        return QTY_NULL
    if not (math.isfinite(value) and value == int(value) and 0 <= value < QTY_NULL):
        raise ValueError("source quantity cannot be represented by SBE QtyNULL")
    return int(value)


def qty_from_wire(value: int) -> float:
    return math.nan if value == QTY_NULL else float(value)


def pack_packet(entries: np.ndarray | bytes, sequence: int) -> bytes:
    """One datagram of ``entries`` (an ``ENTRY_DTYPE`` array or its bytes) with ``sequence``
    the first entry's. A message with entries needs ``sequence`` >= 1; an empty one (a
    heartbeat) carries the next sequence."""
    if isinstance(entries, (bytes, bytearray, memoryview)):
        data = bytes(entries)
    else:
        data = np.ascontiguousarray(entries).view(np.uint8).tobytes() if len(entries) else b""
    if len(data) % ENTRY_BLOCK_LENGTH:
        raise ValueError("SBE entries must be whole blocks")
    count = len(data) // ENTRY_BLOCK_LENGTH
    if count > 0xFFFF:
        raise ValueError("too many SBE entries")
    if not (1 if count else 0) <= sequence <= MAX_SEQUENCE - count:
        raise ValueError(_SEQUENCE_ERROR)
    if PACKET_HEADER + len(data) > MAX_PAYLOAD:
        raise ValueError("SBE replay packet exceeds the unfragmented IPv4 UDP budget")
    return (
        HEADER.pack(ROOT_BLOCK_LENGTH, TEMPLATE_ID, SCHEMA_ID, VERSION)
        + int(sequence).to_bytes(8, "little" if ORDER == "<" else "big")
        + bytes(ROOT_BLOCK_LENGTH - 8)
        + GROUP_SIZE.pack(ENTRY_BLOCK_LENGTH, count)
        + data
    )


def packet_count(data: bytes) -> int:
    """The entry count of a packet that passes the framing check, else 0."""
    try:
        return inspect_packet(data)[1]
    except ValueError:
        return 0


def _frame(data: bytes) -> tuple[str | None, int, int, int, int]:
    """``(error, sequence, count, entries offset, entry length)`` of the framing check: our
    template and schema, a root block that holds SeqNum and keeps the entries 8-octet
    aligned, dimensions present, entries at least the schema's and 8-octet multiples, an
    empty message exactly its prefix, a sequenced one with room for its entries and a
    sequence of at least 1 that fits the receivers' int64 arithmetic. Sequence and count
    are reported (as the gather kernel reports them) once the dimensions are readable."""
    if len(data) < HEADER.size:
        return "short SBE message", 0, 0, 0, 0
    block_length, template_id, schema_id, _version = HEADER.unpack_from(data)
    if template_id != TEMPLATE_ID or schema_id != SCHEMA_ID:
        return "not a replay MarketDataIncrementalRefresh message", 0, 0, 0, 0
    if block_length < ROOT_BLOCK_LENGTH or block_length % 8 != ROOT_BLOCK_LENGTH % 8:
        return "unsupported SBE root block length", 0, 0, 0, 0
    at = HEADER.size + block_length
    if len(data) < at + GROUP_SIZE.size:
        return "short SBE message", 0, 0, 0, 0
    entry_length, count = GROUP_SIZE.unpack_from(data, at)
    at += GROUP_SIZE.size
    if entry_length < ENTRY_BLOCK_LENGTH or entry_length % 8:
        return "unsupported SBE entry block length", 0, 0, 0, 0
    sequence = int.from_bytes(
        data[HEADER.size : HEADER.size + 8], "little" if ORDER == "<" else "big"
    )
    error = None
    if count == 0:
        if len(data) != at:
            error = "empty SBE message has entries"
    elif len(data) < at + count * entry_length:
        error = "invalid SBE entry count"
    elif sequence == 0 or sequence > MAX_SEQUENCE - count:
        error = _SEQUENCE_ERROR
    return error, sequence, count, at, entry_length


def frame_fields(data: bytes) -> tuple[bool, int, int]:
    """``(framing ok, sequence, entry count)`` of a UDP payload as the gather kernel reports
    them in the receive header (never raises; the sequence is the raw 64-bit field)."""
    error, sequence, count, _, _ = _frame(data)
    return error is None, sequence, count


def inspect_packet(data: bytes) -> tuple[int, int]:
    """``(sequence, entry count)`` after the receiver's framing check (ValueError if it
    fails); the GPU gather kernel reports the same two numbers per frame."""
    error, sequence, count, _, _ = _frame(data)
    if error is not None:
        raise ValueError(error)
    return sequence, count


def parse_packet(data: bytes) -> tuple[int, np.ndarray]:
    """``(sequence, entries)``: the framing check, then the exact length (no trailing
    octets) and the entries as an ``ENTRY_DTYPE`` view (the wire's entry length)."""
    error, sequence, count, at, entry_length = _frame(data)
    if error is not None:
        raise ValueError(error)
    if len(data) != at + count * entry_length:
        raise ValueError("trailing SBE octets")
    entries = np.frombuffer(data, entry_dtype(entry_length), count, at)
    return sequence, entries
