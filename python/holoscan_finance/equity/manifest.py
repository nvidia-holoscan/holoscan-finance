# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The feed directory of one replay generation: symbols, sender port, market day.

``validate_manifest`` applies every manifest check and resolves the market midnight, so an
invalid date is rejected before the active generation is replaced. The result is an immutable ``FeedDirectory`` that ``ReplayDirectory``
publishes through a ``Knob``; decode and bars read it once per activation without a lock.
Generation ordering and sender-port reuse are the directory's checks, not this module's.
"""

from __future__ import annotations

import copy
import math
import types
from dataclasses import dataclass

from .mapping import PROTOCOL, TIMEZONE, day_start_ns

MANIFEST_FORMAT = "equity-replay-v1"


@dataclass(frozen=True)
class FeedDirectory:
    generation: int
    sender_port: int
    symbols: tuple[str, ...]
    date: str
    midnight_ns: int
    start_ns: int
    end_ns: int
    events: int
    profile: tuple
    manifest: types.MappingProxyType  # the accepted manifest (for /snapshot.profile)

    @property
    def universe(self) -> int:
        return len(self.symbols)


def _symbol(value: object) -> bool:
    """1-128 characters that encode as UTF-8 (``/snapshot`` serves them; a lone surrogate
    from a JSON escape cannot be encoded)."""
    if not isinstance(value, str) or not 0 < len(value) <= 128:
        return False
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _number(value: object) -> bool:
    """A JSON number ``/snapshot`` can serve back: no bool, no inf or NaN."""
    return type(value) is int or (type(value) is float and math.isfinite(value))


def validate_manifest(payload: dict) -> FeedDirectory:
    """Every manifest check plus the market day; ValueError on any failure."""
    if not isinstance(payload, dict):
        raise ValueError("invalid replay directory or generation")
    symbols = payload.get("symbols")
    generation = payload.get("generation")
    if (
        payload.get("format") != MANIFEST_FORMAT
        or payload.get("wire_protocol") != PROTOCOL
        or payload.get("market_timezone") != TIMEZONE
        or type(payload.get("sender_port")) is not int
        or not 1 <= payload["sender_port"] <= 65535
        or not isinstance(symbols, list)
        or not 1 <= len(symbols) <= 10000
        or not all(_symbol(s) for s in symbols)
        or len(set(symbols)) != len(symbols)
        or type(generation) is not int
        or generation <= 0
        or not isinstance(payload.get("date"), str)
        or type(payload.get("start_ns")) is not int
        or type(payload.get("end_ns")) is not int
        or payload["start_ns"] >= payload["end_ns"]
        or type(payload.get("events")) is not int
        or payload["events"] < 1
        or not isinstance(payload.get("profile"), list)
        or not all(  # [minute start (s), events] pairs
            isinstance(p, list) and len(p) == 2 and all(_number(x) for x in p)
            for p in payload["profile"]
        )
    ):
        raise ValueError("invalid replay directory or generation")
    midnight_ns = day_start_ns(payload["date"])  # ValueError for an invalid date
    # A private deep copy: the caller (an HTTP body, a replay loop's base manifest) may reuse
    # or mutate its dict after publication.
    manifest = copy.deepcopy(payload)
    return FeedDirectory(
        generation=generation,
        sender_port=manifest["sender_port"],
        symbols=tuple(manifest["symbols"]),
        date=manifest["date"],
        midnight_ns=midnight_ns,
        start_ns=manifest["start_ns"],
        end_ns=manifest["end_ns"],
        events=manifest["events"],
        profile=tuple(manifest["profile"]),
        manifest=types.MappingProxyType(manifest),
    )
