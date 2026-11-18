# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``validate_manifest``: every manifest rejection, and the immutable ``FeedDirectory`` it
publishes."""

from __future__ import annotations

import dataclasses
import types

import pytest

from holoscan_finance.equity.manifest import MANIFEST_FORMAT, FeedDirectory, validate_manifest
from holoscan_finance.equity.mapping import PROTOCOL, TIMEZONE, day_start_ns

MIDNIGHT = day_start_ns("2026-06-01")


def manifest(**changes):
    payload = dict(
        format=MANIFEST_FORMAT,
        wire_protocol=PROTOCOL,
        market_timezone=TIMEZONE,
        symbols=["LONG_SYMBOL", "SECOND.SYN"],
        generation=3,
        sender_port=50001,
        date="2026-06-01",
        start_ns=MIDNIGHT + 34200 * 10**9,
        end_ns=MIDNIGHT + 57600 * 10**9,
        events=100,
        profile=[[48600, 7]],
        kinds={"quotes": 90},
    )
    payload.update(changes)
    return payload


def test_a_valid_manifest_becomes_a_feed_directory():
    d = validate_manifest(manifest())
    assert isinstance(d, FeedDirectory)
    assert (d.generation, d.sender_port, d.symbols, d.universe) == (
        3,
        50001,
        ("LONG_SYMBOL", "SECOND.SYN"),
        2,
    )
    assert (d.date, d.midnight_ns, d.events) == ("2026-06-01", MIDNIGHT, 100)
    assert (d.start_ns, d.end_ns, d.profile) == (
        manifest()["start_ns"],
        manifest()["end_ns"],
        ([48600, 7],),
    )
    assert isinstance(d.manifest, types.MappingProxyType) and d.manifest["kinds"] == {"quotes": 90}
    with pytest.raises(dataclasses.FrozenInstanceError):
        d.generation = 4
    with pytest.raises(TypeError):
        d.manifest["generation"] = 4


def test_the_directory_owns_a_deep_copy():
    payload = manifest()
    d = validate_manifest(payload)
    payload["symbols"].append("C")
    payload["kinds"]["quotes"] = 0
    payload["generation"] = 9
    assert d.symbols == ("LONG_SYMBOL", "SECOND.SYN") and d.manifest["symbols"] == [
        "LONG_SYMBOL",
        "SECOND.SYN",
    ]
    assert d.manifest["kinds"] == {"quotes": 90} and d.manifest["generation"] == 3


@pytest.mark.parametrize(
    "changes",
    [
        dict(format="equity-replay-v0"),
        dict(wire_protocol="unsupported-equity-replay-v1"),
        dict(market_timezone="UTC"),
        dict(sender_port="50001"),
        dict(sender_port=True),
        dict(sender_port=0),
        dict(sender_port=65536),
        dict(symbols="LONG_SYMBOL"),
        dict(symbols=[]),
        dict(symbols=[f"S{i}" for i in range(10001)]),
        dict(symbols=["A", 7]),
        dict(symbols=["A", ""]),
        dict(symbols=["A", "x" * 129]),
        dict(symbols=["A", "A"]),
        dict(generation=0),
        dict(generation=1.0),
        dict(generation=False),
        dict(date=20260601),
        dict(date="2025-13-01"),
        dict(start_ns=1.5),
        dict(end_ns=None),
        dict(end_ns=MIDNIGHT),
        dict(events=0),
        dict(events="100"),
        dict(profile={}),
    ],
)
def test_every_rejection(changes):
    with pytest.raises(ValueError):
        validate_manifest(manifest(**changes))


@pytest.mark.parametrize("missing", ["format", "symbols", "generation", "date", "profile"])
def test_missing_keys_are_rejected(missing):
    payload = manifest()
    del payload[missing]
    with pytest.raises(ValueError):
        validate_manifest(payload)


def test_non_dict_payloads_are_rejected():
    for payload in (None, [], "manifest"):
        with pytest.raises(ValueError):
            validate_manifest(payload)
