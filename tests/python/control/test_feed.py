# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FeedControl: the 4.3 sender pacing rules (/feedrate, /feedstatus, /replay/resume)."""

from __future__ import annotations

import pytest

from holoscan_finance.control.directory import ReplayDirectory
from holoscan_finance.control.feed import FeedControl
from holoscan_finance.nyq.mapping import PROTOCOL, TIMEZONE, day_start_ns
from holoscan_finance.ref import Knob

MIDNIGHT = day_start_ns("2025-04-01")


def manifest(generation=1, port=50001):
    return dict(
        format="nyq-equity-replay-v1",
        wire_protocol=PROTOCOL,
        market_timezone=TIMEZONE,
        symbols=["A"],
        generation=generation,
        sender_port=port,
        date="2025-04-01",
        start_ns=MIDNIGHT + 34200 * 10**9 + 7,
        end_ns=MIDNIGHT + 57600 * 10**9,
        events=1,
        profile=[],
    )


def test_defaults_and_partial_updates():
    feed = FeedControl(ReplayDirectory(Knob()))
    assert feed.control() == dict(speed=10.0, rate=50000.0, paused=False, seek=48600.0, seek_seq=1)
    c = feed.control({"speed": 25, "rate": 200000})
    assert (c["speed"], c["rate"], c["seek_seq"]) == (25, 200000, 1)
    c["paused"] = True  # a copy: callers never mutate the controls
    assert feed.control()["paused"] is False


def test_a_seek_bumps_seek_seq():
    feed = FeedControl(ReplayDirectory(Knob()))
    assert feed.control({"seek": 36000})["seek_seq"] == 2
    assert feed.control({"paused": True})["seek_seq"] == 2
    assert feed.control({"seek": 0.0})["seek_seq"] == 3


@pytest.mark.parametrize(
    "payload",
    [
        {"paused": "false"},
        {"speed": float("nan")},
        {"speed": 101},
        {"rate": 0},
        {"rate": 200001},
        {"seek": 86400},
        {"seek": -1},
        {"loops": 2},
    ],
)
def test_invalid_controls_are_rejected_and_change_nothing(payload):
    feed = FeedControl(ReplayDirectory(Knob()))
    before = feed.control()
    with pytest.raises(ValueError):
        feed.control(payload)
    assert feed.control() == before


def test_a_value_of_the_wrong_type_raises_type_error():
    feed = FeedControl(ReplayDirectory(Knob()))
    with pytest.raises(TypeError):
        feed.control({"speed": None})


def test_sender_status_accepts_only_the_current_generation_and_issued_seeks():
    d = ReplayDirectory(Knob())
    feed = FeedControl(d)
    report = dict(seek_seq=1, generation=1, position=34200.5, sent=10, loops=0)
    assert feed.sender_status(report) == {"status": "received"}
    assert feed.sender is None  # no generation yet
    d.set_manifest(manifest(1))
    feed.sender_status(report)
    assert feed.sender == report
    for stale in (dict(report, generation=2), dict(report, seek_seq=2), dict(report, seek_seq="1")):
        assert feed.sender_status(stale) == {"status": "received"}
        assert feed.sender == report
    feed.control({"seek": 40000})
    feed.sender_status(dict(report, seek_seq=2, position=40000.0))
    assert feed.sender["position"] == 40000.0


def test_resume_unpauses_at_the_session_open():
    d = ReplayDirectory(Knob())
    feed = FeedControl(d)
    feed.control({"paused": True, "seek": 50000})
    c = feed.resume()  # no manifest yet: keep the current cursor
    assert (c["paused"], c["seek"], c["seek_seq"]) == (False, 50000.0, 3)
    d.set_manifest(manifest(1))
    feed.control({"paused": True})
    c = feed.resume()  # the slider's clock is UTC: the 09:30 New York open is 13:30 UTC
    assert (c["paused"], c["seek"], c["seek_seq"]) == (False, 48600.0, 4)
