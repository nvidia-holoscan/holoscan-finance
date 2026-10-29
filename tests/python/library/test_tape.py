# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic tape reading: format, size and read-only memory-map checks."""

from __future__ import annotations

import json

import numpy as np
import pytest

from holoscan_finance.marketdata.records import EVENT_DTYPE
from holoscan_finance.nyq.tape import open_tape


def write_tape(path, rows, **changes):
    path.mkdir()
    rows.tofile(path / "events.bin")
    manifest = dict(
        format="nyq-equity-replay-v1", events=len(rows), record_bytes=EVENT_DTYPE.itemsize
    )
    manifest.update(changes)
    (path / "manifest.json").write_text(json.dumps(manifest))


def test_open_tape_maps_the_records_read_only(tmp_path):
    rows = np.zeros(3, EVENT_DTYPE)
    rows["timestamp_ns"] = [1, 2, 3]
    write_tape(tmp_path / "t", rows, date="2025-04-01")
    manifest, tape = open_tape(tmp_path / "t")
    assert manifest["date"] == "2025-04-01"
    assert isinstance(tape, np.memmap) and tape.dtype == EVENT_DTYPE
    assert tape["timestamp_ns"].tolist() == [1, 2, 3]
    with pytest.raises(ValueError):
        tape["timestamp_ns"][0] = 9


@pytest.mark.parametrize(
    "changes,match",
    [
        (dict(format="other"), "format"),
        (dict(record_bytes=EVENT_DTYPE.itemsize + 1), "format"),
        (dict(events=4), "incomplete"),
    ],
)
def test_open_tape_rejects_other_formats_and_incomplete_tapes(tmp_path, changes, match):
    write_tape(tmp_path / "t", np.zeros(3, EVENT_DTYPE), **changes)
    with pytest.raises(ValueError, match=match):
        open_tape(tmp_path / "t")
