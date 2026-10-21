# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepared tapes: ``open_tape`` checks, and ``prepare`` on synthetic Parquet (skips without
duckdb and pyarrow)."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import numpy as np
import pytest

from holoscan_finance.marketdata.records import CONDITION, EVENT_DTYPE, HALTED, QUOTE
from holoscan_finance.nyq.tape import open_tape, prepare


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


def test_preparation_selects_session_sorts_and_retains_quote_fields(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    pytest.importorskip("duckdb")

    def timestamp(hour, minute):
        return int(datetime(2025, 4, 1, hour, minute, tzinfo=UTC).timestamp() * 1e9)

    source = tmp_path / "part.parquet"
    pq.write_table(
        pa.table(
            dict(
                messageTimestamp=[timestamp(14, 0), timestamp(13, 0), timestamp(13, 30)],
                ricName=["VERY_LONG_RIC"] * 3,
                internalSequence=[2, 1, 1],
                type=["Quote", "Trade", "Mkt. Condition"],
                Qualifiers=["", "", "1[HALT_REASN]"],
                Bid_Price=[812345.12, None, None],
                Ask_Price=[812345.14, None, None],
                Bid_Size=[10.0, None, None],
                Ask_Size=[20.0, None, None],
                Price=[None, 4.0, None],
                Volume=[None, 100.0, None],
            )
        ),
        source,
    )
    prepare(source, tmp_path / "out")
    m, tape = open_tape(tmp_path / "out")
    assert m["raw_rows"] == 3 and m["events"] == 2
    assert m["start_ns"] == timestamp(13, 30)
    assert m["symbols"] == ["VERY_LONG_RIC"]
    assert tape["kind"].tolist() == [CONDITION, QUOTE]
    assert tape["flags"][0] == HALTED
    assert tape["bid"][1] == 812345.12 and tape["ask_size"][1] == 20
    assert np.isnan(tape["price"][1])
    assert sum(n for _, n in m["profile"]) == 2
    assert json.loads((tmp_path / "out/manifest.json").read_text())["date"] == "2025-04-01"
    with pytest.raises(ValueError, match="already contains a tape"):
        prepare(source, tmp_path / "out")
    with pytest.raises(ValueError, match="no licensed"):
        prepare(tmp_path / "missing-*.parquet", tmp_path / "other")
