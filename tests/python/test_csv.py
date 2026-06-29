# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CSV ingestion tests for deterministic offline market-data replay."""

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from holoscan.finance import load_market_ticks_csv


def test_load_market_ticks_csv_preserves_decimal_precision_and_row_order(tmp_path):
    csv_path = tmp_path / "ticks.csv"
    csv_path.write_text(
        "timestamp,sequence,symbol,price,currency\n"
        "2026-06-27T14:30:00Z,100,NVDA,157.25000001,USD\n"
        "2026-06-27T14:30:00.000001+00:00,101,MSFT,497.50,USD\n",
        encoding="utf-8",
    )

    ticks = load_market_ticks_csv(csv_path)

    assert [tick.sequence for tick in ticks] == [100, 101]
    assert [tick.symbol for tick in ticks] == ["NVDA", "MSFT"]
    assert ticks[0].timestamp == datetime(2026, 6, 27, 14, 30, tzinfo=timezone.utc)
    assert ticks[0].price == Decimal("157.25000001")
    assert ticks[1].price == Decimal("497.50")


def test_load_market_ticks_csv_accepts_a_string_path(tmp_path):
    csv_path = tmp_path / "one.csv"
    csv_path.write_text(
        "timestamp,sequence,symbol,price,currency\n2026-06-27T14:30:00+00:00,1,NVDA,100,USD\n",
        encoding="utf-8",
    )

    ticks = load_market_ticks_csv(str(csv_path))

    assert len(ticks) == 1
    assert ticks[0].sequence == 1


@pytest.mark.parametrize(
    "header",
    [
        "sequence,symbol,price,currency",
        "timestamp,symbol,price,currency",
        "timestamp,sequence,price,currency",
        "timestamp,sequence,symbol,currency",
        "timestamp,sequence,symbol,price",
    ],
)
def test_load_market_ticks_csv_requires_the_documented_columns(tmp_path, header):
    csv_path = tmp_path / "missing-column.csv"
    csv_path.write_text(f"{header}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="column|required|header"):
        load_market_ticks_csv(csv_path)


def test_load_market_ticks_csv_rejects_duplicate_column_names(tmp_path):
    csv_path = tmp_path / "duplicate-column.csv"
    csv_path.write_text(
        "timestamp,sequence,symbol,price,currency,price\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="column names must be unique"):
        load_market_ticks_csv(csv_path)


@pytest.mark.parametrize(
    "row",
    [
        "not-a-date,1,NVDA,100,USD",
        "2026-06-27T14:30:00+00:00,one,NVDA,100,USD",
        "2026-06-27T14:30:00+00:00,1,NVDA,not-a-price,USD",
        "2026-06-27T14:30:00+00:00,1,NVDA,-10,USD",
        "2026-06-27T14:30:00+00:00,1,NVDA,100,US",
    ],
)
def test_load_market_ticks_csv_reports_invalid_rows(tmp_path, row):
    csv_path = tmp_path / "bad-row.csv"
    csv_path.write_text(
        f"timestamp,sequence,symbol,price,currency\n{row}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="row|line|invalid|price|sequence|timestamp|currency"):
        load_market_ticks_csv(csv_path)


def test_load_market_ticks_csv_rejects_header_only_file(tmp_path):
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("timestamp,sequence,symbol,price,currency\n", encoding="utf-8")

    with pytest.raises(ValueError, match="at least one|empty"):
        load_market_ticks_csv(csv_path)
