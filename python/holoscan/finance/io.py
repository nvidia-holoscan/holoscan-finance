# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Input helpers for deterministic finance reference data."""

from __future__ import annotations

import csv
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

from .models import MarketTick, validate_market_ticks

_REQUIRED_COLUMNS = frozenset({"timestamp", "sequence", "symbol", "price", "currency"})


def _parse_timestamp(value: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError("timestamp must be text")
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = f"{normalized[:-1]}+00:00"
    try:
        return datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"invalid ISO 8601 timestamp {value!r}") from exc


def load_market_ticks_csv(path: str | Path) -> tuple[MarketTick, ...]:
    """Load a validated market-tick sequence from a UTF-8 CSV file.

    The file must contain ``timestamp``, ``sequence``, ``symbol``, ``price``,
    and ``currency`` columns. Extra columns are ignored.
    """

    csv_path = Path(path)
    try:
        stream = csv_path.open("r", encoding="utf-8", newline="")
    except OSError as exc:
        raise OSError(f"unable to open market data CSV {csv_path}: {exc}") from exc

    ticks: list[MarketTick] = []
    with stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames or []
        columns = set(fieldnames)
        if len(columns) != len(fieldnames):
            raise ValueError(f"{csv_path}: column names must be unique")
        missing = sorted(_REQUIRED_COLUMNS - columns)
        if missing:
            raise ValueError(f"{csv_path}: missing required columns: {', '.join(missing)}")

        for line_number, row in enumerate(reader, start=2):
            try:
                tick = MarketTick(
                    timestamp=_parse_timestamp(row["timestamp"]),
                    sequence=int(row["sequence"]),
                    symbol=row["symbol"],
                    price=Decimal(row["price"]),
                    currency=row["currency"],
                )
            except (InvalidOperation, TypeError, ValueError) as exc:
                raise ValueError(f"{csv_path}:{line_number}: {exc}") from exc
            ticks.append(tick)

    try:
        return validate_market_ticks(ticks)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{csv_path}: {exc}") from exc


__all__ = ["load_market_ticks_csv"]
