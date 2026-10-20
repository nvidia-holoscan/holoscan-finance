# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepared NYQ observation tapes: bounded-memory preparation and memory-mapped reading.

``prepare`` turns one UTC day of licensed NYQ Parquet into ``events.bin`` (``EVENT_DTYPE``
records of the 09:30-16:00 New York session, time ordered) plus ``manifest.json``, with a
DuckDB external sort and Arrow batches so memory stays bounded. ``open_tape`` maps a tape
without reading it. Both verbatim from 4.3 ``equities.data`` (its CLI is now
``tools.prepare_tape``); the two progress lines are log records. duckdb and pyarrow are
imported inside ``prepare``.
"""

from __future__ import annotations

import glob
import json
import logging
import tempfile
from datetime import UTC, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from ..marketdata.records import EVENT_DTYPE

logger = logging.getLogger(__name__)


def prepare(
    pattern: str | Path, output: str | Path, *, memory: str = "2GB", threads: int = 2
) -> dict:
    import duckdb

    files = sorted(glob.glob(str(pattern)))
    if not files:
        raise ValueError(f"no licensed NYQ Parquet files match {pattern}")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists() or (out / "events.bin").exists():
        raise ValueError("output already contains a tape; use a new output directory")
    with tempfile.TemporaryDirectory(prefix="nyq-sort-", dir=out) as temp:
        db = duckdb.connect(
            config={
                "memory_limit": memory,
                "threads": threads,
                "temp_directory": temp,
                "preserve_insertion_order": False,
            }
        )
        db.from_parquet(files).create_view("raw")
        names = [x[0] for x in db.execute("DESCRIBE raw").fetchall()]
        if "ricName" not in names:
            raise ValueError("expected NYQ equity schema with ricName")
        lo, hi, raw_rows = db.execute(
            "SELECT min(messageTimestamp), max(messageTimestamp), count(*) FROM raw"
        ).fetchone()
        ny = ZoneInfo("America/New_York")
        # LSEG partitions are UTC days, including the preceding NY evening.
        day = datetime.fromtimestamp(lo / 1e9, UTC).date()
        if datetime.fromtimestamp(hi / 1e9, UTC).date() != day:
            raise ValueError("prepare exactly one UTC source date at a time")
        start = int(datetime.combine(day, time(9, 30), ny).timestamp()) * 10**9
        end = int(datetime.combine(day, time(16), ny).timestamp()) * 10**9
        # Directory membership is metadata, not a liquidity ranking or future prices.
        symbols = [
            r[0]
            for r in db.execute(
                "SELECT DISTINCT ricName FROM raw WHERE ricName IS NOT NULL ORDER BY ricName"
            ).fetchall()
        ]
        import pyarrow as pa

        db.register("directory", pa.table({"symbol": symbols, "symbol_id": range(len(symbols))}))
        query = f"""
          SELECT messageTimestamp::BIGINT AS timestamp_ns, symbol_id::UINTEGER AS symbol_id,
            coalesce(internalSequence, 0)::INTEGER AS internal_sequence,
            CASE type WHEN 'Quote' THEN 1 WHEN 'Trade' THEN 2 ELSE 3 END::UTINYINT AS kind,
            CASE WHEN regexp_matches(coalesce(Qualifiers, ''),
                 '[1-9][0-9]*\\[HALT_REASN\\]') THEN 1 ELSE 0 END::UTINYINT AS flags,
            Bid_Price::DOUBLE AS bid, Ask_Price::DOUBLE AS ask,
            Bid_Size::DOUBLE AS bid_size, Ask_Size::DOUBLE AS ask_size,
            Price::DOUBLE AS price, Volume::DOUBLE AS volume
          FROM raw JOIN directory ON ricName = symbol
          WHERE messageTimestamp >= {start} AND messageTimestamp < {end}
            AND type IN ('Quote', 'Trade', 'Mkt. Condition')
          ORDER BY timestamp_ns, internal_sequence, symbol_id, kind
        """
        logger.info(
            f"Preparing {len(files)} files / {raw_rows:,} rows / {len(symbols):,} RICs; "
            f"{day} 09:30–16:00 America/New_York"
        )
        reader = db.execute(query).to_arrow_reader(131072)
        count = 0
        profile = np.zeros(390, dtype=np.int64)
        kinds = np.zeros(4, dtype=np.int64)
        partial = out / "events.partial"
        with partial.open("wb") as target:
            for batch in reader:
                rows = np.empty(batch.num_rows, dtype=EVENT_DTYPE)
                for name in EVENT_DTYPE.names:
                    rows[name] = batch.column(name).to_numpy(zero_copy_only=False)
                rows.tofile(target)
                count += len(rows)
                profile += np.bincount(
                    ((rows["timestamp_ns"] - start) // (60 * 10**9)).astype(int), minlength=390
                )
                kinds += np.bincount(rows["kind"], minlength=4)
        if not count:
            partial.unlink()
            raise ValueError("no regular-session observations")
        partial.rename(out / "events.bin")
        manifest = {
            "format": "nyq-equity-replay-v1",
            "date": day.isoformat(),
            "symbols": symbols,
            "events": count,
            "raw_rows": raw_rows,
            "kinds": {
                "quotes": int(kinds[1]),
                "trades": int(kinds[2]),
                "conditions": int(kinds[3]),
            },
            "start_ns": start,
            "end_ns": end,
            "record_bytes": EVENT_DTYPE.itemsize,
            "profile": [
                [int(start // 10**9 % 86400 + i * 60), int(c)] for i, c in enumerate(profile)
            ],
            "source_files": [
                {"name": Path(f).name, "bytes": Path(f).stat().st_size} for f in files
            ],
            "prepared_utc": datetime.now(UTC).isoformat(),
        }
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        logger.info(f"Prepared {count:,} observations at {out}")
        return manifest


def open_tape(path: str | Path) -> tuple[dict, np.memmap]:
    p = Path(path)
    m = json.loads((p / "manifest.json").read_text())
    if m.get("format") != "nyq-equity-replay-v1" or m.get("record_bytes") != EVENT_DTYPE.itemsize:
        raise ValueError("unsupported prepared tape format")
    if (p / "events.bin").stat().st_size != m["events"] * EVENT_DTYPE.itemsize:
        raise ValueError("incomplete prepared tape")
    return m, np.memmap(p / "events.bin", dtype=EVENT_DTYPE, mode="r", shape=(m["events"],))
