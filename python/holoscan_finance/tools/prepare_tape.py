# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare one day of licensed NYQ Parquet as a replay tape (bounded memory).

    python -m holoscan_finance.tools.prepare_tape --data 'DAY/*.parquet' --output TAPE
        [--memory 2GB] [--threads 2]

The CLI around ``nyq.tape.prepare``: a DuckDB external sort into ``events.bin`` plus
``manifest.json``, the format ``open_tape``, the replay sender and the replay source read.
Needs duckdb and pyarrow; CPU only.
"""

from __future__ import annotations

import argparse
import logging

from ..nyq.tape import prepare


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m holoscan_finance.tools.prepare_tape", description=__doc__.splitlines()[0]
    )
    p.add_argument("--data", required=True, help="one day's NYQ Parquet file or glob")
    p.add_argument("--output", required=True)
    p.add_argument("--memory", default="2GB")
    p.add_argument("--threads", type=int, default=2)
    return p


def main(argv: list[str] | None = None) -> int:
    a = parser().parse_args(argv)
    # prepare() reports its progress as log records.
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    prepare(a.data, a.output, memory=a.memory, threads=a.threads)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
