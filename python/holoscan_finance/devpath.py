# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Write the two-file dist-info that makes a source checkout importable as ``holoscan.finance``.

Holoscan 5 resolves ``holoscan.<name>`` from ``holoscan.modules`` entry points, so a plain
``PYTHONPATH=<repo>/python`` gives only the canonical ``holoscan_finance``. Putting the
directory written here on ``PYTHONPATH`` as well adds the alias, as the CMake staging and
the wheel do (verified with holoscan-cli 5.0; no dist-info is committed). Apps and tests import the
canonical package and do not need this.

    python -m holoscan_finance.devpath DIR
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import __version__


def write_dist_info(target_dir: str | Path) -> Path:
    """Write ``holoscan_finance-<version>.dist-info/{METADATA,entry_points.txt}`` under
    ``target_dir``; returns ``target_dir`` (the directory to put on ``PYTHONPATH``)."""
    target = Path(target_dir)
    dist = target / f"holoscan_finance-{__version__}.dist-info"
    dist.mkdir(parents=True, exist_ok=True)
    (dist / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: holoscan-finance\nVersion: {__version__}\n"
    )
    (dist / "entry_points.txt").write_text("[holoscan.modules]\nfinance = holoscan_finance\n")
    return target


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("dir", help="directory to add to PYTHONPATH")
    a = p.parse_args(argv)
    print(write_dist_info(a.dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
