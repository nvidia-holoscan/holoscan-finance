# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read synthetic observation tapes produced by ``tools.synth``.

``open_tape`` validates the manifest and maps ``events.bin`` without loading it into RAM.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..marketdata.records import EVENT_DTYPE


def open_tape(path: str | Path) -> tuple[dict, np.memmap]:
    """The manifest and a read-only map of the records; ValueError for another format or an
    incomplete tape."""
    p = Path(path)
    m = json.loads((p / "manifest.json").read_text())
    if m.get("format") != "equity-replay-v1" or m.get("record_bytes") != EVENT_DTYPE.itemsize:
        raise ValueError("unsupported prepared tape format")
    if (p / "events.bin").stat().st_size != m["events"] * EVENT_DTYPE.itemsize:
        raise ValueError("incomplete prepared tape")
    return m, np.memmap(p / "events.bin", dtype=EVENT_DTYPE, mode="r", shape=(m["events"],))
