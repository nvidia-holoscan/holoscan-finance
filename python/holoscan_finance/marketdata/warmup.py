# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compile (or load from ``CUPY_CACHE_DIR``) the GPU kernels before live data.

Without this the first market-data batches stall behind CUDA JIT compilation. Split from
4.3 ``equities.warmup.warmup`` so each operator warms only its own kernels in ``start()``:
the sources the gather and decode RawModule, the bars operator both bar paths and
``apply_sequence``. Both functions run once per process and device and log
"GPU <x> kernels ready in N s".
"""

from __future__ import annotations

from collections.abc import Mapping


def warm_decode_kernels(gpu_device: int = 0) -> float:
    """Compile and run ``gather_frame`` and ``mitch_decode`` on synthetic units; returns seconds."""
    raise NotImplementedError


def warm_bars_kernels(settings: Mapping[str, float], gpu_device: int = 0) -> float:
    """The 4.3 ``warmup()`` body: bars across several bar boundaries; returns seconds."""
    raise NotImplementedError
