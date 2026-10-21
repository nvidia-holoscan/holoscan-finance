# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""When a CPU and a GPU risk result agree, and when their timings may be compared.

``agreement`` is verbatim from 4.3 ``equities.analytics`` (the tolerances behind every
published CPU/GPU comparison): the same eligible and displayed instruments, and volatilities,
covariance probes and heatmap correlations within tolerance. ``compare_results`` adds the
4.3 ``publish_comparison`` rules: the work matched (same method, window and incremental
additions/expirations), and a speedup is usable only if both agree on at least two names.
"""

from __future__ import annotations

import numpy as np

# Identical windows imply identical incremental work: these must match exactly.
MATCHED_KEYS = ("method", "added_bars", "expired_bars", "bars", "universe")


def agreement(cpu, gpu):
    same = cpu["eligible_indices"] == gpu["eligible_indices"]
    vol_ok = same and np.allclose(cpu["annual_vol"], gpu["annual_vol"], rtol=1e-8, atol=1e-10)
    cov_ok = same and np.allclose(
        cpu["covariance_probes"], gpu["covariance_probes"], rtol=1e-8, atol=1e-14
    )
    corr_ok = (
        same
        and cpu["heatmap_indices"] == gpu["heatmap_indices"]
        and np.allclose(cpu["heatmap"], gpu["heatmap"], rtol=1e-8, atol=1e-8)
    )
    return bool(vol_ok and cov_ok and corr_ok)


def compare_results(cpu: dict, gpu: dict) -> tuple[bool, bool, bool]:
    """``(agreement, matched_work, usable)`` of a CPU result against its GPU twin."""
    ok = agreement(cpu, gpu)
    matched = all(cpu[k] == gpu[k] for k in MATCHED_KEYS)
    return ok, matched, ok and matched and gpu["eligible"] >= 2
