# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Separate streaming speedups require the predefined inputs and equivalent work
(``tools.compare_runs``)."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from holoscan_finance.marketdata.covariance import CovarianceEngine
from holoscan_finance.tools.compare_runs import compare_runs, shared_interval


def runs():
    result = CovarianceEngine().update(
        np.random.default_rng(12).normal(size=(10, 5)), np.arange(10)
    )
    base = dict(
        settings={"bar_seconds": 5},
        source_date="2026-06-01",
        requested_rate=75000,
        ingest="daqiri-host",
        minimum_update_interval_seconds=0.05,
        valid=True,
        created_utc="2026-09-12",
        seconds=120,
        observations=9_000_000,
        received_observations_per_second=75000,
        analytics_workers={},
        network_counters_before={},
        network_counters_after={},
        transport_scope="DAQIRI",
        single_calculations=[],
    )
    for stamp in (10_000_000_000, 15_000_000_000):
        base["single_calculations"].append(
            dict(
                bar_ns=stamp,
                input_sha256=str(stamp),
                analytics_ms=10,
                receive_to_result_ms=20,
                method="incremental",
                added_bars=1,
                expired_bars=1,
                eligible=5,
                validation=result,
            )
        )
    cpu, gpu = deepcopy(base), deepcopy(base)
    cpu["analytics_backend"], gpu["analytics_backend"] = "cpu", "gpu"
    for row in gpu["single_calculations"]:
        row["analytics_ms"] = 2
    return cpu, gpu


def test_separate_run_ratio_uses_identical_predefined_windows():
    report = compare_runs(*runs(), 10_000_000_000, 20_000_000_000)
    assert report["valid"] and report["median_analytics_speedup"] == 5
    assert len(report["matched_calculations"]) == 2


def test_shared_interval_uses_all_overlap_and_rejects_internal_holes():
    cpu, gpu = runs()
    for run in (cpu, gpu):
        extra = deepcopy(run["single_calculations"][-1])
        extra["bar_ns"] = 20_000_000_000
        run["single_calculations"].append(extra)
    assert shared_interval(cpu, gpu) == (10_000_000_000, 25_000_000_000)
    gpu["single_calculations"].pop(1)
    report = compare_runs(cpu, gpu, *shared_interval(cpu, gpu))
    assert (
        not report["valid"] and "gpu: missing or duplicate predefined windows" in report["errors"]
    )


@pytest.mark.parametrize(
    "fault", ["input", "missing", "duplicate", "output", "work", "capture", "protocol"]
)
def test_separate_run_comparison_rejects_unmatched_or_invalid_work(fault):
    cpu, gpu = runs()
    first = gpu["single_calculations"][0]
    if fault == "input":
        first["input_sha256"] = "different"
    elif fault == "missing":
        gpu["single_calculations"].pop()
    elif fault == "duplicate":
        gpu["single_calculations"].append(deepcopy(first))
    elif fault == "output":
        first["validation"]["annual_vol"][0] *= 2
    elif fault == "work":
        first["added_bars"] = 2
    elif fault == "capture":
        gpu["valid"] = False
    elif fault == "protocol":
        gpu["wire_protocol"] = "mit303-11.9-equity-replay-v1"  # the previous wire format
    report = compare_runs(cpu, gpu, 10_000_000_000, 20_000_000_000)
    assert not report["valid"] and report["median_analytics_speedup"] is None
