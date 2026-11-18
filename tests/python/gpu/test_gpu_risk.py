# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU risk: heatmap selection independent of the BLAS backend, and ``RollingRisk(gpu)`` on
device windows matching the engine, validated or not."""

from __future__ import annotations

import json

import numpy as np
import pytest

from holoscan_finance.analytics.agreement import agreement
from holoscan_finance.analytics.rolling import RollingRisk
from holoscan_finance.control.gate import CrossCheckGate
from holoscan_finance.marketdata.covariance import CovarianceEngine
from holoscan_finance.payloads import BarWindow

pytestmark = pytest.mark.gpu
cp = pytest.importorskip("cupy")
pytest.importorskip("scipy")  # the NumPy engine's dgemm


def test_equal_volatility_heatmap_selection_is_backend_independent():
    rng = np.random.default_rng(912)
    rows = rng.normal(0, 0.01, (80, 40))
    rows[20:] = np.tile(rng.normal(0, 0.001, (60, 1)), (1, 40))
    cpu, gpu = CovarianceEngine(), CovarianceEngine("gpu")
    for end in range(60, 81):
        a = cpu.update(rows[end - 60 : end], np.arange(end - 60, end))
        b = gpu.update(rows[end - 60 : end], np.arange(end - 60, end))
        assert agreement(a, b)
    assert a["heatmap_indices"] == list(range(24))


def _device_windows(rows: np.ndarray) -> list[BarWindow]:
    return [
        BarWindow(
            tuple(cp.asarray(r) for r in rows[i : i + 60]),
            np.arange(i, i + 60),
            1,
            60 + i,
            0,
            device=True,
        )
        for i in (0, 1)
    ]


@pytest.mark.parametrize("validate", [True, False])
def test_rolling_risk_on_device_windows_matches_the_engine(validate):
    rng = np.random.default_rng(3)
    rows = rng.normal(0, 0.001, (61, 40))
    rows[5:9, 2] = np.nan
    rows[:, 7] = 0
    gate = CrossCheckGate()
    risk = RollingRisk(backend="gpu", min_interval_ns=0, cross_check=validate, gate=gate)
    risk.start()
    engine = CovarianceEngine("gpu")
    for w in _device_windows(rows):
        risk.offer(w)
        update = risk.poll()
        gate.release()  # stands in for the published comparison
        host = rows[w.times[0] : w.times[-1] + 1]
        observed = np.var(host, axis=0, ddof=1) if validate else None
        want = engine.update(host, w.times, observed_variance=observed)
        assert update.window is w and update.validated is validate and update.backend == "gpu"
        if validate:
            np.testing.assert_array_equal(update.rows_host, host)
        else:
            assert update.rows_host is None
        for key in ("method", "added_bars", "expired_bars", "eligible_indices", "heatmap_indices"):
            assert update.result[key] == want[key], key
        np.testing.assert_allclose(update.result["annual_vol"], want["annual_vol"], rtol=1e-10)
        assert update.result["gpu_name"] and update.result["finished_wall_ns"] > 0
    assert update.result["method"] == "incremental"  # second window: one bar added, one expired
    risk.stop()


def test_compare_mode_matches_work_on_windows_longer_than_64_bars():
    """A 120-bar window (--window-seconds 600): every validated GPU window and its CPU
    comparison (``CpuCrossCheck``, new bars only) do the same work and agree, from the
    growth after a start (20..120 bars) through slides past the 256-call rebuild."""
    pytest.importorskip("threadpoolctl")
    from holoscan_finance.analytics.crosscheck import CpuCrossCheck
    from holoscan_finance.analytics.rolling import RollingRisk

    rows = np.random.default_rng(600).normal(0, 0.001, (420, 30))
    rows[150:160, 3] = np.nan
    parts = tuple(cp.asarray(r) for r in rows)
    cp.cuda.Device().synchronize()
    times = np.arange(len(rows), dtype=np.int64) * 5_000_000_000
    gate = CrossCheckGate()
    risk = RollingRisk(
        backend="gpu", min_interval_ns=0, cross_check=True, gate=gate, full_window_bars=120
    )
    xc = CpuCrossCheck(cpu_threads=1, full_window_bars=120)
    risk.start()
    xc.start()
    methods = []
    try:
        for end in [*range(20, 121), *range(121, 421)]:
            start = max(0, end - 120)
            risk.offer(BarWindow(parts[start:end], times[start:end], 1, end, 0, device=True))
            update = risk.poll()
            assert update.validated
            xc.submit(update)
            while (comparison := xc.poll()) is None:
                pass
            gate.release()  # stands in for the published comparison
            assert comparison.error is None and comparison.agreement, end
            assert comparison.matched_work, (end, comparison.cpu["method"], update.result["method"])
            methods.append(update.result["method"])
    finally:
        risk.stop()
        xc.stop()
    assert methods.count("rebuild") == 2 and methods[0] == methods[256] == "rebuild"


def test_risk_bench_gpu_backends_agree_with_the_cpu_and_the_oracle(tmp_path):
    pytest.importorskip("threadpoolctl")
    from holoscan_finance.tools import risk_bench

    out = tmp_path / "risk_bench.json"
    argv = ["--sizes", "40", "--backends", "gpu,gpu-host,cpu", "--windows", "6", "--check"]
    assert risk_bench.main([*argv, "--cpu-threads", "1", "--out", str(out)]) == 0
    check = json.loads(out.read_text())["sizes"]["40"]["check"]
    assert set(check.values()) == {"7/7"} and len(check) == 5


def test_risk_bench_kinds_every_gpu_path_and_compare_mode_agree_and_are_judged(tmp_path):
    """``--kinds`` on the GPU: both GPU paths and compare mode's processors (every window
    validated, its CPU twin in the spawned worker tuned ahead) agree with the oracle and the
    CPU on every kept window and do the same work; the verdict judges every GPU path per
    pass and kind against the CPU at p50 and p99."""
    pytest.importorskip("threadpoolctl")
    from holoscan_finance.tools import risk_bench

    out = tmp_path / "kinds.json"
    argv = ["--sizes", "40", "--backends", "gpu,gpu-host,cpu,compare", "--kinds", "2"]
    argv += ["--cadence-ms", "1", "--cadence-cycles", "1", "--check", "--check-cycles", "2"]
    assert risk_bench.main([*argv, "--cpu-threads", "1", "--out", str(out)]) == 0
    report = json.loads(out.read_text())
    entry = report["sizes"]["40"]
    assert len(entry["check"]) == 2 * 9 and all(
        v.split("/")[0] == v.split("/")[1] for v in entry["check"].values()
    )
    for name in ("back_to_back", "cadence_1ms"):
        runs = entry["passes"][name]
        assert runs["compare"]["cpu_tunings"] == 1  # tuned before the first validated window
        for kind in risk_bench.KINDS:
            work = [
                runs[b]["kinds"][kind[0]]["work"] for b in ("gpu", "gpu-host", "cpu", "compare")
            ]
            assert work[0] == work[1] == work[2] == work[3] and len(work[0]) == 1
            assert {"gpu_ms", "cpu_ms"} <= set(runs["compare"]["kinds"][kind[0]])
    assert report["verdict"]["cells"] == 2 * 6 * 3
