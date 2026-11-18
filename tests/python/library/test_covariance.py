# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``CovarianceEngine`` on the CPU against an independent NumPy oracle: rolling, superseded
and rebuilt windows, settling to zero eligibility, the parts path and invalid input; the
garbage collector held off while a window is timed; and the cached GPU name. The shared
checks take the backend and the array module, and ``gpu/test_gpu_covariance.py`` runs them
on the GPU."""

from __future__ import annotations

import gc
import threading

import numpy as np
import pytest

from holoscan_finance.marketdata import covariance
from holoscan_finance.marketdata.covariance import CovarianceEngine, slide


def check_incremental_covariance_matches_numpy_oracle(backend):
    """Shared with the GPU test: rolling, rebuilt and settled windows on ``backend``."""
    rng = np.random.default_rng(120)
    rows = rng.normal(0, 0.001, (50, 12))
    rows[3:7, 1] = np.nan
    rows[:, 2] = 0
    rows[18:20, 4] = np.nan
    engine = CovarianceEngine(backend)
    for end in (10, 11, 15, 23, 30, 45):
        sample = rows[end - 10 : end]
        result = engine.update(sample, np.arange(end - 10, end), return_matrix=True)
        eligible = np.flatnonzero(
            np.isfinite(sample).all(axis=0) & (np.nanvar(sample, axis=0) > 1e-20)
        )
        assert result["eligible_indices"] == eligible.tolist()
        oracle = np.cov(sample[:, eligible], rowvar=False, ddof=1)
        np.testing.assert_allclose(result["matrix"], oracle, rtol=1e-10, atol=1e-16)
        np.testing.assert_allclose(
            result["annual_vol"], np.sqrt(oracle.diagonal() * 252 * 23400 / 5), rtol=1e-10
        )
    changed = rows[35:45].copy()
    changed[0, 0] += 0.01
    with pytest.raises(ValueError, match="completed bars changed"):
        engine.update(changed, np.arange(35, 45))
    reset = engine.update(rows[:10], np.arange(10), return_matrix=True)
    assert reset["method"] == "rebuild"
    # All movement expires before the periodic rebuild. Subtraction roundoff
    # must not leave phantom volatility or inconsistent CPU/GPU eligibility.
    settling = rng.normal(0, 0.1, (120, 12))
    settling[60:] = 0
    engine = CovarianceEngine(backend)
    for end in range(60, 121):
        result = engine.update(settling[end - 60 : end], np.arange(end - 60, end))
    assert result["eligible_indices"] == []
    assert result["mean_correlation"] is None


def test_incremental_covariance_matches_independent_numpy_oracle():
    check_incremental_covariance_matches_numpy_oracle("cpu")


def assert_matches_oracle(result, sample, *, rtol=1e-9):
    """Every result field against an explicit NumPy computation; the mean correlation is the
    unclipped mean of the off-diagonal correlations."""
    var = np.var(sample, axis=0, ddof=1)
    eligible = np.flatnonzero(np.isfinite(var) & (var > 1e-20))
    count = len(eligible)
    assert result["eligible_indices"] == eligible.tolist() and result["eligible"] == count
    if count == 0:
        assert result["annual_vol"] == [] and result["mean_correlation"] is None
        return
    cov = np.atleast_2d(np.cov(sample[:, eligible], rowvar=False, ddof=1))
    std = np.sqrt(np.diag(cov))
    np.testing.assert_allclose(result["annual_vol"], std * np.sqrt(252 * 23400 / 5), rtol=rtol)
    top = np.argsort(-var[eligible], kind="stable")[:24]
    assert result["heatmap_indices"] == eligible[top].tolist()
    corr = cov / np.outer(std, std)
    np.testing.assert_allclose(result["heatmap"], corr[np.ix_(top, top)], rtol=rtol, atol=1e-12)
    probe = np.linspace(0, count - 1, min(32, count), dtype=int)
    np.testing.assert_allclose(
        result["covariance_probes"], cov[np.ix_(probe, probe)], rtol=rtol, atol=1e-16
    )
    if count > 1:
        mean = (corr.sum() - np.trace(corr)) / (count * (count - 1))
        assert abs(result["mean_correlation"] - mean) < 1e-12
    else:
        assert result["mean_correlation"] is None


def rolling_rows() -> np.ndarray:
    """Returns whose windows see NaN gaps enter and leave, tied variances, a never-moving
    name and one that stops moving for a full window."""
    rng = np.random.default_rng(31)
    rows = rng.normal(0, 0.001, (420, 15))
    rows[40:45, 1] = np.nan
    rows[100:, 3] = rows[100:, 4]
    rows[:, 5] = 0
    rows[200:265, 6] = 0.0005
    return rows


def check_rolling_windows_match_the_oracle_through_a_rebuild(backend, xp=np):
    """Ramp-up, slides of one to three bars (superseded windows), the 256-call rebuild;
    ``xp`` given: the window travels as per-bar parts of that array module."""
    rows = rolling_rows()
    parts = tuple(xp.asarray(r) for r in rows)
    if xp is not np:
        xp.cuda.Device().synchronize()  # complete parts, as QuoteBarsOp emits them
    engine = CovarianceEngine(backend)
    methods, end, steps = [], 20, [1, 1, 2, 1, 3, 1]
    while end <= len(rows):
        start = max(0, end - 60)
        sample, times = rows[start:end], np.arange(start, end)
        result = engine.update(None, times, parts=parts[start:end])
        assert_matches_oracle(result, sample)
        methods.append((result["method"], result["expired_bars"], result["added_bars"]))
        end += steps[len(methods) % len(steps)]
    assert len(methods) > 257
    assert methods[0] == ("rebuild", 0, 20) and methods[256][0] == "rebuild"
    rest = methods[1:256] + methods[257:]
    assert {m for m, _, _ in rest} == {"incremental"}
    assert {(e, a) for _, e, a in rest if e} == {(1, 1), (2, 2), (3, 3)}


def test_rolling_windows_match_the_oracle_through_a_rebuild():
    check_rolling_windows_match_the_oracle_through_a_rebuild("cpu")


def check_parts_and_invalid_input(backend, xp=np):
    """Parts: identical data in other objects is accepted, changed data under the same times
    raises and changes nothing; an infinite return raises, and the next call rebuilds."""
    rows = np.random.default_rng(4).normal(0, 0.001, (14, 6))
    parts = tuple(xp.asarray(r) for r in rows)
    changed = rows[1].copy()
    changed[2] += 0.01
    infinite = rows[11].copy()
    infinite[3] = np.inf
    extra = (xp.asarray(changed), xp.asarray(infinite))
    if xp is not np:
        xp.cuda.Device().synchronize()
    engine = CovarianceEngine(backend)
    times = np.arange(14)
    assert engine.update(None, times[:10], parts=parts[:10])["method"] == "rebuild"
    with pytest.raises(ValueError, match="completed bars changed"):
        engine.update(None, times[1:11], parts=(extra[0], *parts[2:11]))
    copies = tuple(xp.asarray(r) for r in rows[1:11])
    if xp is not np:
        xp.cuda.Device().synchronize()
    result = engine.update(None, times[1:11], parts=copies)
    assert (result["method"], result["expired_bars"], result["added_bars"]) == ("incremental", 1, 1)
    assert_matches_oracle(result, rows[1:11])
    with pytest.raises(ValueError, match="finite-or-NaN"):
        engine.update(None, times[2:12], parts=(*parts[2:11], extra[1]))
    result = engine.update(None, times[2:12], parts=parts[2:12])
    assert result["method"] == "rebuild"
    assert_matches_oracle(result, rows[2:12])
    with pytest.raises(ValueError, match="finite-or-NaN"):
        engine.update(rows[3:13], times[3:13][::-1])
    with pytest.raises(ValueError, match="finite-or-NaN"):
        engine.update(None, times[3:13], parts=parts[3:12])


def test_parts_and_invalid_input():
    check_parts_and_invalid_input("cpu")


def check_windows_longer_than_the_workspaces_grow_them_and_stay_incremental(backend, xp=np):
    """With no window-length hint, growth windows past the 64-bar workspaces grow them,
    keeping the statistics: every growth window from 21 to 130 bars and the slides after
    are incremental, match the oracle, and the workspaces are allocated once."""
    rows = np.random.default_rng(64).normal(0, 0.001, (200, 9))
    rows[70:75, 2] = np.nan
    parts = tuple(xp.asarray(r) for r in rows)
    if xp is not np:
        xp.cuda.Device().synchronize()
    engine = CovarianceEngine(backend)
    allocate, calls = engine._core.allocate, []
    engine._core.allocate = lambda *a: calls.append(a) or allocate(*a)
    methods = []
    for end in [*range(20, 131), *range(132, 200, 2)]:
        start = max(0, end - 130)
        result = engine.update(None, np.arange(start, end), parts=parts[start:end])
        assert_matches_oracle(result, rows[start:end])
        methods.append(result["method"])
    assert methods[0] == "rebuild" and set(methods[1:]) == {"incremental"}
    assert len(calls) == 1 and engine._core.capacity >= 130
    hinted = CovarianceEngine(backend, window_bars=130)
    hinted.update(None, np.arange(20), parts=parts[:20])
    assert hinted._core.capacity == 130


def test_windows_longer_than_the_workspaces_grow_them_and_stay_incremental():
    check_windows_longer_than_the_workspaces_grow_them_and_stay_incremental("cpu")


def quiet_after_a_large_move(seed: int = 1001) -> np.ndarray:
    """Columns 0-1: a large return, then quiet trading (a 10x quote glitch and back, then
    1 bp ticks in 20% of bars; a 26% jump, then half-cent ticks on $1,000 in 5% of bars).
    Column 2: one 0.5 return, then +-1.5e-10 (a 1e-8 price unit on $100). Others: noise."""
    rng = np.random.default_rng(seed)
    rows = rng.normal(0, 3e-4, (300, 10))
    rows[75, 0], rows[76, 0] = np.log(10.0), -np.log(10.0)
    rows[77:, 0] = np.where(rng.random(223) < 0.2, rng.choice([-1e-4, 1e-4], 223), 0.0)
    rows[75, 1] = np.log(1.26)
    rows[76:, 1] = np.where(rng.random(224) < 0.05, rng.choice([-5e-6, 5e-6], 224), 0.0)
    rows[80, 2] = 0.5
    rows[81:, 2] = 1.5e-10 * np.where(np.arange(219) % 2, 1.0, -1.0)
    return rows


def check_a_large_move_then_quiet_trading_keeps_every_statistic_exact(backend, xp=np):
    """Once a large return expires, subtracting it cancels most of its column's diagonal of
    ``C``, and the rounding it left there could dominate a quiet name's statistics (or zero
    its variance while the name is still eligible: NaN correlations). The engine recomputes
    such a window from its bars (``rebuild``), so every window before the 256-call rebuild
    matches the oracle, on one-bar slides and, near the end, two-bar slides."""
    rows = quiet_after_a_large_move()
    parts = tuple(xp.asarray(r) for r in rows)
    if xp is not np:
        xp.cuda.Device().synchronize()
    engine = CovarianceEngine(backend, window_bars=60)
    recomputed = []
    for end in [*range(60, 280), *range(281, 301, 2)]:
        window = rows[end - 60 : end]
        result = engine.update(None, np.arange(end - 60, end), parts=parts[end - 60 : end])
        assert_matches_oracle(result, window)
        assert np.isfinite(result["heatmap"]).all() and np.isfinite(result["annual_vol"]).all()
        if result["method"] == "rebuild" and end > 60:
            recomputed.append(end - 60)
    assert recomputed and min(recomputed) >= 76  # once a large move has expired (bar 75)


def test_a_large_move_then_quiet_trading_keeps_every_statistic_exact():
    check_a_large_move_then_quiet_trading_keeps_every_statistic_exact("cpu")


def test_a_window_slides_only_by_a_shared_run_of_bars():
    old = np.arange(10, 20)
    assert slide(old, np.arange(10, 21)) == (0, 1)
    assert slide(old, np.arange(13, 23)) == (3, 3)
    assert slide(old, np.arange(19, 25)) == (9, 5)
    assert slide(old, np.arange(20, 30)) is None  # nothing shared
    assert slide(old, np.arange(5, 15)) is None  # back in time
    assert slide(old, np.r_[12:19, 20, 21]) is None  # a bar missing in the middle
    assert slide(old, np.arange(12, 18)) is None  # ends before the old window


def check_no_collection_starts_while_a_window_is_timed(backend, monkeypatch, xp=np):
    """A collection starts at whichever allocation crosses the collector's threshold; one
    inside a window added its whole pause to the window's time. ``update`` holds automatic
    collection off while it times a window, so with a threshold of one allocation the
    collections still run (after each window), none starts inside one, and the collector
    is on again afterwards."""
    rows = np.random.default_rng(3).normal(0, 0.001, (40, 8))
    parts = tuple(xp.asarray(row) for row in rows)
    engine = CovarianceEngine(backend, window_bars=20)
    engine.update(rows[:20], np.arange(20))  # allocates outside the measurement
    timing, enabled, starts = [], [], []
    timed = CovarianceEngine._window

    def window(self, *args):
        timing.append(True)
        enabled.append(gc.isenabled())
        try:
            return timed(self, *args)
        finally:
            timing.pop()

    def callback(phase, info):
        if phase == "start":
            starts.append(bool(timing))

    monkeypatch.setattr(CovarianceEngine, "_window", window)
    threshold = gc.get_threshold()
    gc.callbacks.append(callback)
    gc.set_threshold(1, 1000, 1000)
    try:
        for end in (*range(20, 31), 40):  # slides, then a rebuild
            engine.update(None, np.arange(end - 20, end), parts=parts[end - 20 : end])
    finally:
        gc.set_threshold(*threshold)
        gc.callbacks.remove(callback)
    assert len(enabled) == 12 and not any(enabled)
    assert starts and not any(starts)
    assert gc.isenabled()


def test_no_collection_starts_while_a_window_is_timed(monkeypatch):
    check_no_collection_starts_while_a_window_is_timed("cpu", monkeypatch)


def test_the_collector_hold_spans_concurrent_windows_and_keeps_a_disabled_collector_off():
    """Windows timed at once in two threads: the collector comes back on with the last one,
    and only if it was on before the first."""
    assert gc.isenabled()
    covariance._hold_collector()
    other = threading.Thread(target=covariance._hold_collector)
    other.start()
    other.join()
    covariance._release_collector()
    assert not gc.isenabled()  # the other window is still being timed
    covariance._release_collector()
    assert gc.isenabled()
    gc.disable()
    try:
        covariance._hold_collector()
        covariance._release_collector()
        assert not gc.isenabled()
    finally:
        gc.enable()


def test_reset_keeps_the_workspaces_and_rebuilds():
    rows = np.random.default_rng(9).normal(0, 0.001, (12, 5))
    engine = CovarianceEngine()
    engine.update(rows[:10], np.arange(10))
    c = engine._core.c
    engine.reset()
    result = engine.update(rows[2:12], np.arange(2, 12))
    assert result["method"] == "rebuild" and engine._core.c is c and engine.width == 5


def test_gpu_name_is_queried_once():
    class Runtime:
        calls = 0

        def getDeviceProperties(self, device):
            Runtime.calls += 1
            return {"name": b"Test GPU"}

    class Xp:
        class cuda:
            runtime = Runtime()

    covariance._gpu_names.pop(99, None)
    assert [covariance.gpu_name(Xp, 99) for _ in range(3)] == ["Test GPU"] * 3
    assert Runtime.calls == 1
