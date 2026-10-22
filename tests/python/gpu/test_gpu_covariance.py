# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``CovarianceEngine("gpu")``: the NumPy oracle tests on the GPU backend, device windows
consumed in place, the device statistics bit-identical to the CPU engine's NumPy
expressions given the same matrix, graphs for every window kind, the eager fallback after
an invalidated capture, and windows that allocate nothing."""

from __future__ import annotations

import threading

import numpy as np
import pytest
from library.test_covariance import (
    assert_matches_oracle,
    check_a_large_move_then_quiet_trading_keeps_every_statistic_exact,
    check_incremental_covariance_matches_numpy_oracle,
    check_no_collection_starts_while_a_window_is_timed,
    check_parts_and_invalid_input,
    check_rolling_windows_match_the_oracle_through_a_rebuild,
    check_windows_longer_than_the_workspaces_grow_them_and_stay_incremental,
    rolling_rows,
)

from holoscan_finance.analytics.rolling import RollingRisk
from holoscan_finance.marketdata import covariance
from holoscan_finance.marketdata.covariance import CovarianceEngine
from holoscan_finance.payloads import BarWindow

pytestmark = pytest.mark.gpu
cp = pytest.importorskip("cupy")


def test_incremental_covariance_matches_independent_numpy_oracle():
    check_incremental_covariance_matches_numpy_oracle("gpu")


def test_no_collection_starts_while_a_device_window_is_timed(monkeypatch):
    """The GPU path allocated more tracked objects per window than the CPU path, and its
    slowest windows were collections."""
    check_no_collection_starts_while_a_window_is_timed("gpu", monkeypatch, xp=cp)


def test_covariance_engine_consumes_a_device_window_in_place():
    """GPU ingest feeds analytics a CuPy window; results must match the host-window
    path, including incremental updates, a supplied variance and a completed-bar check."""
    rng = np.random.default_rng(5)
    rows = rng.normal(0, 0.001, (90, 40))
    rows[3:9, 2] = np.nan
    rows[:, 7] = 0
    host_engine, dev_engine = CovarianceEngine("gpu"), CovarianceEngine("gpu")
    for end in (60, 61, 62, 70, 90):
        window, times = rows[end - 60 : end], np.arange(end - 60, end)
        h = host_engine.update(window, times, return_matrix=True)
        d = dev_engine.update(cp.asarray(window), times, return_matrix=True)
        assert (h["method"], h["added_bars"], h["expired_bars"]) == (
            d["method"],
            d["added_bars"],
            d["expired_bars"],
        )
        assert h["eligible_indices"] == d["eligible_indices"]
        assert h["heatmap_indices"] == d["heatmap_indices"]
        np.testing.assert_allclose(h["matrix"], d["matrix"], rtol=1e-10, atol=1e-16)
        np.testing.assert_allclose(h["annual_vol"], d["annual_vol"], rtol=1e-10)
    supplied = dev_engine.update(
        cp.asarray(rows[30:90]),
        np.arange(30, 90),
        observed_variance=np.var(rows[30:90], axis=0, ddof=1),
    )
    assert supplied["eligible_indices"] == d["eligible_indices"]
    changed = rows[30:90].copy()
    changed[0, 0] += 0.01
    with pytest.raises(ValueError, match="completed bars changed"):
        dev_engine.update(cp.asarray(changed), np.arange(30, 90))


def test_rolling_windows_match_the_oracle_through_a_rebuild():
    check_rolling_windows_match_the_oracle_through_a_rebuild("gpu", cp)


def test_parts_and_invalid_input():
    check_parts_and_invalid_input("gpu", cp)


def test_windows_longer_than_the_workspaces_grow_them_and_stay_incremental():
    check_windows_longer_than_the_workspaces_grow_them_and_stay_incremental("gpu", cp)


def test_a_large_move_then_quiet_trading_keeps_every_statistic_exact():
    check_a_large_move_then_quiet_trading_keeps_every_statistic_exact("gpu", cp)


def test_device_variance_is_numpy_variance_bit_for_bit():
    rows = np.random.default_rng(17).normal(0, 0.01, (60, 300))
    rows[10:12, 7] = np.nan
    rows[:, 9] = 0.25
    rows *= np.random.default_rng(18).uniform(1e-3, 1e3, 300)
    engine = CovarianceEngine("gpu")
    engine.update(rows, np.arange(60))
    np.testing.assert_array_equal(engine._core.obs.get(), np.var(rows, axis=0, ddof=1))


def tied_window(count: int, width: int, seed: int) -> np.ndarray:
    """``count`` eligible names scattered over ``width``, in triples of equal returns (so
    their variances tie, also at the 24th rank); the rest constant or with a NaN."""
    rng = np.random.default_rng(seed)
    rows = np.zeros((30, width))
    rows[:, 1::3] = np.nan
    names = rng.permutation(width)[:count]
    base = rng.normal(0, 0.001, (30, count // 3 + 1))
    rows[:, names] = base[:, np.arange(count) // 3]
    return rows


@pytest.mark.parametrize("count", [0, 1, 2, 23, 24, 25, 31, 32, 33, 1500])
def test_device_selection_and_finishing_equal_the_numpy_expressions(count):
    """Eligible order, heatmap names (stable ties at rank 24), probe slots (linspace) and the
    volatility, heatmap and probe values: bit-identical to the CPU engine's NumPy finishing
    applied to the same matrix and variance; the mean correlation sums in another order."""
    width = 2600 if count > 40 else 80
    rows = tied_window(count, width, count)
    engine = CovarianceEngine("gpu")
    times = np.arange(30)
    result = engine.update(rows, times)
    for step in (1, 2):  # and after incremental updates of the same matrix
        shifted = np.roll(rows, -step, axis=0)
        result = engine.update(shifted[: 30 - step], times[step:30])
    sample = rows[2:30]
    c = engine._core.c.get()
    want = covariance._finish(c, np.var(sample, axis=0, ddof=1), 28, width, engine.annualize)
    assert result["eligible"] == want["count"] == count
    assert result["eligible_indices"] == want["indices"].tolist()
    assert result["heatmap_indices"] == want["heat_index"].tolist()
    np.testing.assert_array_equal(result["annual_vol"], want["vol"])
    np.testing.assert_array_equal(
        np.reshape(result["heatmap"], want["heatmap"].shape), want["heatmap"]
    )
    np.testing.assert_array_equal(
        np.reshape(result["covariance_probes"], want["probes"].shape), want["probes"]
    )
    if count > 1:
        assert abs(result["mean_correlation"] - want["mean_correlation"]) < 1e-13
    else:
        assert result["mean_correlation"] is want["mean_correlation"] is None
    assert_matches_oracle(result, sample)


class _Counting:
    """Counts the allocations of the default device and pinned pools (installed as CuPy's
    allocators, so every cupy and alloc_pinned_memory allocation goes through it)."""

    def __init__(self) -> None:
        self.device = self.pinned = 0
        self._pool, self._pinned = cp.get_default_memory_pool(), cp.get_default_pinned_memory_pool()

    def _device(self, size):
        self.device += 1
        return self._pool.malloc(size)

    def _pinned_malloc(self, size):
        self.pinned += 1
        return self._pinned.malloc(size)

    def __enter__(self):
        cp.cuda.set_allocator(self._device)
        cp.cuda.set_pinned_memory_allocator(self._pinned_malloc)
        return self

    def __exit__(self, *exc):
        cp.cuda.set_allocator(self._pool.malloc)
        cp.cuda.set_pinned_memory_allocator(self._pinned.malloc)


def _schedule(rows: np.ndarray, generation: int, *, first: int = 20, window: int = 60):
    """``(window, phase)`` of one generation as after a start or seek: the first window,
    growth to the full window, one-, two- and three-bar slides."""
    parts = tuple(cp.asarray(r) for r in rows)
    cp.cuda.Device().synchronize()
    times = np.arange(len(rows), dtype=np.int64)
    spans = [(0, end, "growth") for end in range(first, window + 1)]
    spans[0] = (0, first, "first")
    start = 0
    for step in [1] * 30 + [2] * 5 + [3] * 5:
        start += step
        if start + window > len(rows):
            break
        spans.append((start, start + window, f"slide{step}"))
    return [
        (BarWindow(parts[a:b], times[a:b], generation, b, 0, device=True), phase)
        for a, b, phase in spans
    ]


@pytest.mark.parametrize("supplied", [False, True])
def test_no_window_kind_allocates_device_or_pinned_memory(supplied):
    """After a universe's first window, no window allocates: growth after a start or seek,
    one- to three-bar slides and a changing eligible count over two generations, through
    ``RollingRisk``, and the same windows with a supplied host variance through the engine.
    Counted at CuPy's allocators, so a pinned allocation kept per window, or one in an eager
    window, is seen."""
    rows = np.random.default_rng(21).normal(0, 0.001, (330, 24))
    for j in range(10):  # staggered gaps: the eligible count changes
        rows[40 + 23 * j : 43 + 23 * j, j : j + 3] = np.nan
    windows = _schedule(rows[:165], 1) + _schedule(rows[165:], 2)
    if supplied:
        engine = CovarianceEngine("gpu", window_bars=60)
        variances = [
            np.var(np.stack([p.get() for p in w.parts]), axis=0, ddof=1) for w, _ in windows
        ]

        def analyze(i, window):
            if i and window.generation != windows[i - 1][0].generation:
                engine.reset()
            return engine.update(
                None, window.times, parts=window.parts, observed_variance=variances[i]
            )
    else:
        risk = RollingRisk(backend="gpu", min_interval_ns=0)
        risk.start()

        def analyze(i, window):
            risk.offer(window)
            return risk.poll().result

    analyze(0, windows[0][0])  # allocates the workspaces and captures the graphs
    eligible, phases = set(), set()
    with _Counting() as counted:
        for i, (window, phase) in enumerate(windows[1:], 1):
            result = analyze(i, window)
            eligible.add(result["eligible"])
            phases.add(phase)
    if not supplied:
        risk.stop()
    assert len(eligible) > 3 and phases == {"growth", "slide1", "slide2", "slide3", "first"}
    assert (counted.device, counted.pinned) == (0, 0)


def _results_equal(want: dict, got: dict, where) -> None:
    for key in want:
        if key != "milliseconds":
            assert got[key] == want[key], (where, key)


def test_graph_windows_equal_eager_windows_bit_for_bit():
    """Every window runs as one of the two graphs captured at allocation (while another
    thread works on the legacy stream): growth, one- to three-bar slides, the 256-call
    rebuild, supplied variances, host rows and a forced rebuild. Bit-identical to the same
    work issued eagerly."""
    rows = rolling_rows()
    parts = tuple(cp.asarray(r) for r in rows)
    cp.cuda.Device().synchronize()
    stop = threading.Event()

    def legacy_stream_work():
        x = cp.arange(4096, dtype=cp.float64)
        while not stop.is_set():
            float((x * 2).sum())

    busy = threading.Thread(target=legacy_stream_work)
    busy.start()
    try:
        graphed = CovarianceEngine("gpu", window_bars=60)
        graphed.update(None, np.arange(60), parts=parts[:60])
    finally:
        stop.set()
        busy.join()
    assert sorted(graphed._core.graphs) == [0, 1]
    eager = CovarianceEngine("gpu", window_bars=60)
    eager.update(None, np.arange(60), parts=parts[:60])
    eager._core.graphs = {}
    eager_calls = []
    run_eagerly = graphed._core._enqueue
    graphed._core._enqueue = lambda *a: eager_calls.append(a) or run_eagerly(*a)
    graphed.reset()
    eager.reset()
    end, steps, kinds, calls = 20, [1, 1, 2, 1, 1, 3, 1, 1], set(), 0
    while end <= len(rows):
        start, times = max(0, end - 60), np.arange(max(0, end - 60), end)
        observed = np.var(rows[start:end], axis=0, ddof=1) if end % 5 == 0 else None
        want = eager.update(None, times, parts=parts[start:end], observed_variance=observed)
        got = graphed.update(None, times, parts=parts[start:end], observed_variance=observed)
        _results_equal(want, got, end)
        kinds.add((want["method"], want["bars"] < 60, want["added_bars"]))
        calls += 1
        end += steps[calls % len(steps)]
    assert ("rebuild", False, 60) in kinds and ("incremental", True, 1) in kinds
    assert {("incremental", False, k) for k in (1, 2, 3)} <= kinds
    # Host rows as parts (a spawned worker's window), with a forced full-window rebuild,
    # and device rows.
    host_parts = tuple(rows)
    for end in (200, 201, 202):
        times = np.arange(end - 60, end)
        kwargs = dict(parts=host_parts[end - 60 : end], force_rebuild=end == 201)
        _results_equal(
            eager.update(None, times, **kwargs), graphed.update(None, times, **kwargs), end
        )
    for end in (203, 205):
        window, times = cp.asarray(rows[end - 60 : end]), np.arange(end - 60, end)
        _results_equal(eager.update(window, times), graphed.update(window, times), end)
    assert eager_calls == []


def test_an_invalidated_capture_runs_windows_eagerly_until_the_next_capture(monkeypatch):
    """A capture that fails (here a device-wide synchronize during it) leaves the engine
    running the same work eagerly, correct, and capturing again ``REBUILD_EVERY`` windows
    later."""
    rows = np.random.default_rng(3).normal(0, 0.001, (300, 30))
    rows[:, 4] = 0
    enqueue, fail = covariance._GpuCore._enqueue, [True]

    def failing_enqueue(core, parity):
        if fail[0] and core.stream.is_capturing():
            cp.cuda.Device().synchronize()  # not permitted while capturing
        enqueue(core, parity)

    monkeypatch.setattr(covariance._GpuCore, "_enqueue", failing_enqueue)
    engine = CovarianceEngine("gpu", window_bars=20)
    assert_matches_oracle(engine.update(rows[:20], np.arange(20)), rows[:20])
    assert engine._core.graphs == {}
    assert engine._core._capture_in == covariance.REBUILD_EVERY - 1
    fail[0] = False
    for end in range(21, 21 + covariance.REBUILD_EVERY):
        window, times = rows[end - 20 : end], np.arange(end - 20, end)
        assert_matches_oracle(engine.update(window, times), window)
        if end == 18 + covariance.REBUILD_EVERY:  # the last eager window
            assert engine._core.graphs == {} and engine._core._capture_in == 1
    assert sorted(engine._core.graphs) == [0, 1] and engine._core._capture_in is None


def test_a_device_wide_synchronize_in_another_thread_never_fails_a_window():
    """Another thread's ``Device().synchronize()`` during the captures at a universe's first
    window (an operator elsewhere in the graph) invalidates them; the window still succeeds."""
    rows = np.random.default_rng(4).normal(0, 0.001, (20, 30))
    rows[:, 4] = 0
    want = CovarianceEngine("cpu").update(rows, np.arange(20))
    stop = threading.Event()

    def device_wide_synchronize():
        while not stop.is_set():
            try:
                cp.cuda.Device().synchronize()
            except cp.cuda.runtime.CUDARuntimeError:  # while the engine captures
                pass

    other = threading.Thread(target=device_wide_synchronize)
    other.start()
    try:
        for _ in range(20):
            got = CovarianceEngine("gpu", window_bars=20).update(rows, np.arange(20))
            assert got["eligible_indices"] == want["eligible_indices"]
            np.testing.assert_allclose(got["annual_vol"], want["annual_vol"], rtol=1e-10)
    finally:
        stop.set()
        other.join()


def test_a_strided_device_variance_is_read_as_its_values():
    """A device variance may be any float64 view, such as a column of a device table."""
    rows = np.random.default_rng(8).normal(0, 0.001, (30, 40))
    rows[:, [3, 17]] = 0
    times = np.arange(30)
    variance = np.var(rows, axis=0, ddof=1)
    want = CovarianceEngine("cpu").update(rows, times, observed_variance=variance)
    table = cp.zeros((40, 3))
    table[:, 1] = cp.asarray(variance)
    for column in (table[:, 1], cp.asarray(variance)[::-1][::-1]):
        got = CovarianceEngine("gpu").update(rows, times, observed_variance=column)
        assert got["eligible_indices"] == want["eligible_indices"]
        assert got["heatmap_indices"] == want["heatmap_indices"]
        np.testing.assert_allclose(got["annual_vol"], want["annual_vol"], rtol=1e-10)


_SPIN = r"""
extern "C" __global__ void spin(const long long cycles) {
    const long long start = clock64();
    while (clock64() - start < cycles) {}
}
"""


def test_device_inputs_written_on_the_callers_stream_are_read_after_it():
    """The engine reads on its own non-blocking stream. A device variance written by the
    caller's queued work, and device parts the engine itself converts to float64 on the
    caller's stream, are read only after that work (here held back ~50 ms by a spin)."""
    rows = np.random.default_rng(9).normal(0, 0.001, (30, 40))
    rows[:, 5] = 0
    times = np.arange(30)
    d_rows = cp.asarray(rows)
    observed = cp.var(d_rows, axis=0, ddof=1).get()
    rows32 = rows.astype(np.float32).astype(np.float64)
    want = {
        "variance": CovarianceEngine("cpu").update(rows, times, observed_variance=observed),
        "parts": CovarianceEngine("cpu").update(rows32, times),
    }
    parts32 = tuple(cp.asarray(r, dtype=cp.float32) for r in rows)
    parts32[0].astype(cp.float64)  # the conversion kernel is loaded before the spin
    variance = cp.zeros(40)  # read too early, no name would be eligible
    spin = cp.RawKernel(_SPIN, "spin")
    engine = CovarianceEngine("gpu")
    engine.update(rows, times)  # allocates the workspaces
    for case in ("variance", "parts"):
        engine.reset()
        spare = [cp.empty(40) for _ in range(60)]  # pooled blocks: no cudaMalloc below
        del spare
        cp.cuda.Device().synchronize()
        spin((1,), (1,), (np.int64(10**8),))  # the caller's (legacy) stream stays busy
        if case == "variance":
            variance[...] = cp.var(d_rows, axis=0, ddof=1)
            got = engine.update(rows, times, observed_variance=variance)
        else:
            got = engine.update(None, times, parts=parts32)
        assert got["eligible_indices"] == want[case]["eligible_indices"], case
        assert got["heatmap_indices"] == want[case]["heatmap_indices"], case
        np.testing.assert_allclose(got["annual_vol"], want[case]["annual_vol"], rtol=1e-10)
