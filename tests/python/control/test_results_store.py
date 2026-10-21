# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ResultsStore: 4.3 publish_gpu/publish_comparison formatting, filters, gate and failures."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from holoscan_finance.control.directory import ReplayDirectory
from holoscan_finance.control.gate import CrossCheckGate
from holoscan_finance.control.results import ResultsStore
from holoscan_finance.marketdata.covariance import CovarianceEngine
from holoscan_finance.nyq.mapping import PROTOCOL, TIMEZONE, day_start_ns
from holoscan_finance.payloads import BarWindow, RiskComparison, RiskUpdate
from holoscan_finance.ref import Knob
from holoscan_finance.settings import BarSettings
from holoscan_finance.telemetry.capture import CaptureTap, LatencyCapture

pytest.importorskip("scipy")

MIDNIGHT = day_start_ns("2025-04-01")
SYMBOLS = ["A", "B", "C", "D", "E"]
COMPARISON_KEYS = {
    "identifier", "generation", "timestamp_ns", "eligible", "pairs", "cpu_ms", "gpu_ms",
    "speedup", "agreement", "matched_work", "cpu_threads", "cpu_affinity",
    "cpu_thread_trials_ms", "cpu_tuned_bars", "gpu_name", "blas", "method", "added_bars",
    "expired_bars", "cpu_receive_to_result_ms", "published_wall_ns", "validation",
}  # fmt: skip


def manifest(generation=1, port=50001):
    return dict(
        format="nyq-equity-replay-v1",
        wire_protocol=PROTOCOL,
        market_timezone=TIMEZONE,
        symbols=SYMBOLS,
        generation=generation,
        sender_port=port,
        date="2025-04-01",
        start_ns=MIDNIGHT + 34200 * 10**9,
        end_ns=MIDNIGHT + 57600 * 10**9,
        events=100,
        profile=[],
    )


def window(generation=1, bars=8, recv_ns=None, seed=1):
    rng = np.random.default_rng(seed)
    parts = tuple(rng.normal(0.0, 1e-3, len(SYMBOLS)) for _ in range(bars))
    times = MIDNIGHT + (34200 + 5 * np.arange(1, bars + 1, dtype=np.int64)) * 10**9
    times.flags.writeable = False
    recv_ns = time.time_ns() - 3_000_000 if recv_ns is None else recv_ns
    return BarWindow(parts, times, generation, bars, recv_ns)


def engine_result(w, **extra):
    result = CovarianceEngine("cpu", bar_seconds=5).update(w.rows, w.times)
    return {**result, "finished_wall_ns": time.time_ns(), **extra}


def gpu_update(w, validated=False):
    result = engine_result(w, gpu_name="Test GPU")
    return RiskUpdate(w, result, w.rows if validated else None, "gpu", validated)


def cpu_result(w):
    return engine_result(
        w,
        threads=2,
        cpu_thread_trials_ms={1: 1.0, 2: 0.8},
        cpu_tuned_bars=8,
        cpu_affinity=[0, 1],
        blas=[{"name": "openblas", "version": "0"}],
    )


def comparison(update, *, ok=True, matched=True, method="incremental", error=None):
    gpu = dict(update.result, method=method)
    cpu = dict(cpu_result(update.window), method=method)
    if not ok:
        cpu["heatmap_indices"] = list(reversed(cpu["heatmap_indices"]))
    usable = ok and matched and gpu["eligible"] >= 2
    upd = RiskUpdate(update.window, gpu, update.rows_host, "gpu", True)
    return RiskComparison(upd, None if error else cpu, ok, matched, usable, error)


def make_store(backend="compare", capture=None, gate=None):
    directory = ReplayDirectory(Knob())
    directory.set_manifest(manifest())
    tap = CaptureTap()
    tap.capture = capture  # a capture whose interval covers the test windows
    store = ResultsStore(
        backend=backend,
        interval_s=0.05,
        settings=BarSettings(window_seconds=40, min_bars=2),
        directory=directory,
        tap=tap,
        gate=gate,
    )
    directory.add_listener(store.reset_generation)
    return store, directory


def test_on_risk_publishes_the_filtered_result():
    store, _ = make_store()
    w = window()
    update = gpu_update(w)
    before = time.time_ns()
    store.on_risk(update)
    live = store.live
    assert "matrix" not in live and "covariance_probes" not in live
    assert set(update.result) - {"matrix", "covariance_probes"} < set(live)
    assert live["timestamp_ns"] == int(w.times[-1]) and live["generation"] == 1
    assert before <= live["published_wall_ns"] <= time.time_ns()
    assert live["receive_to_result_ms"] == (live["published_wall_ns"] - w.recv_ns) / 1e6
    assert live["receive_to_result_ms"] >= 3.0
    assert "matrix" in update.result  # the payload's dict is never modified
    bar_s, vol, corr = store.history[-1]
    assert bar_s == int(w.times[-1]) // 10**9
    assert vol == float(np.median(live["annual_vol"])) and corr == live["mean_correlation"]
    assert (store.published, store.received_risk) == (1, 1)
    assert list(store.latencies_ms) == [live["receive_to_result_ms"]]
    assert store.analytics_max_ms == live["milliseconds"]


def test_results_of_another_generation_are_counted_but_not_published():
    store, directory = make_store()
    store.on_risk(gpu_update(window(generation=7)))
    assert store.live is None and not store.history
    assert (store.published, store.received_risk) == (0, 1)
    directory.set_manifest(manifest(2, 50002))
    store.on_risk(gpu_update(window(generation=1)))
    assert store.live is None and store.received_risk == 2


def test_on_comparison_formats_the_4_3_comparison_and_timings():
    store, _ = make_store()
    update = gpu_update(window(), validated=True)
    store.on_risk(update)
    cmp = comparison(update)
    store.on_comparison(cmp)
    c = store.comparison
    assert set(c) == COMPARISON_KEYS
    assert c["identifier"] == store.comparisons == 1 and c["generation"] == 1
    assert c["agreement"] and c["matched_work"]
    assert c["speedup"] == cmp.cpu["milliseconds"] / cmp.update.result["milliseconds"]
    assert c["cpu_threads"] == 2 and c["gpu_name"] == "Test GPU"
    assert c["cpu_receive_to_result_ms"] == (
        (cmp.cpu["finished_wall_ns"] - update.window.recv_ns) / 1e6
    )
    assert list(store.timings) == [[cmp.cpu["milliseconds"], cmp.update.result["milliseconds"]]]
    rolling = store.rolling_timing()
    assert set(rolling) == {"samples", "cpu_median_ms", "gpu_median_ms", "speedup"}
    assert rolling["samples"] == 1 and store.comparison_failure is None and store.error is None


def test_rebuild_windows_are_compared_but_not_timed():
    store, _ = make_store()
    update = gpu_update(window(), validated=True)
    store.on_comparison(comparison(update, method="rebuild"))
    assert store.comparisons == 1 and not store.timings and store.rolling_timing() is None


def test_a_disagreement_records_the_failure_and_the_error():
    store, _ = make_store()
    update = gpu_update(window(), validated=True)
    store.on_comparison(comparison(update, ok=False))
    c = store.comparison
    assert c["speedup"] is None and not c["agreement"] and not store.timings
    failure = store.comparison_failure
    assert failure["comparison"] == c and failure["comparison"] is not c
    assert failure["heatmap_indices_same"] is False and failure["eligible_indices_same"] is True
    assert set(failure["cpu"]) == set(failure["gpu"]) == {
        "method", "added_bars", "expired_bars", "bars", "universe", "heatmap_indices",
        "heatmap", "annual_vol", "eligible_indices", "covariance_probes",
    }  # fmt: skip
    assert store.error == "CPU/GPU comparison failed; inspect /snapshot before using timings"
    assert not store.disabled


def test_the_gate_is_released_after_publication_outside_the_store_lock():
    gate = CrossCheckGate()
    store, _ = make_store(gate=gate)
    update = gpu_update(window(), validated=True)
    seen = []
    release = gate.release

    def lock_is_free():
        free = store.lock.acquire(blocking=False)
        if free:
            store.lock.release()
        seen.append((store.comparisons, free))

    def check_release():
        t = threading.Thread(target=lock_is_free)  # the store lock is an RLock: ask a thread
        t.start()
        t.join()
        release()

    gate.release = check_release
    assert gate.try_claim()
    store.on_comparison(comparison(update))
    assert seen == [(1, True)] and gate.ready() and gate.releases == 1


def test_comparisons_of_an_old_generation_still_release_the_gate():
    gate = CrossCheckGate()
    store, directory = make_store(gate=gate)
    update = gpu_update(window(), validated=True)
    gate.try_claim()
    directory.set_manifest(manifest(2, 50002))
    store.on_comparison(comparison(update))
    assert store.comparison is None and store.received_comparisons == 1 and gate.ready()


def counting_releases(gate):
    """Count every release() call, including those that find nothing in flight."""
    calls = []
    release = gate.release
    gate.release = lambda: (calls.append(1), release())
    return calls


def test_a_worker_error_disables_publication_and_releases_the_gate():
    gate = CrossCheckGate()
    store, _ = make_store(gate=gate)
    update = gpu_update(window(), validated=True)
    gate.try_claim()
    releases = counting_releases(gate)
    store.on_comparison(comparison(update, error="BrokenProcessPool: worker died"))
    assert store.error == "Analytics failed: BrokenProcessPool: worker died"
    assert store.disabled and gate.ready() and store.comparison is None
    # Once: a second release could free the claim risk already took for its next window.
    assert len(releases) == 1
    store.on_risk(gpu_update(window()))
    assert store.live is None and store.received_risk == 1


def test_a_formatting_failure_releases_the_gate_once():
    gate = CrossCheckGate()
    store, _ = make_store(gate=gate)
    cmp = comparison(gpu_update(window(), validated=True))
    del cmp.cpu["milliseconds"]
    gate.try_claim()
    releases = counting_releases(gate)
    store.on_comparison(cmp)
    assert store.error == "Analytics failed: 'milliseconds'" and store.disabled
    assert gate.ready() and len(releases) == 1


def test_an_exception_in_on_risk_never_reaches_the_operator():
    gate = CrossCheckGate()
    store, _ = make_store(gate=gate)
    w = window()
    broken = RiskUpdate(w, {"milliseconds": 1.0}, None, "gpu", True)  # no annual_vol
    gate.try_claim()
    store.on_risk(broken)
    assert store.error.startswith("Analytics failed: ") and "annual_vol" in store.error
    assert store.disabled and gate.ready() and store.received_risk == 1
    assert store.live is None and store.published == 0  # nothing half-published
    store.on_comparison(comparison(gpu_update(w, validated=True)))
    assert store.comparison is None and store.received_comparisons == 1


def test_reset_generation_clears_the_published_state_only():
    store, directory = make_store()
    update = gpu_update(window(), validated=True)
    store.on_risk(update)
    store.on_comparison(comparison(update, ok=False))
    directory.set_manifest(manifest(2, 50002))
    assert store.live is None and store.comparison is None
    assert not store.history and not store.timings
    assert store.comparisons == 1 and store.comparison_failure is not None  # as 4.3


def test_view_copies_what_the_snapshot_shows():
    store, _ = make_store()
    update = gpu_update(window(), validated=True)
    store.on_risk(update)
    store.on_comparison(comparison(update))
    view = store.view()
    assert set(view) == {
        "error", "comparison_failure", "live", "comparison", "comparisons", "history",
        "rolling_timing",
    }  # fmt: skip
    assert view["live"] is store.live and view["rolling_timing"]["samples"] == 1
    view["history"].clear()
    assert len(store.history) == 1


def test_compare_mode_records_gpu_results_and_cpu_comparisons_into_the_capture():
    w = window()
    capture = LatencyCapture(60, 1, start_ns=w.recv_ns - 1, backend="compare")
    store, _ = make_store(capture=capture)
    update = gpu_update(w, validated=True)
    store.on_risk(update)
    store.on_comparison(comparison(update))
    assert capture.gpu[0]["receive_to_result_ms"] == store.live["receive_to_result_ms"]
    assert capture.cpu[0]["valid"] and capture.cpu[0]["generation"] == 1
    assert capture.status()["calculations"] == 1


def test_single_backend_runs_record_hashed_inputs_into_the_capture():
    w = window()
    capture = LatencyCapture(60, 1, start_ns=w.recv_ns - 1, backend="cpu")
    store, _ = make_store(backend="cpu", capture=capture)
    update = RiskUpdate(w, cpu_result(w), None, "cpu", False)
    store.on_risk(update)
    (single,) = capture.single
    assert single["backend"] == "cpu" and len(single["input_sha256"]) == 64
    assert single["receive_to_result_ms"] == store.live["receive_to_result_ms"]
    assert "covariance_probes" in single["validation"]  # recorded from the unfiltered result
