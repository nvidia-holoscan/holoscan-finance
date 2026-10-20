# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent statistics, market-data semantics and replay isolation checks."""

# ruff: noqa: E402

import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from holoscan.equities.analytics import agreement, cpu_window, gpu_window
from holoscan.equities.stream import EquityState
from holoscan.marketdata.bars import QuoteBars
from holoscan.marketdata.covariance import CovarianceEngine
from holoscan.marketdata.wire import (
    CONDITION,
    EVENT_DTYPE,
    HALTED,
    QUOTE,
    TRADE,
    ReplayBatch,
    combine_batches,
)


def event(ts, mid=100.0, kind=QUOTE, flags=0, symbol=0, size=10):
    row = np.zeros(1, dtype=EVENT_DTYPE)
    row["timestamp_ns"] = int(ts * 1e9)
    row["symbol_id"] = symbol
    row["kind"], row["flags"] = kind, flags
    row["bid"], row["ask"] = mid - 0.01, mid + 0.01
    row["bid_size"] = row["ask_size"] = size
    row["price"], row["volume"] = mid, 2
    return row


def _require_gpu():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("CUDA GPU required")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA unavailable")


def batch(rows, sequence=1, gen=1):
    return ReplayBatch(np.concatenate(rows), gen, 200, sequence)


def manifest(gen=1):
    return dict(
        format="nyq-equity-replay-v1",
        wire_protocol="mit303-11.9-nyq-replay-v1",
        market_timezone="America/New_York",
        sender_port=50000 + gen,
        symbols=["LONG_INSTRUMENT_NAME"],
        generation=gen,
        date="2025-04-01",
        start_ns=1,
        end_ns=10**14,
        events=100,
        profile=[],
    )


def test_quote_returns_exclude_trades_and_stale_marks():
    bars = QuoteBars(["A"], window_seconds=20, min_bars=2, stale_seconds=5)
    bars.observe(
        batch(
            [
                event(1, 100),
                event(6, 110),
                event(7, 999, TRADE),
                event(11, 121),
                event(16, kind=CONDITION),
                event(21, kind=CONDITION),
            ]
        )
    )
    np.testing.assert_allclose(np.array(bars.rows)[1:3, 0], np.log(1.1))
    assert np.isnan(bars.rows[-1][0])
    assert bars.trade[0] == 999 and bars.volume[0] == 2


def test_halts_invalid_quotes_and_packet_boundaries_do_not_change_bars():
    rows = [
        event(1),
        event(2, kind=CONDITION, flags=HALTED),
        event(3, 101),
        event(6, 105),
        event(7, kind=CONDITION, flags=HALTED),
        event(11, 110),
        event(12, 111, size=0),
        event(16, 120),
        event(21, 123),
        event(26, kind=CONDITION),
    ]
    one = QuoteBars(["A"], window_seconds=30, min_bars=2)
    split = QuoteBars(["A"], window_seconds=30, min_bars=2)
    one.observe(batch(rows))
    packets = []
    for i, row in enumerate(rows):
        b = ReplayBatch(row, 1, i * 100 + 10, i + 1)
        packets.append(b)
        split.observe(b)
    merged = QuoteBars(["A"], window_seconds=30, min_bars=2)
    merged.observe(combine_batches(packets))
    np.testing.assert_equal(np.asarray(one.rows), np.asarray(split.rows))
    np.testing.assert_equal(np.asarray(one.rows), np.asarray(merged.rows))
    assert merged.bar_recv_ns == split.bar_recv_ns == 910
    assert one.marks[0][0] == 101
    assert np.isnan(one.marks[1][0]) and np.isnan(one.marks[2][0])
    assert one.invalid_quotes == 1


def test_seek_and_sequence_loss_invalidate_unknown_marks():
    state = EquityState(window_seconds=20, min_bars=2)
    state.set_manifest(manifest())
    state.observe([batch([event(1), event(6, 110)])])
    state.observe([batch([event(11, kind=CONDITION)], sequence=4)])
    assert state.sequence_gaps == 1
    assert np.isnan(state.bars.mid).all()
    state.set_manifest(manifest(2))
    state.observe([batch([event(16, 120)], sequence=5)])
    assert state.bars.events == 0 and state.stale_packets == 1
    assert state.bars.window() is None
    assert state.expected_sequence == 1
    for payload in ({"paused": "false"}, {"speed": float("nan")}, {"seek": 86400}):
        with pytest.raises(ValueError):
            state.feed_control(payload)


def test_cpu_validation_pauses_during_a_full_day_burst():
    state = EquityState(window_seconds=20, min_bars=2, analytics_backend="compare")
    state.set_manifest(manifest())
    state.observe([batch([event(t, 100 + t) for t in (1, 6, 11, 16)])])
    assert state.take_work()["validate"]
    state.fullday = dict(status="running")
    state.observe([batch([event(t, 100 + t) for t in (21, 26)], sequence=state.expected_sequence)])
    state.last_started = 0.0
    assert not state.take_work()["validate"]


def test_gpu_name_is_queried_once():
    from holoscan.marketdata import covariance

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


def test_missing_oldest_bar_does_not_shorten_an_established_window():
    bars = QuoteBars(["A"], window_seconds=20, min_bars=2)
    bars.observe(
        batch(
            [
                event(1),
                event(6, size=0),
                event(11, 102),
                event(16, 103),
                event(21, 104),
                event(26, 105),
            ]
        )
    )
    rows, times = bars.window()
    assert len(rows) == 4
    assert times.tolist() == [10_000_000_000, 15_000_000_000, 20_000_000_000, 25_000_000_000]
    assert np.isnan(rows[0, 0])
    assert CovarianceEngine().update(rows, times)["eligible"] == 0


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_incremental_covariance_matches_independent_numpy_oracle(backend):
    if backend == "gpu":
        _require_gpu()
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


def test_persistent_workers_match_after_roll_and_seek():
    _require_gpu()
    rows = np.random.default_rng(42).normal(0, 0.001, (30, 12))
    rows[2:4, 3] = np.nan
    ctx = multiprocessing.get_context("spawn")
    with (
        ProcessPoolExecutor(1, mp_context=ctx) as cpu,
        ProcessPoolExecutor(1, mp_context=ctx) as gpu,
    ):
        for generation, end, method in (
            (1, 10, "rebuild"),
            (1, 13, "incremental"),
            (2, 10, "rebuild"),
        ):
            sample, times = rows[end - 10 : end], np.arange(end - 10, end)
            cf = cpu.submit(cpu_window, sample, times, generation, 5, 1)
            gf = gpu.submit(gpu_window, sample, times, generation, 5)
            cr, gr = cf.result(timeout=30), gf.result(timeout=30)
            assert cr["method"] == gr["method"] == method
            assert cr["added_bars"] == gr["added_bars"]
            assert cr["expired_bars"] == gr["expired_bars"]
            assert agreement(cr, gr)
            oracle = np.cov(sample[:, cr["eligible_indices"]], rowvar=False)
            np.testing.assert_allclose(gr["covariance_probes"], oracle, rtol=1e-10, atol=1e-16)


def test_agreement_requires_the_same_displayed_instruments():
    rows = np.random.default_rng(61).normal(size=(10, 5))
    result = CovarianceEngine().update(rows, np.arange(10))
    wrong_labels = {**result, "heatmap_indices": list(reversed(result["heatmap_indices"]))}
    assert not agreement(result, wrong_labels)


def test_equal_volatility_heatmap_selection_is_backend_independent():
    _require_gpu()
    rng = np.random.default_rng(912)
    rows = rng.normal(0, 0.01, (80, 40))
    rows[20:] = np.tile(rng.normal(0, 0.001, (60, 1)), (1, 40))
    cpu, gpu = CovarianceEngine(), CovarianceEngine("gpu")
    for end in range(60, 81):
        a = cpu.update(rows[end - 60 : end], np.arange(end - 60, end))
        b = gpu.update(rows[end - 60 : end], np.arange(end - 60, end))
        assert agreement(a, b)
    assert a["heatmap_indices"] == list(range(24))


def test_preparation_selects_session_sorts_and_retains_quote_fields(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    pytest.importorskip("duckdb")
    from holoscan.equities.data import open_tape, prepare

    def timestamp(hour, minute):
        return int(datetime(2025, 4, 1, hour, minute, tzinfo=timezone.utc).timestamp() * 1e9)

    source = tmp_path / "part.parquet"
    pq.write_table(
        pa.table(
            dict(
                messageTimestamp=[timestamp(14, 0), timestamp(13, 0), timestamp(13, 30)],
                ricName=["VERY_LONG_RIC"] * 3,
                internalSequence=[2, 1, 1],
                type=["Quote", "Trade", "Mkt. Condition"],
                Qualifiers=["", "", "1[HALT_REASN]"],
                Bid_Price=[812345.12, None, None],
                Ask_Price=[812345.14, None, None],
                Bid_Size=[10.0, None, None],
                Ask_Size=[20.0, None, None],
                Price=[None, 4.0, None],
                Volume=[None, 100.0, None],
            )
        ),
        source,
    )
    prepare(source, tmp_path / "out")
    m, tape = open_tape(tmp_path / "out")
    assert m["raw_rows"] == 3 and m["events"] == 2
    assert m["start_ns"] == timestamp(13, 30)
    assert m["symbols"] == ["VERY_LONG_RIC"]
    assert tape["kind"].tolist() == [CONDITION, QUOTE]
    assert tape["flags"][0] == HALTED
    assert tape["bid"][1] == 812345.12 and tape["ask_size"][1] == 20
    assert np.isnan(tape["price"][1])
    assert sum(n for _, n in m["profile"]) == 2
    assert json.loads((tmp_path / "out/manifest.json").read_text())["date"] == "2025-04-01"
