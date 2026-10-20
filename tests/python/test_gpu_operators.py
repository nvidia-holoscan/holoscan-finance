# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GpuQuoteBarsOp and GpuRollingRiskOp: the same results as the classes they wrap, and
ports that let slow analytics skip windows without ever blocking the bars.

The port test runs a real Holoscan graph and needs only the SDK; the others need a GPU.
"""

import time

import numpy as np
import pytest


def _gpu():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("CUDA GPU required")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA unavailable")
    return cp


SETTINGS = dict(
    bar_seconds=5, window_seconds=60, stale_seconds=30, max_spread_bps=100.0, min_bars=4
)


def _gpu_batches(cp, generation=1):
    """The synthetic MITCH stream of test_equities_gpu, decoded on the GPU in 97-packet batches."""
    from holoscan.equities import gpu_validate as gv
    from holoscan.equities import mitch as replay
    from holoscan.marketdata.gpu_bars import GpuReplayBatch
    from holoscan.marketdata.gpu_mitch import decode_packets

    midnight = replay.day_start_ns("2025-04-01")
    payloads = gv.synthetic_stream(np.random.default_rng(99), 900, 50, midnight, 34200)
    del payloads[400:403]  # a sequence gap
    batches = []
    for lo in range(0, len(payloads), 97):
        part = payloads[lo : lo + 97]
        recv = np.arange(lo, lo + len(part), dtype=np.int64) + 1
        slots, lens = gv.to_slots(part)
        dec = decode_packets(slots, lens, midnight_ns=midnight, universe=50)
        batches.append(
            GpuReplayBatch(
                dec.columns,
                generation,
                dec.sequence,
                dec.wire_count,
                dec.offsets,
                recv,
                int(recv[-1]),
            )
        )
    return batches


def test_quote_bars_op_windows_match_gpu_quote_bars(fragment, op_input_factory, op_output_factory):
    cp = _gpu()
    from holoscan.marketdata import GpuQuoteBarsOp
    from holoscan.marketdata.gpu_bars import GpuQuoteBars, apply_sequence

    symbols = [f"S{i}" for i in range(50)]
    op = GpuQuoteBarsOp(fragment, **SETTINGS, name="bars_test")
    op.reset(symbols, generation=1)
    reference, expected, gaps = GpuQuoteBars(symbols, **SETTINGS), 1, 0
    batches = _gpu_batches(cp)
    stale = _gpu_batches(cp, generation=7)[0]
    emitted = 0
    for batch in batches:
        out = op_output_factory()
        op.compute(op_input_factory(events=[[stale, batch]]), out, None)
        expected, _, g = apply_sequence(reference, batch, expected)
        gaps += g
        window = reference.window()
        if out.emissions:
            emitted += 1
            ((w, port),) = out.emissions
            assert port == "window" and w.generation == 1
            assert w.completed_bars == reference.completed_bars
            assert w.recv_ns == reference.bar_recv_ns
            np.testing.assert_array_equal(w.times, window[1])
            np.testing.assert_array_equal(
                w.rows.get().view(np.int64), window[0].get().view(np.int64)
            )
        else:
            assert window is None or reference.completed_bars == 0
    assert emitted >= 5 and gaps > 0
    assert op.sequence_gaps == gaps and op.expected_sequence == expected
    assert op.stale_packets == stale.n_packets * len(batches)
    assert op.first_batch_ns <= op.last_batch_ns
    op.reset(symbols, generation=2)
    assert op.bars.completed_bars == 0 and op.first_batch_ns is None


def test_rolling_risk_op_matches_the_engine(fragment, op_input_factory, op_output_factory):
    cp = _gpu()
    from holoscan.marketdata import BarWindow, GpuRollingRiskOp
    from holoscan.marketdata.covariance import CovarianceEngine

    rng = np.random.default_rng(3)
    rows = rng.normal(0, 0.001, (61, 40))
    rows[5:9, 2] = np.nan
    rows[:, 7] = 0
    windows = [
        BarWindow([cp.asarray(r) for r in rows[i : i + 60]], np.arange(i, i + 60), 1, 60 + i, 0)
        for i in (0, 1)
    ]
    ready = {"ok": False}
    op = GpuRollingRiskOp(fragment, ready=lambda: ready["ok"], host_variance=True, name="risk_test")
    held = op_input_factory(window=[windows[0]])
    op.compute(held, op_output_factory(), None)
    assert held.received == 0  # not ready: the window stays queued
    ready["ok"] = True
    engine = CovarianceEngine("gpu")
    for w in windows:
        out = op_output_factory()
        op.compute(op_input_factory(window=[w]), out, None)
        ((update, port),) = out.emissions
        host = rows[w.times[0] : w.times[-1] + 1]
        want = engine.update(host, w.times, observed_variance=np.var(host, axis=0, ddof=1))
        assert port == "risk" and update.window is w
        np.testing.assert_array_equal(update.rows_host, host)
        for key in ("method", "added_bars", "expired_bars", "eligible_indices", "heatmap_indices"):
            assert update.result[key] == want[key], key
        np.testing.assert_allclose(update.result["annual_vol"], want["annual_vol"], rtol=1e-10)
        assert update.result["gpu_name"] and update.result["finished_wall_ns"] > 0
    assert update.result["method"] == "incremental"  # second window: one bar added, one expired


def test_slow_risk_skips_windows_without_blocking_bars():
    """Bars emit faster than risk consumes: the emitter must never wait, risk must see
    the newest window (also the one left queued when the stream ends), and nothing faults."""
    pytest.importorskip("holoscan.schedulers")
    from holoscan.conditions import CountCondition
    from holoscan.core import Application, Operator
    from holoscan.marketdata import BarWindow, GpuQuoteBarsOp, GpuRollingRiskOp
    from holoscan.schedulers import MultiThreadScheduler

    n, seen, emitted = 40, [], []

    class Feed(Operator):
        def setup(self, spec):
            spec.output("out")

        def compute(self, op_input, op_output, context):
            op_output.emit(len(emitted), "out")

    class Bars(GpuQuoteBarsOp):  # real ports, stand-in compute
        def compute(self, op_input, op_output, context):
            i = op_input.receive("events")
            emitted.append(time.monotonic())
            op_output.emit(BarWindow([], np.arange(2), 1, i, 0), "window")

    class Risk(GpuRollingRiskOp):  # real ports and rate limit, slow stand-in compute
        def compute(self, op_input, op_output, context):
            w = op_input.receive("window")
            if w is not None:
                seen.append(w.completed_bars)
                time.sleep(0.03)

    class Sink(Operator):
        def setup(self, spec):
            spec.input("risk")

        def compute(self, op_input, op_output, context):
            op_input.receive("risk")

    class App(Application):
        def compose(self):
            feed = Feed(self, CountCondition(self, n), name="feed")
            bars, risk = Bars(self, name="bars"), Risk(self, min_interval_s=0.01, name="risk")
            self.add_flow(feed, bars, {("out", "events")})
            self.add_flow(bars, risk, {("window", "window")})
            self.add_flow(risk, Sink(self, name="sink"), {("risk", "risk")})
            self.scheduler(
                MultiThreadScheduler(
                    self, worker_thread_number=3, stop_on_deadlock=True,
                    stop_on_deadlock_timeout=500, name="scheduler",
                )
            )  # fmt: skip

    App().run()
    assert len(emitted) == n
    assert emitted[-1] - emitted[0] < 0.5 * n * 0.03  # never paced by the 30 ms consumer
    assert seen[-1] == n - 1  # the newest window, though it arrived after the last emit
    assert len(seen) < n and seen == sorted(seen)
