# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``DrainedCondition``: each stage must catch up before a finite run may stop, and ``why()``
names the first that has not."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from holoscan_finance.app.drained import DrainedCondition
from holoscan_finance.control.gate import CrossCheckGate
from holoscan_finance.telemetry.stats import BarsStats, IngestStats, RiskStats


def drained_run() -> SimpleNamespace:
    """Stats of a replay that has decoded, barred, analyzed and compared everything:
    10 event batches, 3 windows (the last of generation 7), 2 analyses, 1 comparison."""
    ingest = IngestStats()
    ingest.eof = True
    ingest.event_batches = 10
    bars = BarsStats(batches_in=10, windows_emitted=3, last_window_key=(7, 24))
    risk = RiskStats(analyzed=2, last_window_key=(7, 24))
    gate = CrossCheckGate()
    assert gate.try_claim()
    gate.release()
    results = SimpleNamespace(received_risk=2, received_comparisons=1)  # a ResultsStore's counters
    return SimpleNamespace(ingest=ingest, bars=bars, risk=risk, gate=gate, results=results)


def condition(run: SimpleNamespace, **kw: object) -> DrainedCondition:
    return DrainedCondition(
        ingest=run.ingest,
        bars=[run.bars],
        risk=run.risk,
        results=run.results,
        gate=run.gate,
        **kw,
    )


def test_a_caught_up_run_is_drained() -> None:
    drained = condition(drained_run())
    assert drained() and drained.why() == ""


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        (lambda r: setattr(r.ingest, "eof", False), "ingest not drained: eof=False"),
        (lambda r: setattr(r.ingest, "held", 1), "1 batches held"),
        (lambda r: setattr(r.ingest, "packet_batches", 4), "0 of 4 packet batches decoded"),
        (lambda r: setattr(r.bars, "batches_in", 9), "bars[0] took 9 of 10 event batches"),
        (lambda r: setattr(r.risk, "busy", True), "risk is analyzing a window"),
        (lambda r: setattr(r.risk, "last_window_key", (7, 23)), "risk analyzed window (7, 23)"),
        (lambda r: setattr(r.gate, "in_flight", True), "a cross-check is in flight"),
        (lambda r: setattr(r.results, "received_risk", 1), "results received 1 of 2 risk updates"),
        (lambda r: setattr(r.results, "received_comparisons", 0), "received 0 of 1 comparisons"),
    ],
)
def test_each_lagging_stage_keeps_the_run_going(change, reason: str) -> None:
    run = drained_run()
    change(run)
    drained = condition(run)
    assert not drained()
    assert reason in drained.why()


def test_pending_packets_in_the_receive_ring_are_not_drained() -> None:
    run = drained_run()
    with run.ingest.lock:
        run.ingest.ring_packets = 3
    assert "3 packets pending" in condition(run).why()


def test_risk_is_matched_by_generation_not_by_bar_count() -> None:
    """A looped replay closes the same bars every pass: the previous pass's last window has the
    same completed_bars, so only the generation tells that risk has not seen this pass's."""
    run = drained_run()
    run.risk.last_window_key = (6, 24)
    drained = condition(run)
    assert not drained()
    assert drained.why() == "risk analyzed window (6, 24), bars emitted (7, 24)"


def test_without_windows_risk_has_nothing_to_wait_for() -> None:
    run = drained_run()
    run.bars.windows_emitted, run.bars.last_window_key = 0, None
    run.risk.analyzed = run.results.received_risk = 0
    run.risk.last_window_key = None
    assert condition(run)()


def test_collectors_must_have_every_window() -> None:
    run = drained_run()
    received = [3]
    drained = condition(run, collectors=[(lambda: received[0], run.bars)])
    assert drained()
    received[0] = 2
    assert drained.why() == "collector 0 received 2 of 3 windows"


def test_every_bars_stage_must_take_every_batch() -> None:
    run = drained_run()
    reference = BarsStats(batches_in=8)
    drained = DrainedCondition(ingest=run.ingest, bars=[run.bars, reference])
    assert drained.why() == "bars[1] took 8 of 10 event batches"
    reference.batches_in = 10
    assert drained()


def test_without_a_gate_comparisons_are_not_counted() -> None:
    """GPU-only and CPU-only runs pass no gate: nothing is ever cross-checked."""
    run = drained_run()
    run.results.received_comparisons = 0
    drained = DrainedCondition(
        ingest=run.ingest, bars=[run.bars], risk=run.risk, results=run.results
    )
    assert drained()


def test_results_without_risk_is_a_usage_error() -> None:
    run = drained_run()
    with pytest.raises(ValueError, match="pass risk too"):
        DrainedCondition(ingest=run.ingest, bars=[run.bars], results=run.results)
