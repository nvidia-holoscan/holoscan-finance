# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""When a finite run has nothing left in flight, judged from the stats objects alone.

The runner polls the predicate every 50 ms and then calls ``session.request_stop()``;
nothing is queued by then, so the stop loses nothing. Operators never end a run themselves:
``complete_after`` counts returned calls (C9) and ``request_completion`` drops data queued
between lanes (v07b). Risk is matched to bars by the ``(generation, completed_bars)`` key
of the last window, never by the bar count alone, because every pass of a looped replay
closes the same number of bars. Only GIL-atomic reads of host objects: safe from any thread.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..control.gate import CrossCheckGate
    from ..control.results import ResultsStore
    from ..telemetry.stats import BarsStats, IngestStats, RiskStats


class DrainedCondition:
    """True once input has ended and every stage downstream has caught up.

    ``bars`` are the bars stats fed by the source (the first one feeds risk);
    ``collectors`` pair a window counter (a ``BarsChecker`` side, a test sink) with the bars
    stats whose windows it receives.
    """

    def __init__(
        self,
        *,
        ingest: IngestStats,
        bars: Sequence[BarsStats],
        collectors: Sequence[tuple[Callable[[], int], BarsStats]] = (),
        risk: RiskStats | None = None,
        results: ResultsStore | None = None,
        gate: CrossCheckGate | None = None,
    ) -> None:
        if results is not None and risk is None:
            raise ValueError("results are matched against risk.analyzed: pass risk too")
        self.ingest = ingest
        self.bars = list(bars)
        self.collectors = list(collectors)
        self.risk, self.results, self.gate = risk, results, gate

    def __call__(self) -> bool:
        return not self.why()

    def why(self) -> str:
        """The first unmet condition, in pipeline order; empty once drained."""
        ingest = self.ingest
        if not ingest.drained():
            return (
                f"ingest not drained: eof={ingest.eof}, {ingest.pending_packets()} packets "
                f"pending, {ingest.held} batches held, {ingest.packet_batches_decoded} of "
                f"{ingest.packet_batches} packet batches decoded"
            )
        for i, bars in enumerate(self.bars):
            if bars.batches_in != ingest.event_batches:
                return f"bars[{i}] took {bars.batches_in} of {ingest.event_batches} event batches"
        for i, (count, bars) in enumerate(self.collectors):
            if (received := count()) != bars.windows_emitted:
                return f"collector {i} received {received} of {bars.windows_emitted} windows"
        risk = self.risk
        if risk is not None:
            if risk.busy:
                return "risk is analyzing a window"
            last = self.bars[0].last_window_key
            if self.bars[0].windows_emitted and risk.last_window_key != last:
                return f"risk analyzed window {risk.last_window_key}, bars emitted {last}"
        gate = self.gate
        if gate is not None and gate.in_flight:
            return "a cross-check is in flight"
        results = self.results
        if results is not None:
            if results.received_risk != risk.analyzed:
                return f"results received {results.received_risk} of {risk.analyzed} risk updates"
            if gate is not None and results.received_comparisons != gate.claims:
                return (
                    f"results received {results.received_comparisons} of {gate.claims} comparisons"
                )
        return ""
