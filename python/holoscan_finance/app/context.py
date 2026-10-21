# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Everything one run shares outside the graph: control plane, telemetry, results, status.

``EquityContext.from_args`` builds the objects that 4.3's ``EquityState`` held in one
class. Operators reach them only as ``Ref`` Params, the dashboard reads them, and none of
them is a port. The directory listeners keep 4.3's order (results, capture, full day), and
the cross-check gate pauses while a full-day burst runs. ``register_replay`` publishes the
first replay generation as 4.3 ``replay.py`` did over ``POST /manifest``, and hands back the
hook that registers a new generation on a new sender port for every further pass.
``RunContext`` is what the runner and the dashboard server need from any app's context.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from ..control.directory import ReplayDirectory, ReplayGenerations
from ..control.feed import FeedControl
from ..control.fullday import FullDayController
from ..control.gate import CrossCheckGate
from ..control.results import ResultsStore
from ..control.status import RunStatus
from ..ingest.affinity import CpuSets
from ..marketdata.framing import PcapFrames
from ..nyq.manifest import MANIFEST_FORMAT, FeedDirectory
from ..nyq.mapping import PROTOCOL, TIMEZONE
from ..nyq.tape import open_tape
from ..ref import Knob
from ..settings import BarSettings
from ..telemetry.capture import CaptureTap, percentiles
from ..telemetry.stalls import StallMonitor
from ..telemetry.stats import BarsStats, IngestStats, RiskStats
from .cli import bar_settings
from .drained import DrainedCondition
from .runner import report_diagnostics

if TYPE_CHECKING:
    import holoscan as hs


class RunContext(Protocol):
    """What ``runner.run_headless`` and ``dashboard.serve`` need from an app's context."""

    status: RunStatus
    drained: DrainedCondition
    stop_reason: str | None  # set by the runner: "drained" | "signal" | "timeout" | "terminated"
    stall_monitor: StallMonitor | None  # None: no heartbeat thread
    plan_summary: dict | None

    def report_dict(self, session: hs.RunSession) -> dict: ...


def register_replay(
    directory: ReplayDirectory, *, tape: str, pcap: str, sender_port: int | None = None
) -> ReplayGenerations:
    """Publish the tape's manifest as the first generation, sent from ``sender_port`` (default:
    the capture's most common UDP source port), and return the hook for later passes."""
    manifest, _ = open_tape(tape)
    base = {
        **manifest,
        "format": MANIFEST_FORMAT,
        "wire_protocol": PROTOCOL,
        "market_timezone": TIMEZONE,
    }
    port = sender_port or PcapFrames.read(pcap, limit=256).dominant_src_port()
    if port is None:
        raise ValueError(f"{pcap}: no UDP frame to take the sender port from")
    directory.set_manifest({**base, "sender_port": port, "generation": time.time_ns()})
    return ReplayGenerations(directory, base, port)


@dataclass(eq=False)
class EquityContext:
    """The objects of one ``equity_analytics`` run (implements ``RunContext``)."""

    args: argparse.Namespace
    cpus: CpuSets
    settings: BarSettings
    directory_knob: Knob[FeedDirectory]
    directory: ReplayDirectory
    feed: FeedControl
    ingest_stats: IngestStats
    bars_stats: BarsStats
    risk_stats: RiskStats
    tap: CaptureTap
    gate: CrossCheckGate
    results: ResultsStore
    stall_monitor: StallMonitor
    fullday: FullDayController
    status: RunStatus
    replay_generations: ReplayGenerations | None
    drained: DrainedCondition
    stop_reason: str | None = None
    plan_summary: dict | None = None

    @classmethod
    def from_args(cls, args: argparse.Namespace, cpus: CpuSets) -> EquityContext:
        """Build every object; with ``--ingest replay`` (and not ``--plan-only``) also publish
        the first generation, after the listeners are in place."""
        settings = bar_settings(args)
        knob: Knob[FeedDirectory] = Knob()
        directory = ReplayDirectory(knob)
        feed = FeedControl(directory)
        ingest, bars, risk = IngestStats(), BarsStats(), RiskStats()
        tap = CaptureTap()
        fullday: FullDayController | None = None
        # The gate asks the full-day controller, which needs the results store, which
        # releases the gate: the lambda reads `fullday` once it exists.
        gate = CrossCheckGate(allowed=lambda: fullday.validation_allowed())
        results = ResultsStore(
            backend=args.analytics_backend,
            interval_s=args.interval,
            settings=settings,
            directory=directory,
            tap=tap,
            gate=gate,
        )
        stall_monitor = StallMonitor(ingest)
        fullday = FullDayController(
            replay_control=args.replay_control,
            feed=feed,
            ingest=ingest,
            bars=bars,
            results=results,
            stalls=stall_monitor,
            backend=args.analytics_backend,
        )
        for listener in (results.reset_generation, tap.on_generation, fullday.on_generation):
            directory.add_listener(listener)
        drained = DrainedCondition(
            ingest=ingest,
            bars=[bars],
            risk=risk,
            results=results,
            gate=gate if args.analytics_backend == "compare" else None,
        )
        replay_generations = None
        if args.ingest == "replay" and not args.plan_only:
            replay_generations = register_replay(
                directory, tape=args.tape, pcap=args.pcap, sender_port=args.replay_sender_port
            )
        return cls(
            args=args,
            cpus=cpus,
            settings=settings,
            directory_knob=knob,
            directory=directory,
            feed=feed,
            ingest_stats=ingest,
            bars_stats=bars,
            risk_stats=risk,
            tap=tap,
            gate=gate,
            results=results,
            stall_monitor=stall_monitor,
            fullday=fullday,
            status=RunStatus(),
            replay_generations=replay_generations,
            drained=drained,
        )

    def expectations(self) -> list[str]:
        """Unmet ``--expect-windows`` checks (in compare mode: a comparison was published and
        none disagreed) and any error the run recorded without failing."""
        unmet = [
            f"{where} error: {error}"
            for where, error in (
                ("ingest", self.ingest_stats.error),
                ("risk", self.risk_stats.error),
                ("results", self.results.error),
            )
            if error
        ]
        expected = self.args.expect_windows
        if expected:
            analyzed = self.risk_stats.analyzed
            if analyzed < expected:
                unmet.append(f"{analyzed} windows analyzed, expected at least {expected}")
            if self.args.analytics_backend == "compare":
                if self.results.comparisons < 1:
                    unmet.append("no CPU/GPU comparison was published")
                if self.results.comparison_failure is not None:
                    unmet.append("a CPU/GPU comparison disagreed")
        return unmet

    def report_dict(self, session: hs.RunSession) -> dict:
        """The run report: termination, counters per stage, latency percentiles, placement."""
        ingest, bars, risk = self.ingest_stats, self.bars_stats, self.risk_stats
        results, compare = self.results, self.args.analytics_backend == "compare"
        counters = bars.counters
        seconds = None
        if counters is not None and counters.first_batch_ns and counters.last_batch_ns:
            seconds = (counters.last_batch_ns - counters.first_batch_ns) / 1e9
        with results.lock:
            latencies = list(results.latencies_ms)
            rolling_timing = results.rolling_timing()
        return {
            "termination": session.termination.name,
            "stop_reason": self.stop_reason,
            "errors": report_diagnostics(session),
            "unmet_expectations": self.expectations(),
            "config": dict(vars(self.args)),
            "generation": self.directory.generation,
            "ingest": {
                "front_end": ingest.front_end,
                "packets": ingest.packets,
                "received_messages": ingest.received_messages,
                "messages": ingest.messages,
                "sequence_gaps": ingest.dropped,
                "ring_dropped": ingest.ring_dropped,
                "stale_packets": ingest.stale_packets,
                "malformed_packets": ingest.malformed_packets,
                "decode_malformed": ingest.decode_malformed,
                "event_batches": ingest.event_batches,
                "retries": ingest.retries,
                "replay": ingest.replay,
                "lane_cpus": ingest.lane_cpus,
            },
            "bars": {
                "observations": bars.observations(),
                "seconds": seconds,
                "batches_in": bars.batches_in,
                "windows_emitted": bars.windows_emitted,
                "sequence_gaps": bars.sequence_gaps,
                "stale_packets": bars.stale_packets,
                "lane_cpus": bars.lane_cpus,
            },
            "risk": {
                "analyzed": risk.analyzed,
                "windows_received": risk.windows_received,
                "windows_superseded": risk.windows_superseded,
                "gate_waits": risk.gate_waits,
                "late_activations": risk.late_activations,
                "max_ms": risk.max_ms,
                "crosschecks_done": risk.crosschecks_done,
                "lane_cpus": risk.lane_cpus,
            },
            "results": {
                "published": results.published,
                "comparisons": results.comparisons,
                "agreement": results.comparison_failure is None if compare else None,
                "receive_to_result_ms": percentiles(latencies),
                "rolling_timing": rolling_timing,
            },
            "plan": self.plan_summary,
        }
