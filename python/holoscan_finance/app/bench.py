# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``market_data_bench`` plumbing: its flags, its run context, and its report and verdict.

The composition is the application file; everything else it needs is here: the feed
directory with the capture's generation registered, the stats objects the drained predicate
reads, the window checker of ``--check``, and after the run the report, the host reference
check and the gates. It lives with the bench, so the dashboard plumbing never depends on it.
Every number comes from stats objects the lanes wrote; nothing here touches the device
except the post-run comparison of the GPU windows. Holoscan is imported only by the
``runner`` functions the application calls.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ..control.directory import ReplayDirectory, ReplayGenerations
from ..control.results import ResultsStore
from ..control.status import RunStatus
from ..ingest.affinity import CpuSets, parse_cpu_list
from ..marketdata.framing import PcapFrames
from ..ref import Knob
from ..settings import BarSettings, IngestDefaults
from ..telemetry.capture import CaptureTap, percentiles
from ..telemetry.stats import BarsStats, IngestStats, RiskStats
from ..validation.checkers import BarsChecker, Gates, bench_verdict, final_state_equal
from ..validation.compare import HostReference
from . import cli
from .context import register_replay
from .drained import DrainedCondition
from .runner import ExitCode, usage_error

if TYPE_CHECKING:
    from ..equity.manifest import FeedDirectory
    from ..telemetry.stalls import StallMonitor

logger = logging.getLogger(__name__)


def bench_parser() -> argparse.ArgumentParser:
    """The bench flags; the bar, placement and plan groups are shared with equity_analytics."""
    d = IngestDefaults()
    p = argparse.ArgumentParser(
        description="Replay an SBE capture from GPU memory through the production GPU chain "
        "(gather, decode, quote bars, optionally risk); measure throughput, loss and lag, and "
        "with --check verify every bar against the CPU backend and a host decode.",
    )
    g = p.add_argument_group("data and pacing")
    g.add_argument("--pcap", default="", help="capture to replay (Ethernet II / IPv4 / UDP)")
    g.add_argument("--tape", default="", help="prepared tape of the capture's day (manifest)")
    g.add_argument("--pps", type=int, default=200000, help="replay rate, packets/s (0: unpaced)")
    g.add_argument(
        "--loops", type=int, default=1, help="passes over the capture, each a new generation"
    )
    g.add_argument(
        "--burst-packets", type=int, default=d.burst_packets, help="packets per emulated NIC burst"
    )
    g.add_argument(
        "--rx-timeout-us", type=int, default=d.rx_timeout_us, help="partial-burst timeout"
    )
    g.add_argument(
        "--nic-bufs",
        type=int,
        default=0,
        help="NIC buffer model: drop due packets beyond this backlog (0: hold, never drop)",
    )
    g = p.add_argument_group("sizes")
    g.add_argument("--batch-messages", type=int, default=d.gpu_batch_messages)
    g.add_argument("--ring-size", type=int, default=d.gpu_ring_size)
    g.add_argument("--gpu-device", type=int, default=0)
    cli.add_bar_flags(p)
    g = p.add_argument_group("verification and load")
    g.add_argument(
        "--check",
        action="store_true",
        help="also build CPU bars from the same decoded stream and compare every window, then "
        "compare every closed bar with a host decode of the capture",
    )
    g.add_argument("--risk", choices=("none", "gpu"), default="none", help="add the GPU risk lane")
    g.add_argument(
        "--interval", type=float, default=d.interval_s, help="minimum seconds between risk runs"
    )
    g = p.add_argument_group("output")
    g.add_argument("--report", metavar="PATH", help="also write the report JSON here")
    g.add_argument(
        "--timeout-s", type=float, default=1800, help="fail if not drained by then (0: never)"
    )
    g = p.add_argument_group("gates")
    g.add_argument(
        "--require-lossless", action="store_true", help="no gap, drop, stale or malformed packet"
    )
    g.add_argument(
        "--require-realtime",
        action="store_true",
        help="paced runs: the graph never held a burst (no retries, no ring waits)",
    )
    g.add_argument("--min-msgs-per-s", type=float, default=0.0, help="throughput floor (0: off)")
    g.add_argument("--max-lag-p99-ms", type=float, default=0.0, help="replay lag p99 cap (0: off)")
    cli.add_placement_flags(p)
    cli.add_plan_flags(p)
    return p


def normalize_bench(args: argparse.Namespace) -> argparse.Namespace:
    """Cross-flag checks (exit 2 when rejected) and ``args.gates``."""
    for flag in ("pcap", "tape"):
        if getattr(args, flag):
            continue
        if not args.plan_only:
            usage_error(f"--{flag} is required")
        setattr(args, flag, "none")  # the source's setup() needs a path; start() never runs
    if args.loops < 1:
        usage_error("--loops must be at least 1: the bench is a finite run")
    if args.check and args.loops != 1:
        usage_error("--check needs --loops 1: the reference decodes the capture once")
    if args.require_realtime and (args.check or args.pps == 0):
        usage_error("--require-realtime needs a paced run (--pps > 0) without --check")
    if not 0.001 <= args.interval <= 60:
        usage_error("--interval must be within 0.001-60 seconds")
    try:
        cli.bar_settings(args)
        if args.ingest_cpus:
            parse_cpu_list(args.ingest_cpus)
    except ValueError as exc:
        usage_error(str(exc))
    args.gates = Gates(
        require_lossless=args.require_lossless,
        require_realtime=args.require_realtime,
        check=args.check,
        min_msgs_per_s=args.min_msgs_per_s,
        max_lag_p99_ms=args.max_lag_p99_ms,
    )
    return args


@dataclass(eq=False)
class BenchContext:
    """The objects ``market_data_bench`` shares with its operators and reads after the run;
    implements ``app.context.RunContext``."""

    args: argparse.Namespace
    cpus: CpuSets
    settings: BarSettings
    directory_knob: Knob[FeedDirectory]
    directory: ReplayDirectory
    replay_generations: ReplayGenerations | None
    ingest_stats: IngestStats
    bars_stats: BarsStats
    ref_bars_stats: BarsStats
    risk_stats: RiskStats
    tap: CaptureTap  # never started; ResultsStore requires one
    results: ResultsStore | None
    checker: BarsChecker
    status: RunStatus
    drained: DrainedCondition
    stall_monitor: StallMonitor | None = None  # the bench runs no heartbeat thread
    stop_reason: str | None = None
    plan_summary: dict | None = None

    @classmethod
    def from_args(cls, args: argparse.Namespace, cpus: CpuSets) -> BenchContext:
        settings = cli.bar_settings(args)
        knob: Knob[FeedDirectory] = Knob()
        directory = ReplayDirectory(knob)
        # --plan-only compiles with placeholder paths and must not read them.
        generations = (
            None if args.plan_only else register_replay(directory, tape=args.tape, pcap=args.pcap)
        )
        ingest, bars, ref_bars, risk = IngestStats(), BarsStats(), BarsStats(), RiskStats()
        tap = CaptureTap()
        results = None
        if args.risk == "gpu":
            results = ResultsStore(
                backend="gpu",
                interval_s=args.interval,
                settings=settings,
                directory=directory,
                tap=tap,
            )
        checker = BarsChecker()
        drained = DrainedCondition(
            ingest=ingest,
            bars=[bars, ref_bars] if args.check else [bars],
            collectors=(
                [(checker.count_left, bars), (checker.count_right, ref_bars)] if args.check else []
            ),
            risk=risk if results is not None else None,
            results=results,
        )
        return cls(
            args=args,
            cpus=cpus,
            settings=settings,
            directory_knob=knob,
            directory=directory,
            replay_generations=generations,
            ingest_stats=ingest,
            bars_stats=bars,
            ref_bars_stats=ref_bars,
            risk_stats=risk,
            tap=tap,
            results=results,
            checker=checker,
            status=RunStatus(),
            drained=drained,
        )

    def report_dict(self, session: object = None) -> dict:
        """The report without the verdict, from the stats objects (the session is not needed:
        the runner has mirrored it into ``status``). A value the run never produced is None,
        so a gate on it fails instead of passing on a default."""
        a, ing, bars = self.args, self.ingest_stats, self.bars_stats
        c = bars.counters
        # First to last bars batch: the graph's own processing span. The bar counters restart
        # with every generation, so after several --loops passes they time the last pass only:
        # such a run reports no span and no rates rather than rates several times too high.
        seconds = None
        if a.loops == 1 and c is not None and None not in (c.first_batch_ns, c.last_batch_ns):
            seconds = (c.last_batch_ns - c.first_batch_ns) / 1e9
        replay = ing.replay or {}
        stalls = ing.stalls_view()
        native = stalls.get("native") or {}
        sent = replay.get("sent")

        def per_s(count: int | None) -> float | None:
            return count / seconds if count is not None and seconds else None

        return {
            "termination": self.status.termination,
            "stop_reason": self.stop_reason,
            "config": {k: v for k, v in vars(a).items() if k != "gates"},
            "packets": ing.packets,
            "received_messages": ing.received_messages,
            "messages": ing.messages,
            "events": c.events if c is not None else None,
            "completed_bars": c.completed_bars if c is not None else None,
            "seconds": seconds,
            "msgs_per_s": per_s(ing.received_messages),
            "packets_per_s": per_s(ing.packets),
            "requested_pps": a.pps,
            "achieved_pps": per_s(sent),  # what the software NIC delivered
            "sequence_gaps": ing.dropped,
            "ring_dropped": ing.ring_dropped,
            "nic_dropped": replay.get("nic_dropped"),
            "stale_packets": ing.stale_packets + bars.stale_packets,
            "malformed_packets": ing.malformed_packets,
            "decode_malformed": ing.decode_malformed,
            "ring_peak": ing.ring_peak(),
            "ring_waits": native.get("ring_waits"),
            "ring_wait_ms": native.get("ring_wait_ms"),
            "lag_p50_ms": replay.get("lag_p50_ms"),
            "lag_p99_ms": replay.get("lag_p99_ms"),
            "lag_max_ms": replay.get("lag_max_ms"),
            "retries": ing.retries,
            "replay_thread_cpu_s": replay.get("thread_cpu_s"),
            "worst_compute_ms": _worst_ms(stalls, "compute"),
            "worst_burst_ms": _worst_ms(stalls, "burst"),
            "lanes": {
                lane["name"]: {"operators": lane["operators"], "cpus": lane["cpus"]}
                for lane in (self.plan_summary or {}).get("lanes", [])
            },
            "risk": self._risk_report() if self.results is not None else None,
            "check": None,
        }

    def finish_report(self, session_rc: int) -> int:
        """Complete the report (``--check`` results, verdict), print it, write ``--report``,
        and return OK only if the run and every gate passed."""
        report = self.report_dict()
        if self.args.check:
            report["check"] = self._check_report()
        ok, failures = bench_verdict(report, self.args.gates)
        if session_rc != ExitCode.OK:
            ok = False
            failures.insert(0, f"the run exited with status {int(session_rc)}")
        report = {"ok": ok, "failures": failures, **report}
        text = json.dumps(report, indent=2, default=str)
        print(text)
        if self.args.report:
            path = Path(self.args.report)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text + "\n")
        if ok:
            return ExitCode.OK
        return ExitCode.RUN_FAILED if session_rc != ExitCode.OK else ExitCode.EXPECTATION

    def _risk_report(self) -> dict:
        r = self.risk_stats
        latency = percentiles(list(self.results.latencies_ms))
        return {
            "analyzed": r.analyzed,
            "windows_superseded": r.windows_superseded,
            "late_activations": r.late_activations,
            "receive_to_result_ms": {k: latency[k] for k in ("p50", "p99", "max")},
        }

    def _check_report(self) -> dict:
        """The in-graph GPU/CPU pairs, both backends' final state, and every GPU bar against
        a host decode of the capture (one generation: ``--check`` needs ``--loops 1``)."""
        pairs = self.checker.summary()
        logger.info(
            "check: %d GPU and %d CPU windows, %d paired; decoding the capture on the host",
            pairs["windows_left"],
            pairs["windows_right"],
            pairs["windows_compared"],
        )
        ref = HostReference(self.directory.current, self.settings)
        ref.feed(PcapFrames.read(self.args.pcap))
        reference = self.checker.compare_reference(ref, self.checker.left_windows)
        for name, want, have in (
            ("counters", ref.counters(), _bar_counters(self.bars_stats)),
            ("decode_counts", ref.decode_counts(), _decode_counts(self.ingest_stats)),
        ):
            reference[f"{name}_equal"] = want == have
            if want != have:
                logger.warning("check: %s differ: reference %s, run %s", name, want, have)
        return {
            "windows_compared": pairs["windows_compared"],
            "times_equal": pairs["times_equal"],
            "nan_pattern_equal": pairs["nan_pattern_equal"],
            "max_ulps": pairs["max_ulps"],
            "final_state_equal": final_state_equal(self.bars_stats, self.ref_bars_stats),
            "reference": reference,
        }


def _worst_ms(stalls: dict, kind: str) -> float | None:
    worst = stalls.get(f"worst_{kind}")
    return worst["ms"] if worst else None


def _bar_counters(bars: BarsStats) -> dict[str, int] | None:
    """The run's side of ``HostReference.counters()``."""
    c = bars.counters
    if c is None:
        return None
    return dict(
        events=c.events,
        quotes=c.quotes,
        trades=c.trades,
        invalid_quotes=c.invalid_quotes,
        late_events=c.late_events,
        completed_bars=c.completed_bars,
        sequence_gaps=bars.sequence_gaps,
        stale_packets=bars.stale_packets,
    )


def _decode_counts(ingest: IngestStats) -> dict[str, int]:
    """The run's side of ``HostReference.decode_counts()``."""
    return dict(
        packets=ingest.packets,
        received_messages=ingest.received_messages,
        messages=ingest.messages,
        rejected_packets=ingest.malformed_packets + ingest.decode_malformed,
    )
