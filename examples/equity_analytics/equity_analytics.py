# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""equity MITCH -> quote bars -> rolling risk -> web dashboard, composed from holoscan.finance operators.

The front end (kernel socket, DAQIRI host memory, DAQIRI GPUDirect, or NIC-free GPU replay) is
chosen in compose(); everything downstream is shared. `holoscan run equity_analytics <mode>`.
"""

from __future__ import annotations

import argparse

import holoscan as hs

from holoscan_finance.app import cli, plan, runner
from holoscan_finance.app.context import EquityContext
from holoscan_finance.app.placement import Lanes
from holoscan_finance.dashboard.server import serve
from holoscan_finance.operators import (
    CallbackSinkOp,
    DaqiriPacketSourceOp,
    GpuMitchReplaySourceOp,
    GpuMitchSourceOp,
    MitchDecodeOp,
    QuoteBarsOp,
    RiskCrossCheckOp,
    RiskSummaryOp,
    RollingRiskOp,
    UdpPacketSourceOp,
)
from holoscan_finance.operators.edges import LATEST, lossless, topic_export_kwargs
from holoscan_finance.ref import Ref


def compose(args: argparse.Namespace, ctx: EquityContext) -> tuple[hs.Graph, Lanes]:
    g = hs.Graph("equity_analytics", default_clock=hs.RealtimeClock)
    lanes = Lanes(ctx.cpus, ingest_busy_poll=args.ingest_busy_poll)
    directory, ingest = Ref(ctx.directory_knob), Ref(ctx.ingest_stats)
    # drain and receiver threads: never on the (possibly busy-polled) ingest lane
    helpers = sorted(ctx.cpus.helper)
    gpu_front_end = args.ingest in ("gpudirect", "replay")
    gpu_common = dict(
        batch_size=args.batch_messages,
        ring_size=args.ring_size,
        gpu_device=args.gpu_device,
        helper_cpus=helpers,
        directory=directory,
        stats=ingest,
        tap=Ref(ctx.tap),
    )

    # ingest lane: the front end is the only mode-specific part of the graph
    if args.ingest == "gpudirect":
        # ConnectX -> GPU memory (DAQIRI GPUDirect) -> GPU gather + MITCH decode
        source = g.op(
            GpuMitchSourceOp,
            "source",
            port=args.udp_port,
            **cli.nic_params(args, ctx.cpus, gpudirect=True),
            **gpu_common,
        )
    elif args.ingest == "replay":
        # NIC-free: pcap frames in GPU memory, the same gather/ring/decode code
        source = g.op(
            GpuMitchReplaySourceOp,
            "source",
            pcap_path=args.pcap,
            pps=args.replay_pps,
            loops=args.replay_loops,
            on_loop=Ref(ctx.replay_generations),
            settled=Ref(ctx.drained.batches_consumed),  # a pass's tail is built before the next
            **gpu_common,
        )
    elif args.ingest == "udp":  # kernel UDP socket (host baseline)
        source = g.op(
            UdpPacketSourceOp,
            "source",
            host=args.udp_host,
            port=args.udp_port,
            batch_size=args.batch_messages,
            ring_size=args.ring_size,
            helper_cpus=helpers,
            stats=ingest,
        )
    else:  # daqiri-host: DAQIRI into pinned host memory (host baseline)
        source = g.op(
            DaqiriPacketSourceOp,
            "source",
            port=args.udp_port,
            **cli.nic_params(args, ctx.cpus, gpudirect=False),
            batch_size=args.batch_messages,
            ring_size=args.ring_size,
            helper_cpus=helpers,
            stats=ingest,
        )
    if gpu_front_end:  # decode is fused in the GPU sources: ring slots stay leased until decoded
        events, front = source.events, [source]
    else:  # host paths: the Python reference decoder is its own operator
        decode = g.op(MitchDecodeOp, "decode", directory=directory, stats=ingest, tap=Ref(ctx.tap))
        g.add_flow(source.packets, decode.packets, **lossless(16))
        events, front = decode.events, [source, decode]
    bars = g.op(
        QuoteBarsOp,
        "bars",
        backend="gpu" if gpu_front_end else "cpu",
        gpu_device=args.gpu_device,
        **ctx.settings.as_params(),
        directory=directory,
        stats=Ref(ctx.bars_stats),
    )
    g.add_flow(events, bars.events, **lossless(4 if gpu_front_end else 16))
    lanes.add("ingest", *front, bars)

    # analytics lane: the newest window only, at most every --interval, published at once
    compare = args.analytics_backend == "compare"
    risk = g.op(
        RollingRiskOp,
        "risk",
        backend="cpu" if args.analytics_backend == "cpu" else "gpu",
        executor="inline" if gpu_front_end else "process",  # host paths: off the decoder's GIL
        bar_seconds=ctx.settings.bar_seconds,
        gpu_device=args.gpu_device,
        min_interval_us=round(args.interval * 1e6),
        poll_period_us=args.risk_poll_us,
        cpu_threads=args.cpu_threads,
        full_window_bars=ctx.settings.window_bars,
        worker_cpus=sorted(ctx.cpus.worker),
        directory=directory,  # a CPU worker is tuned for each universe before its first window
        cross_check=compare,
        gate=Ref(ctx.gate),
        deadline_us=args.risk_deadline_us,
        stats=Ref(ctx.risk_stats),
    )
    publish = g.op(CallbackSinkOp, "publish", callback=Ref(ctx.results.on_risk))
    g.add_flow(bars.window, risk.window, **LATEST)  # bars never wait for analytics
    g.add_flow(risk.risk, publish.items, **lossless(8))
    lanes.add("analytics", risk, publish)

    if compare:  # crosscheck lane: every validated GPU window gets its CPU pair, one in flight
        crosscheck = g.op(
            RiskCrossCheckOp,
            "crosscheck",
            bar_seconds=ctx.settings.bar_seconds,
            cpu_threads=args.cpu_threads,
            full_window_bars=ctx.settings.window_bars,
            worker_cpus=sorted(ctx.cpus.worker),
            directory=directory,  # tuned per universe ahead of its first window
            stats=Ref(ctx.risk_stats),
        )
        publish_comparison = g.op(
            CallbackSinkOp, "publish_comparison", callback=Ref(ctx.results.on_comparison)
        )
        g.add_flow(risk.validate, crosscheck.risk, **lossless(2))
        g.add_flow(crosscheck.comparison, publish_comparison.items, **lossless(2))
        lanes.add("crosscheck", crosscheck, publish_comparison)

    if args.topics:  # optional: compact risk summaries for other processes over holomq
        summary = g.op(RiskSummaryOp, "risk_summary", stats=Ref(ctx.risk_stats))
        g.add_flow(risk.risk, summary.risk, **LATEST)
        # optional peer, 10 s attach bound, bounded subscribers
        topic = g.export_topic(
            "holomq://finance.risk",
            hs.payload.schema.String,
            endpoint_name="finance.risk",
            **topic_export_kwargs(),
        )
        # a stalled external reader never blocks the graph
        g.add_flow(summary.summary, topic.input, **LATEST)
        lanes.add("topics", summary)

    lanes.apply(g)
    return g, lanes


def rules(ctx: EquityContext) -> list[plan.Rule]:
    """What this composition promises; checked on plan.json() before the run starts."""
    checks = [
        plan.same_lane("source", "bars"),  # no thread hop on the packet path
        plan.edge("bars.window", "risk.window", policy="DROP_OLDEST", depth=1),  # bars never block
        # a poll, not a 50 ms grid (default 250 us)
        plan.clock_period_max("risk", ctx.args.risk_poll_us),
        plan.no_automatic_partitions(),
        plan.outputs_connected(),
        plan.inputs_connected(),
        plan.lanes_avoid_cpus(ctx.cpus.reserved),  # the DAQIRI lcores stay free
        plan.no_runtime_parameter_updates(),
    ]
    if ctx.args.topics:  # an external reader can never backpressure the graph
        checks.append(plan.edge("risk_summary.summary", "*", policy="DROP_OLDEST", depth=1))
    return checks


def main(argv: list[str] | None = None) -> int:
    args = cli.normalize(cli.equity_analytics_parser().parse_args(argv))
    cpus = runner.prepare_process(args)  # switch interval, logging, CPU sets, main-thread mask
    ctx = EquityContext.from_args(args, cpus)  # control plane, telemetry, results store
    g, lanes = compose(args, ctx)
    compiled = runner.compile_plan(
        g,
        lanes,
        rules=rules(ctx),
        ctx=ctx,
        plan_json=args.plan_json,
        topics_catalog=args.topics_catalog if args.topics else None,
    )
    if args.plan_only:
        return runner.print_plan(compiled)
    if args.port == 0:  # headless: CI smoke runs and reports
        return runner.run_headless(
            compiled,
            ctx,
            until=ctx.drained if args.exit_when_drained else None,
            timeout_s=args.timeout_s,
            report=args.report,
            expect=ctx.expectations,
        )
    return serve(
        compiled,
        ctx,
        host=args.host,
        port=args.port,
        exit_when_drained=args.exit_when_drained,
        report=args.report,
        expect=ctx.expectations,
    )


if __name__ == "__main__":
    raise SystemExit(main())
