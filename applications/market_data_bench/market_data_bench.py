# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Replay MITCH frames from GPU memory through gather -> decode -> quote bars (-> risk); measure and verify.

Finite and NIC-free: the run stops when every packet is decoded and every bar is built. With --check
the same decoded stream also feeds QuoteBarsOp(backend="cpu"), and every GPU window is compared with
its CPU twin: one operator class, two backends, one graph.
"""

from __future__ import annotations

import holoscan as hs

from holoscan_finance.app import bench, plan, runner
from holoscan_finance.app.bench import BenchContext
from holoscan_finance.app.placement import Lanes
from holoscan_finance.operators import (
    CallbackSinkOp,
    GpuMitchReplaySourceOp,
    QuoteBarsOp,
    RollingRiskOp,
)
from holoscan_finance.operators.edges import LATEST, lossless
from holoscan_finance.ref import Ref


def compose(args, ctx: BenchContext) -> tuple[hs.Graph, Lanes]:
    g = hs.Graph("market_data_bench", default_clock=hs.RealtimeClock)
    lanes = Lanes(ctx.cpus, ingest_busy_poll=args.ingest_busy_poll)
    directory = Ref(ctx.directory_knob)
    source = g.op(
        GpuMitchReplaySourceOp,
        "source",
        pcap_path=args.pcap,
        pps=args.pps,
        loops=args.loops,
        burst_packets=args.burst_packets,
        rx_timeout_us=args.rx_timeout_us,
        nic_bufs=args.nic_bufs,
        batch_size=args.batch_messages,
        ring_size=args.ring_size,
        gpu_device=args.gpu_device,
        # hf-replay is never created on a narrowed or busy-polled ingest core
        helper_cpus=sorted(ctx.cpus.helper),
        on_loop=Ref(ctx.replay_generations),
        directory=directory,
        stats=Ref(ctx.ingest_stats),
    )
    bars = g.op(
        QuoteBarsOp,
        "bars",
        backend="gpu",
        gpu_device=args.gpu_device,
        **ctx.settings.as_params(),
        emit_windows=args.check or args.risk != "none",
        directory=directory,
        stats=Ref(ctx.bars_stats),
    )
    g.add_flow(source.events, bars.events, **lossless(4))
    ingest = [source, bars]

    # --check: every GPU window, in order, and a CPU twin of the bars built from the same batches
    if args.check:
        windows = g.op(CallbackSinkOp, "windows", callback=Ref(ctx.checker.left))
        g.add_flow(bars.window, windows.items, **lossless(64))
        ingest.append(windows)
        ref_bars = g.op(
            QuoteBarsOp,
            "ref_bars",
            backend="cpu",
            **ctx.settings.as_params(),
            gpu_device=args.gpu_device,
            directory=directory,
            stats=Ref(ctx.ref_bars_stats),
        )
        ref_windows = g.op(CallbackSinkOp, "ref_windows", callback=Ref(ctx.checker.right))
        # fan-out: the source holds and retries a batch until both bars accept it
        g.add_flow(source.events, ref_bars.events, **lossless(8))
        g.add_flow(ref_bars.window, ref_windows.items, **lossless(64))
        lanes.add("reference", ref_bars, ref_windows)
    lanes.add("ingest", *ingest)

    if args.risk == "gpu":  # the analytics load of the full chain
        risk = g.op(
            RollingRiskOp,
            "risk",
            backend="gpu",
            executor="inline",
            gpu_device=args.gpu_device,
            bar_seconds=ctx.settings.bar_seconds,
            min_interval_us=round(args.interval * 1e6),
            stats=Ref(ctx.risk_stats),
        )
        publish = g.op(CallbackSinkOp, "publish", callback=Ref(ctx.results.on_risk))
        g.add_flow(bars.window, risk.window, **LATEST)
        g.add_flow(risk.risk, publish.items, **lossless(8))
        lanes.add("analytics", risk, publish)

    lanes.apply(g)
    return g, lanes


def rules(args, ctx: BenchContext) -> list[plan.Rule]:
    checks = [
        plan.same_lane("source", "bars"),
        plan.edge("source.events", "bars.events", policy="BLOCKING", min_depth=4),
        plan.no_automatic_partitions(),
        plan.outputs_connected(),
        plan.inputs_connected(),
    ]
    if args.check:
        checks += [
            plan.same_lane("bars", "windows"),
            plan.same_lane("ref_bars", "ref_windows"),
            plan.edge("bars.window", "windows.items", policy="BLOCKING", min_depth=64),
        ]
    if args.risk == "gpu":
        checks += [
            plan.edge("bars.window", "risk.window", policy="DROP_OLDEST", depth=1),
            plan.clock_period_max("risk", 1000),  # the bench keeps RollingRiskOp's default poll
        ]
    return checks


def main(argv: list[str] | None = None) -> int:
    args = bench.normalize_bench(bench.bench_parser().parse_args(argv))
    cpus = runner.prepare_process(args)
    ctx = BenchContext.from_args(args, cpus)
    g, lanes = compose(args, ctx)
    compiled = runner.compile_plan(
        g, lanes, rules=rules(args, ctx), ctx=ctx, plan_json=args.plan_json
    )
    if args.plan_only:
        return runner.print_plan(compiled)
    rc = runner.run_headless(compiled, ctx, until=ctx.drained, timeout_s=args.timeout_s)
    return ctx.finish_report(rc)  # report JSON, HostReference check, gates


if __name__ == "__main__":
    raise SystemExit(main())
