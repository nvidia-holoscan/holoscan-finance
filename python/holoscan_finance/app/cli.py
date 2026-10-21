# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ``equity_analytics`` flags, their checks, and the flag groups the bench shares.

``--ingest`` takes ``udp|daqiri-host|gpudirect|replay`` and the NIC flags are ``--nic-*``.
``normalize`` fills the defaults that depend on the front end (the ``IngestDefaults``
GPUDirect batch and ring sizes) and rejects what cannot run, among them a GPU front end
with CPU analytics, which would put market data on the CPU (invariant 3). A rejected flag
exits with status 2, as argparse does. ``nic_params`` builds the DAQIRI Params of the NIC
sources.
"""

from __future__ import annotations

import argparse
import dataclasses

from ..ingest.affinity import CpuSets, parse_cpu_list
from ..settings import GPU_INGEST_MODES, INGEST_MODES, BarSettings, IngestDefaults
from .runner import usage_error


def add_bar_flags(p: argparse.ArgumentParser) -> None:
    """The five quote-bar settings (defaults from ``BarSettings``)."""
    d = BarSettings()
    g = p.add_argument_group("quote bars")
    g.add_argument("--bar-seconds", type=int, default=d.bar_seconds)
    g.add_argument("--window-seconds", type=int, default=d.window_seconds)
    g.add_argument("--stale-seconds", type=int, default=d.stale_seconds)
    g.add_argument("--min-bars", type=int, default=d.min_bars)
    g.add_argument("--max-spread-bps", type=float, default=d.max_spread_bps)


def add_placement_flags(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("placement")
    g.add_argument(
        "--ingest-cpus",
        metavar="LIST",
        help="CPUs of the ingest lane, such as '4-5' (default: every CPU the app may use)",
    )
    g.add_argument(
        "--ingest-busy-poll",
        action="store_true",
        help="busy-poll the ingest lane; pair it with --ingest-cpus so other threads keep off",
    )


def add_plan_flags(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("plan")
    g.add_argument(
        "--plan-only",
        action="store_true",
        help="compile, check the plan rules, print the lanes, edges and clocks, and exit",
    )
    g.add_argument("--plan-json", metavar="PATH", help="write the compiled plan.json here")


def equity_analytics_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="NYQ MITCH -> quote bars -> rolling risk -> web dashboard on Holoscan 5."
    )

    g = p.add_argument_group("dashboard")
    g.add_argument("--host", default="127.0.0.1")
    g.add_argument("--port", type=int, default=8080, help="0: headless (CI runs, reports)")
    g.add_argument(
        "--replay-control",
        metavar="URL",
        help="replay controller (tcpreplay of a captured day) for the full-day button",
    )

    g = p.add_argument_group("ingest")
    g.add_argument(
        "--ingest",
        default="udp",
        choices=INGEST_MODES,
        help="udp: kernel UDP socket; daqiri-host: DAQIRI into pinned host memory; gpudirect: "
        "DAQIRI GPUDirect into GPU memory, GPU gather + MITCH decode + quote bars; replay: pcap "
        "frames from GPU memory through that same GPU chain, no NIC",
    )
    g.add_argument("--udp-host", default="127.0.0.1")
    g.add_argument("--udp-port", type=int, default=26400)

    g = p.add_argument_group("NIC (daqiri-host, gpudirect)")
    g.add_argument("--nic-pcie", default="0004:03:00.0")
    g.add_argument(
        "--nic-config",
        default="",
        metavar="YAML",
        help="DAQIRI config file; empty: build it from these flags",
    )
    g.add_argument("--nic-master-core", type=int, default=3)
    g.add_argument("--nic-rx-core", type=int, default=9)
    g.add_argument(
        "--nic-num-bufs",
        type=int,
        default=IngestDefaults().gpu_num_bufs,
        help="(gpudirect) NIC receive buffers: ~0.57 s of a 200k pkt/s burst; the GPU's BAR1 "
        "window bounds it (256 MiB holds ~120k 2 KiB buffers); ignored with --nic-config",
    )

    g = p.add_argument_group("replay (NIC-free GPU chain)")
    g.add_argument("--pcap", metavar="PATH", help="Ethernet capture of the MITCH feed")
    g.add_argument("--tape", metavar="DIR", help="the prepared tape the capture was made from")
    g.add_argument("--replay-pps", type=int, default=200000, help="packets/s; 0: unpaced")
    g.add_argument(
        "--replay-loops",
        type=int,
        default=1,
        help="passes over the capture, each a new generation on a new sender port; 0: forever",
    )
    g.add_argument(
        "--replay-sender-port",
        type=int,
        help="UDP source port of the first generation (default: the capture's most common); "
        "a port the capture carries picks that sender's frames as the feed, any other port "
        "re-sources the most common sender's frames",
    )

    g = p.add_argument_group("batching")
    g.add_argument(
        "--batch-messages",
        type=int,
        help="most messages staged per source compute: 131072 for GPU front ends (amortizes "
        "launches), else 16384. Sources take what is queued and never wait to fill a batch",
    )
    g.add_argument(
        "--ring-size",
        type=int,
        help="receive ring: 1048576 slots for GPU front ends (~5 s of a 200k pkt/s burst), "
        "else 65536 packets",
    )

    add_bar_flags(p)

    g = p.add_argument_group("analytics")
    g.add_argument("--analytics-backend", choices=("compare", "cpu", "gpu"), default="compare")
    g.add_argument(
        "--interval",
        type=float,
        default=IngestDefaults().interval_s,
        help="minimum seconds between risk updates",
    )
    g.add_argument("--cpu-threads", type=int, default=0, help="CPU BLAS threads; 0: tune")
    g.add_argument("--gpu-device", type=int, default=0)
    g.add_argument(
        "--risk-poll-us",
        type=int,
        default=1000,
        help="how often risk looks for a new window (100-100000)",
    )
    g.add_argument(
        "--risk-deadline-us",
        type=int,
        default=0,
        help="count risk poll ticks that start later than this; 0: off",
    )

    add_placement_flags(p)

    g = p.add_argument_group("run")
    g.add_argument(
        "--exit-when-drained",
        action="store_true",
        help="(replay) stop once every packet is decoded and every window analyzed",
    )
    g.add_argument(
        "--timeout-s", type=float, default=0, help="(headless) fail after this long; 0: no limit"
    )
    g.add_argument("--report", metavar="PATH", help="write the run report (JSON) here")
    g.add_argument(
        "--expect-windows",
        type=int,
        default=0,
        metavar="N",
        help="exit 5 unless N windows were analyzed (compare: and at least one comparison, "
        "none failed)",
    )
    add_plan_flags(p)

    g = p.add_argument_group("holomq")
    g.add_argument(
        "--topics",
        action="store_true",
        help="export a compact risk summary to holomq://finance.risk for other processes",
    )
    g.add_argument(
        "--topics-catalog",
        metavar="PATH",
        default="/tmp/holoscan-finance/endpoints.json",
        help="where subscribers find the endpoint (the directory is created)",
    )
    return p


def bar_settings(args: argparse.Namespace) -> BarSettings:
    """The bar flags as ``BarSettings``; ValueError under the ``QuoteBars`` rules."""
    return BarSettings(**{f.name: getattr(args, f.name) for f in dataclasses.fields(BarSettings)})


def normalize(args: argparse.Namespace) -> argparse.Namespace:
    """Front-end defaults and cross-flag checks (exit 2 when rejected)."""
    gpu_front_end = args.ingest in GPU_INGEST_MODES
    defaults = IngestDefaults()
    if args.batch_messages is None:
        gpu, host = defaults.gpu_batch_messages, defaults.host_batch_messages
        args.batch_messages = gpu if gpu_front_end else host
    if args.ring_size is None:
        gpu, host = defaults.gpu_ring_size, defaults.host_ring_size
        args.ring_size = gpu if gpu_front_end else host

    if gpu_front_end and args.analytics_backend == "cpu":
        usage_error(
            f"--ingest {args.ingest} runs GPU analytics: use --analytics-backend compare or gpu"
        )
    if args.ingest == "replay":
        for flag in ("pcap", "tape"):
            if getattr(args, flag):
                continue
            if not args.plan_only:
                usage_error(f"--ingest replay needs --{flag}")
            setattr(args, flag, "none")  # the source's setup() needs a path; start() never runs
        if args.exit_when_drained and args.replay_loops == 0:
            usage_error("--exit-when-drained needs a finite replay: --replay-loops 0 never ends")
    elif args.exit_when_drained:
        usage_error("--exit-when-drained needs --ingest replay: the other front ends never end")
    if not 100 <= args.risk_poll_us <= 100_000:
        usage_error("--risk-poll-us must be within 100-100000")
    if not 0.001 <= args.interval <= 60:
        usage_error("--interval must be within 0.001-60 seconds")
    try:
        bar_settings(args)
        if args.ingest_cpus:
            parse_cpu_list(args.ingest_cpus)
    except ValueError as exc:
        usage_error(str(exc))
    return args


def nic_params(args: argparse.Namespace, cpus: CpuSets, *, gpudirect: bool) -> dict:
    """DAQIRI Params of ``GpuMitchSourceOp`` (``gpudirect``) or ``DaqiriPacketSourceOp``.

    ``daqiri_init`` runs widened to every start-up CPU, and its caller's mask is restored
    afterwards; ``num_bufs`` only applies to GPUDirect.
    """
    params = dict(
        pcie_address=args.nic_pcie,
        daqiri_config=args.nic_config or "",
        master_core=args.nic_master_core,
        rx_core=args.nic_rx_core,
        init_cpus=sorted(cpus.all),
    )
    if gpudirect:
        params["num_bufs"] = args.nic_num_bufs
    return params
