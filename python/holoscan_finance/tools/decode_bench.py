# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline decoder benchmark: Python vs compiled CPU (numba) vs GPU, same packets.

Each decoder starts where its receive path leaves the packets -- Python: one
``bytes`` object per packet; CPU: a host slot array; GPU: a device slot array --
and ends with decoded observations in its own memory (GPU timing includes the
per-packet header copy back to the host and a synchronize). Packets are staged in
groups of at least ``--group-msgs`` MITCH messages, as the live sources do.
"Decode + bars" adds the quote-bar stage each path uses: NumPy ``QuoteBars`` for
the Python and CPU decoders, the fused CUDA bars for the GPU decoder.

Warm-up (numba/CUDA compilation) runs before timing. This measures the decode and
bars stages only; the live ingest ceiling also includes receive, scheduling and
analytics, so do not compare these numbers with live packets/s figures.

    python -m holoscan_finance.tools.decode_bench --pcap data/nyq/day.pcap --tape data/nyq/tape

The bar settings come from ``BarSettings``. CuPy and numba are imported only by the paths
that use them.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from typing import TYPE_CHECKING

import numpy as np

from ..marketdata.bars import QuoteBars
from ..marketdata.framing import PcapFrames
from ..marketdata.records import EVENT_DTYPE, ReplayBatch, combine_batches
from ..nyq.mapping import day_start_ns, decode_packet
from ..nyq.tape import open_tape
from ..settings import BarSettings

if TYPE_CHECKING:
    from ..marketdata.gpu_bars import GpuQuoteBars

SETTINGS = BarSettings().as_params()
NATIVE = EVENT_DTYPE.newbyteorder("=")


def groups_of(payloads: list[bytes], group_msgs: int) -> list[tuple[int, int]]:
    out, i = [], 0
    while i < len(payloads):
        j, staged = i, 0
        while j < len(payloads) and staged < group_msgs:
            staged += payloads[j][2]
            j += 1
        out.append((i, j))
        i = j
    return out


def run_python(
    payloads: list[bytes],
    groups: list[tuple[int, int]],
    midnight: int,
    universe: int,
    symbols: list[str],
    bars: bool,
) -> tuple[float, QuoteBars | None]:
    qb = QuoteBars(symbols, **SETTINGS) if bars else None
    t0 = time.perf_counter()
    for a, b in groups:
        batches = [
            decode_packet(payloads[k], k, generation=1, midnight_ns=midnight, universe=universe)[0][
                0
            ]
            for k in range(a, b)
        ]
        if qb is not None:
            qb.observe(combine_batches(batches))
    return time.perf_counter() - t0, qb


def host_slots(
    payloads: list[bytes], groups: list[tuple[int, int]]
) -> list[tuple[np.ndarray, np.ndarray]]:
    from ..marketdata.gpu_mitch import SLOT_BYTES

    out = []
    for a, b in groups:
        lens = np.fromiter((len(payloads[k]) for k in range(a, b)), np.int64, count=b - a)
        slots = np.zeros((b - a, SLOT_BYTES), np.uint8)
        for r, k in enumerate(range(a, b)):
            slots[r, : lens[r]] = np.frombuffer(payloads[k], np.uint8)
        out.append((slots, lens))
    return out


def run_cpu(
    staged: list[tuple[np.ndarray, np.ndarray]],
    groups: list[tuple[int, int]],
    midnight: int,
    universe: int,
    symbols: list[str],
    threads: int,
    bars: bool,
) -> tuple[float, QuoteBars | None]:
    import numba

    from .cpu_mitch import decode_packets

    numba.set_num_threads(threads)
    decode_packets(*staged[0], midnight_ns=midnight, universe=universe)  # compile / warm
    qb = QuoteBars(symbols, **SETTINGS) if bars else None
    t0 = time.perf_counter()
    for (slots, lens), (a, _) in zip(staged, groups):
        dec = decode_packets(slots, lens, midnight_ns=midnight, universe=universe)
        if qb is not None:
            ev = np.zeros(dec.total_events, NATIVE)
            for name, col in dec.columns.items():
                ev[name] = col
            n = len(lens)
            recv = np.arange(a, a + n, dtype=np.int64)
            qb.observe(ReplayBatch(ev, 1, int(recv[-1]), int(dec.sequence[0]), dec.offsets, recv))
    return time.perf_counter() - t0, qb


def run_gpu(
    staged: list[tuple[np.ndarray, np.ndarray]],
    groups: list[tuple[int, int]],
    midnight: int,
    universe: int,
    symbols: list[str],
    bars: bool,
) -> tuple[float, GpuQuoteBars | None]:
    import cupy as cp

    from ..marketdata.gpu_bars import GpuQuoteBars, GpuReplayBatch, apply_sequence
    from ..marketdata.gpu_mitch import decode_packets
    from ..marketdata.warmup import warm_bars_kernels, warm_decode_kernels

    dev = [(cp.asarray(s), cp.asarray(n.astype(np.int32))) for s, n in staged]
    warm_decode_kernels()
    warm_bars_kernels(SETTINGS)
    decode_packets(*dev[0], midnight_ns=midnight, universe=universe)
    gb = GpuQuoteBars(symbols, **SETTINGS) if bars else None
    expected = 1
    cp.cuda.Device().synchronize()
    t0 = time.perf_counter()
    for (slots, lens), (a, _) in zip(dev, groups):
        dec = decode_packets(slots, lens, midnight_ns=midnight, universe=universe)
        if gb is not None:
            n = int(lens.shape[0])
            recv = np.arange(a, a + n, dtype=np.int64)
            batch = GpuReplayBatch(
                dec.columns, 1, dec.sequence, dec.wire_count, dec.offsets, recv, int(recv[-1])
            )
            expected, _, _ = apply_sequence(gb, batch, expected)
    cp.cuda.Device().synchronize()
    return time.perf_counter() - t0, gb


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m holoscan_finance.tools.decode_bench",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--pcap", required=True)
    p.add_argument("--tape", required=True)
    p.add_argument("--packets", type=int, default=200_000)
    p.add_argument("--group-msgs", type=int, default=16384)
    p.add_argument("--threads", default="1,2,4,8,14")
    p.add_argument("--skip-gpu", action="store_true")
    p.add_argument("--out")
    return p


def main(argv: list[str] | None = None) -> int:
    a = parser().parse_args(argv)
    manifest, _ = open_tape(a.tape)
    midnight, symbols = day_start_ns(manifest["date"]), manifest["symbols"]
    universe = len(symbols)
    payloads = PcapFrames.read(a.pcap, a.packets).payloads()
    groups = groups_of(payloads, a.group_msgs)
    msgs = sum(p[2] for p in payloads)
    staged = host_slots(payloads, groups)
    rows = []

    def record(path, threads, stage, secs, bars_obj):
        r = dict(path=path, threads=threads, stage=stage, seconds=round(secs, 3),
                 msgs_per_s=msgs / secs, packets_per_s=len(payloads) / secs)  # fmt: skip
        if bars_obj is not None:
            r["bars_closed"] = int(bars_obj.completed_bars)
        rows.append(r)
        print(f"{path:>6} {str(threads):>3} thr  {stage:<14} {secs:8.3f} s  "
              f"{r['msgs_per_s'] / 1e6:7.2f} M msgs/s  {r['packets_per_s']:>10,.0f} pkts/s", flush=True)  # fmt: skip

    print(f"{len(payloads):,} packets, {msgs:,} MITCH messages, {len(groups)} groups", flush=True)
    for bars in (False, True):
        stage = "decode+bars" if bars else "decode"
        secs, qb = run_python(payloads, groups, midnight, universe, symbols, bars)
        record("python", 1, stage, secs, qb)
        for t in (int(x) for x in a.threads.split(",")):
            secs, qb = run_cpu(staged, groups, midnight, universe, symbols, t, bars)
            record("cpu", t, stage, secs, qb)
        if not a.skip_gpu:
            secs, gb = run_gpu(staged, groups, midnight, universe, symbols, bars)
            record("gpu", "-", stage, secs, gb)
    report = dict(packets=len(payloads), messages=msgs, groups=len(groups), cpu=platform.processor()
                  or platform.machine(), rows=rows)  # fmt: skip
    if a.out:
        with open(a.out, "w") as f:
            json.dump(report, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
