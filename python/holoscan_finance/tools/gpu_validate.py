# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Field-for-field validation of the GPU MITCH decoder against ``nyq.mapping.decode_packet``.

Reads a pcap of the MITCH replay (as captured on the wire), decodes every packet
with the host decoder (multiprocessing) and with the GPU kernels, and requires
bit-identical results: per-packet accept/reject stage, observation counts, MITCH
message counts, sequence numbers and every observation field (floats compared as
raw bit patterns, so NaN placement and signed zeros must match too). A fuzz pass
mutates real packets to exercise every rejection rule. ``--bars`` replays the day
through host ``QuoteBars`` and ``GpuQuoteBars`` with identical batching instead;
``--decoder cpu`` validates the numba decoder (needs numba>=0.68).

    python -m holoscan_finance.tools.gpu_validate --pcap base.pcap --tape /data/tape

From 4.3 ``equities.gpu_validate`` (same flags). The oracle pieces moved to the library:
``validation.compare`` (host decode, comparison, ULPs), ``nyq.synthetic`` (``fuzz``) and
``marketdata.sequence`` (the host sequence rules, was ``_observe_host``).
"""

from __future__ import annotations

import argparse
import json
import struct
import time
from typing import TYPE_CHECKING

import numpy as np

from ..marketdata.framing import PcapFrames
from ..marketdata.gpu_mitch import SLOT_BYTES
from ..marketdata.sequence import SequenceTracker
from ..nyq.mapping import day_start_ns, decode_packet
from ..nyq.synthetic import fuzz
from ..nyq.tape import open_tape
from ..settings import BarSettings
from ..validation.compare import compare_decoded, host_decode, to_host, ulps

if TYPE_CHECKING:
    import cupy as cp


def to_slots(payloads: list[bytes]) -> tuple[cp.ndarray, cp.ndarray]:
    """Device slot rows and int32 lengths of ``payloads``, as the receive ring holds them."""
    import cupy as cp

    n = len(payloads)
    lens = np.fromiter((len(p) for p in payloads), np.int32, count=n)
    host = np.zeros((n, SLOT_BYTES), np.uint8)
    for i, p in enumerate(payloads):
        host[i, : len(p)] = np.frombuffer(p, np.uint8)
    return cp.asarray(host), cp.asarray(lens)


def run(args: argparse.Namespace) -> int:
    """Decode validation: the whole capture in ``--chunk`` packets, then the fuzz pass."""
    import cupy as cp

    from ..marketdata.gpu_mitch import decode_packets

    manifest, _ = open_tape(args.tape)
    midnight_ns = day_start_ns(manifest["date"])
    universe = len(manifest["symbols"])
    t0 = time.perf_counter()
    payloads = PcapFrames.read(args.pcap, args.limit).payloads()
    report = {
        "pcap": args.pcap,
        "packets": len(payloads),
        "read_s": round(time.perf_counter() - t0, 1),
        "midnight_ns": midnight_ns,
        "universe": universe,
    }
    print(f"read {len(payloads):,} packets in {report['read_s']} s", flush=True)
    totals = {}
    gpu_s = 0.0
    host_s = 0.0
    step = args.chunk
    for lo in range(0, len(payloads), step):
        part = payloads[lo : lo + step]
        t = time.perf_counter()
        host = host_decode(part, midnight_ns, universe, args.workers)
        host_s += time.perf_counter() - t
        if args.decoder == "cpu":
            from .cpu_mitch import decode_packets as cpu_decode

            slots, lens = (x.get() for x in to_slots(part))
            t = time.perf_counter()
            gpu = cpu_decode(
                slots, lens.astype(np.int64), midnight_ns=midnight_ns, universe=universe
            )
        else:
            slots, lens = to_slots(part)
            cp.cuda.Device().synchronize()
            t = time.perf_counter()
            gpu = decode_packets(slots, lens, midnight_ns=midnight_ns, universe=universe)
            cp.cuda.Device().synchronize()
        gpu_s += time.perf_counter() - t
        res = compare_decoded(host, gpu)
        for k, v in res.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                totals[k] = (
                    totals.get(k, 0) + v if k != "max_price_magnitude" else max(totals.get(k, 0), v)
                )
            elif isinstance(v, bool):
                totals[k] = totals.get(k, False) or v
        print(
            f"  packets {lo:>9,}-{lo + len(part):>9,}: "
            + ", ".join(f"{k}={v}" for k, v in res.items() if k.endswith("mismatch") and v),
            flush=True,
        )
        del slots, lens, gpu
    report["day"] = totals
    report["host_decode_s"] = round(host_s, 1)
    report["decoder_s"] = round(gpu_s, 3)
    # Fuzz: every rejection path must agree.
    fz = fuzz(payloads, args.fuzz)
    host = host_decode(fz, midnight_ns, universe, args.workers)
    slots, lens = to_slots(fz)
    if args.decoder == "cpu":
        from .cpu_mitch import decode_packets as cpu_decode

        gpu = cpu_decode(
            slots.get(), lens.get().astype(np.int64), midnight_ns=midnight_ns, universe=universe
        )
    else:
        gpu = decode_packets(slots, lens, midnight_ns=midnight_ns, universe=universe)
    report["decoder"] = args.decoder
    report["fuzz"] = compare_decoded(host, gpu)
    mism = [k for k, v in {**totals, **report["fuzz"]}.items() if k.endswith("mismatch") and v]
    report["bit_identical"] = (
        not mism
        and not totals.get("event_count_mismatch")
        and not report["fuzz"].get("event_count_mismatch")
    )
    report["mismatched_fields"] = mism
    print(json.dumps(report, indent=2), flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
    return 0 if report["bit_identical"] else 1


def run_bars(args: argparse.Namespace) -> int:
    """Replay the day through host QuoteBars and GpuQuoteBars with identical batching."""
    import cupy as cp

    from ..marketdata.bars import QuoteBars
    from ..marketdata.gpu_bars import GpuQuoteBars, GpuReplayBatch, apply_sequence
    from ..marketdata.gpu_mitch import decode_packets

    manifest, _ = open_tape(args.tape)
    midnight_ns = day_start_ns(manifest["date"])
    symbols = manifest["symbols"]
    universe = len(symbols)
    payloads = PcapFrames.read(args.pcap, args.limit).payloads()
    if args.drop_every:  # simulate wire loss so gap handling is exercised identically
        payloads = [p for i, p in enumerate(payloads) if (i + 1) % args.drop_every]
    settings = BarSettings().as_params()
    hb, gb = QuoteBars(symbols, **settings), GpuQuoteBars(symbols, **settings)
    tracker = SequenceTracker(generation=1)
    g_expected, g_stale, g_gaps = 1, 0, 0
    stats = dict(
        groups=0,
        bars_compared=0,
        marks_bitwise_mismatch=0,
        returns_nan_pattern_mismatch=0,
        returns_finite=0,
        returns_ulp_max=0,
        returns_ulp_nonzero=0,
        times_mismatch=0,
        bar_recv_mismatch=0,
        windows_compared=0,
        window_mismatch=0,
    )
    i = 0
    n = len(payloads)
    group_msgs = args.group_msgs
    host_s = gpu_s = 0.0
    while i < n:
        # Stage packets the way the source's compute() does: until >= group_msgs messages.
        j, staged = i, 0
        while j < n and staged < group_msgs:
            staged += payloads[j][2] if len(payloads[j]) > 2 else 0
            j += 1
        part = payloads[i:j]
        recv = np.arange(i, j, dtype=np.int64) * 1000 + 1
        before_h, before_g = hb.completed_bars, gb.completed_bars
        t = time.perf_counter()
        batches = []
        for k, data in enumerate(part):
            try:
                out, _ = decode_packet(
                    data, int(recv[k]), generation=1, midnight_ns=midnight_ns, universe=universe
                )
                batches.extend(out)
            except (ValueError, struct.error):
                pass
        tracker.observe(hb, batches)
        host_s += time.perf_counter() - t
        slots, lens = to_slots(part)
        cp.cuda.Device().synchronize()
        t = time.perf_counter()
        dec = decode_packets(slots, lens, midnight_ns=midnight_ns, universe=universe)
        ok = dec.status == 0
        idx = np.flatnonzero(ok)
        if len(idx):
            if len(idx) != len(part):
                offs = dec.offsets[idx]
            else:
                offs = dec.offsets
            gbatch = GpuReplayBatch(
                dec.columns,
                1,
                dec.sequence[idx],
                dec.wire_count[idx],
                offs,
                recv[idx],
                int(recv[idx[-1]]),
            )
            g_expected, s1, g1 = apply_sequence(gb, gbatch, g_expected)
            g_stale += s1
            g_gaps += g1
        cp.cuda.Device().synchronize()
        gpu_s += time.perf_counter() - t
        # Compare every bar closed during this group.
        dh, dg = hb.completed_bars - before_h, gb.completed_bars - before_g
        if dh != dg:
            stats["closed_count_mismatch_at_group"] = stats["groups"]
            break
        k = min(dh, len(hb.rows))
        for r in range(1, k + 1):
            a_r, b_r = hb.rows[-r], to_host(gb.rows[-r])
            a_m, b_m = hb.marks[-r], to_host(gb.marks[-r])
            stats["bars_compared"] += 1
            stats["marks_bitwise_mismatch"] += int((a_m.view(np.int64) != b_m.view(np.int64)).sum())
            fa, fb = np.isfinite(a_r), np.isfinite(b_r)
            stats["returns_nan_pattern_mismatch"] += int(
                (np.isnan(a_r) != np.isnan(b_r)).sum() + (fa != fb).sum()
            )
            both = fa & fb
            u = ulps(a_r[both], b_r[both])
            stats["returns_finite"] += int(both.sum())
            stats["returns_ulp_nonzero"] += int((u > 0).sum())
            stats["returns_ulp_max"] = max(stats["returns_ulp_max"], int(u.max()) if len(u) else 0)
            stats["times_mismatch"] += int(hb.times[-r] != gb.times[-r])
        if dh:
            stats["bar_recv_mismatch"] += int(hb.bar_recv_ns != gb.bar_recv_ns)
            wa, wb = hb.window(), gb.window()
            if wa is not None or wb is not None:
                stats["windows_compared"] += 1
                if (
                    (wa is None) != (wb is None)
                    or not np.array_equal(wa[1], wb[1])
                    or wa[0].shape != wb[0].shape
                    or not np.array_equal(np.isnan(wa[0]), np.isnan(to_host(wb[0])))
                ):
                    stats["window_mismatch"] += 1
        stats["groups"] += 1
        i = j
        del slots, lens, dec
    final = {}
    for name in ("mid", "trade", "volume", "previous"):
        final[f"{name}_bitwise_mismatch"] = int(
            (getattr(hb, name).view(np.int64) != getattr(gb, name).view(np.int64)).sum()
        )
    final["quote_time_mismatch"] = int((hb.quote_time != gb.quote_time).sum())
    for name in ("events", "quotes", "trades", "invalid_quotes", "late_events", "completed_bars"):
        final[f"{name}"] = [int(getattr(hb, name)), int(getattr(gb, name))]
    hst = {
        "expected": tracker.expected,
        "stale": tracker.stale_packets,
        "gaps": tracker.sequence_gaps,
    }
    final["sequence"] = {
        "host": hst,
        "gpu": {"expected": g_expected, "stale": g_stale, "gaps": g_gaps},
    }
    report = {
        "packets": n,
        "drop_every": args.drop_every,
        "group_msgs": group_msgs,
        "bars": stats,
        "final_state": final,
        "host_decode_plus_bars_s": round(host_s, 1),
        "gpu_decode_plus_bars_s": round(gpu_s, 1),
    }
    exact = (
        all(v == 0 for k, v in stats.items() if k.endswith("mismatch"))
        and "closed_count_mismatch_at_group" not in stats
        and all(v == 0 for k, v in final.items() if k.endswith("mismatch"))
        and all(a == b for k, (a, b) in ((k, v) for k, v in final.items() if isinstance(v, list)))
        and hst["gaps"] == g_gaps
        and hst["stale"] == g_stale
        and hst["expected"] == g_expected
    )
    report["state_identical"] = exact
    report["returns_note"] = (
        "returns use CUDA log vs NumPy log; compared in ULPs, NaN pattern must match exactly"
    )
    print(json.dumps(report, indent=2), flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
    return 0 if exact else 1


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m holoscan_finance.tools.gpu_validate",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--pcap", required=True)
    p.add_argument(
        "--tape", required=True, help="tape directory of the capture's day (for the manifest)"
    )
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--chunk", type=int, default=250_000)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--fuzz", type=int, default=240_000)
    p.add_argument("--out")
    p.add_argument("--bars", action="store_true", help="validate GPU quote bars instead of decode")
    p.add_argument("--decoder", choices=("gpu", "cpu"), default="gpu", help="decoder to validate")
    p.add_argument("--drop-every", type=int, default=0, help="--bars: drop every Nth packet (loss)")
    p.add_argument(
        "--group-msgs", type=int, default=16384, help="--bars: messages per compute batch"
    )
    return p


def main(argv: list[str] | None = None) -> int:
    a = parser().parse_args(argv)
    return run_bars(a) if a.bars else run(a)


if __name__ == "__main__":
    raise SystemExit(main())
