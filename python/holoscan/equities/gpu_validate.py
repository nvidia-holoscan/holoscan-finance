# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Field-for-field validation of the GPU MITCH decoder against ``mitch.decode_packet``.

Reads a pcap of the MITCH replay (as captured on the wire), decodes every packet
with the host decoder (multiprocessing) and with the GPU kernels, and requires
bit-identical results: per-packet accept/reject stage, observation counts, MITCH
message counts, sequence numbers and every observation field (floats compared as
raw bit patterns, so NaN placement and signed zeros must match too). A fuzz pass
mutates real packets to exercise every rejection rule.

    python -m holoscan.equities.gpu_validate --pcap base.pcap --tape /data/tape
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import struct
import time

import numpy as np

from holoscan.marketdata import mitch as unit
from holoscan.marketdata.daqiri import strip_udp_datagram
from holoscan.marketdata.gpu_mitch import FIELDS, SLOT_BYTES, decode_packets
from holoscan.marketdata.wire import EVENT_DTYPE

from .mitch import MAX_PACKET_EVENTS as MAX_EVENTS
from .mitch import day_start_ns, decode_packet, pack_events

_CTX = {}


def read_pcap(path, limit=0):
    """UDP payloads of every IPv4/UDP frame, via the host strip path."""
    payloads = []
    with open(path, "rb") as f:
        magic = f.read(24)[:4]
        endian = "<" if magic in (b"\xd4\xc3\xb2\xa1", b"\x4d\x3c\xb2\xa1") else ">"
        rec = struct.Struct(endian + "IIII")
        while not limit or len(payloads) < limit:
            h = f.read(16)
            if len(h) < 16:
                break
            incl = rec.unpack(h)[2]
            got = strip_udp_datagram(f.read(incl), minimum_payload=8)
            if got is not None:
                payloads.append(got[0])
    return payloads


def _host_decode(chunk):
    midnight_ns, universe = _CTX["midnight_ns"], _CTX["universe"]
    n = len(chunk)
    status = np.zeros(n, np.int32)
    nev = np.zeros(n, np.int32)
    wire = np.zeros(n, np.int32)
    seq = np.zeros(n, np.uint32)
    rows = []
    for i, data in enumerate(chunk):
        try:
            _, s, c = unit.inspect_unit(data)
        except (ValueError, struct.error):
            status[i] = 1
            continue
        seq[i], wire[i] = s, c
        try:
            batches, count = decode_packet(
                data, 0, generation=1, midnight_ns=midnight_ns, universe=universe
            )
        except (ValueError, struct.error):
            status[i] = 2
            continue
        ev = batches[0].events
        nev[i] = len(ev)
        rows.append(ev)
    events = np.concatenate(rows) if rows else np.zeros(0, dtype=EVENT_DTYPE)
    return status, nev, wire, seq, events


def _init(midnight_ns, universe):
    _CTX.update(midnight_ns=midnight_ns, universe=universe)


def host_decode(payloads, midnight_ns, universe, workers):
    parts = max(1, workers * 4)
    size = (len(payloads) + parts - 1) // parts
    chunks = [payloads[i : i + size] for i in range(0, len(payloads), size)]
    with mp.get_context("fork").Pool(
        workers, initializer=_init, initargs=(midnight_ns, universe)
    ) as pool:
        results = pool.map(_host_decode, chunks)
    cat = [np.concatenate([r[k] for r in results]) for k in range(5)]
    return cat


def to_slots(payloads):
    import cupy as cp

    n = len(payloads)
    lens = np.fromiter((len(p) for p in payloads), np.int32, count=n)
    host = np.zeros((n, SLOT_BYTES), np.uint8)
    for i, p in enumerate(payloads):
        host[i, : len(p)] = np.frombuffer(p, np.uint8)
    return cp.asarray(host), cp.asarray(lens)


def _to_host(array):
    return (
        array.get()
        if hasattr(array, "get") and not isinstance(array, np.ndarray)
        else np.asarray(array)
    )


def synthetic_stream(rng, packets, universe, midnight_ns, start_s=34200):
    """Random but valid replay units: quotes (incl. one-sided, halted, negative),
    trades, halt/resume conditions, several Time seconds per packet."""
    from holoscan.marketdata.wire import CONDITION, EVENT_DTYPE, HALTED, QUOTE, TRADE

    out, seq, t = [], 1, midnight_ns + start_s * 10**9
    for _ in range(packets):
        n = int(rng.integers(1, MAX_EVENTS + 1))
        ev = np.zeros(n, EVENT_DTYPE)
        t += int(rng.integers(0, 3 * 10**8))
        ev["timestamp_ns"] = t + np.sort(rng.integers(0, 2 * 10**9, n))
        t = int(ev["timestamp_ns"][-1])
        ev["symbol_id"] = rng.integers(0, universe, n)
        ev["kind"] = rng.choice([QUOTE, QUOTE, QUOTE, TRADE, TRADE, CONDITION], n)
        for f in ("bid", "ask", "bid_size", "ask_size", "price", "volume"):
            ev[f] = np.nan
        q = ev["kind"] == QUOTE
        mid = rng.uniform(1, 500, n)
        ev["bid"][q] = np.round(mid[q] * rng.uniform(0.97, 1.0, q.sum()), 4)
        ev["ask"][q] = np.round(mid[q] * rng.uniform(1.0, 1.03, q.sum()), 4)
        ev["bid"][q & (rng.random(n) < 0.05)] = np.nan
        ev["bid"][q & (rng.random(n) < 0.03)] *= -1
        ev["bid_size"][q] = rng.integers(0, 2000, q.sum())
        ev["ask_size"][q] = rng.integers(0, 2000, q.sum())
        tr = ev["kind"] == TRADE
        ev["price"][tr] = np.round(rng.uniform(1, 900_000, tr.sum()), 6)
        ev["volume"][tr] = rng.integers(0, 10_000, tr.sum())
        ev["flags"][(ev["kind"] == CONDITION) & (rng.random(n) < 0.5)] = HALTED
        ev["flags"][q & (rng.random(n) < 0.02)] = HALTED
        packed, count = pack_events(ev, seq, midnight_ns)
        out.append(packed)
        seq += count
    return out


def compare(host, gpu):
    """Return a dict of mismatch counts (all zero == bit-identical). ``gpu`` may hold
    NumPy (CPU decoder) or CuPy (GPU decoder) columns."""
    status, nev, wire, seq, events = host
    out = {
        "packets": int(len(status)),
        "status_mismatch": int((status != gpu.status).sum()),
        "status_counts_host": {
            int(k): int(v) for k, v in zip(*np.unique(status, return_counts=True))
        },
    }
    ok = status == 0
    out["events_per_packet_mismatch"] = int((nev[ok] != gpu.events_per_packet[ok]).sum())
    fr = status != 1  # framing fields are defined whenever inspect_unit passed
    out["wire_count_mismatch"] = int((wire[fr] != gpu.wire_count[fr]).sum())
    out["sequence_mismatch"] = int((seq[fr] != gpu.sequence[fr]).sum())
    out["events_host"] = int(len(events))
    out["events_gpu"] = int(gpu.total_events)
    if len(events) != gpu.total_events:
        out["event_count_mismatch"] = True
        return out
    native = events.astype(events.dtype.newbyteorder("="))
    for name, dt in FIELDS:
        h = native[name]
        g = _to_host(gpu.columns[name])
        if np.dtype(dt).kind == "f":
            bad = h.view(np.int64) != g.view(np.int64)  # bitwise: NaN and signed zero exact
        else:
            bad = h.astype(np.int64) != g.astype(np.int64)
        out[f"field_{name}_mismatch"] = int(bad.sum())
    out["max_price_magnitude"] = float(
        np.nanmax(np.abs(np.r_[native["bid"], native["ask"], native["price"], 0.0]))
    )
    return out


def fuzz(payloads, count, seed=7):
    """Mutations aimed at each rejection rule, applied to real packets."""
    rng = np.random.default_rng(seed)
    out = []
    picks = rng.integers(0, len(payloads), count)
    for j, i in enumerate(picks):
        b = bytearray(payloads[int(i)])
        mode = j % 12
        if mode == 0:  # random byte flip anywhere
            b[rng.integers(0, len(b))] ^= 1 << int(rng.integers(0, 8))
        elif mode == 1:  # truncate
            b = b[: int(rng.integers(0, len(b)))]
        elif mode == 2:  # wrong unit length
            struct.pack_into("<H", b, 0, (len(b) + int(rng.integers(1, 5))) & 0xFFFF)
        elif mode == 3:  # wrong group
            b[3] = int(rng.choice([ord("B"), 0, 255]))
        elif mode == 4:  # sequence 0 / overflow
            struct.pack_into("<I", b, 4, int(rng.choice([0, 0xFFFFFFFF])))
        elif mode == 5:  # message kind / length byte corruption inside the unit
            off = 8
            k = int(rng.integers(0, max(1, b[2])))
            for _ in range(k):
                if off + 1 >= len(b):
                    break
                off += b[off]
            if off + 1 < len(b):
                b[off + int(rng.integers(0, 2))] = int(rng.integers(0, 256))
        elif mode == 6:  # Time out of range
            if len(b) > 13 and b[9] == ord("T"):
                struct.pack_into("<I", b, 10, 86400 + int(rng.integers(0, 5)))
        elif mode == 7:  # nanoseconds >= 1e9 on the first observation
            if len(b) > 21 and b[9] == ord("T") and b[14] in (34, 35, 28):
                struct.pack_into("<I", b, 16, 10**9 + int(rng.integers(0, 100)))
        elif mode == 8:  # symbol 0 / beyond universe on the first observation
            if len(b) > 25 and b[9] == ord("T") and b[15] == ord("q"):
                struct.pack_into("<I", b, 20, int(rng.choice([0, 0xFFFFFFFF, 99999])))
        elif mode == 9:  # unmutated control packet
            pass
        elif mode == 10:  # sign bit on a price (negative prices decode, not reject)
            if len(b) > 30 and b[9] == ord("T") and b[15] == ord("q"):
                b[31] |= 0x80
        elif mode == 11:  # message count mismatch
            b[2] = (b[2] + int(rng.integers(1, 4))) & 0xFF
        out.append(bytes(b))
    return out


def run(args):
    import cupy as cp

    from .data import open_tape

    manifest, _ = open_tape(args.tape)
    midnight_ns = day_start_ns(manifest["date"])
    universe = len(manifest["symbols"])
    t0 = time.perf_counter()
    payloads = read_pcap(args.pcap, args.limit)
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
        res = compare(host, gpu)
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
    report["fuzz"] = compare(host, gpu)
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


def _observe_host(bars, batches, st):
    """EquityState.observe, verbatim semantics, for host ReplayBatch objects."""
    from holoscan.marketdata.wire import combine_batches

    pending = []
    for batch in batches:
        if batch.sequence < st["expected"]:
            st["stale"] += 1
            continue
        if st["expected"] and batch.sequence > st["expected"]:
            if pending:
                bars.observe(combine_batches(pending))
                pending.clear()
            st["gaps"] += batch.sequence - st["expected"]
            bars.mid[:] = np.nan
            bars.previous[:] = np.nan
        st["expected"] = batch.sequence + batch.sequence_count
        pending.append(batch)
    if pending:
        bars.observe(combine_batches(pending))


def _ulps(a, b):
    ai, bi = a.view(np.int64), b.view(np.int64)
    ai = np.where(ai < 0, np.int64(-(2**63)) - ai, ai)
    bi = np.where(bi < 0, np.int64(-(2**63)) - bi, bi)
    return np.abs(ai - bi)


def run_bars(args):
    """Replay the day through host QuoteBars and GpuQuoteBars with identical batching."""
    import cupy as cp

    from holoscan.marketdata.bars import QuoteBars
    from holoscan.marketdata.gpu_bars import GpuQuoteBars, GpuReplayBatch, apply_sequence
    from holoscan.marketdata.gpu_mitch import decode_packets

    from .data import open_tape

    manifest, _ = open_tape(args.tape)
    midnight_ns = day_start_ns(manifest["date"])
    symbols = manifest["symbols"]
    universe = len(symbols)
    payloads = read_pcap(args.pcap, args.limit)
    if args.drop_every:  # simulate wire loss so gap handling is exercised identically
        payloads = [p for i, p in enumerate(payloads) if (i + 1) % args.drop_every]
    settings = dict(
        bar_seconds=5, window_seconds=300, stale_seconds=30, max_spread_bps=100.0, min_bars=20
    )
    hb, gb = QuoteBars(symbols, **settings), GpuQuoteBars(symbols, **settings)
    hst = {"expected": 1, "stale": 0, "gaps": 0}
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
        _observe_host(hb, batches, hst)
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
            a_r, b_r = hb.rows[-r], _to_host(gb.rows[-r])
            a_m, b_m = hb.marks[-r], _to_host(gb.marks[-r])
            stats["bars_compared"] += 1
            stats["marks_bitwise_mismatch"] += int((a_m.view(np.int64) != b_m.view(np.int64)).sum())
            fa, fb = np.isfinite(a_r), np.isfinite(b_r)
            stats["returns_nan_pattern_mismatch"] += int(
                (np.isnan(a_r) != np.isnan(b_r)).sum() + (fa != fb).sum()
            )
            both = fa & fb
            u = _ulps(a_r[both], b_r[both])
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
                    or not np.array_equal(np.isnan(wa[0]), np.isnan(_to_host(wb[0])))
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pcap", required=True)
    p.add_argument(
        "--tape", required=True, help="directory produced by equities.data (for the manifest)"
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
    a = p.parse_args()
    raise SystemExit(run_bars(a) if a.bars else run(a))


if __name__ == "__main__":
    main()
