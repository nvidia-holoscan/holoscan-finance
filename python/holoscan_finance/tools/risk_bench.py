# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rolling-analytics microbenchmark: per-window calculation time of the GPU and CPU engines.

Each backend runs the code path an application runs, in this process:

- ``gpu``: ``RollingRisk(gpu, inline)`` on a ``BarWindow`` of per-bar CuPy vectors, as
  ``RollingRiskOp`` analyzes the device-resident window of the GPU front ends;
- ``gpu-host``: ``workers.gpu_window`` on host rows, the spawned GPU worker of the host
  front ends;
- ``cpu``: ``workers.cpu_window``, the CPU worker of CPU-only runs, with its BLAS thread
  tuning.

The two workers get a window's new bars only when it slides from the one they analyzed
last (else the full window), as ``RollingRisk`` and the CPU cross-check send them; compare
mode's validated windows are the same windows on both sides.

Windows hold 60 bars and slide by one bar: the first full window, then ``--windows``
slides (``--large-windows`` above 3,000 symbols); the engine rebuilds every 256 calls, so
rebuild windows are reported separately. ``--phases G`` instead times every window kind of
a live run, ``G`` generations per size as after a start or seek: ``first`` (the first,
20-bar window of a generation, a rebuild), ``growth`` (21 to 60 bars), ``slide1`` to
``slide3`` (one- to three-bar slides of the full window, as superseded windows give),
and ``rebuild`` (a full window after a gap, as every 256th call rebuilds).

``--kinds C`` times every window kind ``C`` times per size and backend (300 or more for a
p99): a cycle is a new generation with one window of each kind, ``first`` (bars 0-19),
``growth`` (0-20), ``rebuild`` (22-81, nothing shared with the window before), ``slide1``
(23-82), ``slide2`` (25-84) and ``slide3`` (28-87), after one untimed warm-up cycle (the
allocation). Each backend runs it back to back, then with ``--cadence-ms`` idle before
every window (270 ms, the live cadence at 75,000 observations/s; 0 skips that pass;
``--cadence-cycles`` sets its cycles). The ``compare`` backend runs compare mode's two
processors: ``RollingRisk`` validating every window on the GPU (as ``gpu``) and
``CpuCrossCheck``'s spawned CPU worker recomputing it, one validated window at a time as
the gate keeps them; it reports both engines' times per kind and checks every pair
(agreement and matched work). The CPU workers are tuned for the universe before the first
window, as the example applications tune them when the manifest arrives. With ``--check`` the
results of the first ``--check-cycles`` cycles of each pass are checked against the NumPy
oracle and, for ``gpu`` and ``gpu-host``, against the ``cpu`` backend's. The run ends with
a table of p50 / p99 per size, pass and kind (also written to ``--table``), and
``verdict`` in the report: every GPU path below the CPU at p50 and p99, or the cells where
it is not.

Returns are synthetic N(0, 1e-3), every symbol eligible, or with ``--returns`` the columns
of a bars ``.npz`` (``returns``, bars x symbols, and ``times``) or of a prepared tape
directory (``tools.synth``; the day's bars at the default bar settings, built with
host ``QuoteBars``; ``--save-returns`` keeps them as such an ``.npz``), from mid-session,
shuffled, and time-shifted copies above its width. ``--gap-ms`` sleeps between windows
(the live cadence at 75,000 obs/s is about 270 ms); ``--compete-us`` adds a Python thread
that takes the interpreter lock for that long every 500 µs. ``--check`` compares every
window of every backend with an explicit NumPy oracle and every GPU result with the CPU
result (``agreement``).

    python -m holoscan_finance.tools.risk_bench --sizes 50,200,2886 --backends gpu,cpu \\
        --returns bars.npz --check --out risk_bench.json
    python -m holoscan_finance.tools.risk_bench --returns data/synth/tape --save-returns bars.npz \\
        --sizes 50 --phases 4 --backends gpu,gpu-host,cpu
    python -m holoscan_finance.tools.risk_bench --kinds 300 --backends gpu,gpu-host,cpu,compare \\
        --returns bars.npz --check --out kinds.json --table kinds.md

Reported per size and backend (and phase or kind): the engine's own ``milliseconds`` (the
dashboard's calculation time) and the wall time of the call (``compare``: the validated
window's round trip, both engines and the worker's interprocess transfer), p50 / p99 / mean
/ max. CuPy is imported only by the GPU backends.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import platform
import sys
import threading
import time

import numpy as np

from ..analytics import workers
from ..analytics.agreement import agreement, compare_results
from ..analytics.rolling import RollingRisk
from ..marketdata.covariance import slide
from ..payloads import BarWindow

BARS = 60
BAR_SECONDS = 5
BACKENDS = ("gpu", "gpu-host", "cpu")
LARGE = 3000  # --large-windows applies above this many symbols
KINDS = (  # --kinds: (kind, first bar, end bar) of one cycle, a new generation
    ("first", 0, 20),  # the first window of a generation: a 20-bar rebuild
    ("growth", 0, 21),  # one-bar growth
    ("rebuild", 22, 82),  # a full window sharing nothing with the one before (every 256th call)
    ("slide1", 23, 83),  # one-, two- and three-bar slides of the full window
    ("slide2", 25, 85),
    ("slide3", 28, 88),
)
KIND_BARS = max(end for _, _, end in KINDS)
KIND_BACKENDS = (*BACKENDS, "compare")
WORK = ("method", "added_bars", "expired_bars", "bars")  # what a window did


def tape_returns(path: str) -> tuple[np.ndarray, np.ndarray]:
    """``(returns, times)`` of every bar of a prepared tape's session, at the default bar
    settings: host ``QuoteBars`` over the tape in order, the bars to the session end
    closed. The first bar has no previous mark (NaN)."""
    from ..marketdata.bars import QuoteBars
    from ..marketdata.records import EVENT_DTYPE, ReplayBatch
    from ..nyq.tape import open_tape
    from ..settings import BarSettings

    manifest, events = open_tape(path)
    bars = QuoteBars(manifest["symbols"], **BarSettings().as_params())
    span = 30 * bars.bar_ns  # per batch: fewer bars than the engine keeps
    stamps = events["timestamp_ns"]
    returns, times = [], []
    end = np.zeros(1, EVENT_DTYPE)  # no kind: closes the bars up to the session end
    end["timestamp_ns"] = manifest["end_ns"]
    lo = 0
    while lo <= len(events):
        if lo < len(events):
            hi = int(np.searchsorted(stamps, int(stamps[lo]) // span * span + span))
            batch = np.asarray(events[lo:hi])
        else:
            hi, batch = lo + 1, end
        before = bars.completed_bars
        bars.observe(ReplayBatch(batch, 1, 0, 0))
        closed = bars.completed_bars - before
        if closed:
            returns += list(bars.rows)[-closed:]
            times += list(bars.times)[-closed:]
        lo = hi
    return np.asarray(returns), np.asarray(times, dtype=np.int64)


@functools.cache
def day_returns(path: str, save: str | None = None) -> np.ndarray:
    """The ``returns`` of a bars ``.npz``, or of a prepared tape directory (``save``: kept
    as such an ``.npz``)."""
    if os.path.isdir(path):
        day, times = tape_returns(path)
        if save:
            np.savez(save, returns=day, times=times)
        return day
    return np.load(path)["returns"]


def returns(symbols: int, bars: int, path: str | None, seed: int, save: str | None = None):
    """``bars x symbols`` returns: synthetic N(0, 1e-3), or columns of ``path``."""
    rng = np.random.default_rng(seed)
    if path is None:
        return rng.normal(0.0, 1e-3, size=(bars, symbols))
    day = day_returns(path, save)
    total, width = day.shape
    start = min(total // 2 - bars // 2, 1800) if total > bars else 0
    out = np.empty((bars, symbols))
    columns = rng.permutation(width)
    for j in range(symbols):
        shift = (j // width) * 937  # a copy above the day's width comes from later bars
        out[:, j] = day[(start + shift + np.arange(bars)) % total, columns[j % width]]
    return out


def summary(values: list[float]) -> dict | None:
    if not values:
        return None
    a = np.asarray(values, float)
    return dict(
        n=len(a),
        p50=float(np.median(a)),
        p99=float(np.percentile(a, 99)),
        mean=float(a.mean()),
        max=float(a.max()),
    )


def oracle(rows: np.ndarray, bar_seconds: int) -> dict:
    """An explicit NumPy computation of the documented statistics of the window ``rows``."""
    var = np.var(rows, axis=0, ddof=1)
    eligible = np.flatnonzero(np.isfinite(var) & (var > 1e-20))
    count = len(eligible)
    want = dict(eligible=eligible.tolist(), count=count)
    if count == 0:
        return want
    x = rows[:, eligible]
    cov = np.atleast_2d(np.cov(x, rowvar=False, ddof=1))
    std = np.sqrt(np.diag(cov))
    top = np.argsort(-var[eligible], kind="stable")[:24]
    probe = np.linspace(0, count - 1, min(32, count), dtype=int)
    want.update(
        vol=std * (252 * 6.5 * 3600 / bar_seconds) ** 0.5,
        heat_index=eligible[top].tolist(),
        heatmap=np.clip(cov[np.ix_(top, top)] / np.outer(std[top], std[top]), -1, 1),
        probes=cov[np.ix_(probe, probe)],
        mean=None,
    )
    if count > 1:
        full = cov / np.outer(std, std)
        want["mean"] = (full.sum() - np.trace(full)) / (count * (count - 1))
    return want


def matches_oracle(want: dict, result: dict) -> bool:
    """``result`` against ``oracle()`` of its window."""
    if result["eligible_indices"] != want["eligible"] or result["eligible"] != want["count"]:
        return False
    if want["count"] == 0:
        return result["annual_vol"] == [] and result["mean_correlation"] is None
    ok = (
        np.allclose(result["annual_vol"], want["vol"], rtol=1e-8, atol=1e-10)
        and result["heatmap_indices"] == want["heat_index"]
        and np.allclose(result["heatmap"], want["heatmap"], rtol=1e-8, atol=1e-8)
        and np.allclose(result["covariance_probes"], want["probes"], rtol=1e-8, atol=1e-14)
    )
    if want["mean"] is None:
        return bool(ok and result["mean_correlation"] is None)
    return bool(ok and abs(result["mean_correlation"] - want["mean"]) <= 1e-9)


def oracle_agrees(rows: np.ndarray, result: dict, bar_seconds: int) -> bool:
    """``result`` against an explicit NumPy computation of the documented statistics."""
    return matches_oracle(oracle(rows, bar_seconds), result)


class Competitor:
    """A Python thread holding the interpreter lock ``busy_us`` every 500 µs."""

    def __init__(self, busy_us: float) -> None:
        self._busy_s, self._stop = busy_us / 1e6, threading.Event()
        sys.setswitchinterval(0.001)  # as app.runner.prepare_process
        self._thread = threading.Thread(target=self._run, name="risk-bench-gil", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            time.sleep(0.0005)
            end = time.perf_counter() + self._busy_s
            while time.perf_counter() < end:
                pass

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()


def phases() -> list[tuple[str, int, int]]:
    """``(phase, first bar, end bar)`` of the windows of one ``--phases`` generation."""
    out = [("first", 0, 20)] + [("growth", 0, end) for end in range(21, BARS + 1)]
    start = 0
    for step, count in ((1, 40), (2, 10), (3, 10)):
        for _ in range(count):
            start += step
            out.append((f"slide{step}", start, start + BARS))
    for _ in range(5):  # nothing shared with the window before: a full rebuild
        start += BARS + 1
        out.append(("rebuild", start, start + BARS))
    return out


PHASE_BARS = max(end for _, _, end in phases())


def caller(backend: str, r: np.ndarray, times: np.ndarray, args):
    """``(call(generation, a, b) -> result, stop)``: bars ``a..b-1`` as one window of
    ``backend``, the way the application hands it over."""
    workers.reset()
    if backend == "gpu":
        import cupy as cp

        cp.cuda.Device(args.gpu_device).use()
        parts = [cp.asarray(row) for row in r]
        cp.cuda.Device(args.gpu_device).synchronize()  # parts are complete, as bars emit them
        risk = RollingRisk(backend="gpu", min_interval_ns=0, gpu_device=args.gpu_device)
        risk.start()

        def call(generation: int, a: int, b: int) -> dict:
            window = BarWindow(tuple(parts[a:b]), times[a:b], generation, b, 0, device=True)
            risk.offer(window)
            return risk.poll().result

        return call, risk.stop
    sent = [None]  # the worker's window: generation, times

    def new_rows(generation: int, a: int, b: int) -> np.ndarray:
        """The window's new bars when it slides from the one sent last, else all of it."""
        step = None
        if sent[0] is not None and sent[0][0] == generation:
            step = slide(sent[0][1], times[a:b])
        sent[0] = (generation, times[a:b])
        return r[a:b] if step is None else r[b - step[1] : b]

    if backend == "gpu-host":

        def call(generation: int, a: int, b: int) -> dict:
            rows = new_rows(generation, a, b)
            return workers.gpu_window(
                rows, times[a:b], generation, BAR_SECONDS, args.gpu_device, BARS
            )

    else:
        # Tuned for the universe before its first window, as the manifest has it tuned.
        workers.tune_cpu(r.shape[1], BARS, BAR_SECONDS, args.cpu_threads)

        def call(generation: int, a: int, b: int) -> dict:
            rows = new_rows(generation, a, b)
            return workers.cpu_window(
                rows, times[a:b], generation, BAR_SECONDS, args.cpu_threads, BARS
            )

    return call, lambda: None


def compare_caller(r: np.ndarray, times: np.ndarray, args):
    """``(call(generation, a, b) -> RiskComparison, stop)``: compare mode's processors, the
    GPU ``RollingRisk`` validating every window and ``CpuCrossCheck`` recomputing it in its
    spawned worker (tuned for the universe first), one validated window at a time."""
    import cupy as cp

    from ..analytics.crosscheck import CpuCrossCheck
    from ..control.gate import CrossCheckGate

    workers.reset()
    cp.cuda.Device(args.gpu_device).use()
    parts = [cp.asarray(row) for row in r]
    cp.cuda.Device(args.gpu_device).synchronize()
    gate = CrossCheckGate()
    risk = RollingRisk(
        backend="gpu",
        min_interval_ns=0,
        gpu_device=args.gpu_device,
        cross_check=True,
        gate=gate,
        full_window_bars=BARS,
    )
    xc = CpuCrossCheck(bar_seconds=BAR_SECONDS, cpu_threads=args.cpu_threads, full_window_bars=BARS)
    risk.start()
    xc.start()
    xc.prepare(r.shape[1])  # as the manifest has it tuned
    while xc.tuning_pending:
        xc.poll()
        time.sleep(0.001)

    def call(generation: int, a: int, b: int):
        risk.offer(BarWindow(tuple(parts[a:b]), times[a:b], generation, b, 0, device=True))
        update = risk.poll()
        xc.submit(update)
        while (comparison := xc.poll()) is None:
            time.sleep(0.0002)
        if comparison.error:
            raise RuntimeError(f"the CPU cross-check failed: {comparison.error}")
        gate.release()  # as publishing the comparison does
        return comparison

    def stop() -> None:
        risk.stop()
        xc.stop()

    return call, stop


def timed(call, backend: str, args, gap_ms: float | None = None):
    """``call`` wrapped: the gap before it (``gap_ms``, else ``--gap-ms``), an NVTX range
    (GPU backends), and its wall time in ms."""
    nvtx = None
    if args.nvtx and backend != "cpu":
        import cupy as cp

        nvtx = cp.cuda.nvtx
    gap = args.gap_ms if gap_ms is None else gap_ms

    def run(k: int, *window: int) -> tuple[dict, float]:
        if k and gap:
            time.sleep(gap / 1e3)
        if nvtx:
            nvtx.RangePush(f"window {k}")
        started = time.perf_counter()
        result = call(*window)
        wall = (time.perf_counter() - started) * 1e3
        if nvtx:
            nvtx.RangePop()
        return result, wall

    return run


def run_backend(backend: str, r: np.ndarray, times: np.ndarray, slides: int, args) -> tuple:
    """Every window of one backend: ``(stats, results)``; results only with ``--check``."""
    generation = int(time.monotonic_ns())  # a fresh generation: the workers start over
    call, stop = caller(backend, r, times, args)
    run = timed(call, backend, args)
    first = None
    engine_inc, wall_inc, engine_rebuild, results = [], [], [], []
    for k in range(slides + 1):
        result, wall = run(k, generation, k, k + BARS)
        if k == 0:
            first = dict(engine_ms=result["milliseconds"], wall_ms=wall, method=result["method"])
        elif result["method"] == "incremental":
            engine_inc.append(result["milliseconds"])
            wall_inc.append(wall)
        else:
            engine_rebuild.append(result["milliseconds"])
        if args.check:
            results.append(result)
    stop()
    stats = dict(
        eligible=result["eligible"],
        first_window=first,
        engine_ms=summary(engine_inc),
        wall_ms=summary(wall_inc),
        rebuild_engine_ms=summary(engine_rebuild),
    )
    if backend == "cpu":
        stats["threads"] = result["threads"]
        stats["thread_trials_ms"] = result["cpu_thread_trials_ms"]
    return stats, results


def run_phases(backend: str, r: np.ndarray, times: np.ndarray, args) -> tuple:
    """``--phases`` generations of one backend: ``(stats per phase, results)``. The first
    generation's first window allocates the universe's workspaces and is reported apart
    (``allocate``); results (``--check``) of the first generation only."""
    call, stop = caller(backend, r, times, args)
    run = timed(call, backend, args)
    engine, wall, methods, results = {}, {}, {}, []
    k = 0
    for g in range(args.phases):
        generation = int(time.monotonic_ns())
        for phase, a, b in phases():
            result, w = run(k, generation, a, b)
            k += 1
            name = "allocate" if (g, phase) == (0, "first") else phase
            engine.setdefault(name, []).append(result["milliseconds"])
            wall.setdefault(name, []).append(w)
            methods.setdefault(name, set()).add(result["method"])
            if args.check and g == 0:
                results.append(result)
    stop()
    stats = {
        name: dict(
            engine_ms=summary(engine[name]),
            wall_ms=summary(wall[name]),
            methods=sorted(methods[name]),
        )
        for name in engine
    }
    if backend == "cpu":
        stats["threads"] = result["threads"]
    return stats, results


def run_kinds(
    backend: str, r: np.ndarray, times: np.ndarray, cycles: int, gap_ms: float, args
) -> tuple:
    """One ``--kinds`` pass of one backend: an untimed warm-up cycle, then ``cycles`` cycles
    of ``KINDS`` with ``gap_ms`` idle before every window. ``(stats, kept)``: per kind the
    engine milliseconds (``compare``: the GPU's and its CPU twin's) and the call's, and the
    work every window did; the results of the first ``--check-cycles`` cycles (``--check``)."""
    compare = backend == "compare"
    call, stop = compare_caller(r, times, args) if compare else caller(backend, r, times, args)
    run = timed(call, backend, args, gap_ms)
    sides = ("gpu", "cpu") if compare else ("engine",)
    engine = {side: {kind: [] for kind, _, _ in KINDS} for side in sides}
    wall = {kind: [] for kind, _, _ in KINDS}
    work = {kind: set() for kind, _, _ in KINDS}
    warmup, kept = {}, {side: [] for side in sides}
    agreed = matched = 0
    k = 0
    try:
        for cycle in range(cycles + 1):
            generation = int(time.monotonic_ns())  # a new generation: every engine resets
            for kind, a, b in KINDS:
                out, ms = run(k, generation, a, b)
                k += 1
                results = (out.update.result, out.cpu) if compare else (out,)
                if cycle == 0:
                    warmup[kind] = dict(
                        engine_ms=[x["milliseconds"] for x in results],
                        wall_ms=ms,
                        method=results[0]["method"],
                    )
                    continue
                for side, x in zip(sides, results):
                    engine[side][kind].append(x["milliseconds"])
                wall[kind].append(ms)
                work[kind].add(tuple(results[0][key] for key in WORK))
                if compare:
                    agreed += out.agreement
                    matched += out.matched_work
                if args.check and cycle <= args.check_cycles:
                    for side, x in zip(sides, results):
                        kept[side].append(x)
    finally:
        stop()
    unit = {side: "engine_ms" if side == "engine" else f"{side}_ms" for side in sides}
    stats = dict(
        kinds={
            kind: dict(
                **{unit[side]: summary(engine[side][kind]) for side in sides},
                wall_ms=summary(wall[kind]),
                work=[dict(zip(WORK, w)) for w in sorted(work[kind])],
            )
            for kind, _, _ in KINDS
        },
        warmup=warmup,
        eligible=results[0]["eligible"],
    )
    cpu = results[-1] if backend in ("cpu", "compare") else None
    if cpu is not None:
        stats.update(threads=cpu["threads"], cpu_tunings=cpu["cpu_tunings"])
    if compare:
        n = cycles * len(KINDS)
        stats.update(agreement=f"{agreed}/{n}", matched_work=f"{matched}/{n}")
    return stats, kept


def kinds_check(r: np.ndarray, kept: dict) -> dict:
    """``--kinds --check`` of one pass: every kept result against the NumPy oracle of its
    window, and ``gpu``/``gpu-host`` against the ``cpu`` backend (agreement and matched
    work); ``kept[backend][side]`` lists the results in ``KINDS`` order, cycle by cycle."""
    want = {(a, b): oracle(r[a:b], BAR_SECONDS) for _, a, b in KINDS}
    check = {}
    for backend, sides in kept.items():
        for side, results in sides.items():
            name = backend if side == "engine" else f"{backend}-{side}"
            windows = [(a, b) for _, a, b in KINDS] * (len(results) // len(KINDS))
            ok = sum(matches_oracle(want[w], x) for w, x in zip(windows, results))
            check[f"{name}_vs_oracle"] = f"{ok}/{len(results)}"
        if backend in ("gpu", "gpu-host") and "cpu" in kept:
            pairs = [compare_results(c, g) for c, g in zip(kept["cpu"]["engine"], sides["engine"])]
            check[f"{backend}_vs_cpu"] = f"{sum(a and m for a, m, _ in pairs)}/{len(pairs)}"
    return check


def kinds_cells(entry: dict) -> list[tuple[str, str, str, bool]]:
    """``(pass, kind, pair, below)`` of one size: each GPU path (``gpu``, ``gpu-host``
    against ``cpu``; ``compare``'s GPU against its CPU twin) below the CPU at p50 and p99."""
    cells = []
    for name, runs in entry["passes"].items():
        pairs = [(b, b, "engine_ms", "cpu", "engine_ms") for b in ("gpu", "gpu-host")]
        pairs.append(("compare", "compare", "gpu_ms", "compare", "cpu_ms"))
        for kind, _, _ in KINDS:
            for pair, gpu, gpu_unit, cpu, cpu_unit in pairs:
                if gpu not in runs or cpu not in runs:
                    continue
                g, c = runs[gpu]["kinds"][kind][gpu_unit], runs[cpu]["kinds"][kind][cpu_unit]
                below = g["p50"] < c["p50"] and g["p99"] < c["p99"]
                cells.append((name, kind, pair, below))
    return cells


def kinds_table(report: dict) -> str:
    """The ``--kinds`` results as a Markdown table: engine milliseconds, p50 / p99, per size,
    pass and kind; the last column names the GPU paths not below the CPU at p50 and p99."""
    columns = [(b, "engine_ms", b) for b in BACKENDS] + [
        ("compare", "gpu_ms", "compare GPU"),
        ("compare", "cpu_ms", "compare CPU"),
    ]
    lines = []
    for size, entry in report["sizes"].items():
        cells = kinds_cells(entry)
        for name, runs in entry["passes"].items():
            shown = [c for c in columns if c[0] in runs]
            if not lines:
                head = ["symbols (eligible)", "pass", "kind", *(c[2] for c in shown), "GPU < CPU"]
                lines += ["| " + " | ".join(head) + " |", "|" + " --- |" * len(head)]
            eligible = next(iter(runs.values()))["eligible"]
            for kind, _, _ in KINDS:
                row = [f"{size} ({eligible})", name, kind]
                for backend, unit, _ in shown:
                    st = runs[backend]["kinds"][kind][unit]
                    row.append(f"{st['p50']:.3f} / {st['p99']:.3f}")
                failing = [
                    pair for p, k, pair, below in cells if (p, k) == (name, kind) and not below
                ]
                compared = any((p, k) == (name, kind) for p, k, _, _ in cells)
                row.append(("no: " + ", ".join(failing)) if failing else "yes" if compared else "")
                lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict:
    backends = args.backends.split(",")
    allowed = KIND_BACKENDS if args.kinds else BACKENDS
    if not set(backends) <= set(allowed):
        raise SystemExit(f"--backends: choose from {', '.join(allowed)}")
    if args.kinds and (args.phases or args.gap_ms):
        raise SystemExit("--kinds: use --cadence-ms, not --phases or --gap-ms")
    report = dict(
        args=vars(args),
        host=dict(machine=platform.machine(), python=platform.python_version()),
        sizes={},
    )
    competitor = Competitor(args.compete_us) if args.compete_us else None
    try:
        for size in (int(s) for s in args.sizes.split(",") if s):
            if args.kinds:
                report["sizes"][size] = run_size_kinds(size, backends, args)
                report["verdict"] = verdict(report)
            else:
                report["sizes"][size] = run_size(size, backends, args)
            if args.out:
                with open(args.out, "w") as f:
                    json.dump(report, f, indent=1)
    finally:
        if competitor is not None:
            competitor.stop()
    if args.kinds:
        table = kinds_table(report)
        print(table, flush=True)
        if args.table:
            with open(args.table, "w") as f:
                f.write(table + "\n")
    return report


def run_size(size: int, backends: list[str], args) -> dict:
    """The one-bar slides (or ``--phases``) of every backend at ``size`` symbols."""
    slides = args.large_windows if size > LARGE else args.windows
    bars = PHASE_BARS if args.phases else BARS + slides + 1
    r = returns(size, bars, args.returns, args.seed, args.save_returns)
    times = (1_743_514_200 + BAR_SECONDS * np.arange(len(r), dtype=np.int64)) * 10**9
    times.flags.writeable = False
    entry, outputs = {}, {}
    for backend in backends:
        if args.phases:
            entry[backend], outputs[backend] = run_phases(backend, r, times, args)
        else:
            entry[backend], outputs[backend] = run_backend(backend, r, times, slides, args)
        print(size, backend, json.dumps(entry[backend]), flush=True)
    if args.check:
        if args.phases:
            windows = [r[a:b] for _, a, b in phases()]
        else:
            windows = [r[k : k + BARS] for k in range(slides + 1)]
        check = {}
        for backend, results in outputs.items():
            ok = sum(oracle_agrees(w, x, BAR_SECONDS) for w, x in zip(windows, results))
            check[f"{backend}_vs_oracle"] = f"{ok}/{len(results)}"
            if backend != "cpu" and "cpu" in outputs:
                pairs = zip(outputs["cpu"], results)
                ok = sum(agreement(c, g) for c, g in pairs)
                check[f"{backend}_vs_cpu"] = f"{ok}/{len(results)}"
        entry["check"] = check
        print(size, "check", json.dumps(check), flush=True)
    return entry


def run_size_kinds(size: int, backends: list[str], args) -> dict:
    """``--kinds`` at ``size`` symbols: every backend back to back, then at the cadence."""
    r = returns(size, KIND_BARS, args.returns, args.seed, args.save_returns)
    times = (1_743_514_200 + BAR_SECONDS * np.arange(len(r), dtype=np.int64)) * 10**9
    times.flags.writeable = False
    passes = [("back_to_back", args.kinds, 0.0)]
    if args.cadence_ms:
        cycles = args.cadence_cycles or args.kinds
        passes.append((f"cadence_{args.cadence_ms:g}ms", cycles, args.cadence_ms))
    entry = dict(passes={}, check={})
    for name, cycles, gap_ms in passes:
        runs, kept = {}, {}
        for backend in backends:
            started = time.perf_counter()
            runs[backend], kept[backend] = run_kinds(backend, r, times, cycles, gap_ms, args)
            runs[backend]["seconds"] = time.perf_counter() - started
            brief = {
                kind: {
                    u: [round(v["p50"], 4), round(v["p99"], 4)]
                    for u, v in st.items()
                    if u.endswith("_ms") and u != "wall_ms"
                }
                for kind, st in runs[backend]["kinds"].items()
            }
            print(size, name, backend, json.dumps(brief), flush=True)
        entry["passes"][name] = runs
        if args.check:
            check = kinds_check(r, kept)
            for backend in backends:
                if backend == "compare":
                    check["compare_agreement"] = runs[backend]["agreement"]
                    check["compare_matched_work"] = runs[backend]["matched_work"]
            entry["check"].update({f"{name}:{k}": v for k, v in check.items()})
            print(size, name, "check", json.dumps(check), flush=True)
        del kept
    return entry


def verdict(report: dict) -> dict:
    """Every GPU path below the CPU at p50 and p99, per size, pass and kind: the count of
    cells and the ones where it is not."""
    cells = [
        (size, *cell) for size, entry in report["sizes"].items() for cell in kinds_cells(entry)
    ]
    misses = [dict(size=s, run=p, kind=k, path=pair) for s, p, k, pair, below in cells if not below]
    return dict(cells=len(cells), gpu_below_cpu=len(cells) - len(misses), not_below=misses)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m holoscan_finance.tools.risk_bench",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--sizes", default="50,100,200,500,1000,2886,5000,10000", help="symbols")
    p.add_argument(
        "--backends",
        default="gpu,cpu",
        help=f"comma list of {', '.join(BACKENDS)}; --kinds also compare",
    )
    p.add_argument("--windows", type=int, default=300, help="slides after the first window")
    p.add_argument("--large-windows", type=int, default=40, help=f"slides above {LARGE:,} symbols")
    p.add_argument(
        "--returns", help="bars .npz ('returns') or a prepared tape directory; default synthetic"
    )
    p.add_argument("--save-returns", help="--returns is a tape: keep its bars as this .npz")
    p.add_argument(
        "--phases", type=int, default=0, help="generations timing every window kind (see above)"
    )
    p.add_argument(
        "--kinds", type=int, default=0, help="cycles timing every window kind (see above); 300+"
    )
    p.add_argument(
        "--cadence-ms",
        type=float,
        default=270.0,
        help="--kinds: idle time before every window of the second pass; 0 skips it",
    )
    p.add_argument(
        "--cadence-cycles", type=int, default=0, help="--kinds: cycles of that pass; 0: --kinds"
    )
    p.add_argument(
        "--check-cycles",
        type=int,
        default=50,
        help="--kinds --check: the cycles of each pass whose results are checked",
    )
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--gap-ms", type=float, default=0.0, help="idle time between windows")
    p.add_argument("--compete-us", type=float, default=0.0, help="GIL load every 500 µs")
    p.add_argument("--cpu-threads", type=int, default=0, help="cpu BLAS threads; 0 tunes")
    p.add_argument("--gpu-device", type=int, default=0)
    p.add_argument("--check", action="store_true", help="oracle and CPU/GPU agreement")
    p.add_argument("--nvtx", action="store_true", help="one NVTX range per GPU window")
    p.add_argument("--out", help="JSON report")
    p.add_argument("--table", help="--kinds: also write the table (Markdown) here")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    report = run(args)
    if args.check:
        failed = [
            (size, name, value)
            for size, entry in report["sizes"].items()
            for name, value in entry["check"].items()
            if value.split("/")[0] != value.split("/")[1]
        ]
        if failed:
            print("check failed:", failed, flush=True)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
