# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verdicts on a finished run: paired bar windows, final bar state, reference bars, gates.

``BarsChecker`` collects the windows of two ``QuoteBarsOp`` backends fed one stream (the
``market_data_bench --check`` fan-out) and pairs them by ``(generation, completed_bars)``.
Times and NaN patterns must be identical and finite returns within 1 ULP (CUDA vs NumPy
``log``), the 4.3 ``gpu_validate --bars`` criteria. The callbacks only append under a lock,
so collecting costs a lane one list append per window; the comparisons, with the device
copies they need, run after the run. ``final_state_equal`` compares both backends' final
counters and symbol state, and ``bench_verdict`` applies the bench gates to its report.
"""

from __future__ import annotations

import dataclasses
import numbers
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..payloads import BarCounters, BarWindow
from .compare import to_host, ulps

if TYPE_CHECKING:
    from ..telemetry.stats import BarsStats
    from .compare import HostReference

MAX_ULPS = 1  # CUDA vs NumPy log; GPU_INGEST measured no difference above 1 ULP
LOSS_KEYS = (
    "sequence_gaps",
    "ring_dropped",
    "nic_dropped",
    "stale_packets",
    "malformed_packets",
    "decode_malformed",
)


def _key(window: BarWindow) -> tuple[int, int]:
    return window.generation, window.completed_bars


def _new_parts(window: BarWindow, newest: dict[int, int]) -> range | list[int]:
    """Indices of the window's bars that are newer than the bars already compared in its
    generation (every bar for the first window of a generation), so each bar is compared
    once even when several close between two windows."""
    seen = newest.get(window.generation)
    newest[window.generation] = int(window.times[-1])
    if seen is None:
        return range(len(window.parts))
    return [i for i, t in enumerate(window.times) if t > seen]


class _ReturnsDiff:
    """Accumulated return comparisons of bar pairs."""

    def __init__(self) -> None:
        self.bars = 0
        self.times_equal = True
        self.nan_pattern_equal = True
        self.max_ulps = 0

    def add(self, a: np.ndarray, b: np.ndarray) -> None:
        self.bars += 1
        if a.shape != b.shape:
            self.nan_pattern_equal = False
            return
        fa, fb = np.isfinite(a), np.isfinite(b)
        if not (np.array_equal(fa, fb) and np.array_equal(np.isnan(a), np.isnan(b))):
            self.nan_pattern_equal = False
        both = fa & fb
        if both.any():
            self.max_ulps = max(self.max_ulps, int(ulps(a[both], b[both]).max()))

    def result(self) -> dict:
        return dict(
            times_equal=self.times_equal,
            nan_pattern_equal=self.nan_pattern_equal,
            max_ulps=self.max_ulps,
        )


class BarsChecker:
    """Pairs the windows of two bar backends by ``(generation, completed_bars)``.

    ``left`` and ``right`` are ``CallbackSinkOp`` callbacks on two lanes; ``count_left`` and
    ``count_right`` feed ``DrainedCondition``'s collectors."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.left_windows: list[BarWindow] = []
        self.right_windows: list[BarWindow] = []

    def left(self, window: BarWindow) -> None:
        with self._lock:
            self.left_windows.append(window)

    def right(self, window: BarWindow) -> None:
        with self._lock:
            self.right_windows.append(window)

    def count_left(self) -> int:
        return len(self.left_windows)

    def count_right(self) -> int:
        return len(self.right_windows)

    def summary(self) -> dict:
        """Compare every pair: times fully, and each bar once (its first appearance).

        ``times_equal`` also requires both sides to hold the same window keys, once each."""
        with self._lock:
            left, right = list(self.left_windows), list(self.right_windows)
        twins = {_key(w): w for w in right}
        diff = _ReturnsDiff()
        diff.times_equal = len(twins) == len(right) == len(left) and twins.keys() == {
            _key(w) for w in left
        }
        compared = 0
        newest: dict[int, int] = {}
        for w in left:
            twin = twins.get(_key(w))
            if twin is None:
                continue
            compared += 1
            if not np.array_equal(w.times, twin.times):
                diff.times_equal = False
                continue
            for i in _new_parts(w, newest):
                diff.add(to_host(w.parts[i]), to_host(twin.parts[i]))
        return dict(
            windows_left=len(left),
            windows_right=len(right),
            windows_compared=compared,
            **diff.result(),
        )

    def compare_reference(self, ref: HostReference, windows: Iterable[BarWindow]) -> dict:
        """Compare ``windows`` with the bars ``ref`` recorded from the same capture.

        A window's times must be the reference's consecutive closed bars ending at its newest
        bar; each bar's returns are compared once."""
        times, returns = ref.closed_bars()
        index = {int(t): i for i, t in enumerate(times)}
        generation = ref.directory.generation
        diff = _ReturnsDiff()
        newest: dict[int, int] = {}
        for w in windows:
            k = len(w.times)
            j = index.get(int(w.times[-1]), -1) if k else -1
            first = j + 1 - k
            if (
                w.generation != generation
                or j < 0
                or first < 0
                or not np.array_equal(times[first : j + 1], w.times)
            ):
                diff.times_equal = False
                continue
            for i in _new_parts(w, newest):
                diff.add(to_host(w.parts[i]), returns[first + i])
        return dict(closed_bars_compared=diff.bars, **diff.result())


def _same_values(a: np.ndarray, b: np.ndarray) -> bool:
    """Bitwise equal, with any NaN equal to any NaN (payload bits may differ by backend)."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype.kind != "f":
        return bool(np.array_equal(a, b))
    nan = np.isnan(a)
    bits = f"i{a.dtype.itemsize}"
    return bool(
        np.array_equal(nan, np.isnan(b)) and np.array_equal(a[~nan].view(bits), b[~nan].view(bits))
    )


def final_state_equal(left: BarsStats, right: BarsStats) -> bool:
    """The final ``BarCounters`` agree except the wall-clock fields, and the final symbol
    snapshots (published by ``QuoteBarsOp.stop()``) agree bit for bit; False if either side
    has no counters or no snapshot."""
    lc, rc, ls, rs = left.counters, right.counters, left.symbol_snapshot, right.symbol_snapshot
    if lc is None or rc is None or ls is None or rs is None:
        return False
    timing = set(BarCounters.TIMING_FIELDS)
    for f in dataclasses.fields(BarCounters):
        if f.name not in timing and getattr(lc, f.name) != getattr(rc, f.name):
            return False
    return tuple(ls.symbols) == tuple(rs.symbols) and all(
        _same_values(getattr(ls, n), getattr(rs, n))
        for n in ("mid", "trade", "volume", "quote_time")
    )


@dataclass(frozen=True)
class Gates:
    """The ``market_data_bench`` gates (built from its CLI by ``app.bench.normalize_bench``)."""

    require_lossless: bool = False
    require_realtime: bool = False
    check: bool = False
    min_msgs_per_s: float = 0.0  # 0 = off
    max_lag_p99_ms: float = 0.0  # 0 = off


_MISSING = object()


def _lookup(report: dict, path: str) -> object:
    node: object = report
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return _MISSING
        node = node[part]
    return node


def _number(v: object) -> bool:
    # numbers.Real also admits NumPy scalars: an np.int64 count must not fail its gate.
    return isinstance(v, numbers.Real) and not isinstance(v, bool)


def bench_verdict(report: dict, gates: Gates) -> tuple[bool, list[str]]:
    """Apply ``gates`` to a bench report: ``(ok, failure messages)``. A missing or
    non-numeric value fails its gate; this never raises."""
    failures: list[str] = []

    def require(path: str, ok: Callable[[object], bool], why: str) -> None:
        value = _lookup(report, path)
        if value is _MISSING:
            failures.append(f"{path} is missing from the report ({why})")
        elif not ok(value):
            failures.append(f"{path} = {value!r}: {why}")

    def zero(v: object) -> bool:
        return _number(v) and v == 0

    def true(v: object) -> bool:
        return v is True

    def at_most(limit: float) -> Callable[[object], bool]:
        return lambda v: _number(v) and v <= limit

    def at_least(limit: float) -> Callable[[object], bool]:
        return lambda v: _number(v) and v >= limit

    require("termination", lambda v: v == "STOPPED", "the run must end STOPPED")
    require("stop_reason", lambda v: v == "drained", "the run must stop because it drained")
    if gates.require_lossless:
        for key in LOSS_KEYS:
            require(key, zero, "--require-lossless")
    if gates.require_realtime:  # hold-and-retry is lossless, but then the graph fell behind
        for key in ("retries", "ring_waits"):
            require(key, zero, "--require-realtime: the graph must never hold a burst")
    if gates.check:
        for prefix, count in (
            ("check", "windows_compared"),
            ("check.reference", "closed_bars_compared"),
        ):
            require(f"{prefix}.{count}", at_least(1), "--check must compare something")
            require(f"{prefix}.times_equal", true, "--check")
            require(f"{prefix}.nan_pattern_equal", true, "--check")
            require(f"{prefix}.max_ulps", at_most(MAX_ULPS), f"--check allows {MAX_ULPS} ULP")
        require("check.final_state_equal", true, "--check")
        require("check.reference.counters_equal", true, "--check")
        require("check.reference.decode_counts_equal", true, "--check")
    if gates.min_msgs_per_s:
        require("msgs_per_s", at_least(gates.min_msgs_per_s), "--min-msgs-per-s")
    if gates.max_lag_p99_ms:
        require("lag_p99_ms", at_most(gates.max_lag_p99_ms), "--max-lag-p99-ms")
    return not failures, failures
