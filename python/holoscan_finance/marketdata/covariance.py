# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rolling volatility, covariance and correlation of a bar window, on the CPU or the GPU.

One algorithm on both backends (docs/METHODOLOGY.md, "Statistics"):

- the augmented second-moment matrix ``C = sum_t [f(r_t); 1][f(r_t); 1]^T`` of the window,
  ``(N+1) x (N+1)`` with NaN returns as 0 (``f``), so row and column ``N`` hold the column
  totals;
- one signed rank-k update per window: the expired bars with sign -1, then the new bars
  with +1, for a window that slides (``slide``); any other window, and every 256th call,
  rebuilds ``C`` from the window, which bounds the drift of the subtractions;
- a recomputation of ``C`` from the window when the subtractions cancelled: an eligible
  column whose largest squared return since the last rebuild exceeds ``CANCEL`` (1e5)
  times its diagonal of ``C`` (a large move left a quiet window), so the rounding left by
  the large move never dominates the column's statistics;
- eligibility from the window itself: finite returns throughout, a sample variance
  (``np.var(rows, axis=0, ddof=1)``) above 1e-20 and a positive variance from ``C``, so
  missing returns are excluded, never treated as zero, and subtraction residue never makes
  a name eligible;
- volatilities from ``diag(C)``, the 24 x 24 heatmap (the highest-variance names) and the
  32 x 32 covariance probes from small blocks of ``C``, and the mean correlation as one
  matrix-vector product, ``((d.Cd - (d.t)^2/n)/(n-1) - count)/(count (count-1))`` with
  ``d = 1/sigma`` on eligible names. No ``N_e x N_e`` matrix is built.

CPU: NumPy and SciPy BLAS (a signed ``dgemm`` update, ``gemv``). GPU: CuPy kernels
(``_risk_kernels``) on the engine's own non-blocking CUDA stream, with every buffer
allocated for the universe width and the window capacity, so a window allocates nothing:
it writes the window's shape and its new bars' addresses to pinned memory and launches one
CUDA graph (a control-block copy, seven kernels and one packed result to pinned host
memory), whatever the window's kind. Selection and finishing round exactly as the CPU's
NumPy expressions; ``C`` itself differs between the backends only by the summation order
and fused multiply-add rounding of the rank-k update (SciPy ``dgemm`` on the CPU, a CUDA
kernel on the GPU), which the agreement tolerances cover.

The interpreter's automatic garbage collection is held off while a window is timed, on
both backends (any thread of the process): a collection starts at whichever allocation
crosses the collector's threshold, and one that started inside a window added its whole
pause to that window's time, which made the slowest windows of a kind. A collection that
falls due meanwhile runs right after the window instead.
"""

from __future__ import annotations

import gc
import threading
import time

import numpy as np

from ._risk_kernels import CANCEL, CHECK, CONTROL, HEATMAP, PROBES, REBUILD, RECOMPUTE, SHIFT

REBUILD_EVERY = 256  # calls between rebuilds of C
MIN_CAPACITY = 64  # bars a window buffer holds at least
_INVALID = "need >=2 increasing bar timestamps and finite-or-NaN returns"
_CHANGED = "completed bars changed; reset engine after seek"


def slide(old_times: np.ndarray, times: np.ndarray) -> tuple[int, int] | None:
    """``(expired, added)`` bars when ``times`` continues ``old_times`` (a prefix expires,
    the rest is shared and new bars follow), else None."""
    if times[-1] < old_times[-1]:
        return None
    expired = int(np.searchsorted(old_times, times[0]))
    shared = len(old_times) - expired
    if (
        shared <= 0
        or shared > len(times)
        or not np.array_equal(old_times[expired:], times[:shared])
    ):
        return None
    return expired, len(times) - shared


class CovarianceEngine:
    """Incremental window statistics; ``update`` returns one result dict per window.

    ``reset()`` forgets the statistics and keeps the workspaces (a new feed generation).
    The GPU backend runs on the device that is current in ``update`` (the constructor
    makes ``gpu_device`` current, as the callers keep it). ``window_bars``: the full
    window's length, so the workspaces allocated at a universe's first window already hold
    every growth window after it (a longer window grows them, keeping the statistics).
    """

    def __init__(self, backend="cpu", *, bar_seconds=5, gpu_device=0, window_bars=None):
        if backend not in ("cpu", "gpu"):
            raise ValueError("backend must be cpu or gpu")
        self.backend, self.bar_seconds, self.gpu_device = backend, bar_seconds, gpu_device
        self.window_bars = window_bars or 0
        self.annualize = (252 * 6.5 * 3600 / bar_seconds) ** 0.5
        self.xp = np
        if backend == "gpu":
            import cupy as cp

            cp.cuda.Device(gpu_device).use()
            self.xp = cp
            self._core = _GpuCore(cp, gpu_device, self.annualize)
        else:
            self._core = _CpuCore(self.annualize)
        self.times = self.parts = None
        self.calls = 0

    @property
    def width(self) -> int | None:
        """The universe the workspaces are allocated for (None before the first window)."""
        return self._core.width

    def reset(self) -> None:
        """Forget the window and the statistics: the next update rebuilds."""
        self.times = self.parts = None
        self.calls = 0

    def update(
        self,
        rows,
        times,
        *,
        parts=None,
        force_rebuild=False,
        return_matrix=False,
        observed_variance=None,
    ):
        """One immutable window; the timer includes the input copies, the statistics and the
        transfer of the compact results (GPU: synchronized before it returns), and holds
        automatic garbage collection off meanwhile (see the module docstring).

        ``rows`` is the ``(bars, N)`` window (NumPy, or CuPy on the GPU backend), or None
        with ``parts``: the per-bar return vectors, oldest first, which are never modified
        (``BarWindow.parts``). Shared bars of a sliding window are then proven by identity,
        and only the new bars are copied in; ``rows`` is compared with the previous window
        instead. Device parts must be complete when passed (their producer synchronized, as
        ``QuoteBarsOp`` does before it emits); a device ``rows`` array, a device
        ``observed_variance`` and parts the engine converts to float64 are read after the
        caller's current stream. ``observed_variance`` optionally supplies the
        per-instrument sample variance used for eligibility and display order, so a
        separately computed CPU comparison uses identical coordinates.
        """
        times = np.asarray(times, dtype=np.int64)
        n = len(times)
        if parts is None and not hasattr(rows, "__cuda_array_interface__"):
            rows = np.asarray(rows, dtype=np.float64)
        source = parts if parts is not None else rows
        width = _width(source, n)
        if width is None or n < 2 or np.any(np.diff(times) <= 0):
            raise ValueError(_INVALID)
        _hold_collector()
        try:
            started = time.perf_counter()
            out, rebuild, expired, added = self._window(
                source, parts, times, n, width, force_rebuild, observed_variance
            )
            elapsed_ms = (time.perf_counter() - started) * 1000
        finally:
            _release_collector()
        count = out["count"]
        matrix = self._core.matrix(out["indices"], n) if return_matrix else None
        return {
            "backend": self.backend,
            "milliseconds": elapsed_ms,
            "method": "rebuild" if rebuild else "incremental",
            "added_bars": added,
            "expired_bars": expired,
            "bars": n,
            "universe": width,
            "eligible_indices": out["indices"].tolist(),
            "eligible": count,
            "pairs": count * (count - 1) // 2,
            "annual_vol": out["vol"].tolist(),
            "mean_correlation": out["mean_correlation"] if count > 1 else None,
            "heatmap_indices": out["heat_index"].tolist(),
            "heatmap": out["heatmap"].tolist(),
            "covariance_probes": out["probes"].tolist(),
            "matrix": matrix,
        }

    def _window(self, source, parts, times, n, width, force_rebuild, observed_variance):
        """The timed part of ``update``: ``(out, rebuild, expired, added)``."""
        core = self._core
        if core.width != width:
            core.allocate(width, max(MIN_CAPACITY, n, self.window_bars))
            self.reset()
        elif n > core.capacity:  # longer than the hint: grow, keeping C and the window
            core.grow(max(2 * core.capacity, n), 0 if self.times is None else len(self.times))
        step = None
        if not force_rebuild and self.times is not None and self.calls % REBUILD_EVERY:
            step = slide(self.times, times)
        rebuild = step is None
        expired, added = (0, n) if rebuild else step
        shared = n - added
        proven = (
            not rebuild
            and parts is not None
            and self.parts is not None
            and all(a is b for a, b in zip(self.parts[expired:], parts[:shared]))
        )
        if observed_variance is not None:
            observed_variance = core.asarray(observed_variance, width)
        out = core.run(
            source, parts is not None, n, expired, added, rebuild, proven, observed_variance
        )
        if out is None:  # the shared bars differ: nothing changed
            raise ValueError(_CHANGED)
        if out["invalid"]:
            self.reset()  # C holds the non-finite input: the next call rebuilds
            raise ValueError(_INVALID)
        if out["recomputed"]:
            rebuild, expired, added = True, 0, n
        core.advance()
        self.times = times.copy()
        self.parts = tuple(parts) if parts is not None else None
        self.calls += 1
        return out, rebuild, expired, added


_collector = threading.Lock()
_holds = 0  # windows being timed now, in any thread of the process
_resume = False  # the collector was on when the first of them began


def _hold_collector() -> None:
    """Hold automatic garbage collection off while a window is timed (re-entrant across
    threads: the last ``_release_collector`` turns it back on if it was on)."""
    global _holds, _resume
    with _collector:
        if _holds == 0:
            _resume = gc.isenabled()
            gc.disable()
        _holds += 1


def _release_collector() -> None:
    global _holds
    with _collector:
        _holds -= 1
        if _holds == 0 and _resume:
            gc.enable()


def _width(source, n: int) -> int | None:
    """The universe of ``rows`` (2-D) or ``parts`` (1-D vectors of one length), else None."""
    if source is None or len(source) != n:
        return None
    if isinstance(source, (tuple, list)):
        widths = {p.shape for p in source}
        return widths.pop()[0] if len(widths) == 1 and len(next(iter(widths))) == 1 else None
    return source.shape[1] if source.ndim == 2 else None


def _finish(c, observed, n, width, annualize, peak=None):
    """Selection and finishing in NumPy from ``C`` and the observed variance; the GPU kernels
    reproduce these expressions operation for operation. With ``peak`` (the columns' largest
    squared returns since the last rebuild): ``{"cancel": True}`` when an eligible column
    cancelled (``CANCEL``), so the caller recomputes ``C`` from the window."""
    totals = c[:width, width]
    if not (np.isfinite(c.diagonal()).all() and np.isfinite(totals).all()):
        return {"invalid": True}
    diagonal = c.diagonal()[:width]
    variance = np.maximum((diagonal - totals**2 / n) / (n - 1), 0)
    moving = np.isfinite(observed) & (observed > 1e-20)
    if peak is not None and (moving & (peak > CANCEL * diagonal)).any():
        return {"invalid": False, "cancel": True}
    eligible = moving & (variance > 0)
    indices = np.flatnonzero(eligible)
    count = len(indices)
    sd = np.sqrt(variance)
    std = sd[indices]
    order = np.argsort(-observed[indices], kind="stable")
    heat = indices[order[:HEATMAP]]
    probe = indices[np.linspace(0, max(0, count - 1), min(PROBES, count), dtype=int)]
    mean = None
    if count > 1:
        d = np.zeros(width + 1)
        d[indices] = 1.0 / std
        y = c.T @ d
        s1, s2 = float(np.dot(d[:width], y[:width])), float(y[width])
        mean = ((s1 - s2 * s2 / n) / (n - 1) - count) / (count * (count - 1))

    def block(u):
        cov = c[np.ix_(u, u)].T  # C.T[u[a], u[b]], as the covariance gather always read it
        cov = cov - np.outer(totals[u] / n, totals[u])
        cov /= n - 1
        return cov

    heatmap = block(heat)
    correlation = np.outer(sd[heat], sd[heat])
    np.divide(heatmap, correlation, out=correlation)
    np.clip(correlation, -1, 1, out=correlation)
    return {
        "invalid": False,
        "cancel": False,
        "count": count,
        "indices": indices,
        "vol": std * annualize,
        "mean_correlation": mean,
        "heat_index": heat,
        "heatmap": correlation,
        "probes": block(probe),
    }


def _compact_covariance(c, indices, n, width):
    """The eligible names' covariance matrix (tests only: ``return_matrix``)."""
    totals = c[:width, width][indices]
    cov = c.T[np.ix_(indices, indices)] - np.outer(totals / n, totals)
    cov /= n - 1
    return cov


def _same(old: np.ndarray, new: np.ndarray) -> bool:
    """Equal returns, NaN equal to NaN: a bitwise compare first, values only if needed."""
    if np.array_equal(old.view(np.uint64), new.view(np.uint64)):
        return True
    return bool(((old == new) | (np.isnan(old) & np.isnan(new))).all())


class _CpuCore:
    """NumPy/SciPy mechanics: ``C`` in Fortran order, two host window buffers in turn."""

    def __init__(self, annualize: float) -> None:
        self.annualize = annualize
        self.width = None
        self.capacity = 0

    def allocate(self, width: int, capacity: int) -> None:
        self.width, self.capacity = width, capacity
        self.c = np.zeros((width + 1, width + 1), order="F")
        self.windows = [np.zeros((capacity, width)), np.zeros((capacity, width))]
        self.peak = np.zeros(width)
        self.current = 0

    def grow(self, capacity: int, held: int) -> None:
        """Window buffers of ``capacity`` bars holding the current window's ``held`` bars."""
        windows = [np.zeros((capacity, self.width)), np.zeros((capacity, self.width))]
        windows[self.current][:held] = self.windows[self.current][:held]
        self.windows, self.capacity = windows, capacity

    def asarray(self, values, width: int) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (width,):
            raise ValueError("observed_variance needs one value per instrument")
        return values

    def run(self, source, is_parts, n, expired, added, rebuild, proven, observed):
        old, new = self.windows[self.current], self.windows[1 - self.current]
        shared = n - added
        if proven:
            new[:shared] = old[expired : expired + shared]
            _copy_rows(new, source, shared, n)
        else:
            _copy_rows(new, source, 0, n)
            if not rebuild and not _same(old[expired : expired + shared], new[:shared]):
                return None
        if observed is None:
            with np.errstate(invalid="ignore"):  # an infinite return: rejected below
                observed = np.var(new[:n], axis=0, ddof=1)
        out = self._update(old, new, n, expired, added, rebuild, observed)
        recomputed = out.get("cancel", False)
        if recomputed:  # as a rebuild of this window
            out = self._update(old, new, n, 0, n, True, observed)
        out["recomputed"] = recomputed
        return out

    def _update(self, old, new, n, expired, added, rebuild, observed):
        from scipy.linalg.blas import dgemm

        width, shared = self.width, n - added
        x = np.empty((expired + added, width + 1))
        x[:expired, :width] = old[:expired]
        x[expired:, :width] = new[shared:n]
        x[:, width] = 1.0
        np.nan_to_num(x, copy=False, nan=0.0)
        y = x * np.r_[-np.ones(expired), np.ones(added)][:, None]
        if rebuild:
            self.c[...] = 0
        self.c = dgemm(1.0, x.T, y, beta=1.0, c=self.c, overwrite_c=1)
        if added:  # the columns' largest squared return since the rebuild
            with np.errstate(over="ignore"):  # an infinite return: rejected in _finish
                squares = np.square(x[expired:, :width]).max(axis=0)
            if rebuild:
                self.peak[...] = squares
            else:
                np.maximum(self.peak, squares, out=self.peak)
        return _finish(self.c, observed, n, width, self.annualize, self.peak)

    def advance(self) -> None:
        self.current = 1 - self.current

    def matrix(self, indices, n):
        return _compact_covariance(self.c, indices, n, self.width)


def _copy_rows(window: np.ndarray, source, start: int, stop: int) -> None:
    """``window[start:stop] = source[start:stop]`` for a host 2-D array or host parts."""
    if isinstance(source, np.ndarray):
        window[start:stop] = source[start:stop]
    else:
        for j in range(start, stop):
            window[j] = source[j]


class _GpuCore:
    """CuPy mechanics: one non-blocking stream, every buffer allocated per width and
    capacity, pinned staging. A window writes its control block (its shape and its new
    bars' addresses) to pinned memory and launches the CUDA graph of the current
    window-buffer parity, captured at allocation: the control-block copy, the flags reset,
    seven kernels and the packed result's copy to the host; then one synchronize. If a
    capture fails (another thread synchronized the whole device meanwhile), windows run the
    same work eagerly, and the capture is retried ``REBUILD_EVERY`` windows later."""

    def __init__(self, cp, device: int, annualize: float) -> None:
        from . import _risk_kernels

        self.cp, self.device, self.annualize = cp, device, annualize
        self._layout = _risk_kernels.layout
        self.k = _risk_kernels.kernels(cp, device)
        self.width = None
        self.capacity = 0
        self.graphs: dict[int, object] = {}
        self._capture_in: int | None = None  # eager windows left before the next capture
        self.stream = cp.cuda.Stream(non_blocking=True)
        self._ready = cp.cuda.Event(disable_timing=True)
        self._hold: list = []  # inputs the stream reads until the window's synchronize

    def allocate(self, width: int, capacity: int) -> None:
        cp = self.cp
        self.width = width
        n1 = width + 1
        self.at = self._layout(width)
        with self.stream:
            self.c = cp.zeros((n1, n1), order="F")
            self.obs, self.sd, self.obsc, self.peak = (cp.zeros(width) for _ in range(4))
            self.d, self.y = cp.zeros(n1), cp.zeros(n1)
            self.elig, self.idx = cp.zeros(width, cp.int32), cp.zeros(width, cp.int32)
            self.pidx, self.hidx = cp.zeros(PROBES, cp.int32), cp.zeros(HEATMAP, cp.int32)
            self.cnt = cp.zeros(1, cp.int32)
            self.out = cp.zeros(self.at["size"])
            self.flags = self.out[4:6].view(cp.int32)  # in the header: one D2H brings them
        pinned = cp.cuda.alloc_pinned_memory
        self._pinned = [pinned(8 * self.at["size"]), pinned(8 * width)]
        self.h_out = np.frombuffer(self._pinned[0], np.float64, self.at["size"])
        self.h_obs = np.frombuffer(self._pinned[1], np.float64, width)
        self.current = 0
        self._windows(capacity, 0)

    def grow(self, capacity: int, held: int) -> None:
        self._windows(capacity, held)

    def _windows(self, capacity: int, held: int) -> None:
        """Window buffers, control block and pinned staging for ``capacity`` bars (keeping
        the current window's ``held`` bars), and the graphs over them."""
        cp, width = self.cp, self.width
        with self.stream:
            windows = [cp.zeros((capacity, width)) for _ in range(2)]
            if held:
                windows[self.current][:held] = self.windows[self.current][:held]
            self.ctl = cp.zeros(CONTROL + capacity, cp.int64)
        pinned = cp.cuda.alloc_pinned_memory
        self._staging = [pinned(8 * capacity * width), pinned(8 * (CONTROL + capacity))]
        self.h_stage = np.frombuffer(self._staging[0], np.float64, capacity * width)
        self.h_stage = self.h_stage.reshape(capacity, width)
        self.h_ctl = np.frombuffer(self._staging[1], np.int64, CONTROL + capacity)
        self.h_ctl[:] = 0
        self.stream.synchronize()
        self.windows, self.capacity = windows, capacity
        self._capture()

    def _capture(self) -> None:
        """The graph of each window-buffer parity, captured and uploaded here."""
        cp, s = self.cp, self.stream
        self.graphs, self._capture_in = {}, None
        try:
            for parity in (0, 1):
                s.begin_capture()
                try:
                    self._enqueue(parity)
                finally:
                    graph = s.end_capture()
                graph.upload(s)
                self.graphs[parity] = graph
            s.synchronize()
        except (cp.cuda.runtime.CUDARuntimeError, cp.cuda.driver.CUDADriverError):
            # Invalidated (in relaxed mode by a device-wide synchronize elsewhere): the
            # capture leaves the stream usable, nothing captured ran.
            self.graphs, self._capture_in = {}, REBUILD_EVERY

    def asarray(self, values, width: int):
        if not isinstance(values, np.ndarray) and hasattr(values, "__cuda_array_interface__"):
            # Contiguous: the window copies 8 * width bytes from its address.
            values = self.cp.ascontiguousarray(values, dtype=self.cp.float64)
        else:
            values = np.asarray(values, dtype=np.float64)
        if values.shape != (width,):
            raise ValueError("observed_variance needs one value per instrument")
        return values

    def _stage(self, source, is_parts: bool, start: int, stop: int) -> None:
        """The addresses of bars ``start:stop`` of the input into the control block: host
        rows through pinned staging (the device reads them there), device rows and parts
        where they are."""
        row, cp = 8 * self.width, self.cp
        addresses = self.h_ctl[CONTROL : CONTROL + stop - start]
        if isinstance(source[start], np.ndarray) if is_parts else isinstance(source, np.ndarray):
            _copy_rows(self.h_stage, source, start, stop)
            addresses[:] = self.h_stage.ctypes.data + row * np.arange(start, stop)
        elif not is_parts:
            rows = cp.ascontiguousarray(source, dtype=cp.float64)
            self._hold.append(rows)
            self._follow_caller()
            addresses[:] = rows.data.ptr + row * np.arange(start, stop)
        else:
            parts = [_device_vector(cp, source[j]) for j in range(start, stop)]
            converted = [p for j, p in enumerate(parts) if p is not source[start + j]]
            if converted:  # converted on the caller's stream
                self._hold += converted
                self._follow_caller()
            addresses[:] = [p.data.ptr for p in parts]

    def _follow_caller(self) -> None:
        """Order this engine's stream after the caller's current stream, where the device
        inputs were written or converted (``rows``, a converted part, a device variance)."""
        self._ready.record(self.cp.cuda.get_current_stream())
        self.stream.wait_event(self._ready)

    def run(self, source, is_parts, n, expired, added, rebuild, proven, observed):
        rt, s, ctl = self.cp.cuda.runtime, self.stream, self.h_ctl
        if self._capture_in is not None:
            self._capture_in -= 1
            if self._capture_in <= 0:
                self._capture()
        start = n - added if proven else 0
        if start < n:
            self._stage(source, is_parts, start, n)
        given = observed is not None
        ctl[:5] = n, expired, added, REBUILD if rebuild else SHIFT if proven else CHECK, given
        if given:
            if isinstance(observed, np.ndarray):
                self.h_obs[:] = observed
                source_ptr, copy = self.h_obs.ctypes.data, rt.memcpyHostToDevice
            else:
                source_ptr, copy = observed.data.ptr, rt.memcpyDeviceToDevice
                self._hold.append(observed)
                self._follow_caller()
            rt.memcpyAsync(self.obs.data.ptr, source_ptr, 8 * self.width, copy, s.ptr)
        out = self._launch()
        recomputed = out is not None and out.get("cancel", False)
        if recomputed:  # as a rebuild of the window already in place
            ctl[3] = RECOMPUTE
            out = self._launch()
        self._hold.clear()
        if out is not None:
            out["recomputed"] = recomputed
        return out

    def _launch(self) -> dict | None:
        graph = self.graphs.get(self.current)
        if graph is not None:
            graph.launch(self.stream)
        else:
            self._enqueue(self.current)
        self.stream.synchronize()
        return self._unpack()

    def _enqueue(self, parity: int) -> None:
        """The work of one window, as the control block describes it (capturable)."""
        k, s = self.k, self.stream
        rt = self.cp.cuda.runtime
        width, n1 = self.width, self.width + 1
        old, new = self.windows[parity], self.windows[1 - parity]
        rt.memcpyAsync(
            self.ctl.data.ptr,
            self.h_ctl.ctypes.data,
            self.ctl.nbytes,
            rt.memcpyHostToDevice,
            s.ptr,
        )
        rt.memsetAsync(self.flags.data.ptr, 0, self.flags.nbytes, s.ptr)
        k["assemble"](
            (min(2048, (self.capacity * width + 255) // 256),),
            (256,),
            (self.ctl, old, new, np.int32(width), self.flags),
            stream=s,
        )
        k["rank_update"](
            (min(4096, (n1 * n1 + 255) // 256),),
            (256,),
            (self.c, np.int32(n1), old, new, self.ctl, self.flags, self.peak),
            stream=s,
        )
        columns = ((width + 127) // 128,)
        k["colstats"](
            columns,
            (128,),
            (
                new,
                np.int32(width),
                self.c,
                self.ctl,
                self.obs,
                self.elig,
                self.sd,
                self.d,
                self.peak,
                self.flags,
            ),
            stream=s,
        )
        k["compact"](
            (1,),
            (1024,),
            (
                np.int32(width),
                self.elig,
                self.sd,
                self.obs,
                np.float64(self.annualize),
                self.idx,
                self.obsc,
                self.pidx,
                self.cnt,
                self.out,
            ),
            stream=s,
        )
        k["toprank"](columns, (128,), (self.cnt, self.obsc, self.idx, self.hidx), stream=s)
        k["colreduce"]((n1,), (256,), (self.c, self.d, np.int32(n1), self.y), stream=s)
        k["final_"](
            (1,),
            (1024,),
            (
                self.c,
                np.int32(width),
                self.ctl,
                self.cnt,
                self.d,
                self.y,
                self.sd,
                self.hidx,
                self.pidx,
                self.out,
            ),
            stream=s,
        )
        rt.memcpyAsync(
            self.h_out.ctypes.data, self.out.data.ptr, self.out.nbytes, rt.memcpyDeviceToHost, s.ptr
        )

    def _unpack(self) -> dict | None:
        h, at = self.h_out, self.at
        flags = h[4:6].view(np.int32)
        if flags[1]:
            return None
        if flags[0]:
            return {"invalid": True}
        if flags[2]:
            return {"invalid": False, "cancel": True}
        count = int(h[0])
        heat, probes = min(HEATMAP, count), min(PROBES, count)
        return {
            "invalid": False,
            "count": count,
            "indices": h[at["indices"] : at["indices"] + count].astype(np.int64),
            "vol": h[at["vol"] : at["vol"] + count],
            "mean_correlation": float(h[1]),
            "heat_index": h[at["heat_index"] : at["heat_index"] + heat].astype(np.int64),
            "heatmap": h[at["heat"] : at["heat"] + heat * heat].reshape(heat, heat),
            "probes": h[at["probes"] : at["probes"] + probes * probes].reshape(probes, probes),
        }

    def advance(self) -> None:
        self.current = 1 - self.current

    def matrix(self, indices, n):
        with self.stream:
            c = self.c.get(stream=self.stream)
        return _compact_covariance(c, indices, n, self.width)


def _device_vector(cp, part):
    """A contiguous float64 device vector (bars' parts already are)."""
    if part.dtype == cp.float64 and part.flags.c_contiguous:
        return part
    return cp.ascontiguousarray(part, dtype=cp.float64)


_gpu_names = {}


def gpu_name(xp, device):
    """Device name, queried once: getDeviceProperties reads ~100 driver attributes while
    holding the GIL, which stalled the receiver for hundreds of ms when run per window."""
    if device not in _gpu_names:
        _gpu_names[device] = xp.cuda.runtime.getDeviceProperties(device)["name"].decode()
    return _gpu_names[device]
