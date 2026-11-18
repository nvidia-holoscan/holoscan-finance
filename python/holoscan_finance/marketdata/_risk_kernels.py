# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CUDA kernels of the GPU rolling-risk engine (``covariance.CovarianceEngine("gpu")``).

Every kernel reads the window's shape from a small control block (``CONTROL`` int64 values,
then one source address per gathered bar) that the window copies to the device first, so
one CUDA graph per window buffer serves every window: a one-bar or multi-bar slide, a
growing window, a rebuild or a recomputation. The modes: ``REBUILD`` (all bars gathered,
``C`` from zero), ``SHIFT`` (the shared bars, proven identical by the caller, shifted from
the previous window buffer; the new bars gathered), ``CHECK`` (all bars gathered, the shared
ones compared with the previous window's) and ``RECOMPUTE`` (``C`` from zero over the window
already in place).

Two CuPy ``RawModule``s. ``UPDATE`` maintains the augmented second-moment matrix
``C = sum_t [f(r_t); 1][f(r_t); 1]^T`` (``(N+1) x (N+1)``, column-major, ``f``: NaN -> 0):
``assemble`` writes the window, ``rank_update`` applies the expired bars with sign -1, then
the new bars with +1, and tracks each column's largest squared return since the last
rebuild (``peak``). ``FINISH`` is compiled with ``--fmad=false`` so that every result
rounds as the NumPy expressions of the CPU engine do (``covariance._finish``), operation for
operation:

- ``colstats``: the window's sample variance exactly as ``np.var(rows, axis=0, ddof=1)``
  (sequential axis-0 sums), eligibility, the standard deviations from ``C``, ``d = 1/sigma``
  on eligible names, and the cancellation flag (a column with a window variance above
  1e-20 whose ``peak`` exceeds ``CANCEL`` times its diagonal of ``C``);
- ``compact``: one block; the eligible indices in ascending order, their volatilities and
  variances, and the probe slots of ``np.linspace(0, count - 1, min(32, count), dtype=int)``;
- ``toprank``: the stable descending variance rank of each eligible name (the
  ``argsort(-variance, kind="stable")`` order); ranks below 24 are the heatmap names;
- ``colreduce``: ``y = C d`` by column (C is symmetric), only for the columns that use it;
- ``final``: one block; the mean correlation from ``d.y`` and ``y[N]``, the 24 x 24
  heatmap and the 32 x 32 covariance probes.

Every result goes to one packed buffer (``layout``), so a window costs one device-to-host
copy. Flags: ``flags[0]`` a non-finite statistic (an infinite input), ``flags[1]`` changed
completed bars (``rank_update`` then leaves ``C`` untouched), ``flags[2]`` cancellation
(the engine recomputes the window).
"""

from __future__ import annotations

import threading

HEATMAP, PROBES = 24, 32
HEADER = 8  # count, mean correlation, heatmap size, probe count, four int32 flags, unused
CONTROL = 8  # n, expired, added, mode, given, unused; then one address per gathered bar
REBUILD, SHIFT, CHECK, RECOMPUTE = 0, 1, 2, 3
# A column whose largest squared return since the last rebuild exceeds CANCEL times its
# current sum of squares has lost about CANCEL * 1e-16 * (window length) of its relative
# precision to subtraction: the window is recomputed (docs/METHODOLOGY.md, "Statistics").
CANCEL = 1e5

UPDATE = r"""
extern "C" __global__ void assemble(const long long* __restrict__ ctl,
                                    const double* __restrict__ old, double* __restrict__ win,
                                    const int width, int* __restrict__ flags) {
    // Window row j: SHIFT, j < shared: row expired + j of the previous window; otherwise
    // the width doubles at address ctl[CONTROL + j - start]. CHECK also compares the
    // shared rows with the previous window's (NaN equals NaN, as the CPU check).
    const long long n = ctl[0], expired = ctl[1], added = ctl[2], mode = ctl[3];
    if (mode == RECOMPUTE) return;
    const long long start = mode == SHIFT ? n - added : 0;
    const long long shared = (n - added) * width, total = n * width, base = expired * width;
    for (long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x; idx < total;
         idx += (long long)gridDim.x * blockDim.x) {
        if (mode == SHIFT && idx < shared) {
            win[idx] = old[base + idx];
            continue;
        }
        const long long j = idx / width, i = idx - j * width;
        const double x = ((const double*)ctl[CONTROL + j - start])[i];
        win[idx] = x;
        if (mode == CHECK && idx < shared) {
            const double y = old[base + idx];
            if (!(x == y || (isnan(x) && isnan(y)))) flags[1] = 1;
        }
    }
}

extern "C" __global__ void rank_update(double* __restrict__ c, const int n1,
                                       const double* __restrict__ old,
                                       const double* __restrict__ win,
                                       const long long* __restrict__ ctl,
                                       const int* __restrict__ flags,
                                       double* __restrict__ peak) {
    // c += sum_k s_k f(x_k) f(x_k)^T over the expired rows of the previous window (s = -1),
    // then the new rows of this one (+1); from zero over the whole window for a rebuild. A
    // row has n1 - 1 returns and an implied trailing 1.
    if (flags[1]) return;  // the completed bars changed: the statistics stay as they were
    const long long n = ctl[0], mode = ctl[3];
    const bool fresh = mode == REBUILD || mode == RECOMPUTE;
    const int k_out = fresh ? 0 : (int)ctl[1];
    const int k_in = fresh ? (int)n : (int)ctl[2];
    const int width = n1 - 1;
    const double* add = win + (n - k_in) * width;
    const long long total = (long long)n1 * n1;
    for (long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x; idx < total;
         idx += (long long)gridDim.x * blockDim.x) {
        const int r = (int)(idx % n1), q = (int)(idx / n1);
        double acc = fresh ? 0.0 : c[idx];
        for (int j = 0; j < k_out; ++j) {
            double a = r < width ? old[(long long)j * width + r] : 1.0;
            double b = q < width ? old[(long long)j * width + q] : 1.0;
            a = isnan(a) ? 0.0 : a;
            b = isnan(b) ? 0.0 : b;
            acc += a * (-b);
        }
        for (int j = 0; j < k_in; ++j) {
            double a = r < width ? add[(long long)j * width + r] : 1.0;
            double b = q < width ? add[(long long)j * width + q] : 1.0;
            a = isnan(a) ? 0.0 : a;
            b = isnan(b) ? 0.0 : b;
            acc += a * b;
        }
        c[idx] = acc;
        if (r == q && r < width) {  // the largest squared return taken in since the rebuild
            double m = fresh ? 0.0 : peak[r];
            for (int j = 0; j < k_in; ++j) {
                const double a = add[(long long)j * width + r];
                m = fmax(m, isnan(a) ? 0.0 : __dmul_rn(a, a));
            }
            peak[r] = m;
        }
    }
}
"""

FINISH = r"""
extern "C" __global__ void colstats(const double* __restrict__ w, const int width,
        const double* __restrict__ c, const long long* __restrict__ ctl,
        double* __restrict__ obs, int* __restrict__ elig, double* __restrict__ sd,
        double* __restrict__ d, const double* __restrict__ peak, int* __restrict__ flags) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= width) return;
    const int n = (int)ctl[0];
    double o;
    if (ctl[4]) {  // a supplied variance
        o = obs[j];
    } else {  // np.var(w, axis=0, ddof=1): sequential sums over the bars, then the mean
        double s = 0.0;
        for (int t = 0; t < n; ++t) s += w[(long long)t * width + j];
        const double m = s / n;
        double ss = 0.0;
        for (int t = 0; t < n; ++t) {
            const double x = w[(long long)t * width + j] - m;
            ss += x * x;
        }
        o = ss / (n - 1);
        obs[j] = o;
    }
    const long long n1 = width + 1;
    const double dg = c[(long long)j * n1 + j], tt = c[(long long)width * n1 + j];
    if (!isfinite(dg) || !isfinite(tt)) flags[0] = 1;
    double v = (dg - tt * tt / n) / (n - 1);
    v = (v >= 0.0 || isnan(v)) ? v : 0.0;  // np.maximum(v, 0)
    const double st = sqrt(v);
    const bool moving = isfinite(o) && o > 1e-20;
    const int e = moving && v > 0.0;
    elig[j] = e;
    sd[j] = st;
    d[j] = e ? 1.0 / st : 0.0;
    if (moving && peak[j] > CANCEL * dg) flags[2] = 1;
}

extern "C" __global__ void compact(const int width, const int* __restrict__ elig,
        const double* __restrict__ sd, const double* __restrict__ obs, const double ann,
        int* __restrict__ idx, double* __restrict__ obsc, int* __restrict__ pidx,
        int* __restrict__ cnt, double* __restrict__ out) {
    // One block of 1024 threads: a stable compaction in chunks of 1024 columns (warp
    // scans), then the probe slots.
    __shared__ int warp_total[32];
    __shared__ int base;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (threadIdx.x == 0) base = 0;
    __syncthreads();
    for (int start = 0; start < width; start += blockDim.x) {
        const int j = start + threadIdx.x;
        const int f = j < width ? elig[j] : 0;
        int v = f;
        for (int o = 1; o < 32; o <<= 1) {
            const int u = __shfl_up_sync(0xffffffffu, v, o);
            if (lane >= o) v += u;
        }
        if (lane == 31) warp_total[warp] = v;
        __syncthreads();
        if (warp == 0) {
            int t = warp_total[lane];
            for (int o = 1; o < 32; o <<= 1) {
                const int u = __shfl_up_sync(0xffffffffu, t, o);
                if (lane >= o) t += u;
            }
            warp_total[lane] = t;
        }
        __syncthreads();
        const int inclusive = v + (warp ? warp_total[warp - 1] : 0);
        if (f) {
            const int p = base + inclusive - 1;
            idx[p] = j;
            obsc[p] = obs[j];
            out[HEADER + p] = (double)j;
            out[HEADER + width + p] = sd[j] * ann;
        }
        __syncthreads();
        if (threadIdx.x == blockDim.x - 1) base += inclusive;
        __syncthreads();
    }
    const int count = base;
    const int m_probes = count < PROBES ? count : PROBES;
    if (threadIdx.x == 0) {
        *cnt = count;
        out[0] = count;
        out[2] = count < HEATMAP ? count : HEATMAP;
        out[3] = m_probes;
    }
    if (threadIdx.x < m_probes) {  // np.linspace(0, count - 1, m_probes, dtype=int), NumPy 2
        const int m = threadIdx.x, div = m_probes - 1;
        const double delta = (double)(count - 1);
        double y = (double)m;
        if (div > 0) {
            const double step = delta / div;
            if (step == 0.0) {
                y = y / div;
                y = y * delta;
            } else {
                y = y * step;
            }
        } else {
            y = y * delta;
        }
        y = y + 0.0;
        if (m_probes > 1 && m == m_probes - 1) y = delta;
        pidx[m] = idx[(int)floor(y)];
    }
}

extern "C" __global__ void toprank(const int* __restrict__ cnt, const double* __restrict__ obsc,
                                   const int* __restrict__ idx, int* __restrict__ hidx) {
    const int count = *cnt;
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= count) return;
    const double v = obsc[p];
    int r = 0;
    for (int k = 0; k < count && r < HEATMAP; ++k) {
        const double u = obsc[k];
        r += (u > v) || (u == v && k < p);
    }
    if (r < HEATMAP) hidx[r] = idx[p];
}

extern "C" __global__ void colreduce(const double* __restrict__ c, const double* __restrict__ d,
                                     const int n1, double* __restrict__ y) {
    // y[q] = sum_r C[r, q] d[r] for the eligible columns and the totals' column.
    __shared__ double warp_sum[32];
    const int q = blockIdx.x;
    if (q < n1 - 1 && d[q] == 0.0) return;
    const double* col = c + (long long)q * n1;
    double acc = 0.0;
    for (int r = threadIdx.x; r < n1; r += blockDim.x) acc += col[r] * d[r];
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if ((threadIdx.x & 31) == 0) warp_sum[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x < 32) {
        acc = threadIdx.x < (blockDim.x >> 5) ? warp_sum[threadIdx.x] : 0.0;
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
        if (threadIdx.x == 0) y[q] = acc;
    }
}

extern "C" __global__ void final_(const double* __restrict__ c, const int width,
        const long long* __restrict__ ctl, const int* __restrict__ cnt,
        const double* __restrict__ d, const double* __restrict__ y,
        const double* __restrict__ sd, const int* __restrict__ hidx,
        const int* __restrict__ pidx, double* __restrict__ out) {
    // One block of 1024 threads. Heatmap at out[HEADER + 2 width], probes 576 later, the
    // heatmap's indices 1024 after the probes.
    __shared__ double warp_sum[32];
    const long long n1 = width + 1;
    const int n = (int)ctl[0];
    const int count = *cnt;
    double acc = 0.0;
    for (int j = threadIdx.x; j < width; j += blockDim.x) {
        if (d[j] != 0.0) acc += d[j] * y[j];
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if ((threadIdx.x & 31) == 0) warp_sum[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x < 32) {
        acc = warp_sum[threadIdx.x];
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
        if (threadIdx.x == 0) {
            const double s2 = y[width];
            out[1] = count > 1 ? ((acc - s2 * s2 / n) / (n - 1) - count)
                                     / ((double)count * (count - 1))
                               : 0.0;
        }
    }
    const int h = count < HEATMAP ? count : HEATMAP, p = count < PROBES ? count : PROBES;
    double* heat = out + HEADER + 2 * width;
    double* probes = heat + HEATMAP * HEATMAP;
    double* heat_index = probes + PROBES * PROBES;
    const double* totals = c + (long long)width * n1;
    for (int e = threadIdx.x; e < h * h + p * p; e += blockDim.x) {
        const bool in_heat = e < h * h;
        const int f = in_heat ? e : e - h * h, side = in_heat ? h : p;
        const int a = f / side, b = f % side;
        const int* u = in_heat ? hidx : pidx;
        const int ua = u[a], ub = u[b];
        // The CPU engine's covariance[a, b] = (C.T[ua, ub] - (t[ua] / n) * t[ub]) / (n - 1).
        double cov = c[ub + ua * n1] - (totals[ua] / n) * totals[ub];
        cov = cov / (n - 1);
        if (in_heat) {
            double r = cov / (sd[ua] * sd[ub]);
            r = r < -1.0 ? -1.0 : (r > 1.0 ? 1.0 : r);  // np.clip(r, -1, 1), NaN stays NaN
            heat[f] = r;
            if (b == 0) heat_index[a] = (double)ua;
        } else {
            probes[f] = cov;
        }
    }
}
"""

# The constants above, as the kernels see them: one definition for Python and CUDA.
_DEFINES = (
    f"-DHEATMAP={HEATMAP}",
    f"-DPROBES={PROBES}",
    f"-DHEADER={HEADER}",
    f"-DCONTROL={CONTROL}",
    f"-DREBUILD={REBUILD}",
    f"-DSHIFT={SHIFT}",
    f"-DCHECK={CHECK}",
    f"-DRECOMPUTE={RECOMPUTE}",
    f"-DCANCEL={CANCEL!r}",
)

_lock = threading.Lock()
_functions: dict[int, dict] = {}  # device -> kernel name -> RawKernel


def kernels(cp, device: int) -> dict:
    """The kernels on ``device`` (the current device), compiled once per process (or loaded
    from ``CUPY_CACHE_DIR``)."""
    with _lock:
        found = _functions.get(device)
        if found is None:
            update = cp.RawModule(code=UPDATE, options=_DEFINES)
            finish = cp.RawModule(code=FINISH, options=("--fmad=false", *_DEFINES))
            found = {name: update.get_function(name) for name in ("assemble", "rank_update")}
            for name in ("colstats", "compact", "toprank", "colreduce", "final_"):
                found[name] = finish.get_function(name)
            _functions[device] = found
        return found


def layout(width: int) -> dict[str, int]:
    """Offsets into the packed result buffer of a ``width``-symbol engine (float64 slots)."""
    heat = HEADER + 2 * width
    return dict(
        indices=HEADER,
        vol=HEADER + width,
        heat=heat,
        probes=heat + HEATMAP * HEATMAP,
        heat_index=heat + HEATMAP * HEATMAP + PROBES * PROBES,
        size=heat + HEATMAP * HEATMAP + PROBES * PROBES + HEATMAP,
    )
