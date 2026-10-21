# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rolling sample covariance with NumPy/SciPy BLAS or CuPy/cuBLAS.

No custom GPU kernels. Maintain sum(r) and sum(r r.T); additions and expirations
are BLAS matrix products. Missing returns never become displayed zero returns:
only instruments valid throughout the current window enter the reported matrix.
The CPU uses incremental BLAS too, rather than an artificially slow Python loop.
Its eligibility and display rules are the ones the CPU/GPU agreement checks assume.
"""

from __future__ import annotations

import time

import numpy as np


class CovarianceEngine:
    def __init__(self, backend="cpu", *, bar_seconds=5, gpu_device=0):
        if backend not in ("cpu", "gpu"):
            raise ValueError("backend must be cpu or gpu")
        self.backend, self.bar_seconds = backend, bar_seconds
        self.xp = np
        if backend == "gpu":
            import cupy as cp

            cp.cuda.Device(gpu_device).use()
            self.xp = cp
        self.rows = self.times = self.cross = self.total = None
        self.calls = 0

    def sync(self):
        if self.backend == "gpu":
            self.xp.cuda.get_current_stream().synchronize()

    def host(self, array):
        return np.asarray(array) if self.backend == "cpu" else self.xp.asnumpy(array)

    def _accumulate(self, host_rows, sign):
        if not len(host_rows):
            return
        if isinstance(host_rows, np.ndarray):
            x = self.xp.asarray(np.nan_to_num(host_rows, nan=0.0), dtype=self.xp.float64)
        else:  # device-resident window (GPU ingest): no host round trip
            x = self.xp.nan_to_num(host_rows, nan=0.0)
        if self.backend == "cpu":
            from scipy.linalg.blas import dgemm

            self.cross = dgemm(sign, x.T, x, beta=1.0, c=self.cross, overwrite_c=1)
        else:
            from cupy.cublas import gemm

            gemm("T", "N", x, x, out=self.cross, alpha=sign, beta=1.0)
        self.total += sign * x.sum(axis=0)

    def update(
        self, rows, times, *, force_rebuild=False, return_matrix=False, observed_variance=None
    ):
        """One immutable window; includes input transfer, math and compact output transfer.

        ``rows`` may be a CuPy array on the GPU backend (device-resident window from
        GPU ingest); it is then used in place. ``observed_variance`` optionally supplies
        the per-instrument sample variance used for eligibility and display order, so
        a separately computed CPU comparison uses identical coordinates.
        """
        device = self.backend == "gpu" and not isinstance(rows, np.ndarray)
        rq = self.xp if device else np
        rows = rq.asarray(rows, dtype=rq.float64)
        times = np.asarray(times, dtype=np.int64)
        if (
            rows.ndim != 2
            or rows.shape[0] != len(times)
            or len(times) < 2
            or np.any(np.diff(times) <= 0)
            or bool(rq.isinf(rows).any())
        ):
            raise ValueError("need >=2 increasing bar timestamps and finite-or-NaN returns")
        self.sync()
        started = time.perf_counter()
        xp = self.xp
        n, width = rows.shape
        reset = (
            force_rebuild
            or self.rows is None
            or self.rows.shape[1] != width
            or self.calls % 256 == 0
            or times[-1] < self.times[-1]
        )
        if reset:
            self.cross = xp.zeros((width, width), dtype=xp.float64, order="F")
            self.total = xp.zeros(width, dtype=xp.float64)
            removed, added = rq.empty((0, width)), rows
        else:
            if device and isinstance(self.rows, np.ndarray):
                self.rows = rq.asarray(self.rows)
            removed = self.rows[rq.asarray(~np.isin(self.times, times))]
            added = rows[rq.asarray(~np.isin(times, self.times))]
            # A seek/reset may reuse timestamps with different partial observations.
            overlap = np.intersect1d(times, self.times)
            if len(overlap):
                old = self.rows[rq.asarray(np.searchsorted(self.times, overlap))]
                new = rows[rq.asarray(np.searchsorted(times, overlap))]
                same = (old == new) | (rq.isnan(old) & rq.isnan(new))
                if not bool(same.all()):
                    raise ValueError("completed bars changed; reset engine after seek")
        self._accumulate(removed, -1.0)
        self._accumulate(added, 1.0)
        self.rows, self.times = rows.copy(), times.copy()
        self.calls += 1
        # Eligibility comes from the actual window. An expired price move can
        # leave roundoff in incremental cross-products even when every current
        # return is constant; that residue must never create an eligible name.
        if observed_variance is None:
            observed_variance = rq.var(rows, axis=0, ddof=1)
        observed_variance = rq.asarray(observed_variance)
        eligible = rq.isfinite(observed_variance) & (observed_variance > 1e-20)
        indices = np.flatnonzero(self.host(eligible) if device else eligible)
        variance = xp.maximum((xp.diag(self.cross) - self.total**2 / n) / (n - 1), 0)
        ii = xp.asarray(indices)
        total = self.total[ii]
        # cross is symmetric and Fortran contiguous; transposing gives contiguous
        # rows for the gather. Use the same allocation-saving path on CPU and GPU.
        covariance = self.cross.T[xp.ix_(ii, ii)]
        covariance -= xp.outer(total / n, total)
        covariance /= n - 1
        variance = variance[ii]
        std = xp.sqrt(variance)
        correlation = xp.outer(std, std)
        xp.divide(covariance, correlation, out=correlation)
        xp.clip(correlation, -1, 1, out=correlation)
        count = len(indices)
        annual_vol = self.host(std * (252 * 6.5 * 3600 / self.bar_seconds) ** 0.5)
        # Full-universe aggregate is computed, while only a compact heatmap goes to HTTP.
        mean_correlation = (
            float(self.host((correlation.sum() - count) / (count * (count - 1))))
            if count > 1
            else None
        )
        # Choose display coordinates from the identical host window. Equal
        # volatilities can differ by rounding across BLAS backends, including
        # at the top-24 cutoff; that must not change which correlations we compare.
        order = np.argsort(-self.host(observed_variance)[indices], kind="stable")
        chosen = order[:24]
        ci = xp.asarray(chosen)
        heatmap = self.host(correlation[xp.ix_(ci, ci)])
        # Deterministic distributed probes plus all variances for each live comparison.
        probe = np.linspace(0, max(0, count - 1), min(32, count), dtype=int)
        pi = xp.asarray(probe)
        probes = self.host(covariance[xp.ix_(pi, pi)])
        matrix = self.host(covariance) if return_matrix else None
        self.sync()
        elapsed_ms = (time.perf_counter() - started) * 1000
        return {
            "backend": self.backend,
            "milliseconds": elapsed_ms,
            "method": "rebuild" if reset else "incremental",
            "added_bars": len(added),
            "expired_bars": len(removed),
            "bars": n,
            "universe": width,
            "eligible_indices": indices.tolist(),
            "eligible": count,
            "pairs": count * (count - 1) // 2,
            "annual_vol": annual_vol.tolist(),
            "mean_correlation": mean_correlation,
            "heatmap_indices": indices[chosen].tolist(),
            "heatmap": heatmap.tolist(),
            "covariance_probes": probes.tolist(),
            "matrix": matrix,
        }


_gpu_names = {}


def gpu_name(xp, device):
    """Device name, queried once: getDeviceProperties reads ~100 driver attributes while
    holding the GIL, which stalled the receiver for hundreds of ms when run per window."""
    if device not in _gpu_names:
        _gpu_names[device] = xp.cuda.runtime.getDeviceProperties(device)["name"].decode()
    return _gpu_names[device]
