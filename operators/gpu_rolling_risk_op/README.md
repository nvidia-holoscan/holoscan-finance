# GpuRollingRiskOp

Rolling volatility, covariance and correlation on the GPU. Each run takes the newest
window of bar returns, already in GPU memory, and updates running sums with cuBLAS
products for only the bars added and expired since the last run:

```text
window (BarWindow) → incremental cuBLAS sums → volatility, covariance, correlation → risk (RiskUpdate)
```

Only instruments with returns throughout the window enter the matrix; missing returns
are never treated as zero. The engine
([covariance.py](../../python/holoscan/marketdata/covariance.py)) has a NumPy/SciPy
backend that produces the same results on the CPU.

## Ports

| Port | Type | Notes |
| --- | --- | --- |
| `window` (in) | `BarWindow` | Holds one message; a newer window replaces a waiting one |
| `risk` (out) | `RiskUpdate` | `window`, `result` (below) and `rows_host` (with `host_variance`) |

`result` has `annual_vol` and `eligible_indices` (instruments in the matrix),
`mean_correlation`, a 24 x 24 `heatmap` of the most volatile names, `covariance_probes`,
`eligible`, `pairs`, `milliseconds`, `method` (`incremental` or `rebuild`), `gpu_name`
and `finished_wall_ns`.

## Arguments

| Argument | Default | Purpose |
| --- | --- | --- |
| `min_interval_s` | `0.05` | The operator runs at most this often, always on the newest window |
| `bar_seconds` | `5` | Annualizes volatility (252 days of 6.5 hours) |
| `gpu_device` | `0` | CUDA device |
| `ready` | none | Callable; returning False leaves the newest window queued, so a consumer that must see every result can hold the operator back |
| `host_variance` | `False` | Bool or callable; ranks and selects instruments by the NumPy variance of a host copy of the window, so a NumPy reference makes identical choices |

The first window of each generation also runs on a throwaway engine, so kernel
compilation and workspace allocation stay out of the reported times. Connect `risk`:
an unconnected output stops Holoscan from scheduling the operator.
