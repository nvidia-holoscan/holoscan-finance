# RollingRiskOp

Rolling volatility, covariance and correlation of the newest window of bar returns. Each
analysis updates running sums with BLAS products for only the bars added and expired since
the last one (cuBLAS through CuPy, or NumPy/SciPy on the CPU):

```text
window (BarWindow) → newest-window slot → incremental covariance → risk (RiskUpdate) [+ validate]
```

Only instruments with returns throughout the window enter the matrix; missing returns are
never treated as zero. The engine is
[covariance.py](../../python/holoscan_finance/marketdata/covariance.py).

Processor: `holoscan_finance.analytics.rolling.RollingRisk`. Derived from 4.3
`GpuRollingRiskOp` (inline GPU), the host `RiskOp` with `gpu_window`/`cpu_window` (process
executor) and `validate_next` (the cross-check gate).

## Ports

| Port | Type | Notes |
| --- | --- | --- |
| `window` (in) | `BarWindow` | every received window replaces a waiting one (`windows_superseded`) |
| `risk` (out) | `RiskUpdate` | `window`, `result`, `rows_host`, `backend`, `validated` |
| `validate` (out) | `RiskUpdate` | declared only with `cross_check`: the validated updates, with `rows_host` |

`result` holds `annual_vol` and `eligible_indices` (the instruments in the matrix),
`mean_correlation`, a 24 x 24 `heatmap` of the most volatile names, `covariance_probes`,
`eligible`, `pairs`, `milliseconds` (engine time), `method` (`incremental` or `rebuild`) and
`finished_wall_ns`; the GPU adds `gpu_name`; the CPU worker adds its BLAS threads, trials,
affinity and library.

Contract: `OnClock(poll_period_us).input(window, DrainAll())`, plus
`max_start_delay(deadline_us).on_violation(EMIT_WITNESS)` when `deadline_us > 0`.

## Why a 1 ms poll with an internal minimum interval

4.3 ran risk as soon as a window was available *and* `--interval` had passed since the last
run. A 50 ms clock alone would add 28 ms at p50 (judge J1); polling every 1 ms and enforcing
`min_interval_us` in the operator measured a 0.99 ms pick-up at about 0.01 core.

## Params

| Param | Type | Default | Meaning |
| --- | --- | --- | --- |
| `backend` | str | `"gpu"` | `gpu` or `cpu` |
| `executor` | str | `"inline"` | `inline` (in the lane) or `process` (one spawned worker; the host front ends use it to keep BLAS and CUDA off the decoder's GIL). `cpu`+`inline` is for tests: SciPy holds the GIL |
| `bar_seconds` | int | `5` | annualizes volatility (252 days of 6.5 hours) |
| `gpu_device` | int | `0` | CUDA device |
| `min_interval_us` | int | `50000` | at most one analysis per interval (4.3 `--interval 0.05`) |
| `poll_period_us` | int | `1000` | the poll clock |
| `host_variance` | bool | `False` | inline: rank and select instruments by the NumPy variance of a host copy, as a NumPy reference would |
| `cross_check` | bool | `False` | claim `gate` per analysis and emit claimed (validated) updates on `validate`; requires `gate` and `backend="gpu"` |
| `gate` | `Ref` | `Ref()` | `Ref(CrossCheckGate)`; while a claim is open no window is analyzed |
| `cpu_threads` | int | `0` | process + cpu: BLAS threads; `0` tunes over {1, 2, 4, 8, all} |
| `full_window_bars` | int | `60` | process + cpu: the worker re-tunes at the first full window |
| `worker_cpus` | list | `[]` | CPUs of the spawned worker (`pin_worker`); `[]` inherits |
| `deadline_us` | int | `0` | `>0`: count poll ticks that started this late (`late_activations`): almost always the ticks right after an analysis that overran the period, not late windows |
| `stats` | `Ref` | `Ref()` | `Ref(RiskStats)`; private if empty |

## Behaviour

- The first window of each generation also runs on a throwaway engine, so kernel
  compilation and workspace allocation stay out of the reported times.
- A validated window carries a host copy of its rows, and its variance ranks the
  instruments, so the CPU comparison makes identical choices.
- Outputs go through hold-and-retry: on `BackpressuredError` the update is retried at the
  next poll, before any new analysis.
- Process executor: the worker is spawned and pinged in `start()`; a worker exception sets
  `RiskStats.error` (`"Analytics failed: ..."`) and stops further analyses, as 4.3 did, and
  releases the gate if that window was validated (no comparison will come for it).
  `host_variance` needs the inline executor: the workers rank by their own variance.
  An inline exception fails the run.

## Edges and lane

- In: `LATEST` from `QuoteBarsOp.window`, another lane.
- `risk` → `CallbackSinkOp` (`ResultsStore.on_risk`): `lossless(8)`, same lane.
- `validate` → `RiskCrossCheckOp.risk`: `lossless(2)`, the `crosscheck` lane.
- Lane: the entry operator of `analytics`. It waits on GPU syncs or futures, which release
  the GIL. Inline GPU work runs on the legacy stream, as in 4.3.

## Telemetry

`RiskStats`: `windows_received`, `windows_superseded`, `analyzed`, `gate_waits` (polls that
found a comparison in flight), `late_activations`, `last_completed_bars`, `last_generation`,
`last_window_key` (`(generation, completed_bars)` of the last analyzed window), `busy`,
`last_ms`, `max_ms`, `error`, `lane_cpus`.

## Example

```python
from holoscan_finance.operators import CallbackSinkOp, RiskCrossCheckOp, RollingRiskOp
from holoscan_finance.operators.edges import LATEST, lossless
from holoscan_finance.ref import Ref

risk = g.op(RollingRiskOp, "risk", backend="gpu", cross_check=True, gate=Ref(gate), stats=Ref(risk_stats))
publish = g.op(CallbackSinkOp, "publish", callback=Ref(results.on_risk))
crosscheck = g.op(RiskCrossCheckOp, "crosscheck", stats=Ref(risk_stats))
publish_comparison = g.op(CallbackSinkOp, "publish_comparison", callback=Ref(results.on_comparison))
g.add_flow(bars.window, risk.window, **LATEST)
g.add_flow(risk.risk, publish.items, **lossless(8))
g.add_flow(risk.validate, crosscheck.risk, **lossless(2))
g.add_flow(crosscheck.comparison, publish_comparison.items, **lossless(2))
g.partition("analytics").add(risk, publish)
g.partition("crosscheck").add(crosscheck, publish_comparison)
```

`results.on_comparison` publishes each comparison and then releases `gate`; without a
`cross_check`, drop the last four operators and edges.
