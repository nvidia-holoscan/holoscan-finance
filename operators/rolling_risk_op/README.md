# RollingRiskOp

Rolling volatility, covariance and correlation of the newest window of bar returns. Each
analysis updates the window's second-moment matrix with one signed rank update for only
the bars added and expired since the last one (CUDA kernels through CuPy, or NumPy/SciPy
BLAS on the CPU: the same algorithm, and in `compare` mode the same work on both sides):

```text
window (BarWindow) → newest-window slot → incremental covariance → risk (RiskUpdate) [+ validate]
```

Only instruments with returns throughout the window enter the matrix; missing returns are
never treated as zero. The engine is
[covariance.py](../../python/holoscan_finance/marketdata/covariance.py) with the kernels in
[_risk_kernels.py](../../python/holoscan_finance/marketdata/_risk_kernels.py).

Processor: `holoscan_finance.analytics.rolling.RollingRisk`.

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

## Why a 250 µs poll with an internal minimum interval

Risk should run as soon as a window is available *and* `--interval` has passed since the
last run. A 50 ms clock alone would leave a ready window waiting for up to 50 ms; polling
every 250 µs and enforcing `min_interval_us` in the operator picks a window up within the
poll period.

## Params

| Param | Type | Default | Meaning |
| --- | --- | --- | --- |
| `backend` | str | `"gpu"` | `gpu` or `cpu` |
| `executor` | str | `"inline"` | `inline` (in the lane) or `process` (one spawned worker; the host front ends use it to keep BLAS and CUDA off the decoder's GIL). `cpu`+`inline` is for tests: SciPy holds the GIL |
| `bar_seconds` | int | `5` | annualizes volatility (252 days of 6.5 hours) |
| `gpu_device` | int | `0` | CUDA device |
| `min_interval_us` | int | `50000` | at most one analysis per interval (`--interval 0.05`) |
| `poll_period_us` | int | `250` | the poll clock (`--risk-poll-us`) |
| `host_variance` | bool | `False` | inline: rank and select instruments by the NumPy variance of a host copy, as a NumPy reference would |
| `cross_check` | bool | `False` | claim `gate` per analysis and emit claimed (validated) updates on `validate`; requires `gate` and `backend="gpu"` |
| `gate` | `Ref` | `Ref()` | `Ref(CrossCheckGate)`; while a claim is open no window is analyzed |
| `cpu_threads` | int | `0` | process + cpu: BLAS threads; `0` tunes over {1, 2, 4, 8, all} once per universe (the fewest within 10% of the fastest steady one-bar slide, timed in a second pass) |
| `full_window_bars` | int | `60` | the bars' full window, `window_seconds // bar_seconds` (set it to that: the engine's workspaces hold it, so growth windows never reallocate; a longer window grows them while data flows); process + cpu: the window the worker's BLAS tuning times |
| `worker_cpus` | list | `[]` | CPUs of the spawned worker (`pin_worker`); `[]` inherits |
| `directory` | `Ref` | `Ref()` | `Ref(Knob[FeedDirectory])`; process + cpu: the worker tunes its BLAS threads for each new universe as its manifest arrives, before the first window (in `start()` for a universe registered before the graph runs, as `replay` does); empty: at the universe's first window |
| `deadline_us` | int | `0` | `>0`: count poll ticks that started this late (`late_activations`): almost always the ticks right after an analysis that overran the period, not late windows |
| `stats` | `Ref` | `Ref()` | `Ref(RiskStats)`; private if empty |

## Behaviour

- One engine per operator: `start()` loads the GPU kernels, the first window of a universe
  runs once untimed to allocate the workspaces (for `full_window_bars`) and capture the
  CUDA graphs, and a new generation resets the engine. No later window allocates, so none
  can wake a runtime-suspended GPU.
- Inline GPU: the engine copies in only the new bars of a sliding window (the window's
  parts are immutable, so shared bars are proven by identity) and runs on its own
  non-blocking CUDA stream: every window, whatever its kind, is one launch of a CUDA
  graph and one device-to-host copy.
- The graphs are captured at a universe's first window, in `compute`. A device-wide
  synchronize from another thread during the capture (`Device().synchronize()`,
  `torch.cuda.synchronize()`) invalidates it, and fails that thread's call: the engine
  then runs the same work eagerly and captures again 256 windows later, but the other
  operator should synchronize its own stream instead.
- A validated window carries a host copy of its rows for the CPU comparison. Both engines
  rank by their own variance (the GPU's equals NumPy's bit for bit), so the comparison
  makes identical choices and times the same work.
- Outputs go through hold-and-retry: on `BackpressuredError` the update is retried at the
  next poll, before any new analysis.
- Process executor: the worker is spawned and pinged in `start()`; a worker exception sets
  `RiskStats.error` (`"Analytics failed: ..."`) and stops further analyses, and releases the
  gate if that window was validated (no comparison will come for it).
  `host_variance` needs the inline executor: the workers rank by their own variance. The
  worker keeps its last window, so only each window's new bars are sent to it.
  An inline exception fails the run.

## Edges and lane

- In: `LATEST` from `QuoteBarsOp.window`, another lane.
- `risk` → `CallbackSinkOp` (`ResultsStore.on_risk`): `lossless(8)`, same lane.
- `validate` → `RiskCrossCheckOp.risk`: `lossless(2)`, the `crosscheck` lane.
- Lane: the entry operator of `analytics`. It waits on GPU syncs or futures, which release
  the GIL. Inline GPU work runs on the engine's own stream, after the bars' explicit
  synchronize.

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
