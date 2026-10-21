# RiskCrossCheckOp

The CPU twin of every validated GPU risk window. Each validated `RiskUpdate` carries a host
copy of its window; the operator recomputes it with `cpu_window` in one spawned worker
(pinned, BLAS threads tuned over {1, 2, 4, 8, all}) and emits the comparison with the GPU
result. The dashboard's live CPU/GPU speedup (`rolling_timing`) comes from these pairs; the
published 9.55x was measured in separate CPU-only and GPU-only runs of the same workers.

```text
risk (validated RiskUpdate) → FIFO → spawned cpu_window → comparison (RiskComparison)
```

Processor: `holoscan_finance.analytics.crosscheck.CpuCrossCheck`. Derived from the compare
branch of 4.3 `PublishOp` (spawn pool, `cpu_window`, 5 ms poll) and the rules of
`publish_comparison` (`analytics.agreement.compare_results`).

## Ports

| Port | Type | Notes |
| --- | --- | --- |
| `risk` (in) | `RiskUpdate` | validated, with `rows_host` (any other update fails the run) |
| `comparison` (out) | `RiskComparison` | `update`, `cpu` (the `cpu_window` result), `agreement`, `matched_work`, `usable`, `error` |

`agreement`: same eligible and displayed instruments, and volatilities, covariance probes and
heatmap correlations within the 4.3 tolerances. `matched_work`: same method, window and
incremental additions/expirations. `usable` (timings comparable): both, and at least two
eligible names. A worker exception gives `cpu=None` and `error`.

Contract: `OnClock(poll_period_us).input(risk, DrainAll())`.

## Params

| Param | Type | Default | Meaning |
| --- | --- | --- | --- |
| `bar_seconds` | int | `5` | annualizes volatility |
| `cpu_threads` | int | `0` | BLAS threads; `0` tunes at start and at the first full window |
| `full_window_bars` | int | `60` | bars of a full window |
| `poll_period_us` | int | `5000` | the poll clock (4.3 `PublishOp` period) |
| `worker_cpus` | list | `[]` | CPUs of the worker (`pin_worker`); `[]` inherits. The apps pass `CpuSets.worker`, the 4.3 core set |
| `stats` | `Ref` | `Ref()` | `Ref(RiskStats)`: writes `crosschecks_submitted` and `crosschecks_done` only; private if empty |

## Edges and lane

- In: `lossless(2)` from `RollingRiskOp.validate`, another lane. The gate keeps one update in
  flight; the queue is FIFO anyway.
- Out: `lossless(2)` to a `CallbackSinkOp` (`ResultsStore.on_comparison`) in the same lane;
  that callback publishes the comparison and then releases the gate.
- Lane: the entry operator of `crosscheck`.
- Threads: the spawned worker process and the pool's management thread. The worker is spawned
  and pinged in `start()`.

## Example

```python
from holoscan_finance.operators import CallbackSinkOp, RiskCrossCheckOp
from holoscan_finance.operators.edges import lossless
from holoscan_finance.ref import Ref

crosscheck = g.op(RiskCrossCheckOp, "crosscheck", bar_seconds=5, full_window_bars=60,
                  worker_cpus=sorted(cpus.worker), stats=Ref(risk_stats))
publish_comparison = g.op(CallbackSinkOp, "publish_comparison", callback=Ref(results.on_comparison))
g.add_flow(risk.validate, crosscheck.risk, **lossless(2))
g.add_flow(crosscheck.comparison, publish_comparison.items, **lossless(2))
g.partition("crosscheck").add(crosscheck, publish_comparison)
```
