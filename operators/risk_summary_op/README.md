# RiskSummaryOp

A compact JSON summary of every risk update, for other processes over a holomq topic. Used
only with `equity_analytics --topics`; off by default and absent from every measured
configuration.

```text
risk (RiskUpdate) → {"g","t","n","vol","corr","ms","bars","r2r"} → summary (String, ≤ max_bytes) → holomq://finance.risk
```

| Key | Value |
| --- | --- |
| `g` | generation |
| `t` | close time of the newest bar (ns) |
| `n` | eligible instruments |
| `vol` | median annualized volatility |
| `corr` | mean pairwise correlation |
| `ms` | engine milliseconds |
| `bars` | completed bars |
| `r2r` | receive-to-result milliseconds at summary time |

Trailing keys are dropped until the encoding fits `max_bytes`; `g` and `t` always stay.

## Ports

| Port | Type | Notes |
| --- | --- | --- |
| `risk` (in) | `RiskUpdate` | |
| `summary` (out) | `holoscan.schema.String` | at most `max_bytes`: a `String` over 512 B makes `emit` raise |

Contract: `OnEach(risk)`.

## Params

| Param | Type | Default | Meaning |
| --- | --- | --- | --- |
| `max_bytes` | int | `480` | range 64-512 |
| `stats` | `Ref` | `Ref()` | `Ref(RiskStats)`: writes `summaries` and `summary_drops` only; private if empty |

## Drop, never block

This is the one operator that drops instead of holding. A summary is a latest value, its
reader is an external process the graph does not control, and the rate is at most 20 Hz. So
an emit that raises `BackpressuredError` or `NotReadyError` is counted in
`RiskStats.summary_drops` and the run goes on. The export edge is always `LATEST`: with a
`BLOCKING` export one reader that attached and stopped reading failed the whole dashboard run
with `HOLOMQ_BACKPRESSURED` after 17 emits.

## Edges and lane

- In: `LATEST` from `RollingRiskOp.risk`, a fan-out beside the publish edge.
- Out: `LATEST` into `export_topic(..., **edges.topic_export_kwargs())` (optional peer, a
  10 s attach bound, bounded subscribers).
- Lane: `topics`, of which it is the entry operator.

## Example

```python
import holoscan as hs
from holoscan_finance.operators import RiskSummaryOp
from holoscan_finance.operators.edges import LATEST, topic_export_kwargs
from holoscan_finance.ref import Ref

summary = g.op(RiskSummaryOp, "risk_summary", stats=Ref(risk_stats))
g.add_flow(risk.risk, summary.risk, **LATEST)
topic = g.export_topic("holomq://finance.risk", hs.payload.schema.String,
                       endpoint_name="finance.risk", **topic_export_kwargs())
g.add_flow(summary.summary, topic.input, **LATEST)
g.partition("topics").add(summary)
```

Read it with `python -m holoscan_finance.tools.topic_tail --catalog PATH` from the engine's
IPC namespace (`docker exec`, or `--ipc=container:<engine>`; never `--ipc=host`).
