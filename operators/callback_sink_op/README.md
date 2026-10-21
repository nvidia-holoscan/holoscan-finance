# CallbackSinkOp

Hands every received item to a Python callable, in order, on the producer's runtime lane. It
is the one sink of the module: risk results and comparisons go to the dashboard's
`ResultsStore`, and the bench collects bar windows with it.

```text
items (any pyobject) → callback(item)
```

Derived from the publication half of 4.3 `PublishOp`. Results are published as soon as risk
emits them, where 4.3 polled every 5 ms.

## Ports

| Port | Type | Notes |
| --- | --- | --- |
| `items` (in) | any pyobject | |

Contract: `OnEach(items)`.

## Params

| Param | Type | Default | Meaning |
| --- | --- | --- | --- |
| `callback` | `Ref` | `Ref()` | `Ref(callable(item))`; required: `graph.op` fails without a callable |

A bound method in a `Ref` is shared, not copied (Holoscan deep-copies every Param value, and
`Ref` deep-copies to itself), so `Ref(results.on_risk)` reaches the application's store.

## Errors

An exception from the callback fails the run, with its traceback in the session
diagnostics. Callbacks that must not stop the graph (`ResultsStore.on_risk`,
`on_comparison`) catch internally, record `"Analytics failed: ..."` and release the gate.

## Edges and lane

- `lossless(8)` from `RollingRiskOp.risk`, `lossless(2)` from `RiskCrossCheckOp.comparison`,
  `lossless(64)` from `QuoteBarsOp.window` when every window matters (bench, tests).
- Lane: the producer's. Fed only from another lane it would be that lane's second entry
  operator, so it would need its own lane.
- Threads, streams: none.

## Example

```python
from holoscan_finance.operators import CallbackSinkOp
from holoscan_finance.operators.edges import lossless
from holoscan_finance.ref import Ref

publish = g.op(CallbackSinkOp, "publish", callback=Ref(results.on_risk))
g.add_flow(risk.risk, publish.items, **lossless(8))
g.partition("analytics").add(risk, publish)
```
