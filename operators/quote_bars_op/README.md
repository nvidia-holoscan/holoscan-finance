# QuoteBarsOp

Event-time quote bars, on the GPU or with NumPy. The operator keeps each instrument's last
valid quote midpoint, last trade and volume, and closes a bar every `bar_seconds` of market
time with one log return per instrument. After a bar closes it emits the rolling window of
returns.

```text
events (list[GpuReplayBatch] | list[ReplayBatch]) → generation and sequence checks → bars → window (BarWindow)
```

With `backend="gpu"` the bars stay in GPU memory (`GpuQuoteBars`). `gpu_validate --bars`
checks the kernels against the NumPy reference `QuoteBars`: marks, quote times, trades,
volumes and counters bit for bit, returns within 1 ULP (CUDA vs NumPy `log`)
([correctness](../../docs/GPU_INGEST.md#correctness)). With `backend="cpu"` the host
`QuoteBars` runs behind the host sequence rules (`marketdata.sequence`). The cpu backend
also accepts `GpuReplayBatch`es (one D2H per batch): that is the in-graph reference of
`market_data_bench --check`, never a production path.

Processor: `holoscan_finance.marketdata.bar_builder.BarBuilder`.

## Ports

| Port | Type | Notes |
| --- | --- | --- |
| `events` (in) | `list[GpuReplayBatch]` (GPU sources) or `list[ReplayBatch]` (`MitchDecodeOp`) | |
| `window` (out) | `BarWindow` | declared only with `emit_windows`; `parts` (per-bar return vectors, CuPy on the gpu backend), `times`, `generation`, `completed_bars`, `recv_ns`; `rows` is stacked by the consumer |

Contract: `OnEach(events)`.

## Params

| Param | Type | Default | Meaning |
| --- | --- | --- | --- |
| `backend` | str | `"gpu"` | `gpu` or `cpu` |
| `bar_seconds`, `window_seconds` | int | `5`, `300` | bar length and rolling window |
| `stale_seconds` | int | `30` | a quote older than this gives no mark at the bar close |
| `min_bars` | int | `20` | returns needed before the first window |
| `max_spread_bps` | float | `100.0` | quotes with a wider spread are invalid |
| `gpu_device` | int | `0` | CUDA device; ignored by the cpu backend |
| `warmup` | bool | `True` | gpu: compile and run the bar kernels in `start()` |
| `emit_windows` | bool | `True` | declare `window`, which must then be wired: an unwired `window` compiles cleanly but raises `NotReadyError` at the first window (the rule `holoscan_finance.app.plan.outputs_connected()` catches it before the run); `False` when nothing consumes windows |
| `snapshot_period_ms` | int | `250` | symbol snapshot cadence; `0` = none during the run |
| `directory` | `Ref` | `Ref()` | `Ref(Knob[FeedDirectory])`; empty: input is counted and ignored |
| `stats` | `Ref` | `Ref()` | `Ref(BarsStats)`; private if empty |

The five bar Params are validated with the `QuoteBars` rules at `graph.op`
(`BarSettings`).

## Behaviour

- **Generations.** The directory is read once per input message. A new generation resets
  the bars (new engine, next expected sequence 1) at the first message decoded under it;
  messages of the bars' generation still in flight when the directory moved are built
  first, and batches of any other generation are stale. The dashboard hides the old
  generation until then.
- **Sequences.** A sequence below the expected one is stale; a gap resets the marks, so no
  return spans missing updates.
- **Host mirrors.** After every batch the exact counters go to `BarsStats.counters` (an
  immutable `BarCounters`; gpu: one 40-byte D2H), and at most every `snapshot_period_ms`, on
  the first batch and after a reset, a `SymbolSnapshot` of mid, trade, volume and quote time
  (gpu: one batched D2H). The HTTP thread reads only these objects, never CUDA.
- **Final state.** `stop()` runs on the lane thread after the last batch and publishes the
  final counters and snapshot, so finite runs never read a stale snapshot.

## Edges and lane

- In: `lossless(4)` from a GPU source or `lossless(16)` from `MitchDecodeOp`, same lane.
- Out: `LATEST` to `RollingRiskOp` in another lane (bars never block on analytics; risk
  skips windows), or `lossless(64)` to a same-lane `CallbackSinkOp` when every window matters.
- Lane: the source's (`ingest`), or the bench's `reference` lane.
- Streams: the legacy stream, plus the engine's per-batch D2H of segment starts.
- Errors: an unexpected `BackpressuredError` propagates (the run fails loudly); a wrong batch
  type raises `TypeError`.

## Telemetry

`BarsStats`: `generation`, `symbols`, `device`, `batches_in`, `stale_packets`,
`sequence_gaps`, `expected_sequence`, `windows_emitted`, `last_window_bars`,
`last_window_key` (`(generation, completed_bars)`), `counters`, `symbol_snapshot`,
`lane_cpus`.

## Example

Every window of a NIC-free replay, in order (the dashboard app instead sends `bars.window` to
`RollingRiskOp` over `LATEST`):

```python
from holoscan_finance.operators import CallbackSinkOp, GpuMitchReplaySourceOp, QuoteBarsOp
from holoscan_finance.operators.edges import lossless
from holoscan_finance.ref import Ref
from holoscan_finance.settings import BarSettings

source = g.op(GpuMitchReplaySourceOp, "source", pcap_path="day.pcap", directory=Ref(knob))
bars = g.op(QuoteBarsOp, "bars", backend="gpu", **BarSettings().as_params(),
            directory=Ref(knob), stats=Ref(bars_stats))
windows = g.op(CallbackSinkOp, "windows", callback=Ref(collected.append))
g.add_flow(source.events, bars.events, **lossless(4))
g.add_flow(bars.window, windows.items, **lossless(64))
g.partition("ingest").add(source, bars, windows)
```
