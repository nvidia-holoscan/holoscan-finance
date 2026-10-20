# GpuQuoteBarsOp

Event-time quote bars on the GPU. Decoded observations arrive in GPU memory; the
operator keeps each instrument's last valid quote midpoint, last trade and volume, and
closes a bar every `bar_seconds` of market time with a log return per instrument. Bars
stay in GPU memory and leave as a rolling window:

```text
events (GpuReplayBatch list) → sequence and generation checks → bar kernels → window (BarWindow)
```

The bar kernels (`bars_scatter`, `bars_apply`, `bars_close` in
[gpu_bars.py](../../python/holoscan/marketdata/gpu_bars.py)) match the NumPy reference
`holoscan.marketdata.bars.QuoteBars` bit for bit over a full recorded day; returns are
within 1 ULP ([validation](../../docs/GPU_INGEST.md#correctness)).

## Ports

| Port | Type | Notes |
| --- | --- | --- |
| `events` (in) | list of `GpuReplayBatch` | GPU observation columns plus per-packet sequence numbers, as `GpuMitchSourceOp`'s decoder emits |
| `window` (out) | `BarWindow` | After a bar closes: `rows` (bars x instruments, CuPy, stacked on first access), `times`, `generation`, `completed_bars`, `recv_ns` |

`window` never blocks. Connect it to an input that keeps only the newest message, as
`GpuRollingRiskOp`'s does, so slow analytics skip windows rather than stall ingest.

## Arguments and state

| Argument | Default | Purpose |
| --- | --- | --- |
| `bar_seconds`, `window_seconds` | `5`, `300` | Bar length and rolling window length |
| `stale_seconds` | `30` | A quote older than this gives no mark at the bar close |
| `min_bars` | `20` | Returns needed before the first window is emitted |
| `max_spread_bps` | `100` | Quotes with a wider spread are invalid |
| `lock` | own `RLock` | Guards the bar state; pass the application's lock to read `bars` from other threads |

`reset(symbols, generation)` starts a new instrument universe and feed generation; call
it when the feed restarts or seeks. Within a generation, stale packets are skipped and a
sequence gap clears the marks, so no return spans missing updates. Counters:
`stale_packets`, `sequence_gaps`, and `first_batch_ns` / `last_batch_ns` (arrival of the
generation's first and last batch). `bars` exposes the per-instrument state and event
counters.
