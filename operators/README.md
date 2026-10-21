# Operators

Ten Holoscan SDK 5.0 operators for streaming market data. Each is a thin adapter (ports,
Params, contract, lifecycle) around a library processor in `holoscan_finance` that holds the
logic and is tested without Holoscan. Import them from `holoscan_finance.operators` (or the
module alias `holoscan.finance.operators`); edge presets live in
`holoscan_finance.operators.edges`. Each folder here is a catalog entry: `metadata.json` and a
`README.md` with the ports, Params, contract, recommended edges, lane and an example.

| Operator | Folder | What it does |
| --- | --- | --- |
| `GpuMitchSourceOp` | [gpu_mitch_source_op](gpu_mitch_source_op/README.md) | DAQIRI GPUDirect into GPU memory; native drain, GPU gather and MITCH decode |
| `GpuMitchReplaySourceOp` | [gpu_mitch_replay_source_op](gpu_mitch_replay_source_op/README.md) | NIC-free: pcap frames in GPU memory through the same gather, ring and decode |
| `UdpPacketSourceOp` | [udp_packet_source_op](udp_packet_source_op/README.md) | Kernel UDP socket (host baseline) |
| `DaqiriPacketSourceOp` | [daqiri_packet_source_op](daqiri_packet_source_op/README.md) | DAQIRI into pinned host memory (host baseline) |
| `MitchDecodeOp` | [mitch_decode_op](mitch_decode_op/README.md) | Python reference MITCH decode of host packet batches |
| `QuoteBarsOp` | [quote_bars_op](quote_bars_op/README.md) | Event-time quote bars on the GPU or with NumPy; rolling windows of returns |
| `RollingRiskOp` | [rolling_risk_op](rolling_risk_op/README.md) | Volatility, covariance and correlation of the newest window at a bounded rate |
| `RiskCrossCheckOp` | [risk_cross_check_op](risk_cross_check_op/README.md) | CPU recomputation and agreement check of validated GPU windows |
| `CallbackSinkOp` | [callback_sink_op](callback_sink_op/README.md) | Every item to a Python callable, on the producer's lane |
| `RiskSummaryOp` | [risk_summary_op](risk_summary_op/README.md) | Compact JSON risk summary over holomq (`--topics` only) |

## Data flow

The applications compose these operators in plain sight
([equity_analytics](../applications/equity_analytics/README.md),
[market_data_bench](../applications/market_data_bench/README.md)). Bracketed names are
authored lanes (runtime threads); the `market_data_bench --check` reference adds a
`reference` lane with a second `QuoteBarsOp(cpu)` fed by the same GPU source:

```text
[ingest]     GpuMitchSourceOp | GpuMitchReplaySourceOp ──events──▶ QuoteBarsOp(gpu) ──┐
             UdpPacketSourceOp | DaqiriPacketSourceOp ──packets──▶ MitchDecodeOp       │
                                        MitchDecodeOp ──events───▶ QuoteBarsOp(cpu) ──┤
                                                                     window (LATEST)  │
[analytics]  RollingRiskOp ◀──────────────────────────────────────────────────────────┘
               ├──risk──▶ CallbackSinkOp (ResultsStore.on_risk)
               ├──validate (compare mode)──▶ [crosscheck] RiskCrossCheckOp
               │                               └──comparison──▶ CallbackSinkOp
               │                                     (ResultsStore.on_comparison, which then releases the gate)
               └──risk (LATEST, --topics only)──▶ [topics] RiskSummaryOp
                                                   └──summary (LATEST)──▶ holomq export finance.risk
```

Decode is fused into the GPU sources: ring slots stay leased until decode synchronizes, so
the lease has exactly one consumer and no extra hop. On the host paths packets are owned
bytes, so decode is its own operator.

## Payloads

Every port is `hs.pyobject` (zero copy, shared across a fan-out) except
`RiskSummaryOp.summary`. Producers never touch a payload after `emit()`; consumers treat it
as read-only. GPU work runs on the CUDA legacy stream, and producers synchronize before
emitting device data, so no payload carries a CUDA event.

| Payload | From → to | Notes |
| --- | --- | --- |
| `PacketBatch` | host sources `.packets` → `MitchDecodeOp.packets` | `(datagram, recv_wall_ns, wire_count, peer)` per packet; host memory |
| `list[ReplayBatch]` | `MitchDecodeOp.events` → `QuoteBarsOp(cpu).events` | one batch per accepted packet (4.3 layout) |
| `list[GpuReplayBatch]` | GPU sources `.events` → `QuoteBarsOp.events` | always one element; columns in GPU memory, complete before emit |
| `BarWindow` | `QuoteBarsOp.window` → `RollingRiskOp.window`, `CallbackSinkOp` | per-bar return vectors never mutated after their bar closed; `rows` is stacked by the consumer, so a skipped window costs no GPU work |
| `RiskUpdate` | `RollingRiskOp.risk` / `.validate` → `CallbackSinkOp`, `RiskCrossCheckOp`, `RiskSummaryOp` | host results; `rows_host` on validated windows |
| `RiskComparison` | `RiskCrossCheckOp.comparison` → `CallbackSinkOp` | the GPU update and its CPU twin |
| `holoscan.schema.String` | `RiskSummaryOp.summary` → `export_topic` | compact JSON, at most 512 B, crosses processes |

Live objects (the feed directory `Knob`, stats holders, the cross-check gate, the capture
tap, callbacks) arrive as `holoscan_finance.ref.Ref` Params; decode and bars read the
directory once per activation, without a lock. Every stats `Ref` is optional: an operator
creates a private stats object when it is empty.

## Edges

| Edge | Preset | Why |
| --- | --- | --- |
| source → bars (GPU), source → decode → bars (host) | `lossless(4)` / `lossless(16)`, same lane | every packet; no thread hop per packet |
| bars → risk | `LATEST` (`DROP_OLDEST`, depth 1), across lanes | bars never block on analytics; risk analyzes the newest window |
| risk → publish | `lossless(8)`, same lane | every result |
| risk.validate → crosscheck → publish_comparison | `lossless(2)` | one comparison in flight (the gate) |
| bars → a window collector | `lossless(64)`, same lane | every window (bench, tests) |
| GPU source → reference `QuoteBarsOp(cpu)` (bench `--check`) | `lossless(8)`, the `reference` lane | a fan-out: the source retries until both consumers accept |
| risk → summary → export | `LATEST` | a stalled external reader can never fail the run |

Sources, risk and the cross-check hold and retry an output that meets a full `BLOCKING`
edge; decode and bars feed same-lane input-triggered operators or `DROP_OLDEST` edges, where
the runtime throttles them instead.

## The lane rule

Applications author every lane with `graph.partition(name)` and place it with
`ThreadPlacement`. Each lane must:

- be weakly connected;
- be convex: no path leaves it and comes back;
- have exactly one entry operator: a clock-triggered operator, or one fed only from other
  lanes or from an import topic. Every other member has an input edge from inside the lane.

That is why a `CallbackSinkOp` shares its producer's lane, why `RiskSummaryOp` (fed only from
the `analytics` lane) gets its own `topics` lane, and why the cross-check and its publisher
form the `crosscheck` lane. Every operator is authored, so the compiler never creates an
automatic partition; the applications check this and their edge policies on `plan.json()`
before running.
