# MitchDecodeOp

The Python reference MITCH decoder of the host front ends. Each `PacketBatch` from
`UdpPacketSourceOp` or `DaqiriPacketSourceOp` is decoded under the current feed directory into
one `ReplayBatch` per accepted packet. This is the host baseline, about 0.27M msgs/s on one
core (`decode_bench`, 2026-10-20); the GPU front ends decode inside the source instead.

```text
packets (PacketBatch) → sender-port check → MIT303 unit parse → NYQ mapping → events (list[ReplayBatch])
```

Processor: `holoscan_finance.nyq.host_decoder.HostMitchDecoder`.

## Ports

| Port | Type | Notes |
| --- | --- | --- |
| `packets` (in) | `PacketBatch` | `items`: `(datagram, recv_wall_ns, wire_count, peer)` in receive order |
| `events` (out) | `list[ReplayBatch]` | one batch per accepted packet; nothing is emitted when no packet was accepted |

Contract: `OnEach(packets)`.

## Params

| Param | Type | Default | Meaning |
| --- | --- | --- | --- |
| `directory` | `Ref` | `Ref()` | `Ref(Knob[FeedDirectory])`, read once per batch right before decoding; empty: every packet is stale |
| `stats` | `Ref` | `Ref()` | `Ref(IngestStats)`, normally the source's; private if empty |
| `tap` | `Ref` | `Ref()` | `Ref(CaptureTap)`; an active capture records every decoded list |

## Rules

- No directory, or a sender port other than `directory.sender_port`: the packet is stale.
- Otherwise `nyq.mapping.decode_packet` runs under the directory's generation, market
  midnight and universe; `ValueError` or `struct.error` counts the packet as malformed.

## Edges and lane

- In: `lossless(16)` from the host source; out: `lossless(16)` into `QuoteBarsOp(backend="cpu")`.
- Lane: `ingest`, with the source and the bars, so no packet crosses a thread.
- Threads, streams: none.

## Telemetry

Writes only these `IngestStats` fields (the sources never write them): `messages`,
`stale_packets`, `decode_malformed`, `event_batches`, `packet_batches_decoded`. The last one
is written last, so a finite run's drained check never runs ahead of the event batches.

## Example

```python
from holoscan_finance.operators import MitchDecodeOp, QuoteBarsOp, UdpPacketSourceOp
from holoscan_finance.operators.edges import lossless
from holoscan_finance.ref import Ref

directory, ingest = Ref(directory_knob), Ref(ingest_stats)
source = g.op(UdpPacketSourceOp, "source", port=26400, stats=ingest)
decode = g.op(MitchDecodeOp, "decode", directory=directory, stats=ingest)
# nothing consumes bar windows here: emit_windows=False leaves `window` undeclared
bars = g.op(QuoteBarsOp, "bars", backend="cpu", emit_windows=False, directory=directory)
g.add_flow(source.packets, decode.packets, **lossless(16))
g.add_flow(decode.events, bars.events, **lossless(16))
g.partition("ingest").add(source, decode, bars)
```
