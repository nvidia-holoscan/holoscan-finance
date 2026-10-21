# UdpPacketSourceOp

MITCH datagrams from a kernel UDP socket, the host baseline that runs anywhere. A drain
thread moves every datagram into a host ring after only a header peek, so decode load can
never make the kernel drop packets; overload is counted as `ring_dropped` instead. Each
activation emits the oldest packets as one `PacketBatch` for `MitchDecodeOp`:

```text
socket → udp-drain: recvfrom, receive accounting, ring → packets (PacketBatch) → MitchDecodeOp
```

Processor: `holoscan_finance.ingest.host_receivers.UdpReceiver`. Derived from 4.3
`AsyncUdpSourceOp`, with decode moved into `MitchDecodeOp`.

## Ports

| Port | Type | Notes |
| --- | --- | --- |
| `packets` (out) | `PacketBatch` | `items`: `(datagram, recv_wall_ns, wire_count, peer)` in receive order; `messages`: their wire messages (up to `batch_size`, at least one packet); `index`: batch number since `start()` |

Contract: `OnClock(poll_period_us)`.

## Params

| Param | Type | Default | Meaning |
| --- | --- | --- | --- |
| `host` | str | `"127.0.0.1"` | address to bind |
| `port` | int | `26400` | UDP port to bind |
| `recv_buf` | int | `67108864` | `SO_RCVBUF` bytes (capped by `net.core.rmem_max`) |
| `ring_size` | int | `65536` | host ring, in packets |
| `batch_size` | int | `16384` | max messages per `PacketBatch` |
| `poll_period_us` | int | `500` | below 100 needs a busy-polled lane |
| `keep_alive` | bool | `True` | `False`: `recv_timeout_s` of silence ends the input (`stats.eof`) |
| `recv_timeout_s` | float | `5.0` | |
| `helper_cpus` | list | `[]` | mask the receiver thread starts under; `[]` = the process mask, never the lane mask |
| `stats` | `Ref` | `Ref()` | `Ref(IngestStats)`, shared with `MitchDecodeOp`; private if empty |

## Behaviour

- **Accounting** (4.3 rules, per datagram): sessions are keyed by sender address and MITCH
  group; a sequence above the expected one adds the gap to `dropped` (messages), one below it
  is out of order; malformed units are counted and not queued.
- **Backpressure.** A full edge to the decoder is held and retried at the next activation;
  a compute with a backlog emits nothing new.
- **End of input.** With `keep_alive` off, `stats.eof` is set once the socket was silent for
  `recv_timeout_s` and every packet was emitted. A warning names the sender tool when nothing
  arrived.

## Edges and lane

- Out: `lossless(16)` into `MitchDecodeOp.packets`, same lane.
- Lane: the entry of `ingest`.
- Threads: `udp-drain` (Python; `recvfrom` releases the GIL), created under
  `helper_mask(helper_cpus)`.
- Errors: a bind error fails `start()`; a receiver error goes to `stats.error` and the next
  compute raises.

## Telemetry

`IngestStats` (`front_end = "socket"`): `packets`, `received_messages`, `dropped`,
`out_of_order_packets`, `malformed_packets`, `ring_dropped`, `last_packet_wall_ns`,
`packet_batches`, `retries`, `held`, `eof`, `error`, `lane_cpus`, plus `feed_buckets()`,
`ring_fill()` and `ring_peak()`. `MitchDecodeOp` adds `messages`, `stale_packets`,
`decode_malformed`, `event_batches` and `packet_batches_decoded`.

## Example

```python
from holoscan_finance.operators import MitchDecodeOp, QuoteBarsOp, UdpPacketSourceOp
from holoscan_finance.operators.edges import lossless
from holoscan_finance.ref import Ref

source = g.op(UdpPacketSourceOp, "source", host="0.0.0.0", port=26400,
              helper_cpus=sorted(cpus.helper), stats=Ref(ingest_stats))
decode = g.op(MitchDecodeOp, "decode", directory=Ref(knob), stats=Ref(ingest_stats))
# nothing consumes bar windows here: emit_windows=False leaves `window` undeclared
bars = g.op(QuoteBarsOp, "bars", backend="cpu", emit_windows=False, directory=Ref(knob),
            stats=Ref(bars_stats))
g.add_flow(source.packets, decode.packets, **lossless(16))
g.add_flow(decode.events, bars.events, **lossless(16))
g.partition("ingest").add(source, decode, bars)
```
