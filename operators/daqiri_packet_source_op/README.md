# DaqiriPacketSourceOp

MITCH datagrams received through DAQIRI into pinned host memory: the DAQIRI host-memory
baseline (4,500 packets/s with Python decode, measured on Holoscan 4.3). The NIC DMAs
flow-matched frames into host buffers; a drain thread copies each frame out, strips the
Ethernet/IPv4/UDP headers, accounts it and queues the datagram, and returns every burst to
DAQIRI at once. Each activation emits the oldest packets as one `PacketBatch` for
`MitchDecodeOp`:

```text
ConnectX → DAQIRI RX buffers in pinned host memory
         → daqiri-host-drain: get_packet_bytes, strip_udp_datagram, receive accounting, ring
         → packets (PacketBatch) → MitchDecodeOp
```

Processor: `holoscan_finance.ingest.host_receivers.DaqiriHostReceiver`. Derived from 4.3
`AnoSourceOp` (`--ingest ano`), with decode moved into `MitchDecodeOp`.

## Requirements

DAQIRI with Python bindings (`get_packet_bytes`), built with the DPDK engine
([deploy/igx_thor/Dockerfile](../../deploy/igx_thor/Dockerfile)); a ConnectX NIC and
hugepages. Without the bindings `start()` fails explicitly; there is no socket fallback.

## Ports

| Port | Type | Notes |
| --- | --- | --- |
| `packets` (out) | `PacketBatch` | as `UdpPacketSourceOp`; `peer` is the frame's source address |

Contract: `OnClock(poll_period_us)`.

## Params

| Param | Type | Default | Meaning |
| --- | --- | --- | --- |
| `port` | int | `26400` | UDP destination port of the flow rule |
| `pcie_address` | str | `"0004:03:00.0"` | NIC PCIe address (non-empty) |
| `daqiri_config` | str | `""` | DAQIRI YAML path; overrides the built config (and its lcores) |
| `interface_name`, `queue_id` | str, int | `"rx_port"`, `0` | DAQIRI interface and RX queue |
| `master_core`, `rx_core` | int | `3`, `9` | DPDK lcores |
| `num_bufs`, `buf_size` | int | `51200`, `2048` | pinned host receive buffers |
| `rx_batch_size`, `rx_timeout_us` | int | `1024`, `5000` | DAQIRI burst size and partial-burst timeout |
| `daqiri_log_level` | str | `"warn"` | `trace`, `debug`, `info`, `warn`, `error`, `critical` or `off` |
| `isolate_cores` | bool | `False` | move this process's threads off the lcores after `daqiri_init` (4.3 `ano` never did, and its baseline was measured that way) |
| `init_cpus` | list | `[]` | mask to widen to around `daqiri_init` (the apps pass every CPU) |
| `ring_size` | int | `65536` | host ring, in packets |
| `batch_size` | int | `16384` | max messages per `PacketBatch` |
| `poll_period_us` | int | `500` | below 100 needs a busy-polled lane |
| `keep_alive` | bool | `True` | `False`: `recv_timeout_s` of silence ends the input (`stats.eof`) |
| `recv_timeout_s` | float | `5.0` | |
| `helper_cpus` | list | `[]` | mask the drain thread starts under; `[]` = the process mask, never the lane mask |
| `stats` | `Ref` | `Ref()` | `Ref(IngestStats)`, shared with `MitchDecodeOp`; private if empty |

CPU lists are validated in `setup()`: a list or tuple of ints ≥ 0.

## Behaviour

- **`start()`.** Bindings checked; `daqiri_init` (memory kind `host_pinned`, engine
  `dpdk`) runs inside `thread_mask(init_cpus or None)`, which always restores and checks the
  lane mask, because DPDK's EAL pins its caller and the runtime never re-applies a lane's
  placement; then the optional lcore sweep, the interface lookup and the drain thread,
  started under `helper_mask(helper_cpus)`. A failure after a successful `daqiri_init` (the
  mask check included) stops a started drain and shuts DAQIRI down.
- **Drain.** Per burst: one `get_packet_bytes` copy per frame, IPv4/UDP stripping that
  respects the IHL and datagram lengths (non-IPv4 and fragments are skipped), the 4.3
  receive accounting, then `free_all_packets_and_burst_rx`, also when a read fails.
- **Backpressure and end of input** as `UdpPacketSourceOp`.

## Edges and lane

- Out: `lossless(16)` into `MitchDecodeOp.packets`, same lane.
- Lane: the entry of `ingest`.
- Threads: `daqiri-host-drain` (Python); the DPDK lcores, pinned by DPDK.
- Errors: a `start()` failure fails the run; a fatal RX status or a failed packet read goes
  to `stats.error` (`DAQIRI RX failed: ...`) and the next compute raises.

## Telemetry

`IngestStats` (`front_end = "daqiri-host"`), with the same fields as `UdpPacketSourceOp`.

## Example

```python
from holoscan_finance.operators import DaqiriPacketSourceOp, MitchDecodeOp, QuoteBarsOp
from holoscan_finance.operators.edges import lossless
from holoscan_finance.ref import Ref

source = g.op(DaqiriPacketSourceOp, "source", port=26400, pcie_address="0004:03:00.0",
              init_cpus=sorted(cpus.all), helper_cpus=sorted(cpus.helper),
              stats=Ref(ingest_stats))
decode = g.op(MitchDecodeOp, "decode", directory=Ref(knob), stats=Ref(ingest_stats))
bars = g.op(QuoteBarsOp, "bars", backend="cpu", directory=Ref(knob), stats=Ref(bars_stats))
g.add_flow(source.packets, decode.packets, **lossless(16))
g.add_flow(decode.events, bars.events, **lossless(16))
g.partition("ingest").add(source, decode, bars)
```
