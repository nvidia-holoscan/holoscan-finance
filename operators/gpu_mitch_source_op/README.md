# GpuMitchSourceOp

GPUDirect receive for MITCH market data. The NIC writes frames straight into GPU memory and
CUDA kernels unpack and decode them, so no packet payload ever reaches the CPU:

```text
ConnectX → DAQIRI RX buffers in GPU memory (dma-buf)
         → native drain thread hf-drain (no Python lock): gather_frame copies the UDP payloads
           into a device slot ring and checks MITCH framing; the burst returns to the NIC
         → compute(): receive accounting on a 7-column header per packet, staging,
           GPU MITCH decode, legacy-stream sync, ring release → events (list[GpuReplayBatch])
```

The drain ([daqiri_drain.cpp](../../python/holoscan_finance/ingest/native/daqiri_drain.cpp),
byte-identical to 4.3) is a C++ thread that never takes the Python interpreter lock. It
polls DAQIRI bursts, gathers them into a preallocated ring of `ring_size` device slots,
copies the per-packet header to pinned host memory and returns the burst to the NIC, with
no allocation in steady state. A Python thread that stalls while holding the lock therefore
cannot stop NIC buffers from being recycled: the ring absorbs the stall and `compute()`
catches up. When the ring is full the drain holds the burst, so overflow shows up as NIC
drops and sequence gaps, never as silently discarded packets. Decode is fused into the
source: ring rows stay leased until decode has synchronized, so the lease has exactly one
consumer. On IGX Thor this sustains 200,000 packets/s (6.6M MITCH messages/s, measured on
Holoscan 4.3); see [GPUDirect ingest](../../docs/GPU_INGEST.md).

Processor: `holoscan_finance.ingest.gpu_ingest.GpuIngest` with
`holoscan_finance.marketdata.gpu_decoder.GpuMitchDecoder`. Derived from the 4.3 operator of
the same name, `EquityState.decode_gpu` and `warmup.py`.

## Requirements

- CuPy, and DAQIRI with Python bindings that expose `get_packet_ptr` and
  `get_packet_length`, built with the DPDK engine
  ([deploy/igx_thor/Dockerfile](../../deploy/igx_thor/Dockerfile)).
- A ConnectX NIC with GPUDirect over dma-buf (no `nvidia_peermem`), hugepages, and the
  pybind11 headers to build the native drain on first start. Tested on IGX Thor (aarch64)
  with an RTX PRO 6000 and a ConnectX-7; platform setup is in
  [Thor deployment](../../deploy/igx_thor/README.md).

## Ports

| Port | Type | Notes |
| --- | --- | --- |
| `events` (out) | `list[GpuReplayBatch]` | always one batch; columns in GPU memory, complete before emit; per-packet sequence, wire count, offsets and receive times on the host |

Contract: `OnClock(poll_period_us)`; no idle sleep, the clock paces the source.

## Params

Shared with `GpuMitchReplaySourceOp`:

| Param | Type | Default | Meaning |
| --- | --- | --- | --- |
| `batch_size` | int | `131072` | max MITCH messages staged per batch |
| `ring_size` | int | `1048576` | device slots, 2 KiB each; covers a stall of `ring_size` / packet rate |
| `poll_period_us` | int | `500` | below 100 needs a busy-polled lane |
| `gpu_device` | int | `0` | CUDA device of the ring and kernels |
| `recv_timeout_s` | float | `5.0` | silence that ends the input when `keep_alive` is off |
| `keep_alive` | bool | `True` | `False`: end of input sets `stats.eof` once everything was emitted |
| `warmup` | bool | `True` | compile and run the gather and decode kernels in `start()` |
| `helper_cpus` | list | `[]` | mask the drain thread starts under; `[]` = the process mask, never the lane mask |
| `directory` | `Ref` | `Ref()` | `Ref(Knob[FeedDirectory])`; empty: every packet is stale |
| `stats` | `Ref` | `Ref()` | `Ref(IngestStats)`; private if empty |
| `tap` | `Ref` | `Ref()` | `Ref(CaptureTap)`; no capture recording if empty |

`GpuMitchSourceOp` only:

| Param | Type | Default | Meaning |
| --- | --- | --- | --- |
| `port` | int | `26400` | UDP destination port of the flow rule |
| `pcie_address` | str | `"0004:03:00.0"` | NIC PCIe address (non-empty) |
| `daqiri_config` | str | `""` | DAQIRI YAML path; overrides the built config and disables `isolate_cores` |
| `interface_name`, `queue_id` | str, int | `"rx_port"`, `0` | DAQIRI interface and RX queue |
| `master_core`, `rx_core` | int | `3`, `9` | DPDK lcores |
| `num_bufs`, `buf_size` | int | `114688`, `2048` | GPU receive buffers (bounded by BAR1) |
| `rx_batch_size`, `rx_timeout_us` | int | `4096`, `20000` | DAQIRI burst size and partial-burst timeout: few, large bursts |
| `daqiri_log_level` | str | `"warn"` | `trace`, `debug`, `info`, `warn`, `error`, `critical` or `off` |
| `native_drain` | bool | `True` | `False` (or `HOLOSCAN_FINANCE_NATIVE_DRAIN=0`): the Python drain, the A/B switch |
| `isolate_cores` | bool | `True` | after `daqiri_init`, move this process's threads off the two lcores |
| `init_cpus` | list | `[]` | mask to widen to around `daqiri_init` (the apps pass every CPU) |

CPU lists are validated in `setup()`: a list or tuple of ints ≥ 0.

## Behaviour

- **`start()` order.** Stats reset and lane CPUs recorded, kernels warmed, bindings checked;
  the native drain is built **before** `daqiri_init` (a cold build takes about 5 s, which must
  not pass with the NIC up and unpolled) and the ring allocated; `daqiri_init` runs inside
  `thread_mask(init_cpus or None)`, which always restores and checks the lane mask (DPDK's
  EAL pins its caller, and the runtime never re-applies a lane's placement); then the lcore
  sweep, the interface lookup, and the drain, started under `helper_mask(helper_cpus)`. A
  failure after a successful `daqiri_init` (the mask check included) stops a started drain
  and shuts DAQIRI down before it propagates.
- **Accounting.** Per burst, vectorized for one contiguous session (4.3 rules): sequence gaps
  (`dropped`, in messages), reordering, framing rejects, non-UDP frames; rows beyond
  `ring_size` queued packets count as `ring_dropped` messages.
- **Staging.** Consecutive ring rows are decoded through one zero-copy view; rejected rows,
  or rows that straddle the ring end, are gathered by copy.
- **Decode.** The directory is read immediately before decoding; packets from another sender
  port are stale. Ring rows are released only after the legacy stream synchronized.
- **Backpressure.** A full downstream edge is held and retried at the next activation
  (`PendingEmits`); a compute with a backlog emits nothing new.
- **Fallback.** Without the native helper the Python drain `gpu-mitch-drain` does the same
  with one binding call per packet and the GIL per burst (`frame_fetch = "per-packet
  binding"`); the 4.3 `burst_frames` helper is gone, so the native-vs-Python A/B now compares
  against this weaker baseline.

## Edges and lane

- Out: `lossless(4)` into `QuoteBarsOp(backend="gpu")`, in the same lane.
- Lane: the entry of `ingest`; the drain and DPDK threads run outside the graph.
- Threads: `hf-drain` (C++, never takes the GIL) or `gpu-mitch-drain` (Python); the DPDK
  lcores pinned by DPDK; the DAQIRI stats thread (moved off the lcores by the sweep).
- Streams: the drains gather on their own non-blocking stream and synchronize each burst;
  decode runs on the legacy stream.
- Errors: a `start()` failure fails the run; a drain error goes to `stats.error` and the next
  compute raises, so the run is `FAILED` with the message.

## Telemetry

`IngestStats` (4.3 names): `packets`, `received_messages`, `messages`, `dropped`,
`ring_dropped`, `malformed_packets`, `decode_malformed`, `stale_packets`,
`out_of_order_packets`, `non_udp_frames`, `bursts`, `last_packet_wall_ns`, `drain_host_ns`,
`frame_fetch`, `event_batches`, `retries`, `held`, `eof`, `error`, `lane_cpus`, and
`stalls` (`burst_gaps_ms`, `worst_burst`, `worst_compute`, `events`, and `native`:
`poll_gap_max_ms`, `poll_gap_at`, `ring_waits`, `ring_wait_ms` and every `poll_gaps_*`
count over 10, 50 and 100 ms), plus `feed_buckets()`, `ring_fill()`, `ring_peak()`,
`pending_packets()` and `drain_alive()`.

## Example

```python
from holoscan_finance.operators import GpuMitchSourceOp, QuoteBarsOp
from holoscan_finance.operators.edges import lossless
from holoscan_finance.ref import Ref

source = g.op(GpuMitchSourceOp, "source", port=26400, pcie_address="0004:03:00.0",
              init_cpus=sorted(cpus.all), helper_cpus=sorted(cpus.helper),
              directory=Ref(knob), stats=Ref(ingest_stats))
bars = g.op(QuoteBarsOp, "bars", backend="gpu", directory=Ref(knob), stats=Ref(bars_stats))
g.add_flow(source.events, bars.events, **lossless(4))
g.partition("ingest").add(source, bars)
```
