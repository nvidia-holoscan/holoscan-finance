# GpuMitchSourceOp

GPUDirect receive for MITCH market data. The NIC writes packets straight into GPU
memory and CUDA kernels unpack them, so packet payloads never touch the CPU:

```text
ConnectX → DAQIRI RX buffers in GPU memory (dma-buf)
         → native drain thread (no Python lock): gather_frame kernel copies UDP payloads
           into a device slot ring and checks MITCH framing; the burst returns to the NIC
         → compute(): loss accounting from a 7-field header per packet,
           batch_decoder on the device slots → emit on "tick"
```

The drain is a C++ thread ([daqiri_drain.cpp](../../python/holoscan/marketdata/native/daqiri_drain.cpp))
that never takes the Python interpreter lock. It polls DAQIRI bursts, launches the gather
kernel into a preallocated ring of `ring_size` device slots, copies the per-packet header
to pinned host memory and returns the burst to the NIC, with no allocation in steady
state. A Python thread that stalls while holding the lock (the cause of the occasional
full-day loss before this drain) no longer stops the NIC buffers from being recycled: the
ring keeps filling and `compute()` catches up afterwards. When the ring is full the drain
holds the burst, so overflow appears as NIC drops and sequence gaps, never as silently
discarded packets. `compute()` accounts the gathered bursts, stages them up to
`batch_size` messages and calls the decoder once per batch.
On IGX Thor this sustains 200,000 packets/s (6.6M MITCH messages/s); see
[GPUDirect ingest](../../docs/GPU_INGEST.md) for the data path, measurements and limits.

## Requirements

- Holoscan 4.3+, CuPy, and DAQIRI with Python bindings that expose `get_packet_ptr` and
  `get_packet_length`, built with the DPDK engine.
- A ConnectX NIC with GPUDirect over dma-buf (no `nvidia_peermem`), and hugepages.
- Tested on IGX Thor (aarch64) with an RTX PRO 6000 and ConnectX-7. DAQIRI on Thor
  needs [daqiri_thor_steering.patch](../../deploy/igx_thor/daqiri_thor_steering.patch);
  platform setup is in [Thor deployment](../../deploy/igx_thor/README.md).

## Arguments

| Argument | Default | Purpose |
| --- | --- | --- |
| `batch_decoder` | required | `(slots, lengths, src_port, recv_ns) → (batches, decoded, malformed)`; `batches` is emitted on `tick` |
| `port`, `pcie_address` | `26400`, `0004:03:00.0` | UDP destination port for the flow match; NIC PCIe address |
| `daqiri_config` | built | Custom DAQIRI config (dict or YAML path) with the same interface and queue |
| `interface_name`, `queue_id` | `rx_port`, `0` | DAQIRI interface and RX queue |
| `master_core`, `rx_core` | `3`, `9` | DPDK cores; the operator moves its process's other threads off them |
| `num_bufs`, `buf_size` | `51200`, `2048` | GPU receive buffers (at most `SLOT_BYTES` each) |
| `rx_batch_size`, `rx_timeout_us` | `4096`, `20000` | DAQIRI burst size and partial-burst timeout |
| `ring_size`, `batch_size` | `65536`, `16384` | Device ring slots (2 KiB each; covers a stall of `ring_size` / packet rate); messages per decode |
| `native_drain` | `True` | Use the native drain; `False`, or `HOLOSCAN_FINANCE_NATIVE_DRAIN=0`, selects the Python drain thread |
| `keep_alive`, `recv_timeout` | `True`, `5.0` | Keep listening, or end the stream after this many idle seconds |
| `gpu_device` | `0` | CUDA device for the slots and kernels |

`batch_decoder` receives a CuPy `(n, 2048)` uint8 slot buffer, CuPy payload lengths and
NumPy source ports and receive times. It would normally call
`holoscan.marketdata.gpu_mitch.decode_packets`, which returns per-packet status and
device-resident observation columns. The application's `EquityState.decode_gpu`
([stream.py](../../python/holoscan/equities/stream.py)) is a complete example: it drops
packets from stale senders, decodes, and returns GPU quote-bar batches.

## Counters and diagnostics

`packets`, `received_messages` (wire messages), `messages` (decoded), `dropped`
(sequence gaps), `ring_dropped`, `malformed_packets`, `out_of_order_packets`,
`non_udp_frames` and `bursts`, plus `feed_buckets()` (100 ms arrival counts),
`ring_fill()`, `ring_peak()` (most packets held since `reset_stalls()`) and `drain_alive()`. A new session (sender address or MITCH group) restarts sequence tracking.

`stalls` records the longest gaps between NIC bursts, the slowest drain and decode
passes with per-step times, and every stall over 50 ms; `reset_stalls()` clears it. A
drain stall longer than the NIC buffer pool (`num_bufs` packets) overflows the receive
queue, which DAQIRI logs as `Rx: Dropped N packets`.

The native drain and the frame-address helper
([daqiri_frames.cpp](../../python/holoscan/marketdata/native/daqiri_frames.cpp)) are
compiled on first start against the installed DAQIRI and CUDA headers. If the drain cannot
be built or started, a Python drain thread does the same work (allocating per burst, and
stopping whenever another Python thread holds the lock); if the helper cannot be built
either, that thread makes one binding call per packet. `HOLOSCAN_FINANCE_NATIVE=0`
disables both. `frame_fetch` names the path in use.
