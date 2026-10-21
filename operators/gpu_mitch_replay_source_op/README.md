# GpuMitchReplaySourceOp

The NIC-free GPU MITCH source. It replays the Ethernet frames of a pcap as a software NIC:
the frames are uploaded once into GPU memory, as GPUDirect leaves received frames, and
everything after the drain is the production code of `GpuMitchSourceOp`:

```text
pcap → GPU memory (one copy) → software NIC hf-replay: DAQIRI-sized bursts at pps,
       the same gather_frame kernel into the same device slot ring
     → compute(): receive accounting, staging, GPU MITCH decode, sync, ring release
     → events (list[GpuReplayBatch])
```

The drain (`holoscan_finance.ingest.drains.DeviceReplayDrain`, thread `hf-replay`)
implements the native drain's protocol (records, ring positions, releases, statistics). It
closes a burst at `burst_packets` frames or `rx_timeout_us` after its first frame was due,
as DAQIRI does, never wraps a burst around the ring end, and holds a burst while the ring is
full, sleeping on a condition that each release signals (a full ring costs it no CPU).
This is how the whole 5.0 GPU graph is run, tested and A/B-compared without a NIC: bit
identity, throughput headroom and paced lossless gates. The drain is Python, so it is not a
test of resilience to GIL stalls; those claims stay with the native drain on the NIC.

Processor: `holoscan_finance.ingest.gpu_ingest.GpuIngest` with `DeviceReplayDrain`. New in
the 5.0 port; frame reading from 4.3 `gpu_validate.read_pcap`, loop and generation logic
from `replay.py`.

## Requirements

A CUDA GPU and CuPy. No NIC, DAQIRI or hugepages. The capture must fit in GPU memory next
to the ring (`ring_size` × 2 KiB).

## Ports

| Port | Type | Notes |
| --- | --- | --- |
| `events` (out) | `list[GpuReplayBatch]` | as `GpuMitchSourceOp` |

Contract: `OnClock(poll_period_us)`.

## Params

The shared Params of [GpuMitchSourceOp](../gpu_mitch_source_op/README.md#params)
(`batch_size`, `ring_size`, `poll_period_us`, `gpu_device`, `recv_timeout_s`, `warmup`,
`helper_cpus`, `directory`, `stats`, `tap`), plus:

| Param | Type | Default | Meaning |
| --- | --- | --- | --- |
| `pcap_path` | str | `""` | the capture; required (`setup()` raises if empty, `start()` if unreadable) |
| `pps` | int | `200000` | packets/s; `0` = unpaced (as fast as the ring and the consumer allow) |
| `loops` | int | `1` | passes over the capture; `0` = forever |
| `burst_packets` | int | `4096` | DAQIRI `rx_batch_size`; at most `ring_size` |
| `rx_timeout_us` | int | `20000` | DAQIRI partial-burst timeout |
| `nic_bufs` | int | `0` | `0`: hold forever. Otherwise a NIC buffer model (paced only): due frames beyond `nic_bufs` are dropped oldest first and show up as sequence gaps. A forming burst occupies buffers too, so it must cover `pps × rx_timeout` |
| `start_delay_ms` | int | `0` | delay before the first frame is due |
| `on_loop` | `Ref` | `Ref()` | `callable(pass_index) -> sender port \| None`, called only before pass k ≥ 1, after everything gathered was released; a returned port is written into every frame's UDP source port |
| `keep_alive` | bool | `False` | `False`: the input ends after the last pass (`stats.eof`) |

## Behaviour

- **Schedule.** Frame `i` of a pass is due at `t0 + i / pps`; the burst deadline is
  `min(due(first) + rx_timeout, due(last))`, and a burst holds the frames due by then.
  `lag` is how late a burst was gathered against its deadline.
- **Loops.** Before pass k ≥ 1 the drain waits until the consumer released every row (ring
  and staging empty), calls `on_loop(k)` (`ReplayGenerations` registers a new generation on
  a fresh sender port) and re-bases the schedule, so no packet of pass k − 1 is decoded under
  pass k's directory and the wait never counts as lag. One pass registers no extra
  generation.
- **Everything else** (accounting, staging, decode, release, backpressure, errors) is
  `GpuMitchSourceOp`'s.

## Edges and lane

- Out: `lossless(4)` into `QuoteBarsOp(backend="gpu")`, same lane; in `market_data_bench
  --check` also `lossless(8)` into a reference `QuoteBarsOp(backend="cpu")` (fan-out: the
  source retries until both accept).
- Lane: the entry of `ingest`.
- Threads: `hf-replay` (Python), created under `helper_mask(helper_cpus)`; it holds the GIL
  for about 100-200 µs per burst, about 1% of a core at 200k packets/s.
- Streams: the drain gathers on its own non-blocking stream and synchronizes each burst.
- Errors: a pcap error fails `start()`; a drain exception goes to `stats.error` and the next
  compute raises.

## Telemetry

`IngestStats` as for `GpuMitchSourceOp` (`front_end = "replay"`, `frame_fetch = "device frame
pool"`; `stalls["native"]` carries the drain's ring waits and wake-up gaps), plus
`IngestStats.replay`: `pps`, `pass_index`, `sent`, `nic_dropped`, `lag_p50_ms`,
`lag_p99_ms`, `lag_max_ms`, `loop_wait_ms`, `thread_cpu_s` (CPU time of `hf-replay`) and
`eof`, copied every 100 ms.

## Example

```python
from holoscan_finance.operators import GpuMitchReplaySourceOp, QuoteBarsOp
from holoscan_finance.operators.edges import lossless
from holoscan_finance.ref import Ref

source = g.op(GpuMitchReplaySourceOp, "source", pcap_path="day.pcap", pps=200_000,
              nic_bufs=114_688, loops=1, helper_cpus=sorted(cpus.helper),
              on_loop=Ref(replay_generations), directory=Ref(knob), stats=Ref(ingest_stats))
bars = g.op(QuoteBarsOp, "bars", backend="gpu", directory=Ref(knob), stats=Ref(bars_stats))
g.add_flow(source.events, bars.events, **lossless(4))
g.partition("ingest").add(source, bars)
```
