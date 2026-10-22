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
full, sleeping on a condition that each release signals (a full ring costs it no CPU). This
is how the whole GPU graph is run and tested without a NIC: bit identity, throughput
headroom and paced lossless gates. The drain is Python, so it is not a test of resilience to
GIL stalls; those claims stay with the native drain on the NIC.

Processor: `holoscan_finance.ingest.gpu_ingest.GpuIngest` with `DeviceReplayDrain`.

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
| `on_loop` | `Ref` | `Ref()` | `callable(pass_index) -> sender port \| None`, called only before pass k ≥ 1, after everything gathered was released and `settled` holds; a returned port is written into the feed frames' UDP source port |
| `settled` | `Ref` | `Ref()` | `callable() -> bool`: before pass k ≥ 1 the drain also waits until it is true, so that every decoded batch has reached the bars (the applications pass `DrainedCondition.batches_consumed`); empty: no wait |
| `keep_alive` | bool | `False` | `False`: the input ends after the last pass (`stats.eof`) |

## Behaviour

- **Schedule.** Frame `i` of a pass is due at `t0 + i / pps`; the burst deadline is
  `min(due(first) + rx_timeout, due(last))`, and a burst holds the frames due by then.
  `lag` is how late a burst was gathered against its deadline.
- **Feed and sender port.** The feed is the capture's frames from the first generation's
  `sender_port` when the capture carries that port, else its most common UDP source port's
  frames, which are rewritten to `sender_port` before pass 0. Each later pass re-sources
  only the feed's frames (like `tcprewrite --portmap feed:new`); frames from any other
  sender stay stale in every pass.
- **Loops.** Before pass k ≥ 1 the drain waits until the consumer released every row (ring
  and staging empty) and `settled()` holds (every decoded batch reached the bars), calls
  `on_loop(k)` (`ReplayGenerations` registers a new generation on a fresh sender port) and
  re-bases the schedule, so no packet of pass k − 1 is decoded or built into bars under
  pass k's directory and the wait never counts as lag. One pass registers no extra
  generation.
- **Everything else** (accounting, staging, decode, release, backpressure, errors) is
  `GpuMitchSourceOp`'s.

## Edges and lane

- Out: `lossless(4)` into `QuoteBarsOp(backend="gpu")`, same lane; in `market_data_bench
  --check` also `lossless(8)` into a reference `QuoteBarsOp(backend="cpu")` (fan-out: the
  source retries until both accept).
- Lane: the entry of `ingest`.
- Threads: `hf-replay` (Python, so it takes the GIL for every burst), created under
  `helper_mask(helper_cpus)`.
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
              on_loop=Ref(replay_generations), settled=Ref(drained.batches_consumed),
              directory=Ref(knob), stats=Ref(ingest_stats))
# nothing consumes bar windows here: emit_windows=False leaves `window` undeclared
bars = g.op(QuoteBarsOp, "bars", backend="gpu", emit_windows=False, directory=Ref(knob),
            stats=Ref(bars_stats))
g.add_flow(source.events, bars.events, **lossless(4))
g.partition("ingest").add(source, bars)
```
