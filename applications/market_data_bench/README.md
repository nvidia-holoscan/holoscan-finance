# market_data_bench

A headless, finite benchmark of the production GPU market-data chain, with no NIC. It
replays the Ethernet frames of a MITCH capture from GPU memory through the same gather
kernel, device ring, receive accounting, GPU MITCH decode and GPU quote bars as the
GPUDirect path, optionally adds the GPU risk lane, and stops as soon as everything is
decoded and built. It reports throughput, loss, replay lag and placement, applies gates
to them, and with `--check` proves that every bar the GPU built is bit-identical to the
CPU backend and to an independent host decode of the capture.

The application is one plain composition of `holoscan.finance` operators
([`market_data_bench.py`](market_data_bench.py), with no operator classes of its own):

```text
lane ingest:     source ─lossless(4)─► bars ─lossless(64)─► windows        (windows: --check)
                   │                    │ LATEST (newest window only)       (--risk gpu)
                   │                    ▼
lane analytics:    │                   risk ─lossless(8)─► publish
                   │ fan-out, lossless(8)                                     (--check)
lane reference:    └──────────────────► ref_bars ─lossless(64)─► ref_windows
```

| Operator | Class | Role |
| --- | --- | --- |
| `source` | `GpuMitchReplaySourceOp` | pcap frames in GPU memory, a software NIC with DAQIRI's burst rule (4,096 packets or 20 ms), paced at `--pps`, optional NIC buffer drop model (`--nic-bufs`) |
| `bars` | `QuoteBarsOp(backend="gpu")` | event-time quote bars on the GPU, same lane as the source |
| `windows` | `CallbackSinkOp` | every GPU window, in order, to the checker |
| `ref_bars` | `QuoteBarsOp(backend="cpu")` | the CPU twin, fed the very same decoded batches |
| `ref_windows` | `CallbackSinkOp` | every CPU window to the checker |
| `risk`, `publish` | `RollingRiskOp(gpu, inline)`, `CallbackSinkOp` | the analytics load of the full chain |

`--check` is one operator class with two backends in one graph: the source's output fans
out to both `QuoteBarsOp`s, and the source holds and retries a batch until both accepted
it. After the run, every GPU window is paired with its CPU twin by
`(generation, completed_bars)`, and every GPU bar is compared with a host decode of the
pcap (`holoscan_finance.validation.HostReference`). Times and NaN patterns must be
identical, finite returns within 1 ULP (CUDA vs NumPy `log`), and the final counters and
per-symbol state of both backends equal. Because the source waits for the CPU twin, a
`--check` run's rates measure the CPU bars, not the GPU chain: measure throughput without
`--check`.

## Run

With the Holoscan CLI (`holoscan run market_data_bench <mode>`), on any GPU host:

| Mode | What it does |
| --- | --- |
| `smoke` (default) | a small synthetic day, unpaced, `--check --require-lossless` (CI) |
| `check` | the same on one million synthetic packets |
| `paced` | a full synthetic day (2,886 symbols, 2.88M packets) at 200,000 packets/s with the NIC buffer model and GPU risk: lossless, real time, replay lag p99 ≤ 5 ms |
| `throughput` | the full synthetic day unpaced: at least 13.2M MITCH messages/s (twice the 200k packets/s line rate) |
| `drops` | a capture with every 997th packet missing, `--check`: the gaps and the bars after them match the CPU backend and the host decode |
| `pcap` | a recorded day, `<data>/nyq/day.pcap` and its prepared tape `<data>/nyq/tape`, at 200,000 packets/s |
| `plan` | compiles the `--check --risk gpu` composition, checks its rules and writes `market_data_bench.plan.json` |

Or directly, from the repository root inside the Holoscan 5.0 image
(`scripts/dev5.sh '<command>'`):

```bash
python3 -m holoscan_finance.tools.synth --out data/synth --symbols 500 --packets 50000 --seconds 1800 --reuse
python3 applications/market_data_bench/market_data_bench.py \
    --pcap data/synth/day.pcap --tape data/synth/tape --pps 0 --check --require-lossless --report out/check.json
```

| Flags | Meaning |
| --- | --- |
| `--pcap`, `--tape` | the capture and the prepared tape of its day (the manifest: symbols, date) |
| `--pps` (200000), `--loops` (1) | replay rate (0: unpaced, as fast as the ring and the graph allow); passes over the capture, each later pass a new generation from a new source port |
| `--burst-packets` (4096), `--rx-timeout-us` (20000), `--nic-bufs` (0) | DAQIRI's burst rule; the NIC buffer model (0: the drain holds instead of dropping) |
| `--batch-messages`, `--ring-size`, `--gpu-device` | the measured 4.3 GPUDirect sizes by default |
| `--bar-seconds` ... `--max-spread-bps` | the bar settings, as in `equity_analytics` |
| `--check`, `--risk {none,gpu}`, `--interval` | verification; the risk lane and its minimum interval |
| `--ingest-cpus`, `--ingest-busy-poll` | placement of the ingest lane |
| `--report PATH`, `--timeout-s` (1800) | the report file; fail if not drained by then |
| `--plan-only`, `--plan-json PATH` | compile and check only |

## Report and gates

The report is printed and written to `--report`. It holds the termination, the stop
reason and the parsed flags; the counters of every stage (`packets`, `received_messages`,
`messages`, `events`, `completed_bars`); the rates over the span from the first to the
last bars batch (`msgs_per_s`, `packets_per_s`, `achieved_pps` against `requested_pps`);
every loss counter; the ring (`ring_peak`, `ring_waits`, `ring_wait_ms`); the replay lag
percentiles; the replay drain's own CPU time; the worst drain and decode passes; the lanes
with their CPUs; and, when enabled, `risk` (windows analyzed and superseded,
receive-to-result latency) and `check`. A value the run never produced is `null`, and a
gate on it fails. The bar counters restart with every generation, so after several
`--loops` passes `events` and `completed_bars` are the last pass's, and `seconds` and the
rates are `null`.

`ok` is true only if the run stopped because it drained, ended `STOPPED`, and every
requested gate holds:

| Gate | Requires |
| --- | --- |
| `--require-lossless` | no sequence gap, ring or NIC drop, stale, malformed or undecodable packet (holding a burst is not loss) |
| `--require-realtime` | paced runs without `--check`: no retried emit and no ring wait, so the graph kept up with the requested rate |
| `--check` | every comparison above holds, and something was compared |
| `--min-msgs-per-s X` | `msgs_per_s >= X` |
| `--max-lag-p99-ms X` | replay lag p99 ≤ X (the gather time against each burst's due time) |

| Exit | Meaning |
| --- | --- |
| 0 | every gate holds |
| 1 | the run failed, timed out or left an error diagnostic |
| 2 | invalid flags |
| 3, 4 | the plan was rejected, or breaks one of the composition's rules |
| 5 | the run finished cleanly but a gate failed |

The replay drain is a Python thread, so these runs validate function, bit identity,
throughput headroom and paced losslessness, not resilience to GIL stalls: that claim stays
with the native drain on the NIC. Results and the exact commands are in
[`docs/VALIDATION.md`](../../docs/VALIDATION.md).

Python code imports the canonical package, `holoscan_finance`; with the module installed,
the same operators are available as `holoscan.finance`:

```python
from holoscan.finance.operators import GpuMitchReplaySourceOp, QuoteBarsOp
```
