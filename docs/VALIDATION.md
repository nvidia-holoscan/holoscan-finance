# NIC-free validation on Holoscan SDK 5.0

This page records how this Holoscan SDK 5.0 module is validated on a GPU host without the
NIC, and the results. Every run uses synthetic data and the NIC-free replay source
(`GpuMitchReplaySourceOp`): pcap frames staged in GPU memory go through the production
gather kernel, device ring, receive accounting, GPU MITCH decode and GPU quote bars. No run
touches the NIC, hugepages, IRQs or the demo containers.

> **Status.** The functional steps 1, 2 and 4 ran on 2026-10-20 and pass ([results
> below](#functional-results-2026-10-20)), and step 9 passed in the integration pass the same
> day. The serial timing pass (steps 3, 5, 6, 7 and 8 and `pytest -m timing`) ran the same
> evening and passes ([results below](#timing-results-2026-10-20)). Step 5 passed only after a
> fix to the replay source: freeing the capture's host copy after the replay had started
> delayed the first bursts by about 200 ms.

## What runs where

- **Container.** Every command runs inside the Holoscan 5.0 image through
  [`scripts/dev5.sh`](../scripts/dev5.sh): non-root, private IPC (never `--ipc=host`), CPUs
  `0-2,4-8,10-13` (off the DAQIRI lcores 3 and 9), the RTX PRO 6000 by PCI bus order, and a
  timeout inside the container. Full-day steps set `RUN5_TIMEOUT=3600`, because the bench's
  own `--timeout-s 1800` exceeds the 1500 s default.
- **Data.** `data/` and `out/` under the repository root (both gitignored).
- **Shared GPU.** Another demo container may share the GPU, so timing compares **medians only**,
  and each run records the host state (below).
- **IGX Thor iGPU power.** The Thor iGPU (`0000:01:00.0`) runtime-suspends after about 5 s
  idle, and the next new CUDA allocation in the process wakes it, which blocks every CUDA
  call of the process for about 130 ms. The NIC launcher keeps the iGPU powered for
  `INGEST=gpudirect`; `scripts/dev5.sh` does not. In the paced runs this shows up as a replay lag
  of about 110 ms once per wake-up (see [anomalies](#anomalies-and-findings)). To time
  without it, keep a CUDA context open on the iGPU for the length of the run (an idle
  non-root container with `CUDA_VISIBLE_DEVICES=<iGPU UUID>`), or set its
  `power/control` to `on` as the launcher does.
- **Functional steps** (1, 2, 4, 9) are part of the acceptance of the validation work
  package. **Timing steps** (3, 5, 6, 7, 8) run only in the serial timing pass, after
  the integration pass, with no other container running.

## Host state

Recorded before and after each session.

| When | `docker ps` | `nvidia-smi` GPU utilization / memory | Notes |
| --- | --- | --- | --- |
| functional runs, before (2026-10-20 19:49 EDT) | `nervous_morse` and one unrelated demo container | RTX PRO 6000: 0 %, 47 MiB | `nervous_morse` is an auto-named container, not a demo container; the demo container was not touched |
| functional runs, after (2026-10-20 20:17 EDT) | the unrelated demo container | RTX PRO 6000: 0 %, 47 MiB | steps 2 and 4 overlapped each other and the `synth-drops` generation: functional results only |
| step 9, before and after (2026-10-20 20:30 and 20:31 EDT) | the unrelated demo container | RTX PRO 6000: 0 %, 47 MiB | integration pass |
| timing pass, before (2026-10-20 21:55 EDT) | the unrelated demo container | RTX PRO 6000: 0 %, 47 MiB | load average 2.08; no docker build; the demo container was not touched |
| timing pass, every step (21:55-22:58 EDT, 139 records) | the unrelated demo container only, for steps 3, 5, 6 and 7 | 0 %, 47 MiB at every step start and end | load average 1.88-6.28; no docker build at any check; the diagnostic runs with the iGPU kept awake also had a sidecar container keeping the iGPU awake |
| timing pass, after (2026-10-20 23:00 EDT) | the unrelated demo container | RTX PRO 6000: 0 %, 47 MiB | load average 3.23 |

Commands: `docker ps --format '{{.Names}}'` and
`nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv`.

## Procedure

All commands run inside `scripts/dev5.sh '<command>'` from the repository root.

### 1. Data

```bash
python3 -m holoscan_finance.tools.synth --out data/synth-day --symbols 2886 --packets 2880000 --seconds 23400 --reuse
python3 -m holoscan_finance.tools.synth --out data/synth-drops --symbols 2886 --packets 2880000 --seconds 23400 --drop-every 5000 --reuse
```

Pass: the files are written and `synth.json` holds the totals. The full day is the size of
the recorded NYQ day (2,880,565 packets, 92,178,039 observations, 2,886 symbols): 2,880,001
packets including the end-of-session marker, and 92,160,000 observations.

### 2. Library bit identity (GPU vs host decode and bars)

```bash
python3 -m holoscan_finance.tools.gpu_validate --pcap data/synth-day/day.pcap --tape data/synth-day/tape --out out/gpu_validate.json
python3 -m holoscan_finance.tools.gpu_validate --pcap data/synth-day/day.pcap --tape data/synth-day/tape --bars --out out/gpu_validate_bars.json
python3 -m holoscan_finance.tools.gpu_validate --pcap data/synth-day/day.pcap --tape data/synth-day/tape --bars --drop-every 5000 --out out/gpu_validate_bars_drops.json
```

Pass: decode `bit_identical`, including the fuzz pass's rejection decisions; bars
`state_identical` (marks, times, trades, volumes and counters identical, returns within
1 ULP).

### 3. Offline kernels (timing pass)

```bash
python3 -m holoscan_finance.tools.decode_bench --pcap data/synth-day/day.pcap --tape data/synth-day/tape --group-msgs 16384 --threads 1,4,8,12 --out out/decode_bench_16k.json
python3 -m holoscan_finance.tools.decode_bench --pcap data/synth-day/day.pcap --tape data/synth-day/tape --group-msgs 131072 --threads 1,4,8,12 --out out/decode_bench_131k.json
```

Report only (shared GPU). The compiled CPU rows need numba>=0.68, which
`scripts/dev5.sh` installs: the 5.0 image ships numba 0.64, which refuses its NumPy 2.5.
Pass `--threads` with at most 12 threads: `scripts/dev5.sh` gives the container 12 CPUs, numba
refuses more, and the default `--threads 1,2,4,8,14` (meant for a 14-core host without a
CPU set) stops with `ValueError: The number of threads must be between 1 and 12`.

### 4. Graph bit identity (`market_data_bench --check`)

```bash
python3 applications/market_data_bench/market_data_bench.py --pcap data/synth-day/day.pcap --tape data/synth-day/tape --pps 0 --check --require-lossless --report out/check.json
python3 applications/market_data_bench/market_data_bench.py --pcap data/synth-drops/day.pcap --tape data/synth-drops/tape --pps 0 --check --report out/check_drops.json
```

Pass: `ok`; every `check.*` value true; `max_ulps ≤ 1`; on the drops capture the sequence
gaps equal the host reference's (`check.reference.counters_equal`). The source holds every
batch until the CPU twin has accepted it, so the rates of a `--check` run measure the CPU
bars, not the GPU chain (step 6 measures throughput).

### 5. Paced lossless (timing pass, 3 runs)

```bash
python3 applications/market_data_bench/market_data_bench.py --pcap data/synth-day/day.pcap --tape data/synth-day/tape --pps 200000 --nic-bufs 114688 --risk gpu --require-lossless --require-realtime --max-lag-p99-ms 5 --report out/paced_<i>.json
```

Pass: all three runs `ok`.

### 6. Throughput headroom (timing pass, 5 runs)

```bash
python3 applications/market_data_bench/market_data_bench.py --pcap data/synth-day/day.pcap --tape data/synth-day/tape --pps 0 --report out/tput_<i>.json
```

Pass: median `msgs_per_s ≥ 13.2e6`, twice the 6.6M msgs/s line rate; otherwise the
fallback of a separate `bars` lane is evaluated and reported.

### 7. Full chain with compare (timing pass)

```bash
python3 applications/equity_analytics/equity_analytics.py --ingest replay --pcap data/synth-day/day.pcap --tape data/synth-day/tape --replay-pps 200000 --analytics-backend compare --port 0 --exit-when-drained --report out/chain.json --expect-windows 10
```

Pass: exit 0; every comparison `agreement and matched_work`; `receive_to_result_ms` p50 and
p99 reported. The replay source emulates DAQIRI's 20 ms partial-burst timeout, which the
arrival stamp includes.

### 8. Analytics speedup (timing pass)

The `rolling_timing.speedup` median of the step 7 run.

### 9. Dashboard contract

```bash
python3 -m pytest -q -m slow tests/python/e2e/test_dashboard_live.py
```

Pass: the test passes (every documented `/snapshot` key present, `network.ingest == "replay"`).

## Results

| Step | Command (report) | Result | Pass |
| --- | --- | --- | --- |
| 1 Data | `synth` (`data/synth-day/synth.json`) | 2,880,001 packets, 95,062,715 messages; drops: 576 packets missing | pass |
| 2 Library bit identity | `gpu_validate` (`out/gpu_validate*.json`) | decode and fuzz bit-identical; bars identical, 41 and 39 returns differ by 1 ULP | pass |
| 3 Offline kernels | `decode_bench` (`out/decode_bench_*.json`) | GPU decode / decode + bars: 43.1 / 17.6M msgs/s at 16k-message groups, 306 / 66.4M at 131k | report |
| 4 Graph bit identity | `market_data_bench --check` (`out/check*.json`) | both `ok`; 701 and 700 window pairs, 4,679 bars against the host decode, `max_ulps` 1; drops: 19,011 gaps on every side | pass |
| 5 Paced lossless | `market_data_bench` paced (`out/paced_*.json`) | 3 of 3 `ok` (after the replay-source fix): 2,880,001 packets, no loss, no retry or ring wait, lag p99 0.61-1.22 ms; before the fix 0 of 3 (lag p99 86-89 ms) | pass |
| 6 Throughput headroom | `market_data_bench` unpaced (`out/tput_*.json`) | median 62.1M msgs/s (61.2-64.5M, 5 runs), 4.7× the 13.2M gate, lossless | pass |
| 7 Full chain with compare | `equity_analytics --ingest replay` (`out/chain.json`) | exit 0; 41 windows, 41 comparisons, all agree with matched work; receive-to-result p50 25.2 ms, p99 143.9 ms | pass |
| 8 Analytics speedup | step 7 `rolling_timing.speedup` | 18.3× (CPU 97.3 ms, GPU 5.31 ms, 40 windows) on the synthetic day, where all 2,886 symbols are eligible; 4.7× (6.3× with the iGPU kept awake) on the recorded day | report |
| 9 Dashboard contract | `e2e/test_dashboard_live.py` | 1 passed in 28.5 s (integration pass, 2026-10-20) | pass |

The timing tests of the test suite (`pytest -m timing`) also run in the timing pass:

```bash
RUN5_TIMEOUT=3600 scripts/dev5.sh 'python3 -m pytest -q -m timing tests/python'
```

| Timing tests | Result |
| --- | --- |
| `pytest -m timing` | 3 passed (risk pickup latency, paced replay drain, paced replay source), before and after the replay-source fix |

## Functional results (2026-10-20)

Holoscan 5.0.0 image (`nvcr.io/nvidia/clara-holoscan/holoscan:v5.0.0-cuda13`) through
`scripts/dev5.sh` with `RUN5_TIMEOUT=3600`, on the RTX PRO 6000. Durations are wall times on
a shared host and are not results.

**Step 1, data.** Both commands exited 0 (about 11 minutes each).

| Data set | Packets | Observations | MITCH messages | Missing | Files |
| --- | --- | --- | --- | --- | --- |
| `data/synth-day` | 2,880,001 | 92,160,001 | 95,062,715 | none | `day.pcap` 3.36 GB, tape 6.08 GB |
| `data/synth-drops` (`--drop-every 5000`) | 2,879,425 | 92,141,569 | 95,043,704 | 576 packets, 18,432 observations, 19,011 messages | `day.pcap` 3.36 GB |

The observation and message counts include the end-of-session Status marker; the tape
holds 92,160,000 observations of 2,886 symbols.

**Step 2, library bit identity.** All three runs exited 0.

- Decode: every one of the 2,880,001 packets has the same accept/reject stage, counts,
  sequence and fields, bit for bit (92,160,001 observations; `bit_identical` true). The fuzz
  pass (240,000 mutated packets: 60,518 accepted, 60,036 rejected by the framing check,
  119,446 by the decoder) is bit-identical too. Host decode 84.5 s, GPU 0.27 s.
- Bars: 4,680 bars and 4,660 windows compared, marks, times, trades, volumes, the final
  per-symbol state and every counter identical (`state_identical` true). Of 13,414,033 finite
  returns, 41 differ by 1 ULP (CUDA vs NumPy `log`) and none by more.
- Bars with every 5,000th packet dropped: identical as well, with 19,011 sequence gaps on
  both sides; 39 of 11,434,555 finite returns differ by 1 ULP.

**Step 4, graph bit identity.** Both runs exited 0 with `ok` true, termination `STOPPED`
and stop reason `drained`.

| Capture | Packets | Gaps | Window pairs | Bars vs host decode | `max_ulps` (pairs, host) | Counters and decode counts | Final state |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `synth-day` (`--require-lossless`) | 2,880,001 | 0 | 701 | 4,679 | 1, 1 | equal | equal |
| `synth-drops` | 2,879,425 | 19,011 | 700 | 4,679 | 1, 1 | equal | equal |

- Every loss counter of the lossless run is 0; the drops run's 19,011 sequence gaps are the
  synthesizer's 19,011 missing messages, and the CPU twin and the host reference counted
  the same gaps (`counters_equal`).
- 4,680 bars closed; the first has no previous midpoint and is in no window, so 4,679 are
  compared.
- Each run took about 200 s: about 50 s of graph (paced by the CPU twin: 2.1M and 1.7M
  messages/s) and about 2 minutes of host decode for the reference.

## Timing results (2026-10-20)

The serial timing pass, 21:55-22:58 EDT, on the same host and image as the functional runs,
with the replay-source fix below for steps 5-7. Only the unrelated demo container was running
at every step start and end (139 host-state records), the GPU was idle at each of them and no
docker build ran. Steps 5-7 ran with the iGPU in its default runtime power management (see
[What runs where](#what-runs-where)).

**Timing tests.** `RUN5_TIMEOUT=3600 scripts/dev5.sh 'python3 -m pytest -q -m timing
tests/python'`: 3 passed in 7.4 s, and again after the fix.

**Step 3, offline kernels** (`--threads 1,4,8,12`; 200,000 packets = 6,601,587 messages of
`synth-day`, M msgs/s, one run each):

| Path | 16k decode | 16k decode + bars | 131k decode | 131k decode + bars |
| --- | --- | --- | --- | --- |
| Python, 1 core | 0.27 | 0.25 | 0.27 | 0.24 |
| numba, 1 / 4 / 8 / 12 threads | 36.1 / 94.0 / 140.1 / 26.7 | 5.8 / 5.6 / 5.9 / 5.7 | 33.7 / 103.5 / 151.4 / 97.8 | 5.7 / 6.0 / 6.3 / 6.2 |
| GPU | 43.1 | 17.6 | 306.1 | 66.4 |

The numba multi-thread rows are noisy (timed steps of 40-250 ms): the 12-thread 16k figure
was 26.7, 49.0 and 38.1M msgs/s in three runs. The other rows are stable.

`decode_bench` times one pass per process, so its GPU decode + bars figure includes a
one-time cost of the first pass, about 43 ms (at 131k groups, a 58 ms pass). In steady
state (passes 2-5 of five in one process) GPU decode + bars reaches 19.9 and 113.3M msgs/s
at 16k and 131k groups. The applications pay the first-pass cost in the operators'
`start()` warm-up, before data.

**Step 5, paced at 200,000 packets/s** (`--nic-bufs 114688 --risk gpu --require-lossless
--require-realtime --max-lag-p99-ms 5`):

| Run | `ok` | Packets | Loss (gaps, ring, NIC, stale, malformed) | Bars span | Achieved pps | Retries, ring waits | Ring peak | Lag p50 / p99 / max (ms) | Receive to result p50 / p99 (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | true | 2,880,001 | 0 | 14.380 s | 200,282 | 0, 0 | 28,007 | 0.11 / 0.73 / 112.0 | 20.9 / 75.6 |
| 2 | true | 2,880,001 | 0 | 14.379 s | 200,288 | 0, 0 | 24,006 | 0.11 / 1.22 / 109.8 | 21.1 / 71.8 |
| 3 | true | 2,880,001 | 0 | 14.379 s | 200,294 | 0, 0 | 28,007 | 0.10 / 0.61 / 112.9 | 21.6 / 70.3 |

Before the fix, the same three runs were lossless and real time but not `ok`: lag p99 86.4,
88.5 and 88.3 ms (max 194-204 ms), ring peak 48,012 each time. The remaining lag maximum of
about 110 ms is one iGPU wake-up at the first risk window; with the iGPU kept awake it is
20.8 ms (p99 0.85 ms).

**Step 6, unpaced** (5 runs): 64.45, 61.15, 62.08, 64.49 and 61.26M msgs/s, median
**62.08M msgs/s** (1.53 s for the day's 95,062,715 messages), all lossless. The 1M-slot ring
is full most of the time (447 ring waits, 0.93-0.98 s waited), as intended; the replay drain
used 0.11-0.16 s of CPU per run. The fallback of a separate `bars` lane was not needed.
Graph overhead, in absolute terms: steady-state GPU decode + bars (step 3) take 8.8 ns per
message (0.84 s for the full day), and the unpaced graph 16.1 ns (1.53 s). The graph adds
0.69 s per day for gather, ring and accounting, header copies, per-batch Python and
scheduling: about 0.95 ms per 131,072-message batch.

**Steps 7 and 8, full chain with compare** (`out/chain.json`): exit 0, termination `STOPPED`,
stop reason `drained`, no error, no unmet expectation. 2,880,001 packets in 720 batches,
nothing lost; 717 windows emitted, 712 received by risk, 671 superseded (latest-only); 41
analyzed, 41 comparisons, `agreement` true. Receive to result: p50 **25.2 ms**, p99 **143.9
ms**, max 158.3 ms (41 samples, so p99 is the slowest window). Replay lag p50 / p99 / max
0.10 / 2.74 / 7.71 ms. `rolling_timing`: 40 samples, CPU median 97.3 ms, GPU median 5.31 ms,
**speedup 18.3×**. On the synthetic day every one of the 2,886 symbols becomes eligible, so
the CPU window is larger than on the recorded day (below), where the same run gives 4.7×.

**The recorded day, NIC-free** (the licensed capture of 2,880,565 packets, mounted
read-only; same commands as steps 3, 5, 6 and 7):

| Measurement | Holoscan 5.0, replay (no NIC) |
| --- | --- |
| `decode_bench` GPU decode / decode + bars, 16k | 44.1 / 17.7M msgs/s |
| `decode_bench` GPU decode / decode + bars, 131k | 304.3 / 66.6M msgs/s |
| Full day at 200,000 packets/s | 3 of 3 lossless and real time, 14.38 s |
| The same with `--max-lag-p99-ms 5` | 0 of 3 (lag p99 45-49 ms, two iGPU wake-ups per run); with the iGPU kept awake 3 of 3 (p99 0.69-0.73 ms, max 11-15 ms) |
| Unpaced | 60.0M msgs/s median (59.5-61.9M) |
| Compare mode at 200,000 packets/s: receive to result p50 / p99 | 26.9 / 150.7 ms; iGPU kept awake 25.2 / 50.5 ms |
| Compare mode: rolling CPU / GPU median, speedup | 31.7 / 6.69 ms, 4.7×; iGPU kept awake 35.4 / 5.62 ms, 6.3× |

### Anomalies and findings

- **Fixed: the replay source freed the capture's host copy after starting the replay.**
  `GpuMitchReplaySourceOp.start()` started the drain thread and then returned, which freed
  the 3.4 GB host copy of the capture: about 240 ms with the interpreter lock held (measured
  in isolation: 238.8 ms). The drain, already on its schedule, gathered its first bursts up to
  200 ms late, so step 5 failed its 5 ms lag gate in 3 of 3 runs, and the ring peaked at 12
  bursts (48,012 packets). A GIL watchdog (a heartbeat thread plus `faulthandler`) showed a
  255 ms hold by the Holoscan thread right after `start()`. The fix frees the host copy before
  the drain starts (`del frames`), with a regression test
  (`test_the_host_copy_of_the_capture_is_freed_before_the_drain_starts`, which fails without
  the fix). After it, step 5 passes 3 of 3.
- **iGPU runtime power management.** The remaining lag spikes of about 110-130 ms happen at the
  exact times the Thor iGPU resumes from runtime suspend (sysfs `runtime_status` logged
  against the drain's per-burst timestamps: the drain's host-to-device copy was blocked from
  6,901.7 to 7,032.0 ms while the iGPU was resuming from 6,902.9 to 7,031.5 ms, and the same at
  13.2 s). New CUDA allocations of the risk engine, when the number of eligible symbols grows,
  trigger them. On the synthetic day it happens once, at the first window; on the recorded
  day twice more, which breaks the 5 ms lag gate there. With an idle CUDA context holding the
  iGPU awake, no transition occurs and every run passes. It is a host setting the NIC launcher
  already handles, not a code change.
- **The step 3 command needed `--threads`.** See [step 3](#3-offline-kernels-timing-pass).
- **Compare-mode speedups are not the separate-runs measurement.** Compare mode runs both
  backends on the same host at once, and the CPU job starts when the GPU result is ready, so
  its ratio is not that of separate CPU-only and GPU-only runs
  ([METHODOLOGY](METHODOLOGY.md#separate-cpu-only-and-gpu-only-benchmark)). On the
  synthetic day the CPU side analyzes all 2,886 symbols.

## What this does not show

The replay drain is a Python thread, so these runs establish function, bit identity,
throughput headroom and paced losslessness, not resilience to GIL stalls: that claim stays
with the native drain on the NIC. Three runs need the user's explicit approval and are not
part of this page until they run: the NIC smoke run, the 250-run full-day series and the
`/debug/hold_gil` fault-injection A/B at 1.1 s and 3 s (native vs Python drain). No claim
is made for them until then.
