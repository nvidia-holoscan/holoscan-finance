# NIC-free validation on Holoscan SDK 5.0

This page records how this Holoscan SDK 5.0 module is validated on a GPU host without the
NIC, and the results. The runs use a recorded NYQ day prepared with `tools.prepare_tape`
and its MITCH capture through two front ends: the NIC-free
replay source (`GpuMitchReplaySourceOp`: pcap frames staged in GPU memory go through the
production gather kernel, device ring, receive accounting, GPU MITCH decode and GPU quote
bars) and the kernel UDP receiver fed by `replay_sender` over localhost. Steps 10 and 11
time the analytics. None of these runs touches the NIC, hugepages or IRQs; the short NIC
check has [its own section](#short-nic-check).

> **Status.** Every number on this page is from the final code of this module, measured on
> 2026-10-21 (evening) and 2026-10-22 on one host with no other GPU application during the
> timed steps ([host state](#host-state)). Every step, the timing tests, the `holoscan`
> smoke and test runs, the full test suite and the headless-browser dashboard checks ran on
> it. Every step passes: decode and bars are bit-identical (steps 2 and 4), the paced gate
> at 200,000 packets/s is lossless (step 5), and in step 10 the GPU is faster than the CPU at p50 and p99 for every window kind at
> every size from 50 to 10,000 symbols, back to back and at the 270 ms live cadence, on both
> GPU paths, with one exception: one back-to-back p99 at 50 symbols in one of 16 runs
> ([step 10](#step-10-analytics-per-window)). A short NIC check on 2026-10-22 passed at the
> paced sender's rate, about 4,600-4,700 packets/s ([short NIC check](#short-nic-check));
> 200,000 packets/s on the NIC and the full-day NIC series are not recorded
> ([what this does not show](#what-this-does-not-show)).

## What runs where

- **Host.** IGX Thor (aarch64, 14 CPU cores) with an RTX PRO 6000 Blackwell Max-Q GPU,
  which runs every CUDA kernel, and a ConnectX-7 NIC, which these runs do not use.
- **Container.** Every command runs inside the Holoscan 5.0.0 image
  (`nvcr.io/nvidia/clara-holoscan/holoscan:v5.0.0-cuda13`) through
  [`scripts/dev5.sh`](../scripts/dev5.sh): non-root, private IPC (never `--ipc=host`), CPUs
  `0-2,4-8,10-13` (off the DAQIRI lcores 3 and 9), the RTX PRO 6000 by PCI bus order, and a
  timeout inside the container. Full-day steps set `RUN5_TIMEOUT=3600`, because the bench's
  own `--timeout-s 1800` exceeds the 1500 s default. A step that runs a receiver, a sender
  and `measure` (step 11) runs them in one `scripts/dev5.sh` shell, over the container's own
  localhost.
- **Data.** Under the repository root (gitignored), with `out/` for the reports; the layout
  is below.
- **IGX Thor iGPU power.** The Thor iGPU (`0000:01:00.0`) runtime-suspends about 5 s after
  it goes idle, and a resume takes 120-140 ms (its `power/runtime_status` polled every
  20 ms); the next new CUDA allocation in a process wakes it and waits for the resume. The
  timed runs of 2026-10-22 therefore kept an idle CUDA context open on the iGPU (a sidecar:
  an idle non-root container with `CUDA_VISIBLE_DEVICES=<iGPU UUID>`), except the runs
  marked "no sidecar"; the earlier runs (steps 7 and 8, step 10's back-to-back `compare`
  runs) had none. The NIC launcher sets the iGPU's `power/control` to `on` instead.
- **Timing discipline.** Timed steps ran one at a time, with the host state recorded at
  the start and end of each. From 2026-10-22 01:00 each started after 60 s with no other
  container on the host, which was checked every 5 s; a run during which one appeared was
  repeated, or, in the two long step 10 runs, the affected size was taken from the other
  run.

| Path | Contents | How it is made |
| --- | --- | --- |
| `data/nyq/tape` | a recorded NYQ day prepared with `tools.prepare_tape` (2,886 symbols) | step 1 |
| `data/nyq/day.pcap` | that day's MITCH packets as the replay sender sends them, one generation (2,880,565 packets) | step 1 |
| `data/nyq/bars.npz` | the day's five-second bars, for `risk_bench` | step 10 |
| `data/synth-50` | a deterministic synthetic day of 50 symbols | step 1 |

## Host state

| When (EDT) | Other containers | RTX PRO 6000 utilization / memory | Notes |
| --- | --- | --- | --- |
| 2026-10-21 22:52-22:59: steps 7 and 8 (two runs), the timing tests, both `holoscan run ... smoke` modes, `holoscan test` and the headless-browser dashboard checks | an idle DPDK container; no iGPU sidecar | 0 %, 47 MiB at every start and end | load average 3.7-7.1, all of it from the running step |
| 2026-10-21 22:59-23:05: step 10 back to back with `compare`, 50 to 1,000 symbols | the idle DPDK container; no iGPU sidecar. A CI job's container started at about 23:05, during the 2,886-symbol size, which was re-run | 0 %, 47 MiB at the start | load average 3.7 at the start |
| 2026-10-21 23:33-23:37: the full test suite (step 9; no timing) | the idle DPDK container and a CI job's container (about one CPU core, on the Thor iGPU) | 0 %, 47 MiB at the start and end | load average 5.5-8.2 |
| 2026-10-22 01:02-02:45: step 10 back to back with `compare` at 2,886, 5,000 and 10,000 symbols, and `compare` at the 270 ms cadence at 50 | the idle DPDK container only; no iGPU sidecar | 0 %, 47 MiB at every start and end | load average 2.2-10.0, all of it from the running step |
| 2026-10-22 03:04-09:55: step 10, both passes at every size, two runs | the idle DPDK container and the iGPU sidecar. Short-lived containers of other users at 03:43 (run 1, 100 symbols: that size is taken from run 2), 05:04-05:06 (run 1, 1,000 symbols, the CPU pass; under 2 % of a core, kept) and 09:00 (run 2, 5,000 symbols: taken from run 1) | 0 %, 47 MiB at the start and end of each run | load average 1.6-7.6 |
| 2026-10-22 09:56-11:00: steps 11, 3 and 6, and two follow-ups of step 10 at 50 to 200 symbols | the idle DPDK container and the iGPU sidecar (stopped for the two no-sidecar runs of step 11); one step 11 run during which other containers ran was discarded and repeated | 0 %, 47 MiB at every start | load average 2.0-6.8, all of it from the running step |
| 2026-10-22 11:44-12:10: steps 2, 4 and 5 | the idle DPDK container only; no iGPU sidecar | 0 %, 47 MiB at the start and end | load average 2.4-4.2, all of it from the running step |

Commands: `uptime`, `docker ps --format '{{.Names}}'` and
`nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv`.

## Procedure

All commands run inside `scripts/dev5.sh '<command>'` from the repository root.

### 1. Data

```bash
python3 -m holoscan_finance.tools.prepare_tape --data '<day>/*.parquet' --output data/nyq/tape
python3 -m holoscan_finance.tools.bench_replay inspect data/nyq/day.pcap --port <PORT>
python3 -m holoscan_finance.tools.synth --out data/synth-50 --symbols 50 --packets 400000 --seconds 7200 --reuse
```

`data/nyq/day.pcap` is one replay generation of `replay_sender` captured from its egress
and filtered to its source port (`<PORT>`), as in
[gpu_ingest_bench, step 1](../deploy/igx_thor/gpu_ingest_bench/README.md). Pass: the
capture's sequences are continuous, and `synth.json` holds the synthetic totals. Without
the licensed data, `synth --out data/synth-day --symbols 2886 --packets 2880000 --seconds
23400` writes a synthetic day of the same size that every step accepts in place of
`data/nyq`; no result on this page comes from it.

### 2. Library bit identity (GPU vs host decode and bars)

```bash
python3 -m holoscan_finance.tools.gpu_validate --pcap data/nyq/day.pcap --tape data/nyq/tape --out out/gpu_validate.json
python3 -m holoscan_finance.tools.gpu_validate --pcap data/nyq/day.pcap --tape data/nyq/tape --decoder cpu --out out/gpu_validate_cpu.json
python3 -m holoscan_finance.tools.gpu_validate --pcap data/nyq/day.pcap --tape data/nyq/tape --bars --out out/gpu_validate_bars.json
python3 -m holoscan_finance.tools.gpu_validate --pcap data/nyq/day.pcap --tape data/nyq/tape --bars --drop-every 5000 --out out/gpu_validate_bars_drops.json
```

Pass: decode `bit_identical` (GPU, and the compiled CPU decoder of step 3), including the
fuzz pass's rejection decisions; bars `state_identical` (marks, times, trades, volumes and
counters identical, returns within 1 ULP).

### 3. Offline kernels (timing pass)

```bash
python3 -m holoscan_finance.tools.decode_bench --pcap data/nyq/day.pcap --tape data/nyq/tape --group-msgs 16384 --threads 1,4,8,12 --out out/decode_bench_16k.json
python3 -m holoscan_finance.tools.decode_bench --pcap data/nyq/day.pcap --tape data/nyq/tape --group-msgs 131072 --threads 1,4,8,12 --out out/decode_bench_131k.json
```

Report only. The compiled CPU rows need numba 0.68, which `scripts/dev5.sh` installs: the
5.0 image ships numba 0.64, which refuses its NumPy 2.5. Pass `--threads` with at most 12
threads: `scripts/dev5.sh` gives the container 12 CPUs, numba refuses more, and the default
`--threads 1,2,4,8,14` (meant for a 14-core host without a CPU set) stops with
`ValueError: The number of threads must be between 1 and 12`.

### 4. Graph bit identity (`market_data_bench --check`)

```bash
python3 examples/market_data_bench/market_data_bench.py --pcap data/nyq/day.pcap --tape data/nyq/tape --pps 0 --check --require-lossless --report out/check.json
python3 examples/market_data_bench/market_data_bench.py --pcap data/nyq/day.pcap --tape data/nyq/tape --pps 200000 --check --require-lossless --report out/check_200k.json
```

Pass: `ok`; every `check.*` value true; `max_ulps ≤ 1`. The source holds every batch until
the CPU twin has accepted it, so the rates of a `--check` run measure the CPU bars, not the
GPU chain (step 6 measures throughput).

### 5. Paced lossless (timing pass, 3 runs)

```bash
python3 examples/market_data_bench/market_data_bench.py --pcap data/nyq/day.pcap --tape data/nyq/tape --pps 200000 --nic-bufs 114688 --risk gpu --require-lossless --require-realtime --max-lag-p99-ms 5 --report out/paced_<i>.json
```

Pass: all three runs `ok`.

### 6. Throughput headroom (timing pass)

```bash
python3 examples/market_data_bench/market_data_bench.py --pcap data/nyq/day.pcap --tape data/nyq/tape --pps 0 --report out/tput_<i>.json
```

Pass: `msgs_per_s ≥ 13.2e6`, twice the 6.6M msgs/s line rate, in every run (two runs
here); otherwise the fallback of a separate `bars` lane is evaluated and reported.

### 7. Full chain with compare (timing pass)

```bash
python3 examples/equity_analytics/equity_analytics.py --ingest replay --pcap data/nyq/day.pcap --tape data/nyq/tape --replay-pps 200000 --analytics-backend compare --port 0 --exit-when-drained --report out/chain.json --expect-windows 10
```

Pass: exit 0; every comparison `agreement and matched_work`; `receive_to_result_ms` p50 and
p99 reported. The replay source emulates DAQIRI's 20 ms partial-burst timeout, and the
arrival stamp is taken when a burst is gathered, after that timeout: at low packet rates up
to 20 ms of a packet's wait is not in the figure ([timing scopes](METHODOLOGY.md#timing-scopes)).

### 8. Analytics speedup (timing pass)

The `rolling_timing.speedup` median of the step 7 run.

### 9. Dashboard contract

```bash
python3 -m pytest -q -m slow tests/python/e2e/test_dashboard_live.py
```

Pass: the test passes (every documented `/snapshot` key present, `network.ingest == "replay"`).

### 10. Analytics per window (timing pass)

```bash
python3 -m holoscan_finance.tools.risk_bench --sizes 50 --phases 1 --returns data/nyq/tape --save-returns data/nyq/bars.npz   # once: the day's bars
python3 -m holoscan_finance.tools.risk_bench --kinds 300 --sizes 50,100,200,500,1000,2886,5000,10000 --backends gpu,gpu-host,cpu --returns data/nyq/bars.npz --check --out out/kinds.json --table out/kinds.md
python3 -m holoscan_finance.tools.risk_bench --kinds 2000 --cadence-ms 0 --sizes 50,100,200,500,1000,2886 --backends gpu,gpu-host,cpu,compare --returns data/nyq/bars.npz --check --out out/kinds_b2b.json --table out/kinds_b2b.md
python3 -m holoscan_finance.tools.risk_bench --kinds 500 --cadence-ms 0 --sizes 5000,10000 --backends gpu,gpu-host,cpu,compare --returns data/nyq/bars.npz --check --out out/kinds_b2b_large.json --table out/kinds_b2b_large.md
python3 -m holoscan_finance.tools.risk_bench --kinds 300 --sizes 50 --backends gpu,gpu-host,cpu,compare --returns data/nyq/bars.npz --check --out out/kinds_50.json --table out/kinds_50.md
```

The engines as the example applications call them: `gpu` is `RollingRiskOp` inline on device bars
(the `replay` and `gpudirect` front ends); `gpu-host` and `cpu` are the spawned workers'
functions, fed each window's new bars, as risk and the CPU cross-check send them (the GPU
worker of the host front ends, and the CPU-only run); `compare` runs compare mode's two
processors, the GPU `RollingRisk` validating every window and the CPU cross-check's spawned
worker recomputing it, one validated window at a time, and reports both times. The CPU
workers are tuned for the universe before the first window, as the example applications tune them
when the manifest arrives. `--kinds C` times every window kind `C` times per size and
backend, one cycle per new generation (as after a start or a seek): the first, 20-bar window
of a generation (`first`), one-bar growth (`growth`), a full window that shares nothing with
the window before (`rebuild`, the same work as every 256th call), and one-, two- and
three-bar slides of the full window (`slide1` to `slide3`; superseded windows give the
longer slides). A validated window in `compare` mode is one of these kinds on both sides.
One untimed warm-up cycle allocates first. Each backend runs back to back, then with
270 ms idle before every window, the live cadence at 75,000 observations/s (`--cadence-ms`;
0 skips that pass). `--returns data/nyq/tape` builds the day's bars with host `QuoteBars`
at the default settings (about two minutes) and `--save-returns` keeps them for the later
runs: real missing returns and eligibility, where synthetic returns (no `--returns`) make
every symbol eligible. With `--check` the first 50 cycles of every pass and backend are
checked against the NumPy oracle and the GPU paths against the CPU (agreement and matched
work), and every `compare` pair is checked. The run ends with the table of p50 / p99 per
size, pass and kind and the report's `verdict`: every GPU path below the CPU at p50 and p99,
or the cells where it is not. `--phases G` (`G` generations of a live sequence: growth from
21 to 60 bars, 40 one-bar, 10 two-bar and 10 three-bar slides and 5 rebuilds) and the
default one-bar slides time the same engines with fewer windows of each kind. Pass: every
`check` value complete; GPU p50 and p99 below the CPU's for every window kind at every size,
on both GPU paths and in `compare` mode.

Most of a run's time is the 270 ms pass: 300 cycles of six kinds at 270 ms take about eight
minutes per backend and size, so the second command runs about 3.4 hours (both runs here
took 12,284 s), while the back-to-back pass alone (`--cadence-ms 0`) takes minutes at
every size. For a quick check, run back to back only, or only the sizes of interest at the
cadence.

### 11. Separate CPU-only and GPU-only runs, NIC-free (timing pass)

The [separate-runs method](METHODOLOGY.md#separate-cpu-only-and-gpu-only-benchmark) over
localhost UDP, and the replay front end, each in one `scripts/dev5.sh` shell:

```bash
python3 examples/equity_analytics/equity_analytics.py --ingest udp --analytics-backend cpu --port 8080 &   # then gpu
# the sender waits until the receiver's /healthz answers 200 (its UDP socket is bound):
python3 -m holoscan_finance.tools.replay_sender --data data/nyq/tape --port 26400 --control-url http://127.0.0.1:8080 &
python3 -m holoscan_finance.tools.measure --seconds 120 --rate 75000 --output out/cpu-only.json                # gpu-only.json
python3 -m holoscan_finance.tools.compare_runs --cpu out/cpu-only.json --gpu out/gpu-only.json --all-shared --output out/separate.json

python3 examples/equity_analytics/equity_analytics.py --ingest replay --pcap data/nyq/day.pcap --tape data/nyq/tape --replay-pps 2344 --analytics-backend gpu --port 8080 &   # or compare
python3 -m holoscan_finance.tools.measure --seconds 120 --rate 75000 --output out/replay-gpu.json
```

2,344 packets/s replays the capture at 75,000 observations/s (32 per packet). `measure`
seeks the sender to 14:00 UTC (the replay front end keeps its own position, from the start
of the session), waits for a full window and records 120 s. Pass:
`compare_runs` valid (identical inputs, agreement, matched work); every GPU-only packet to
result p50 and p99 below the CPU-only run's.

## Results

| Step | Command (report) | Result | Pass |
| --- | --- | --- | --- |
| 1 Data | `prepare_tape`, capture, `synth` | recorded day: 2,880,565 packets, 95,081,226 MITCH messages, 92,178,039 observations (the end-of-session marker included), 2,886 symbols; `synth-50`: 400,001 packets, 13,206,974 messages, 50 symbols | pass |
| 2 Library bit identity | `gpu_validate` (`out/gpu_validate*.json`) | GPU decode and the compiled CPU decoder `bit_identical` on all 2,880,565 packets and the 240,000-packet fuzz pass (0 mismatches); bars `state_identical`: 4,680 bars and 4,196 windows, returns within 1 ULP (518 of 11,612,346 differ by 1 ULP); with every 5,000th packet dropped, 19,012 gaps on both sides and `state_identical` | pass |
| 3 Offline kernels | `decode_bench` (`out/decode_bench_*.json`) | GPU decode + bars 17.79M msgs/s at 16k-message groups and 65.91M at 131k: 2.47× and 8.67× the best compiled CPU; GPU decode alone 44.0M at 16k (0.28× twelve numba threads) and 276.1M at 131k (2.45× eight) | report |
| 4 Graph bit identity | `market_data_bench --check` (`out/check*.json`) | `ok` unpaced and at 200,000 packets/s: 694 and 712 windows compared, times, NaN pattern and final state equal, `max_ulps` 1, lossless | pass |
| 5 Paced lossless | `market_data_bench` paced (`out/paced_*.json`) | three runs `ok`: 2,880,565 packets each, no sequence gap or ring drop, 6.61M msgs/s, lag p99 0.30, 0.24 and 0.37 ms | pass |
| 6 Throughput headroom | `market_data_bench` unpaced (`out/tput_*.json`) | 60.45M and 62.27M msgs/s (two runs), 4.6× and 4.7× the 13.2M gate; 2,880,565 packets, 95,081,226 messages, 4,680 bars, lossless | pass |
| 7 Full chain with compare | `equity_analytics --ingest replay` (`out/chain.json`) | two runs: exit 0; 284 and 286 windows analyzed, every one compared and agreeing with matched work; receive to result p50 15.7 and 16.1 ms, p99 25.7 and 26.1 ms | pass |
| 8 Analytics speedup | step 7 `rolling_timing.speedup` | 11.9× (CPU 19.2 ms, GPU 1.61 ms) and 6.8× (10.7 / 1.56 ms), 120 windows each; both engines run the same algorithm and the same work (new bars only, own variance) | report |
| 9 Dashboard contract | `e2e/test_dashboard_live.py` | passed (full test suite) | pass |
| 10 Analytics per window | `risk_bench --kinds` (`out/kinds*.json`) | every kept window agrees with the oracle and GPU with CPU, and every `compare` pair agrees with matched work, at every size; GPU p50 and p99 below the CPU's for every window kind at every size from 50 to 10,000 symbols: back to back on both GPU paths and in `compare` mode (144 of 144 cells; p50 1.4-1.7× at 50 symbols, 7.4-9.7× at 2,886, 8.6-16.6× at 10,000), and at the 270 ms cadence on both GPU paths (96 of 96 cells; p50 1.9-2.8× at 50, 9.6-12.8× at 2,886, 8.3-15.7× at 10,000) and in `compare` mode at 50 symbols; one exception: one back-to-back p99 at 50 symbols in one of 16 runs that time that size | pass, one exception |
| 11 Separate CPU-only and GPU-only runs | `measure`, `compare_runs` (`out/separate.json`) | recorded day at 75,000 observations/s over UDP: `compare_runs` valid for all 9 CPU/GPU pairs (366-368 shared windows, identical inputs, agreement, matched work), analytics 9.7-10.0× faster; packet to result GPU 5.38-5.41 / 8.34-8.89 ms vs CPU 10.63-10.77 / 15.03-15.52 ms (p50 / p99, three runs each) | pass |

The timing tests of the test suite (`pytest -m timing`: risk pickup latency, paced replay
drain, paced replay source) pass on this code:

```bash
RUN5_TIMEOUT=3600 scripts/dev5.sh 'python3 -m pytest -q -m timing tests/python'
```

`holoscan run equity_analytics smoke --local`, `holoscan run market_data_bench smoke
--local` and `holoscan test --local --no-xvfb` (CTest: `python.unit` and `python.gpu`)
inside `scripts/dev5.sh` pass too. The dashboard was also driven outside the test suite, in
a headless browser on replay `compare` and UDP GPU-only runs: every polled `/snapshot` was
consistent, and the browser logged no console error and showed the values of the
`/snapshot` it rendered.

## Timing results

Measured on the final code ([host state](#host-state)): steps 7 and 8 on 2026-10-21 at
22:52-22:54 EDT and step 10's back-to-back `compare` runs at 22:59-02:45, without the iGPU
sidecar; step 10's two runs of both passes (2026-10-22 03:04-09:55), steps 11, 3 and 6
(09:56-10:56) and step 10's follow-ups (10:57-11:00) with it, except the two no-sidecar
runs of step 11; steps 2, 4 and 5 on 2026-10-22 at 11:44-12:10, without it.

### Step 3, offline kernels

The first 200,000 packets of the recorded day (6,601,280 messages), `--threads 1,4,8,12`, one
run each, M msgs/s:

| Path | 16k decode | 16k decode + bars | 131k decode | 131k decode + bars |
| --- | --- | --- | --- | --- |
| Python, 1 core | 0.275 | 0.249 | 0.271 | 0.250 |
| numba, 1 / 4 / 8 / 12 threads | 34.2 / 90.7 / 144.0 / 160.1 | 6.75 / 6.79 / 7.19 / 6.31 | 33.0 / 86.0 / 112.7 / 85.2 | 6.57 / 7.18 / 7.51 / 7.60 |
| GPU | 44.0 | **17.79** | **276.1** | **65.91** |

Decode + bars is faster on the GPU at both group sizes: 2.47× the best CPU at 16k-message
groups and 8.67× at 131k. GPU decode alone is not faster at 16k-message groups: 1.3× one
numba thread, but 0.49× four, 0.31× eight and 0.28× twelve. The numba rows are single runs
of short timed steps, and their thread scaling is uneven (twelve threads are the fastest at
16k-message groups and slower than eight at 131k). `decode_bench` times one pass per process,
so its GPU figures include a one-time cost of the first pass, which the example applications pay in
the operators' `start()` warm-up, before data.

### Steps 5 and 6, paced and unpaced

Step 5, paced at 200,000 packets/s (three runs): all `ok`, 2,880,565 packets and 6.61M
msgs/s each, real time, no sequence gap, ring drop or NIC drop, lag p99 0.30, 0.24 and
0.37 ms against the 5 ms gate. Unpaced (step 6, two
runs): 60.45 and 62.27M msgs/s (1.83M and 1.89M packets/s), both `ok`: 2,880,565 packets,
95,081,226 messages, 4,680 bars, and no sequence gap, ring or NIC drop, stale or malformed
packet. The source waited for ring space 447 times in each run: unpaced, the 1M-slot ring
fills, as intended. The fallback of a separate `bars` lane is not needed.

### Steps 7 and 8, full chain with compare

Both runs: exit 0, termination `STOPPED`, stop reason `drained`, no error, no unmet
expectation; 2,880,565 packets, nothing lost; 712 windows emitted. Each CPU twin was tuned
in `start()` (four BLAS threads, 1.4 and 1.5 s), before the replay began.

| Run | Analyzed (superseded) | Comparisons, agreement | Receive to result p50 / p99 / max (ms) | `rolling_timing` CPU / GPU median, speedup |
| --- | --- | --- | --- | --- |
| 1 | 284 (428) | 284, true | 15.7 / 25.7 / 31.0 | 19.2 / 1.61 ms, 11.9× |
| 2 | 286 (426) | 286, true | 16.1 / 26.1 / 32.3 | 10.7 / 1.56 ms, 6.8× |

At 200,000 packets/s the analyzed windows mostly slide by several bars, ingest kernels
share the GPU and the interpreter lock is busy, so both medians sit above step 10's, and
the CPU median varies between runs. Both engines copy in only a window's new bars and
compute their own variance, so the ratio compares the same work.

### Step 10, analytics per window

`risk_bench --kinds`, engine milliseconds, p50 / p99, on the recorded day's bars (eligible
count in brackets). Back to back (the third and fourth commands), 2,000 windows of every
kind and backend up to 2,886 symbols and 500 above; the last column is `compare` mode's validated windows, its GPU and
its CPU twin at p50. The 2,886-symbol rows come from the same command with `--sizes 2886`
alone, re-run because a CI job's container started on the host during that size of the
first run. Every `check` complete: 300 kept windows of every backend against the oracle and
of both GPU paths against the CPU at every size, and every `compare` pair (12,000 and 3,000
per size) agreeing with matched work:

| Symbols (eligible) | Kind | GPU (`RollingRiskOp`) | GPU worker | CPU worker | `compare`: GPU / CPU p50 |
| --- | --- | --- | --- | --- | --- |
| 50 (30) | first | 0.077 / 0.083 | 0.075 / 0.082 | 0.130 / 0.143 | 0.104 / 0.393 |
| 50 (30) | growth | 0.065 / 0.072 | 0.068 / 0.077 | 0.117 / 0.127 | 0.103 / 0.361 |
| 50 (30) | rebuild | 0.120 / 0.128 | 0.110 / 0.116 | 0.167 / 0.181 | 0.145 / 0.479 |
| 50 (30) | one-bar | 0.076 / 0.082 | 0.078 / 0.084 | 0.130 / 0.157 | 0.120 / 0.388 |
| 50 (30) | two-bar | 0.076 / 0.086 | 0.078 / 0.084 | 0.131 / 0.145 | 0.107 / 0.391 |
| 50 (30) | three-bar | 0.077 / 0.083 | 0.078 / 0.085 | 0.133 / 0.145 | 0.119 / 0.394 |
| 100 (58) | first | 0.079 / 0.086 | 0.078 / 0.085 | 0.171 / 0.184 | 0.108 / 0.455 |
| 100 (58) | growth | 0.075 / 0.083 | 0.077 / 0.083 | 0.137 / 0.149 | 0.109 / 0.390 |
| 100 (58) | rebuild | 0.122 / 0.131 | 0.112 / 0.120 | 0.259 / 0.272 | 0.161 / 0.627 |
| 100 (58) | one-bar | 0.079 / 0.085 | 0.083 / 0.090 | 0.160 / 0.171 | 0.115 / 0.419 |
| 100 (58) | two-bar | 0.079 / 0.087 | 0.083 / 0.091 | 0.160 / 0.172 | 0.114 / 0.425 |
| 100 (58) | three-bar | 0.081 / 0.087 | 0.084 / 0.095 | 0.166 / 0.177 | 0.118 / 0.442 |
| 200 (127) | first | 0.089 / 0.096 | 0.090 / 0.096 | 0.290 / 0.315 | 0.137 / 0.786 |
| 200 (127) | growth | 0.076 / 0.083 | 0.079 / 0.086 | 0.176 / 0.195 | 0.133 / 0.513 |
| 200 (127) | rebuild | 0.133 / 0.140 | 0.129 / 0.137 | 0.540 / 0.577 | 0.188 / 1.422 |
| 200 (127) | one-bar | 0.088 / 0.095 | 0.091 / 0.097 | 0.211 / 0.236 | 0.128 / 0.596 |
| 200 (127) | two-bar | 0.088 / 0.095 | 0.091 / 0.098 | 0.224 / 0.246 | 0.150 / 0.616 |
| 200 (127) | three-bar | 0.089 / 0.095 | 0.092 / 0.099 | 0.240 / 0.263 | 0.131 / 0.675 |
| 500 (311) | first | 0.128 / 0.135 | 0.137 / 0.146 | 0.940 / 0.956 | 0.183 / 2.193 |
| 500 (311) | growth | 0.101 / 0.109 | 0.106 / 0.115 | 0.347 / 0.361 | 0.154 / 0.927 |
| 500 (311) | rebuild | 0.222 / 0.232 | 0.234 / 0.248 | 2.160 / 2.203 | 0.298 / 4.296 |
| 500 (311) | one-bar | 0.128 / 0.135 | 0.131 / 0.139 | 0.445 / 0.463 | 0.176 / 1.108 |
| 500 (311) | two-bar | 0.127 / 0.135 | 0.130 / 0.138 | 0.498 / 0.516 | 0.186 / 1.273 |
| 500 (311) | three-bar | 0.128 / 0.135 | 0.131 / 0.140 | 0.551 / 0.565 | 0.186 / 1.367 |
| 1,000 (622) | first | 0.230 / 0.247 | 0.254 / 0.267 | 1.233 / 1.327 | 0.305 / 4.517 |
| 1,000 (622) | growth | 0.152 / 0.168 | 0.155 / 0.170 | 0.590 / 0.645 | 0.191 / 1.587 |
| 1,000 (622) | rebuild | 0.456 / 0.474 | 0.492 / 0.516 | 2.586 / 2.851 | 0.540 / 9.912 |
| 1,000 (622) | one-bar | 0.179 / 0.193 | 0.183 / 0.197 | 0.602 / 0.683 | 0.244 / 1.833 |
| 1,000 (622) | two-bar | 0.189 / 0.203 | 0.194 / 0.206 | 0.641 / 0.731 | 0.255 / 2.103 |
| 1,000 (622) | three-bar | 0.192 / 0.208 | 0.197 / 0.213 | 0.687 / 0.756 | 0.266 / 2.378 |
| 2,886 (1,763) | first | 0.979 / 0.989 | 1.029 / 1.043 | 8.645 / 8.780 | 1.045 / 9.062 |
| 2,886 (1,763) | growth | 0.339 / 0.351 | 0.346 / 0.358 | 3.355 / 3.490 | 0.436 / 3.549 |
| 2,886 (1,763) | rebuild | 2.403 / 2.441 | 2.522 / 2.555 | 18.57 / 19.03 | 2.487 / 20.44 |
| 2,886 (1,763) | one-bar | 0.427 / 0.439 | 0.434 / 0.444 | 3.713 / 3.835 | 0.533 / 3.955 |
| 2,886 (1,763) | two-bar | 0.489 / 0.499 | 0.497 / 0.509 | 3.899 / 4.014 | 0.556 / 4.123 |
| 2,886 (1,763) | three-bar | 0.553 / 0.568 | 0.562 / 0.577 | 4.200 / 4.312 | 0.638 / 4.428 |
| 5,000 (3,008) | first | 2.489 / 2.511 | 2.564 / 2.591 | 24.70 / 25.74 | 2.575 / 25.29 |
| 5,000 (3,008) | growth | 0.723 / 0.758 | 0.730 / 0.741 | 9.221 / 10.06 | 0.806 / 9.439 |
| 5,000 (3,008) | rebuild | 6.591 / 6.626 | 6.782 / 6.847 | 57.38 / 58.11 | 6.690 / 55.41 |
| 5,000 (3,008) | one-bar | 0.823 / 0.834 | 0.831 / 0.841 | 10.37 / 10.79 | 0.911 / 9.572 |
| 5,000 (3,008) | two-bar | 0.994 / 1.002 | 1.004 / 1.021 | 10.02 / 10.41 | 1.072 / 9.961 |
| 5,000 (3,008) | three-bar | 1.184 / 1.198 | 1.194 / 1.210 | 11.12 / 11.45 | 1.267 / 11.23 |
| 10,000 (5,213) | first | 8.915 / 8.973 | 9.043 / 9.092 | 96.18 / 97.61 | 9.083 / 94.76 |
| 10,000 (5,213) | growth | 2.174 / 2.197 | 2.187 / 2.201 | 35.30 / 36.43 | 2.291 / 35.73 |
| 10,000 (5,213) | rebuild | 24.70 / 24.86 | 25.05 / 25.15 | 216.1 / 219.4 | 24.96 / 219.9 |
| 10,000 (5,213) | one-bar | 2.280 / 2.316 | 2.290 / 2.307 | 37.90 / 39.51 | 2.360 / 36.82 |
| 10,000 (5,213) | two-bar | 2.734 / 2.752 | 2.755 / 2.795 | 37.68 / 39.24 | 2.849 / 38.27 |
| 10,000 (5,213) | three-bar | 3.499 / 3.531 | 3.518 / 3.536 | 41.64 / 42.60 | 3.633 / 41.24 |

GPU p50 and p99 are below the CPU's for every kind at every size, on both GPU paths and in
`compare` mode (the reports' `verdict`: 144 of 144 cells). At p50 the slower GPU path is
faster than the CPU worker by 1.4-1.7× at 50 symbols, 1.8-2.2× at 100, 2.2-4.1× at 200,
3.3-9.2× at 500, 3.3-5.3× at 1,000, 7.4-9.7× at 2,886, 8.5-12.6× at 5,000 and 8.6-16.6× at
10,000, and by 1.4× or more at p99.
The GPU worker (`gpu_window`, host rows) is within 0.04 ms of the in-process GPU up to 1,000
symbols. Every window is one launch of a CUDA graph: the engine writes the window's shape
and its new bars' addresses to pinned memory, launches, and synchronizes (plus one copy of a
supplied variance). In `compare` mode the CPU twin runs in its own spawned worker, one
validated window at a time after the GPU side's, with its own tuning, so its times differ
from the CPU worker's: up to 500 symbols, where both kept one BLAS thread, they are higher
(1.108 against 0.445 ms per one-bar slide at 500), and at 1,000 symbols its tuning kept two
threads where the CPU worker's kept four.

The second command times the same three engines back to back and at the cadence, 300
windows of every kind, at every size; it ran twice. Back to back, its cells match the table
above (GPU p50 within 0.01 ms up to 1,000 symbols) and the GPU is below the CPU in 192 of
192 cells of run 1 and 191 of 192 of run 2. The exception is the GPU worker's three-bar
slide at 50 symbols in run 2: p99 0.182 against the CPU's 0.154 ms (p50 0.078 against
0.128), one slow window among 300, where a p99 is about the third slowest. Ten more runs of
that size (`--sizes 50 --cadence-ms 0`, 120 cells) and three runs of 2,000 windows at 50,
100 and 200 symbols (`--kinds 2000 --cadence-ms 0 --sizes 50,100,200`, 108 cells) had every
cell below the CPU's, by at least 1.35× at p50 and 1.36× at p99. The margin at 50 symbols is
small in absolute terms: 0.04-0.05 ms per window.

With 270 ms idle before every window, the live cadence at 75,000 observations/s (the
second command; 300 windows of every kind and backend; `check` complete). Run 1, except 100
symbols from run 2: a container of another user ran on the host during run 1's 100-symbol
cadence pass (and during run 2's 5,000-symbol one). Each pass tunes the CPU worker anew; the
cadence pass kept one BLAS thread at 50, 200 and 500 symbols, eight at 100 (where every
candidate times alike) and four from 1,000:

| Symbols (eligible) | Kind | GPU (`RollingRiskOp`) | GPU worker | CPU worker |
| --- | --- | --- | --- | --- |
| 50 (30) | first | 0.170 / 0.278 | 0.192 / 0.295 | 0.415 / 0.786 |
| 50 (30) | growth | 0.191 / 0.295 | 0.188 / 0.330 | 0.393 / 0.759 |
| 50 (30) | rebuild | 0.241 / 0.372 | 0.223 / 0.362 | 0.674 / 0.887 |
| 50 (30) | one-bar | 0.194 / 0.302 | 0.202 / 0.311 | 0.415 / 0.778 |
| 50 (30) | two-bar | 0.192 / 0.289 | 0.208 / 0.371 | 0.433 / 0.760 |
| 50 (30) | three-bar | 0.204 / 0.293 | 0.214 / 0.321 | 0.398 / 0.764 |
| 100 (58) | first | 0.193 / 0.270 | 0.208 / 0.278 | 0.474 / 0.955 |
| 100 (58) | growth | 0.186 / 0.303 | 0.189 / 0.299 | 0.423 / 0.764 |
| 100 (58) | rebuild | 0.265 / 0.367 | 0.254 / 0.354 | 0.808 / 1.832 |
| 100 (58) | one-bar | 0.221 / 0.336 | 0.206 / 0.307 | 0.419 / 0.788 |
| 100 (58) | two-bar | 0.211 / 0.304 | 0.198 / 0.313 | 0.442 / 0.819 |
| 100 (58) | three-bar | 0.209 / 0.309 | 0.215 / 0.321 | 0.449 / 0.840 |
| 200 (127) | first | 0.189 / 0.267 | 0.203 / 0.307 | 0.547 / 1.182 |
| 200 (127) | growth | 0.197 / 0.296 | 0.190 / 0.303 | 0.660 / 0.872 |
| 200 (127) | rebuild | 0.268 / 0.406 | 0.254 / 0.365 | 1.145 / 1.803 |
| 200 (127) | one-bar | 0.214 / 0.317 | 0.221 / 0.344 | 0.606 / 0.959 |
| 200 (127) | two-bar | 0.214 / 0.320 | 0.210 / 0.309 | 0.503 / 1.004 |
| 200 (127) | three-bar | 0.221 / 0.308 | 0.212 / 0.331 | 0.505 / 1.066 |
| 500 (311) | first | 0.238 / 0.339 | 0.261 / 0.357 | 1.507 / 2.934 |
| 500 (311) | growth | 0.233 / 0.319 | 0.236 / 0.369 | 0.750 / 1.363 |
| 500 (311) | rebuild | 0.353 / 0.501 | 0.383 / 0.506 | 3.419 / 6.175 |
| 500 (311) | one-bar | 0.249 / 0.358 | 0.247 / 0.373 | 0.874 / 1.598 |
| 500 (311) | two-bar | 0.244 / 0.345 | 0.254 / 0.468 | 1.078 / 1.750 |
| 500 (311) | three-bar | 0.255 / 0.363 | 0.270 / 0.368 | 1.052 / 1.940 |
| 1,000 (622) | first | 0.330 / 0.442 | 0.383 / 0.475 | 3.126 / 4.755 |
| 1,000 (622) | growth | 0.281 / 0.363 | 0.266 / 0.387 | 1.574 / 3.497 |
| 1,000 (622) | rebuild | 0.579 / 0.732 | 0.636 / 0.799 | 5.074 / 8.317 |
| 1,000 (622) | one-bar | 0.299 / 0.406 | 0.311 / 0.408 | 1.766 / 3.042 |
| 1,000 (622) | two-bar | 0.305 / 0.406 | 0.341 / 0.438 | 1.750 / 4.040 |
| 1,000 (622) | three-bar | 0.333 / 0.419 | 0.342 / 0.455 | 1.958 / 3.283 |
| 2,886 (1,763) | first | 1.072 / 1.545 | 1.172 / 1.274 | 15.03 / 21.07 |
| 2,886 (1,763) | growth | 0.456 / 0.564 | 0.485 / 0.585 | 5.125 / 6.421 |
| 2,886 (1,763) | rebuild | 2.543 / 2.671 | 2.714 / 2.848 | 34.03 / 46.46 |
| 2,886 (1,763) | one-bar | 0.547 / 0.643 | 0.546 / 0.658 | 5.580 / 7.612 |
| 2,886 (1,763) | two-bar | 0.596 / 0.703 | 0.632 / 0.744 | 6.085 / 9.099 |
| 2,886 (1,763) | three-bar | 0.654 / 0.761 | 0.683 / 0.807 | 6.557 / 9.134 |
| 5,000 (3,008) | first | 2.611 / 2.730 | 2.748 / 2.854 | 39.71 / 47.05 |
| 5,000 (3,008) | growth | 0.842 / 0.939 | 0.862 / 0.961 | 10.91 / 13.97 |
| 5,000 (3,008) | rebuild | 6.750 / 7.052 | 7.004 / 7.170 | 86.86 / 99.05 |
| 5,000 (3,008) | one-bar | 0.946 / 1.055 | 0.978 / 1.073 | 10.93 / 13.19 |
| 5,000 (3,008) | two-bar | 1.112 / 1.235 | 1.132 / 1.234 | 12.18 / 15.18 |
| 5,000 (3,008) | three-bar | 1.315 / 1.398 | 1.331 / 1.446 | 14.92 / 17.86 |
| 10,000 (5,213) | first | 9.110 / 9.207 | 9.318 / 9.431 | 98.77 / 104.7 |
| 10,000 (5,213) | growth | 2.303 / 2.406 | 2.320 / 2.473 | 36.33 / 40.61 |
| 10,000 (5,213) | rebuild | 25.01 / 25.17 | 25.48 / 25.75 | 212.3 / 217.0 |
| 10,000 (5,213) | one-bar | 2.410 / 2.514 | 2.427 / 2.557 | 37.02 / 41.23 |
| 10,000 (5,213) | two-bar | 2.876 / 2.975 | 2.899 / 2.985 | 37.20 / 40.89 |
| 10,000 (5,213) | three-bar | 3.658 / 3.760 | 3.698 / 3.822 | 40.72 / 45.92 |

GPU p50 and p99 are below the CPU's for every kind at every size on both GPU paths (96 of 96
cells here, and in both runs). Against the CPU worker, the slower GPU path is faster at p50
by 1.9-2.8× at 50 symbols, 1.9-3.1× at 100, 2.3-4.3× at 200, 3.2-8.9× at 500, 5.1-8.2× at
1,000, 9.6-12.8× at 2,886, 10.8-14.5× at 5,000 and 8.3-15.7× at 10,000, and by 2.0× or more
at p99. Both engines are slower with 270 ms idle before each window than back to back, the
CPU more so. `compare` mode at the cadence, at 50 symbols (the last command; 300 validated
windows of every kind, every pair agreeing with matched work): GPU p50 0.155-0.229 ms and
p99 0.262-0.372 ms against the CPU twin's 0.396-0.599 and 0.794-1.006 ms, all 36 of that
run's cells below the CPU.

### Step 11, separate CPU-only and GPU-only runs

Recorded day at 75,000 observations/s, 120 s each, every capture `valid` with 0 sequence
gaps; p50 / p99 / max in ms, each from the run's `measure` report. The `compare` rows ran
with `/snapshot` polled every 500 ms, as the dashboard polls it:

| Run | Windows | Analytics p50 / p99 | Packet to result |
| --- | --- | --- | --- |
| UDP, CPU only (three runs) | 368, 368, 368 | 5.53 / 9.22, 5.63 / 9.70, 5.61 / 8.87 | 10.63 / 15.03 / 42.7, 10.77 / 15.52 / 44.8, 10.66 / 15.17 / 48.4 |
| UDP, GPU only (three runs) | 368, 368, 368 | 0.572 / 1.065, 0.563 / 1.004, 0.567 / 0.803 | **5.41 / 8.89** / 9.4, **5.39 / 8.34** / 10.0, **5.38 / 8.46** / 11.2 |
| UDP, GPU only, no sidecar | 369 | 0.557 / 0.735 | 5.18 / 7.19 / 10.4 |
| UDP, `compare` | 368 | GPU 0.597 / 0.859, CPU 5.63 / 9.79 | GPU 7.51 / 14.36 / 18.5, CPU 17.45 / 27.49 / 48.9 |
| Replay front end, GPU only (two runs) | 351, 351 | 0.513 / 0.974, 0.515 / 0.691 | 4.10 / 5.45 / 6.2, 4.02 / 4.97 / 6.4 |
| Replay front end, GPU only, no sidecar | 351 | 0.503 / 1.006 | 4.18 / 5.14 / 5.4 |
| Replay front end, `compare` | 351 | GPU 0.522 / 0.819, CPU 5.60 / 8.03 | GPU 4.79 / 7.08 / 10.4, CPU 14.28 / 18.54 / 55.1 |

`compare_runs` (each CPU run against each GPU run, 9 pairs): valid, identical inputs,
agreement and matched work (the same method and added and expired bars in every window) on
366-368 shared windows, analytics 9.7-10.0× faster at the median; every GPU-only run is
below every CPU-only run at p50 (2.0×) and p99 (1.7-1.9×). UDP against UDP is the
like-for-like pair: both stamp each datagram's arrival right after `recvfrom` returns. The
replay front end stamps a burst when it is gathered, after the 20 ms partial-burst timeout
(at 2,344 packets/s every burst closes on it, about 47 packets), so its figure excludes up to
20 ms of a packet's wait and is not comparable with UDP
([timing scopes](METHODOLOGY.md#timing-scopes)). Every run's dashboard served `/`,
`/snapshot` (`error` null), `/capture`, `/feedrate`, `/plan` and `/healthz`, rejected an
empty `/manifest` with 422, and the receiver exited 0. `/healthz` answered 503 until the
front end received (1.0-2.5 s after start over UDP, 7.1-9.4 s with the replay front end,
which stages the capture in GPU memory first) and the sender started on its 200.

Without the iGPU sidecar, the UDP run's iGPU stayed suspended through the capture; in the
replay run it was resumed once, 77 s in, by something outside the run, for about 5 s. The
steady state allocates nothing, so nothing waited for it: both runs are within or below the
range of the runs with the sidecar.

In `compare` mode every window of the capture was compared (UDP: 417 comparisons over the
run, replay: 399), every one agreeing with matched work, with no gate wait and no
superseded comparison; `rolling_timing` CPU 5.66 / GPU 0.598 ms (9.5×) over UDP and 5.61 /
0.564 ms (9.9×) with the replay front end, 120 windows each. The UDP `compare` run's GPU
packet to result is above the GPU-only runs': in `compare` mode both backends run on the
host at once (see the anomalies).

## Anomalies and findings

- **iGPU runtime power management.** See [what runs where](#what-runs-where). The risk
  engine allocates its workspaces once per universe, for the bars' full window, and nothing
  per window of any kind (`gpu/test_gpu_covariance.py` counts every call into CuPy's device
  and pinned allocators), so a steady run never waits for a resume (step 11, no sidecar).
  The first window of a universe still allocates.
- **Fixed: `/healthz` answered before the UDP socket was bound.** A sender started on it
  lost its first datagrams, and for that leading loss the two gap counters differed.
  `/healthz` now answers 200 only once the ingest front end receives (`network.ready`) and
  503 before (`test_healthz_answers_200_only_while_the_ingest_front_end_receives`), and
  `replay_sender` waits for it. After a finite replay's last pass the front end no longer
  receives, so `/healthz` is 503 again while the graph and the dashboard keep running; its
  body's `input_ended` tells that state from one that has not started
  (`test_healthz_tells_a_finished_replay_from_one_not_started`). The counters stay distinct and are labelled on the page:
  "Sequence gaps" counts what the receiver saw missing between a sender's packets, "Bar
  input gaps" what the quote bars did not get from each generation's sequence 1 (ring drops
  and anything sent before the receiver was ready included) ([architecture](ARCHITECTURE.md)).
- **Fixed: one delayed publication per process in `compare` mode.** The CPU worker re-tuned
  its BLAS threads at the first full window, inside a validated window's job, so the next
  GPU window waited for the gate. The workers now tune once per universe when its manifest
  arrives, long before its first window, and no window tunes
  (`test_the_worker_tunes_before_the_first_window_so_no_validated_window_waits_for_it`).
  In `replay` mode the universe is registered before the graph runs and the replay starts
  with the source, so a tuning started at the operator's first `compute` still overlapped
  the first windows, and the first comparison, and with it the next GPU result, waited for
  it. The operators now tune such a universe in `start()`, before any window
  (`test_a_universe_registered_before_the_run_is_tuned_in_start`).
- **Fixed: the BLAS tuning misread its first candidate.** A process's first engine slides
  faster than every later one, the engine the worker keeps included, and the tuning timed
  its first candidate, one thread, on it. At 1,000 symbols the 10% margin then could keep
  one thread where four were clearly faster. The tuning now times every candidate twice and
  keeps the second pass (`test_the_tuning_keeps_the_times_of_a_second_pass`).
- **Fixed: garbage collections inside the analytics timer.** A collection starts at
  whichever allocation crosses the collector's threshold; one that started inside a window
  added its pause to that window, and the GPU path, which allocates more tracked objects
  per window, met one more often than the CPU path. These made the slowest windows of a
  kind: with 40 windows per cell (`--phases 4`), one of them could make a p99. Both engines
  now hold automatic collection off while they time a window
  (`test_no_collection_starts_while_a_window_is_timed`).
- **Fixed: the replay source freed the capture's host copy after starting the replay.**
  Freeing it held the interpreter lock while the drain was already on its schedule, so the
  first bursts were gathered late and the paced lag gate failed. The source now frees the
  host copy before the drain starts
  (`test_the_host_copy_of_the_capture_is_freed_before_the_drain_starts`).
- **Fixed: windows longer than 64 bars reset the risk engine at every growth step.** With
  `--window-seconds 600` (120 bars) or `--bar-seconds` below 5, each growth window past 64
  bars reallocated the workspaces and rebuilt the statistics, so `compare` mode reported
  unmatched work and failed. The engines are now sized for the bars' full window and grow,
  keeping their statistics, if a window is longer
  (`gpu/test_gpu_risk.py::test_compare_mode_matches_work_on_windows_longer_than_64_bars`).
- **Fixed: a large move followed by quiet trading.** Once the move expires, its rounding in
  `C` could dominate a quiet name's variance, and a variance that cancelled to zero gave NaN
  correlations, which then failed every `/snapshot`. The engines recompute such a window
  from its bars ([METHODOLOGY](METHODOLOGY.md#statistics)), and `/snapshot` encodes a
  non-finite value as null (`library/test_covariance.py`, `gpu/test_gpu_covariance.py`,
  `control/test_dashboard_contract.py`).
- **Open: a sequence jump after two UDP captures.** In two of the three UDP GPU-only runs of
  step 11 the receiver's two gap counters rose together, by 4,787 and 4,028, in the two
  seconds after the capture ended (none inside any capture, none in the other runs). The
  packets received around the capture match the runs without it, so the counters saw a
  jump in sequence numbers rather than missing traffic. Not investigated yet.
- **The step 3 command needs `--threads`.** See [step 3](#3-offline-kernels-timing-pass).
- **Compare-mode speedups are not the separate-runs measurement.** Compare mode runs both
  backends on the same host at once, and the CPU job starts when the GPU result is ready, so
  its ratio is not that of separate CPU-only and GPU-only runs
  ([METHODOLOGY](METHODOLOGY.md#separate-cpu-only-and-gpu-only-benchmark), step 11).

## Short NIC check

Three short runs of the final code on the ConnectX-7, on 2026-10-22 between 11:16 and 11:24
EDT: `gpudirect` in `compare` mode, then `daqiri-host` GPU-only and CPU-only. The
[Thor launchers](../deploy/igx_thor/README.md) start the receiver image
([Dockerfile](../deploy/igx_thor/Dockerfile): Holoscan SDK 5.0.0 and DAQIRI) with the
checkout mounted read-only, the port in PHY loopback (`LOOPBACK=1`) and the launcher's
paced `replay_sender` on the recorded day's tape (the default `TAPE_DIR`). These commands
run on the host, from the repository root, not in `scripts/dev5.sh`:

```bash
export LOOPBACK=1 SPEED=0 RATE=200000 GPU_UUID=GPU-<uuid>   # the RTX PRO 6000 (nvidia-smi -L)
bash deploy/igx_thor/igx_loopback.sh --check                 # before, and after each teardown
INGEST=gpudirect ANALYTICS_BACKEND=compare bash deploy/igx_thor/igx_run.sh
PYTHONPATH=python python3 -m holoscan_finance.tools.measure --seconds 60 --rate 200000 --output out/nic-gpudirect-compare.json
bash deploy/igx_thor/igx_down.sh
INGEST=daqiri-host ANALYTICS_BACKEND=gpu bash deploy/igx_thor/igx_run.sh   # then cpu
PYTHONPATH=python python3 -m holoscan_finance.tools.measure --seconds 60 --rate 200000 --output out/nic-host-gpu.json   # nic-host-cpu.json
bash deploy/igx_thor/igx_down.sh
PYTHONPATH=python python3 -m holoscan_finance.tools.compare_runs --cpu out/nic-host-cpu.json --gpu out/nic-host-gpu.json --output out/nic-host-separate.json
```

`RATE` is in observations/s, and 200,000 is the most `/feedrate` accepts. The Python sender
tops out below it: it delivered 4,582-4,694 packets/s (146,610-150,203 observations/s).
200,000 *packets*/s on the NIC needs `tcpreplay` (the dashboard's full-day burst or
[`gpu_ingest_bench/sweep.sh`](../deploy/igx_thor/gpu_ingest_bench/README.md)), which was
not run, so this check makes no claim at that rate. `measure` seeks to 14:00 UTC, waits for
a complete window and records 60 s.

| Run | Packets (packets/s) | Windows | Analytics p50 / p99 (ms) | Packet to published result p50 / p99 (ms) |
| --- | --- | --- | --- | --- |
| `gpudirect`, `compare` | 281,630 (4,694) | 370 | GPU 0.555 / 1.82, CPU 4.88 / 12.2 | GPU **4.21 / 6.49**, CPU 12.7 / 21.8 |
| `daqiri-host`, GPU only | 275,259 (4,588) | 359 | 0.552 / 1.06 | **9.82 / 18.0** |
| `daqiri-host`, CPU only | 274,893 (4,582) | 359 | 5.40 / 7.94 | 14.7 / 21.3 |

- **Ingest.** Every capture is `valid`, with 0 sequence gaps, ring drops, malformed, stale
  and out-of-order packets, and no ring overflow; the `gpudirect` receiver logged no
  `Rx: Dropped`.
- **`compare` mode.** All 370 GPU windows of the capture were compared, and every one agreed
  with matched work (analytics 8.8× faster, matched).
- **Separate runs.** `compare_runs` is valid: on the 180 predefined windows (14:10-14:25
  UTC) both runs had identical inputs, agreed and did the same work (178 incremental, 2
  rebuilds). Analytics are 9.8× faster at the median on the GPU. On `daqiri-host`, decode
  and bars run in Python on the host, which is why its packet to result is about 2.3× the
  `gpudirect` figure at the same rate (p50 9.82 against 4.21 ms).
- **Dashboard.** During the `gpudirect` run, a headless browser found no console error or
  failed request, every rendered value matched `/snapshot`, pause and resume worked, the
  full-day panel was hidden (no `FULLDAY_PCAP`) and a 390 px wide view did not scroll
  sideways.
- **Host.** The `daqiri-host` runs had only the idle DPDK container beside them. The
  `gpudirect` run shared the RTX PRO 6000 with a `risk_bench` run of this module and the
  iGPU with an idle CUDA context, so its latencies are an upper bound. Load average was
  5.3-7.3. After the last teardown, `igx_loopback.sh --check` passed, the port had no
  address or neighbor entry, and the iGPU power control, `fwupd-refresh.timer` and the
  hugepages were as before.

## What this does not show

The replay drain is a Python thread, so these runs establish function, throughput headroom
and the analytics timings, not resilience to GIL stalls: that claim stays with the native
drain on the NIC. Bit identity and the paced lossless gate (steps 2, 4 and 5) are not
recorded for the final code. Beyond the [short NIC check](#short-nic-check) at the paced
sender's rate, these NIC runs need explicit approval and are not recorded for this code:
200,000 packets/s with `tcpreplay`, the 250-run full-day series and the `/debug/hold_gil`
fault-injection A/B at 1.1 s and 3 s (native vs Python drain). No claim is made for them.
