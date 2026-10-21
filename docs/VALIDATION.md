# NIC-free validation on Holoscan SDK 5.0

This page records how the Holoscan 5.0 port is validated on a GPU host without the NIC, and
the results. Every run uses synthetic data and the NIC-free replay source
(`GpuMitchReplaySourceOp`): pcap frames staged in GPU memory go through the production
gather kernel, device ring, receive accounting, GPU MITCH decode and GPU quote bars. No run
touches the NIC, hugepages, IRQs or the demo containers.

> **Status.** The functional steps 1, 2 and 4 ran on 2026-10-20 and pass ([results
> below](#functional-results-2026-10-20)), and step 9 passed in the integration pass the same
> day. Steps 3, 5, 6, 7, 8 and 10 run in the serial timing pass. Nothing on this page is a
> measured 5.0 result until its row is filled in.

## What runs where

- **Container.** Every command runs inside the Holoscan 5.0 image through
  [`scripts/dev5.sh`](../scripts/dev5.sh): non-root, private IPC (never `--ipc=host`), CPUs
  `0-2,4-8,10-13` (off the DAQIRI lcores 3 and 9), the RTX PRO 6000 by PCI bus order, and a
  timeout inside the container. Full-day steps set `RUN5_TIMEOUT=3600`, because the bench's
  own `--timeout-s 1800` exceeds the 1500 s default.
- **Data.** `data/` and `out/` under the repository root (both gitignored).
- **Shared GPU.** The live 4.3 demo may share the GPU, so timing compares **medians only**,
  and each run records the host state (below).
- **Functional steps** (1, 2, 4, 9) are part of the acceptance of the validation work
  package. **Timing steps** (3, 5, 6, 7, 8, 10) run only in the serial timing pass, after
  the integration pass, with no other agent container running.

## Host state

Recorded before and after each session.

| When | `docker ps` (names) | `nvidia-smi` GPU utilization / memory | Notes |
| --- | --- | --- | --- |
| functional runs, before (2026-10-20 19:49 EDT) | `nervous_morse`, `ano-igx` | RTX PRO 6000: 0 %, 47 MiB | `nervous_morse` is an auto-named container, not a demo container; `ano-igx` was not touched |
| functional runs, after (2026-10-20 20:17 EDT) | `ano-igx` | RTX PRO 6000: 0 %, 47 MiB | steps 2 and 4 overlapped each other and the `synth-drops` generation: functional results only |
| step 9, before and after (2026-10-20 20:30 and 20:31 EDT) | `ano-igx` | RTX PRO 6000: 0 %, 47 MiB | integration pass |
| timing pass, before | _pending_ | _pending_ | |
| timing pass, after | _pending_ | _pending_ | |

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
the published 4.3 day (2,880,565 packets, 92,178,039 observations, 2,886 symbols): 2,880,001
packets including the end-of-session marker, and 92,160,000 observations.

### 2. Library bit identity (GPU vs host decode and bars)

```bash
python3 -m holoscan_finance.tools.gpu_validate --pcap data/synth-day/day.pcap --tape data/synth-day/tape --out out/gpu_validate.json
python3 -m holoscan_finance.tools.gpu_validate --pcap data/synth-day/day.pcap --tape data/synth-day/tape --bars --out out/gpu_validate_bars.json
python3 -m holoscan_finance.tools.gpu_validate --pcap data/synth-day/day.pcap --tape data/synth-day/tape --bars --drop-every 5000 --out out/gpu_validate_bars_drops.json
```

Pass: decode `bit_identical`, including the fuzz pass's rejection decisions; bars
`state_identical` (marks, times, trades, volumes and counters identical, returns within
1 ULP). Reference (Holoscan 4.3, GPU_INGEST): on the real day 518 of 11.6M returns differ by
1 ULP.

### 3. Offline kernels (timing pass)

```bash
python3 -m holoscan_finance.tools.decode_bench --pcap data/synth-day/day.pcap --tape data/synth-day/tape --group-msgs 16384 --out out/decode_bench_16k.json
python3 -m holoscan_finance.tools.decode_bench --pcap data/synth-day/day.pcap --tape data/synth-day/tape --group-msgs 131072 --out out/decode_bench_131k.json
```

Report only (shared GPU). Reference (Holoscan 4.3): 17.7M msgs/s at 16k-message groups,
76M msgs/s at 131k and up (decode + bars). The compiled CPU rows need numba>=0.68, which
`scripts/dev5.sh` installs: the 5.0 image ships numba 0.64, which refuses its NumPy 2.5.

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

Pass: all three runs `ok`. Reference (Holoscan 4.3, NIC): lossless at 200k packets/s in
242 of 250 full-day runs.

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
p99 reported. Reference (Holoscan 4.3, NIC, 20 ms burst timeout): GPU 34.22 / 49.83 ms, CPU
66.91 / 100.30 ms; the same order is expected, as the replay source emulates the 20 ms
timeout.

### 8. Analytics speedup (timing pass)

The `rolling_timing.speedup` median of the step 7 run. Reference (Holoscan 4.3): 9.55×
(GPU 3.89 ms).

### 9. Dashboard contract

```bash
python3 -m pytest -q -m slow tests/python/e2e/test_dashboard_live.py
```

Pass: the test passes (every 4.3 `/snapshot` key present, `network.ingest == "replay"`).

### 10. Holoscan 4.3 vs 5.0 A/B (timing pass)

The 4.3 sources come from a read-only export of `43bf4e1`, run in a new throwaway 4.3
container (non-root, private IPC, the same CPU set); the running demo containers are never
touched:

```bash
mkdir -p .cache/ab43-src && git archive 43bf4e1 | tar -x -C .cache/ab43-src
docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e CUDA_VISIBLE_DEVICES=1 -u $(id -u):$(id -g) -e HOME=/tmp -e CUPY_CACHE_DIR=/tmp/cupy \
  --cpuset-cpus=0-2,4-8,10-13 --name ab43-$USER \
  -v $PWD/.cache/ab43-src:/workspace/holoscan-finance:ro -v $PWD/data:$PWD/data:ro \
  -e HOLOSCAN_FINANCE_REPO=/workspace/holoscan-finance -w /workspace/holoscan-finance \
  --entrypoint bash holoscan-finance:nyq-20260911 \
  -c 'timeout -s KILL 1200 bash deploy/igx_thor/ano_python.sh -m holoscan.equities.decode_bench --pcap <...> --tape <...>'
```

The same for `holoscan.equities.gpu_validate`, then the 5.0 tools on the same
`data/synth-day` capture. Pass: kernel medians agree within noise (the kernels are
byte-identical); the 5.0 graph overhead is reported in absolute terms.

## Results

| Step | Command (report) | Result | Pass |
| --- | --- | --- | --- |
| 1 Data | `synth` (`data/synth-day/synth.json`) | 2,880,001 packets, 95,062,715 messages; drops: 576 packets missing | pass |
| 2 Library bit identity | `gpu_validate` (`out/gpu_validate*.json`) | decode and fuzz bit-identical; bars identical, 41 and 39 returns differ by 1 ULP | pass |
| 3 Offline kernels | `decode_bench` (`out/decode_bench_*.json`) | _pending (timing pass)_ | report |
| 4 Graph bit identity | `market_data_bench --check` (`out/check*.json`) | both `ok`; 701 and 700 window pairs, 4,679 bars against the host decode, `max_ulps` 1; drops: 19,011 gaps on every side | pass |
| 5 Paced lossless | `market_data_bench` paced (`out/paced_*.json`) | _pending (timing pass)_ | _pending_ |
| 6 Throughput headroom | `market_data_bench` unpaced (`out/tput_*.json`) | _pending (timing pass)_ | _pending_ |
| 7 Full chain with compare | `equity_analytics --ingest replay` (`out/chain.json`) | _pending (timing pass)_ | _pending_ |
| 8 Analytics speedup | step 7 `rolling_timing.speedup` | _pending (timing pass)_ | report |
| 9 Dashboard contract | `e2e/test_dashboard_live.py` | 1 passed in 28.5 s (integration pass, 2026-10-20) | pass |
| 10 4.3 vs 5.0 A/B | `decode_bench`, `gpu_validate` in both images | _pending (timing pass)_ | report |

The timing tests of the test suite (`pytest -m timing`) also run in the timing pass:

```bash
RUN5_TIMEOUT=3600 scripts/dev5.sh 'python3 -m pytest -q -m timing tests/python'
```

| Timing tests | Result |
| --- | --- |
| `pytest -m timing` | _pending (timing pass)_ |

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

## What this does not show

The replay drain is a Python thread, so these runs establish function, bit identity,
throughput headroom and paced losslessness, not resilience to GIL stalls: that claim stays
with the native drain on the NIC. Three runs need the user's explicit approval and are not
part of this page until they run: the NIC smoke run, the 250-run full-day series (4.3:
242/250 lossless) and the `/debug/hold_gil` fault-injection A/B at 1.1 s and 3 s (native vs
Python drain; 4.3: 8/9 clean with the native drain). No Holoscan 5.0 parity claim is made
for them until then.
