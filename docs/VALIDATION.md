# Synthetic validation on Holoscan SDK 5.0

This guide exercises the GPU ingest, decode, quote bars and rolling analytics with
deterministic synthetic input. It specifies commands and pass gates; it does not report
measurements. Keep each generated input summary with its reports so a result can be
reproduced on the same code and host.

## Environment and timing discipline

Run from the repository root in a Holoscan SDK 5.0 environment with the project's
requirements installed. GPU commands need CuPy and a CUDA device; compiled CPU decode
also needs numba. `holoscan run` builds and uses the dev container; add `--local` when
already inside that environment. See the [development setup](../README.md#develop-and-test).

Run timing checks serially on an otherwise idle host. Retain the commit, SDK and CUDA
versions, GPU model, CPU affinity, thread settings, other host workloads and GPU power
state with the reports. On Thor, an idle integrated GPU may suspend and delay a later
CUDA allocation while resuming; record whether it was kept awake. Warm-up and compilation
must finish before a streaming timing interval begins.

```bash
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$PWD/python"
mkdir -p out
```

The software replay exercises the production gather, ring, decode and bars without a
NIC. Its drain is a Python thread; hardware receive and native-drain stall resilience
need the [NIC checks](#nic-validation).

## Synthetic inputs

Generate the small input for library checks and the full-size input for timing and
streaming measurements. The small input matches the `market_data_bench smoke` preset:

```bash
python3 -m holoscan_finance.tools.synth \
    --out data/synth-small --symbols 500 --packets 50000 --seconds 1800 --seed 7 --reuse
python3 -m holoscan_finance.tools.synth \
    --out data/synth-day --symbols 2886 --packets 1824000 --seconds 23400 --seed 7 --reuse
python3 -m holoscan_finance.tools.bench_replay inspect data/synth-small/day.pcap --port 40000
python3 -m holoscan_finance.tools.bench_replay inspect data/synth-day/day.pcap --port 40000
```

Each output directory contains `tape/`, `day.pcap` and `synth.json`. The summary records
the seed, arguments, file sizes and exact counts expected by a lossless receiver. The
`--packets` argument specifies data packets; one final status-marker packet closes the
last bar. An SBE entry is one observation, so entry counts equal observation counts.
`--reuse` reuses files only when the generation arguments and recorded sizes match.
`--encoder gpu` packs and frames the capture with the CUDA encoder (`gpu_sbe.encode_units`
and `frame_units`); it writes the same bytes as the NumPy packer, so either encoder may
reuse the other's output. Compare the two with `cmp` on the pcaps when changing either.

Pass: generation and inspection complete, and inspected packet and message
counts match the generated capture summary. The replay directory comes from the matching
tape; do not pair a capture with a different manifest.

## Library bit identity

```bash
python3 -m holoscan_finance.tools.gpu_validate \
    --pcap data/synth-small/day.pcap --tape data/synth-small/tape --out out/decode_gpu.json
python3 -m holoscan_finance.tools.gpu_validate \
    --pcap data/synth-small/day.pcap --tape data/synth-small/tape --decoder cpu --out out/decode_cpu.json
python3 -m holoscan_finance.tools.gpu_validate \
    --pcap data/synth-small/day.pcap --tape data/synth-small/tape --bars --out out/bars.json
python3 -m holoscan_finance.tools.gpu_validate \
    --pcap data/synth-small/day.pcap --tape data/synth-small/tape --bars --drop-every 5000 --out out/bars_drops.json
```

Pass: each command exits zero. Decode reports must be `bit_identical`, including the
default 240,000 mutated packets. Bar state, counters and NaN patterns must match the host
reference, and finite returns must be within 1 ULP. The dropped-packet run must produce
identical sequence-gap handling and recovery. For a larger correctness pass, use
`data/synth-day/day.pcap` and its tape with the same commands.

The wire format itself is checked against the FIX SBE reference tool: `sbe-tool`
(`uk.co.real-logic:sbe-all`) validates
[`sbe_schema.xml`](../python/holoscan_finance/marketdata/sbe_schema.xml) with warnings
fatal and generates a C++ codec, from which a probe prints the layout (identifiers, block
lengths, every field's offset, length, id and null value, the enum values) that must equal
what `sbe_codec` derives from the XML, and which must decode the hand-transcribed golden
packet as the codec does. The check needs `java` (17 or later), `g++` and the jar:

```bash
bash scripts/fetch_sbe_tool.sh      # pinned version and SHA-256, into .cache/sbe
python3 -m pytest -q tests/python/library/test_sbe_tool.py   # skipped without the three
```

## Graph bit identity

```bash
holoscan run market_data_bench smoke
holoscan run market_data_bench check
holoscan run market_data_bench drops
```

The modes generate their own synthetic inputs. With `--check`, the same decoded stream
feeds GPU and CPU `QuoteBarsOp` instances, and the report compares every GPU window with
its CPU twin. An independent host decode also checks every closed bar after the run.

Pass: each mode exits zero and its report is `ok`; window times, state and NaN patterns
match, and finite returns differ by at most 1 ULP. `smoke` and `check` require lossless
replay. `drops` deliberately omits every 997th data packet, so its gate is identical
recovery and expected gap accounting.

## Paced replay and throughput

```bash
holoscan run market_data_bench paced
holoscan run market_data_bench throughput
holoscan run market_data_bench pcap
```

These modes generate or reuse `data/synth-day`. `paced` includes the NIC-buffer model
and GPU risk. Its gates are zero loss, real-time completion and replay lag p99 ≤ 5 ms at
200,000 packets/s. `throughput` runs unpaced and requires at least 12M SBE entries/s.
`pcap` replays the same synthetic input at 200,000 packets/s with lossless and real-time
gates. The numeric thresholds are requirements, not reported performance.

Repeat timing runs separately and retain every report together with host conditions.
The [timing scopes](METHODOLOGY.md#timing-scopes) explain where the receive timestamp is
taken; software replay latency does not include all waiting before a burst is gathered.

## Full chain with compare

```bash
python3 examples/equity_analytics/equity_analytics.py \
    --ingest replay --pcap data/synth-day/day.pcap --tape data/synth-day/tape \
    --replay-pps 200000 --analytics-backend compare --port 0 --exit-when-drained \
    --report out/chain.json --expect-windows 10
```

Pass: exit zero, at least ten analyzed windows, and every comparison agrees with matched
work (the same incremental or rebuild method, added bars and expired bars). Retain
packet-to-result p50/p99 and `rolling_timing` as measurements from this run. `compare`
executes both backends together, so its timing ratio is distinct from separate CPU-only
and GPU-only runs.

## Offline decode and bars

```bash
python3 -m holoscan_finance.tools.decode_bench \
    --pcap data/synth-day/day.pcap --tape data/synth-day/tape \
    --group-msgs 16384 --threads 1,4,8,12 --out out/decode_16k.json
python3 -m holoscan_finance.tools.decode_bench \
    --pcap data/synth-day/day.pcap --tape data/synth-day/tape \
    --group-msgs 131072 --threads 1,4,8,12 --out out/decode_131k.json
```

These compare Python, compiled CPU and GPU decode on the same packets, both alone and
with quote bars, and the NumPy and GPU encoders on the tape's observations. The default
limit is 200,000 packets. Select thread counts available inside the process's CPU
affinity. Compilation is excluded; GPU timing includes the per-packet results' copy to
the host and synchronization. The CPU bars use NumPy; a compiled, multi-core CPU bars
implementation is not measured.

Pass: commands complete after the correctness checks above. Report throughput for each
batch size and thread count without treating a speedup as a correctness gate.

## Analytics per window

```bash
python3 -m holoscan_finance.tools.risk_bench \
    --kinds 50 --cadence-ms 0 --sizes 50,500,2886 \
    --backends gpu,gpu-host,cpu,compare --returns data/synth-day/tape \
    --save-returns out/synthetic-bars.npz --check --out out/risk.json --table out/risk.md
```

`gpu` runs inline on device bars. `gpu-host` and `cpu` use the worker functions with the
new bars of each window. `compare` pairs the GPU processor with the spawned CPU
cross-check. The benchmark covers a generation's first window, growth, rebuilds and
one-, two- and three-bar slides, with an untimed warm-up cycle first. `--returns` builds
bars from the synthetic tape, including its invalid marks and missing returns.

Pass: every requested oracle and CPU/GPU check completes, and compared results agree
with matched work. The report's timing verdict identifies cells where a GPU path is not
below the CPU at p50 or p99; retain those cells along with the complete table. For more
samples, increase `--kinds`; for idle intervals between windows, use `--cadence-ms 270`.
Neither a lower latency nor a speedup is assumed before measuring.

## Separate CPU and GPU runs

Use the same full-size synthetic tape, host front end and rate for both runs. In one
terminal, start the CPU receiver:

```bash
python3 examples/equity_analytics/equity_analytics.py \
    --ingest udp --analytics-backend cpu --port 8080
```

In a second terminal, start the sender:

```bash
python3 -m holoscan_finance.tools.replay_sender \
    --data data/synth-day/tape --port 26400 --control-url http://127.0.0.1:8080
```

Once the dashboard shows the registered feed, capture from a third terminal:

```bash
python3 -m holoscan_finance.tools.measure \
    --seconds 120 --rate 75000 --output out/cpu-only.json
```

Stop the receiver and sender. Repeat the same commands with `--analytics-backend gpu`
and `--output out/gpu-only.json`, then compare:

```bash
python3 -m holoscan_finance.tools.compare_runs \
    --cpu out/cpu-only.json --gpu out/gpu-only.json --all-shared --output out/separate.json
```

`measure` seeks the sender to 14:00 UTC, waits for a complete window, captures for
120 seconds and restores its controls. The full-session input covers that seek and the
measurement interval. Pass: both captures and their comparison are valid, with identical
input hashes, agreeing results and matched work on shared windows. Report analytics and
packet-to-result p50/p99 separately; compare packet-to-result only between the same front
end. See the [measurement conventions](METHODOLOGY.md#separate-cpu-only-and-gpu-only-benchmark).

## Test suite and dashboard

```bash
python3 -m pytest -q -m "not timing" tests/python
python3 -m pytest -q -m timing tests/python
holoscan run equity_analytics smoke
```

Run timing tests separately on an idle host. Unsupported SDK, GPU or DAQIRI tiers skip;
retain the skip summary so it is clear which paths ran. The non-timing suite includes
`tests/python/e2e/test_dashboard_live.py` when its dependencies are available.
Pass: applicable tests and smoke gates pass, and no required validation tier is skipped.

## NIC validation

Use the [Thor launchers](../deploy/igx_thor/README.md) and
[GPU ingest sweep](../deploy/igx_thor/gpu_ingest_bench/README.md) with the generated tape and
pcap. Follow their host coordination and ConnectX setup requirements. Check NIC drops,
receiver sequence gaps, ring drops and decoded observations against the synthetic input
summary. Evaluate both loss and backlog: a run can lose no packets while falling behind.

The [native-drain procedure](GPU_INGEST.md#native-drain) compares interpreter-lock stalls
with the native and Python drains. Keep these results separate from software replay:
the software source cannot establish hardware receive rate or native-drain resilience.

## Interpreting reports

Record the input's `synth.json`, code version and host settings with every report. A
correctness pass establishes agreement on those synthetic inputs; a timing pass applies
to that hardware, batch size, pacing and workload. Uniform replay does not model every
external-feed burst pattern. Report failures and skipped checks alongside successful
runs, and preserve the [timing boundaries](METHODOLOGY.md#timing-scopes) when comparing
measurements.
