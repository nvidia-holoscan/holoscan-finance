# Holoscan Module for Financial Services Industry

`holoscan-finance` is a self-contained [Holoscan SDK 5](https://developer.nvidia.com/holoscan-sdk)
Module for building streaming financial-services applications. It provides reusable
operators for market-data ingest, decoding, bar building and rolling risk analytics, the
libraries behind them, command-line tools, and two example applications composed from the
operators. Everything is pure Python: the package `holoscan_finance` is published by
Holoscan as `holoscan.finance`, and the module runs on the
`nvcr.io/nvidia/clara-holoscan/holoscan:v5.0.0-cuda13` image with `holoscan-cli` 5.0.0.

| The module provides | Where |
| --- | --- |
| Ten [operators](#operators): four receive sources (kernel UDP, DAQIRI into pinned host memory, DAQIRI GPUDirect into GPU memory, and a NIC-free GPU replay of a capture), SBE decode, event-time quote bars, rolling GPU risk, a CPU cross-check, a callback sink and a holomq risk summary | `holoscan_finance.operators`; a catalog entry per operator under [operators/](operators/README.md) |
| The libraries the operators wrap: the FIX SBE schema and codec, CUDA encode/decode and quote-bar kernels, the covariance engine, receive accounting, the control plane, telemetry and validation, all tested without Holoscan | `holoscan_finance.marketdata`, `equity`, `ingest`, `analytics`, `control`, `telemetry`, `validation` |
| Application plumbing (flags, authored lanes and CPU placement, plan rules, runner, drained predicate) and a web dashboard | `holoscan_finance.app`, `holoscan_finance.dashboard` |
| CPU-only [tools](#tools): synthetic tape and pcap generation, a paced replay sender, measurement and benchmarks | `holoscan_finance.tools` |
| Two [example applications](#examples): a live dashboard and a headless, verified benchmark, both `holoscan run` targets | [examples/](examples/) |

An application built on the module is a composition: it picks operators, wires them with
the module's edge presets, places its lanes and states the plan rules it relies on. It
defines no operator classes. [Composing an application](#composing-an-application) shows
the pattern, [architecture](docs/ARCHITECTURE.md) explains how the examples are laid out,
and [adding an operator](docs/ARCHITECTURE.md#adding-an-operator) extends the module.

## Examples

The examples replay deterministic synthetic equity trades and quotes encoded with
[FIX Simple Binary Encoding 1.0](https://www.fixtrading.org/standards/sbe/) over UDP (one
SBE message per datagram under the module's own [schema](python/holoscan_finance/marketdata/sbe_schema.xml):
a repeating group of fixed 48-byte entries, one per observation) and continuously compute
five-second quote returns, historical volatility and cross-instrument correlation. A
Holoscan SDK 5 graph runs the receiver, SBE decode, quote bars and analytics; custom CUDA
kernels (CuPy) compute the rolling statistics on the GPU, and a NumPy/SciPy worker computes
the same windows with the same algorithm on the CPU for comparison. With GPUDirect ingest,
packets go from the NIC straight into GPU memory and CUDA kernels decode them (one warp per
packet, one lane per entry) and build the quote bars; the synthetic captures can be encoded
on the GPU too.

Generate the input with `python -m holoscan_finance.tools.synth`; the examples' replay
modes generate it automatically. See [methodology](docs/METHODOLOGY.md) for the synthetic
model, wire format and calculation conventions, and [validation](docs/VALIDATION.md) for
reproducible commands and pass gates.

| Example | What it is | Front ends |
| --- | --- | --- |
| [`equity_analytics`](examples/equity_analytics/README.md) | The live dashboard: SBE feed → quote bars → rolling risk → web UI, with an optional live CPU cross-check of every validated GPU window (`--analytics-backend compare`) | `udp` (kernel socket), `daqiri-host` (DAQIRI into pinned host memory), `gpudirect` (DAQIRI GPUDirect into GPU memory), `replay` (pcap frames staged in GPU memory: the GPUDirect chain without a NIC) |
| [`market_data_bench`](examples/market_data_bench/README.md) | Headless, finite and verified: replays a pcap from GPU memory through gather → decode → quote bars (→ risk) and reports throughput, loss, lag and placement against pass/fail gates. With `--check`, the same decoded stream also feeds a CPU `QuoteBarsOp`, and every GPU window is compared with its CPU twin in the graph | `replay` |

Both are plain compositions: an example file builds the graph from
`holoscan_finance.operators`, states the plan rules it promises, and calls the module's
shared plumbing. It defines no operator classes. The Holoscan 5 features they show, none of
them on the per-packet path:

| Feature | `equity_analytics` | `market_data_bench` |
| --- | --- | --- |
| Declarative operators, typed `Param`s validated at `graph.op`, live objects injected as `Ref`s | yes | yes |
| `Param`-gated optional ports (`emit_windows`, `cross_check`) | yes | yes |
| `OnClock` sources with hold-and-retry on `BLOCKING` edges | yes | yes, including fan-out |
| A lossless same-lane `OnEach` chain: no thread hop per packet | yes | yes |
| A `DROP_OLDEST`/1 latest-only mailbox into an `OnClock(250 µs)` + `DrainAll` consumer | yes | `--risk gpu` |
| Authored lanes with `ThreadPlacement` (CPU affinity, optional `BUSY_POLL`), off the DAQIRI cores | yes | yes |
| Downstream-first `start()` warm-up: kernels are compiled before the NIC starts | yes | yes |
| `run_async` under a FastAPI lifespan, with run diagnostics in `/snapshot` | yes | - |
| Plan rules checked on `plan.json()`; `--plan-only`, `--plan-json`, `GET /plan` | yes | yes |
| One stream fanned out to two backends of the same operator (in-graph bit identity) | - | `--check` |
| A deterministic finite run: a drained predicate, then `request_stop()` | `--exit-when-drained` | yes |
| `max_start_delay` + `EMIT_WITNESS` lateness counter | `--risk-deadline-us` | - |
| holomq `export_topic` of a compact risk summary, latest-only | `--topics` | - |
| `holoscan run <app> <mode>` and the `holoscan.finance` module alias | yes | yes |

How the graphs are laid out, and why: [architecture](docs/ARCHITECTURE.md).

## Quick start: the full GPU chain on any GPU host, no NIC

```bash
python3 -m pip install -r requirements-cli.txt   # holoscan-cli 5.0.0 (the SDK image does not ship it)
holoscan run equity_analytics replay             # then open http://localhost:8080/
holoscan run market_data_bench smoke             # headless: GPU vs CPU bars, lossless gate, JSON report
```

`holoscan run` builds the dev image from the [Dockerfile](Dockerfile) (the 5.0 SDK image
plus holoscan-cli and the test dependencies), mounts the checkout, writes a synthetic day
under `data/` (`python -m holoscan_finance.tools.synth`) and runs the mode's command. The
`replay` mode stages the day's pcap frames in GPU memory and replays them at 2,000
packets/s, in a loop, through the production gather, ring, SBE decode and quote-bar
code, with live CPU validation. Select a GPU with
`CUDA_VISIBLE_DEVICES=<n> CUDA_DEVICE_ORDER=PCI_BUS_ID`, which the CLI forwards. Inside the
dev container (`holoscan run-container`), or anywhere the 5.0 SDK is installed, add `--local`.

| `holoscan run equity_analytics <mode>` | What runs |
| --- | --- |
| `replay` (default) | Synthetic day, replay front end, `compare` analytics, dashboard on port 8080 |
| `replay_pcap` | A generated full-size synthetic day (`data/synth-day/day.pcap`, `data/synth-day/tape`) at 200,000 packets/s |
| `udp`, `udp_cpu` | Kernel UDP front end (`compare` or CPU analytics); start the sender separately (below) |
| `smoke` | Headless CI run: exits non-zero if the run fails, no window is analyzed or a comparison disagrees |
| `plan` | Compiles the graph and prints the plan summary; no GPU work, no I/O |

| `holoscan run market_data_bench <mode>` | What runs |
| --- | --- |
| `smoke` (default) | 50,000 synthetic packets, unpaced, `--check --require-lossless` |
| `check` | 1M packets, in-graph GPU vs CPU bars and an independent host decode of every closed bar |
| `paced` | A full-size synthetic day at 200,000 packets/s with the NIC-buffer drop model and GPU risk: lossless, real time, lag p99 ≤ 5 ms |
| `throughput` | The same day unpaced; gate ≥ 12M entries/s (2× the 6M line rate) |
| `drops` | A capture with every 997th packet missing: GPU and CPU recover identically |
| `pcap` | A generated full-size synthetic day at 200,000 packets/s, with lossless and real-time gates |
| `plan` | Compiles the `--check --risk gpu` graph; no GPU work, no I/O |

The NIC front ends (`gpudirect`, `daqiri-host`) are not `holoscan run` modes: they need the
DAQIRI image, root, hugepages and the ConnectX device, and `holoscan run` starts a
non-root container without DAQIRI. They run only through the Thor launchers in
[deploy/igx_thor](deploy/igx_thor/README.md), which use a private IPC namespace.

## Operators

| Operator | In → out | Trigger | Lane | What it does |
| --- | --- | --- | --- | --- |
| `GpuSbeSourceOp` | → `events` | `OnClock` 500 µs | ingest | DAQIRI GPUDirect into GPU memory; a native drain thread (never takes the Python lock) gathers payloads into a 1M-slot device ring; GPU SBE decode |
| `GpuSbeReplaySourceOp` | → `events` | `OnClock` 500 µs | ingest | The same ring, accounting, staging and decode, fed from pcap frames in GPU memory by a software NIC (DAQIRI burst cadence, optional NIC-buffer drop model) |
| `UdpPacketSourceOp` | → `packets` | `OnClock` 500 µs | ingest | Kernel UDP socket receiver with sequence accounting |
| `DaqiriPacketSourceOp` | → `packets` | `OnClock` 500 µs | ingest | DAQIRI into pinned host memory |
| `SbeDecodeOp` | `packets` → `events` | `OnEach` | ingest | Python reference SBE decode (host front ends) |
| `QuoteBarsOp` | `events` → `window` | `OnEach` | ingest | Event-time five-second quote bars, `backend="gpu"` (CUDA kernels) or `"cpu"` (NumPy), bit-identical |
| `RollingRiskOp` | `window` → `risk`, `validate` | `OnClock` 250 µs + `DrainAll` | analytics | Rolling volatility, covariance and correlation on the newest window at a bounded rate; GPU or CPU, inline or in a spawned process |
| `RiskCrossCheckOp` | `risk` → `comparison` | `OnClock` 5 ms + `DrainAll` | crosscheck | Recomputes validated GPU windows on the CPU and checks agreement |
| `CallbackSinkOp` | `items` → | `OnEach` | producer's | Hands every item to a callable (results store, checkers, tests) |
| `RiskSummaryOp` | `risk` → `summary` | `OnEach` | topics | Compact JSON risk summary for a holomq topic (`--topics` only) |

Each has a catalog entry with its ports, parameters, contract, recommended edges and
threads under [operators/](operators/README.md). They are thin adapters around library
classes in `holoscan_finance` that hold the logic and are tested without Holoscan.

### Composing an application

The examples import the canonical package `holoscan_finance`; an application outside this
checkout uses the alias `holoscan.finance`, as in this simplified version of the
`equity_analytics` replay chain:

```python
import holoscan as hs
from holoscan.finance import Knob, Ref
from holoscan.finance.operators import CallbackSinkOp, GpuSbeReplaySourceOp, QuoteBarsOp, RollingRiskOp
from holoscan.finance.operators.edges import LATEST, lossless

g = hs.Graph("replay_chain", default_clock=hs.RealtimeClock)
directory, results = Ref(Knob()), []  # live objects travel as Ref parameters
source = g.op(GpuSbeReplaySourceOp, "source", pcap_path="day.pcap", pps=200_000, directory=directory)
bars = g.op(QuoteBarsOp, "bars", backend="gpu", directory=directory)
risk = g.op(RollingRiskOp, "risk", backend="gpu", executor="inline", min_interval_us=50_000)
publish = g.op(CallbackSinkOp, "publish", callback=Ref(results.append))
g.add_flow(source.events, bars.events, **lossless(4))  # the app keeps these in one lane
g.add_flow(bars.window, risk.window, **LATEST)  # bars never wait for analytics
g.add_flow(risk.risk, publish.items, **lossless(8))  # every declared output is wired
```

Until the directory knob holds the feed's `FeedDirectory` (the manifest), every packet
counts as stale; the application sets it from the tape's manifest or `POST /manifest`.

The alias needs the `holoscan.modules` entry point: the wheel, the CMake build tree that
`holoscan build|run|test` stages, or `python -m holoscan_finance.devpath DIR` for a plain
checkout. `PYTHONPATH=python` alone gives `holoscan_finance`.

## Tools

CPU-only unless noted; none of them imports the Holoscan SDK except `topic_tail`:

```bash
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$PWD/python"
python3 -m holoscan_finance.tools.synth --out data/synth --symbols 1000 --packets 400000 --seconds 23400 --seed 7 --reuse   # synthetic tape + pcap
python3 -m holoscan_finance.tools.replay_sender --data data/synth/tape --port 26400 --control-url http://127.0.0.1:8080   # starts once /healthz is 200
python3 -m holoscan_finance.tools.bench_replay inspect data/synth/day.pcap    # and: register --port N
python3 -m holoscan_finance.tools.measure --seconds 120 --rate 75000 --output cpu-only.json
python3 -m holoscan_finance.tools.compare_runs --cpu cpu-only.json --gpu gpu-only.json --all-shared --output separate.json
python3 -m holoscan_finance.tools.gpu_validate --pcap data/synth/day.pcap --tape data/synth/tape   # GPU; add --bars for quote bars
python3 -m holoscan_finance.tools.decode_bench --pcap data/synth/day.pcap --tape data/synth/tape            # GPU; numba >= 0.68 for the CPU decoder
python3 -m holoscan_finance.tools.risk_bench --sizes 50,2886,10000 --check --out risk.json     # GPU and CPU analytics per window
python3 -m holoscan_finance.tools.risk_bench --kinds 300 --backends gpu,gpu-host,cpu,compare --returns data/synth/tape --check   # every window kind, back to back and at 270 ms
python3 -m holoscan_finance.tools.topic_tail --catalog /tmp/holoscan-finance/endpoints.json     # holomq subscriber (SDK)
```

They run from a checkout with its `python/` directory added to `PYTHONPATH` (Python 3.12+,
NumPy). Preserve the SDK environment's existing entries. The measurement commands need a
running receiver and sender; see the [separate CPU and GPU
runs](docs/VALIDATION.md#separate-cpu-and-gpu-runs) for the full synthetic input and sequence.

## Synthetic UDP replay

Generate a tape and pcap, then start the receiver and the paced sender in separate terminals:

```bash
PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$PWD/python" python3 -m holoscan_finance.tools.synth \
    --out data/synth --symbols 1000 --packets 400000 --seconds 23400 --seed 7 --reuse
holoscan run equity_analytics udp                     # receiver and dashboard on port 8080
PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$PWD/python" python3 -m holoscan_finance.tools.replay_sender \
    --data data/synth/tape --port 26400 --control-url http://127.0.0.1:8080
```

Open <http://localhost:8080/>: scrub the timeline, pause, and set the replay speed or a
fixed observation rate (these drive the external sender; with `--ingest replay` the pace is
`--replay-pps` and the page disables them). A seek, loop or sender restart resets the
rolling history; the first estimate needs 20 return bars and the window grows to 60 (five
minutes).
DAQIRI/ConnectX, GPUDirect and the full-day button run on Thor through the launchers
in [deploy/igx_thor](deploy/igx_thor/README.md).

## Synthetic validation

[docs/VALIDATION.md](docs/VALIDATION.md) defines the reproducible checks and the reports to
retain. These are validation targets; results depend on the input, hardware and run
conditions.

| Check | Command | Pass gate |
| --- | --- | --- |
| Decode and quote bars | [`gpu_validate`](docs/VALIDATION.md#library-bit-identity) | Decode fields and bar state identical to the host reference; returns within 1 ULP |
| In-graph GPU/CPU bars | `holoscan run market_data_bench check` | Every compared window and the independent host decode agree |
| Loss recovery | `holoscan run market_data_bench drops` | GPU and CPU recover identically from the generated sequence gaps |
| Paced full-size replay | `holoscan run market_data_bench paced` | Lossless at 200,000 packets/s, real time, replay lag p99 ≤ 5 ms |
| Throughput headroom | `holoscan run market_data_bench throughput` | At least 12M SBE entries/s |
| Complete analytics graph | `holoscan run equity_analytics smoke` | At least one analyzed window; every CPU/GPU comparison agrees |
| Analytics and latency | [`risk_bench`, `measure`, `compare_runs`](docs/VALIDATION.md#analytics-per-window) | Correctness and matched-work checks pass; report timing separately |

### Host-memory path vs GPUDirect path

| Step | `daqiri-host` | `gpudirect` |
| --- | --- | --- |
| NIC delivers packets to | pinned CPU memory | GPU memory (dma-buf; no `nvidia_peermem`) |
| Packet handling | Python copies each packet into a ring | CUDA kernel gathers payloads into slots |
| SBE decode | Python, one packet at a time (NumPy over its entries) | CUDA kernel, one warp per packet, one lane per entry |
| Five-second quote bars | NumPy on the CPU | CUDA kernels |
| Rolling covariance / correlation | CUDA kernels (CuPy) in a separate process, sent each window's new bars | CUDA kernels (CuPy) in-process on the device-resident window (`RollingRiskOp`) |
| Market data on the CPU | every packet, observation and bar | none (control metadata and dashboard results only) |

The host path decodes each message in interpreted Python. To distinguish interpreter
cost from the GPU's contribution, `decode_bench` also measures a compiled CPU decoder
(numba), both alone and with the NumPy quote bars. Compare the same synthetic capture at
multiple batch sizes and CPU thread counts; see
[Is it the GPU, or is it removing Python?](docs/GPU_INGEST.md#is-it-the-gpu-or-is-it-removing-python).

Holoscan's contribution is structure: receiver, bars and analytics are operators on one
graph, which is why the front end is a single flag with no change downstream. The GPU
path is three operators:

```text
GpuSbeSourceOp ──events──▶ QuoteBarsOp(gpu) ──window──▶ RollingRiskOp(gpu)
NIC → GPU memory,             5 s quote bars              volatility, covariance and
GPU gather + SBE decode       on the GPU                  correlation on the GPU (CUDA)
```

### What runs where

| Stage | Library | Device |
| --- | --- | --- |
| Generate synthetic observations, tape and pcap | NumPy, project codec; `--encoder gpu` packs and frames the capture with CUDA kernels | CPU (GPU encode optional), before playback |
| Replay sender (SBE over UDP) | NumPy memory map, project codec | CPU |
| Receive | `GpuSbeSourceOp` (DAQIRI GPUDirect) or `GpuSbeReplaySourceOp` (pcap in GPU memory), or the host baselines `UdpPacketSourceOp` and `DaqiriPacketSourceOp` | NIC → GPU memory, or CPU |
| SBE decode, quote bars | CUDA kernels on the GPU front ends; `SbeDecodeOp` (Python) and `QuoteBarsOp(cpu)` (NumPy) on the host front ends | GPU or CPU |
| Volatility, covariance, correlation | CuPy `RawModule` kernels: a fused rank update and device reductions, one CUDA graph launch per window of any kind (slides of any length, growth after a seek, rebuilds) | GPU |
| CPU comparator | The same algorithm in NumPy + SciPy BLAS (`dgemm`, `gemv`), threadpoolctl | CPU, separate process |
| Dashboard and controls | FastAPI / Uvicorn, JavaScript | CPU / browser |

The custom CUDA kernels are in
[gpu_sbe.py](python/holoscan_finance/marketdata/gpu_sbe.py),
[gpu_bars.py](python/holoscan_finance/marketdata/gpu_bars.py) and, for the analytics,
[_risk_kernels.py](python/holoscan_finance/marketdata/_risk_kernels.py) (driven by
[covariance.py](python/holoscan_finance/marketdata/covariance.py)). The graph is
[equity_analytics.py](examples/equity_analytics/equity_analytics.py).

### Synthetic inputs

`tools.synth` writes a memory-mapped tape, `day.pcap` and `synth.json`, which records the
seed, arguments and expected capture counts. The full-size modes use 2,886 synthetic
symbols, 2,880,000 data packets and a 23,400-second session; a final marker packet closes
the last bar. Use the generated summary for exact message and observation counts.

## Layout

| Path | Contents |
| --- | --- |
| `python/holoscan_finance/` | The package: `operators/` (the ten operators and edge presets), the libraries they wrap (`marketdata`, `equity`, `ingest`, `analytics`, `telemetry`, `control`, `validation`), the dashboard, the shared app plumbing (`app`) and the tools |
| `operators/<name>_op/` | Catalog entries: `metadata.json` and a README per operator ([index](operators/README.md)) |
| `examples/<name>/` | The two example applications: the composition, `metadata.json` (the `holoscan run` modes) and a README |
| `metadata.json`, `pyproject.toml`, `CMakeLists.txt`, `Dockerfile` | Module metadata for holoscan-cli (the examples are its `subprojects.applications`, the key the module schema uses), the wheel and the CLI search paths, the build-tree staging (nothing is compiled) and the dev image |
| `tests/python/` | pytest tiers: `library`, `control`, `gpu`, `operators`, `graphs`, `app`, `e2e`, `packaging`, `deploy` |
| `deploy/igx_thor/`, `docs/` | Thor launchers, the DAQIRI image and benchmarks; architecture, GPU ingest, methodology and validation docs |

## Develop and test

```bash
holoscan test                        # CTest in the dev container: python.unit and python.gpu
holoscan lint                        # pre-commit: ruff, codespell, markdownlint, metadata validation
holoscan package --pkg-generator WHEEL   # build/dist/holoscan_finance-0.2.0-py3-none-any.whl
```

The tests are tiered by pytest markers: `sdk` needs Holoscan 5, `gpu` CuPy and a CUDA
device, and `daqiri` the DAQIRI module; a test whose tier is unavailable skips. `slow`
marks tests over 30 s, and `timing` (always also `slow`) the latency and throughput bounds,
which need an otherwise idle host. CTest deselects `slow`; add `-m "not timing"` to any
other run that shares the host.

```bash
pytest -m "not sdk and not gpu and not daqiri"                        # any Python 3.12+ (requirements-dev.txt)
scripts/with_stub_libcuda.sh python3 -m pytest -m "sdk and not gpu and not slow"   # 5.0 image, no GPU driver
python3 -m pytest -m "not slow" tests/python                          # 5.0 image with a GPU
```

On the IGX Thor development host, `scripts/dev5.sh '<command>'` and `scripts/dev5cpu.sh`
run a command in the 5.0 image non-root, with a private IPC namespace and off the DAQIRI
cores.

The wheel installs `holoscan_finance` and the `holoscan.modules` entry point
(`python3 -m pip install build/dist/*.whl`). The Holoscan SDK itself is never a pip
dependency: Holoscan SDK 5.0 is not published on PyPI, so use the 5.0 image or the SDK's
Debian packages. Extras:
`gpu` (CuPy), `analytics` (SciPy, threadpoolctl), `dashboard` (FastAPI, Uvicorn),
`bench` (numba), `test`.

## Limits

The examples use synthetic replay with live computation. A full window has 60 returns,
so the covariance matrix has rank at most 59; correlations are descriptive and noisy, and
annualized volatility is not a forecast. The generated quote fields do not represent NBBO.
Uniformly paced synthetic captures exercise the pipeline and its validation gates; they
do not establish performance on an external, bursty feed.

[Apache-2.0 license](LICENSE)
