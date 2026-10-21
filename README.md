# Streaming NYQ equity analytics with Holoscan SDK 5

Replays one recorded trading day of LSEG NYQ trades and quotes as MITCH over UDP and
continuously computes five-second quote returns, historical volatility and
cross-instrument correlation. A Holoscan SDK 5 graph runs the receiver, MITCH decode,
quote bars and analytics; CuPy/cuBLAS computes the rolling statistics on the GPU, and a
NumPy/SciPy worker computes the same windows on the CPU for comparison. With GPUDirect
ingest, packets go from the NIC straight into GPU memory and CUDA kernels decode them and
build the quote bars.

The repository is the Holoscan module `holoscan-finance`: ten reusable, pure-Python
operators in the package `holoscan_finance` (published by Holoscan as
`holoscan.finance`), and two applications composed from them. It runs on the
`nvcr.io/nvidia/clara-holoscan/holoscan:v5.0.0-cuda13` image with `holoscan-cli` 5.0.0.

**Data: historical LSEG/Refinitiv equity data — NYQ dataset** (April 1, 2025;
licensed, not in Git). See [methodology](docs/METHODOLOGY.md) for the data, wire
format and calculation conventions. Without it, the applications run on a deterministic
synthetic day (`python -m holoscan_finance.tools.synth`), which their `holoscan run` modes
generate.

## What the applications demonstrate

| Application | What it is | Front ends |
| --- | --- | --- |
| [`equity_analytics`](applications/equity_analytics/README.md) | The live dashboard: MITCH feed → quote bars → rolling risk → web UI, with an optional live CPU cross-check of every validated GPU window (`--analytics-backend compare`) | `udp` (kernel socket), `daqiri-host` (DAQIRI into pinned host memory), `gpudirect` (DAQIRI GPUDirect into GPU memory), `replay` (pcap frames staged in GPU memory: the GPUDirect chain without a NIC) |
| [`market_data_bench`](applications/market_data_bench/README.md) | Headless, finite and verified: replays a pcap from GPU memory through gather → decode → quote bars (→ risk) and reports throughput, loss, lag and placement against pass/fail gates. With `--check`, the same decoded stream also feeds a CPU `QuoteBarsOp`, and every GPU window is compared with its CPU twin in the graph | `replay` |

Both are plain compositions: an application file builds the graph from
`holoscan_finance.operators`, states the plan rules it promises, and calls shared plumbing.
It defines no operator classes. The Holoscan 5 features they show, none of them on the
per-packet path:

| Feature | `equity_analytics` | `market_data_bench` |
| --- | --- | --- |
| Declarative operators, typed `Param`s validated at `graph.op`, live objects injected as `Ref`s | yes | yes |
| `Param`-gated optional ports (`emit_windows`, `cross_check`) | yes | yes |
| `OnClock` sources with hold-and-retry on `BLOCKING` edges | yes | yes, including fan-out |
| A lossless same-lane `OnEach` chain: no thread hop per packet | yes | yes |
| A `DROP_OLDEST`/1 latest-only mailbox into an `OnClock(1 ms)` + `DrainAll` consumer | yes | `--risk gpu` |
| Authored lanes with `ThreadPlacement` (CPU affinity, optional `BUSY_POLL`), off the DAQIRI cores | yes | yes |
| Downstream-first `start()` warm-up: kernels are compiled before the NIC starts | yes | yes |
| `run_async` under a FastAPI lifespan, with run diagnostics in `/snapshot` | yes | - |
| Plan rules checked on `plan.json()`; `--plan-only`, `--plan-json`, `GET /plan` | yes | yes |
| One stream fanned out to two backends of the same operator (in-graph bit identity) | - | `--check` |
| A deterministic finite run: a drained predicate, then `request_stop()` | `--exit-when-drained` | yes |
| `max_start_delay` + `EMIT_WITNESS` lateness counter | `--risk-deadline-us` | - |
| holomq `export_topic` of a compact risk summary, latest-only | `--topics` | - |
| `holoscan run <app> <mode>` and the `holoscan.finance` module alias | yes | yes |

How the graphs are laid out, and why: [architecture](docs/ARCHITECTURE.md). What changed
from the Holoscan 4.3 version: [porting notes](docs/PORTING_TO_5_0.md).

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
packets/s, in a loop, through the production gather, ring, MITCH decode and quote-bar
code, with live CPU validation. Select a GPU with
`CUDA_VISIBLE_DEVICES=<n> CUDA_DEVICE_ORDER=PCI_BUS_ID`, which the CLI forwards. Inside the
dev container (`holoscan run-container`), or anywhere the 5.0 SDK is installed, add `--local`.

| `holoscan run equity_analytics <mode>` | What runs |
| --- | --- |
| `replay` (default) | Synthetic day, replay front end, `compare` analytics, dashboard on port 8080 |
| `replay_pcap` | The recorded day (`data/nyq/day.pcap`, `data/nyq/tape`) at 200,000 packets/s |
| `udp`, `udp_cpu` | Kernel UDP front end (`compare` or CPU analytics); start the sender separately (below) |
| `smoke` | Headless CI run: exits non-zero if the run fails, no window is analyzed or a comparison disagrees |
| `plan` | Compiles the graph and prints the plan summary; no GPU work, no I/O |

| `holoscan run market_data_bench <mode>` | What runs |
| --- | --- |
| `smoke` (default) | 50,000 synthetic packets, unpaced, `--check --require-lossless` |
| `check` | 1M packets, in-graph GPU vs CPU bars and an independent host decode of every closed bar |
| `paced` | A full-size synthetic day at 200,000 packets/s with the NIC-buffer drop model and GPU risk: lossless, real time, lag p99 ≤ 5 ms |
| `throughput` | The same day unpaced; gate ≥ 13.2M messages/s (2× the 6.6M line rate) |
| `drops` | A capture with every 997th packet missing: GPU and CPU recover identically |
| `pcap` | The recorded day at 200,000 packets/s: lossless and real time |
| `plan` | Compiles the `--check --risk gpu` graph; no GPU work, no I/O |

The NIC front ends (`gpudirect`, `daqiri-host`) are not `holoscan run` modes: they need the
DAQIRI image, root, hugepages and the ConnectX device, and `holoscan run` starts a
non-root container without DAQIRI. They run only through the Thor launchers in
[deploy/igx_thor](deploy/igx_thor/README.md), which use a private IPC namespace.

## Operators

| Operator | In → out | Trigger | Lane | What it does |
| --- | --- | --- | --- | --- |
| `GpuMitchSourceOp` | → `events` | `OnClock` 500 µs | ingest | DAQIRI GPUDirect into GPU memory; a native drain thread (never takes the Python lock) gathers payloads into a 1M-slot device ring; GPU MITCH decode |
| `GpuMitchReplaySourceOp` | → `events` | `OnClock` 500 µs | ingest | The same ring, accounting, staging and decode, fed from pcap frames in GPU memory by a software NIC (DAQIRI burst cadence, optional NIC-buffer drop model) |
| `UdpPacketSourceOp` | → `packets` | `OnClock` 500 µs | ingest | Kernel UDP socket receiver with sequence accounting |
| `DaqiriPacketSourceOp` | → `packets` | `OnClock` 500 µs | ingest | DAQIRI into pinned host memory |
| `MitchDecodeOp` | `packets` → `events` | `OnEach` | ingest | Python reference MITCH decode (host front ends) |
| `QuoteBarsOp` | `events` → `window` | `OnEach` | ingest | Event-time five-second quote bars, `backend="gpu"` (CUDA kernels) or `"cpu"` (NumPy), bit-identical |
| `RollingRiskOp` | `window` → `risk`, `validate` | `OnClock` 1 ms + `DrainAll` | analytics | Rolling volatility, covariance and correlation on the newest window at a bounded rate; GPU or CPU, inline or in a spawned process |
| `RiskCrossCheckOp` | `risk` → `comparison` | `OnClock` 5 ms + `DrainAll` | crosscheck | Recomputes validated GPU windows on the CPU and checks agreement |
| `CallbackSinkOp` | `items` → | `OnEach` | producer's | Hands every item to a callable (results store, checkers, tests) |
| `RiskSummaryOp` | `risk` → `summary` | `OnEach` | topics | Compact JSON risk summary for a holomq topic (`--topics` only) |

Each has a catalog entry with its ports, parameters, contract, recommended edges and
threads under [operators/](operators/README.md). They are thin adapters around library
classes in `holoscan_finance` that hold the logic and are tested without Holoscan.
Applications import the canonical package; elsewhere use the alias, as in this
simplified version of the `equity_analytics` replay chain:

```python
import holoscan as hs
from holoscan.finance import Knob, Ref
from holoscan.finance.operators import CallbackSinkOp, GpuMitchReplaySourceOp, QuoteBarsOp, RollingRiskOp
from holoscan.finance.operators.edges import LATEST, lossless

g = hs.Graph("replay_chain", default_clock=hs.RealtimeClock)
directory, results = Ref(Knob()), []  # live objects travel as Ref parameters
source = g.op(GpuMitchReplaySourceOp, "source", pcap_path="day.pcap", pps=200_000, directory=directory)
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
python3 -m holoscan_finance.tools.synth --out data/synth --symbols 500 --packets 20000   # synthetic tape + pcap
python3 -m holoscan_finance.tools.prepare_tape --data '/licensed/nyq/2025-04-01/*.parquet' --output data/nyq-equities-2025-04-01
python3 -m holoscan_finance.tools.replay_sender --data data/nyq-equities-2025-04-01 --port 26400 --control-url http://127.0.0.1:8080
python3 -m holoscan_finance.tools.bench_replay inspect capture.pcap          # and: register --port N
python3 -m holoscan_finance.tools.measure --seconds 120 --rate 75000 --output cpu-only.json
python3 -m holoscan_finance.tools.compare_runs --cpu cpu-only.json --gpu gpu-only.json --all-shared --output separate.json
python3 -m holoscan_finance.tools.gpu_validate --pcap capture.pcap --tape <tape dir> [--bars]   # GPU
python3 -m holoscan_finance.tools.decode_bench --pcap capture.pcap --tape <tape dir>            # GPU; numba >= 0.68 for the CPU decoder
python3 -m holoscan_finance.tools.topic_tail --catalog /tmp/holoscan-finance/endpoints.json     # holomq subscriber (SDK)
```

They run from a checkout with `PYTHONPATH=python` (Python 3.12+, NumPy; `prepare_tape` adds
DuckDB and PyArrow).

## The licensed day

Prepare it once, then run the receiver and the paced sender:

```bash
PYTHONPATH=python python3 -m holoscan_finance.tools.prepare_tape \
    --data '/licensed/nyq/2025-04-01/*.parquet' --output data/nyq-equities-2025-04-01
holoscan run equity_analytics udp                     # receiver and dashboard on port 8080
PYTHONPATH=python python3 -m holoscan_finance.tools.replay_sender \
    --data data/nyq-equities-2025-04-01 --port 26400 --control-url http://127.0.0.1:8080
```

Open <http://localhost:8080/>: scrub the timeline, pause, and set the replay speed or a
fixed observation rate. A seek, loop or sender restart resets the rolling history; the
first estimate needs 20 return bars and the window grows to 60 (five minutes).
DAQIRI/ConnectX, GPUDirect and the full-day button run on Thor through the launchers
in [deploy/igx_thor](deploy/igx_thor/README.md).

## Results (Holoscan 4.3)

The numbers in this section were measured with the Holoscan SDK 4.3 version of this code
(the first commit of this branch and earlier). The 5.0 port keeps the kernels and the native drain byte
for byte, and the GPU path's batch, burst, buffer and ring sizes; its NIC-free results are in
[docs/VALIDATION.md](docs/VALIDATION.md), and the NIC series has not been repeated on 5.0
yet ([status](docs/GPU_INGEST.md#holoscan-50-revalidation)).

Measured on IGX Thor (14 Arm64 cores), RTX PRO 6000 Blackwell Max-Q and ConnectX-7 in
PHY local loopback, with DAQIRI/DPDK receive.

**GPUDirect ingest (`--ingest gpu`): 200,000 packets/s — 6.6M MITCH messages/s —
received, decoded, barred and analyzed in real time, with zero loss in 242 of 250
full-day runs and market data never leaving GPU memory.** The whole recorded day
(92,178,039 observations, 2,880,565 packets) arrives in 14.4 s, about 1,620× real time.
Five lossy runs dropped 2–4% of packets during rare ~1 s receiver stalls; one rejected
35 packets the replay tool sent with a stale source port; in two, every packet arrived
but 26–41 arrived out of order and were discarded. Those stalls were Python threads
holding the interpreter lock, which stopped the receive drain. The drain now runs in a
native thread that never takes the lock: with injected lock holds of 1.1 s and 3 s, 11
of 12 runs delivered every packet, where the Python drain lost ~3.8% in 3 of 3
([details](docs/GPU_INGEST.md#native-drain)). The
host-memory path (`--ingest ano`) keeps up in real time only to 4,500 packets/s (~149k
messages/s), about 44× lower; at 6,000–7,500 packets/s it loses nothing within 15 s
but falls 4–7 s behind, and it starts dropping near 10,000. GPU decode and bars are bit-identical to the CPU
implementation over the full day (returns within 1 ULP). Method and caveats:
[GPUDirect ingest](docs/GPU_INGEST.md).

**Analytics: 9.55× by median calculation time** in separate CPU-only and GPU-only
streaming runs at 75,000 observations/s (host-memory ingest), over all 362 shared
full-window updates. It covers the rolling analytics block, not the whole pipeline.

| Same 362 windows | p50 | p99 |
| --- | --- | --- |
| Packet arrival → published GPU result (GPU-only run) | 34.22 ms | 49.83 ms |
| Packet arrival → published CPU result (CPU-only run) | 66.91 ms | 100.30 ms |
| GPU analytics calculation | 3.89 ms | 6.44 ms |
| CPU analytics calculation (8 BLAS threads) | 37.13 ms | 65.30 ms |

Each run captured 120 s at ~75,002 observations/s (CPU run 9,000,192 observations,
GPU run 9,000,256) with no sequence gaps, ring drops or malformed packets. The windows
are every update in 14:05:40–14:35:50 UTC; input hashes match and numerical agreement
passes. CPU BLAS threads were chosen from 1, 2, 4, 8 and 14.

### Host-memory path vs GPUDirect path

| Step | `daqiri-host` (4.3 `--ingest ano`) | `gpudirect` (4.3 `--ingest gpu`) |
| --- | --- | --- |
| NIC delivers packets to | pinned CPU memory | GPU memory (dma-buf; no `nvidia_peermem`) |
| Packet handling | Python copies each packet into a ring | CUDA kernel gathers payloads into slots |
| MITCH decode | Python, one message at a time | CUDA kernel, one thread per packet |
| Five-second quote bars | NumPy on the CPU | CUDA kernels |
| Rolling covariance / correlation | CuPy + cuBLAS in a separate process, fed host arrays | CuPy + cuBLAS in-process on the device-resident window (`RollingRiskOp`) |
| Market data on the CPU | every packet, observation and bar | none (control metadata and dashboard results only) |
| Highest rate kept up in real time (4.3) | 4,500 packets/s (149k msgs/s) | 200,000 packets/s (6.6M msgs/s) |

**Where the gain comes from — mostly not the GPU.** The host path is limited by
interpreted Python decoding each message. On the same packets a compiled CPU decoder
(numba, bit-identical) is ~100× faster than Python on one core (27M vs 0.28M
messages/s) and, at 16k-message batches, faster than the GPU decoder (43.5M). The
GPU's advantage appears at scale: with large batches GPU decode reaches 303M
messages/s (compiled CPU, 8 cores: 104M) and decode + bars 76M (CPU with NumPy bars:
5.3M). The live 6.6M messages/s is above what compiled CPU decode plus NumPy bars
reach even offline; a CPU path with compiled multi-core bars was not built. See
[Is it the GPU, or is it removing Python?](docs/GPU_INGEST.md#is-it-the-gpu-or-is-it-removing-python)

Holoscan's contribution is structure: receiver, bars and analytics are operators on one
graph, which is why the front end is a single flag with no change downstream. The GPU
path is three operators:

```text
GpuMitchSourceOp ──events──▶ QuoteBarsOp(gpu) ──window──▶ RollingRiskOp(gpu)
NIC → GPU memory,             5 s quote bars              volatility, covariance and
GPU gather + MITCH decode     on the GPU                  correlation on the GPU (cuBLAS)
```

### What runs where

| Stage | Library | Device |
| --- | --- | --- |
| Prepare the day (Parquet → sorted replay tape) | DuckDB, PyArrow | CPU, before playback |
| Replay sender (MITCH over UDP) | NumPy memory map, project codec | CPU |
| Receive | `GpuMitchSourceOp` (DAQIRI GPUDirect) or `GpuMitchReplaySourceOp` (pcap in GPU memory), or the host baselines `UdpPacketSourceOp` and `DaqiriPacketSourceOp` | NIC → GPU memory, or CPU |
| MITCH decode, quote bars | CUDA kernels on the GPU front ends; `MitchDecodeOp` (Python) and `QuoteBarsOp(cpu)` (NumPy) on the host front ends | GPU or CPU |
| Volatility, covariance, correlation | CuPy + cuBLAS (`gemm`) | GPU |
| CPU comparator | NumPy + SciPy BLAS (`dgemm`), threadpoolctl | CPU, separate process |
| Dashboard and controls | FastAPI / Uvicorn, JavaScript | CPU / browser |

The custom CUDA kernels are in
[gpu_mitch.py](python/holoscan_finance/marketdata/gpu_mitch.py) and
[gpu_bars.py](python/holoscan_finance/marketdata/gpu_bars.py); the analytics use CuPy and
cuBLAS only ([covariance.py](python/holoscan_finance/marketdata/covariance.py)). The graph
is [equity_analytics.py](applications/equity_analytics/equity_analytics.py).

### Data

| Measure | Amount |
| --- | --- |
| Source | 24 NYQ Parquet files; 95,746,717 records; 3,350,638,952 bytes |
| Instrument directory | 2,886 RICs (shares, units, warrants, preferred) |
| Regular-session replay | 92,178,038 observations, 09:30–16:00 America/New_York |
| Replay tape | 6,083,750,508 bytes, memory mapped on the sender |

## Layout

| Path | Contents |
| --- | --- |
| `python/holoscan_finance/` | The package: `operators/` (the ten operators and edge presets), the libraries they wrap (`marketdata`, `nyq`, `ingest`, `analytics`, `telemetry`, `control`, `validation`), the dashboard, the shared app plumbing (`app`) and the tools |
| `operators/<name>_op/` | Catalog entries: `metadata.json` and a README per operator ([index](operators/README.md)) |
| `applications/<app>/` | The two applications: the composition, `metadata.json` (the `holoscan run` modes) and a README |
| `metadata.json`, `pyproject.toml`, `CMakeLists.txt`, `Dockerfile` | Module metadata for holoscan-cli, the wheel, the build-tree staging (nothing is compiled) and the dev image |
| `tests/python/` | pytest tiers: `library`, `control`, `gpu`, `operators`, `graphs`, `app`, `e2e`, `packaging`, `deploy` |
| `deploy/igx_thor/`, `docs/` | Thor launchers, the DAQIRI image and benchmarks; architecture, porting, GPU ingest, methodology and validation docs |

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
dependency: PyPI stops at 4.6, so use the 5.0 image or the SDK's Debian packages. Extras:
`gpu` (CuPy), `analytics` (SciPy, threadpoolctl), `dashboard` (FastAPI, Uvicorn), `data`
(DuckDB, PyArrow), `bench` (numba), `test`.

## Limits

This is historical replay with live computation, not a production risk model. A full
window has 60 returns, so the covariance matrix has rank at most 59; correlations are
descriptive and noisy, and annualized volatility is not a forecast. The demo uses the
supplied NYQ quote fields and does not reconstruct NBBO. Throughput is measured on
one host in PHY loopback with uniform `tcpreplay` traffic, not an external feed.

[Apache-2.0 license](LICENSE)
