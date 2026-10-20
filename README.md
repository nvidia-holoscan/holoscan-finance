# Streaming NYQ equity analytics with Holoscan

Replays one recorded trading day of LSEG NYQ trades and quotes as MITCH over UDP and
continuously computes five-second quote returns, historical volatility and
cross-instrument correlation. NVIDIA Holoscan runs the receiver, quote bars and
analytics as one graph; CuPy/cuBLAS computes the rolling statistics on the GPU, and a
NumPy/SciPy worker computes the same windows on the CPU for comparison. With
`--ingest gpu`, packets go from the NIC straight into GPU memory and CUDA kernels
decode them and build the quote bars.

**Data: historical LSEG/Refinitiv equity data — NYQ dataset** (April 1, 2025;
licensed, not in Git). See [methodology](docs/METHODOLOGY.md) for the data, wire
format and calculation conventions.

## Results

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

## Host-memory path vs GPUDirect path

| Step | `--ingest ano` (host memory) | `--ingest gpu` (GPUDirect) |
| --- | --- | --- |
| NIC delivers packets to | pinned CPU memory | GPU memory (dma-buf; no `nvidia_peermem`) |
| Packet handling | Python copies each packet into a ring | CUDA kernel gathers payloads into slots |
| MITCH decode | Python, one message at a time | CUDA kernel, one thread per packet |
| Five-second quote bars | NumPy on the CPU | CUDA kernels |
| Rolling covariance / correlation | CuPy + cuBLAS in a separate process, fed host arrays | CuPy + cuBLAS in-process on the device-resident window (`GpuRollingRiskOp`) |
| Market data on the CPU | every packet, observation and bar | none (control metadata and dashboard results only) |
| Highest rate kept up in real time | 4,500 packets/s (149k msgs/s) | 200,000 packets/s (6.6M msgs/s) |

**Where the gain comes from — mostly not the GPU.** The host path is limited by
interpreted Python decoding each message. On the same packets a compiled CPU decoder
(numba, bit-identical) is ~100× faster than Python on one core (27M vs 0.28M
messages/s) and, at 16k-message batches, faster than the GPU decoder (43.5M). The
GPU's advantage appears at scale: with large batches GPU decode reaches 303M
messages/s (compiled CPU, 8 cores: 104M) and decode + bars 76M (CPU with NumPy bars:
5.3M). The live 6.6M messages/s is above what compiled CPU decode plus NumPy bars
reach even offline; a CPU path with compiled multi-core bars was not built. See
[Is it the GPU, or is it removing Python?](docs/GPU_INGEST.md#is-it-the-gpu-or-is-it-removing-python)

Holoscan's contribution is structure: receiver, bars and analytics are operators on
one graph run concurrently, which is why the ingest mode is a single flag with no
change downstream. The GPU path ships as three reusable operators
([overview](operators/README.md)):

```text
GpuMitchSourceOp → GpuQuoteBarsOp → GpuRollingRiskOp
NIC → GPU memory,   5 s quote bars,   volatility, covariance and
MITCH decode        on the GPU        correlation on the GPU (cuBLAS)
```

The host-memory receivers, with their own bars and analytics, stay in the application
as the baseline.

## What runs where

| Stage | Library | Device |
| --- | --- | --- |
| Prepare the day (Parquet → sorted replay tape) | DuckDB, PyArrow | CPU, before playback |
| Replay sender (MITCH over UDP) | NumPy memory map, project codec | CPU |
| Receive | `GpuMitchSourceOp` (DAQIRI GPUDirect), or the host-memory baselines (kernel UDP, DAQIRI into pinned memory) | NIC → GPU memory, or CPU |
| MITCH decode, quote bars | Python + NumPy, or CUDA kernels with `--ingest gpu` | CPU or GPU |
| Volatility, covariance, correlation | CuPy + cuBLAS (`gemm`) | GPU |
| CPU comparator | NumPy + SciPy BLAS (`dgemm`), threadpoolctl | CPU, separate process |
| Dashboard and controls | FastAPI / Uvicorn, JavaScript | CPU / browser |

The custom CUDA kernels are in [gpu_mitch.py](python/holoscan/marketdata/gpu_mitch.py)
and [gpu_bars.py](python/holoscan/marketdata/gpu_bars.py); the analytics use CuPy and
cuBLAS only ([covariance.py](python/holoscan/marketdata/covariance.py)). The graph is
[stream.py](python/holoscan/equities/stream.py).

## Data

| Measure | Amount |
| --- | --- |
| Source | 24 NYQ Parquet files; 95,746,717 records; 3,350,638,952 bytes |
| Instrument directory | 2,886 RICs (shares, units, warrants, preferred) |
| Regular-session replay | 92,178,038 observations, 09:30–16:00 America/New_York |
| Replay tape | 6,083,750,508 bytes, memory mapped on the sender |

## Run

Python 3.10+, Holoscan 4.3+ and a CuPy build for your CUDA runtime:

```bash
python3 -m venv .venv
.venv/bin/pip install holoscan cupy-cuda13x   # match your CUDA version
.venv/bin/pip install '.[equities]'

# Prepare the licensed day once
./scripts/runenv -m holoscan.equities.data \
    --data '/licensed/nyq/2025-04-01/*.parquet' --output data/nyq-equities-2025-04-01

# Receiver and dashboard, then the sender in a second terminal
./scripts/runenv -m holoscan.equities.stream --port 8080 --udp-port 26400
./scripts/runenv -m holoscan.equities.replay --data data/nyq-equities-2025-04-01 \
    --port 26400 --control-url http://127.0.0.1:8080
```

Open <http://localhost:8080/>: scrub the timeline, pause, and set the replay speed or a
fixed observation rate. A seek, loop or sender restart resets the rolling history; the
first estimate needs 20 return bars and the window grows to 60 (five minutes).
`scripts/runenv` uses `.venv`; set `PY=/usr/bin/python3` inside a prepared container.

DAQIRI/ConnectX, GPUDirect and the full-day button run on Thor through the launchers
in [deploy/igx_thor](deploy/igx_thor/README.md).

## Layout and build

The repository is a HoloHub module:

| Path | Contents |
| --- | --- |
| `operators/<name>/` | The shipped operators, `GpuMitchSourceOp`, `GpuQuoteBarsOp` and `GpuRollingRiskOp`: module, `metadata.json`, `CMakeLists.txt`, README ([overview](operators/README.md)) |
| `applications/equity_analytics/` | The application: entry point and `metadata.json` |
| `python/holoscan/equities/` | Application package: graph and dashboard, MITCH-to-NYQ mapping, host-memory receivers and analytics workers, replay and measurement tools |
| `python/holoscan/marketdata/` | What the operators use: MITCH codec and GPU kernels, DAQIRI helpers, observation records, quote bars, covariance engine; the operators resolve into this package |
| `CMakeLists.txt`, `cmake/`, `pkg/`, `metadata.json`, `holohub` | HoloHub build, packaging and CLI |
| `deploy/igx_thor/`, `docs/` | Thor launchers and benchmarks; GPU ingest and methodology docs |

```bash
./holohub build equity_analytics            # or with CMake directly:
cmake -S . -B build -DBUILD_ALL=ON -DFINANCE_BUILD_TESTING=ON
cmake --build build && ctest --test-dir build
```

## Reproduce

```bash
./scripts/runenv -m pytest                       # GPU tests skip without a GPU
./scripts/runenv -m holoscan.equities.gpu_validate --pcap capture.pcap --tape <tape dir>
./scripts/runenv -m holoscan.equities.gpu_validate --bars --pcap capture.pcap --tape <tape dir>
./scripts/runenv -m holoscan.equities.decode_bench --help
```

Ingest rates: [gpu_ingest_bench](deploy/igx_thor/gpu_ingest_bench/README.md).
Separate CPU/GPU analytics runs: [methodology](docs/METHODOLOGY.md#separate-cpu-only-and-gpu-only-benchmark).
The dashboard's default `compare` mode runs both analytics workers at once for live
validation; its timing ratio is not the headline benchmark.

## Limits

This is historical replay with live computation, not a production risk model. A full
window has 60 returns, so the covariance matrix has rank at most 59; correlations are
descriptive and noisy, and annualized volatility is not a forecast. The demo uses the
supplied NYQ quote fields and does not reconstruct NBBO. Throughput is measured on
one host in PHY loopback with uniform `tcpreplay` traffic, not an external feed.

[Apache-2.0 license](LICENSE)
