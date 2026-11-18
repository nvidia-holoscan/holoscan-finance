# Holoscan Module for Financial Services Industry

`holoscan-finance` provides reusable components and example applications for streaming
market data and GPU risk analytics. It receives and decodes equity trades and quotes,
builds quote bars (five seconds by default), and computes rolling volatility, covariance
and correlation.

The module uses [NVIDIA Holoscan](https://developer.nvidia.com/holoscan-sdk), a framework
for connecting processing steps, called *operators*, into a streaming pipeline. The
included applications assemble these operators for you; you can run them without writing
a Holoscan application.

```text
Market data → Decode → Quote bars → Rolling risk → Dashboard or benchmark report
```

## What you can run

| Application | Use it to |
| --- | --- |
| [Equity analytics](examples/equity_analytics/README.md) | Watch synthetic market data drive a live dashboard, with GPU analytics and a CPU cross-check. |
| [Market data benchmark](examples/market_data_bench/README.md) | Run a finite GPU replay and check correctness, packet loss, throughput and replay lag. |

Both can replay a packet capture from GPU memory without a network card. The dashboard
also supports UDP input and, on a configured IGX Thor system, DAQIRI/ConnectX input into
host or GPU memory. The [operator catalog](operators/README.md) describes the reusable
components for building your own application.

## Quick start

The default demo generates synthetic data and replays it through the GPU pipeline.
You need:

- A Linux host (`x86_64` or `aarch64`) with a supported NVIDIA GPU and a CUDA 13 compatible
  driver. Check the [Holoscan SDK requirements](https://docs.nvidia.com/holoscan/sdk-user-guide/setup/sdk-installation).
- Docker with the NVIDIA Container Toolkit configured, and permission to run containers.
- Python 3.12 or newer with `pip` and `venv` on the host, plus this repository checked out.

Run from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements-cli.txt
holoscan run equity_analytics replay
```

The CLI builds the project's [container image](Dockerfile), based on
`nvcr.io/nvidia/clara-holoscan/holoscan:v5.0.0-cuda13`, mounts the checkout and generates
input under `data/synth/`. The first run downloads a large SDK image and installs
dependencies. The repository pins Holoscan SDK 5.0.0 and `holoscan-cli` 5.0.0; the CLI
is the host-side launcher, and the SDK runs inside the container.

Open <http://localhost:8080/> on the host running the application. The demo loops at
2,000 packets/s and compares GPU analytics with CPU results. Estimates start after
20 return bars and grow to a 60-bar window (five minutes of market time). Stop with
`Ctrl+C`. Pause, seek and speed controls are available with an external UDP sender;
they are disabled for this GPU replay mode.

For a remote host, forward the dashboard port with
`ssh -N -L 8080:127.0.0.1:8080 <host>`, then open the same URL locally.
The dashboard binds to localhost by default and has no authentication; see
[SECURITY.md](SECURITY.md) before exposing it to a network.

To check the GPU pipeline without a dashboard:

```bash
holoscan run market_data_bench smoke
```

This generates 50,000 data packets, compares GPU quote bars with CPU references, checks
for packet loss and writes `build/market_data_bench/bench_smoke.json`. A nonzero exit
status means the run or a validation gate failed.

On a multi-GPU host, prefix a run with `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0`
to select a GPU. If you are already inside the project's container or an equivalent
SDK environment, add `--local` to `holoscan run`.

## Choose an input

| Input | Requirements and behavior |
| --- | --- |
| GPU replay | The quickest start: generated packet captures are staged in GPU memory. No NIC or external sender is needed. |
| UDP | A kernel socket receives packets; decode and quote bars run on the CPU. Choose GPU analytics with a CPU cross-check (`udp`) or CPU analytics (`udp_cpu`). Both modes still use the Holoscan SDK. |
| DAQIRI / GPUDirect | A configured ConnectX NIC receives into pinned host memory or directly into GPU memory. Follow the [IGX Thor deployment guide](deploy/igx_thor/README.md); these modes need its DAQIRI image and launchers. |

Inputs use the module's [FIX Simple Binary Encoding (SBE) schema](python/holoscan_finance/marketdata/sbe_schema.xml),
with one SBE message per UDP datagram. A replay capture needs its matching tape directory,
which supplies the symbol manifest. Arbitrary exchange feeds and packet captures need
adaptation to this format.

### Synthetic UDP replay

Use two terminals on the same host. First, generate the data and start the receiver
from the repository root with the virtual environment above activated:

```bash
python3 -m pip install "numpy>=1.26"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$PWD/python"
python3 -m holoscan_finance.tools.synth \
    --out data/synth --symbols 1000 --packets 400000 --seconds 23400 --seed 7 --reuse
holoscan run equity_analytics udp
```

In the second terminal, also from the repository root:

```bash
source .venv/bin/activate
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$PWD/python"
python3 -m holoscan_finance.tools.replay_sender \
    --data data/synth/tape --port 26400 --control-url http://127.0.0.1:8080
```

The sender waits for the receiver to be ready. Open <http://localhost:8080/> to pause,
seek or change the replay rate. A seek, loop or sender restart resets the rolling history.
Use `udp_cpu` in place of `udp` for the CPU analytics baseline.

## More examples and validation

List the available presets with:

```bash
holoscan modes equity_analytics
holoscan modes market_data_bench
```

The application guides explain each preset and its options. For larger correctness
checks, loss recovery and performance measurements, follow the
[validation guide](docs/VALIDATION.md). Its throughput and latency gates are validation
targets, not guaranteed performance on your hardware.

The [methodology](docs/METHODOLOGY.md) explains the synthetic data and calculations;
[GPU ingest](docs/GPU_INGEST.md) explains the receive paths and their measurement limits.
Data generation and UDP sending run on the CPU without Holoscan; GPU validation and
benchmark tools require their respective GPU and SDK dependencies.

## Use the operators in your application

From the repository root, install the module into your application's Holoscan SDK 5.0
environment:

```bash
python3 -m pip install -c requirements-dev.txt ".[gpu,analytics]"
```

Then import the operators through the module alias:

```python
from holoscan.finance.operators import QuoteBarsOp, RollingRiskOp
```

The [operator catalog](operators/README.md) describes their inputs and settings, and the
[architecture guide](docs/ARCHITECTURE.md) explains how the examples connect them.
Installing the module does not install the Holoscan SDK.

## Develop and test

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, repository layout, tests,
packaging and contribution requirements. The [architecture guide](docs/ARCHITECTURE.md)
and [operator catalog](operators/README.md) cover composing and extending applications.

## Limits

The examples use synthetic data with live computation. They are demonstrations and
validation tools, not production risk controls. Generated quotes do not represent the
National Best Bid and Offer (NBBO). With 60 returns, the covariance matrix has rank at
most 59; correlations are descriptive and annualized volatility is not a forecast.
Uniformly paced replay does not establish performance on a bursty external feed or
validate physical NIC behavior.

[Apache-2.0 license](LICENSE)
