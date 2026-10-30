# Thor deployment: DAQIRI/ConnectX and GPUDirect ingest

These launchers run the `equity_analytics` application on IGX Thor with NVIDIA
DAQIRI/DPDK and a ConnectX NIC, in containers, optionally with the sender on the same host
through PHY loopback. They are the only way to run the NIC ingest modes: `holoscan run`
starts an image without DAQIRI, as a non-root user, with the host IPC namespace.

```text
INGEST=daqiri-host: MITCH / UDP → ConnectX / DAQIRI → pinned host memory
            → Python decode / NumPy quote bars → custom CUDA kernels (CuPy) → dashboard
INGEST=gpudirect:   MITCH / UDP → ConnectX / DAQIRI → GPU memory (GPUDirect, dma-buf)
            → CUDA gather / MITCH decode / quote bars → custom CUDA kernels (CuPy) → dashboard
```

To run the same GPU chain without a NIC, use the `replay` mode of `equity_analytics`
(pcap frames staged in GPU memory; see the [main README](../../README.md)).

Design and method: [GPUDirect ingest](../../docs/GPU_INGEST.md). Follow the
[synthetic validation guide](../../docs/VALIDATION.md) for correctness checks and record
fresh hardware measurements for any NIC throughput claim.

## Prepare the platform

A supported ConnectX Ethernet device, NVIDIA container runtime, mlx5/verbs,
`/dev/infiniband`, and hugepages mounted at `/dev/hugepages` (Thor resets
`nr_hugepages` to 0 at boot; 1,024–4,096 2-MiB pages work). Follow NVIDIA's
[DAQIRI setup guide](https://nvidia.github.io/daqiri/getting-started/) and
[Python bindings build](https://nvidia.github.io/daqiri/api-reference/python/).

The receiver image is built by [Dockerfile](Dockerfile) on the Holoscan SDK 5.0 image
(`nvcr.io/nvidia/clara-holoscan/holoscan:v5.0.0-cuda13`). It builds DPDK 25.11 and DAQIRI
(DPDK engine, Python binding) from source at a pinned commit, installs rdma-core from the
DOCA 3.3.0 repository, and adds the app dependencies; the build takes about 8 minutes on
Thor. On Thor (DOCA-OFED 25.10) DAQIRI must be built with
[daqiri_thor_steering.patch](daqiri_thor_steering.patch), which the Dockerfile applies:
mlx5 hardware steering and non-zero flow groups fail there, and the patch adds the
`DAQIRI_DV_FLOW_EN=1` / `DAQIRI_RX_FLOW_GROUP=0` overrides the launcher sets.

```bash
docker build -f deploy/igx_thor/Dockerfile -t holoscan-finance:hsdk5-daqiri .   # the launchers' default RECEIVER_IMAGE
```

In Holoscan 5, `import holoscan` needs the GPU driver, and the SDK is found only through
the image's `PYTHONPATH` (`/usr/lib/python3.12/site-packages`).
[run_python.sh](run_python.sh) therefore appends the checkout's `python/` directory to it
and never replaces it. The sender and the replay controller run without the GPU runtime;
they import only `holoscan_finance` tools, which never import the SDK.

The launcher mounts this checkout read-only and checks imports (Holoscan SDK 5, the
operators, DAQIRI with the bindings the ingest mode needs), GPU, NIC vendor, CPU affinity
and hugepages before starting; DAQIRI failure never falls back to sockets. The receiver
runs `examples/equity_analytics/equity_analytics.py --ingest daqiri-host|gpudirect` with
`--nic-pcie`, `--nic-master-core` and `--nic-rx-core` from the variables below. It gets a
private IPC namespace with `--shm-size=4g`, never `--ipc=host`: DPDK needs the
bind-mounted `/dev/hugepages`, not the host's IPC namespace, and a root Holoscan 5 process
with `--ipc=host` leaves a root-owned shared-memory file in the host's `/dev/shm` that
breaks later non-root Holoscan runs.

The receiver needs no tape; generate the sender's `manifest.json` and `events.bin`
from the repository root (see [synthetic UDP replay](../../README.md#synthetic-udp-replay)):

```bash
PYTHONPATH=python python3 -m holoscan_finance.tools.synth --out data/synth \
    --symbols 1000 --packets 400000 --seconds 23400 --seed 7 --reuse
```

This writes `data/synth/tape`, `data/synth/day.pcap` and `data/synth/synth.json`, using
the same generation settings as the default equity replay demo.

## One-host PHY loopback

This changes NIC PHY, address and neighbor state. Use a dedicated test port (not the
management/SSH port) with no global IP address and PHY loopback off; `MST_IMAGE` must
contain `mstlink` and `ip`. The launcher checks these conditions. This repository does not
build `MST_IMAGE`; use any image with NVIDIA MFT (`mstlink`, `mstreg`), `iproute2`,
`tcpdump` and `nsenter`, tagged `holoscan-finance:mst` or named by `MST_IMAGE`.

```bash
export NIC_PCIE=0004:03:00.0 NIC_NAME=enP4p3s0f0np0   # TAPE_DIR: the repo's data/synth/tape
LOOPBACK=1 bash deploy/igx_thor/igx_run.sh --dry-run
LOOPBACK=1 bash deploy/igx_thor/igx_run.sh
bash deploy/igx_thor/igx_down.sh   # stop
```

Launch order: preflight, receiver, HTTP readiness, PHY loopback, sender. Existing
containers are never replaced or adopted; an ownership label identifies this demo. A failed
start removes only what it created. Teardown restores the NIC's addresses, PHY
loopback, link state and NetworkManager management from the receiver's labels; a
failed cleanup keeps the labels so it can be retried. Pass `APP_NAME` again if you
customized it.

## External sender

```bash
NIC_PCIE=0004:03:00.0 NIC_MASTER_CORE=3 NIC_RX_CORE=9 \
    WEB_HOST=0.0.0.0 bash deploy/igx_thor/igx_run.sh
# on the sender:
PYTHONPATH=python python3 -m holoscan_finance.tools.replay_sender --data data/synth/tape \
    --host RECEIVER_CONNECTX_IP --port 26400 \
    --control-url http://RECEIVER_MANAGEMENT_IP:8080
```

With `LOOPBACK=0` (default) only the receiver starts and the NIC is not changed.
Packets must be untagged IPv4/UDP without fragmentation; HTTP can use a management
network.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `RECEIVER_IMAGE`, `MST_IMAGE` | `holoscan-finance:hsdk5-daqiri`, `holoscan-finance:mst` | Runtime image ([Dockerfile](Dockerfile)) and PHY tools (see above) |
| `NIC_PCIE`, `NIC_NAME` | `0004:03:00.0`, `enP4p3s0f0np0` | Receiver function and test interface |
| `NIC_MASTER_CORE`, `NIC_RX_CORE` | `3`, `9` | Distinct DAQIRI cores |
| `NIC_DRAIN_CORE` | `10` | `INGEST=gpudirect`: the native drain's own core, kept free of other threads and interrupts; the drain busy-polls it (`HOLOSCAN_FINANCE_DRAIN_SPIN=0`: sleeps). `-1`: unpinned |
| `GPU_UUID` | unset | Select one GPU; on Thor device index 0 can be the integrated GPU (the receiver also gets `CUDA_DEVICE_ORDER=PCI_BUS_ID`) |
| `INGEST` | `daqiri-host` | `daqiri-host`: host memory, Python decode. `gpudirect`: GPUDirect, CUDA gather/decode/bars |
| `KERNEL_CACHE_DIR` | `~/.cache/holoscan-finance/cupy` | CUDA kernels and the native drain for `INGEST=gpudirect` (first start compiles ~15 s) |
| `LOOPBACK`, `START_FEED` | `0`, follows loopback | PHY setup and local sender |
| `LOOPBACK_CIDR`, `LOOPBACK_PEER` | `192.168.100.1/24`, `192.168.100.2` | Test network |
| `FEED_HOST` | loopback peer if enabled | Local sender destination |
| `WEB_HOST`, `WEB_PORT`, `UDP_PORT` | `127.0.0.1`, `8080`, `26400` | HTTP bind and UDP flow match |
| `TAPE_DIR` | repo `data/synth/tape` | Generated synthetic tape ([data layout](../../docs/VALIDATION.md)) |
| `CPU_THREADS` | `0` | CPU BLAS threads (0 = tune automatically) |
| `ANALYTICS_BACKEND` | `compare` | `compare` runs both workers; `cpu` or `gpu` runs one, for separate measurements (`INGEST=gpudirect` accepts `compare` or `gpu`) |
| `SPEED`, `RATE` | `10`, `50000` | Replay multiplier; `SPEED=0` selects fixed observations/s |
| `READY_TIMEOUT` | `90` | Receiver readiness timeout, seconds |
| `APP_NAME`, `FEED_NAME` | `fin-equities-receiver`, `fin-equities-feed` | Container names |
| `DAQIRI_DV_FLOW_EN`, `DAQIRI_RX_FLOW_GROUP` | `1`, `0` | Steering-patch overrides (see above) |
| `FULLDAY_PCAP` | unset | Generated synthetic pcap, such as `data/synth/day.pcap`; enables the full-day button (`INGEST=gpudirect`) |
| `FULLDAY_BASE_PORT`, `FULLDAY_PACKETS` | unset | The capture's sender port and packet count |
| `TCPREPLAY_DIR`, `TCPREPLAY_LIBS` | unset | `tcpreplay`/`tcprewrite` binaries and libraries for the replay controller |
| `FULLDAY_WORK` | `replay_control/` beside the capture | Rewritten per-run copies of the capture |
| `REPLAY_NAME`, `REPLAY_PORT` | `$APP_NAME-replay`, `8095` | Replay controller container and port |
| `HOLOSCAN_FINANCE_NATIVE_DRAIN` | unset | `0` selects the Python drain thread instead of the native drain (A/B tests) |
| `HOLOSCAN_FINANCE_FAULTS` | unset | `1` enables `POST /debug/hold_gil {"seconds": s}`, which holds the Python interpreter lock (fault-injection tests only) |

DAQIRI config: flow isolation on `rx_port` queue 0. `INGEST=daqiri-host` uses 51,200 pinned
host buffers, bursts of up to 1,024 packets and a 5,000 µs partial-burst timeout.
`INGEST=gpudirect` uses 114,688 `device` buffers (the GPU's 256 MiB BAR1 window holds about
120,000 2-KiB buffers; ~0.57 s at 200,000 packets/s) and bursts of up to 4,096 packets
with a 20,000 µs timeout. The native drain copies each burst into a 1,048,576-slot ring in
GPU memory (2 GiB, ~5 s at 200,000 packets/s) and returns it to the NIC without the
Python interpreter lock; the receiver also moves its other threads off the DAQIRI cores.
`--nic-config PATH` takes a custom YAML that keeps that port and queue.

## Synthetic capture replay

With `FULLDAY_PCAP` set and `INGEST=gpudirect`, the launcher also starts a privileged replay
controller and the dashboard shows a capture replay panel. The button
pauses the paced sender, announces a fresh generation and `tcpreplay`s the synthetic
capture out of the test port (200,000 packets/s by default). The panel times first to last
arrival and reports packets, sequence gaps, ring drops, events, bars and the multiple
of the synthetic event-time span; "Resume replay" returns to the paced feed. `tcpreplay` runs off the
DAQIRI cores, and `compare` mode skips CPU validation while a burst is in flight.
The capture covers the span chosen with `synth --seconds`; the generated example above
covers a 6.5-hour synthetic session and is suitable for functional checks.

`fullday.stalls` in `/snapshot` records the longest gaps between NIC bursts, the
slowest drain and decode passes (per step), the native drain's longest gap between NIC
polls (`native`), process-wide stalls, garbage-collection pauses and every stall over
50 ms; `network.gpu_ingest.ring_peak` is the most packets the ring held. NIC queue
overflow shows in the receiver log as DAQIRI's `Rx: Dropped N packets`: the drain fell
behind by more than the NIC buffer pool (~0.57 s at 200,000 packets/s). A Python stall
does not cause it unless it outlasts the ring (~5 s).

Three host settings are applied while the receiver runs and restored by teardown (the
previous values are kept in the receiver's labels; a reboot also restores them):

- With `LOOPBACK=1`, the DAQIRI cores are banned from irqbalance's IRQ placement
  ([irq_isolate.py](irq_isolate.py)); otherwise the local sender's transmit-completion
  interrupts can land on the RX core and delay it long enough to lose part of a burst.
- With `INGEST=gpudirect`, NVIDIA GPUs that runtime-suspend when idle (Thor's integrated
  GPU) are kept powered. While suspended, the receiver's first new CUDA allocation
  wakes it and can delay the receiver's CUDA calls. The replay button can leave the
  GPU idle while `tcpreplay` preloads the capture.
- With `FULLDAY_PCAP`, `fwupd-refresh.timer` (an hourly firmware-metadata refresh) is
  stopped, as a precaution against a refresh during a burst.

NIC sweeps and fault-injection runs require approval
(method: [GPUDirect ingest](../../docs/GPU_INGEST.md)).

## Check it

```bash
docker logs --tail 50 "$APP_NAME"
curl --fail http://127.0.0.1:8080/snapshot
```

The snapshot should show the graph running, the selected `network.ingest`, readiness,
and advancing observations, bars and results with no RX error; in `compare` mode
CPU/GPU agreement should pass. Use sequence gaps, ring drops, malformed packets and
DAQIRI counters over the same interval for any throughput claim.

## Example: full GPUDirect demo on one Thor

A complete launch with PHY loopback, synthetic capture replay and the receiver image. Pick the GPU by
UUID (`nvidia-smi -L`), since CUDA device index 0 can be the integrated GPU:

```bash
export RECEIVER_IMAGE=holoscan-finance:hsdk5-daqiri INGEST=gpudirect LOOPBACK=1 \
    GPU_UUID=GPU-<uuid> \
    TAPE_DIR=$PWD/data/synth/tape APP_NAME=fin-gpu-demo FEED_NAME=fin-gpu-feed \
    FULLDAY_PCAP=$PWD/data/synth/day.pcap FULLDAY_BASE_PORT=40000 FULLDAY_PACKETS=400001 \
    TCPREPLAY_DIR=~/tools/root/usr/bin TCPREPLAY_LIBS=~/tools/root/usr/lib/aarch64-linux-gnu
bash deploy/igx_thor/igx_run.sh
```

For the generation command above, `data/synth/day.pcap` contains 400,000 data packets
and a final marker packet from source port 40000. If you change the generator arguments,
use the capture's actual sender port and packet count:

```bash
PYTHONPATH=python python3 -m holoscan_finance.tools.bench_replay inspect data/synth/day.pcap
```

For sustained rate measurements, generate a larger fixture and follow
[gpu_ingest_bench](gpu_ingest_bench/README.md).

The dashboard binds to `127.0.0.1`. From another machine, tunnel to it:
`ssh -N -L 8080:127.0.0.1:8080 <user>@<thor-host>`, then open <http://localhost:8080/>.
