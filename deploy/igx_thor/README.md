# Thor deployment: DAQIRI/ConnectX and GPUDirect ingest

These launchers run the `equity_analytics` application on IGX Thor with NVIDIA
DAQIRI/DPDK and a ConnectX NIC, in containers, optionally with the sender on the same host
through PHY loopback. They are the only way to run the NIC ingest modes: `holoscan run`
starts an image without DAQIRI, as a non-root user, with the host IPC namespace.

```text
INGEST=ano (daqiri-host): MITCH / UDP → ConnectX / DAQIRI → pinned host memory
            → Python decode / NumPy quote bars → CuPy / cuBLAS → dashboard
INGEST=gpu (gpudirect):   MITCH / UDP → ConnectX / DAQIRI → GPU memory (GPUDirect, dma-buf)
            → CUDA gather / MITCH decode / quote bars → CuPy / cuBLAS → dashboard
```

To run the same GPU chain without a NIC, use the `replay` mode of `equity_analytics`
(pcap frames staged in GPU memory; see the [main README](../../README.md)).

Results and validation: [GPUDirect ingest](../../docs/GPU_INGEST.md). The published
numbers were measured with Holoscan SDK 4.3; the Holoscan 5.0 port has not yet been run
against the NIC (the revalidation status is in that document).

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
docker build -f deploy/igx_thor/Dockerfile -t holoscan-finance:hsdk5-daqiri .   # the launchers' default ANO_IMAGE
```

In Holoscan 5, `import holoscan` needs the GPU driver, and the SDK is found only through
the image's `PYTHONPATH` (`/usr/lib/python3.12/site-packages`).
[ano_python.sh](ano_python.sh) therefore appends the checkout's `python/` directory to it
and never replaces it. The sender and the replay controller run without the GPU runtime;
they import only `holoscan_finance` tools, which never import the SDK.

The launcher mounts this checkout read-only and checks imports (Holoscan SDK 5, the
operators, DAQIRI with the bindings the ingest mode needs), GPU, NIC vendor, CPU affinity
and hugepages before starting; DAQIRI failure never falls back to sockets. The receiver
runs `applications/equity_analytics/equity_analytics.py --ingest daqiri-host|gpudirect` with
`--nic-pcie`, `--nic-master-core` and `--nic-rx-core` from the variables below. It gets a
private IPC namespace with `--shm-size=4g`, never `--ipc=host`: DPDK needs the
bind-mounted `/dev/hugepages`, not the host's IPC namespace, and a root Holoscan 5 process
with `--ipc=host` leaves a root-owned shared-memory file in the host's `/dev/shm` that
breaks later non-root Holoscan runs.

The receiver needs no tape; the sender needs the prepared `manifest.json` and
`events.bin` (see the [main README](../../README.md#the-licensed-day)).

## One-host PHY loopback

This changes NIC PHY, address and neighbor state. Use a dedicated test port (not the
management/SSH port) with no global IP address and PHY loopback off; `MST_IMAGE` must
contain `mstlink` and `ip`. The launcher checks these conditions.

```bash
export ANO_PCIE=0004:03:00.0 ANO_NIC=enP4p3s0f0np0 TAPE_DIR=/licensed/prepared-nyq
LOOPBACK=1 bash deploy/igx_thor/igx_ano_run.sh --dry-run
LOOPBACK=1 bash deploy/igx_thor/igx_ano_run.sh
bash deploy/igx_thor/igx_ano_down.sh   # stop
```

Launch order: preflight, receiver, HTTP readiness, PHY loopback, sender. Existing
containers are never replaced or adopted (ownership label `nyq-equities`); a failed
start removes only what it created. Teardown restores the NIC's addresses, PHY
loopback, link state and NetworkManager management from the receiver's labels; a
failed cleanup keeps the labels so it can be retried. Pass `APP_NAME` again if you
customized it.

## External sender

```bash
ANO_PCIE=0004:03:00.0 ANO_MASTER_CORE=3 ANO_RX_CORE=9 \
    WEB_HOST=0.0.0.0 bash deploy/igx_thor/igx_ano_run.sh
# on the sender:
PYTHONPATH=python python3 -m holoscan_finance.tools.replay_sender --data /licensed/prepared-nyq \
    --host RECEIVER_CONNECTX_IP --port 26400 \
    --control-url http://RECEIVER_MANAGEMENT_IP:8080
```

With `LOOPBACK=0` (default) only the receiver starts and the NIC is not changed.
Packets must be untagged IPv4/UDP without fragmentation; HTTP can use a management
network.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `ANO_IMAGE`, `MST_IMAGE` | `holoscan-finance:hsdk5-daqiri`, `ano-igx:mst` | Runtime image ([Dockerfile](Dockerfile)) and PHY tools |
| `ANO_PCIE`, `ANO_NIC` | `0004:03:00.0`, `enP4p3s0f0np0` | Receiver function and test interface |
| `ANO_MASTER_CORE`, `ANO_RX_CORE` | `3`, `9` | Distinct DAQIRI cores |
| `GPU_UUID` | unset | Select one GPU; on Thor device index 0 can be the integrated GPU (the receiver also gets `CUDA_DEVICE_ORDER=PCI_BUS_ID`) |
| `INGEST` | `ano` | `ano` or `daqiri-host`: host memory, Python decode. `gpu` or `gpudirect`: GPUDirect, CUDA gather/decode/bars |
| `KERNEL_CACHE_DIR` | `~/.cache/holoscan-finance/cupy` | CUDA kernels and the native drain for `INGEST=gpu` (first start compiles ~15 s) |
| `LOOPBACK`, `START_FEED` | `0`, follows loopback | PHY setup and local sender |
| `LOOPBACK_CIDR`, `LOOPBACK_PEER` | `192.168.100.1/24`, `192.168.100.2` | Test network |
| `FEED_HOST` | loopback peer if enabled | Local sender destination |
| `WEB_HOST`, `WEB_PORT`, `UDP_PORT` | `127.0.0.1`, `8080`, `26400` | HTTP bind and UDP flow match |
| `TAPE_DIR` | repo `data/nyq-equities-2025-04-01` | Prepared licensed tape |
| `CPU_THREADS` | `0` | CPU BLAS threads (0 = tune automatically) |
| `ANALYTICS_BACKEND` | `compare` | `compare` runs both workers; `cpu` or `gpu` runs one, for separate measurements (`INGEST=gpu` accepts `compare` or `gpu`) |
| `SPEED`, `RATE` | `10`, `50000` | Replay multiplier; `SPEED=0` selects fixed observations/s |
| `READY_TIMEOUT` | `90` | Receiver readiness timeout, seconds |
| `APP_NAME`, `FEED_NAME` | `fin-equities-ano`, `fin-equities-feed` | Container names |
| `DAQIRI_DV_FLOW_EN`, `DAQIRI_RX_FLOW_GROUP` | `1`, `0` | Steering-patch overrides (see above) |
| `FULLDAY_PCAP` | unset | Captured day ([capture.sh](gpu_ingest_bench/capture.sh)); enables the full-day button (`INGEST=gpu` or `gpudirect`) |
| `FULLDAY_BASE_PORT`, `FULLDAY_PACKETS` | unset | The capture's sender port and packet count |
| `TCPREPLAY_DIR`, `TCPREPLAY_LIBS` | unset | `tcpreplay`/`tcprewrite` binaries and libraries for the replay controller |
| `FULLDAY_WORK` | `replay_control/` beside the capture | Rewritten per-run copies of the capture |
| `REPLAY_NAME`, `REPLAY_PORT` | `$APP_NAME-replay`, `8095` | Replay controller container and port |
| `HOLOSCAN_FINANCE_NATIVE_DRAIN` | unset | `0` selects the Python drain thread instead of the native drain (A/B tests) |
| `HOLOSCAN_FINANCE_FAULTS` | unset | `1` enables `POST /debug/hold_gil {"seconds": s}`, which holds the Python interpreter lock (fault-injection tests only) |

DAQIRI config: flow isolation on `rx_port` queue 0. `INGEST=ano` uses 51,200 pinned
host buffers, bursts of up to 1,024 packets and a 5,000 µs partial-burst timeout.
`INGEST=gpu` uses 114,688 `device` buffers (the GPU's 256 MiB BAR1 window holds about
120,000 2-KiB buffers; ~0.57 s at 200,000 packets/s) and bursts of up to 4,096 packets
with a 20,000 µs timeout. The native drain copies each burst into a 1,048,576-slot ring in
GPU memory (2 GiB, ~5 s at 200,000 packets/s) and returns it to the NIC without the
Python interpreter lock; the receiver also moves its other threads off the DAQIRI cores.
`--nic-config PATH` takes a custom YAML that keeps that port and queue.

## Full-day button

With `FULLDAY_PCAP` set and `INGEST=gpu`, the launcher also starts a privileged replay
controller and the dashboard shows a "Full trading day at line rate" panel. The button
pauses the paced sender, announces a fresh generation and `tcpreplay`s the captured
day out of the test port (200,000 packets/s by default). The panel times first to last
arrival and reports packets, sequence gaps, ring drops, events, bars and the multiple
of real time; "Resume replay" returns to the paced feed. `tcpreplay` runs off the
DAQIRI cores, and `compare` mode skips CPU validation while a burst is in flight.

`fullday.stalls` in `/snapshot` records the longest gaps between NIC bursts, the
slowest drain and decode passes (per step), the native drain's longest gap between NIC
polls (`native`), process-wide stalls, garbage-collection pauses and every stall over
50 ms; `network.gpu_ingest.ring_peak` is the most packets the ring held. NIC queue
overflow shows in the receiver log as DAQIRI's `Rx: Dropped N packets`: the drain fell
behind by more than the NIC buffer pool (~0.57 s at 200,000 packets/s). A Python stall
no longer causes it unless it outlasts the ring (~5 s).

Three host settings are applied while the receiver runs and restored by teardown (the
previous values are kept in the receiver's labels; a reboot also restores them):

- With `LOOPBACK=1`, the DAQIRI cores are banned from irqbalance's IRQ placement
  ([irq_isolate.py](irq_isolate.py)); otherwise the local sender's transmit-completion
  interrupts can land on the RX core, which lost 4–35% of a burst.
- With `INGEST=gpu`, NVIDIA GPUs that runtime-suspend when idle (Thor's integrated
  GPU) are kept powered. While suspended, the receiver's first new CUDA allocation
  wakes it, taking 0.13–0.56 s with the interpreter lock held, and the button idles
  the GPU for the ~3.4 s `tcpreplay` preload.
- With `FULLDAY_PCAP`, `fwupd-refresh.timer` (an hourly firmware-metadata refresh) is
  stopped. This is a precaution: one lossy run followed a refresh by ~10 s, but forcing
  a refresh or an fwupd restart at the start of 9 bursts caused no stall over 111 ms,
  and stalls still occurred with the timer stopped.

With these settings on Holoscan 4.3, 242 of 250 `compare`-mode runs with a `/snapshot`
poller delivered all 2,880,565 packets in order in ~14.4 s each (~1,620× real time). Five
lost 2–4% to receiver stalls longer than the ~0.57 s of NIC buffering, one rejected 35
stale-port packets, and in two, 26–41 packets arrived out of order and were discarded
([details](../../docs/GPU_INGEST.md#performance)). Those runs used the Python drain; the
native drain survives injected interpreter-lock stalls of up to 3 s
([native drain](../../docs/GPU_INGEST.md#native-drain)). The 5.0 build repeats these
series only with approval for the NIC.

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

A complete launch with PHY loopback, the full-day button and the 5.0 image. Pick the GPU by
UUID (`nvidia-smi -L`), since CUDA device index 0 can be the integrated GPU:

```bash
export ANO_IMAGE=holoscan-finance:hsdk5-daqiri INGEST=gpu LOOPBACK=1 \
    GPU_UUID=GPU-<uuid> \
    TAPE_DIR=/data/nyq-equities-2025-04-01 APP_NAME=fin-gpu-demo FEED_NAME=fin-gpu-feed \
    FULLDAY_PCAP=/data/xportbench/base.pcap FULLDAY_BASE_PORT=$(cat /data/xportbench/baseport) \
    FULLDAY_PACKETS=$(cat /data/xportbench/base.count) \
    TCPREPLAY_DIR=/data/xportbench/root/usr/bin \
    TCPREPLAY_LIBS=/data/xportbench/root/usr/lib/aarch64-linux-gnu
bash deploy/igx_thor/igx_ano_run.sh
```

The dashboard binds to `127.0.0.1`. From another machine, tunnel to it:
`ssh -N -L 8080:127.0.0.1:8080 <user>@<thor-host>`, then open <http://localhost:8080/>.
