# Benchmarking synthetic ingest

Use generated SBE packets with `tcpreplay` to measure the receiver independently
of the paced Python sender.
Method: [docs/GPU_INGEST.md](../../../docs/GPU_INGEST.md).

Prerequisites on the Thor: the `MST_IMAGE`
helper image (default `holoscan-finance:mst`, see the
[deploy README](../README.md#one-host-phy-loopback); privileged `tcpdump`/`sysctl` via
`nsenter`), the receiver image built from [deploy/igx_thor/Dockerfile](../Dockerfile), and
`tcpreplay`/`tcprewrite`. Without root, unpack them from the distribution packages:

```bash
mkdir -p ~/tools && cd ~/tools && apt-get download tcpreplay libdumbnet1
for d in *.deb; do dpkg -x "$d" root; done   # binaries: root/usr/bin
```

These scripts change NIC and host settings. Use a dedicated test port, stop other
workloads using it, and obtain approval before running a hardware benchmark.

1. **Generate** a synthetic tape and its matching capture from the repository root:

   ```bash
   PYTHONPATH=python python3 -m holoscan_finance.tools.synth --out data/synth \
       --symbols 1000 --packets 400000 --seconds 23400 --seed 7 --reuse
   PYTHONPATH=python python3 -m holoscan_finance.tools.bench_replay inspect data/synth/day.pcap
   ```

   This matches the default equity replay demo: a 6.5-hour synthetic session with
   400,000 data packets plus a final marker, from source port 40000. It is suitable for
   functional checks. For sustained throughput measurements, generate a larger fixture
   with enough `--packets` to cover the requested rate and duration. Record the generator
   arguments and `synth.json` alongside each result. `--seconds` controls the synthetic
   event-time span.

2. **Stop anything else using the ConnectX port** (e.g. `igx_down.sh`).

3. **Sweep.** Each rate registers a fresh generation (`bench_replay register`) and
   rewrites the capture's source port, because the receiver only accepts packets
   from the manifest's sender port and each generation needs a new one:

   ```bash
   BASE_PCAP=$PWD/data/synth/day.pcap BASE_PORT=40000 TCPREPLAY_DIR=~/tools/root/usr/bin \
   TAPE_DIR=$PWD/data/synth/tape RECEIVER_IMAGE=holoscan-finance:hsdk5-daqiri \
   GPU_UUID=GPU-... MODES="gpudirect daqiri-host" RATES="4500 10000 20000 50000 75000" \
   deploy/igx_thor/gpu_ingest_bench/sweep.sh
   ```

`MODES` takes the app's ingest names `gpudirect`, `daqiri-host` and `udp`. The generated
pcap uses placeholder Ethernet and IP addresses, which the DAQIRI modes do not filter.
For kernel-socket `udp` measurements, use [capture.sh](capture.sh) to capture a fresh
generation from the synthetic paced sender on the configured PHY loopback. Filter it
to that generation's source port and set `BASE_PCAP` and `BASE_PORT` to the captured
file and port. This preserves the test NIC's Ethernet and loopback IP addresses.
Choose `PACKETS` for the generated tape size; inspect the capture to verify that the
selected generation is complete and its sequences are continuous.

The `udp` mode additionally rewrites the loopback destination to the host address and sets
`accept_local`/`rp_filter` (restored on exit) so kernel sockets can receive frames
returning through the PHY loopback; it raises `rmem_max` so the receiver gets the 64 MiB
socket buffer it requests.

Use the loss counters and timing from the same run when reporting a rate. The fixture
size or requested replay rate alone is not a measured throughput result.
