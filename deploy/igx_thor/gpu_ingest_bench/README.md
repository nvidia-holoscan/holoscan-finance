# Reproducing the ingest benchmark

The Python replay sender tops out at a few thousand packets per second, so the
receiver's ceiling is measured with real MITCH packets replayed by `tcpreplay`.
Results and method: [docs/GPU_INGEST.md](../../../docs/GPU_INGEST.md).

Prerequisites on the Thor: the running demo (for the one-time capture), the
`ano-igx:mst` helper image (privileged `tcpdump`/`sysctl` via `nsenter`), and
`tcpreplay`/`tcprewrite`. Without root, unpack them from the distribution packages:

```bash
mkdir -p ~/tools && cd ~/tools && apt-get download tcpreplay libdumbnet1
for d in *.deb; do dpkg -x "$d" root; done   # binaries: root/usr/bin
```

1. **Capture** one replay generation from the sender's egress (a seek restarts
   sequences at 1; ~3M packets is a full trading day and takes ~11 minutes):

   ```bash
   OUT=~/bench/capture.pcap PACKETS=3000000 deploy/igx_thor/gpu_ingest_bench/capture.sh
   python -m holoscan.equities.bench_replay inspect ~/bench/capture.pcap
   ```

   The largest source port is the new generation. Keep only its packets and check
   that sequences are continuous:

   ```bash
   tcpdump -r ~/bench/capture.pcap -w ~/bench/base.pcap 'udp src port <PORT>'
   python -m holoscan.equities.bench_replay inspect ~/bench/base.pcap --port <PORT>
   ```

2. **Stop anything else using the ConnectX port** (e.g. `igx_ano_down.sh`).

3. **Sweep.** Each rate registers a fresh generation (`bench_replay register`) and
   rewrites the capture's source port, because the receiver only accepts packets
   from the manifest's sender port and each generation needs a new one:

   ```bash
   BASE_PCAP=~/bench/base.pcap BASE_PORT=<PORT> TCPREPLAY_DIR=~/tools/root/usr/bin \
   TAPE_DIR=/path/to/nyq-equities-2025-04-01 ANO_IMAGE=holoscan-finance:nyq-20260911 \
   GPU_UUID=GPU-... MODES="gpu ano" RATES="4500 10000 20000 50000 75000" \
   deploy/igx_thor/gpu_ingest_bench/sweep.sh
   ```

The `async` mode additionally rewrites the destination to the host address and
sets `accept_local`/`rp_filter` (restored on exit) so kernel sockets can receive
frames returning through the PHY loopback; it raises `rmem_max` so the receiver
gets the 64 MiB socket buffer it requests.
