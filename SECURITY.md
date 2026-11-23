# Security Policy: Holoscan Module for Financial Services Industry

NVIDIA is dedicated to the security and trust of our software products and services, including
all source code repositories managed through our organization.

If you need to report a security issue, please use the contact points below.
**Please do not report security vulnerabilities through GitHub issues, pull requests, or
discussions.**

## Reporting a Vulnerability

To report a potential security vulnerability in `holoscan-finance`:

- **Web (preferred):** [NVIDIA Vulnerability Disclosure Program](https://www.nvidia.com/en-us/security/)
- **E-Mail:** [psirt@nvidia.com](mailto:psirt@nvidia.com)
  - We encourage you to encrypt sensitive details with the
    [NVIDIA public PGP key](https://www.nvidia.com/en-us/security/pgp-key).
- **GitHub:** Use this repository's **Security** tab > **Report a vulnerability** to file a
  private advisory.

If a security issue is raised through public channels, maintainers may limit public discussion
and redirect the reporter to the private channels above.

### What to Include

- Project name (`holoscan-finance`) and the affected version or branch (see `version` in
  `pyproject.toml`), plus how the code was obtained: wheel or built from source, and which
  container image (the Holoscan SDK image or the `deploy/igx_thor` receiver image).
- Type of vulnerability (for example: input-validation flaw, resource exhaustion, insecure
  default, information disclosure, supply-chain issue).
- Step-by-step reproduction instructions, including the application, ingest mode, command-line
  flags, and the packet capture, tape or HTTP requests used.
- Proof-of-concept code or a minimal reproducer, if available.
- Impact assessment covering confidentiality, integrity, availability, and scope.

### What to Expect

NVIDIA's Product Security Incident Response Team (PSIRT) will acknowledge receipt, validate the
issue and its severity, coordinate remediation with engineering, and publish a security bulletin
where appropriate. NVIDIA does not operate a public bug bounty program for this project, but
externally reported issues are acknowledged under our coordinated vulnerability disclosure
policy.

## Security Architecture & Context

`holoscan-finance` is the Holoscan Module for Financial Services Industry: a self-contained
[Holoscan SDK 5](https://github.com/nvidia-holoscan) module for building streaming
financial-services applications. The Python package `holoscan_finance` (public alias `holoscan.finance`)
provides ten operators: four receive sources (`GpuSbeSourceOp`, `GpuSbeReplaySourceOp`,
`UdpPacketSourceOp`, `DaqiriPacketSourceOp`), `SbeDecodeOp`, `QuoteBarsOp`, `RollingRiskOp`,
`RiskCrossCheckOp`, `CallbackSinkOp` and `RiskSummaryOp`. It also provides the libraries behind
them: the SBE codec and schema, CUDA encode/decode and quote-bar kernels, the covariance engine, and the control
plane and web dashboard. Two example applications under `examples/` compose these operators:
`equity_analytics` is a live web dashboard; `market_data_bench` is a headless benchmark that
reads only capture files.
Command-line tools cover replay, synthetic data generation, validation and benchmarks, and
`deploy/igx_thor` holds the launchers for an IGX Thor with a ConnectX NIC.

The code is Python, with three kinds of generated machine code:

- the CUDA kernels, compiled at run time by CuPy (NVRTC);
- the native receive drain (`ingest/native/daqiri_drain.cpp`), compiled on first use with the
  system C++ compiler;
- in the benchmark and validation tools only, a numba-compiled CPU decoder.

`equity_analytics` receives **untrusted network input** and exposes a **control plane**:

- UDP market data, through a kernel socket (default `127.0.0.1:26400`) or through NVIDIA DAQIRI
  (DPDK) on a dedicated ConnectX port, into pinned host memory or directly into GPU memory
  (GPUDirect).
- An HTTP dashboard (FastAPI/uvicorn, default `127.0.0.1:8080`) with **no authentication**. Its
  `POST` endpoints change the run: `/manifest` (a new replay generation, symbol universe and
  sender port), `/feedrate`, `/feedstatus`, `/capture`, `/capture/finish`, `/replay/fullday` and
  `/replay/resume`.
- When `FULLDAY_PCAP` is set, the full-day replay controller
  (`deploy/igx_thor/gpu_ingest_bench/replay_control.py`). It is an unauthenticated HTTP server,
  always bound to `127.0.0.1` on `REPLAY_PORT` (default 8095), in a privileged container with host
  networking, and it runs `tcpreplay` from there. Its inputs are `POST /start` with a rate of
  1-250,000 packets/s and `POST /stop`; the capture, NIC and binaries are fixed at launch.
- With `--topics`, an export of compact risk summaries over HoloMQ shared memory, described by
  an endpoint catalog file.

Its primary security responsibilities are:

- the robustness of packet parsing against malformed or hostile input;
- the integrity of the analytics derived from the feed;
- containment of the control plane;
- the integrity of the build and privileged deployment chain.

**Repository Exposure Classification:** Public.
Basis: the repository is on github.com under an NVIDIA Holoscan organization and the project is
released under Apache-2.0; this document is written to public-safe detail.

**Service Exposure Classification:** External / Regulated (medium confidence).
Basis: the Apache-2.0 module is distributed externally as a platform-independent wheel for
financial-services workloads. Its example dashboard application opens network listeners when run. The
repository contains no live service, credentials or customer data. The examples and validation
workflows use generated synthetic data.

Key boundaries and interfaces:

- **Untrusted packets.** Every received frame and datagram is untrusted.
  - *Framing.* On the DAQIRI host path `strip_udp_datagram`, and on the GPU paths the
    `gather_frame` CUDA kernel, accept only untagged Ethernet II, unfragmented IPv4 and exact UDP
    lengths. The kernel-socket path relies on the operating system's IP stack, which reassembles
    fragments, so datagrams up to 64 KiB reach the decoder. No path checks IP or UDP checksums or
    the destination address, and DAQIRI steers on the UDP destination port only.
  - *Decoding.* The SBE message header, block lengths, entry count, sequence and every entry's fields
    are checked by the Python decoder and the `sbe_decode` kernel. Both apply the same rejection rules and are tested bit-for-bit
    against each other, including with mutated packets. On the GPU paths, payloads are copied
    into fixed 2,048-byte device slots, and larger payloads are rejected.
  - *Rejections.* Rejected packets are not decoded. SBE rejections are counted on every path,
    and non-UDP frames on the GPU paths. On the DAQIRI host path, frames that fail the IPv4/UDP
    check are dropped without a counter.
  - *Sessions.* Packets whose UDP source port differs from the current manifest's sender port are
    dropped as stale; the source address is not checked. Sequence gaps and reordering are counted
    per session.
- **Unauthenticated control plane.** The dashboard checks the shape and range of what it acts on:
  - manifests: 1-10,000 unique symbols of at most 128 characters, a sender port in 1-65535 not
    used before, and a positive integer generation newer than the current one and at most 24
    hours ahead of the wall clock (every sender names generations with `time.time_ns()`);
  - feed control: a speed of 0-100, a rate of 1-200,000 and a seek below 86,400 seconds;
  - captures: 1-600 seconds;
  - full-day replays: a rate of 1-250,000 packets/s, checked before the paced sender is paused;
  - `/feedstatus`: only the sender's numeric report fields for the current generation are kept
    and shown in `/snapshot`.

  Request bodies are limited to 64 KiB (2 MiB for `/manifest`, whose largest valid body is about
  1.3 MB); a larger one, declared or streamed in chunks, is answered with 413 before it is
  parsed. There is no rate limiting, and callers are not authenticated or authorized. There is no
  Host-header allowlist. Deployments that expose the port beyond a trusted segment must add
  authentication in front of it (a reverse proxy or a network policy).
- **Files.** Prepared tapes (`manifest.json` and `events.bin`) are checked for format, record
  size and file size and are then memory-mapped. Packet captures are read whole into host memory,
  and the replay source uploads them whole into GPU memory. `holoscan_finance.tools.synth`
  generates the demo tapes and captures locally.
- **Run-time code generation.** On the GPUDirect path, the native drain is built into
  `HOLOSCAN_FINANCE_NATIVE_CACHE`, else `$CUPY_CACHE_DIR/native`, else
  `~/.cache/holoscan-finance/native`. An existing build in that directory is loaded with no
  integrity or ownership check. The build takes the compiler from `CXX` (through `PATH`), CUDA from
  `CUDA_PATH`, DAQIRI from `DAQIRI_ROOT` (also the library path at run time) and the pybind11
  headers from `HOLOSCAN_FINANCE_PYBIND11_INCLUDE`, `/usr/include` or an importable `pybind11`.
  `HOLOSCAN_FINANCE_NATIVE=0` or `HOLOSCAN_FINANCE_NATIVE_DRAIN=0` skips it and uses the Python
  drain. CuPy caches compiled kernels in `CUPY_CACHE_DIR` (default `~/.cupy/kernel_cache`;
  `holoscan run` sets it inside the checkout).
- **Processes and messages.** Operators exchange Python objects within one process. Rolling
  analytics on the host-memory paths, and the CPU cross-check on every path, run in spawned worker
  processes. The validation tools fork decode workers.
- **Privileged deployment.** `deploy/igx_thor/igx_run.sh` starts the receiver as root with
  host networking (so `NET_ADMIN` applies to the host's network stack), `NET_ADMIN`, `IPC_LOCK`,
  `SYS_NICE` and `SYS_PTRACE`, all GPUs, unlimited locked memory, `/dev/infiniband` and
  hugepages; with `INGEST=gpudirect` it mounts `KERNEL_CACHE_DIR` read-write as its CuPy and
  native cache. It runs `--privileged` helper containers, some with `--pid=host` and `nsenter`
  into the host, that set PHY loopback (`mstlink`), link state, NetworkManager management, the
  test address and neighbor, the irqbalance ban, GPU runtime power management and
  `fwupd-refresh.timer`.
  - The launcher records the original NetworkManager, link, irqbalance, GPU power-management and
    timer state in the receiver's labels, and with PHY loopback it first requires a dedicated
    port. `igx_down.sh` removes the test settings and restores the recorded state.
  - The launchers never replace existing containers, and teardown stops only containers with the
    project's owner label.
  - The replay container mounts the capture directory, `TCPREPLAY_DIR`, `TCPREPLAY_LIBS` and
    `FULLDAY_WORK`, and writes rewritten copies of the synthetic capture into `FULLDAY_WORK`.
  - The benchmark scripts `gpu_ingest_bench/sweep.sh` and `capture.sh` act on the host directly:
    they apply loopback and, in `udp` mode, host sysctls (restored on exit), or run `tcpdump`
    through `nsenter`, without the launcher's checks.
- **Developer, build and CI tooling.** The `Dockerfile` (dev image), `deploy/igx_thor/Dockerfile`
  (receiver image with DAQIRI built from source), `scripts/dev5.sh` and `dev5cpu.sh`, the CMake
  glue, `.pre-commit-config.yaml` and `.github/workflows/ci.yml` are not part of the runtime, but
  they are part of the trust chain that produces the distributed wheel and the receiver image.

### Threat Model

The following scenarios represent the primary security concerns for this project, including its
auxiliary build, deployment and tooling paths:

1. **Market-data injection driving incorrect analytics.** SBE over UDP carries no
   authentication or integrity protection. The sender-port filter and per-session sequence checks
   detect loss and stale generations; they are not authentication. A party that can deliver
   datagrams to the receive port, or frames to the NIC queue, with the expected source port and
   sequence can inject quotes that pass every check and shift the quote bars and the reported
   volatility and correlation. A single forged packet with a high sequence number or a late
   timestamp also makes every later genuine packet stale or late until the next generation.

2. **Malformed or hostile packets.** Packets are parsed in Python on the host paths and by CUDA
   kernels in GPU memory on the GPUDirect path. A parsing defect, such as a missed length check,
   could read beyond a packet within the device slot ring, mis-decode prices or fail the run, a
   denial of service of the pipeline. The kernels and the Python reference decoder share their
   rejection rules and are tested bit-for-bit, including with mutated packets. The benchmark
   tools' numba decoder parses capture payloads with the same rules.

3. **Unauthenticated control of a running pipeline.** Anyone who can reach the dashboard port can:
   - announce a manifest, which starts a new generation and resets the bars. A wrong sender port
     makes every later packet stale, a generation up to 24 hours ahead of the clock makes every
     later manifest from a sender that names generations by the clock a no-op until then (a
     larger one is refused), and repeated manifests can use up the sender ports that replay loops need;
   - change the feed controls that the external sender follows;
   - start or finish latency captures;
   - trigger a full-day `tcpreplay` burst through the privileged replay controller.

   The defaults bind to `127.0.0.1`, but with host networking, loopback is shared by every local
   account and host-network container. A browser on the same host, or on a machine with an SSH
   tunnel, can also reach it, including through DNS rebinding. Binding to all interfaces
   (`--host 0.0.0.0` or `WEB_HOST=0.0.0.0`) exposes these endpoints to the network. The
   documented external-sender setup in `deploy/igx_thor/README.md` does this, because the sender
   announces each generation through `POST /manifest`; run it only on a trusted, isolated segment,
   or tunnel the sender's `--control-url`. With `HOLOSCAN_FINANCE_FAULTS=1`, which `igx_run.sh`
   and `holoscan run` forward from the caller's environment, `/debug/hold_gil` holds the Python
   interpreter lock for up to 10 seconds per request. It exists for fault-injection tests only and
   must never be enabled in a deployment.

4. **Resource exhaustion.** The receive rings are bounded by packet count:
   - the GPU paths use 1,048,576 device slots of 2 KiB by default (2 GiB), configurable up to
     4,194,304 (8 GiB);
   - the host ring holds 65,536 packets by default and up to 10,000,000, each up to 64 KiB on the
     kernel-socket path.

   Overload is counted as ring drops, or shows as NIC drops and sequence gaps. Some memory is
   sized by inputs, however:
   - the per-symbol state and the covariance matrices, which grow with the square of the symbol
     count (about 0.8 GB each at the 10,000-symbol manifest limit);
   - latency captures, which keep one record per completed calculation for at most 600 seconds;
   - packet captures, read whole into host memory and, by the replay source, into GPU memory.

   A flood of control-plane requests, or a sustained feed above the measured capacity, degrades
   the run's latency and loss figures.

5. **Code execution through writable caches or build inputs.** The receiver loads the
   shared object it finds in the native cache directory, and CuPy loads kernels from its cache.
   Anyone who can write to those directories, or who controls `CXX`, `PATH`, `CUDA_PATH`,
   `DAQIRI_ROOT`, `HOLOSCAN_FINANCE_PYBIND11_INCLUDE` or the module search path of the receiving
   process, can run code inside the receiver. The cache key covers the source, compiler, flags and
   library versions. It protects against stale builds, not against tampering.

6. **Privileged launchers modifying the host.** The Thor launchers run privileged containers that
   enter the host's namespaces and change NIC, IRQ, power-management, NetworkManager and systemd
   state, and they run code from the mounted checkout (`replay_control.py`, `irq_isolate.py`) and
   the `tcpreplay` binaries as root. Anyone who can modify these scripts, the checkout, the images
   they name (`RECEIVER_IMAGE`, `MST_IMAGE`), `TCPREPLAY_DIR`/`TCPREPLAY_LIBS` or their environment
   effectively has root on the host. Run them only on a dedicated test port. With PHY loopback
   (`LOOPBACK=1`), `igx_run.sh` first refuses a port that has a global IP address, already has
   a neighbor entry for the peer, does not belong to `NIC_PCIE`, or whose PHY loopback cannot be
   verified as off. The benchmark scripts under `gpu_ingest_bench/` perform none of these checks.
   Never run any of them against a management interface.

7. **Supply-chain compromise of the images and tooling.**
   - `deploy/igx_thor/Dockerfile` builds DAQIRI from GitHub at a pinned commit (`DAQIRI_REPO` and
     `DAQIRI_REF` are overridable build arguments) and DPDK from the release tarball on
     fast.dpdk.org, which the build checks against a pinned SHA-256 (`DPDK_SHA256`; a different
     `DPDK_VERSION` needs its own hash). DPDK publishes no checksum next to the tarball, so the
     pinned value was computed from a download over TLS and protects against a later
     substitution, not against a tarball that was already malicious when it was pinned. It
     installs rdma-core from the DOCA apt repository: the build fails unless the downloaded
     signing key carries the pinned fingerprint (`DOCA_GPG_FINGERPRINT`, the Mellanox Signing Key
     v3). apt packages are not version-pinned: the DOCA repository is pinned by version, the
     Ubuntu packages (compilers, meson, libraries) come from the base image's current archive.
   - Both Dockerfiles, `scripts/dev5.sh`, `scripts/dev5cpu.sh`, the CI `unit-sdk` job and
     `[tool.holoscan.base-images]` name the Holoscan SDK image by tag (`v5.0.0-cuda13`), and a tag
     can be re-pointed. For reproducible builds pass a digest through
     `--build-arg BASE_IMAGE=...@sha256:...` (dev image) or
     `--build-arg HOLOSCAN_IMAGE=...@sha256:...` (receiver image).
   - `scripts/dev5.sh` and `dev5cpu.sh` install the pinned test tooling into the
     repository-local user base `.cache/pyuser5`, which later runs reuse; anyone who can write
     that directory controls code in those runs.

   A substituted tag, repository or package index would execute attacker-controlled code in
   developer, CI and receiver environments.

8. **Untrusted contributor code on self-hosted GPU CI.** The GitHub-hosted jobs in
   `.github/workflows/ci.yml` run on pushes to `main` and on pull requests targeting `main`, with a
   read-only `GITHUB_TOKEN` and no secrets. The GPU job targets a self-hosted runner with Docker
   access and the GPUs passed into its container. It runs only on pushes to `main`, and only when
   the repository variable `ENABLE_GPU_CI` is `true`. Do not relax that job's
   `github.event_name == 'push'` condition and do not add `pull_request_target` or `workflow_run`
   triggers. Its Dockerfile, requirements and tests come from the checked-out commit, so either
   change would execute contributor code on persistent, NVIDIA-managed hardware.

9. **Exposure of input data and derived results.** The examples use generated synthetic data.
   `.gitignore` keeps generated inputs and run artifacts out of Git, and `.dockerignore` keeps
   `data` out of the image build contexts. Access to input files, including rewritten capture
   copies in `FULLDAY_WORK`, otherwise relies on filesystem permissions. The dashboard's
   `/snapshot` serves prices and analytics to any client that can reach the port, and run
   reports and logs contain counts and timings. Restrict these files and endpoints to their
   intended users when adapting the module to other inputs.

10. **Shared-memory topics.** With `--topics`, risk summaries are published over HoloMQ
    (iceoryx2) shared memory. Any process that shares the IPC namespace and its `/dev/shm` can
    subscribe; the endpoint catalog names the endpoint but is not an access control.
    - The Thor launchers, `scripts/dev5.sh` and the CI GPU job give their containers a private
      IPC namespace.
    - `holoscan run` (holoscan-cli 5.0.0) starts containers with `--ipc=host`, host networking
      and `/dev` bind-mounted. This project's `docker-run-args` mounts a private tmpfs at
      `/dev/shm`, so the topics are not created in the host's `/dev/shm`.

    Do not add `--ipc=host` or a host `/dev/shm` mount to the launchers, and do not remove that
    tmpfs. The default catalog path is under `/tmp`; on a shared host, pass `--topics-catalog`
    with a private directory.

### Critical Security Assumptions

The following are assumed to be handled by other layers; the module does not enforce them.

- **The market-data network is trusted and isolated.** The UDP feed and the ConnectX test port
  are assumed to sit on a dedicated segment, such as PHY loopback or a private link, that only
  trusted senders can reach. The module checks structure and sequence, not origin.
- **The dashboard and replay-controller ports are reachable only by trusted users.** Use the
  `127.0.0.1` defaults and an SSH tunnel, on a host whose local accounts and host-network
  containers are trusted. Any wider exposure must add authentication, TLS, a Host-header
  allowlist, request-size limits and access control in front of them.
- **The consuming application and its configuration are trusted.** Command-line flags, operator
  Params and the objects passed to operators are validated for type and range, not for
  authorization. There is no sandbox around operator code or parameters.
- **Data provenance and filesystem permissions are enforced by the deployer.** Tapes, captures,
  the native and CuPy caches, `.cache/pyuser5`, and the repository checkout
  mounted into the containers are assumed to be writable only by trusted principals.
- **Message confidentiality and integrity are out of scope in transit.** The UDP feed, the HTTP
  control plane and HoloMQ topics are unencrypted and unauthenticated. A deployment beyond one
  trusted host or segment must add its own authenticated, encrypted transport.
- **No authentication, authorization, rate limiting, or audit trail is provided.** Any identity,
  access control, quota or security-relevant audit logging must be implemented by the deployer.
- **No secrets or cryptographic material are handled here.** The repository contains no
  credentials, keys or key management, and reads no security-relevant secrets at run time.
- **Availability and recovery are the deployer's responsibility.** Bars, rolling windows and
  latest marks are held in memory with no checkpointing, durable log or exactly-once guarantee.
  A restart or a new generation starts them over.
- **The platform stack is trusted and patched.** The deployer supplies a patched OS, Python 3.12
  or later, Holoscan SDK 5.0.0 or later (the `v5.0.0-cuda13` image), the GPU driver and CUDA 13
  runtime, DAQIRI and DPDK, rdma-core, and the container runtime. The module does not verify the
  integrity of the SDK, the base images or the installed packages.
- **The host OS enforces process isolation and file permissions.** Unprivileged users cannot
  modify the installed package, the checkout, the data files or the caches used by a running
  receiver; no untrusted local process can bind the receive port (the kernel-socket receiver sets
  `SO_REUSEADDR`); and the privileged launchers run only for trusted operators.
- **This module is not a production risk control.** As stated in the README, it is a synthetic
  replay demonstration with live computation. Consumers are assumed to independently validate
  calculations, data sources and controls, and not to rely on it as a regulatory, accounting, or
  risk-management safeguard.

## Supported Versions

This project is in early development. Security fixes are applied to the `main` branch and
released in the next version; earlier pre-release versions are not maintained separately.

| Version | Supported |
| --- | --- |
| `main` (0.2.x, Holoscan SDK 5) | Yes |
| Older pre-releases | No |

## Scope

In scope for a security report:

- The distributed Python package under `python/holoscan_finance/`, including the native drain
  source and the CUDA kernel sources it compiles at run time.
- The example applications under `examples/` and the operator catalog under `operators/`.
- Packet parsing, input validation and resource handling in the receive sources, decoders, bars,
  analytics, dashboard and control plane.
- The deployment launchers, benchmark scripts and Dockerfiles under `deploy/igx_thor/`, and the
  build, packaging and CI configuration in this repository that affects the integrity of released
  artifacts or of the hosts they run on.

Out of scope here (report to the owning project instead):

- Vulnerabilities in the Holoscan SDK, DAQIRI, CUDA, GPU drivers, or container base images. Use
  the same NVIDIA reporting channels above and name the affected product.
- Vulnerabilities in DPDK, rdma-core, or third-party Python packages and developer tooling, such
  as CuPy, FastAPI, uvicorn, numba or the Holoscan CLI.
- The financial correctness or suitability of the reference calculations, which is a functional
  matter documented in the README's scope and limitations.

## Dependency Security

The wheel declares one runtime dependency, NumPy, and optional extras: `gpu` (CuPy), `analytics`
(SciPy, threadpoolctl), `dashboard` (FastAPI, uvicorn, AnyIO), `bench`
(numba) and `test` (pytest, pytest-timeout, httpx). These carry minimum versions only in
`pyproject.toml`. The Holoscan SDK is never a pip dependency: it comes from the SDK container
image or its Debian packages. DAQIRI's Python binding comes from the DAQIRI installation, and
pybind11 headers for the native drain come from the system (`pybind11-dev`).

The direct development, CI and image dependencies are pinned to exact versions, without hashes,
in `requirements-cli.txt` (`holoscan-cli`) and `requirements-dev.txt`; their transitive
dependencies are not pinned. The CI `unit-pure` job also installs NumPy and pytest unpinned, and
isolated wheel builds resolve their build backend (`setuptools>=77`) from PyPI. GitHub Actions
are referenced by major-version tag and pre-commit hooks by release tag, both mutable references
rather than commit SHAs. The receiver image pins DAQIRI by commit, DPDK by release and checksum,
and the DOCA repository by version and signing-key fingerprint (see threat 7 for what is not
verified).
