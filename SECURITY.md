# Security Policy: Holoscan Finance

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
  `pyproject.toml`), plus how the code was obtained: wheel, Debian package, or built from source.
- Type of vulnerability (for example: input-validation flaw, resource exhaustion, insecure
  default, information disclosure, supply-chain issue).
- Step-by-step reproduction instructions, including the market-data input or operator
  configuration used.
- Proof-of-concept code or a minimal reproducer, if available.
- Impact assessment covering confidentiality, integrity, availability, and scope.

### What to Expect

NVIDIA's Product Security Incident Response Team (PSIRT) will acknowledge receipt, validate the
issue and its severity, coordinate remediation with engineering, and publish a security bulletin
where appropriate. NVIDIA does not operate a public bug bounty program for this project, but
externally reported issues are acknowledged under our coordinated vulnerability disclosure
policy.

## Security Architecture & Context

`holoscan-finance` is a self-contained [Holoscan](https://github.com/nvidia-holoscan) module for
building streaming financial-services applications. It provides immutable market and portfolio
message contracts (`python/holoscan/finance/models.py`), a CSV loader
(`python/holoscan/finance/io.py`), three stateful operators (`MarketDataReplayOp`,
`RollingStatisticsOp`, `PortfolioRiskOp`), and one offline reference application
(`applications/intraday_risk_pipeline`). It is pure Python; beyond the separately installed
Holoscan SDK it uses only the standard library.

This software operates at the **library / SDK** level. It is imported into a consumer's Holoscan
application and runs inside that application's process. It opens no network listener, exposes no
service endpoint or RPC surface, uses no database, and implements no authentication, session, or
cryptographic functionality. Its primary security responsibility is the **integrity and validity
of the financial data flowing through the operator graph** — market ticks, derived rolling
features, position marks, and risk-limit breaches — and the robustness of its CSV ingestion and
rolling-state code paths against malformed or hostile input.

**Repository Exposure Classification:** Public.
Basis: the `origin` remote is on github.com under an NVIDIA Holoscan organization and the project
is released under Apache-2.0; this document is written to public-safe detail.

**Service Exposure Classification:** External / Regulated (medium confidence).
Basis: externally distributed Apache-2.0 module published as a platform-independent wheel and a
Debian package for financial-services workloads; the repository contains no live service,
credentials, or customer data, but the artifact is consumed outside NVIDIA and its output feeds
portfolio valuation and risk-limit decisions.

Key boundaries and interfaces:

- **Untrusted file input:** `load_market_ticks_csv` is the only file-parsing entry point. It
  requires the `timestamp`, `sequence`, `symbol`, `price`, and `currency` columns, rejects
  duplicate column names, and validates ISO 8601 timezone-aware timestamps, non-negative integer
  sequences, positive finite `Decimal` prices, three-letter currency codes, strictly increasing
  sequences, and non-decreasing timestamps.
- **Caller-supplied configuration:** operator parameters such as `ticks`, `window_size`,
  `min_periods`, `positions`, and `limits` are type- and range-checked, but are trusted as to
  origin — the module has no notion of an authenticated or authorized caller.
- **In-process messages:** operators exchange frozen dataclasses within one process. There is no
  serialization format, network transport, or IPC layer in this repository.
- **Application-owned sinks:** logging, alerting, persistence, and execution belong to the
  consuming application. The example `ConsoleRiskSinkOp` lives in the application, not in the
  distributed package.
- **Developer, build, and CI tooling:** the `./holohub` wrapper, `Dockerfile`, CMake and
  scikit-build-core packaging, `.github/workflows/ci.yml`, and the maintenance scripts under
  `.github/workflows/scripts/` are not part of the runtime, but they are part of the trust chain
  that produces the distributed artifacts.

### Threat Model

The following scenarios represent the primary security concerns for this project, including its
auxiliary build and tooling paths:

1. **Market-data tampering driving incorrect risk output:** `load_market_ticks_csv` validates
   structure and ordering but performs no authenticity or integrity check — there is no signature,
   checksum, or provenance verification of the CSV. A party who can write the data file consumed
   by an application (for example a deployment's data directory, or the checked-in
   `applications/intraday_risk_pipeline/data/synthetic_market_data.csv`) can shift prices so that
   `PortfolioRiskOp` reports understated exposure or suppresses `GROSS_EXPOSURE`,
   `ABS_NET_EXPOSURE`, and `UNREALIZED_LOSS` breaches, while every value still passes validation.

2. **Memory exhaustion through unbounded state and input size:** `RollingStatisticsOp` keeps a
   per-symbol `_SymbolState` in a dictionary that is never evicted; the price and log-return
   deques are bounded by `window_size`, but the number of symbols is not bounded. Because symbols
   come from the input stream, a high-cardinality or oversized feed grows state without limit. In
   addition, `load_market_ticks_csv` and `MarketDataReplayOp` materialize the entire tick sequence
   in memory before replay, so input size directly determines resident memory in a long-running
   pipeline.

3. **Supply-chain compromise of the developer and build bootstrap:** the `./holohub` wrapper and
   the `Dockerfile` install `holoscan-cli>4.2.0` with `--pre` from an additional package index,
   and the container base image is referenced by a mutable tag rather than a digest. The build
   also installs unpinned `build`, `scikit-build-core`, and test tooling. A compromised or
   substituted package or image tag would execute attacker-controlled code in developer, CI, and
   packaging environments, and could affect the published wheel or Debian package.

4. **Environment-driven injection in the developer CLI wrapper:** `./holohub` expands
   `HOLOSCAN_CLI_INSTALL_ARGS` unquoted into its `pip install` invocation and prepends
   `HOLOSCAN_CLI_SOURCE` to `PYTHONPATH`. In an environment where those variables can be set by
   another party — a shared shell profile, a devcontainer definition, or a CI variable — the
   wrapper can be steered into installing unintended packages or importing a substituted
   `holoscan_cli` module before the intended one.

5. **Disclosure of portfolio detail through application logs:** the reference
   `ConsoleRiskSinkOp` logs entire `PortfolioSnapshot` values at `INFO` and every `RiskBreach` at
   `WARNING`. Snapshots carry position quantities, average prices, market values, gross and net
   exposure, and unrealized P&L. Applications that copy this pattern into a shared or
   multi-tenant environment can leak confidential position and performance data into logs with
   broader read access than the pipeline itself.

6. **Untrusted contributor code on self-hosted GPU CI:** `.github/workflows/ci.yml` GitHub-hosted
   jobs still run on `pull_request`. The optional GPU job (enabled by the `ENABLE_GPU_CI`
   repository variable) targets a self-hosted runner with all GPUs passed into the container, but
   is restricted to `push` events so it runs only after merge to `main`. Do not reintroduce a
   `pull_request` trigger for that job: build and test commands come from the checked-out branch
   and would otherwise execute contributor-controlled code on persistent, NVIDIA-managed hardware.

7. **Shell interpolation in repository maintenance scripts:** `gitutils.py` executes commands
   through `subprocess.check_output(cmd, shell=True)`, and `check_copyright.py` builds git
   commands from branch and path values before running them in pre-commit and CI. Repository
   content or reference names that reach those command strings are interpreted by the shell rather
   than passed as argument vectors.

### Critical Security Assumptions

The following are assumed to be handled by other layers; the module does not enforce them.

- **The consuming application and its operator configuration are trusted.** Positions, limits,
  window parameters, and tick sequences are validated for type and range, not for authorization.
  There is no sandbox around operator code or parameters.
- **Data provenance and filesystem permissions are enforced by the deployer.** Market-data CSV
  files, and the directories containing them, are assumed to be writable only by trusted
  principals; the module verifies structure, not origin.
- **Message confidentiality and integrity are out of scope in transit.** Messages are in-process
  Python objects. A distributed or cross-process deployment must add its own versioned
  serialization plus an authenticated, encrypted transport; nothing in this repository provides
  TLS, signing, or replay protection.
- **No authentication, authorization, rate limiting, or audit trail is provided.** Any identity,
  access control, quota, or security-relevant audit logging must be implemented by the consuming
  application.
- **No secrets or cryptographic material are handled here.** The repository contains no
  credentials, keys, or key management, and reads no security-relevant environment variables at
  runtime; secret handling is assumed to occur upstream.
- **Availability and recovery are the deployer's responsibility.** Operator state is held in
  memory with no checkpointing, durable log, or exactly-once guarantee, so process restart loses
  rolling windows and latest marks.
- **The platform stack is trusted and patched.** A patched OS, Python 3.10-3.13, Holoscan SDK
  4.3.0 or later, GPU driver and CUDA runtime, and container runtime are supplied by the deployer;
  the module does not verify the integrity of the SDK, base images, or installed packages.
- **The host OS enforces process isolation and file permissions**, so unprivileged users cannot
  modify the installed package, data files, or configuration used by a running pipeline.
- **This module is not a production risk control.** As stated in the README, it is a technical
  reference implementation. Consumers are assumed to independently validate calculations, data
  sources, and controls, and not to rely on it as a regulatory, accounting, or risk-management
  safeguard.

## Supported Versions

This project is in early development. Security fixes are applied to the `main` branch and
released in the next version; earlier pre-release versions are not maintained separately.

| Version | Supported |
| --- | --- |
| `main` (0.1.x) | Yes |
| Older pre-releases | No |

## Scope

In scope for a security report:

- The distributed Python package under `python/holoscan/finance/` and the operators under
  `operators/`.
- Input validation and resource handling in the CSV loader, message models, and operator state.
- Build, packaging, and CI configuration in this repository that affects the integrity of released
  artifacts.

Out of scope here (report to the owning project instead):

- Vulnerabilities in the Holoscan SDK, CUDA, GPU drivers, or container base images. Use the same
  NVIDIA reporting channels above and name the affected product.
- Vulnerabilities in third-party developer tooling installed by the build, such as the Holoscan
  CLI or Python build backends.
- The financial correctness or suitability of the reference calculations, which is a functional
  matter documented in the README's scope and limitations.

## Dependency Security

The runtime package declares no third-party dependencies: it uses the Python standard library and
the separately installed Holoscan SDK. Development and CI tooling — pre-commit hooks, `ruff`,
`codespell`, `markdownlint`, `cmakelint`, `pytest`, and the Python build backend — is pinned by
revision in `.pre-commit-config.yaml` where the hook framework supports it. The Holoscan CLI
bootstrap in `./holohub` and the `Dockerfile` are version-ranged rather than pinned; deployments
that require reproducible builds should pin those versions and reference the container base image
by digest.
