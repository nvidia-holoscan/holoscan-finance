# Contributing to holoscan-finance

Contributions to the operators, libraries, examples, tests and documentation are welcome.
For running the examples, start with the [README](README.md). This guide covers changing
the module itself. Run the commands below from the repository root unless noted otherwise.

## Set up a development environment

Use Python 3.12 or newer. The module targets Holoscan SDK 5.0.0 with CUDA 13 and pins
`holoscan-cli` 5.0.0 in [requirements-cli.txt](requirements-cli.txt).
[requirements-dev.txt](requirements-dev.txt) pins the development dependencies used in CI.

### CPU development without the SDK

Library, control-plane and tooling changes can be developed without Holoscan or a GPU:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -c requirements-dev.txt -r requirements-cli.txt \
    -e ".[analytics,dashboard,test]" pre-commit
pre-commit install
python3 -m pytest -m "not sdk and not gpu and not daqiri"
```

The editable install uses the source under `python/`. Installing this module or the CLI
does not install the Holoscan SDK. Operator and graph work needs the SDK environment below.

### SDK and GPU development

With the [README prerequisites](README.md#quick-start) satisfied and the CLI installed:

```bash
holoscan run-container
```

This builds the root [Dockerfile](Dockerfile) and opens a container with the SDK, CLI and
development dependencies, with the checkout mounted. Inside that container, use `--local`
to keep CLI commands in the current environment:

```bash
holoscan test --local --no-xvfb
holoscan run equity_analytics smoke --local
holoscan run market_data_bench smoke --local
```

When running Python files directly, preserve the SDK's existing import paths:

```bash
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$PWD/python"
```

The root image does not include DAQIRI. NIC changes require the separate
[IGX Thor environment](deploy/igx_thor/README.md). The helpers `scripts/dev5.sh` and
`scripts/dev5cpu.sh` are specific to the Thor development host: they hard-code CPU
placement, and `dev5.sh` selects GPU 1. Use them only with that setup.

## Find the code

| Location | Responsibility |
| --- | --- |
| `python/holoscan_finance/{marketdata,equity,ingest,analytics}/` | SBE codec and CUDA kernels, tapes, receivers, quote bars and risk calculations. |
| `python/holoscan_finance/{control,telemetry,validation}/` | Feed state, results, instrumentation and correctness checks. |
| `python/holoscan_finance/operators/` | Holoscan adapters around the library processors. |
| `python/holoscan_finance/{app,dashboard,tools}/` | Shared application setup, web UI and command-line tools. |
| `operators/<name>_op/` | Operator documentation and catalog metadata. |
| `examples/<name>/` | Application composition, README and CLI presets in `metadata.json`. |
| `tests/python/` | Library, control, GPU, operator, graph, application, end-to-end, packaging and deployment tests. |
| `deploy/igx_thor/`, `docs/` | Hardware launchers and detailed design, methodology and validation guides. |

The canonical Python package is `holoscan_finance`. Its `holoscan.modules` entry point
also exposes it as `holoscan.finance` when installed in an SDK environment. CMake stages
the package and entry point for `holoscan build`, `run` and `test`; `PYTHONPATH=python`
alone does not register that alias.

The operators are Python, with GPU work implemented in CUDA kernels through CuPy.
The DAQIRI path also has a C++ drain helper, compiled on first use against the installed
DAQIRI and pybind11. CMake stages the package and registers tests; it does not compile
that helper.

## Make a change

Keep changes focused and put reusable processing logic in the library layer. Read the
[architecture guide](docs/ARCHITECTURE.md) before changing the graph or operator lifecycle.

- Keep the library layer independent of Holoscan. Use relative imports within the
  package and `holoscan_finance` imports in examples and tests.
- Keep operator `setup()` free of device and file I/O. Acquire runtime resources in
  `start()` and release them in `stop()`, including after a partial start. Preserve
  output retry behavior when a downstream queue is full.
- Keep examples as compositions of reusable operators. Set queue policies and lane
  placement explicitly, and update plan rules when the graph's guarantees change.
- Add regression coverage for behavior changes at the relevant layer. Update operator
  docs, example presets and validation instructions when their contracts change.

For a new operator, follow [Adding an operator](docs/ARCHITECTURE.md#adding-an-operator):
add the library processor and adapter, export the class, add its catalog README and
`metadata.json`, register it in the root [metadata.json](metadata.json), and add operator
harness and graph tests. For a new example, add its composition, README and mode metadata,
and register it under `module.subprojects.applications` in the root metadata.

## Run checks

Run formatting, spelling, Markdown and metadata checks before submitting:

```bash
pre-commit run --all-files
```

`holoscan lint` runs the same hooks. Ruff handles Python linting and formatting;
the configured hooks may fix files, so review their changes and rerun if needed.

Choose tests for the code and hardware you changed:

| Environment | Command |
| --- | --- |
| Python environment above, no SDK or GPU | `python3 -m pytest -m "not sdk and not gpu and not daqiri"` |
| SDK image without a GPU driver | `scripts/with_stub_libcuda.sh python3 -m pytest -m "sdk and not gpu and not slow"` |
| SDK container with a GPU | `holoscan test --local --no-xvfb` |
| Additional slow tests in that container | `python3 -m pytest -m "slow and not timing"` |

The stub command is for CPU tests inside the SDK image only; it cannot run GPU work.
The `sdk`, `gpu` and `daqiri` markers skip tests when their runtime is unavailable.
Inspect the skipped-test summary: a passing CPU run does not verify GPU or NIC changes.
CTest runs `python.unit` and `python.gpu` and excludes `slow` tests. The `timing` marker
also implies `slow`; run timing tests separately on an otherwise idle host.

For graph or GPU pipeline changes, run both smoke commands from the SDK setup above.
For decoder, quote-bar, analytics or performance changes, use the relevant checks in
[docs/VALIDATION.md](docs/VALIDATION.md). Record the input, hardware, versions and settings
with performance reports; do not infer NIC performance from software replay.

[CI](.github/workflows/ci.yml) runs lint, CLI discovery, CPU library and SDK tests, and
wheel checks on pull requests. GPU checks run after pushes to `main` only when the
repository's `ENABLE_GPU_CI` variable is enabled and a GPU runner is available.
Include your own GPU or NIC results when the change depends on them.

## Check packaging

For package layout, entry-point or bundled-resource changes, build a wheel:

```bash
python3 -m pip install -c requirements-dev.txt build
python3 -m build --wheel -o dist
```

The CLI equivalent, inside the SDK container, is:

```bash
holoscan package --pkg-generator WHEEL --local
```

The first command writes to `dist/`; the CLI writes to `build/dist/`. Install the wheel
in a fresh environment and test imports from outside the checkout, as the CI `wheel`
job does. Verify bundled schemas, dashboard assets and native sources. Check
`holoscan.finance` imports in an SDK environment as well when changing module registration.

## Submit a pull request

- Explain the problem, resulting behavior and relevant test results. Name any hardware
  checks you could not run, and include reproduction steps for bug fixes.
- Update documentation alongside changes to flags, defaults, ports or supported inputs.
  Keep generated captures, caches and benchmark output out of Git.
- Preserve existing license headers and use the repository's SPDX header format for new
  source files. Contributions use the [Apache-2.0 license](LICENSE).
- Sign off every commit as described below. Commits without sign-off are not accepted.

### Signing off your work

By signing off, you certify that you have the right to submit the contribution under
the project's license, as specified in the
[Developer Certificate of Origin 1.1](https://developercertificate.org/).

```bash
git commit -s -m "Describe the change"
```

This adds `Signed-off-by: Your Name <your@email.com>` to the commit message using your
Git identity. Read the DCO before signing; this sign-off is distinct from cryptographic
commit signing.
