# Porting from Holoscan SDK 4.3 to 5.0

What changed between the Holoscan 4.3 version of this repository (commit `43bf4e1`) and the
5.0 port, and every place where the running system behaves differently. The design goal
was to keep every published 4.3 number reproducible and every invariant intact: the CUDA
kernels, their compile flags and the native drain are byte-identical, the GPU path keeps
its batching and buffer sizes, and the dashboard keeps its HTTP contract.

## Packaging and layout

| 4.3 | 5.0 |
| --- | --- |
| Code in `python/holoscan/marketdata` and `python/holoscan/equities`, spliced into the SDK's namespace by a `holoscan.__path__` hook (`scripts/sitecustomize.py`, the CMake dev hook) | One package, `python/holoscan_finance`, published by Holoscan as `holoscan.finance` through the `holoscan.modules` entry point. 5.0 resolves `holoscan.<name>` only from entry points; a physical `holoscan/<name>` directory outside the SDK no longer imports |
| Operators in `operators/<name>/<name>.py`, copied into the package by CMake | Operators in `holoscan_finance/operators/`; `operators/<name>_op/` holds only the catalog entry (`metadata.json`, README) |
| HoloHub CMake (`add_holohub_*`, `cmake/`, `pkg/`), a `./holohub` wrapper, DEB and wheel | `holoscan-cli` 5.0.0 finds the module from `metadata.json` and `pyproject.toml` (`[tool.holoscan]`); a `LANGUAGES NONE` CMake file only stages the package and registers the tests; wheel only (`holoscan package --pkg-generator WHEEL`) |
| `scripts/runenv` with a pip-installed SDK | The SDK is not on PyPI for 5.0 (PyPI stops at 4.6): run in the `holoscan:v5.0.0-cuda13` image, through `holoscan run`/`holoscan test` or the dev image (`Dockerfile`) |
| `metadata.json` with `urn:holohub:` schemas, SDK 4.3.0 | `urn:holoscan:` schemas, SDK 5.0.0, ten operators and two applications, validated by `scripts/validate_metadata.py` (also a pre-commit hook) |
| `pytest.ini`; a conftest that installed a 4.x `holoscan` stub | `[tool.pytest.ini_options]`; markers `sdk`, `gpu`, `daqiri`, `slow`, `timing` with auto-skip; no stub (it would hide the real 5.0 SDK) |
| `native/daqiri_frames.cpp` (`burst_frames`) | Removed: it only sped up the Python fallback drain. `native/daqiri_drain.cpp` is unchanged |

Where the 4.3 modules went:

| 4.3 | 5.0 |
| --- | --- |
| `marketdata/mitch.py`, `wire.py`, `bars.py`, `gpu_mitch.py`, `gpu_bars.py`, `covariance.py` | `holoscan_finance.marketdata.{mitch_codec, records, bars, gpu_mitch, gpu_bars, covariance}` (verbatim; `bars` and `gpu_bars` gain read-only helpers) |
| `marketdata/daqiri.py`, `native/` | `holoscan_finance.ingest.{daqiri, native}`; `strip_udp_datagram` → `marketdata.framing` |
| `equities/mitch.py`, `data.py` | `holoscan_finance.nyq.{mapping, tape}` |
| `equities/stream.py` | split by role: manifest checks → `nyq.manifest`; host and GPU decode → `nyq.host_decoder`, `marketdata.gpu_decoder`; sequence rules → `marketdata.sequence`; directory, feed control, full day, results, status → `control/`; heartbeat → `telemetry.stalls`; HTTP → `dashboard/`; the graph → `applications/equity_analytics/equity_analytics.py`; `main` → `app/` |
| `equities/host_sources.py` | `ingest.host_receivers` + `ingest.accounting` + `operators.packet_sources`, with decode in `operators.mitch_decode` |
| `operators/gpu_mitch_source_op` | `operators.gpu_mitch_source` (adapter) + `ingest.gpu_ingest` (accounting, staging, decode, release) + `ingest.drains` |
| `operators/gpu_quote_bars_op`, `gpu_rolling_risk_op` | `operators.quote_bars` (`QuoteBarsOp`, GPU or CPU backend) + `marketdata.bar_builder`; `operators.rolling_risk` (`RollingRiskOp`) + `analytics.rolling` |
| `equities/analytics.py`, `capture.py`, `warmup.py` | `analytics.{workers, agreement}`, `telemetry.capture`, `marketdata.warmup` |
| `equities/{replay, data, bench_replay, measure, compare_runs, decode_bench, gpu_validate, cpu_mitch}.py` | `holoscan_finance.tools.{replay_sender, prepare_tape, bench_replay, measure, compare_runs, decode_bench, gpu_validate, cpu_mitch}`, same flags |

## API mapping as applied

| Holoscan 4.3 (as this repo used it) | Holoscan 5.0 (as the port uses it) |
| --- | --- |
| `from holoscan.core import Operator, OperatorSpec, Application, ...` | `import holoscan as hs`; `hs.Operator`, `hs.Graph`. `holoscan.core`, `holoscan.gxf`, `holoscan.conditions` and `holoscan.schedulers` do not exist |
| `Operator.__init__(fragment, *args, x=..., name=...)` | class-level `hs.Param`s; `__init__(self)` takes no arguments; `g.op(Op, "name", x=...)`. Constructor-only objects (locks, callbacks, state) become `Ref` parameters |
| `setup(spec)`: `spec.input("events")`, `spec.output("tick")` | class attributes `events = hs.Input("events", hs.pyobject)`; port name equals attribute name. Optional ports are declared in `setup()` behind a bool `Param` |
| `spec.input("window", size=IOSize(1), policy=POP)` | the edge: `g.add_flow(bars.window, risk.window, queue_policy=DROP_OLDEST, queue_depth=1)` (`edges.LATEST`) |
| `ConditionType.NONE` on outputs and inputs | `DROP_OLDEST` edges, or `BLOCKING` edges with hold-and-retry (`PendingEmits`); clock-paced inputs use `OnClock(...).input(port, hs.DrainAll())` |
| `PeriodicCondition(recess_period)` | `contract()`: `hs.Contract().trigger(hs.OnClock(period=...))` |
| `BooleanCondition.disable_tick()`, `stop_execution()` | no per-operator disable; finite runs end through a drained predicate and `session.request_stop()` |
| `CountCondition` in tests | `OperatorTestHarness` with `run_async`/`wait_for_output`/`request_stop` for unbounded sources |
| `compute(op_input, op_output, context)` | `compute(self, ctx)`; `self.port.receive_data()`, `self.port.selected_batch()`, `self.port.emit(value)` (one emit per port per call) |
| `Application.compose()` + `MultiThreadScheduler(worker_thread_number=4)` | `hs.Graph` built in `compose()`; concurrency is authored partitions ("lanes") with `ThreadPlacement`; there is no scheduler object and no deadlock auto-stop |
| `app.run()` in a worker thread; exceptions raised from it | `plan = hs.compile(g, options)`; `with hs.run_async(plan) as session`; `session.termination` and `session.diagnostics` (the dashboard shows the first error) |
| `_isolate_cores` for every thread | `ThreadPlacement` for lanes, a process mask before the run, `thread_mask` around `daqiri_init`, the 4.3 sweep after it |
| `import holoscan.core` in the Thor preflight | `import holoscan` and a check for `holoscan.compile` |

Not used, on purpose: Tensor ports (multi-field payloads, a cost per hop, stream mixing),
`LatestWithin` (it needs capture times and starves silently without them), runtime
parameter updates, operator restarts, embedded `hs.Graph` subclasses (a parent cannot
place an embedded graph's operators), `complete_after` and `request_completion`.

## Behavioural differences

1. **Results publish immediately.** A risk result is published on the analytics lane right
   after the analysis; 4.3's publisher polled every 5 ms. `receive_to_result_ms` is equal
   or up to 5 ms lower.
2. **Risk polls every 1 ms and enforces the minimum interval itself.** It keeps the newest
   window in a slot and starts an analysis when at least `--interval` (50 ms) has passed
   since the previous start, which reproduces 4.3's run-on-arrival timing. After a
   cross-check gate release, the next window runs within 1 ms; in 4.3 it took up to 50 ms.
   `--risk-poll-us` raises the poll period (the plan rule follows it).
3. **Bars reset at the first batch decoded after `POST /manifest`**, on the ingest lane;
   4.3 reset them inside the HTTP call. The snapshot hides counters of a stale generation,
   so nothing visible changes. A reset requested while no data flows takes effect when
   data arrives.
4. **Host `compare` mode runs the CPU job after the GPU job.** In 4.3 the two workers
   started together; now the CPU cross-check starts when the GPU result is ready. Only
   host-mode `cpu_receive_to_result_ms` grows (by the GPU job time); `cpu_ms`, `gpu_ms`
   and the speedup are measured inside the engines and are unchanged.
5. **The native drain is built before NIC initialization** (about 5 s cold), so the NIC is
   never up while the helper compiles.
6. **The dashboard reads host mirrors.** Counters are mirrored after every bars batch (one
   40-byte copy on the GPU path), so full-day totals stay exact; prices come from a
   symbol snapshot at most 250 ms old. 4.3 copied from the GPU on every HTTP request, under
   a lock shared with the bars operator.
7. **Canonical ingest names.** `--ingest udp|daqiri-host|gpudirect|replay`; `async`, `ano`
   and `gpu` (and the `--ano-*` flags, now `--nic-*`) are accepted for one release with a
   warning. `/snapshot` `network.ingest` keeps the 4.3 tokens (`async`, `ano`, `gpu`, plus
   `replay`); the new key `network.ingest_mode` carries the canonical name.
8. **The Python fallback drain has no `burst_frames`.** It reads each packet's address
   through the binding. The native-vs-Python drain A/B on 5.0 therefore compares against a
   weaker baseline than the published 4.3 A/B, which is labelled as a 4.3 measurement.
9. **Graph operator names** are `source`, `decode`, `bars`, `risk`, `publish`, `crosscheck`
   and `publish_comparison` (plus `risk_summary` with `--topics`).
10. **No `--ipc=host` on the receiver.** The Thor launchers give it a private IPC namespace
    with `--shm-size=4g`. A root Holoscan 5 process with `--ipc=host` leaves a root-owned
    shared-memory file in the host's `/dev/shm` that breaks later non-root runs.
11. **Decode and bars share the ingest lane.** In 4.3 they ran on separate scheduler
    workers; their Python was GIL-serialized anyway, but their CUDA work overlapped. The
    NIC-free throughput gate (≥ 2× line rate) guards this; moving `bars` into its own lane
    is a one-line fallback.
12. **Replay mode ignores `/feedrate`.** The replay pace is a command-line setting
    (`--replay-pps`); pause and seek apply only to the external sender.
13. **`--risk-deadline-us` counts late poll ticks**, which are almost always the ticks
    right after an analysis that overran the poll period. It is not a measure of window
    lateness; that is `receive_to_result_ms` minus the engine time.
14. **Spawn workers re-import the application's top-level imports** (Holoscan, FastAPI) once,
    when the pool starts in `start()`, as 4.3's `-m` entry point did. Work stays under
    `main()`, so nothing runs twice.

## Command lines

| 4.3 | 5.0 |
| --- | --- |
| `python -m holoscan.equities.stream --ingest async\|ano\|gpu --ano-pcie ...` | `python3 applications/equity_analytics/equity_analytics.py --ingest udp\|daqiri-host\|gpudirect\|replay --nic-pcie ...`, or `holoscan run equity_analytics <mode>` |
| (none) | `python3 applications/market_data_bench/market_data_bench.py` (`holoscan run market_data_bench <mode>`) |
| `python -m holoscan.equities.replay` | `python -m holoscan_finance.tools.replay_sender` |
| `python -m holoscan.equities.data` | `python -m holoscan_finance.tools.prepare_tape` |
| `python -m holoscan.equities.{bench_replay, measure, compare_runs, decode_bench, gpu_validate}` | `python -m holoscan_finance.tools.{bench_replay, measure, compare_runs, decode_bench, gpu_validate}` |
| `./scripts/runenv -m pytest` | `holoscan test`, or `pytest` in the 5.0 image (tiers in the README) |

New: `--plan-only`, `--plan-json`, `GET /plan`, `GET /healthz` and the `/snapshot.engine`
block; `--exit-when-drained`, `--report` and `--expect-windows` for headless runs;
`--ingest-cpus` and `--ingest-busy-poll` for placement; `--topics` for the holomq risk
summary.

## Deployment (IGX Thor)

- The receiver image is `deploy/igx_thor/Dockerfile`: the 5.0 image with DAQIRI and DPDK
  built from source at the pinned commit with the Thor steering patch (it replaces
  `Dockerfile.equities`, which extended the local 4.3 `holoscan-finance:ano` image).
  Default tag: `holoscan-finance:hsdk5-daqiri`.
- `igx_ano_run.sh` keeps `INGEST=ano|gpu` and also accepts `daqiri-host|gpudirect`; it runs
  the application script with `--nic-*` flags, passes the mode to the preflight, sets
  `CUDA_DEVICE_ORDER=PCI_BUS_ID`, and replaces `--ipc=host` with `--shm-size=4g`.
- `ano_python.sh` appends the checkout's `python/` to `PYTHONPATH`: the 5.0 image finds the
  SDK only through its own `PYTHONPATH`, which must stay first.
- The sender and replay-controller containers still run without the GPU runtime. In 5.0
  `import holoscan` needs libcuda, so the tools they run never import the SDK.
- `daqiri_init` runs with the calling lane's CPU mask saved and restored; the receive
  configuration names the DPDK engine explicitly (`"engine": "dpdk"`), which newer DAQIRI
  releases need for the steering patch to apply.
