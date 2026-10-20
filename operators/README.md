# Operators

A GPU market-data chain: packets go from the NIC into GPU memory and stay there through
decode, bars and analytics.

```text
GpuMitchSourceOp ──tick/events──▶ GpuQuoteBarsOp ──window──▶ GpuRollingRiskOp ──risk──▶
```

| Operator | Folder | What it does |
| --- | --- | --- |
| `GpuMitchSourceOp` | [gpu_mitch_source_op](gpu_mitch_source_op/README.md) | DAQIRI GPUDirect into GPU memory; CUDA payload gather and MITCH framing, pluggable GPU decoder |
| `GpuQuoteBarsOp` | [gpu_quote_bars_op](gpu_quote_bars_op/README.md) | Event-time quote-midpoint bars on the GPU, with feed sequence and generation handling; emits rolling windows of bar returns |
| `GpuRollingRiskOp` | [gpu_rolling_risk_op](gpu_rolling_risk_op/README.md) | Rolling volatility, covariance and correlation on the GPU (incremental cuBLAS), on the latest window at a bounded rate |

Each folder holds the operator module, its HoloHub `metadata.json` and `CMakeLists.txt`.
In the source tree they resolve as submodules of `holoscan.marketdata`; the CMake build
copies them flat into that package together with the modules they use (`mitch.py`,
`gpu_mitch.py`, `daqiri.py`, `wire.py`, `bars.py`, `gpu_bars.py`, `covariance.py`,
`native/`).

The application ([equity_analytics](../applications/equity_analytics/README.md), graph in
[stream.py](../python/holoscan/equities/stream.py)) runs this chain with `--ingest gpu`
and adds the MITCH-to-NYQ decoder, the dashboard and a CPU cross-check. Its host-memory
paths (`--ingest async | ano`: kernel UDP socket or DAQIRI into pinned host memory, with
Python decode, NumPy bars and analytics in separate processes) are application code in
[host_sources.py](../python/holoscan/equities/host_sources.py) and
[stream.py](../python/holoscan/equities/stream.py). They run without a ConnectX NIC and
are the comparison for the GPU path.
