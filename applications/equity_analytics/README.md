# equity_analytics

Replays licensed NYQ trades and quotes as MITCH over UDP into a Holoscan graph and
computes rolling historical volatility and correlation with CuPy/cuBLAS, with a
NumPy/SciPy worker computing the same windows for comparison.

```bash
./scripts/runenv applications/equity_analytics/equity_demo.py --ingest async
./scripts/runenv -m holoscan.equities.replay --data /licensed/prepared-nyq   # second terminal
```

Open <http://localhost:8080/>. `--ingest ano` and `--ingest gpu` need DAQIRI and a
ConnectX NIC; see [Thor deployment](../../deploy/igx_thor/README.md). Results and setup:
[README](../../README.md).
