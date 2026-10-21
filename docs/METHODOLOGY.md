# Data, wire format, calculations and measurement

## Data

The source is LSEG/Refinitiv NYQ equity trade/quote Parquet for April 1, 2025
(**historical LSEG/Refinitiv equity data — NYQ dataset**). LSEG defines NYQ as
consolidated instruments primarily listed on NYSE; NYS denotes the NYSE venue. See
[LSEG's NYQ/NYS definitions](https://community.developers.lseg.com/discussion/103024/difference-between-idn-exchange-id-nys-and-nyq).
The NYQ label does not certify this extract's quote coverage or NBBO semantics, and
the prepared tape does not keep the contributor field.

Preparation (`python -m holoscan_finance.tools.prepare_tape`) keeps Quote, Trade and
Mkt. Condition records from the regular 09:30–16:00 New York session (IANA timezone),
sorted by source timestamp, internal sequence, instrument ID and kind. Missing numeric fields stay NaN;
values stay float64; instrument names go to a directory.

## Wire format: MITCH replay

Sender and receivers use the binary layouts and unit framing of
[MIT303 issue 11.9](https://docs.londonstockexchange.com/sites/default/files/documents/mit303issue119.pdf).
The UDP payload starts with the MITCH unit header; there is no other envelope or
hidden timing field. This is historical NYQ data re-encoded for replay, not exchange
packets: LSE retired its MITCH service in 2019.

| NYQ input | MITCH message | Replay interpretation |
| --- | --- | --- |
| Market timestamp | Time (`T`) plus message nanosecond offset | Source date and New York midnight come from the HTTP manifest. A Time message starts every unit and repeats when the second changes. |
| Bid/ask prices and displayed sizes | Top of Book (`q`) | Both sides travel together; no individual orders are invented. |
| Trade price and volume | Trade (`P`) | Observed trade with a generated replay trade ID. |
| Recognized halt / other condition | Symbol Status (`H`) | Halt (`H`) or regular-session (`T`) marker. |
| RIC | Instrument ID | One-based replay directory ID; full RICs stay in the HTTP directory. |

The `q` and `P` **layouts** are reused for consolidated quotes and trades; no LSE
auction state, order book or trade-correction service is reconstructed. Instrument
IDs, trade IDs and market-data group `A` are local replay identifiers.

Prices use eight decimal places in the specification's sign-magnitude form; the codec
rejects out-of-range prices. Quantities must be integral UInt32; fractional or
overflowing quantities stop the replay. Unavailable prices and unavailable or
nonpositive quantities become zero, which the quote rules treat as absent. Quote rows
with the halt flag are sent with unavailable prices. Units are the source units; no
lot-size conversion is applied.

MITCH sequence numbers advance for **every wire message**, including Time messages,
so wire-message counts exceed observation counts. The sender packs at most 32
observations per unit within a 1,472-byte payload; a requested rate of 75,000 means
source observations/s. One final Status marker closes the last bar at session end.

Each seek, loop or sender restart registers a new generation and a fresh UDP source
port through the manifest endpoint; the receiver rejects packets from earlier ports
and never reuses one. The source port must arrive unchanged (no source-port
translation). There is no retransmission or feed arbitration; lost or reordered
packets invalidate the affected marks, and malformed messages are rejected and counted.

## Quote rules and returns

A quote is usable when bid and ask are finite and positive, ask ≥ bid, both sizes are
positive, the spread is at most 100 basis points of the midpoint, and `HALT_REASN` is
zero. Invalid updates invalidate the mark. Trades are displayed separately and never
enter the return estimator; observed volume is not an official exchange total.

At each five-second event-time boundary the bar uses the last valid midpoint strictly
before the boundary, if it is at most 30 seconds old. A return is
`log(mid[t] / mid[t-1])` and needs both endpoints. Late observations cannot rewrite
closed bars. Sequence loss invalidates marks and the previous endpoint. A new
generation (seek, loop, restart) resets history, and older queued work is rejected.

## Statistics

The rolling window holds up to 60 returns; 20 are required before estimating. Only
instruments with finite returns across the whole window and sample variance above
1e-20 enter the matrix; missing returns are excluded, not treated as zero. Eligibility
is screened from the actual window before the dense GPU work.

For n returns, sample covariance is `(sum(r rᵀ) - sum(r) sum(r)ᵀ/n)/(n-1)`.
Annualized volatility is `sqrt(sample_variance × 252 × 23,400 / bar_seconds)`.
Correlation divides covariance by the product of sample standard deviations.
Statistics add new and subtract expired rows with BLAS products
(`cupy.cublas.gemm` / `scipy.linalg.blas.dgemm`) and rebuild every 256 calls to limit
drift. Both backends use float64 and identical eligibility.

With 60 returns the centered covariance has rank at most 59, so it is descriptive
monitoring, not an input for portfolio optimization without regularization.
Annualization is a scaling convention, not a forecast; five-second sampling carries
microstructure noise ([Aït-Sahalia and Yu](https://arxiv.org/abs/0906.1444)).

## Workers

The receiver, MITCH decode and the quote bars run on one Holoscan lane (thread); the
risk operator and its publisher run on a second, and in `compare` mode the CPU
cross-check and its publisher on a third ([architecture](ARCHITECTURE.md)). On the host
front ends (`udp`, `daqiri-host`; 4.3 `async`, `ano`) the GPU and CPU analytics each run
in a spawned process; on the GPU front ends (`gpudirect`, `replay`; 4.3 `gpu`) the GPU
analytics (`RollingRiskOp`) run in-process on the device-resident window.

The bars hand each new window to risk through a latest-only mailbox, so a slow analysis
skips windows instead of delaying the bars. The risk operator checks for a new window
every 1 ms (`--risk-poll-us`), keeps only the newest, and starts an analysis once a new
bar has closed and at least the minimum interval (50 ms, `--interval`) has passed since
the previous start. This is 4.3's rule (run on arrival, at a bounded rate), with at most
1 ms added. One matched comparison is in flight at a time: in `compare` mode, risk
claims the comparison slot for a window, sends the CPU cross-check a host copy of it,
and waits for that comparison to be published before analyzing the next window. CPU
BLAS thread tuning runs at startup and when the first full window exists.

`market_data_bench --check` checks the bars inside the graph: the decoded stream feeds a
GPU and a CPU `QuoteBarsOp`, every GPU window is compared with its CPU twin by
generation and bar count (times exactly, returns within 1 ULP, NaN pattern exactly), and
after the run every closed bar is compared with an independent host decode of the pcap.

## Timing scopes

MITCH carries no sender wall-clock time, so sender-to-receiver latency is unavailable.

- **Analytics timer:** host input preparation, incremental statistics, covariance,
  correlation, volatility, heatmap/probes and required transfers. GPU timers
  synchronize before returning.
- **Live `compare` mode:** the CPU recomputes each validated GPU window; every
  comparison checks volatilities, covariance probes and the heatmap, and failures are
  kept in `/snapshot` as `comparison_failure`. The CPU job starts when the GPU result is
  ready (4.3's host modes started both together), so its timing ratio is not the
  headline benchmark.
- **Packet arrival to result:** from the drain of the packet that closes the newest
  bar to publication (GPU) or worker completion (CPU). Results are published on the
  risk lane as soon as the analysis returns (4.3 polled for them every 5 ms). The
  arrival timestamp is taken when the drain takes a DAQIRI burst, after the
  partial-burst timeout, or in `replay` mode when the software NIC gathers the burst; it
  is not a NIC hardware timestamp and excludes browser refresh.

## Separate CPU-only and GPU-only benchmark

The 9.55× headline comes from two separate streaming runs on Holoscan 4.3. On Thor
(with the 5.0 receiver image and module paths):

```bash
for backend in cpu gpu; do
    bash deploy/igx_thor/igx_ano_down.sh
    ANALYTICS_BACKEND="$backend" bash deploy/igx_thor/igx_ano_run.sh
    docker exec fin-equities-ano bash deploy/igx_thor/ano_python.sh \
        -m holoscan_finance.tools.measure --seconds 120 --rate 75000 \
        --output "/tmp/$backend-only.json"
    docker cp "fin-equities-ano:/tmp/$backend-only.json" "/tmp/$backend-only.json"
done
bash deploy/igx_thor/igx_ano_down.sh
ANALYTICS_BACKEND=compare bash deploy/igx_thor/igx_ano_run.sh

PYTHONPATH=python python3 -m holoscan_finance.tools.compare_runs \
    --cpu /tmp/cpu-only.json --gpu /tmp/gpu-only.json \
    --all-shared --output /tmp/separate-streaming.json
```

`measure` seeks to 14:00 UTC, requests 75,000 observations/s, waits for a full window
and an agreeing CPU comparison, records 120 s and restores the previous replay
settings. It keeps every window whose triggering packet arrived in the interval and
computes p50/p99 with NumPy's default interpolation. Sequence gaps, a generation
change, failed agreement, missing comparisons or more than three million packets
invalidate the capture.

`compare_runs` rejects missing or duplicate update timestamps, unequal input hashes,
different protocols or settings, failed agreement and unequal incremental/rebuild
work, then reports CPU median / GPU median over every shared full window.
