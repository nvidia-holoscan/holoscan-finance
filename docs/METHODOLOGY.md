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
comes from the window's own sample variance (`np.var(rows, axis=0, ddof=1)`, which the
GPU reproduces bit for bit), never from the incremental sums; an instrument whose
variance from `C` (below) is not positive is excluded too, so no displayed volatility is
zero and no correlation divides by it.

For n returns, sample covariance is `(sum(r rᵀ) - sum(r) sum(r)ᵀ/n)/(n-1)`.
Annualized volatility is `sqrt(sample_variance × 252 × 23,400 / bar_seconds)`.
Correlation divides covariance by the product of sample standard deviations. The heatmap
shows the 24 eligible instruments with the highest variance, and each comparison checks
32 × 32 covariance probes spread evenly over the eligible instruments.

Both backends run one algorithm in float64 (`marketdata/covariance.py`):

- **One matrix.** The engine keeps the augmented second-moment matrix
  `C = sum_t [r_t; 1][r_t; 1]ᵀ` of the window, `(N+1) × (N+1)` with missing returns as 0;
  its last row and column hold `sum(r)`.
- **One signed rank-k update per window.** A window that slides by k bars subtracts the k
  expired bars and adds the k new ones in one update: SciPy BLAS `dgemm` on the CPU, a
  fused CUDA kernel on the GPU. Any other window (a gap, a seek), and every 256th call,
  rebuilds `C` from the window to limit drift.
- **Recomputed after cancellation.** Subtracting an expired bar leaves the rounding of
  every update it took part in, about 1e-16 of its squared return per update, in `C`. The
  256-call rebuild bounds that only relative to the largest return that passed through:
  after a large move expires from a name that then trades quietly (a jump, then a pinned
  price; a quote glitch and back), it could dominate the name's variance. Each engine
  keeps every column's largest squared return since the last rebuild, and a window in
  which an instrument with a window variance above 1e-20 has that largest squared return
  above 1e5 times its diagonal of `C` is recomputed from its bars (reported as a rebuild).
  The relative error of `C` then stays below about 1e5 × 1e-16 × (updates), about 1e-9.
  The decision reads only the window, the exact squares and `C`, so both backends take it
  on the same windows (only an exact tie at the threshold, between two roundings of `C`,
  could differ). In every comparison of [VALIDATION.md](VALIDATION.md) steps 7 and 11 on
  the recorded day, both backends chose the same method for every window.
- **No N_e × N_e matrix.** Volatilities come from the diagonal, and the heatmap and the
  probes from 24 × 24 and 32 × 32 blocks of `C`. The mean correlation is one
  matrix-vector product, `((d·Cd − (d·t)²/n)/(n−1) − N_e)/(N_e(N_e−1))` with `d = 1/σ` on
  eligible instruments, `t = sum(r)` and `N_e` eligible instruments: the mean of the
  off-diagonal correlations, without clipping each one to ±1 first.

On the GPU the selection and finishing (eligible indices, volatilities, the stable top-24
variance ranking, the probe positions, the heatmap and probes) are small CUDA kernels
compiled without fused multiply-add, so they round exactly as the CPU's NumPy expressions
do given the same `C`; `C` itself differs between the backends only by the summation
order and fused multiply-add rounding of the rank-k update (SciPy `dgemm` on the CPU, a
CUDA kernel on the GPU).
`agreement` checks the eligible and displayed instruments for identity and the
volatilities, probes and heatmap within relative 1e-8. The mean correlation is not part
of the check.

With 60 returns the centered covariance has rank at most 59, so it is descriptive
monitoring, not an input for portfolio optimization without regularization.
Annualization is a scaling convention, not a forecast; five-second sampling carries
microstructure noise ([Aït-Sahalia and Yu](https://arxiv.org/abs/0906.1444)).

## Workers

The receiver, MITCH decode and the quote bars run on one Holoscan lane (thread); the
risk operator and its publisher run on a second, and in `compare` mode the CPU
cross-check and its publisher on a third ([architecture](ARCHITECTURE.md)). On the host
front ends (`udp`, `daqiri-host`) the GPU and CPU analytics each run in a spawned
process; on the GPU front ends (`gpudirect`, `replay`) the GPU analytics
(`RollingRiskOp`) run in-process on the device-resident window.

The bars hand each new window to risk through a latest-only mailbox, so a slow analysis
skips windows instead of delaying the bars. The risk operator checks for a new window
every 250 µs (`--risk-poll-us`), keeps only the newest, and starts an analysis once a new
bar has closed and at least the minimum interval (50 ms, `--interval`) has passed since
the previous start. So an analysis runs on arrival at a bounded rate; the poll adds at
most 250 µs. One matched comparison is in flight at a time: in `compare` mode, risk
claims the comparison slot for a window, sends the CPU cross-check a host copy of it,
and waits for that comparison to be published before analyzing the next window. Both
engines compute the window's variance themselves (the GPU's equals NumPy's bit for bit),
and the cross-check, like risk, sends its worker only the new bars of a window that
slides from the one it sent last, so both sides do the same work on every validated
window.

Each engine is allocated at the first window of a universe, which runs once untimed, for
the bars' full window (`full_window_bars`, so growth windows never reallocate; a longer
window grows the workspaces and keeps the statistics), and is reset, not replaced, on a
new generation. On the GPU nothing is allocated per window, and every window, whatever
its kind, is one launch of a CUDA graph captured at allocation: the window's shape and its
new bars' addresses travel in a small control block that the graph copies to the device
first. If the capture fails (another thread synchronized the whole device meanwhile), the
windows run the same work eagerly and the capture is retried 256 windows later.
A spawned worker keeps the window it analyzed last, so risk sends it only the new bars
of a sliding window (the full window after a gap, a new generation or a new worker). A CPU
worker tunes its BLAS thread count once per universe, before the first window: when the
manifest announces it, or, in `replay` mode, where the universe is registered before the
graph runs and the replay starts with the source, in the operator's `start()`, which
returns once the tuning is done. It times a run of one-bar slides of a full window of that
width at {1, 2, 4, 8, all} threads (the work, a rank-k update of the whole `(N+1) x (N+1)`
matrix, does not depend on the returns), twice, and keeps the second pass's times: a
process's first engine slides faster than every later one, the engine the worker keeps
included, so a single pass understated the first candidate. It keeps the fewest threads
within 10% of the fastest, since threads that win only by noise in an idle tuning stall
once the pipeline's own threads compete for the cores. The tuned limit then stays in force,
and no window waits for a tuning, so in `compare` mode no GPU result does either.

`market_data_bench --check` checks the bars inside the graph: the decoded stream feeds a
GPU and a CPU `QuoteBarsOp`, every GPU window is compared with its CPU twin by
generation and bar count (times exactly, returns within 1 ULP, NaN pattern exactly), and
after the run every closed bar is compared with an independent host decode of the pcap.

## Timing scopes

MITCH carries no sender wall-clock time, so sender-to-receiver latency is unavailable.

- **Analytics timer:** copying the window's new bars in, the incremental statistics,
  volatility, mean correlation, heatmap and probes, and the transfer of the compact
  result (on the GPU, one device-to-host copy, then a synchronize before it returns).
  The worker's interprocess transfer and the result's conversion to lists are outside it.
  On both engines the interpreter's automatic garbage collection is held off while the
  timer runs: a collection starts at whichever allocation in the process crosses the
  collector's threshold, and one that started inside a window added its whole pause to
  it. A collection that falls due meanwhile runs right after the window, so it stays in
  packet arrival to result.
- **Live `compare` mode:** the CPU recomputes each validated GPU window; every
  comparison checks volatilities, covariance probes and the heatmap, and failures are
  kept in `/snapshot` as `comparison_failure`. The dashboard's speedup
  (`rolling_timing.speedup`) is the median CPU analytics time over the median GPU
  analytics time of the last 120 comparisons that agree, did the same incremental work
  (method, added and expired bars) and have at least two eligible instruments; rebuilds
  are excluded. Both times are the analytics timer above for the same window: each
  engine computes its own variance and copies in only the new bars. The CPU job starts
  when the GPU result is ready, and both run on the same host at once, so its timing
  ratio is not the separate-runs benchmark below.
- **Packet arrival to result:** from the arrival stamp of the packet that closes the newest
  bar to publication (GPU) or worker completion (CPU). Results are published on the risk
  lane as soon as the analysis returns. The stamp is the host's wall clock, not a NIC
  hardware timestamp, and the figure excludes browser refresh. Where it is taken depends on
  the front end, so compare runs of the same front end only:
  - `udp`: per datagram, in the receiver thread right after `recvfrom` returns;
  - `daqiri-host` and `gpudirect`: per DAQIRI burst, when the receiver or the drain takes
    it, after the partial-burst timeout (5 ms and 20 ms);
  - `replay`: per burst, when the software NIC gathers it, after DAQIRI's burst rule
    (4,096 packets or 20 ms). Below 204,800 packets/s every burst closes on the timeout,
    and up to 20 ms of a packet's wait is not in the figure.

## Separate CPU-only and GPU-only benchmark

The separate-runs CPU/GPU speedup comes from two separate streaming runs, one with CPU
analytics only and one with GPU analytics only, on the same host front end, so that both
stamp arrivals the same way. Without the NIC, both runs use the `udp` front end and
`replay_sender` over localhost in one container ([VALIDATION.md, step 11](VALIDATION.md#11-separate-cpu-only-and-gpu-only-runs-nic-free-timing-pass)).
On Thor with the NIC, they use the launcher's default host-memory ingest
(`INGEST=daqiri-host`; `fin-equities-receiver` is the launchers' default `APP_NAME`):

```bash
for backend in cpu gpu; do
    bash deploy/igx_thor/igx_down.sh
    ANALYTICS_BACKEND="$backend" bash deploy/igx_thor/igx_run.sh
    docker exec fin-equities-receiver bash deploy/igx_thor/run_python.sh \
        -m holoscan_finance.tools.measure --seconds 120 --rate 75000 \
        --output "/tmp/$backend-only.json"
    docker cp "fin-equities-receiver:/tmp/$backend-only.json" "/tmp/$backend-only.json"
done
bash deploy/igx_thor/igx_down.sh
ANALYTICS_BACKEND=compare bash deploy/igx_thor/igx_run.sh

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
