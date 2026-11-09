# Data, wire format, calculations and measurement

## Data

`python -m holoscan_finance.tools.synth` generates deterministic equity observations,
a replay tape and an Ethernet/IPv4/UDP pcap. A fixed seed controls per-symbol price levels,
spreads, volatility and random walks with a common market factor. The stream includes
quotes, trades and condition markers, with rare invalid quotes and halts to exercise
rejection and recovery. Symbols use generated names in a manifest directory.

The tape stores time-ordered observations, with prices and quantities in float64 fields. `--seconds` sets the
span after 09:30 New York; `--symbols`, `--packets` and `--max-events-per-packet` set the
universe and volume. A final status marker closes the last bar. `synth.json` records the
seed, generation arguments, file sizes and exact expected capture counts. See
[synthetic inputs](VALIDATION.md#synthetic-inputs) for commands.

## Wire format: FIX SBE replay

Sender and receivers use [FIX Simple Binary Encoding 1.0](https://www.fixtrading.org/standards/sbe/)
(the FIX Trading Community's binary encoding) under the replay's own message schema,
[`sbe_schema.xml`](../python/holoscan_finance/marketdata/sbe_schema.xml). Every UDP datagram
is exactly one `MarketDataIncrementalRefresh` message, little-endian: the 8-octet SBE
message header (`blockLength`, `templateId` 1, `schemaId` 18502, `version`), the root block
(`SeqNum`, padded to 12 octets) and the `NoMDEntries` repeating group (`blockLength` 48,
`numInGroup`, then one fixed 48-octet entry per observation). There is no other envelope or
hidden timing field; every byte of a datagram is described by the schema, and the
[reference SBE tool](VALIDATION.md#library-bit-identity) decodes the replay from the XML
alone. The generator encodes synthetic observations in this schema for replay; the packets
do not represent an exchange feed.

| Synthetic input | Entry fields (FIX tag) | Replay interpretation |
| --- | --- | --- |
| Market timestamp | `TransactTime` (60): nanoseconds since the Unix epoch | Must lie in the market day of the HTTP manifest's date and be non-decreasing within a packet; every packet decodes on its own. |
| Bid/ask prices and displayed sizes | `EntryType` Quote; `BidPx` (132), `OfferPx` (133), `BidSize` (134), `OfferSize` (135) | One entry carries both sides; no order book is reconstructed. |
| Trade price and volume | `EntryType` Trade; `LastPx` (31), `LastQty` (32) | Synthetic trade. |
| Recognized halt / other condition | `EntryType` TradingStatus; `SecurityTradingStatus` (326) `TradingHalt` (2) or `ReadyToTrade` (17) | A halt or a resume. `TradingHalt` on a quote or trade entry marks it halted; the quote keeps its prices. |
| Symbol name | `SecurityID` (48) | Zero-based replay directory index, the instrument key; names live in the directory only. |

Prices are the `PriceNULL4` composite: a signed 64-bit mantissa with a constant exponent of
-4 (four implied decimals), `null` (the int64 minimum) when absent. Quantities are
`QtyNULL`, an unsigned 32-bit count, `null` (the uint32 maximum) when absent. Fields that
do not belong to an entry's type are null. Missing source values travel as null and decode
as NaN; nothing is coerced. A price decodes as `float64(mantissa) / 10000.0` (the mantissa
converts to the nearest double, then one IEEE division), the same two roundings on every
decoder. The codec rejects infinite prices, mantissas outside the int64 range and
fractional, negative or overflowing quantities; negative prices are representable. Units
are the source units; no lot-size conversion is applied.

Sequence numbers advance for **every entry**, so an entry count is an observation count and
a sequence gap is a count of lost observations. The sender packs at most 30 observations per
packet within a 1,472-byte payload (30 entries are 1,464 bytes); a requested rate of 75,000
means source observations/s. One final `TradingStatus` marker closes the last bar at
session end.

As SBE requires, the decoders take the root and entry block lengths from the wire, so a
later schema version that appends fields decodes with today's receivers, and a later
`version` is accepted. The replay receivers additionally require the lengths to keep every
entry on an 8-octet boundary (the GPU reads each field with one aligned load) and reject the
message otherwise, as they reject another `schemaId` or `templateId`, an unknown entry type
or trading status, and a `TradingStatus` entry without a status.

Each seek, loop or sender restart registers a new generation and a fresh UDP source
port through the manifest endpoint; the receiver rejects packets from earlier ports
and never reuses one. The source port must arrive unchanged (no source-port
translation). There is no retransmission or feed arbitration; lost or reordered
packets invalidate the affected marks, and malformed messages are rejected and counted.

## Quote rules and returns

A quote is usable when bid and ask are finite and positive, ask ≥ bid, both sizes are
positive, the spread is at most 100 basis points of the midpoint, and the halt flag is
clear. Invalid updates invalidate the mark. Trades are displayed separately and never
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
  could differ). The [synthetic validation](VALIDATION.md#full-chain-with-compare) checks
  that both backends choose the same method on each compared window.
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

The receiver, SBE decode and the quote bars run on one Holoscan lane (thread); the
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

The entries carry market time, not sender wall-clock time, so sender-to-receiver latency is unavailable.

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
  lane as soon as the analysis returns. The figure includes the time the window waited in
  the risk slot for the minimum interval since the previous analysis: with bursts every
  20 ms and the default 50 ms interval, the window analyzed is the newest one when the
  interval elapses, which arrived up to one burst earlier (in the NIC-free `paced` bench
  about 10 ms of a 16 ms median, against 1-2 ms each for decode, bars and the analysis).
  Lower `--interval` to time the pipeline alone. The stamp is the host's wall clock, not a
  NIC hardware timestamp, and the figure excludes browser refresh. Where it is taken
  depends on the front end, so compare runs of the same front end only:
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
`replay_sender` over localhost in one container ([validation procedure](VALIDATION.md#separate-cpu-and-gpu-runs)).
On Thor with the NIC, they use the launcher's default host-memory ingest
(`INGEST=daqiri-host`; `fin-equities-receiver` is the launchers' default `APP_NAME`):

```bash
PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$PWD/python" python3 -m holoscan_finance.tools.synth \
    --out data/synth-day --symbols 2886 --packets 1824000 --seconds 23400 --reuse
export TAPE_DIR="$PWD/data/synth-day/tape"
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

PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$PWD/python" python3 -m holoscan_finance.tools.compare_runs \
    --cpu /tmp/cpu-only.json --gpu /tmp/gpu-only.json \
    --all-shared --output /tmp/separate-streaming.json
```

`measure` seeks to 14:00 UTC, requests 75,000 observations/s, waits for a full window
(and an agreeing CPU comparison in `compare` mode), records 120 s and restores the
previous replay settings. It keeps every window whose triggering packet arrived in the interval and
computes p50/p99 with NumPy's default interpolation. Sequence gaps, a generation
change, failed agreement, missing comparisons or more than three million packets
invalidate the capture.

`compare_runs` rejects missing or duplicate update timestamps, unequal input hashes,
different protocols or settings, failed agreement and unequal incremental/rebuild
work, then reports CPU median / GPU median over every shared full window.
