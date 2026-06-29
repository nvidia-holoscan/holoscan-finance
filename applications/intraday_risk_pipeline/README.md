# Intraday Risk Pipeline

This offline reference application demonstrates how to compose the operators in
`holoscan.finance` into a finite intraday portfolio-risk pipeline.
It uses fictional symbols and checked-in synthetic ticks, so it requires no
network access, credentials, or licensed market data.

> This is an educational reference pipeline, not investment advice or a
> production, accounting, or regulatory risk system.

## Pipeline

```text
MarketDataReplayOp (tick)
  -> RollingStatisticsOp (features)
  -> PortfolioRiskOp (snapshot)
  -> ConsoleRiskSinkOp
```

All connections use explicit port maps. The operators have the stable names
`market_data_replay`, `rolling_statistics`, `portfolio_risk`, and
`console_risk_sink` for readable Holoscan logs and traces.

The default portfolio and thresholds are:

| Setting | Value |
| --- | ---: |
| ALPHA position | Long 100 at USD 100 |
| BETA position | Short 40 at USD 200 |
| Rolling window / minimum periods | 3 / 2 |
| Maximum gross exposure | USD 19,000 |
| Maximum absolute net exposure | USD 3,000 |
| Maximum unrealized loss | USD 1,000 |

The eight events in
[`data/synthetic_market_data.csv`](data/synthetic_market_data.csv) first
produce a partial snapshot, then complete snapshots. Later price moves
deterministically demonstrate gross-exposure, net-exposure, and unrealized-loss
breaches.

## Run

From the module root, run:

```bash
./holohub run intraday_risk_pipeline
```

The console sink logs whether each snapshot is complete or partial and emits a
warning for each limit breach. The source is also reusable from tests or other
applications through `IntradayRiskApp(ticks=..., positions=..., limits=...,
snapshots=...)`.

## Scope and limitations

- Risk snapshots combine the latest asynchronous mark for each position and do
  not perform stale-price checks.
- The rolling deviation is an unannualized per-observation statistic, not VaR,
  expected shortfall, or a forecast.
- The example performs no FX conversion, corporate-action processing, market
  calendar handling, correction/cancel logic, or checkpointing.
- The Python value objects are intended for in-process flows. Distributed
  fragments require an explicitly versioned wire format.
