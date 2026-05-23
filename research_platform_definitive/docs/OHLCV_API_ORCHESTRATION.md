# OHLCV API Orchestration

This layer sits above the OHLCV providers and keeps the public ingest entrypoint
compatible:

```bash
python scripts/sync_ohlcv_prices.py --execute --mode full --start-date 2000-01-01
python scripts/sync_ohlcv_prices.py --execute --mode incremental
python scripts/sync_ohlcv_prices.py --execute --interval 5m --max-assets 10 --preferred-providers alpha_vantage
```

## Components

- `DataProviderRegistry` keeps provider capability profiles: intervals, regions,
  asset types, batch support, expected quality, stability and cost.
- `DataProviderPolicyEngine` ranks providers for a request by quality, coverage,
  cost, stability, priority and live budget availability.
- `ProviderUsageTracker` persists minute and daily usage counters in
  `output/logs/ohlcv_provider_usage.json`.
- `MarketUniverseBuilder` caches NASDAQ Trader listing universes under
  `Database Finanziario/catalog` using `ohlcv.cache.universe_ttl_hours`.
- `kaggle_seed_loader.py` imports local Kaggle static histories and the API
  ingest reads max dates by `provider_symbol`, so API deltas start after the
  latest Kaggle/API observation.
- `APIOrchestrator.fetch_with_fallback(...)` still exposes the old fallback
  behavior, but now also accepts an explicit provider order from the policy
  engine and skips providers whose budget or environment keys are unavailable.
- `OhlcvClient` exposes the stable public helpers while delegating provider
  choice to the policy engine.

## Provider Policy

For daily full-history jobs the engine prefers bulk-capable, low-cost providers.
In the default config this means `yfinance` for broad batches. Stooq and Alpha
Vantage remain configured as fallback providers but are skipped automatically
until their required environment keys are present.

For 5 minute jobs the engine only considers providers that declare
`supports_intraday_5m`. In the default profile this is Alpha Vantage, so 5m jobs
should be small and targeted.

If a daily provider returns a materially incomplete history, `OhlcvClient` can
blend fallback providers. Conflict resolution is deterministic: the policy order
wins on overlapping dates.

Kaggle is registered as `kaggle_seed` with `static_history: true`. It is not a
network fetcher; it is considered by bootstrap/date logic and skipped by live API
calls.

## Budgets And Cache

The configurable knobs live in `config/ohlcv_data_sources.yaml`:

- `ohlcv.cache.enabled`
- `ohlcv.cache.daily_request_ttl_hours`
- `ohlcv.cache.intraday_request_ttl_hours`
- `ohlcv.policy.weights`
- `ohlcv.policy.incomplete_coverage_threshold`
- `ohlcv.providers.<name>.requests_per_minute`
- `ohlcv.providers.<name>.requests_per_day`
- `ohlcv.providers.<name>.supports_bulk_daily`
- `ohlcv.providers.<name>.supports_intraday_5m`

Examples:

```bash
# Full history from 2000 with orchestrated bulk provider selection
python scripts/sync_ohlcv_prices.py --execute --mode full --start-date 2000-01-01 --max-assets 0

# Daily incremental update, respecting all provider budgets
python scripts/sync_ohlcv_prices.py --execute --mode incremental --max-assets 0

# Targeted 5m pull. Keep this small because free intraday APIs are tight.
python scripts/sync_ohlcv_prices.py --execute --interval 5m --markets us_all --max-assets 10 --preferred-providers alpha_vantage
```

## Notes On Current Providers

- `yfinance`: bulk daily ergonomics are excellent, but Yahoo access is unofficial,
  so stability score is lower than documented APIs.
- `Stooq`: useful historical CSV fallback; this project expects `STOOQ_API_KEY`
  for direct CSV API calls.
- `Alpha Vantage`: documented daily and intraday API. The free budget is small,
  so the default config uses it only when its key is present and the policy
  selects a targeted fallback or intraday job.
