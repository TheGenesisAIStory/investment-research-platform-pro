# Global OHLCV Price Pipeline

This pipeline builds a historical and incremental OHLCV store for equities and ETFs.

## Architecture

- `api_orchestrator.py`: shared provider fallback, cache, rate limit and retry/backoff layer.
- `loaders/market_universe.py`: discovers asset universes.
  - US listings: official NASDAQ Trader Symbol Directory.
  - Europe/Asia: existing index constituents plus optional CSV files in `Database Finanziario/catalog/exchange_symbols/`.
  - ETF seed: US ETF flags from NASDAQ Trader plus a small global ETF seed list.
- `loaders/ohlcv_client.py`: provider adapters.
  - `yfinance`: primary daily bulk downloader.
  - `stooq`: optional CSV fallback when `STOOQ_API_KEY` is configured.
  - `alpha_vantage`: optional daily adjusted and 5m intraday fallback when `ALPHAVANTAGE_API_KEY` is configured.
- `ohlcv_store.py`: `asset_master`, `price_ohlcv_daily`, `price_ohlcv_intraday_5m` DB adapter.
- `ohlcv_ingest.py`: full-history and incremental jobs.

## Storage

Default DB:

```text
Database Finanziario/MarketData/ohlcv.sqlite
```

Parquet mirror:

```text
Database Finanziario/MarketData/OHLCV/
  asset_master_candidates.csv
  daily/{exchange}/{ticker}.parquet
  intraday_5m/{exchange}/{ticker}.parquet
  manifests/
```

Set `MARKETDATA_DATABASE_URL` for Postgres/TimescaleDB, for example:

```bash
export MARKETDATA_DATABASE_URL='postgresql+psycopg2://user:pass@localhost:5432/marketdata'
```

## First Historical Load

Run a small smoke slice first:

```bash
python scripts/sync_ohlcv_prices.py --execute --mode full --markets us_all,global_etfs --start-date 2000-01-01 --max-assets 50
```

Then remove the cap:

```bash
python scripts/sync_ohlcv_prices.py --execute --mode full --markets us_all,europe_major,japan_major,global_etfs --start-date 2000-01-01 --max-assets 0
```

## Daily Incremental Update

```bash
python scripts/sync_ohlcv_prices.py --execute --mode incremental --markets us_all,europe_major,japan_major,global_etfs --max-assets 0
```

Cron example:

```cron
45 18 * * 1-5 cd /path/to/research_platform_definitive && python scripts/sync_ohlcv_prices.py --execute --mode incremental --markets us_all,europe_major,japan_major,global_etfs --max-assets 0
```

## 5 Minute Data

Historical 5m is free-tier constrained. The implemented provider is Alpha Vantage:

```bash
export ALPHAVANTAGE_API_KEY='...'
python scripts/sync_ohlcv_prices.py --execute --interval 5m --month 2024-01 --markets us_all --max-assets 10
```

Use small batches because free-tier rate limits are strict.

## Extending Europe/Asia Coverage

Place CSV files in:

```text
Database Finanziario/catalog/exchange_symbols/
```

Expected columns:

```text
ticker,provider_symbol,name,exchange,country,type
```

`provider_symbol` should be the symbol accepted by the primary provider, e.g. Yahoo Finance suffixes such as `.MI`, `.DE`, `.PA`, `.L`, `.T`.

## Monitoring

Main outputs:

- `Database Finanziario/MarketData/OHLCV/manifests/ohlcv_daily_manifest.csv`
- `research_platform_definitive/output/tables/OHLCV_daily_manifest.csv`
- `research_platform_definitive/output/logs/ohlcv_api_orchestrator.jsonl`
- `research_platform_definitive/output/logs/ohlcv_provider_usage.json`

Watch `status`, `rows`, `provider`, `error` and `duration_seconds`.

Provider ranking, fallback and budget rules are documented in
`docs/OHLCV_API_ORCHESTRATION.md`.

Kaggle static seed imports and bootstrap commands are documented in
`docs/KAGGLE_SEED_INTEGRATION.md`.
