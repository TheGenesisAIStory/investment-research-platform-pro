# Kaggle Seed Integration

Kaggle datasets are treated as local static history, not as live providers. Put
downloaded CSV/TXT/ZIP files under:

```text
Database Finanziario/Kaggle/
```

Typical folders:

```text
Kaggle/
  price-volume-data-for-all-us-stocks-etfs/
    Stocks/
    ETFs/
  stock-market-dataset/
    symbols_valid_meta.csv
    stocks/
    etfs/
  historical-etf-data/
  world-stock-prices-daily-updating/
  cryptocurrencypricehistory/
  crypto-5m/
  btcusdt-5-minute-ohlcv/
```

## Flow

1. `kaggle_seed_loader.py` normalizes local Kaggle CSV/TXT/ZIP files.
2. Assets are upserted through `OhlcvDatabase.upsert_assets`.
3. Daily rows go to `price_ohlcv_daily`; crypto 5m rows go to
   `price_ohlcv_intraday_5m`.
4. Price rows are tagged with sources such as:
   - `kaggle_seed_stocks`
   - `kaggle_seed_etf`
   - `kaggle_seed_crypto_daily`
   - `kaggle_seed_crypto_5m`
5. API incremental jobs use max daily date by `provider_symbol`, so API calls
   start after the latest available Kaggle/API date instead of redownloading the
   full history.

## Commands

Dry-run only:

```bash
python scripts/sync_ohlcv_prices.py --use-kaggle-only --kaggle-datasets us_stocks_etfs
```

Import Kaggle only:

```bash
python scripts/sync_ohlcv_prices.py --execute --use-kaggle-only --kaggle-datasets us_stocks_etfs,crypto_daily
```

Bootstrap full data center:

```bash
python scripts/bootstrap_market_datacenter.py \
  --execute \
  --markets us_all,europe_major,japan_major,global_etfs \
  --start-date 2000-01-01 \
  --include-kaggle-seeds true
```

Daily notebook:

```bash
EXECUTE_MARKETDATA_NOTEBOOK=true \
jupyter nbconvert --to notebook --execute notebooks/daily_marketdata_update.ipynb \
  --output daily_marketdata_update_ran.ipynb
```

If you refresh Kaggle files manually, add:

```bash
UPDATE_KAGGLE_SEEDS=true
```

## Configuration

Dataset names, local paths and source tags live in
`config/ohlcv_data_sources.yaml` under `ohlcv.kaggle.datasets`.

The default loader handles common Kaggle layouts:

- one file per ticker, with ticker inferred from the filename;
- multi-ticker files with `ticker`, `symbol`, `pair` or `asset` columns;
- stock/ETF metadata files named `symbols_valid_meta.csv`;
- zipped Kaggle downloads containing CSV/TXT members.
