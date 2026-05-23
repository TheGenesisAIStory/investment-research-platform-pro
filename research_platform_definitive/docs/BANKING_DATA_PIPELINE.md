# Banking Data Pipeline

This module adds an open-source banking data layer for the Italian Banks ML Lab and the broader Research Platform.

## Module

`src/research_platform_core/banking_data.py`

Public entry points:

- `build_banks_universe(cache_dir, output_dir=None)`
- `run_banks_data_pipeline(output_dir, cache_dir=None, ...)`
- `build_yfinance_bank_panel(tickers, start, end, ...)`
- `ecb_get_series(dataset_id, key, cache_dir, params=None)`
- `fetch_bancaditalia_macro_regulatory(cache_dir, exports={...})`
- `load_external_fundamentals(filepath, banks_universe=None)`
- `discover_bank_report_links(bank_name, ir_url, cache_dir)`

## Data Policy

The pipeline is designed for notebook-first research:

1. Use cached Drive/repo artifacts first.
2. Use official/public sources for universe and regulatory context.
3. Use provider/API fallbacks only for listed-bank market data.
4. Persist outputs as CSV and SQLite contracts.

## Main Artifacts

The Italian Banks notebook writes:

- `papers/italian_banks_ml_stock_screening/output/banks_pipeline/banks_universe.csv`
- `papers/italian_banks_ml_stock_screening/output/banks_pipeline/banks_macro_regulatory.csv`
- `papers/italian_banks_ml_stock_screening/output/banks_pipeline/banks_fundamentals_panel.csv`
- `papers/italian_banks_ml_stock_screening/output/banks_pipeline/banks_market_panel.csv`
- `papers/italian_banks_ml_stock_screening/output/banks_pipeline/banks_data.sqlite`

## Source Caveats

- Wikipedia is only a seed list, not a regulatory source of truth.
- ECB SI/LSI PDF parsing is best-effort and should be reviewed after ECB layout changes.
- Banca d'Italia BDS often needs explicit export links configured in `bds_exports`.
- Annual report PDF parsing is intentionally a placeholder until audited extraction rules are added.
- yfinance is used as a market-data fallback for listed banks, with explicit proxy flags.

## Streamlit

The app page `pages/12_Banking_Data_Lab.py` exposes:

- universe metrics;
- filters by country/status;
- banking artifact contracts;
- optional refresh of listed-bank market panel;
- optional ECB BSI call.

