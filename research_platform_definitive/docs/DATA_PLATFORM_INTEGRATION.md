# Data Platform Integration

## Canonical Source Of Truth

Primary data root:

```text
/Users/itsgennymac/Library/CloudStorage/GoogleDrive-sfn.gns@gmail.com/Il mio Drive/Database Finanziario
```

Colab-compatible root:

```text
/content/drive/MyDrive/Database Finanziario
```

The project now discovers this root through:

1. `FINANCIAL_DB_ROOT`
2. `GOOGLE_DRIVE_DB_ROOT`
3. `ML_TRADING_DB_BASE`
4. `DB_BASE`
5. `DATA_PATH`
6. known Mac/Colab Drive paths

## Observed Database Structure

The database is not a single flat cache. It contains:

- `catalog/`: provider registry, credential status, data domains and inventory.
- `europe_stoxx_companies/`: parquet OHLCV/security metadata for European equities.
- `alternative_data/`: Stocktwits, Polymarket and related feature snapshots.
- `alpha_factor_library/` and `alpha_factor_research/`: factor panels, IC diagnostics and model outputs.
- `alpha_factor_library/aqr/`: AQR Data Library raw Excel cache, parsed factor CSVs and discovery catalog.
- `notebook_exports/`: CSV, HTML, markdown and chart exports.
- `analysis_outputs/`: published notebook/app outputs.
- `output/`: legacy notebook output structures.
- multiple research-domain folders for ML4T notebooks.

## Implemented Shared Layer

Core module:

```text
src/research_platform_core/data_platform.py
```

API governance module:

```text
src/research_platform_core/api_management.py
```

Batch/export and bridge modules:

```text
src/research_platform_core/batch_download.py
src/research_platform_core/data_bridge.py
src/research_platform_core/data_center_catalog.py
src/research_platform_core/api_orchestrator.py
src/research_platform_core/cache_manager.py
src/research_platform_core/batch_downloader.py
src/research_platform_core/loaders/
config/data_api_control.yaml
config/data_sources.yaml
config/rate_limits.yaml
config/scheduler_config.yaml
scripts/migrate_data_api_control.py
scripts/smoke_data_api_control.py
scripts/initial_setup.py
scripts/sync_prices.py
scripts/sync_fundamentals.py
scripts/validate_data.py
```

Key functions:

- `discover_financial_database_root`
- `resolve_data_platform_roots`
- `build_dataset_inventory`
- `dataset_status`
- `should_refresh`
- `read_dataset`
- `read_dataset_drive_first`
- `provider_fallback_plan`
- `incremental_merge`
- `write_dataset_incremental`
- `refresh_europe_stoxx_prices_incremental`
- `publish_artifacts_to_financial_db`
- `write_data_platform_status`

API key functions:

- `discover_api_credentials_root`
- `resolve_api_control_roots`
- `api_control_status`
- `accepted_api_env_vars`
- `apply_env_text_to_session`
- `write_api_control_status`

Batch/bridge functions:

- `create_batch_download`
- `enrich_inventory_for_export`
- `estimate_batch_size`
- `DataBridge.push_manifest`
- `FamaFrenchLoader.download_all`
- `AQRDataLoader.download_all`
- `EquityUniverseManager.sync_prices`
- `EquityUniverseManager.sync_fundamentals`
- `ForexCommoditiesManager.sync_fx`
- `ForexCommoditiesManager.sync_commodities`
- `RiskFactorsLoader.sync_fred`

## Operating Policy

The platform follows:

1. Drive-first lookup.
2. Local/repo cache fallback.
3. API only when data are missing, stale or incomplete.
4. Incremental refresh and deduplication before full refresh.
5. Provider/freshness/coverage metadata written with outputs.

## App Integration

Streamlit page:

```text
research_platform_app/pages/7_Data_Platform.py
```

Dedicated Data/API app:

```text
data_api_app.py
research_platform_app/pages/11_Data_API_Control_Center.py
```

It exposes:

- Drive root status,
- dataset role summary,
- inventory search,
- catalog tables,
- provider and credential status,
- dashboard health KPIs,
- batch download/export manager,
- Data/API analytics,
- `ml_stock_lab` manifest bridge,
- artifact publishing to `analysis_outputs/<domain>`,
- status exports to workspace output.

Scheduler job:

```text
data_platform_status_refresh
europe_prices_incremental_refresh
```

`data_platform_status_refresh` writes `DataPlatform_*.csv` contracts into `output/tables` and makes the Drive database visible to Streamlit/orchestration freshness checks.

`europe_prices_incremental_refresh` inspects `europe_stoxx_companies/*_1d.parquet`, calls yfinance only for stale/missing symbols unless forced, deduplicates by `symbol,date_time`, and writes a refresh manifest to:

```text
Database Finanziario/catalog/europe_stoxx_incremental_price_refresh.csv
```

API-ready contract:

```text
output/api_contracts/data_platform_contract.json
output/api_contracts/data_api_control_contract.json
```

## Notebook Integration

Both main notebooks now include a tagged `data-platform-bootstrap` cell after `parameters`.

The common Colab operations notebook is:

```text
data_api_management/notebooks/Data_API_Management_Colab.ipynb
```

This cell:

- resolves `FINANCIAL_DB_ROOT`,
- sets `DB_BASE` and `DATA_PATH`,
- imports the shared data-platform layer,
- displays a concise dataset summary,
- remains Colab-compatible.

## API Credit Control

API providers remain enrichment/fallback sources. Notebook/app refresh code should call APIs only after:

- checking Drive/cache existence,
- checking freshness window,
- checking minimum coverage,
- logging provider and fallback path.

Full-history downloads should be avoided unless a dataset is absent or explicitly forced.

## Scheduler Examples

Safe weekly/data status loop:

```bash
python research_platform_app/scheduler.py --interval-seconds 900 --jobs data_platform_status_refresh screener_refresh
```

Incremental price refresh:

```bash
python research_platform_app/scheduler.py --once --jobs europe_prices_incremental_refresh
```

AQR factor-library refresh:

```bash
python research_platform_app/scheduler.py --once --jobs aqr_factor_library_refresh
```

Use `top_n=0` in the app to refresh every discovered AQR workbook, or a small `top_n` value for smoke checks.

Full notebook jobs can be scheduled, but are guarded by the off-hour window:

```bash
python research_platform_app/scheduler.py --interval-seconds 3600 --jobs valuation_research_refresh portfolio_research_refresh --notebook-offhour-start 20 --notebook-offhour-end 7
```
