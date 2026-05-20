# Research Platform Architecture Notes

## Current Boundary

The notebooks remain the orchestration layer. Local modules provide reusable engines:

- `company_valuation_screener.py`: company-level screener contract.
- `company_valuation_finviz_layer.py`: maps, groups, watchlists, events, macro board, alerts and API-ready exports.
- `portfolio_research_screener.py`: portfolio-native selection layer for allocation candidates.
- `company_valuation_dashboard.py` and `portfolio_dashboard.py`: HTML export builders.
- `src/research_platform_core/`: shared dataframe, alias, export and HTML helpers.
- `src/research_platform_core/data_platform.py`: Drive-first Database Finanziario discovery, inventory, freshness, incremental merge and artifact publishing.

The Streamlit app now adds an operational shell without moving core analytics out of the notebooks:

- notebooks = analytical truth and authoring layer
- exports/artifacts = stable contract layer
- Streamlit = workstation UX, artifact browser, lightweight refresh, notebook launcher and run history

## Stable Contracts

Company valuation outputs:

- `screener_results`, `screener_audit`, `screener_progression`, `screener_summary`
- `finviz_group_summaries`, `finviz_map_payload`, `finviz_watchlists`, `finviz_events`, `finviz_alerts`
- CSV exports under `output/tables`, JSON configs under `output/config`
- HTML hooks: `screener_kpi_cards_html`, `screener_top_table_html`, `screener_audit_html`

Portfolio outputs:

- `portfolio_selection_results`, `portfolio_selection_audit`, `portfolio_selection_progression`, `portfolio_selection_summary`
- CSV exports under `output/tables`, JSON config under `output/config`
- HTML hooks: `portfolio_selection_kpi_html`, `portfolio_selection_results_html`, `portfolio_selection_audit_html`

Missing data behavior is explicit: unavailable fields are skipped in audit/progression tables, not silently fabricated.

## Migration Path

### A. Local Package

Keep notebook cells as orchestration. Move stable entry points into package exports:

- `company_valuation.src.run_integrated_screener`
- `company_valuation.src.run_finviz_platform_layer`
- `portfolio_analysis.src.run_portfolio_selection_layer`
- `src.research_platform_core.*`

Next cleanup would add `pyproject.toml` package metadata and a thin CLI.

### B. Streamlit Prototype

Use exported CSV/JSON contracts as the first backend:

- load `ScreenerResults.csv`, `FinvizMapPayload.csv`, `PortfolioSelectionResults.csv`
- render tabs from dashboard tables
- call module entry points only when a local notebook namespace is available

This avoids coupling Streamlit to notebook execution order.

Implemented MVP:

- `research_platform_app/app.py`
- `research_platform_app/operations.py`
- `research_platform_app/generate_artifacts.py`
- `research_platform_app/pages/1_Valuation_Research.py`
- `research_platform_app/pages/2_Portfolio_Research.py`
- `research_platform_app/pages/3_Screeners_Selection.py`
- `research_platform_app/pages/4_Artifacts_Exports.py`

Run locally with:

```bash
python -m pip install -r research_platform_app/requirements.txt
streamlit run research_platform_app/app.py
```

Optional lightweight regeneration:

```bash
python research_platform_app/generate_artifacts.py --domain all
```

This regenerates Streamlit-facing screener/selection/platform artifacts from existing exports. It does not execute notebook ingestion, valuation, modelling or full refresh cells.

### Orchestration Layer

Files:

- `research_platform_app/orchestration/models.py`: `NotebookJob`, `ParameterSpec`, `ArtifactSpec`, `JobRun`
- `research_platform_app/orchestration/registry.py`: central job registry
- `research_platform_app/orchestration/runner.py`: unified runner interface
- `research_platform_app/orchestration/papermill_runner.py`: primary parameterized notebook runner
- `research_platform_app/orchestration/nbclient_runner.py`: fallback notebook runner
- `research_platform_app/orchestration/job_store.py`: local JSON run history
- `research_platform_app/orchestration/artifact_contracts.py`: artifact validation
- `research_platform_app/orchestration/freshness.py`: artifact freshness monitor
- `research_platform_app/pages/5_Run_Notebooks.py`: operational launcher
- `research_platform_app/pages/6_Run_History.py`: persisted run history

Job lifecycle:

1. User selects a registered job.
2. Streamlit validates parameters.
3. Streamlit either executes synchronously or creates a `PENDING` run and launches a detached worker.
4. Runner writes a `RUNNING` record to `state/job_runs.json`.
5. Papermill executes the notebook and writes an executed copy under `runs/<run_id>/`.
6. If Papermill fails and fallback is allowed, nbclient attempts execution.
7. Artifact contracts are validated against the configured output root.
8. The run is marked `SUCCESS` or `FAILED`, with log and artifact metadata persisted.

Supported base jobs:

- `valuation_research_refresh`
- `portfolio_research_refresh`
- `screener_refresh`

`screener_refresh` is module-only and intentionally low-cost. It rebuilds Streamlit-facing screener/selection/platform artifacts from existing exports.

The local scheduler can keep artifacts fresh:

```bash
python research_platform_app/scheduler.py --interval-seconds 900 --jobs screener_refresh
```

Notebook scheduling is supported by adding notebook job IDs to the command, but should be used deliberately because full notebook runs can be long.

## Canonical Data Platform

`Database Finanziario` on Google Drive is now the primary data source. The app and notebooks discover it through `FINANCIAL_DB_ROOT`, `DB_BASE`, `DATA_PATH`, Colab Drive paths and the local Mac CloudStorage path.

Streamlit exposes it through:

- `research_platform_app/pages/7_Data_Platform.py`

The intended data order is:

1. Database Finanziario
2. local/repo cache
3. free/API provider fallback
4. explicit unavailable state

See `docs/DATA_PLATFORM_INTEGRATION.md`.

Additional data-platform jobs:

- `data_platform_status_refresh`: writes inventory/freshness/provider contracts.
- `europe_prices_incremental_refresh`: refreshes stale Europe/STOXX daily parquet files incrementally.

API-ready contract:

- `output/api_contracts/data_platform_contract.json`

### C. FastAPI + Frontend

Promote the export contracts to API schemas:

- `/company/screener`
- `/company/maps`
- `/company/groups`
- `/portfolio/selection`
- `/portfolio/diagnostics`

The remaining boundary to clean is state management: notebooks currently pass a mutable `globals()` namespace. A future backend should use explicit request/config objects.

## Remaining Technical Debt

- Some dashboard helpers remain domain-specific and intentionally duplicated for readability.
- News/insider/event data are schema-ready but depend on reliable free/provider inputs.
- Notebook execution still owns data ingestion; backend execution needs explicit orchestration objects.
