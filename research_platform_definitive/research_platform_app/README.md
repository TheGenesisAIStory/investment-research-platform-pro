# Research Platform Streamlit App

Lightweight Streamlit frontend over notebook-generated research artifacts.

## Run

From the repository root:

```bash
python -m pip install -r research_platform_app/requirements.txt
streamlit run research_platform_app/app.py
```

Run orchestration smoke checks:

```bash
python research_platform_app/smoke_checks.py
```

## Data Model

The app is notebook-first by design:

1. notebooks generate CSV/JSON/HTML artifacts,
2. local modules define reusable computation,
3. Streamlit presents exported artifacts for exploration,
4. optional lightweight actions can regenerate screener/selection artifacts from existing exports.

Default roots:

- `company_valuation/output`
- `portfolio_analysis/output`
- `output`

Override them in the sidebar or with:

- `COMPANY_VALUATION_OUTPUT_ROOT`
- `PORTFOLIO_OUTPUT_ROOT`
- `RESEARCH_PLATFORM_OUTPUT_ROOT`
- `FINANCIAL_DB_ROOT`

The preferred `FINANCIAL_DB_ROOT` is:

```text
/Users/itsgennymac/Library/CloudStorage/GoogleDrive-sfn.gns@gmail.com/Il mio Drive/Database Finanziario
```

In Colab the compatible path is usually:

```text
/content/drive/MyDrive/Database Finanziario
```

## Data Platform Mode

Open `Data Platform` in the sidebar/app navigation to inspect the canonical Drive database:

- file inventory by role,
- catalog tables,
- provider registry,
- masked credential status,
- freshness summary,
- artifact publish/sync controls.

Operational rule:

1. read from Database Finanziario first,
2. read local/repo cache second,
3. call APIs only for missing/stale/incomplete datasets,
4. write refreshed datasets back to Drive with metadata,
5. avoid repeated full downloads when incremental append/dedup is possible.

Incremental Europe/STOXX price refresh:

```bash
python research_platform_app/scheduler.py --once --jobs europe_prices_incremental_refresh
```

API-ready data-platform contract:

```text
output/api_contracts/data_platform_contract.json
```

## Lightweight Artifact Generation

From the repository root:

```bash
python research_platform_app/generate_artifacts.py --domain all
python research_platform_app/generate_artifacts.py --domain company
python research_platform_app/generate_artifacts.py --domain portfolio
```

This does not run the full notebooks. It only rebuilds low-cost modules such as the company screener, Finviz-style platform tables, and portfolio selection layer when suitable base exports are available.

## Orchestration Mode

The app includes two operational pages:

- `Run Notebooks`: launch registered notebook/module jobs, edit parameters, inspect logs and validate expected artifacts.
- `Run History`: inspect local JSON run history, logs, parameters, output notebooks and rerun previous jobs.

Primary notebook execution uses Papermill. If Papermill fails or the notebook is not fully parameter-ready, the runner falls back to nbclient for compatible notebooks.

Run history is stored locally under:

```text
research_platform_app/state/job_runs.json
research_platform_app/runs/<run_id>/
```

Background execution is supported from the UI. Streamlit creates a run record, spawns `research_platform_app/worker.py`, and returns immediately while the worker updates status/logs.

To run a freshness scheduler:

```bash
python research_platform_app/scheduler.py --once --jobs data_platform_status_refresh screener_refresh
python research_platform_app/scheduler.py --interval-seconds 900 --jobs data_platform_status_refresh screener_refresh
```

The safe default keeps lightweight screener/selection artifacts current. Add notebook job IDs only when you intentionally want scheduled full notebook execution:

```bash
python research_platform_app/scheduler.py --interval-seconds 3600 --jobs data_platform_status_refresh screener_refresh valuation_research_refresh portfolio_research_refresh --notebook-offhour-start 20 --notebook-offhour-end 7
```

The Home and Run Notebooks pages also expose Start/Stop controls for the local scheduler process.

## Environment Variables

- `COMPANY_VALUATION_OUTPUT_ROOT`
- `PORTFOLIO_OUTPUT_ROOT`
- `RESEARCH_PLATFORM_OUTPUT_ROOT`

## Troubleshooting

- If a notebook run fails immediately, install `papermill`, `nbclient` and `nbformat` from `requirements.txt`.
- If Papermill warns about a missing `parameters` cell, use the Run Notebooks page to dry-check or insert a tagged `parameters` cell with backup. The nbclient fallback can still execute many legacy notebooks by injecting parameters at the top.
- If a job succeeds but shows missing required artifacts, run the notebook export cells or verify that output roots in the sidebar point to the intended folders.
- Heavy notebook execution can take time. The app stores logs in `research_platform_app/runs/<run_id>/run.log`.

## Add a Runnable Notebook

1. Add a `NotebookJob` in `research_platform_app/orchestration/registry.py`.
2. Define `ParameterSpec` entries with defaults and descriptions.
3. Define required and optional `ArtifactSpec` contracts.
4. Prefer a tagged Papermill `parameters` cell in the notebook.
5. Test with `python research_platform_app/smoke_checks.py`.
