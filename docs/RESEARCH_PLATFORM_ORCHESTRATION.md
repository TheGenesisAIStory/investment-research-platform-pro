# Research Platform Orchestration

## Purpose

The Streamlit app can now operate as a lightweight research workstation. It launches supported notebook or module jobs, stores run metadata, captures logs, and validates output artifacts.

## Model

- Notebook: analytical source of truth.
- Artifact: CSV/JSON/HTML contract between notebooks, modules and app.
- Streamlit: operational UX for launching, monitoring and inspecting research outputs.

## Runners

Primary runner:

- Papermill for parameterized notebooks.

Fallback runner:

- nbclient for legacy notebooks that can execute without a fully tagged Papermill parameters cell.

Module runner:

- Lightweight refresh for screener, selection and platform artifacts from existing exports.

Background runner:

- Streamlit can now create a run record and spawn `research_platform_app/worker.py` in a detached subprocess.
- The UI remains responsive while the worker updates `state/job_runs.json` and `runs/<run_id>/run.log`.

## Run State

Run history is persisted in:

```text
research_platform_app/state/job_runs.json
```

Run outputs are written under:

```text
research_platform_app/runs/<run_id>/
```

Fields include:

- `run_id`
- `job_id`
- `status`
- `created_at`
- `started_at`
- `finished_at`
- `parameters`
- `runner_type`
- `notebook_path`
- `output_notebook_path`
- `artifacts_detected`
- `log_path`
- `error_message`

## Job Registry

Jobs live in:

```text
research_platform_app/orchestration/registry.py
```

Current jobs:

- `valuation_research_refresh`
- `portfolio_research_refresh`
- `screener_refresh`

Each job declares parameter specs, expected artifacts, runner type, timeout, output domain and enabled state.

## Artifact Contracts

Artifact specs declare:

- label
- relative path
- required/optional
- freshness window

Validation checks existence, size, modified timestamp, age and freshness status.

## Adding a Notebook Job

1. Add a tagged `parameters` cell to the notebook when possible.
2. Register a `NotebookJob`.
3. Add `ParameterSpec` definitions.
4. Add `ArtifactSpec` definitions.
5. Run:

```bash
python research_platform_app/smoke_checks.py
```

## Known Limits

- Notebook execution can run in a background subprocess from Streamlit.
- The app does not manipulate live notebook kernels.
- Full ingestion, valuation and modelling logic still lives in notebooks/modules.
- A local scheduler is available via `research_platform_app/scheduler.py`.
- Production-grade distributed scheduling is still intentionally out of scope.

## Keeping Artifacts Current

Safe one-shot:

```bash
python research_platform_app/scheduler.py --once --jobs screener_refresh
```

Continuous local scheduler:

```bash
python research_platform_app/scheduler.py --interval-seconds 900 --jobs screener_refresh
```

The Streamlit Home and Run Notebooks pages can also start/stop this scheduler. The process state is stored in:

```text
research_platform_app/state/scheduler_state.json
```

Full notebook scheduling is possible but should be intentional:

```bash
python research_platform_app/scheduler.py --interval-seconds 3600 --jobs screener_refresh valuation_research_refresh portfolio_research_refresh
```
