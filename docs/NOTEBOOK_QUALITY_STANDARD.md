# Notebook Quality Standard

This standard applies to every professional notebook in the local ML Trading workspace.

## Canonical Notebook Standard

Use `company_valuation/notebooks/Company_Valuation_Final_Version.ipynb`
as the canonical pattern for future professional notebooks and dashboards.

Even when building a new notebook from scratch, preserve the company valuation pattern:

- central config/state layer with one canonical request object
- visible Control Center near the top of the notebook
- interactive user controls for company/ticker selection, data sources, assumptions, scenarios, model depth, diagnostics, and output options
- robust local-cache/API ingestion with clear source priority
- validated data pipeline with diagnostics before research execution
- lag-safe feature engineering and explicit formulas
- valuation, risk, quality, peer, model, scenario, sensitivity, and dashboard analytics
- governance tables for data quality, source mix, missingness, robustness, and config snapshots
- final polished Plotly/HTML dashboard, not scattered chart output
- exports under `DB_BASE`, never heavy data inside Git

The target user experience is a modular financial analytics application that happens to run inside Colab/Jupyter. The notebook should feel navigable and product-grade, not like a long script.

## Dashboard Requirements

Professional dashboards must use a clear workflow:

```text
Input Setup -> Apply configuration -> Diagnostics -> Run research -> Results Dashboard -> Saved Outputs
```

The first visible operating surface must be a **Control Center** with:

- an obvious header
- a mini summary/status card
- navigation between `Input Setup`, `Diagnostics`, `Results Dashboard`, and `Saved Outputs`
- progressive disclosure for advanced assumptions
- save/reload preset support where possible

The final dashboard must use dedicated tabs adapted to the analysis. For company and valuation work, use the company valuation notebook as the visual and interaction reference. For portfolio work, adapt these views without changing the canonical standard:

```text
Executive Summary
Portfolio
Peers
Valuation
Risk
Models
Backtest
Outputs
```

Data quality and methodology can be embedded inside these views, but should never be hidden as raw output. Plotly charts should be presented as dashboard panels with titles, descriptions, and interactive controls.

## Required User Experience

Every notebook should expose a friendly configuration layer near the start:

- ticker/universe/sector selection
- data source selection
- model depth selection
- date range
- forecast horizon
- run toggles for backtest, ablation, robustness, and output saving

Use:

```python
from src.notebook_standards import (
    NotebookControlDefaults,
    build_control_panel,
    configure_notebook_runtime,
    resolve_notebook_config,
)

runtime = configure_notebook_runtime()
CONTROL_PANEL = build_control_panel(NotebookControlDefaults(analysis_name="my_analysis"))
NOTEBOOK_CONFIG = resolve_notebook_config(CONTROL_PANEL, EXPERIMENT)
```

The widget layer must also work headlessly: if `ipywidgets` is unavailable, the notebook uses defaults.

## Required Structure

Research notebooks use the universal 0-14 sections:

```text
0 Setup & Config
1 Data Ingestion
2 Cleaning & Alignment
3 Feature Engineering
4 Targets & Labels
5 Descriptive Stats
6 Exploratory / Event Study
7 Single-Factor Diagnostics
8 Statistical Models
9 ML Walk-Forward
10 Feature Ablation
11 Backtest / Strategy Evaluation
12 Interpretability
13 Robustness Checks
14 Final Summary
```

Educational chapter notebooks can keep their pedagogical flow, but must still add:

- a setup/config cell
- clear data source handling
- output paths under `DB_BASE`
- chart and table exports where the notebook produces results
- no heavy data saved inside Git

## Graphics

Minimum professional graphics standard:

- at least two Plotly charts for final/professional notebooks
- clear titles, axis labels, and units
- consistent color palette
- export chart HTML to `DB_BASE / "notebook_exports/charts"`
- static matplotlib/seaborn is allowed for educational notebooks, but final notebooks should prefer Plotly

## Model Depth

Use the model depth selector to control compute:

- `quick`: fast smoke test, small universe, minimal models
- `standard`: default professional run
- `institutional`: broader diagnostics, robustness, and reporting
- `research_deep_dive`: heavier model comparison and sensitivity checks

Portfolio notebooks should also expose analysis detail levels:

- `executive`: concise KPI summary and decision layer
- `professional`: core formulas, diagnostics, charts, and tables
- `institutional`: governance, robustness, factor decomposition, and audit trail
- `research_deep_dive`: project-chapter links, advanced model labs, and maximum transparency

At minimum, portfolio model coverage should include:

- linear models: OLS/Ridge/Lasso/ElasticNet where appropriate
- machine learning: Random Forest and Extra Trees
- gradient boosting
- time-series diagnostics and walk-forward logic
- factor research
- unsupervised peer/similarity views
- governed deep-learning/recurrent-net integration hooks when trained artifacts or project datasets exist

## Data Sources

All notebooks should prefer real data first and fall back safely:

1. `database_finanziario`
2. `local_repo_cache`
3. public API source such as yfinance
4. synthetic fallback, clearly marked with `_synthetic`

Persistent data and outputs must live under:

```text
/Users/itsgennymac/Library/CloudStorage/GoogleDrive-sfn.gns@gmail.com/Il mio Drive/Database Finanziario
```

## Benchmark Findings

The Drive benchmark notebooks show the target direction:

- `company_valuation/notebooks/Company_Valuation_Final_Version.ipynb`: canonical enterprise standard with widgets, source handling, valuation depth, formulas, exports, dashboard UX, and Drive integration.
- Portfolio analysis notebooks should copy the company valuation interaction and dashboard standard, not define a competing canonical standard.
- Older `Portfolio-Analysis-Model.ipynb` variants should be treated as source inspiration only.
