# Research Platform Definitive Review

This is the canonical map of the local research platform after consolidation.

## 1. Executive State

The project is now organized as a notebook-first, app-backed, Drive-first research platform.

Primary layers:

1. **Database Finanziario / Data Platform**
   - Canonical data lake.
   - Drive-first / cache-second / API-last.
   - Freshness, provider fallback and artifact contracts.

2. **Research Platform Core**
   - `src/research_platform_core/`
   - Shared dataframe, export, HTML and data platform helpers.

3. **Domain Engines**
   - `company_valuation/src/`
   - `portfolio_analysis/src/`
   - `src/smart_money_engine/`
   - `src/ml_stock_lab/`

4. **Notebook Authoring Layer**
   - `company_valuation/notebooks/Company_Valuation_Final_Version.ipynb`
   - `portfolio_analysis/notebooks/Portfolio-Analysis-Model_RESEARCH_PLATFORM_PRO.ipynb`
   - `machine_learning_lab/notebooks/ML_Stock_Lab_Experiments.ipynb`

5. **Streamlit Workstation**
   - `research_platform_app/`
   - Operational viewer, refresh console, artifact browser and research dashboard.

## 2. Canonical App Pages

| Page | Role |
|---|---|
| Home / Overview | Global status, artifact roots, run/freshness summary |
| Valuation Research | Fair value, valuation models, screener context, Smart Money and ML Lab overlays |
| Portfolio Research | Allocation, selection lab, Smart Money overlay, ML Lab overlay |
| Screeners Selection | Company screener, portfolio selection, Smart Money screener, ML Lab screener |
| Artifacts / Exports | Notebook, Data Platform, Smart Money and ML Lab artifact contracts |
| Run Notebooks | Papermill/nbclient/module job launcher |
| Run History | Persistent local run history and logs |
| Data Platform | Database Finanziario inventory, freshness, providers and credentials |
| Smart Money Gov Data | SEC/CFTC/Treasury/USAspending/EU proxy official-source intelligence |
| ML Stock Lab | ML fair value, mispricing, z-score and quintile diagnostics |

## 3. Canonical Packages

### `research_platform_core`

Stable shared utilities:

- `data_platform.py`
- `data_utils.py`
- `export_utils.py`
- `html_utils.py`

Use this for any future package-level helper before adding duplicated notebook code.

### `company_valuation`

Domain layer for:

- valuation dashboard
- Finviz-inspired screener
- company scoring
- dashboard hooks
- research extensions

### `portfolio_analysis`

Domain layer for:

- portfolio engine
- portfolio-aware selection
- dashboard outputs
- ML time-series helper

### `smart_money_engine`

Official-source-first layer:

- SEC 13F, Form 4, 13D/G normalization
- CFTC COT, Treasury TIC
- USAspending
- EU/Italy proxy registry through ESMA/ECB/TED/national regulators
- explainable smart-money scoring

### `ml_stock_lab`

ML equity lab:

- panel loading from Database Finanziario / existing artifacts
- fair value `V_hat = f_t(X)`
- mispricing `(V_hat - V_mkt) / V_mkt`
- z-score cross-sectional signals
- ranking and quintile portfolios
- expected return prediction
- OOS R2, Sharpe, turnover, transaction costs

## 4. Notebook Integration

The two primary notebooks are now orchestration-friendly:

- Data platform bootstrap.
- Smart Money bridge.
- ML Stock Lab bridge.
- Artifact-writing workflow.
- Colab/VS Code compatible path discovery.

Use notebooks for:

- analytical authoring
- long-running experiments
- methodology narrative
- export generation

Do not duplicate package logic inside notebooks. Add new reusable logic under `src/`.

## 5. Artifact Contracts

Canonical output roots:

- Company valuation: `company_valuation/output/`
- Portfolio research: `portfolio_analysis/output/`
- Platform workspace: `output/`
- Smart Money: `output/smart_money/`
- ML Stock Lab: `output/ml_stock_lab/`

Generated local artifacts under `output/smart_money/` and `output/ml_stock_lab/` are ignored by git and can be regenerated.

Important contracts:

- `DataPlatform_*.csv`
- `Screener*.csv`
- `PortfolioSelection*.csv`
- `SmartMoney_*.csv`
- `MLStockLab_*.csv`

The Streamlit app reads these contracts rather than notebook state.

## 6. Current Quality Review

Strengths:

- Data platform is centralized and Drive-first.
- Streamlit is now a workstation, not a passive viewer.
- Valuation, portfolio, Smart Money and ML Lab are integrated instead of isolated.
- Jobs are registered and run-history aware.
- Missing data produces explicit empty artifacts/coverage diagnostics.

Residual risks:

- Many legacy notebooks in the repository still show local modifications unrelated to this platform.
- Some generated output folders remain local-only and should not be committed.
- ML Lab currently uses proxy targets when market value is unavailable; this is explicit via `target_source`.
- EU/Italy Smart Money coverage is currently a framework/proxy layer until national connectors are added.

## 7. Definitive Operating Workflow

1. Check **Data Platform** page.
2. Run `data_platform_status_refresh`.
3. Run valuation and portfolio notebooks only when methodology/artifacts need refresh.
4. Run lightweight jobs:
   - `screener_refresh`
   - `smart_money_government_refresh`
   - `ml_stock_lab_experiments_refresh`
5. Review app pages:
   - Valuation Research
   - Portfolio Research
   - Screeners Selection
   - Smart Money Gov Data
   - ML Stock Lab
6. Use Artifacts / Exports as the source of truth for downstream APIs.

## 8. What Not To Do

- Do not add new large logic directly into Streamlit pages.
- Do not make notebooks depend on live Streamlit state.
- Do not commit run logs, caches, backup notebooks or generated output folders.
- Do not silently fill missing official-source or fundamentals data.
- Do not claim EU ownership coverage is equivalent to SEC/EDGAR coverage.

## 9. Next Refactor If Needed

If the repo keeps growing, the next clean split should be:

- `src/research_platform_core`
- `src/company_valuation_platform`
- `src/portfolio_research_platform`
- `src/smart_money_engine`
- `src/ml_stock_lab`
- `research_platform_app`

The notebooks should remain thin authoring layers.

## 10. Audit Artifact

A machine-readable inventory was generated at:

`output/review/Research_Platform_Definitive_Audit.csv`

## 11. Maintenance Contract

The operational Codex/developer maintenance guide is:

`docs/CODEX_CANONICAL_MAINTENANCE_GUIDE.md`

Use it before changing notebooks, package contracts, Streamlit pages or artifact schemas. It contains the current validation commands and the anti-regression checklist.
