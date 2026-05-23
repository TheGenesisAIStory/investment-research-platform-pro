# Definitive Notebook And App Review

Date: 2026-05-20

## What Was Reviewed

The workspace contains many legacy/course notebooks. The operational research platform is narrower and now collected in this definitive bundle.

Canonical notebooks:

- `company_valuation/notebooks/Company_Valuation_Final_Version.ipynb`
- `portfolio_analysis/notebooks/Portfolio-Analysis-Model_RESEARCH_PLATFORM_PRO.ipynb`
- `machine_learning_lab/notebooks/ML_Stock_Lab_Experiments.ipynb`

Notebook index:

- `notebooks/README.md`

App:

- `research_platform_app/app.py`
- `research_platform_app/pages/1_Valuation_Research.py`
- `research_platform_app/pages/2_Portfolio_Research.py`
- `research_platform_app/pages/3_Screeners_Selection.py`
- `research_platform_app/pages/4_Artifacts_Exports.py`
- `research_platform_app/pages/5_Run_Notebooks.py`
- `research_platform_app/pages/6_Run_History.py`
- `research_platform_app/pages/7_Data_Platform.py`
- `research_platform_app/pages/8_Smart_Money_Gov_Data.py`
- `research_platform_app/pages/9_ML_Stock_Lab.py`

## Finding

The ML Equity Lab was not missing. It was located at:

```text
machine_learning_lab/notebooks/ML_Stock_Lab_Experiments.ipynb
```

For easier discovery, it is listed in:

```text
notebooks/README.md
```

## Decision

`research_platform_definitive/` is now the practical folder to use for demo, review and future maintenance. It includes:

- a friendly notebook index;
- execution-compatible notebook paths;
- Streamlit app;
- core packages;
- domain modules;
- docs;
- validators;
- lightweight output artifacts.

## Company Valuation Section 0 Update

The Company Valuation notebook setup was hardened after review:

- Section `0.1.7` now creates `config/` and `utils/` safely before later config-writing cells.
- API keys can be pasted either as bulk `.env` style text or as individual provider fields.
- Keys are stored only in runtime environment variables and are not printed.
- Valuation assumptions can be set via preset, manually, or estimated from 5Y market-price history when `yfinance` data are available.
- Discount rate uses a CAPM-style proxy with bounded beta and market risk premium.
- Terminal growth is capped conservatively as a mature-company assumption.
- Section `0.1.8` writes `config/presets.py` without using fragile `%%writefile` magic.
- If the interactive selection was not applied, Section `0.7` now uses an explicit safe fallback unless `COMPANY_VALUATION_STRICT_UI=1`.

## What Is Not Included As Canonical

Backup notebooks, course notebooks, executed notebook copies and Streamlit run history are not canonical. They remain useful references, but should not drive future maintenance decisions.

## 2026-05-22 Harmonization Update

The duplicate notebook copies were removed from active source paths to prevent drift:

- exact duplicate notebook aliases were deleted;
- the non-identical stale company valuation copy was archived in `archive/notebook_duplicates/`;
- generated tested notebook copies were archived in `archive/generated_notebooks/`;
- `notebooks/` is now an index-only folder.

## 2026-05-23 Status And Verification Update

The project status document was added at:

```text
docs/PROJECT_STATUS_2026-05-23.md
```

Verification performed:

- `pytest` over the QuantDinger bridge and definitive bundle tests: 27 passed.
- Definitive bundle, notebook, artifact and laboratorio validators: passed.
- Orchestration, Data/API, Data Center, ML Stock Lab, Smart Money and OHLCV smoke checks: passed.
- Python compile check over canonical packages, integrations, scripts and app: passed.

The only code change needed during this pass was `tests/conftest.py` inside the definitive bundle, so pytest can import canonical packages from `src/` without manual `PYTHONPATH`.
