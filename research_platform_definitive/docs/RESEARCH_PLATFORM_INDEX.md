# Research Platform Index

Start here.

## Run The App

```bash
streamlit run research_platform_app/app.py
```

## Main Notebooks

- `company_valuation/notebooks/Company_Valuation_Final_Version.ipynb`
- `portfolio_analysis/notebooks/Portfolio-Analysis-Model_RESEARCH_PLATFORM_PRO.ipynb`
- `machine_learning_lab/notebooks/ML_Stock_Lab_Experiments.ipynb`

## Core Packages

- `src/research_platform_core`
- `company_valuation/src`
- `portfolio_analysis/src`
- `src/smart_money_engine`
- `src/ml_stock_lab`

## Key Docs

- `docs/CODEX_CANONICAL_MAINTENANCE_GUIDE.md`
- `docs/RESEARCH_PLATFORM_DEFINITIVE_REVIEW.md`
- `docs/RESEARCH_PLATFORM_ARCHITECTURE.md`
- `docs/RESEARCH_PLATFORM_ORCHESTRATION.md`
- `docs/DATA_PLATFORM_INTEGRATION.md`
- `docs/SMART_MONEY_GOVERNMENT_DATA_ENGINE.md`
- `docs/ML_STOCK_LAB_OVERVIEW.md`
- `docs/BANKING_DATA_PIPELINE.md`

## Validation

```bash
python3 -m compileall -q src/ml_stock_lab src/smart_money_engine src/research_platform_core company_valuation/src portfolio_analysis/src research_platform_app
python3 scripts/smoke_ml_stock_lab.py
python3 scripts/smoke_smart_money_engine.py
python3 scripts/smoke_banking_data.py
python3 research_platform_app/smoke_checks.py
python3 scripts/validate_notebooks.py
python3 scripts/validate_artifacts.py
```

## Regenerable Local Artifacts

- `output/smart_money/`
- `output/ml_stock_lab/`
- `research_platform_app/runs/`
- `research_platform_app/state/`

These are intentionally local/runtime outputs.
