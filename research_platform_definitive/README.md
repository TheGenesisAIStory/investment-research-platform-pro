# Research Platform Definitive Bundle

This folder is the curated, demo-ready and maintenance-ready bundle for the Investment Research Platform Pro.

It exists to solve the previous sprawl problem: the canonical notebooks, app, reusable packages, docs, validation scripts and lightweight demo artifacts are collected in one place.

## Start Here

Open these first:

1. `company_valuation/notebooks/Company_Valuation_Final_Version.ipynb`
2. `portfolio_analysis/notebooks/Portfolio-Analysis-Model_RESEARCH_PLATFORM_PRO.ipynb`
3. `machine_learning_lab/notebooks/ML_Stock_Lab_Experiments.ipynb`
4. `data_api_management/notebooks/Data_API_Management_Colab.ipynb`
5. `research_platform_app/app.py`
6. `data_api_app.py`
7. `docs/CODEX_CANONICAL_MAINTENANCE_GUIDE.md`
8. `docs/COLAB_LOCAL_DRIVE_WORKFLOW.md`
9. `docs/DATA_CATALOG.md`
10. `docs/API_USAGE.md`
11. `docs/MAINTENANCE.md`
12. `docs/PROJECT_STATUS_2026-05-23.md`
13. `docs/PROJECT_STATUS_FINAL.md`
14. `docs/DATA_CENTER_OPERATING_MODEL.md`
15. `docs/FORMULE_STRATEGIE_REASONING.md`
16. `docs/LLM_LAB_VIBE_TRADING.md`
17. `docs/MOBILE_QUANTDINGER_HANDOFF.md`

Reference/lab material restored from the old Drive project is in:

```text
laboratorio/
```

Use it as a research library, not as the production source of truth.
It has been curated down to platform-relevant notebooks only: data platform, factor research, ML Stock Lab, portfolio/risk, valuation/regulatory text, smart-money data and optional microstructure research. Generic teaching notebooks and unrelated examples were removed from the definitive bundle.

The Streamlit app exposes the curated inventory in:

```text
research_platform_app/pages/10_Laboratorio_Research_Library.py
```

The common data/API control center is available as a standalone app:

```bash
streamlit run research_platform_definitive/data_api_app.py
```

and inside the main app:

```text
research_platform_app/pages/11_Data_API_Control_Center.py
```

It now includes dashboard KPIs, provider health, credential status, batch exports, analytics, a settings panel backed by `config/data_api_control.yaml`, and an `ml_stock_lab` manifest bridge.

The Data Center enhancement adds Fama-French, AQR, risk-factor, equity-universe, FX and commodity loaders under:

```text
src/research_platform_core/loaders/
```

Safe population and validation entrypoints:

```bash
python scripts/initial_setup.py
python scripts/initial_setup.py --execute --max-items 5
python scripts/populate_research_database.py
python scripts/llm_lab_cli.py packet
python scripts/sync_prices.py --execute --universes sp500,ftsemib --max-symbols 25
python scripts/sync_fundamentals.py --execute --universes sp500,ftsemib --max-symbols 10
python scripts/validate_data.py
```

The ML Equity Lab notebook you were looking for is:

```text
research_platform_definitive/machine_learning_lab/notebooks/ML_Stock_Lab_Experiments.ipynb
```

`notebooks/README.md` is now the human-friendly notebook index. It intentionally does not contain duplicate source notebooks.

## Run The App

From the repository root:

```bash
streamlit run research_platform_definitive/research_platform_app/app.py
```

Or from this folder:

```bash
streamlit run research_platform_app/app.py
```

## Sync To Google Drive For Colab And Local Runs

Preferred from the repository root:

```bash
./scripts/sync_repo_to_google_drive.sh
```

This creates/updates:

```text
MyDrive/GitHub/investment-research-platform-pro/
```

You can also sync only this definitive bundle from this folder:

```bash
./scripts/sync_definitive_to_drive.sh
```

This creates/updates:

```text
MyDrive/machine-learning-for-trading/research_platform_definitive/
```

By default, sync does not copy virtual environments, runtime history, local archives, Python caches or transient data cache. Colab should use `MyDrive/Database Finanziario/` as the data source.

## Validate The Bundle

From this folder:

```bash
python3 scripts/validate_definitive_bundle.py
python3 scripts/validate_laboratorio.py
python3 scripts/validate_notebooks.py
python3 scripts/validate_artifacts.py
python3 scripts/smoke_data_api_control.py
python3 scripts/smoke_data_center_enhancement.py
python3 scripts/smoke_ml_stock_lab.py
python3 scripts/smoke_smart_money_engine.py
python3 research_platform_app/smoke_checks.py
```

From the repository root:

```bash
python3 research_platform_definitive/scripts/validate_definitive_bundle.py
python3 research_platform_definitive/scripts/validate_laboratorio.py
python3 research_platform_definitive/scripts/validate_notebooks.py
python3 research_platform_definitive/scripts/validate_artifacts.py
python3 research_platform_definitive/scripts/smoke_data_api_control.py
python3 research_platform_definitive/scripts/smoke_data_center_enhancement.py
python3 research_platform_definitive/research_platform_app/smoke_checks.py
```

## Folder Map

```text
research_platform_definitive/
├── notebooks/                         # human-friendly notebook index, no duplicate sources
├── laboratorio/                       # curated reference/lab notebooks from old Drive project
├── company_valuation/notebooks/       # execution-compatible valuation notebook path
├── portfolio_analysis/notebooks/      # execution-compatible portfolio notebook path
├── machine_learning_lab/notebooks/    # execution-compatible ML lab notebook path
├── data_api_management/               # common Data/API Colab notebook and operating notes
├── research_platform_app/             # Streamlit workstation
├── src/                               # reusable core packages
├── company_valuation/src/             # valuation domain modules
├── portfolio_analysis/src/            # portfolio domain modules
├── output/                            # lightweight demo/runtime artifacts
├── data/sample/                       # tracked sample extracts from the final research DB
├── docs/                              # canonical documentation
├── config/data_sources.yaml           # strategic Data Center source map
├── config/rate_limits.yaml            # provider throttling defaults
├── config/scheduler_config.yaml       # recurring job template
└── scripts/                           # validators and smoke checks
```

## Review Result

The canonical notebooks are present and validated:

| Notebook | Friendly path | Canonical execution path | Status |
|---|---|---|---|
| Company Valuation | `notebooks/README.md` | `company_valuation/notebooks/Company_Valuation_Final_Version.ipynb` | canonical |
| Portfolio Research | `notebooks/README.md` | `portfolio_analysis/notebooks/Portfolio-Analysis-Model_RESEARCH_PLATFORM_PRO.ipynb` | canonical |
| ML Equity / Stock Lab | `notebooks/README.md` | `machine_learning_lab/notebooks/ML_Stock_Lab_Experiments.ipynb` | canonical |

Legacy notebook dumps are not part of the operating source of truth. Non-identical stale duplicates are kept under `archive/notebook_duplicates/` for traceability; generated executed notebooks are kept under `archive/generated_notebooks/`. Only the curated `laboratorio/` reference library remains, and each notebook has a documented promotion target in `laboratorio/LABORATORIO_MANIFEST.csv`.

## Operating Model

- Notebooks are the analytical authoring layer.
- `src/` packages own reusable logic.
- Streamlit is the operational research console.
- `output/` contains artifact contracts consumed by the app.
- `data/sample/` contains small CSV fixtures generated from the final database.
- `src/research_platform_core/research_database.py` owns the final SQLite schema.
- `src/research_platform_core/llm_lab.py` owns prompt packets and provider readiness.
- Database Finanziario remains the preferred external data source when available.
