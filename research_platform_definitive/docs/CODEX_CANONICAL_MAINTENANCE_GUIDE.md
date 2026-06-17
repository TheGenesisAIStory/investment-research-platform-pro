# Codex Canonical Maintenance Guide

Last definitive review: 2026-05-20

This guide is the operational maintenance contract for the Research Platform. It converts the project manifesto into concrete rules that Codex or any future maintainer should follow before changing notebooks, packages, app pages or artifact contracts.

## 0. Executive Summary

Current validated state:

- Canonical architecture documented and synchronized with the real repository layout.
- ML Stock Lab integrated as `src/ml_stock_lab/`.
- Smart Money Government Data Engine integrated as `src/smart_money_engine/`.
- Streamlit workstation available at `research_platform_app/app.py`.
- Notebook and artifact validators available under `scripts/`.
- Smoke tests available for the core analytical modules and orchestration layer.

Every future change must preserve:

- Colab compatibility.
- Notebook-first workflow.
- Artifact contracts consumed by the app.
- Drive-first / cache-second / API-last data behavior.
- Documentation updates when architecture or public contracts change.

## 1. Source Of Truth

Read these first:

- `docs/CODEX_CANONICAL_MAINTENANCE_GUIDE.md`
- `docs/RESEARCH_PLATFORM_DEFINITIVE_REVIEW.md`
- `docs/RESEARCH_PLATFORM_INDEX.md`
- `docs/ML_STOCK_LAB_OVERVIEW.md`
- `docs/SMART_MONEY_GOVERNMENT_DATA_ENGINE.md`
- `output/review/Research_Platform_Definitive_Audit.csv`

If code and documentation disagree, treat the documentation as the intended contract, then update the code or documentation explicitly. Do not silently drift.

## 2. Canonical Structure

```text
machine-learning-for-trading/
├── src/
│   ├── research_platform_core/
│   ├── smart_money_engine/
│   └── ml_stock_lab/
├── company_valuation/
│   ├── src/
│   └── notebooks/Company_Valuation_Final_Version.ipynb
├── portfolio_analysis/
│   ├── src/
│   └── notebooks/Portfolio-Analysis-Model_RESEARCH_PLATFORM_PRO.ipynb
├── machine_learning_lab/notebooks/ML_Stock_Lab_Experiments.ipynb
├── research_platform_app/
│   ├── app.py
│   └── pages/
│       ├── 0_🏠_Home.py
│       ├── 1_📡_Smart_Money_Macro.py
│       ├── 2_🔬_Valuation_Research.py
│       ├── 3_📁_Portfolio_Research.py
│       ├── 4_🔍_Screener_Builder.py
│       ├── 5_📤_Export_Center.py
│       ├── 6_🧪_Notebook_Runner.py
│       ├── 7_⚙️_Run_Hi_Freq_Engine.py
│       ├── 8_🗄️_Data_Platform.py
│       └── 9_ML_Stock_Lab.py
├── output/
│   ├── ml_stock_lab/
│   ├── smart_money/
│   └── review/
└── docs/
```

Generated runtime outputs under `output/ml_stock_lab/`, `output/smart_money/`, `research_platform_app/runs/` and `research_platform_app/state/` are local/regenerable and should not be treated as source code.

## 3. Non-Negotiable Principles

1. **Single source of truth per domain**
   - Shared helpers belong in `src/research_platform_core/`.
   - ML fair value, signals, screening and quintiles belong in `src/ml_stock_lab/`.
   - Official-source smart money intelligence belongs in `src/smart_money_engine/`.
   - Do not copy-paste analytical functions between notebooks.

2. **Notebook-first, package-backed**
   - Notebooks remain the authoring and methodology layer.
   - Packages hold reusable computation.
   - Streamlit reads artifact contracts and orchestrates runs; it must not become a hidden analytical engine.

3. **Drive-first / cache-second / API-last**
   - Prefer Database Finanziario and local artifacts before API calls.
   - Never silently fabricate data.
   - If a proxy is used, expose it through provenance columns such as `target_source`, `source`, `updated_at` or a dedicated diagnostics table.

4. **Artifact contracts are public interfaces**
   - App pages and future APIs consume files, not notebook state.
   - New artifact files must have stable names, minimal schemas and validation.

## 4. Colab Bootstrap Pattern

Every canonical notebook should have an early setup cell that:

- discovers the project root;
- mounts Google Drive when running in Colab;
- inserts `PROJECT_ROOT`, `PROJECT_ROOT/src`, `company_valuation/src` and `portfolio_analysis/src` into `sys.path`;
- reports the detected environment and data roots;
- fails with a readable error when the repository is not available.

The current canonical notebooks use `Portable Research Platform Bootstrap - Canonical v1.2`.
It supports local VS Code, Google Drive desktop sync, Colab Drive paths and an optional Colab GitHub clone fallback.
For the operating workflow and Drive sync command, see `docs/COLAB_LOCAL_DRIVE_WORKFLOW.md`.

Use package-directory checks, not obsolete single-file checks:

```python
from pathlib import Path
import os
import sys

def setup_colab_environment(verbose=True):
    config = {}
    try:
        from google.colab import drive
        if not os.path.exists("/content/drive/MyDrive"):
            drive.mount("/content/drive")
        config["environment"] = "colab"
    except ImportError:
        config["environment"] = "local"

    candidates = [
        Path("/content/drive/MyDrive/GitHub/machine-learning-for-trading"),
        Path("/content/drive/MyDrive/machine-learning-for-trading"),
        Path.cwd(),
        Path.cwd().parent,
        Path.cwd().parent.parent,
    ]

    project_root = None
    for candidate in candidates:
        if (candidate / "src" / "research_platform_core").exists():
            project_root = candidate
            break

    if project_root is None:
        raise FileNotFoundError(
            "PROJECT_ROOT not found. Open or clone the full repository before running this notebook."
        )

    for rel in ["", "src", "company_valuation/src", "portfolio_analysis/src"]:
        path = str(project_root / rel)
        if path not in sys.path:
            sys.path.insert(0, path)

    config["PROJECT_ROOT"] = project_root
    config["FINANCIAL_DB_ROOT"] = Path("/content/drive/MyDrive/Database Finanziario")
    globals().update(config)

    if verbose:
        print(f"PROJECT_ROOT: {project_root}")
        print(f"Environment: {config['environment']}")
        print(f"FINANCIAL_DB_ROOT: {config['FINANCIAL_DB_ROOT']}")
    return config

CONFIG = setup_colab_environment()
```

## 5. Robust Import Pattern

Use this pattern in notebooks and Streamlit pages when imports need to survive both package and local-path contexts:

```python
try:
    from src.ml_stock_lab import lab
    from src.research_platform_core import read_dataset
except ModuleNotFoundError:
    from ml_stock_lab import lab
    from research_platform_core import read_dataset
```

Avoid broad `except: pass` imports. If an optional dependency is missing, show a clear fallback state.

## 6. Canonical Validation Commands

```bash
python3 -m compileall -q src/ml_stock_lab src/smart_money_engine src/research_platform_core company_valuation/src portfolio_analysis/src research_platform_app
python3 scripts/smoke_ml_stock_lab.py
python3 scripts/smoke_smart_money_engine.py
python3 research_platform_app/smoke_checks.py
python3 scripts/validate_notebooks.py
python3 scripts/validate_artifacts.py
```

The notebook validator intentionally checks only canonical notebooks. Legacy and backup notebooks can remain in the repository, but they are not allowed to define the platform contract.

For a stricter notebook hygiene pass:

```bash
python3 scripts/validate_notebooks.py --strict
```

Default mode reports hardcoded local paths and other hygiene issues as warnings; strict mode promotes those warnings to failures.

Quick pre-commit check:

```bash
python3 -m compileall -q src/ml_stock_lab src/smart_money_engine src/research_platform_core company_valuation/src portfolio_analysis/src research_platform_app && \
python3 scripts/smoke_ml_stock_lab.py && \
python3 scripts/validate_notebooks.py && \
python3 scripts/validate_artifacts.py && \
echo "All validations passed"
```

## 7. Artifact Contract Expectations

### ML Stock Lab

Expected output root: `output/ml_stock_lab/tables/`

- `MLStockLab_panel.csv`
- `MLStockLab_signals.csv`
- `MLStockLab_metrics.csv`
- `MLStockLab_quintile_returns.csv`

Minimum columns:

- Panel: `ticker`, `date`, `market_value`, `target_source`; at least 39 columns
- Signals: `ticker`, `date`, `fair_value_hat`, `mispricing_rel`, `zscore`, `rank`, `signal`, `score`, `target_source`; at least 43 columns
- Metrics: `status`, `model`, `rows`, `target`, `target_source`, `r2_os`, `sharpe_long_short`
- Quintiles: `date`, `quantile`, `return` when non-empty

### Smart Money Government Data Engine

Expected output root: `output/smart_money/`

- `SmartMoneyManifest.json`
- `tables/SmartMoney_source_registry.csv`
- `tables/SmartMoney_coverage.csv`
- `tables/SmartMoney_smart_money_scores.csv`
- `reports/smart_money_government_data_report.html`

Empty analytical tables are allowed when official source files are absent, but coverage diagnostics must explain the gap.

Minimum source registry columns: `dataset_id`, `region`, `authority`, `frequency`, `priority`.

The project currently emits multiple Smart Money tables plus a manifest instead of a single monolithic `SmartMoney_GovernmentData.csv`. The stable public entry points are the manifest, source registry, coverage table, score table and HTML report listed above.

## 8. Automated Anti-Regression Checks

`scripts/validate_notebooks.py` checks:

- canonical notebook files exist;
- notebook JSON is valid;
- required platform bridge text is present;
- early setup/path markers exist;
- hardcoded personal path patterns are reported;
- silent `except: pass` patterns are reported.

`scripts/validate_artifacts.py` checks:

- ML Stock Lab and Smart Money canonical artifacts exist or can be regenerated by lightweight package runners;
- minimum required columns are present;
- ML panel/signals expose `target_source`;
- minimum ML artifact width is preserved;
- Smart Money manifest is readable JSON.

## 9. Anti-Patterns

- Do not duplicate logic between notebooks.
- Do not add Streamlit-only analytics that bypass package modules.
- Do not hardcode personal local paths in core modules.
- Do not silently generate synthetic fundamentals or market values.
- Do not commit run logs, caches, executed notebook copies or local generated outputs.
- Do not claim EU regulatory coverage is equivalent to SEC/EDGAR coverage.

## 10. New Feature Checklist

Before adding a feature:

- Confirm the canonical owner module.
- Search for existing implementation before writing a new helper.
- Keep notebooks as thin orchestration/narrative layers.
- Add type hints and short docstrings for reusable package code.
- Add or update smoke/contract validation when outputs change.
- Update `docs/RESEARCH_PLATFORM_INDEX.md` if a new public component appears.
- Update artifact documentation when a new CSV/JSON/HTML contract is introduced.

## 11. Maintenance Workflow

When adding a feature:

1. Consult `docs/RESEARCH_PLATFORM_INDEX.md`.
2. Identify the canonical owner module.
3. Implement package logic with type hints and short docstrings.
4. Keep notebooks as orchestration/narrative consumers.
5. Add or update smoke tests and artifact validators.
6. Update documentation when public behavior changes.
7. Run the validation suite before commit.

When fixing `ModuleNotFoundError`:

1. Run the notebook setup/bootstrap cell.
2. Verify `PROJECT_ROOT` and `PROJECT_ROOT/src` are on `sys.path`.
3. Confirm the full repository is available, not only the notebook.
4. Re-run `python3 scripts/validate_notebooks.py`.

When changing an artifact contract:

1. Update the producing package.
2. Regenerate the artifact.
3. Update `scripts/validate_artifacts.py`.
4. Update this guide and the relevant domain overview.
5. Run `python3 scripts/validate_artifacts.py`.

## 12. Current Known State

As of 2026-05-20:

- ML Stock Lab package is integrated in `src/ml_stock_lab/`.
- Smart Money Government Data Engine is integrated in `src/smart_money_engine/`.
- Streamlit app has pages for valuation, portfolio, screeners, data platform, run orchestration, Smart Money and ML Lab.
- Canonical notebooks include platform bridges for Smart Money and ML Lab.
- Generated local artifacts are regenerable and ignored by git.
