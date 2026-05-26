#!/usr/bin/env python3
"""Validate the definitive research platform bundle structure."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_PATHS = [
    "README.md",
    "MANIFEST.csv",
    "REVIEW_SUMMARY.md",
    "notebooks/README.md",
    "company_valuation/notebooks/Company_Valuation_Final_Version.ipynb",
    "portfolio_analysis/notebooks/Portfolio-Analysis-Model_RESEARCH_PLATFORM_PRO.ipynb",
    "machine_learning_lab/notebooks/ML_Stock_Lab_Experiments.ipynb",
    "machine_learning_lab/notebooks/ML_Training_Lab_Colab_Ollama.ipynb",
    ".streamlit/config.toml",
    "research_platform_app/app.py",
    "research_platform_app/pages/0_🏠_Home.py",
    "research_platform_app/pages/1_📡_Smart_Money_Macro.py",
    "research_platform_app/pages/2_🔬_Valuation_Research.py",
    "research_platform_app/pages/3_📁_Portfolio_Research.py",
    "research_platform_app/pages/4_🔍_Screener_Builder.py",
    "research_platform_app/pages/5_📤_Export_Center.py",
    "research_platform_app/pages/6_🧪_Notebook_Runner.py",
    "research_platform_app/pages/7_⚙️_Run_Hi_Freq_Engine.py",
    "research_platform_app/pages/8_🗄️_Data_Platform.py",
    "research_platform_app/pages/9_ML_Stock_Lab.py",
    "src/research_platform_core/__init__.py",
    "src/ml_stock_lab/__init__.py",
    "src/smart_money_engine/__init__.py",
    "docs/CODEX_CANONICAL_MAINTENANCE_GUIDE.md",
    "docs/DATA_BOOTSTRAP.md",
    "docs/DATA_COMPLETION_2000_2026.md",
    "docs/SCREENER_BUILDER_WORKSTATION.md",
    "research_platform_app/data_bootstrap.py",
    "research_platform_app/screener_workbench.py",
    "research_platform_app/ui_ops.py",
    "src/research_platform_core/data_completion.py",
    "src/ml_stock_lab/training.py",
    "scripts/bootstrap_research_data_2000_2026.py",
    "scripts/train_ml_models_2000_2026.py",
    "scripts/validate_research_data_coverage.py",
    "scripts/validate_notebooks.py",
    "scripts/validate_artifacts.py",
]


def _valid_notebook(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return "cells" in payload and "nbformat" in payload


def main() -> int:
    failures: list[str] = []
    for rel in REQUIRED_PATHS:
        path = ROOT / rel
        if not path.exists():
            failures.append(f"MISSING {rel}")
            continue
        if rel.endswith(".ipynb") and not _valid_notebook(path):
            failures.append(f"INVALID_NOTEBOOK {rel}")

    if failures:
        for failure in failures:
            print(f"FAIL {failure}")
        return 1

    print("definitive_bundle_OK")
    print(f"root={ROOT}")
    print(f"validated_paths={len(REQUIRED_PATHS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
