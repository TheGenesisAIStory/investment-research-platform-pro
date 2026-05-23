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
    "research_platform_app/app.py",
    "research_platform_app/pages/9_ML_Stock_Lab.py",
    "src/research_platform_core/__init__.py",
    "src/ml_stock_lab/__init__.py",
    "src/smart_money_engine/__init__.py",
    "docs/CODEX_CANONICAL_MAINTENANCE_GUIDE.md",
    "docs/PROJECT_STATUS_FINAL.md",
    "docs/ACADEMIC_METHODS.md",
    "docs/MEMORY_RECOVERY_PLAN.md",
    "docs/DATA_CENTER_OPERATING_MODEL.md",
    "docs/FORMULE_STRATEGIE_REASONING.md",
    "scripts/validate_notebooks.py",
    "scripts/validate_artifacts.py",
    "scripts/populate_research_database.py",
    "scripts/llm_lab_cli.py",
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
