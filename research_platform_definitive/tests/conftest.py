"""Pytest import path setup for the definitive research platform bundle."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent

for candidate in [
    REPO_ROOT,
    PROJECT_ROOT,
    PROJECT_ROOT / "src",
    PROJECT_ROOT / "company_valuation" / "src",
    PROJECT_ROOT / "portfolio_analysis" / "src",
]:
    value = str(candidate)
    if candidate.exists() and value not in sys.path:
        sys.path.insert(0, value)
