#!/usr/bin/env python3
"""Smoke test for Smart Money Government Data Engine."""

from __future__ import annotations

import tempfile
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.smart_money_engine import run_smart_money_engine


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        sample = pd.DataFrame({
            "nameofissuer": ["ACME CORP"],
            "filingmanagername": ["SAMPLE CAPITAL"],
            "cusip": ["000000AA"],
            "ticker": ["ACME"],
            "value": [100],
            "sshprnamt": [1000],
            "periodofreport": ["2026-03-31"],
        })
        sample.to_csv(root / "sec_13f_sample.csv", index=False)
        outputs = run_smart_money_engine(root, root / "out", max_files_per_source=2)
        assert "smart_money_scores" in outputs
        assert len(outputs["smart_money_scores"]) == 1
        assert (root / "out" / "tables" / "SmartMoney_smart_money_scores.csv").exists()
        assert (root / "out" / "reports" / "smart_money_government_data_report.html").exists()
    print("smart_money_smoke_OK")


if __name__ == "__main__":
    main()
