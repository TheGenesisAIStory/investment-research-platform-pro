#!/usr/bin/env python3
"""Validate Research Platform data-completion and ML-training artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

from research_platform_core.data_completion import validate_completion_coverage


def _exists_nonempty(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 1


def _validate_ml(output_root: Path, strict: bool) -> pd.DataFrame:
    lab = output_root / "ml_training_lab"
    rows = []
    artifacts = {
        "ml_training_metrics": lab / "tables" / "MLTraining_metrics.csv",
        "ml_training_predictions": lab / "tables" / "MLTraining_predictions.csv",
        "ml_training_manifest": lab / "MLTraining_manifest.json",
        "ml_stock_lab_model_comparison": output_root / "ml_stock_lab" / "tables" / "MLStockLab_model_comparison.csv",
    }
    for name, path in artifacts.items():
        exists = _exists_nonempty(path)
        status = "OK" if exists else "MISSING"
        passes = bool(exists or not strict)
        if strict and name == "ml_training_metrics" and exists:
            try:
                metrics = pd.read_csv(path)
                passes = bool(not metrics.empty and metrics.get("status", pd.Series(dtype=object)).astype(str).eq("OK").any())
                status = "OK" if passes else "NO_OK_MODEL"
            except Exception as exc:
                passes = False
                status = f"INVALID:{type(exc).__name__}"
        rows.append({"domain": name, "status": status, "passes": passes, "manifest_path": str(path)})
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--financial-db-root", default=None)
    parser.add_argument("--output-root", default=str(ROOT / "output"))
    parser.add_argument("--start-year", type=int, default=2000)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--strict", action="store_true", help="Require completed data and ML artifacts.")
    args = parser.parse_args()

    output_root = Path(args.output_root).expanduser()
    data_report = validate_completion_coverage(args.financial_db_root, output_root, args.start_year, args.end_year, strict=args.strict)
    ml_report = _validate_ml(output_root, strict=args.strict)
    report = pd.concat([data_report, ml_report], ignore_index=True, sort=False)
    target = output_root / "data_completion" / "ResearchDataCoverageValidation.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(target, index=False)
    print(report.to_string(index=False))
    print(f"coverage_report={target}")
    if args.strict and not report["passes"].astype(bool).all():
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
