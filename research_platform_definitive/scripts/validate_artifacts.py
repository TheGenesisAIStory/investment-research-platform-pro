#!/usr/bin/env python3
"""Validate canonical runtime artifact contracts.

The checks are intentionally contract-focused: they verify the files and minimum
columns consumed by the Streamlit app and future API layers. They do not require
large historical datasets to be present.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@dataclass(frozen=True)
class CsvContract:
    path: str
    required_columns: tuple[str, ...] = ()
    allow_empty: bool = False
    min_columns: int | None = None


ML_CONTRACTS = (
    CsvContract("output/ml_stock_lab/tables/MLStockLab_panel.csv", ("ticker", "date", "market_value", "target_source"), min_columns=39),
    CsvContract(
        "output/ml_stock_lab/tables/MLStockLab_signals.csv",
        ("ticker", "date", "fair_value_hat", "mispricing_rel", "zscore", "rank", "signal", "score", "target_source"),
        min_columns=43,
    ),
    CsvContract(
        "output/ml_stock_lab/tables/MLStockLab_metrics.csv",
        ("status", "model", "rows", "target", "target_source", "r2_os", "sharpe_long_short"),
    ),
    CsvContract("output/ml_stock_lab/tables/MLStockLab_quintile_returns.csv", ("date", "quantile", "return"), allow_empty=True),
)

SMART_MONEY_CONTRACTS = (
    CsvContract("output/smart_money/tables/SmartMoney_source_registry.csv", ("dataset_id", "region", "authority", "frequency", "priority")),
    CsvContract("output/smart_money/tables/SmartMoney_coverage.csv", ("dataset", "rows", "status"), allow_empty=True),
    CsvContract("output/smart_money/tables/SmartMoney_smart_money_scores.csv", allow_empty=True),
)


def _generate_missing() -> None:
    try:
        from src.ml_stock_lab.lab import run_ml_stock_lab_experiment
        from src.smart_money_engine.pipeline import run_smart_money_engine
    except ModuleNotFoundError:
        from ml_stock_lab.lab import run_ml_stock_lab_experiment
        from smart_money_engine.pipeline import run_smart_money_engine

    if not (ROOT / "output/ml_stock_lab/tables/MLStockLab_metrics.csv").exists():
        run_ml_stock_lab_experiment(output_root=ROOT / "output/ml_stock_lab")
    if not (ROOT / "output/smart_money/SmartMoneyManifest.json").exists():
        run_smart_money_engine(output_root=ROOT / "output/smart_money")


def _validate_csv(contract: CsvContract) -> dict[str, object]:
    path = ROOT / contract.path
    if not path.exists():
        return {"path": contract.path, "status": "MISSING", "message": "file not found"}

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        df = pd.DataFrame()
    except Exception as exc:
        return {"path": contract.path, "status": "UNREADABLE", "message": str(exc)}

    if df.empty and not contract.allow_empty:
        return {"path": contract.path, "status": "EMPTY", "message": "required artifact is empty", "shape": df.shape}

    missing = [col for col in contract.required_columns if col not in df.columns]
    if missing:
        return {"path": contract.path, "status": "MISSING_COLUMNS", "message": ", ".join(missing), "shape": df.shape}
    if contract.min_columns is not None and df.shape[1] < contract.min_columns:
        return {
            "path": contract.path,
            "status": "TOO_FEW_COLUMNS",
            "message": f"{df.shape[1]} < {contract.min_columns}",
            "shape": df.shape,
        }
    if "target_source" in contract.required_columns and not df.empty and df["target_source"].isna().any():
        return {"path": contract.path, "status": "TARGET_SOURCE_NA", "message": "target_source contains null values", "shape": df.shape}

    return {"path": contract.path, "status": "OK", "message": "", "shape": df.shape}


def _validate_manifest() -> dict[str, object]:
    path = ROOT / "output/smart_money/SmartMoneyManifest.json"
    if not path.exists():
        return {"path": str(path.relative_to(ROOT)), "status": "MISSING", "message": "file not found"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"path": str(path.relative_to(ROOT)), "status": "INVALID_JSON", "message": str(exc)}
    if "generated_at" not in payload and "tables" not in payload:
        return {"path": str(path.relative_to(ROOT)), "status": "WEAK_MANIFEST", "message": "missing generated_at/tables"}
    return {"path": str(path.relative_to(ROOT)), "status": "OK", "message": ""}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-generate", action="store_true", help="Do not regenerate missing lightweight artifacts.")
    args = parser.parse_args()

    if not args.no_generate:
        _generate_missing()

    rows = [_validate_csv(contract) for contract in (*ML_CONTRACTS, *SMART_MONEY_CONTRACTS)]
    rows.append(_validate_manifest())

    width = max(len(str(row["path"])) for row in rows)
    failed = False
    for row in rows:
        status = str(row["status"])
        marker = "OK" if status == "OK" else "FAIL"
        shape = f" shape={row['shape']}" if "shape" in row else ""
        print(f"{marker:4} {str(row['path']):<{width}}  {status}{shape}  {row.get('message', '')}")
        failed = failed or status not in {"OK"}
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
