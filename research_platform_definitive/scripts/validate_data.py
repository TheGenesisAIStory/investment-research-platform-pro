"""Validate Data Center target coverage and write quality reports."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for candidate in [PROJECT_ROOT, PROJECT_ROOT / "src"]:
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from research_platform_core.data_center_catalog import build_target_catalog, summarize_target_catalog
from research_platform_core.data_platform import build_dataset_inventory, resolve_data_platform_roots


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-files", type=int, default=10000)
    args = parser.parse_args()
    roots = resolve_data_platform_roots(repo_output_root=PROJECT_ROOT / "output")
    catalog = build_target_catalog(roots.financial_db)
    summary = summarize_target_catalog(catalog)
    inventory = build_dataset_inventory(roots.financial_db, max_files=args.max_files)
    out_dir = roots.repo_output / "data_quality"
    out_dir.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(out_dir / "DataCenter_target_catalog.csv", index=False)
    summary.to_csv(out_dir / "DataCenter_target_summary.csv", index=False)
    stale = inventory[pd.to_numeric(inventory.get("age_hours", 0), errors="coerce").fillna(0).gt(24 * 30)] if not inventory.empty else pd.DataFrame()
    stale.to_csv(out_dir / "DataCenter_stale_inventory.csv", index=False)
    print("Data Center quality validation")
    print(summary.to_string(index=False))
    print(f"Reports written: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
