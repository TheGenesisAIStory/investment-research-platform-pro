#!/usr/bin/env python3
"""Synchronize the Gen.is.IA multi-asset and academic factor universe.

The script is intentionally restart-friendly.  It writes a text log under
``output/sync_logs`` and leaves unavailable symbols/factors as partial rows
rather than crashing the full run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in [PROJECT_ROOT, SRC_ROOT]:
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import pandas as pd

from research_platform_core import (
    compile_smart_money_source_manifest,
    download_ff_factors,
    ingest_multi_asset_universe,
    multi_asset_universe_catalog,
    resolve_data_platform_roots,
)
from research_platform_core.aqr_factors import construct_it_local_factors


def _log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = f"{pd.Timestamp.now('UTC').isoformat()} {message}"
    path.open("a", encoding="utf-8").write(line + "\n")
    print(line)


def _candidate_factor_panels(output_root: Path) -> list[Path]:
    return [
        output_root / "ml_training_lab" / "tables" / "FactorUniversePanel.csv",
        output_root / "ml_stock_lab" / "tables" / "FactorUniversePanel.csv",
        output_root / "factor_universe" / "FactorUniversePanel.csv",
    ]


def _sync_ff_regions(output_root: Path, log_path: Path, dry_run: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for region in ["US", "EU", "JP", "APAC", "EM", "World"]:
        factor_set = "FF5+MOM" if region in {"US", "EU", "JP", "APAC"} else "FF3"
        try:
            if dry_run:
                rows.append({"region": region, "factor_set": factor_set, "status": "DRY_RUN", "rows": 0})
                continue
            frame = download_ff_factors(region, factor_set=factor_set, output_root=output_root)
            rows.append({"region": region, "factor_set": factor_set, "status": "OK" if not frame.empty else "PARTIAL", "rows": len(frame)})
            _log(log_path, f"ff_factors region={region} factor_set={factor_set} rows={len(frame)}")
        except Exception as exc:
            rows.append({"region": region, "factor_set": factor_set, "status": "FAILED", "rows": 0, "error": str(exc)})
            _log(log_path, f"ff_factors region={region} status=FAILED error={exc}")

    it_panel = next((path for path in _candidate_factor_panels(output_root) if path.exists() and path.stat().st_size > 1), None)
    if it_panel and not dry_run:
        try:
            panel = pd.read_csv(it_panel)
            local = construct_it_local_factors(panel, output_root=output_root, write=True)
            rows.append({"region": "IT", "factor_set": "LOCAL_FACTORS", "status": "OK" if not local.empty else "PARTIAL", "rows": len(local), "source_panel": str(it_panel)})
            _log(log_path, f"ff_factors region=IT factor_set=LOCAL_FACTORS rows={len(local)}")
        except Exception as exc:
            rows.append({"region": "IT", "factor_set": "LOCAL_FACTORS", "status": "FAILED", "rows": 0, "error": str(exc)})
            _log(log_path, f"ff_factors region=IT status=FAILED error={exc}")
    else:
        rows.append({
            "region": "IT",
            "factor_set": "LOCAL_FACTORS",
            "status": "DRY_RUN" if dry_run else "PARTIAL",
            "rows": 0,
            "error": "factor panel not found for local Italy construction" if not dry_run else "",
        })
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["full", "incremental"], default="incremental")
    parser.add_argument("--start", default="2000-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--max-assets", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-ff", action="store_true")
    parser.add_argument("--skip-smart-money", action="store_true")
    args = parser.parse_args(argv)

    roots = resolve_data_platform_roots()
    output_root = roots.repo_output
    run_id = f"sync_{pd.Timestamp.now('UTC').strftime('%Y%m%d_%H%M%S')}"
    log_path = output_root / "sync_logs" / f"{run_id}.log"
    _log(log_path, f"sync_universe start mode={args.mode} start={args.start} end={args.end} dry_run={args.dry_run}")

    status: dict[str, Any] = {"run_id": run_id, "mode": args.mode, "started_at": pd.Timestamp.now("UTC").isoformat()}
    if not args.dry_run:
        symbols = None
        if args.max_assets and args.max_assets > 0:
            symbols = multi_asset_universe_catalog().head(int(args.max_assets))["symbol"].astype(str).tolist()
        bundle = ingest_multi_asset_universe(
            symbols,
            start_date=args.start,
            end_date=args.end,
            output_dir=output_root,
            refresh=args.mode == "full",
        )
        manifest = bundle["manifest"]
        status["multi_asset"] = {
            "rows": int(len(manifest)),
            "ok": int(manifest["data_status"].astype(str).str.upper().eq("OK").sum()) if not manifest.empty else 0,
            "partial": int(manifest["data_status"].astype(str).str.upper().eq("PARTIAL").sum()) if not manifest.empty else 0,
        }
        _log(log_path, f"multi_asset rows={status['multi_asset']['rows']} ok={status['multi_asset']['ok']}")
    else:
        status["multi_asset"] = {"rows": 0, "ok": 0, "partial": 0, "dry_run": True}

    if not args.skip_ff:
        ff_rows = _sync_ff_regions(output_root, log_path, dry_run=args.dry_run)
        status["ff_factors"] = ff_rows
        ff_manifest = pd.DataFrame(ff_rows)
        ff_path = output_root / "ff_factors" / "FFRegionalManifest.csv"
        ff_path.parent.mkdir(parents=True, exist_ok=True)
        ff_manifest.to_csv(ff_path, index=False)

    if not args.skip_smart_money:
        smart_manifest = compile_smart_money_source_manifest(roots.financial_db, output_root, fetch_cot=False)
        status["smart_money"] = {"rows": int(len(smart_manifest))}
        _log(log_path, f"smart_money_manifest rows={len(smart_manifest)}")

    status["completed_at"] = pd.Timestamp.now("UTC").isoformat()
    status["log_path"] = str(log_path)
    platform_status = output_root / "PLATFORM_STATUS.json"
    platform_status.write_text(json.dumps(status, indent=2, default=str), encoding="utf-8")
    _log(log_path, f"sync_universe done platform_status={platform_status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
