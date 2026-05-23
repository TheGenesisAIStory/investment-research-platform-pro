"""Bootstrap the market-data center with Kaggle seeds plus API deltas."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for candidate in [PROJECT_ROOT, PROJECT_ROOT / "src"]:
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from research_platform_core.data_platform import resolve_data_platform_roots, utc_now
from research_platform_core.loaders.kaggle_seed_loader import import_kaggle_seeds
from research_platform_core.ohlcv_ingest import OhlcvIngestJob, summarize_ohlcv_manifest
from research_platform_core.ohlcv_store import OhlcvDatabase


def _split(value: str) -> list[str]:
    return [part.strip() for part in str(value or "").split(",") if part.strip()]


def _bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _source_counts(database: OhlcvDatabase) -> pd.DataFrame:
    database.init_schema()
    queries = {
        "daily": "SELECT source, COUNT(*) AS rows FROM price_ohlcv_daily GROUP BY source",
        "intraday_5m": "SELECT source, COUNT(*) AS rows FROM price_ohlcv_intraday_5m GROUP BY source",
    }
    frames = []
    if database.is_sqlite:
        with database.connect() as conn:
            for table, query in queries.items():
                frame = pd.read_sql_query(query, conn)
                frame["table"] = table
                frames.append(frame)
    else:
        with database.connect() as conn:
            for table, query in queries.items():
                frame = pd.read_sql_query(query, conn)
                frame["table"] = table
                frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame(columns=["source", "rows", "table"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--markets", default="us_all,europe_major,japan_major,global_etfs")
    parser.add_argument("--start-date", default="2000-01-01")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--max-assets", type=int, default=50, help="Safety cap for API phase. Use 0 for no cap.")
    parser.add_argument("--batch-size", type=int, default=80)
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--include-kaggle-seeds", default="true")
    parser.add_argument("--kaggle-root", default=None)
    parser.add_argument("--kaggle-datasets", default="")
    parser.add_argument("--kaggle-limit-files", type=int, default=None)
    parser.add_argument("--skip-api", action="store_true")
    parser.add_argument("--api-mode", default="auto", choices=["auto", "full", "incremental"])
    parser.add_argument("--preferred-providers", default="")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    started = time.time()
    roots = resolve_data_platform_roots(repo_output_root=PROJECT_ROOT / "output")
    dry_run = not args.execute
    include_kaggle = _bool(args.include_kaggle_seeds)
    max_assets = None if args.max_assets == 0 else args.max_assets

    kaggle_manifest = pd.DataFrame()
    if include_kaggle:
        kaggle_manifest = import_kaggle_seeds(
            datasets=_split(args.kaggle_datasets) or None,
            kaggle_root=args.kaggle_root,
            financial_db_root=roots.financial_db,
            output_root=roots.repo_output,
            database_url=args.database_url,
            dry_run=dry_run,
            limit_files=args.kaggle_limit_files,
        )
        print("Kaggle seed manifest")
        print(kaggle_manifest.to_string(index=False) if not kaggle_manifest.empty else "No Kaggle rows")

    api_manifest = pd.DataFrame()
    if not args.skip_api:
        mode = "incremental" if args.api_mode == "auto" and include_kaggle else ("full" if args.api_mode == "auto" else args.api_mode)
        job = OhlcvIngestJob(roots.financial_db, roots.repo_output, database_url=args.database_url)
        api_manifest = job.run_daily(
            markets=_split(args.markets),
            start_date=args.start_date,
            end_date=args.end_date,
            mode=mode,
            max_assets=max_assets,
            batch_size=args.batch_size,
            dry_run=dry_run,
            preferred_providers=_split(args.preferred_providers) or None,
        )
        print("API OHLCV summary")
        print(summarize_ohlcv_manifest(api_manifest).to_string(index=False) if not api_manifest.empty else "No API rows")

    source_counts = pd.DataFrame()
    if not dry_run:
        source_counts = _source_counts(OhlcvDatabase(database_url=args.database_url, financial_db_root=roots.financial_db))
        target = roots.repo_output / "tables" / "MarketData_source_counts.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        source_counts.to_csv(target, index=False)
        print("Source counts")
        print(source_counts.to_string(index=False) if not source_counts.empty else "No source counts")

    summary = pd.DataFrame(
        [
            {
                "status": "dry_run" if dry_run else "completed",
                "include_kaggle": include_kaggle,
                "skip_api": bool(args.skip_api),
                "kaggle_rows": int(kaggle_manifest.get("num_rows", pd.Series(dtype=int)).sum()) if not kaggle_manifest.empty else 0,
                "api_rows": int(api_manifest.get("rows", pd.Series(dtype=int)).sum()) if not api_manifest.empty else 0,
                "duration_seconds": round(time.time() - started, 2),
                "updated_at": utc_now(),
            }
        ]
    )
    target = roots.repo_output / "tables" / "MarketData_bootstrap_summary.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(target, index=False)
    print("Bootstrap summary")
    print(summary.to_string(index=False))
    if dry_run:
        print("Dry run only. Re-run with --execute for Kaggle/API DB upserts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
