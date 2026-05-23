"""Historical/incremental OHLCV ingestion entrypoint.

Default mode is dry-run. Use --execute for network calls and DB upserts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for candidate in [PROJECT_ROOT, PROJECT_ROOT / "src"]:
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from research_platform_core.data_platform import resolve_data_platform_roots
from research_platform_core.loaders.kaggle_seed_loader import import_kaggle_seeds
from research_platform_core.ohlcv_ingest import OhlcvIngestJob, summarize_ohlcv_manifest


def _split(value: str) -> list[str]:
    return [part.strip() for part in str(value or "").split(",") if part.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--markets", default="us_all,europe_major,japan_major,global_etfs")
    parser.add_argument("--start-date", default="2000-01-01")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--mode", default="incremental", choices=["full", "incremental"])
    parser.add_argument("--interval", default="1d", choices=["1d", "5m"])
    parser.add_argument("--month", default=None, help="Alpha Vantage 5m month slice YYYY-MM.")
    parser.add_argument("--max-assets", type=int, default=50, help="Safety cap. Use 0 for no cap.")
    parser.add_argument("--batch-size", type=int, default=80)
    parser.add_argument("--preferred-providers", default="", help="Optional comma-separated provider allow-list, e.g. yfinance,stooq.")
    parser.add_argument("--include-kaggle-seeds", action="store_true", help="Import configured local Kaggle seed files before API sync.")
    parser.add_argument("--skip-kaggle", action="store_true", help="Skip Kaggle seed import even when include-kaggle-seeds is set.")
    parser.add_argument("--use-kaggle-only", action="store_true", help="Only import Kaggle seeds and do not call market-data APIs.")
    parser.add_argument("--kaggle-root", default=None, help="Optional local Kaggle data lake root. Defaults to Database Finanziario/Kaggle.")
    parser.add_argument("--kaggle-datasets", default="", help="Optional comma-separated Kaggle dataset config names to import.")
    parser.add_argument("--kaggle-limit-files", type=int, default=None, help="Optional safety cap for files per Kaggle dataset.")
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--include-etfs", action="store_true", default=True)
    parser.add_argument("--exclude-etfs", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    roots = resolve_data_platform_roots(repo_output_root=PROJECT_ROOT / "output")
    job = OhlcvIngestJob(roots.financial_db, roots.repo_output, database_url=args.database_url)
    max_assets = None if args.max_assets == 0 else args.max_assets
    include_etfs = bool(args.include_etfs and not args.exclude_etfs)
    dry_run = not args.execute
    if (args.include_kaggle_seeds or args.use_kaggle_only) and not args.skip_kaggle:
        kaggle_manifest = import_kaggle_seeds(
            datasets=_split(args.kaggle_datasets) or None,
            kaggle_root=args.kaggle_root,
            financial_db_root=roots.financial_db,
            output_root=roots.repo_output,
            database_url=args.database_url,
            dry_run=dry_run,
            limit_files=args.kaggle_limit_files,
        )
        print(kaggle_manifest.to_string(index=False) if not kaggle_manifest.empty else "No Kaggle manifest rows")
    if args.use_kaggle_only:
        if dry_run:
            print("Dry run only. Re-run with --execute for Kaggle DB upserts.")
        return 0
    if args.interval == "5m":
        manifest = job.run_intraday_5m(
            markets=_split(args.markets),
            month=args.month,
            max_assets=max_assets or 25,
            include_etfs=include_etfs,
            dry_run=dry_run,
            preferred_providers=_split(args.preferred_providers) or None,
        )
    else:
        manifest = job.run_daily(
            markets=_split(args.markets),
            start_date=args.start_date,
            end_date=args.end_date,
            mode=args.mode,
            max_assets=max_assets,
            batch_size=args.batch_size,
            include_etfs=include_etfs,
            dry_run=dry_run,
            preferred_providers=_split(args.preferred_providers) or None,
        )
    print(summarize_ohlcv_manifest(manifest).to_string(index=False) if not manifest.empty else "No manifest rows")
    if dry_run:
        print("Dry run only. Re-run with --execute for downloads and DB upserts.")
    else:
        print("OHLCV sync complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
