#!/usr/bin/env python3
"""Bootstrap/refresh the Research Platform data estate for 2000-2026.

Default mode is dry-run. Pass `--execute` for provider calls and Drive writes.
Use safety caps (`--max-assets`, `--max-symbols`) for desk-day refreshes; remove
or raise them for full historical backfills.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

from research_platform_core.data_completion import CompletionConfig, ResearchDataBootstrapper


def _csv_values(value: str | None) -> tuple[str, ...]:
    if not value:
        return tuple()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Run provider/download jobs. Without this flag only manifests are planned.")
    parser.add_argument("--full", action="store_true", help="Use full-history price mode instead of incremental mode.")
    parser.add_argument("--refresh", action="store_true", help="Force refresh where loaders support cache invalidation.")
    parser.add_argument("--start-year", type=int, default=2000)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--universes", default="sp500,nasdaq100,eurostoxx50,ftsemib")
    parser.add_argument("--markets", default="us_all,europe_major,japan_major,global_etfs")
    parser.add_argument("--max-symbols", type=int, default=25, help="Cap fundamental symbols per universe. Use 0 for no cap.")
    parser.add_argument("--max-assets", type=int, default=250, help="Cap OHLCV assets. Use 0 for no cap.")
    parser.add_argument("--max-factor-datasets", type=int, default=4, help="Cap Fama-French/AQR datasets. Use 0 for no cap.")
    parser.add_argument("--financial-db-root", default=None)
    parser.add_argument("--output-root", default=str(ROOT / "output"))
    parser.add_argument(
        "--ohlcv-parquet-root",
        default=None,
        help="Local/stable OHLCV parquet root. Defaults to local staging when Financial DB is on Google Drive.",
    )
    parser.add_argument(
        "--stage",
        default="all",
        choices=[
            "all",
            "equity_fundamentals",
            "equity_prices",
            "macro_fx",
            "factor_libraries",
            "smart_money",
            "banking",
            "factor_universe_panel",
            "coverage",
        ],
        help="Run one stage instead of the full pipeline. Use equity_prices to resume after fundamentals.",
    )
    parser.add_argument("--skip-smart-money", action="store_true")
    parser.add_argument("--skip-banking", action="store_true")
    parser.add_argument("--skip-macro", action="store_true")
    parser.add_argument("--skip-factors", action="store_true")
    args = parser.parse_args()
    if args.ohlcv_parquet_root:
        os.environ["RESEARCH_PLATFORM_OHLCV_PARQUET_ROOT"] = str(Path(args.ohlcv_parquet_root).expanduser())

    config = CompletionConfig(
        start_year=args.start_year,
        end_year=args.end_year,
        universes=_csv_values(args.universes),
        markets=_csv_values(args.markets),
        execute=bool(args.execute),
        incremental=not bool(args.full),
        refresh=bool(args.refresh),
        max_symbols=None if args.max_symbols == 0 else args.max_symbols,
        max_assets=None if args.max_assets == 0 else args.max_assets,
        max_factor_datasets=None if args.max_factor_datasets == 0 else args.max_factor_datasets,
        include_smart_money=not args.skip_smart_money,
        include_banking=not args.skip_banking,
        include_macro=not args.skip_macro,
        include_factors=not args.skip_factors,
    )
    runner = ResearchDataBootstrapper(args.financial_db_root, args.output_root, config)
    if args.stage == "all":
        results = runner.run_all()
    else:
        runner.write_plan()
        results = {args.stage: runner.run_stage(args.stage)}
    print("research_data_bootstrap_complete")
    print(f"stage={args.stage}")
    print(f"execute={config.execute}")
    print(f"financial_db_root={runner.financial_db_root}")
    print(f"output_root={runner.output_root}")
    if os.environ.get("RESEARCH_PLATFORM_OHLCV_PARQUET_ROOT"):
        print(f"ohlcv_parquet_root={os.environ['RESEARCH_PLATFORM_OHLCV_PARQUET_ROOT']}")
    for name, frame in results.items():
        print(f"{name}: rows={len(frame)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
