"""Initial Data Center population entrypoint.

Use --execute to perform network downloads. Without it, the script writes a
target catalog and exits safely.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for candidate in [PROJECT_ROOT, PROJECT_ROOT / "src"]:
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from research_platform_core.aqr_factors import refresh_aqr_factor_library
from research_platform_core.data_center_catalog import build_target_catalog, summarize_target_catalog
from research_platform_core.data_platform import resolve_data_platform_roots
from research_platform_core.loaders.equity_universe import EquityUniverseManager
from research_platform_core.loaders.fama_french import FamaFrenchLoader
from research_platform_core.loaders.fx_commodities import ForexCommoditiesManager
from research_platform_core.loaders.risk_factors import RiskFactorsLoader


def _split(value: str) -> list[str]:
    return [part.strip().lower() for part in str(value or "").split(",") if part.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--equities", default="sp500,eurostoxx50,ftsemib")
    parser.add_argument("--factors", default="ff,aqr,risk")
    parser.add_argument("--fx", default="majors")
    parser.add_argument("--commodities", default="all")
    parser.add_argument("--start-date", default="2000-01-01")
    parser.add_argument("--max-items", type=int, default=5, help="Smoke cap per family. Use 0 for all.")
    parser.add_argument("--execute", action="store_true", help="Actually download data. Default is catalog-only dry run.")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    roots = resolve_data_platform_roots(repo_output_root=PROJECT_ROOT / "output")
    catalog = build_target_catalog(roots.financial_db)
    catalog_dir = roots.financial_db / "catalog"
    catalog_dir.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(catalog_dir / "data_center_target_catalog.csv", index=False)
    summarize_target_catalog(catalog).to_csv(catalog_dir / "data_center_target_summary.csv", index=False)
    print(f"Target catalog written: {catalog_dir / 'data_center_target_catalog.csv'}")

    if not args.execute:
        print("Dry run only. Re-run with --execute to download/populate data.")
        return 0

    max_items = None if args.max_items == 0 else args.max_items
    if "ff" in _split(args.factors):
        print("Refreshing Fama-French factors...")
        print(FamaFrenchLoader(roots.financial_db, roots.repo_output).download_all(refresh=args.refresh, max_datasets=max_items).to_string(index=False))
    if "aqr" in _split(args.factors):
        print("Refreshing AQR factors...")
        print(refresh_aqr_factor_library(roots.financial_db, roots.repo_output, refresh=args.refresh, max_datasets=max_items))
    if "risk" in _split(args.factors):
        print("Refreshing risk factors...")
        risk = RiskFactorsLoader(roots.financial_db, args.start_date)
        print(risk.sync_fred(refresh=args.refresh).to_string(index=False))
        print(risk.sync_volatility(refresh=args.refresh).to_string(index=False))
        risk.build_derived_risk_factors()

    if args.fx:
        print("Refreshing FX...")
        print(ForexCommoditiesManager(roots.financial_db, args.start_date).sync_fx(refresh=args.refresh, max_symbols=max_items).to_string(index=False))
    if args.commodities:
        print("Refreshing commodities...")
        print(ForexCommoditiesManager(roots.financial_db, args.start_date).sync_commodities(refresh=args.refresh, max_symbols=max_items).to_string(index=False))

    equities = _split(args.equities)
    if equities:
        manager = EquityUniverseManager(roots.financial_db, args.start_date)
        for universe in equities:
            print(manager.write_constituents(universe, refresh=args.refresh))
            print(manager.sync_prices(universe, max_symbols=max_items, refresh=args.refresh).to_string(index=False))
    print("Initial Data Center setup complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
