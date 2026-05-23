"""Incremental price sync for selected Data Center universes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for candidate in [PROJECT_ROOT, PROJECT_ROOT / "src"]:
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from research_platform_core.data_platform import resolve_data_platform_roots
from research_platform_core.loaders.equity_universe import EquityUniverseManager
from research_platform_core.loaders.fx_commodities import ForexCommoditiesManager


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universes", default="sp500,ftsemib,eurostoxx50")
    parser.add_argument("--start-date", default="2000-01-01")
    parser.add_argument("--max-symbols", type=int, default=25)
    parser.add_argument("--mode", default="incremental", choices=["incremental", "force"])
    parser.add_argument("--include-fx", action="store_true")
    parser.add_argument("--include-commodities", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    roots = resolve_data_platform_roots(repo_output_root=PROJECT_ROOT / "output")
    if not args.execute:
        print("Dry run. Re-run with --execute to download prices.")
        return 0
    refresh = args.mode == "force"
    equities = EquityUniverseManager(roots.financial_db, args.start_date)
    for universe in [u.strip() for u in args.universes.split(",") if u.strip()]:
        print(equities.write_constituents(universe))
        print(equities.sync_prices(universe, max_symbols=args.max_symbols, refresh=refresh).to_string(index=False))
    fx = ForexCommoditiesManager(roots.financial_db, args.start_date)
    if args.include_fx:
        print(fx.sync_fx(refresh=refresh).to_string(index=False))
    if args.include_commodities:
        print(fx.sync_commodities(refresh=refresh).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
