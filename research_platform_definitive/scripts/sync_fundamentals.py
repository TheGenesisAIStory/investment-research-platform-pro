"""Weekly fundamentals sync for selected equity universes."""

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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universes", default="sp500,ftsemib")
    parser.add_argument("--max-symbols", type=int, default=10)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    roots = resolve_data_platform_roots(repo_output_root=PROJECT_ROOT / "output")
    if not args.execute:
        print("Dry run. Re-run with --execute to download fundamentals.")
        return 0
    manager = EquityUniverseManager(roots.financial_db)
    for universe in [u.strip() for u in args.universes.split(",") if u.strip()]:
        print(manager.write_constituents(universe))
        print(manager.sync_fundamentals(universe, max_symbols=args.max_symbols, refresh=args.refresh).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
