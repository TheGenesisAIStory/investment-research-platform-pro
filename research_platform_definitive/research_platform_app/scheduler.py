"""CLI scheduler for keeping research artifacts fresh."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from orchestration.scheduler import scheduler_loop, scheduler_tick
from support import default_roots


def main() -> int:
    parser = argparse.ArgumentParser(description="Run freshness-driven research-platform scheduler.")
    parser.add_argument("--once", action="store_true", help="Run one scheduler tick and exit.")
    parser.add_argument("--interval-seconds", type=int, default=900)
    parser.add_argument("--jobs", nargs="*", default=["data_platform_status_refresh", "screener_refresh"], help="Job IDs to keep fresh.")
    parser.add_argument("--company-root", type=Path, default=None)
    parser.add_argument("--portfolio-root", type=Path, default=None)
    parser.add_argument("--workspace-root", type=Path, default=None)
    parser.add_argument("--financial-db-root", type=Path, default=None)
    parser.add_argument("--notebook-offhour-start", type=int, default=20)
    parser.add_argument("--notebook-offhour-end", type=int, default=7)
    args = parser.parse_args()

    roots = default_roots()
    if args.company_root is not None:
        roots["company"] = args.company_root.expanduser()
    if args.portfolio_root is not None:
        roots["portfolio"] = args.portfolio_root.expanduser()
    if args.workspace_root is not None:
        roots["workspace"] = args.workspace_root.expanduser()
    if args.financial_db_root is not None:
        roots["financial_db"] = args.financial_db_root.expanduser()

    if args.once:
        print(json.dumps(scheduler_tick(
            roots,
            args.jobs,
            notebook_offhour_start=args.notebook_offhour_start,
            notebook_offhour_end=args.notebook_offhour_end,
        ), indent=2))
    else:
        scheduler_loop(
            roots,
            args.interval_seconds,
            args.jobs,
            notebook_offhour_start=args.notebook_offhour_start,
            notebook_offhour_end=args.notebook_offhour_end,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
