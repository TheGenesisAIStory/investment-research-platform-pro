"""Detached worker for executing one persisted run."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
if str(APP_DIR.parent) not in sys.path:
    sys.path.insert(0, str(APP_DIR.parent))

from orchestration import JobStore, execute_run


def main() -> int:
    parser = argparse.ArgumentParser(description="Execute a persisted research-platform run.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--company-root", type=Path, required=True)
    parser.add_argument("--portfolio-root", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--financial-db-root", type=Path, default=None)
    args = parser.parse_args()
    roots = {
        "company": args.company_root.expanduser(),
        "portfolio": args.portfolio_root.expanduser(),
        "workspace": args.workspace_root.expanduser(),
        "financial_db": (args.financial_db_root or args.workspace_root).expanduser(),
    }
    os.environ["FINANCIAL_DB_ROOT"] = str(roots["financial_db"])
    os.environ["DB_BASE"] = str(roots["financial_db"])
    os.environ["DATA_PATH"] = str(roots["financial_db"])
    run = execute_run(args.run_id, roots=roots, store=JobStore())
    return 0 if run.status.value == "SUCCESS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
