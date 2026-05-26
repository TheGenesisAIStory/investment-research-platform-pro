#!/usr/bin/env python3
"""Populate the canonical Research Platform SQLite database.

The default run creates:
- a local SQLite DB in ``output/data_cache/research_platform.sqlite``;
- lightweight sample CSVs in ``data/sample/``;
- a JSON summary with row counts and artifact paths.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from research_platform_core.research_database import (  # noqa: E402
    default_research_database_path,
    populate_research_database,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Populate the final Research Platform database with rich local sample data.")
    parser.add_argument("--db-path", default=None, help="SQLite destination path. Defaults to output/data_cache/research_platform.sqlite.")
    parser.add_argument("--sample-dir", default=None, help="Directory for tracked sample CSV extracts. Defaults to data/sample/.")
    parser.add_argument("--start", default="2021-01-04", help="Synthetic history start date.")
    parser.add_argument("--end", default="2026-05-22", help="Synthetic history end date.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed.")
    parser.add_argument("--sample-rows-per-symbol", type=int, default=252, help="Rows per symbol to export in lightweight CSV samples.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    db_path = Path(args.db_path).expanduser() if args.db_path else default_research_database_path(ROOT)
    sample_dir = Path(args.sample_dir).expanduser() if args.sample_dir else ROOT / "data" / "sample"
    summary = populate_research_database(
        db_path=db_path,
        sample_dir=sample_dir,
        start=args.start,
        end=args.end,
        seed=args.seed,
        sample_rows_per_symbol=args.sample_rows_per_symbol,
    )
    print("research_database_OK")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
