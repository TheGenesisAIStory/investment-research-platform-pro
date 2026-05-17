#!/usr/bin/env python3
"""Weekly Analysis Studio batch update.

Runs the eight standard workflows and writes all outputs under DB_BASE.
Use `--offline` for deterministic local/synthetic fallback runs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis import (
    CompetitiveAnalysisEngine,
    EarningsAnalysisEngine,
    MacroAnalysisEngine,
    PortfolioBuilderEngine,
    PortfolioRiskEngine,
    QuantResearchEngine,
    StockScreenerEngine,
    TechnicalAnalysisEngine,
)
from src.analysis.config import DB_BASE, ensure_base_directories
from src.reporting import export_analysis_report


def run_weekly_update(*, offline: bool = False) -> list[dict[str, str]]:
    """Run the standard Analysis Studio workflow set."""
    if offline:
        os.environ["ML_TRADING_OFFLINE"] = "1"
    ensure_base_directories()

    jobs = [
        StockScreenerEngine(universe="core"),
        PortfolioRiskEngine(),
        EarningsAnalysisEngine(ticker="NVDA"),
        PortfolioBuilderEngine(universe="core", risk_profile="moderate"),
        TechnicalAnalysisEngine(ticker="AAPL"),
        CompetitiveAnalysisEngine(sector="semiconductors"),
        QuantResearchEngine(ticker="MSFT"),
        MacroAnalysisEngine(),
    ]

    manifests: list[dict[str, str]] = []
    for engine in jobs:
        engine.run()
        manifests.append(export_analysis_report(engine.last_result))

    batch_dir = DB_BASE / "analysis_outputs" / "weekly_update"
    batch_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = batch_dir / f"weekly_update_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    manifest_path.write_text(json.dumps(manifests, indent=2), encoding="utf-8")
    print(f"Weekly update complete: {manifest_path}")
    return manifests


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Analysis Studio weekly update.")
    parser.add_argument("--offline", action="store_true", help="Skip remote data fetches and use local/synthetic fallbacks.")
    args = parser.parse_args()
    manifests = run_weekly_update(offline=args.offline)
    print(f"Reports generated: {len(manifests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
