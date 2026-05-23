"""End-to-end Smart Money Government Data Engine pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .analytics import event_feed, sector_theme_monitor, smart_money_screener
from .ingestion import ingest_official_sources
from .io import default_output_root, discover_financial_db, ensure_dir, safe_write_csv, safe_write_json
from .reporting import build_html_report
from .resolution import build_entity_master, match_awards_to_issuers
from .schemas import (
    analyst_quick_wins,
    dataset_priority_ranking,
    government_dataset_matrix,
    mvp_30_day_plan,
    open_source_pattern_matrix,
)


def run_smart_money_engine(
    financial_db_root: str | Path | None = None,
    output_root: str | Path | None = None,
    max_files_per_source: int = 5,
) -> dict[str, pd.DataFrame | dict[str, Any]]:
    """Run a cache-first, official-source-first smart-money analytics pass.

    No network calls are made here. The function consumes locally available
    official files from Database Finanziario and always emits stable artifacts.
    """
    financial_db = Path(financial_db_root) if financial_db_root else discover_financial_db()
    out_root = Path(output_root) if output_root else default_output_root()
    tables_dir = ensure_dir(out_root / "tables")
    reports_dir = ensure_dir(out_root / "reports")

    ingested = ingest_official_sources(financial_db, max_files_per_source=max_files_per_source)
    entity_master = build_entity_master(
        ingested["holdings"],
        ingested["insiders"],
        ingested["beneficial_events"],
        ingested["government_spending"],
    )
    spending = match_awards_to_issuers(ingested["government_spending"], entity_master)
    scores = smart_money_screener(
        ingested["holdings"],
        ingested["insiders"],
        ingested["beneficial_events"],
        spending,
        ingested["macro_positioning"],
        ingested["capital_flows"],
    )
    events = event_feed(ingested["insiders"], ingested["beneficial_events"], spending, ingested["holdings"])
    sector_monitor = sector_theme_monitor(scores)

    outputs: dict[str, pd.DataFrame] = {
        **ingested,
        "entity_master": entity_master,
        "government_spending_matched": spending,
        "smart_money_scores": scores,
        "event_feed": events,
        "sector_monitor": sector_monitor,
        "government_dataset_matrix": government_dataset_matrix(),
        "open_source_pattern_matrix": open_source_pattern_matrix(),
        "mvp_30_day_plan": mvp_30_day_plan(),
        "dataset_priority_ranking": dataset_priority_ranking(),
        "analyst_quick_wins": analyst_quick_wins(),
    }

    written: dict[str, str] = {}
    for name, df in outputs.items():
        if isinstance(df, pd.DataFrame):
            written[name] = str(safe_write_csv(df, tables_dir / f"SmartMoney_{name}.csv"))

    report_path = build_html_report(outputs, reports_dir / "smart_money_government_data_report.html")
    manifest = {
        "financial_db_root": str(financial_db),
        "output_root": str(out_root),
        "report_path": str(report_path),
        "tables": written,
        "coverage_rows": len(ingested["coverage"]),
        "score_rows": len(scores),
        "event_rows": len(events),
        "caveats": [
            "13F is quarterly and delayed; it is not trade tape.",
            "Form 4 signals require transaction-code and plan context.",
            "13D/13G events need amendment tracking for stateful ownership monitoring.",
            "COT and TIC are macro proxies, not issuer-level ownership data.",
            "USAspending recipient-to-listed-parent matching requires audited entity resolution.",
            "EU coverage is fragmented and proxy-based unless country connectors are implemented.",
        ],
    }
    safe_write_json(manifest, out_root / "SmartMoneyManifest.json")
    return {**outputs, "manifest": manifest}
