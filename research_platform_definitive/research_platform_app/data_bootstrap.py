"""Data readiness and light bootstrap helpers for Streamlit pages.

The bootstrap is deliberately conservative: it creates the expected folder
structure, seeds available local metadata into the selected Financial DB root,
and writes an audit manifest. Heavy refreshes remain delegated to registered
jobs/scripts so the UI does not surprise the user with long provider calls.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


REQUIRED_DATASETS = {
    "equity_metadata": ["data/us_equities_meta_data.csv", "us_equities_meta_data.csv"],
    "company_screener": ["company_valuation/output/tables/ScreenerResults.csv"],
    "ml_signals": ["output/ml_stock_lab/tables/MLStockLab_signals.csv"],
    "smart_money_scores": ["output/smart_money/tables/SmartMoney_smart_money_scores.csv"],
    "banking_universe": ["output/banks_pipeline/banks_universe.csv"],
}


def _exists_nonempty(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 1


def data_readiness(roots: dict[str, Path]) -> pd.DataFrame:
    financial_db = Path(roots.get("financial_db", PROJECT_ROOT / "archive" / "local_databases_not_on_drive")).expanduser()
    rows = []
    for dataset, candidates in REQUIRED_DATASETS.items():
        paths = []
        for rel in candidates:
            base = financial_db if rel.startswith("data/") or rel == "us_equities_meta_data.csv" else PROJECT_ROOT
            path = base / rel
            paths.append(path)
        found = next((path for path in paths if _exists_nonempty(path)), None)
        rows.append(
            {
                "dataset": dataset,
                "status": "available" if found else "missing",
                "path": str(found or paths[0]),
                "required_for": {
                    "equity_metadata": "Screener universe and issuer identity",
                    "company_screener": "Valuation and core screening",
                    "ml_signals": "ML Stock Lab reasoning and quintiles",
                    "smart_money_scores": "Official-source confirmation layer",
                    "banking_universe": "Banking Data Lab universe",
                }.get(dataset, "platform"),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_financial_database(roots: dict[str, Path]) -> dict[str, Any]:
    financial_db = Path(roots.get("financial_db", PROJECT_ROOT / "archive" / "local_databases_not_on_drive")).expanduser()
    workspace = Path(roots.get("workspace", PROJECT_ROOT / "output")).expanduser()
    manifest_dir = workspace / "bootstrap"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    created_dirs = []
    for rel in ["data", "data/prices", "data/fundamentals", "data/mappings", "data/catalog", "logs"]:
        target = financial_db / rel
        target.mkdir(parents=True, exist_ok=True)
        created_dirs.append(str(target))

    copied = []
    local_metadata = PROJECT_ROOT / "archive" / "local_databases_not_on_drive" / "data" / "us_equities_meta_data.csv"
    target_metadata = financial_db / "data" / "us_equities_meta_data.csv"
    if local_metadata.exists() and not target_metadata.exists():
        shutil.copy2(local_metadata, target_metadata)
        copied.append({"source": str(local_metadata), "target": str(target_metadata)})

    manifest = {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "financial_db": str(financial_db),
        "created_dirs": created_dirs,
        "copied_seed_files": copied,
        "next_cli_steps": [
            "python scripts/bootstrap_research_data_2000_2026.py --max-assets 25 --max-symbols 10",
            "python scripts/train_ml_models_2000_2026.py --models ols,rf --max-rows 5000",
            "python scripts/initial_setup.py --execute --max-items 5",
            "python scripts/sync_prices.py --execute --universes sp500,ftsemib --max-symbols 25",
            "python scripts/sync_fundamentals.py --execute --universes sp500,ftsemib --max-symbols 10",
            "python scripts/sync_official_macro.py --execute --build-features --start 2010-01",
            "python scripts/validate_data.py",
        ],
    }
    manifest_path = manifest_dir / "data_bootstrap_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def render_bootstrap_banner(roots: dict[str, Path], required: list[str] | None = None) -> None:
    import streamlit as st

    readiness = data_readiness(roots)
    if required:
        readiness = readiness[readiness["dataset"].isin(required)]
    missing = readiness[readiness["status"].eq("missing")]
    if missing.empty:
        return
    with st.container(border=True):
        st.warning("Data bootstrap required: one or more datasets needed by this page are missing or empty.")
        st.dataframe(readiness, width="stretch", hide_index=True)
        col1, col2 = st.columns([1, 2])
        if col1.button("Run light bootstrap", width="stretch"):
            try:
                result = bootstrap_financial_database(roots)
                st.success(f"Bootstrap manifest written: {result['manifest_path']}")
                st.json(result)
            except Exception as exc:
                st.error(f"Bootstrap failed cleanly: {type(exc).__name__}. Check paths and permissions.")
        col2.code(
            "python scripts/initial_setup.py --execute --max-items 5\n"
            "python scripts/sync_prices.py --execute --universes sp500,ftsemib --max-symbols 25\n"
            "python scripts/sync_fundamentals.py --execute --universes sp500,ftsemib --max-symbols 10",
            language="bash",
        )
