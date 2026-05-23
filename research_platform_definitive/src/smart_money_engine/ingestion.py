"""Local official-source ingestion adapters.

The ingestion layer is intentionally cache-first and conservative. It scans the
Financial Database for official-source exports and normalizes what is already
available. Network download helpers can be added later without changing the
analytics contract.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .io import find_candidate_files, read_table
from .normalization import (
    standardize_13dg,
    standardize_13f,
    standardize_cot,
    standardize_form4,
    standardize_spending,
    standardize_tic,
)
from .schemas import (
    BENEFICIAL_EVENT_COLUMNS,
    CAPITAL_FLOW_COLUMNS,
    GOV_SPENDING_COLUMNS,
    HOLDING_COLUMNS,
    INSIDER_COLUMNS,
    MACRO_POSITIONING_COLUMNS,
    empty_frame,
    source_registry_frame,
)


def _load_many(root: Path, tokens: list[str], normalizer, columns: list[str], max_files: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    files = find_candidate_files(root, tokens)[:max_files]
    frames: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, object]] = []
    for path in files:
        raw = read_table(path)
        normalized = normalizer(raw, str(path))
        frames.append(normalized)
        manifest_rows.append({
            "dataset": tokens[0],
            "path": str(path),
            "raw_rows": len(raw),
            "normalized_rows": len(normalized),
            "status": "OK" if not normalized.empty else "EMPTY_OR_UNREADABLE",
            "modified": pd.Timestamp(path.stat().st_mtime, unit="s").isoformat(),
        })
    if frames:
        return pd.concat(frames, ignore_index=True), pd.DataFrame(manifest_rows)
    return empty_frame(columns), pd.DataFrame([{"dataset": tokens[0], "path": "", "raw_rows": 0, "normalized_rows": 0, "status": "MISSING", "modified": ""}])


def ingest_official_sources(financial_db_root: Path, max_files_per_source: int = 5) -> dict[str, pd.DataFrame]:
    """Ingest locally available official-source files from Database Finanziario."""
    holdings, man_13f = _load_many(financial_db_root, ["13f", "form13f", "sec_13f"], standardize_13f, HOLDING_COLUMNS, max_files_per_source)
    insiders, man_form4 = _load_many(financial_db_root, ["form4", "insider", "ownership"], standardize_form4, INSIDER_COLUMNS, max_files_per_source)
    beneficial, man_13dg = _load_many(financial_db_root, ["13d", "13g", "sc13"], standardize_13dg, BENEFICIAL_EVENT_COLUMNS, max_files_per_source)
    cot, man_cot = _load_many(financial_db_root, ["cot", "commitments"], standardize_cot, MACRO_POSITIONING_COLUMNS, max_files_per_source)
    tic, man_tic = _load_many(financial_db_root, ["tic", "treasury_international_capital"], standardize_tic, CAPITAL_FLOW_COLUMNS, max_files_per_source)
    spending, man_spending = _load_many(financial_db_root, ["usaspending", "usa_spending", "awards", "procurement"], standardize_spending, GOV_SPENDING_COLUMNS, max_files_per_source)

    manifest = pd.concat([man_13f, man_form4, man_13dg, man_cot, man_tic, man_spending], ignore_index=True)
    coverage = manifest.groupby("dataset", dropna=False).agg(
        files=("path", lambda x: int((x.astype(str) != "").sum())),
        rows=("normalized_rows", "sum"),
        status=("status", lambda x: "OK" if (x == "OK").any() else "MISSING"),
    ).reset_index()

    return {
        "source_registry": source_registry_frame(),
        "holdings": holdings,
        "insiders": insiders,
        "beneficial_events": beneficial,
        "macro_positioning": cot,
        "capital_flows": tic,
        "government_spending": spending,
        "ingestion_manifest": manifest,
        "coverage": coverage,
    }
