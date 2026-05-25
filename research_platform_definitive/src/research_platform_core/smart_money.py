"""Smart Money source catalog and lightweight coverage manifests.

Issuer-level Smart Money scoring remains in `smart_money_engine`.  This module
adds a source/coverage layer for cross-asset positioning and flow datasets such
as CFTC COT, ETF flows and options positioning.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .data_platform import resolve_data_platform_roots, utc_now


SMART_MONEY_TABLE_ROOT = Path("smart_money") / "tables"


@dataclass(frozen=True)
class SmartMoneySource:
    source_id: str
    name: str
    domain: str
    asset_classes: str
    region: str
    provider: str
    source_url: str
    status: str
    refresh_frequency: str
    notes: str


SMART_MONEY_SOURCES: tuple[SmartMoneySource, ...] = (
    SmartMoneySource(
        "cftc_cot_financial_futures",
        "CFTC COT financial futures positioning",
        "COT",
        "rates, fx, equity_index, credit",
        "usa_global",
        "CFTC public reporting",
        "https://publicreporting.cftc.gov/resource/udgc-27he.csv",
        "READY_OPTIONAL",
        "weekly",
        "Official CFTC public reporting endpoint; used for aggregate positioning where fields are available.",
    ),
    SmartMoneySource(
        "cftc_cot_legacy_futures",
        "CFTC legacy futures-only COT",
        "COT",
        "commodities, rates, fx, equity_index",
        "usa_global",
        "CFTC historical reports",
        "https://publicreporting.cftc.gov/resource/6dca-aqww.csv",
        "READY_OPTIONAL",
        "weekly",
        "Official CFTC historical/reporting page. Ingestion can use downloaded CSV/ZIP exports when API fields differ.",
    ),
    SmartMoneySource(
        "etf_flows_public_proxy",
        "ETF flows public/proxy layer",
        "ETF_FLOWS",
        "equity_etf, fixed_income_etf, commodity_etf, crypto_etf",
        "global",
        "provider_or_user_supplied",
        "",
        "PLANNED",
        "weekly",
        "Placeholder for licensed or manually supplied ETF flow files; schema and UI support are present.",
    ),
    SmartMoneySource(
        "options_positioning_proxy",
        "Options positioning and skew proxy",
        "OPTIONS",
        "equity_index, single_name, etf",
        "usa",
        "provider_or_user_supplied",
        "",
        "PLANNED",
        "daily_or_weekly",
        "Placeholder for put/call, skew and dealer positioning providers.",
    ),
    SmartMoneySource(
        "issuer_insider_fund_flows",
        "Issuer insider / fund flow evidence",
        "ISSUER_EVENTS",
        "single_name_equity",
        "usa_eu",
        "smart_money_engine",
        "",
        "PARTIAL",
        "event_driven",
        "Covered by existing smart_money_engine artifacts when source files are available locally.",
    ),
)


def _read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 1:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, **kwargs)
    except Exception:
        return pd.DataFrame()


def smart_money_source_catalog() -> pd.DataFrame:
    return pd.DataFrame([asdict(source) for source in SMART_MONEY_SOURCES])


def refresh_cftc_cot_snapshot(
    output_root: str | Path | None = None,
    *,
    source_url: str | None = None,
    max_rows: int = 5000,
    fetch: bool = False,
) -> pd.DataFrame:
    """Fetch or load a lightweight CFTC COT snapshot.

    Network fetching is opt-in so tests and local UI startup never block.  If a
    previously saved snapshot exists it is returned when `fetch=False`.
    """
    roots = resolve_data_platform_roots(repo_output_root=output_root)
    table_root = roots.repo_output / SMART_MONEY_TABLE_ROOT
    table_root.mkdir(parents=True, exist_ok=True)
    path = table_root / "SmartMoney_COT_snapshot.csv"
    if not fetch:
        return _read_csv(path)
    url = source_url or SMART_MONEY_SOURCES[0].source_url
    try:
        frame = pd.read_csv(url, nrows=int(max_rows))
    except Exception as exc:
        failure = pd.DataFrame(
            [
                {
                    "source_id": "cftc_cot_financial_futures",
                    "status": "FAILED",
                    "error": f"{type(exc).__name__}: {exc}",
                    "updated_at": utc_now(),
                }
            ]
        )
        failure.to_csv(table_root / "SmartMoney_COT_fetch_status.csv", index=False)
        return pd.DataFrame()
    frame["source_id"] = "cftc_cot_financial_futures"
    frame["updated_at"] = utc_now()
    frame.to_csv(path, index=False)
    return frame


def compile_smart_money_source_manifest(
    financial_db_root: str | Path | None = None,
    output_root: str | Path | None = None,
    *,
    fetch_cot: bool = False,
) -> pd.DataFrame:
    """Write source catalog + source manifest for Smart Money coverage."""
    roots = resolve_data_platform_roots(financial_db_root=financial_db_root, repo_output_root=output_root)
    table_root = roots.repo_output / SMART_MONEY_TABLE_ROOT
    table_root.mkdir(parents=True, exist_ok=True)
    catalog = smart_money_source_catalog()
    cot = refresh_cftc_cot_snapshot(roots.repo_output, fetch=fetch_cot)
    existing_coverage = _read_csv(table_root / "SmartMoney_coverage.csv")
    rows: list[dict[str, Any]] = []
    for source in SMART_MONEY_SOURCES:
        rows_count = 0
        last_date = ""
        status = source.status
        if source.source_id == "cftc_cot_financial_futures" and not cot.empty:
            rows_count = len(cot)
            status = "OK"
            date_cols = [col for col in cot.columns if "date" in col.lower()]
            if date_cols:
                dates = pd.to_datetime(cot[date_cols[0]], errors="coerce")
                if dates.notna().any():
                    last_date = dates.max().date().isoformat()
        elif source.source_id == "issuer_insider_fund_flows" and not existing_coverage.empty:
            rows_count = int(existing_coverage.get("rows", pd.Series([0])).fillna(0).sum()) if "rows" in existing_coverage.columns else len(existing_coverage)
            status = "PARTIAL" if rows_count else "PLANNED"
        rows.append(
            {
                **asdict(source),
                "data_status": status,
                "rows": rows_count,
                "last_date": last_date,
                "manifest_updated_at": utc_now(),
            }
        )
    manifest = pd.DataFrame(rows)
    catalog.to_csv(table_root / "SmartMoneySourceCatalog.csv", index=False)
    manifest.to_csv(table_root / "SmartMoneySourceManifest.csv", index=False)
    return manifest


def load_smart_money_source_manifest(
    financial_db_root: str | Path | None = None,
    output_root: str | Path | None = None,
    *,
    refresh_if_missing: bool = True,
) -> pd.DataFrame:
    roots = resolve_data_platform_roots(financial_db_root=financial_db_root, repo_output_root=output_root)
    path = roots.repo_output / SMART_MONEY_TABLE_ROOT / "SmartMoneySourceManifest.csv"
    frame = _read_csv(path)
    if frame.empty and refresh_if_missing:
        frame = compile_smart_money_source_manifest(roots.financial_db, roots.repo_output)
    return frame


def summarize_smart_money_sources(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["domain", "source_count", "ok_count", "ready_optional_count", "planned_count", "partial_count", "rows"])
    view = frame.copy()
    view["data_status"] = view.get("data_status", "PLANNED").fillna("PLANNED").astype(str).str.upper()
    grouped = view.groupby("domain", dropna=False)
    rows: list[dict[str, Any]] = []
    for domain, group in grouped:
        rows_series = (
            pd.to_numeric(group["rows"], errors="coerce")
            if "rows" in group.columns
            else pd.Series(0, index=group.index)
        )
        rows.append(
            {
                "domain": domain,
                "source_count": int(len(group)),
                "ok_count": int(group["data_status"].eq("OK").sum()),
                "ready_optional_count": int(group["data_status"].str.contains("READY", na=False).sum()),
                "planned_count": int(group["data_status"].str.contains("PLANNED", na=False).sum()),
                "partial_count": int(group["data_status"].str.contains("PARTIAL", na=False).sum()),
                "rows": int(rows_series.fillna(0).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("domain").reset_index(drop=True)
