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


COT_MARKET_MAP: tuple[tuple[str, str, str], ...] = (
    ("SP500", "equity_index", "S&P"),
    ("NASDAQ", "equity_index", "NASDAQ"),
    ("US_10Y", "rates", "10-YEAR"),
    ("US_2Y", "rates", "2-YEAR"),
    ("EURO_FX", "fx", "EURO FX"),
    ("JAPANESE_YEN", "fx", "JAPANESE YEN"),
    ("BRITISH_POUND", "fx", "BRITISH POUND"),
    ("WTI_CRUDE", "commodities", "CRUDE OIL"),
    ("GOLD", "commodities", "GOLD"),
    ("COPPER", "commodities", "COPPER"),
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


def _cot_instrument(market_name: object) -> tuple[str, str]:
    name = str(market_name or "").upper()
    for instrument, asset_class, keyword in COT_MARKET_MAP:
        if keyword in name:
            return instrument, asset_class
    return "", ""


def normalize_cot_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize CFTC COT financial/legacy rows to a common positioning schema."""
    if raw.empty:
        return pd.DataFrame()
    frame = raw.copy()
    market_col = "market_and_exchange_names" if "market_and_exchange_names" in frame.columns else "contract_market_name"
    date_col = "report_date_as_yyyy_mm_dd" if "report_date_as_yyyy_mm_dd" in frame.columns else ""
    if market_col not in frame.columns or not date_col:
        return pd.DataFrame()
    mapped = frame[market_col].map(_cot_instrument)
    frame["instrument"] = mapped.map(lambda item: item[0])
    frame["asset_class"] = mapped.map(lambda item: item[1])
    frame = frame[frame["instrument"].astype(str).str.len().gt(0)].copy()
    if frame.empty:
        return pd.DataFrame()
    frame["report_date"] = pd.to_datetime(frame[date_col], errors="coerce").dt.date.astype(str)

    if {"lev_money_positions_long", "lev_money_positions_short"}.issubset(frame.columns):
        long_col = "lev_money_positions_long"
        short_col = "lev_money_positions_short"
        trader_group = "leveraged_money"
    elif {"noncomm_positions_long_all", "noncomm_positions_short_all"}.issubset(frame.columns):
        long_col = "noncomm_positions_long_all"
        short_col = "noncomm_positions_short_all"
        trader_group = "noncommercial"
    else:
        return pd.DataFrame()

    frame["noncommercial_long"] = pd.to_numeric(frame[long_col], errors="coerce")
    frame["noncommercial_short"] = pd.to_numeric(frame[short_col], errors="coerce")
    frame["open_interest"] = pd.to_numeric(frame.get("open_interest_all", pd.Series(index=frame.index, dtype=float)), errors="coerce")
    frame["net_noncommercial"] = frame["noncommercial_long"] - frame["noncommercial_short"]
    frame["net_noncommercial_oi_pct"] = frame["net_noncommercial"] / frame["open_interest"].replace(0, float("nan"))
    frame["trader_group"] = trader_group
    out = frame[
        [
            "report_date",
            "instrument",
            "asset_class",
            market_col,
            "trader_group",
            "noncommercial_long",
            "noncommercial_short",
            "net_noncommercial",
            "net_noncommercial_oi_pct",
            "open_interest",
        ]
    ].copy()
    out = out.rename(columns={market_col: "market_name"})
    out["source"] = "CFTC COT"
    out["updated_at"] = utc_now()
    out = out.sort_values(["instrument", "report_date"]).reset_index(drop=True)
    out["weekly_change_net"] = out.groupby("instrument")["net_noncommercial"].diff()
    out["net_position_percentile_3y"] = out.groupby("instrument")["net_noncommercial"].transform(
        lambda series: series.rolling(156, min_periods=20).rank(pct=True).iloc[:, 0] if False else series.rolling(156, min_periods=20).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    )
    return out


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
    return fetch_cot_data(output_root, source_url=source_url, max_rows=max_rows, fetch=True).get("snapshot", pd.DataFrame())


def fetch_cot_data(
    output_root: str | Path | None = None,
    *,
    source_url: str | None = None,
    max_rows: int = 50_000,
    fetch: bool = False,
    raw_frame: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Fetch/load and normalize CFTC COT data for core macro futures.

    Network is opt-in. Tests can pass `raw_frame`; UI can call with
    `fetch=True` when the user explicitly requests a refresh.
    """
    roots = resolve_data_platform_roots(repo_output_root=output_root)
    table_root = roots.repo_output / SMART_MONEY_TABLE_ROOT
    table_root.mkdir(parents=True, exist_ok=True)
    history_path = table_root / "SmartMoney_COT_history.csv"
    snapshot_path = table_root / "SmartMoney_COT_snapshot.csv"
    sample_path = table_root / "SmartMoney_COT_history_sample.csv"
    if raw_frame is not None:
        raw = raw_frame.copy()
    elif fetch:
        url = source_url or SMART_MONEY_SOURCES[0].source_url
        query_url = f"{url}?$limit={int(max_rows)}" if "?" not in url else url
        try:
            raw = pd.read_csv(query_url)
        except Exception as exc:
            failure = pd.DataFrame(
                [{"source_id": "cftc_cot_financial_futures", "status": "FAILED", "error": f"{type(exc).__name__}: {exc}", "updated_at": utc_now()}]
            )
            failure.to_csv(table_root / "SmartMoney_COT_fetch_status.csv", index=False)
            return {"history": _read_csv(history_path), "snapshot": _read_csv(snapshot_path), "sample": _read_csv(sample_path)}
    else:
        return {"history": _read_csv(history_path), "snapshot": _read_csv(snapshot_path), "sample": _read_csv(sample_path)}

    history = normalize_cot_data(raw)
    if history.empty:
        return {"history": pd.DataFrame(), "snapshot": pd.DataFrame(), "sample": pd.DataFrame()}
    history.to_csv(history_path, index=False)
    snapshot = history.sort_values("report_date").groupby("instrument", as_index=False).tail(1).sort_values(["asset_class", "instrument"])
    snapshot.to_csv(snapshot_path, index=False)
    sample = history.groupby("instrument", group_keys=False).tail(260).reset_index(drop=True)
    sample.to_csv(sample_path, index=False)
    return {"history": history, "snapshot": snapshot, "sample": sample}


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
    cot_bundle = fetch_cot_data(roots.repo_output, fetch=fetch_cot)
    cot = cot_bundle.get("snapshot", pd.DataFrame())
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
