"""Finviz-inspired research platform layer for the company valuation notebook.

This module does not clone Finviz. It implements the reproducible platform
ideas that fit the notebook: maps, groups, watchlists, events, macro/risk,
alerts, quick views, provenance, and API-ready exports.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

try:
    from research_platform_core import (
        as_df as _core_as_df,
        first_available as _core_first,
        normalize_ticker as _core_normalize_ticker,
        output_root_from_namespace,
        safe_write_csv,
        safe_write_json,
    )
except ModuleNotFoundError:
    from src.research_platform_core import (
        as_df as _core_as_df,
        first_available as _core_first,
        normalize_ticker as _core_normalize_ticker,
        output_root_from_namespace,
        safe_write_csv,
        safe_write_json,
    )

try:
    from company_valuation_screener import (
        build_screener_base_frame,
        run_integrated_screener,
        resolve_column,
    )
except ModuleNotFoundError:
    from .company_valuation_screener import (
        build_screener_base_frame,
        run_integrated_screener,
        resolve_column,
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _as_df(value: Any) -> pd.DataFrame:
    return _core_as_df(value)


def _first(namespace: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    return _core_first(namespace, *names, default=default)


def _output_root(namespace: Mapping[str, Any]) -> Path:
    return output_root_from_namespace(namespace)


def _normalize_ticker(value: Any) -> str:
    return _core_normalize_ticker(value)


def _numeric(frame: pd.DataFrame, field: str) -> pd.Series:
    col = resolve_column(frame, field)
    if col is None:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[col], errors="coerce")


def _column(frame: pd.DataFrame, field: str) -> pd.Series:
    col = resolve_column(frame, field)
    if col is None:
        return pd.Series(np.nan, index=frame.index)
    return frame[col]


def _safe_write_csv(df: pd.DataFrame, path: Path) -> None:
    safe_write_csv(df, path)


def _safe_write_json(obj: Any, path: Path) -> None:
    safe_write_json(obj, path)


@dataclass
class FinvizLayerOutputs:
    architecture_review: pd.DataFrame
    data_dictionary: pd.DataFrame
    securities_master: pd.DataFrame
    universe_constituents: pd.DataFrame
    latest_cross_section: pd.DataFrame
    group_summaries: pd.DataFrame
    map_payload: pd.DataFrame
    watchlists: pd.DataFrame
    events: pd.DataFrame
    macro_risk: pd.DataFrame
    alerts: pd.DataFrame
    quick_view: pd.DataFrame
    export_manifest: pd.DataFrame


FINVIZ_MODULE_BLUEPRINT = pd.DataFrame([
    {"module": "Screener", "source": "latestcrosssection + screener_results", "implementation": "Implemented by company_valuation_screener; this layer reuses results and audit."},
    {"module": "Maps / Heatmaps", "source": "screener_base_frame/latestcrosssection", "implementation": "Treemap-ready payload grouped by sector, industry, country and market cap bucket."},
    {"module": "Groups", "source": "latestcrosssection", "implementation": "Aggregate overview, valuation, performance, risk, and data-quality tables."},
    {"module": "Portfolio / Watchlist", "source": "SCREENERCONFIG + screener_results + manual watchlist files", "implementation": "Saved CSV watchlists and candidate portfolios from top-ranked names."},
    {"module": "News / Events / Insider", "source": "local event/news tables and optional provider artifacts", "implementation": "Schema-first optional payload; unavailable data remains explicit, never fabricated."},
    {"module": "Futures / Forex / Crypto / Macro", "source": "macro_risk, risk_factors, refreshed macro artifacts", "implementation": "Cross-asset board from available local/provider data."},
    {"module": "Alerts", "source": "latestcrosssection, screener_results, QA, events", "implementation": "Notebook-triggerable alert log for price, scores, stale data, valuation gaps and new matches."},
    {"module": "Quick View", "source": "latestcrosssection + valuation + diagnostics + events", "implementation": "One-row analyst snapshot per selected ticker."},
    {"module": "Exports / API-ready", "source": "all layer tables", "implementation": "Stable CSV/JSON files under output/tables and output/config."},
])


DATA_MODEL_SCHEMA = pd.DataFrame([
    {"dataset": "securities_master", "grain": "ticker", "purpose": "Identity, sector, industry, country, exchange, market cap and provenance."},
    {"dataset": "universe_constituents", "grain": "ticker x source", "purpose": "Universe membership and source annotations."},
    {"dataset": "latest_cross_section", "grain": "ticker", "purpose": "Primary analytical snapshot from notebook-native latestcrosssection/screener base."},
    {"dataset": "historical_panel", "grain": "ticker x date", "purpose": "Available dfpanel/dfmerged history for future web/API use."},
    {"dataset": "valuation_outputs", "grain": "ticker/model/scenario", "purpose": "DCF, residual income, relative valuation and blended outputs."},
    {"dataset": "ml_outputs", "grain": "ticker/date/model", "purpose": "Predictions, leaderboard, feature importance and validation windows."},
    {"dataset": "diagnostics_qa", "grain": "check/ticker", "purpose": "Coverage, staleness, robustness and leakage checks."},
    {"dataset": "screener_outputs", "grain": "run/ticker", "purpose": "Filtered/ranked screen results and audit trail."},
    {"dataset": "group_summaries", "grain": "group_type/group_value/view", "purpose": "Sector, industry, country and bucket aggregate analytics."},
    {"dataset": "map_payload", "grain": "ticker", "purpose": "Treemap/heatmap-ready metrics and drill-down payload."},
    {"dataset": "watchlists", "grain": "watchlist/ticker", "purpose": "Manual and model-generated lists with timestamps."},
    {"dataset": "events", "grain": "ticker/event_type/date", "purpose": "News, earnings, filings, dividends, splits, insider/proxy events when available."},
    {"dataset": "macro_risk", "grain": "asset/series/date", "purpose": "Rates, FX, crypto, commodities and volatility context."},
    {"dataset": "alerts", "grain": "timestamp/ticker/rule", "purpose": "Monitoring outputs for thresholds and data quality."},
])


def build_architecture_review() -> pd.DataFrame:
    return pd.DataFrame([
        {"area": "Keep", "detail": "Notebook remains source of truth; latestcrosssection powers cross-sectional analytics."},
        {"area": "Keep", "detail": "Existing valuation, ML, QA, scenario and dashboard logic are reused as inputs."},
        {"area": "Add", "detail": "Finviz-inspired layer produces maps, groups, watchlists, events, macro boards, alerts and quick views."},
        {"area": "Add", "detail": "All new outputs are table-first and exportable for Streamlit/web/API migration."},
        {"area": "Constraint", "detail": "Paid-only Finviz features are approximated with local data and free-source-compatible schemas."},
        {"area": "Failure mode", "detail": "Unavailable data produces empty payloads with provenance/quality flags, not fabricated values."},
    ])


def build_securities_master(namespace: Mapping[str, Any], base: pd.DataFrame) -> pd.DataFrame:
    universe = _as_df(_first(namespace, "universe_master", "universemaster"))
    frames = []
    if not base.empty:
        frames.append(base)
    if not universe.empty:
        frames.append(universe)
    if not frames:
        return pd.DataFrame(columns=["ticker", "company_name", "sector", "industry", "country", "exchange", "market_cap", "source", "updated_at", "quality_flag"])
    work = pd.concat(frames, ignore_index=True, sort=False)
    ticker_col = resolve_column(work, "ticker") or "ticker"
    work["ticker"] = work[ticker_col].map(_normalize_ticker)
    out = pd.DataFrame({"ticker": work["ticker"]})
    for field in ["company_name", "sector", "industry", "country", "exchange", "index_membership", "market_cap", "market_cap_bucket", "price", "liquidity"]:
        out[field] = _column(work, field)
    out = out.dropna(subset=["ticker"]).drop_duplicates("ticker", keep="last")
    out["source"] = "latestcrosssection+universe_master"
    out["updated_at"] = utc_now()
    out["quality_flag"] = np.where(out["ticker"].astype(str).str.len() > 0, "PASS", "WARN")
    return out


def build_universe_constituents(namespace: Mapping[str, Any], securities_master: pd.DataFrame) -> pd.DataFrame:
    universe = _as_df(_first(namespace, "universe_master", "universemaster", "universe_constituents"))
    if universe.empty:
        universe = securities_master[["ticker"]].copy() if "ticker" in securities_master.columns else pd.DataFrame(columns=["ticker"])
        universe["source"] = "securities_master"
    if "ticker" not in universe.columns:
        ticker_col = resolve_column(universe, "ticker")
        if ticker_col:
            universe["ticker"] = universe[ticker_col]
    if "ticker" in universe.columns:
        universe["ticker"] = universe["ticker"].map(_normalize_ticker)
    for col, default in [("source", "unknown"), ("index_membership", np.nan), ("country", np.nan), ("sector", np.nan)]:
        if col not in universe.columns:
            universe[col] = default
    universe["updated_at"] = utc_now()
    universe["quality_flag"] = np.where(universe["ticker"].astype(str).str.len() > 0, "PASS", "WARN")
    return universe.dropna(subset=["ticker"]).drop_duplicates(["ticker", "source"], keep="last")


def build_group_summaries(base: pd.DataFrame) -> pd.DataFrame:
    if base.empty:
        return pd.DataFrame()
    work = base.copy()
    group_cols = [c for c in ["sector", "industry", "country", "market_cap_bucket", "index_membership"] if c in work.columns]
    rows = []
    metrics = {
        "market_cap": "median",
        "pe_ratio": "median",
        "pb_ratio": "median",
        "ev_ebitda": "median",
        "ret_21d": "mean",
        "ret_63d": "mean",
        "ret_126d": "mean",
        "volatility": "mean",
        "beta": "mean",
        "valuation_score": "mean",
        "quality_score": "mean",
        "momentum_score": "mean",
        "risk_score": "mean",
        "blended_score": "mean",
        "days_since_fundamental": "median",
    }
    for group_col in group_cols:
        for group_value, g in work.groupby(group_col, dropna=False):
            matched = g.get("fundamental_matched", pd.Series(index=g.index, dtype=object))
            if matched.notna().any():
                fundamentals_coverage = matched.astype(str).str.lower().isin(["true", "1", "yes", "matched", "available"]).mean()
            else:
                fundamentals_coverage = 0.0
            row = {
                "group_type": group_col,
                "group_value": "Unknown" if pd.isna(group_value) else group_value,
                "count": len(g),
                "fundamentals_coverage": float(fundamentals_coverage),
                "diagnostic_quality": "PASS",
                "updated_at": utc_now(),
            }
            for metric, agg in metrics.items():
                col = resolve_column(g, metric)
                if col:
                    nums = pd.to_numeric(g[col], errors="coerce")
                    row[f"{metric}_{agg}"] = getattr(nums, agg)()
            score_col = resolve_column(g, "blended_score")
            ticker_col = resolve_column(g, "ticker")
            if score_col and ticker_col and pd.to_numeric(g[score_col], errors="coerce").notna().any():
                scored = g.assign(_score=pd.to_numeric(g[score_col], errors="coerce")).sort_values("_score", ascending=False)
                row["best_constituent"] = scored.iloc[0][ticker_col]
                row["worst_constituent"] = scored.iloc[-1][ticker_col]
            rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty and "days_since_fundamental_median" in out.columns:
        out["diagnostic_quality"] = np.where(pd.to_numeric(out["days_since_fundamental_median"], errors="coerce") > 540, "WARN", "PASS")
    return out


def build_map_payload(base: pd.DataFrame) -> pd.DataFrame:
    if base.empty:
        return pd.DataFrame()
    out = pd.DataFrame()
    for field in ["ticker", "company_name", "sector", "industry", "country", "market_cap_bucket"]:
        out[field] = _column(base, field)
    for metric in [
        "market_cap", "price", "ret_21d", "ret_63d", "ret_126d", "ret_252d", "volatility",
        "valuation_score", "quality_score", "momentum_score", "risk_score", "blended_score",
        "upside_to_fair_value", "sws_snowflake_score", "scenario_downside",
    ]:
        out[metric] = _numeric(base, metric)
    out["size_metric"] = out["market_cap"].where(out["market_cap"].notna(), 1.0)
    out["color_metric_default"] = out["ret_21d"].combine_first(out["momentum_score"]).combine_first(out["blended_score"])
    out["drilldown_label"] = out["ticker"].astype(str) + " | " + out["company_name"].astype(str)
    out["source"] = "latestcrosssection/screener_base_frame"
    out["updated_at"] = utc_now()
    out["quality_flag"] = np.where(out["ticker"].notna(), "PASS", "WARN")
    return out


def build_watchlists(namespace: Mapping[str, Any], base: pd.DataFrame) -> pd.DataFrame:
    root = _output_root(namespace)
    watchlist_dir = root / "config" / "watchlists"
    watchlist_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    manual = _as_df(namespace.get("watchlists"))
    if not manual.empty:
        rows.append(manual)
    screener = _as_df(namespace.get("screener_results"))
    if screener.empty and not base.empty:
        screener = base.sort_values(resolve_column(base, "blended_score") or base.columns[0], ascending=False).head(25)
    if not screener.empty and "ticker" in screener.columns:
        candidate = screener.head(25).copy()
        candidate["watchlist"] = "candidate_from_screener"
        candidate["source"] = "screener_results"
        rows.append(candidate)
    target = _first(namespace, "MASTER_REQUEST", "MASTERREQUEST", default={}) or {}
    if isinstance(target, dict) and (target.get("ticker") or target.get("target_ticker")):
        rows.append(pd.DataFrame([{"watchlist": "selected_ticker", "ticker": target.get("ticker") or target.get("target_ticker"), "source": "MASTER_REQUEST"}]))
    out = pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame(columns=["watchlist", "ticker", "source"])
    if "ticker" in out.columns:
        out["ticker"] = out["ticker"].map(_normalize_ticker)
    out["updated_at"] = utc_now()
    out["quality_flag"] = np.where(out.get("ticker", pd.Series(dtype=str)).astype(str).str.len() > 0, "PASS", "WARN")
    _safe_write_csv(out, watchlist_dir / "watchlists.csv")
    return out


def build_events_layer(namespace: Mapping[str, Any], base: pd.DataFrame) -> pd.DataFrame:
    candidates = [
        "event_log", "events", "news_events", "news_table", "earnings_calendar",
        "calendar_events", "insider_events", "sec_filings", "dividend_calendar",
    ]
    frames = []
    for name in candidates:
        frame = _as_df(namespace.get(name))
        if not frame.empty:
            frame["source_object"] = name
            frames.append(frame)
    if frames:
        out = pd.concat(frames, ignore_index=True, sort=False)
    else:
        tickers = base["ticker"].dropna().head(0).tolist() if "ticker" in base.columns else []
        out = pd.DataFrame({"ticker": tickers, "event_type": [], "event_date": [], "headline": [], "source": []})
    for col in ["ticker", "event_type", "event_date", "headline", "source", "source_object"]:
        if col not in out.columns:
            out[col] = np.nan
    if "ticker" in out.columns:
        out["ticker"] = out["ticker"].map(_normalize_ticker)
    out["availability"] = np.where(out["event_type"].notna() | out["headline"].notna(), "available", "unavailable")
    out["updated_at"] = utc_now()
    out["quality_flag"] = np.where(out["availability"].eq("available"), "PASS", "MISSING")
    return out


def build_macro_risk_board(namespace: Mapping[str, Any]) -> pd.DataFrame:
    frames = []
    for name in ["macro_risk", "risk_factors", "riskfactorpanel", "factor_data", "fx_rates", "rates_dashboard", "commodity_prices", "crypto_prices"]:
        frame = _as_df(namespace.get(name))
        if not frame.empty:
            tmp = frame.copy()
            tmp["source_object"] = name
            frames.append(tmp)
    if not frames:
        return pd.DataFrame([
            {"asset_class": "rates", "series": "10Y yield", "value": np.nan, "change": np.nan, "source_object": "unavailable", "quality_flag": "MISSING"},
            {"asset_class": "fx", "series": "EUR/USD", "value": np.nan, "change": np.nan, "source_object": "unavailable", "quality_flag": "MISSING"},
            {"asset_class": "commodities", "series": "Gold/Oil proxy", "value": np.nan, "change": np.nan, "source_object": "unavailable", "quality_flag": "MISSING"},
            {"asset_class": "crypto", "series": "BTC proxy", "value": np.nan, "change": np.nan, "source_object": "unavailable", "quality_flag": "MISSING"},
        ]).assign(updated_at=utc_now())
    out = pd.concat(frames, ignore_index=True, sort=False)
    if "asset_class" not in out.columns:
        out["asset_class"] = out["source_object"]
    if "series" not in out.columns:
        out["series"] = out.get("ticker", out.get("factor", out.index.astype(str)))
    out["updated_at"] = utc_now()
    out["quality_flag"] = "PASS"
    return out


def build_alerts(namespace: Mapping[str, Any], base: pd.DataFrame) -> pd.DataFrame:
    rules = _first(namespace, "ALERTCONFIG", "alert_config", default={}) or {}
    thresholds = {
        "min_blended_score": rules.get("min_blended_score", 0.75),
        "min_upside": rules.get("min_upside", 0.15),
        "max_days_since_fundamental": rules.get("max_days_since_fundamental", 540),
        "max_risk_score": rules.get("max_risk_score", 0.80),
    }
    rows = []
    if base.empty:
        return pd.DataFrame(columns=["updated_at", "ticker", "rule", "severity", "value", "threshold", "message"])
    ticker = _column(base, "ticker")
    checks = [
        ("high_blended_score", _numeric(base, "blended_score"), ">=", thresholds["min_blended_score"], "INFO", "High internal blended score"),
        ("valuation_gap", _numeric(base, "upside_to_fair_value"), ">=", thresholds["min_upside"], "INFO", "Upside to fair value above threshold"),
        ("stale_fundamentals", _numeric(base, "days_since_fundamental"), ">", thresholds["max_days_since_fundamental"], "WARN", "Fundamentals may be stale"),
        ("high_risk_score", _numeric(base, "risk_score"), ">=", thresholds["max_risk_score"], "WARN", "Risk score above threshold"),
    ]
    for rule, series, op, threshold, severity, message in checks:
        if series.notna().sum() == 0:
            rows.append({"updated_at": utc_now(), "ticker": "", "rule": rule, "severity": "MISSING", "value": np.nan, "threshold": threshold, "message": f"{message}: source field unavailable"})
            continue
        mask = series > threshold if op == ">" else series >= threshold
        for idx in series[mask.fillna(False)].index:
            rows.append({"updated_at": utc_now(), "ticker": ticker.loc[idx], "rule": rule, "severity": severity, "value": series.loc[idx], "threshold": threshold, "message": message})
    events = _as_df(namespace.get("finviz_events"))
    if not events.empty and "ticker" in events.columns:
        for _, row in events.head(50).iterrows():
            if str(row.get("availability", "")).lower() == "available":
                rows.append({"updated_at": utc_now(), "ticker": row.get("ticker"), "rule": "event_available", "severity": "INFO", "value": row.get("event_type"), "threshold": "", "message": str(row.get("headline", "Event/news flag available"))})
    return pd.DataFrame(rows)


def build_quick_view(namespace: Mapping[str, Any], base: pd.DataFrame) -> pd.DataFrame:
    master = _first(namespace, "MASTER_REQUEST", "MASTERREQUEST", default={}) or {}
    selected = master.get("ticker") or master.get("target_ticker") if isinstance(master, dict) else None
    if base.empty:
        return pd.DataFrame()
    work = base.copy()
    if selected and "ticker" in work.columns:
        match = work[work["ticker"].astype(str).str.upper().eq(str(selected).upper())]
        row = match.iloc[0] if not match.empty else work.iloc[0]
    else:
        score = resolve_column(work, "blended_score")
        row = work.sort_values(score, ascending=False).iloc[0] if score else work.iloc[0]
    fields = [
        "ticker", "company_name", "sector", "industry", "country", "price", "market_cap",
        "pe_ratio", "pb_ratio", "ev_ebitda", "ret_21d", "ret_63d", "volatility",
        "valuation_score", "quality_score", "momentum_score", "risk_score",
        "blended_score", "upside_to_fair_value", "robustness_status",
        "days_since_fundamental", "ml_prediction_score",
    ]
    out = {field: row.get(resolve_column(work, field) or field, np.nan) for field in fields}
    out["updated_at"] = utc_now()
    out["source"] = "latestcrosssection/screener_base_frame"
    return pd.DataFrame([out])


def build_finviz_plotly_figures(namespace: Mapping[str, Any]) -> dict[str, Any]:
    figures: dict[str, Any] = {}
    try:
        import plotly.express as px
    except Exception:
        return figures
    maps = _as_df(namespace.get("finviz_map_payload"))
    groups = _as_df(namespace.get("finviz_group_summaries"))
    if not maps.empty and {"sector", "ticker", "size_metric"}.issubset(maps.columns):
        figures["finviz_sector_map"] = px.treemap(
            maps.fillna({"sector": "Unknown", "industry": "Unknown", "ticker": "Unknown"}),
            path=["sector", "industry", "ticker"],
            values="size_metric",
            color="color_metric_default",
            hover_data=[c for c in ["company_name", "valuation_score", "quality_score", "momentum_score", "upside_to_fair_value"] if c in maps.columns],
            title="Market Map by Sector / Industry / Ticker",
            color_continuous_scale="RdYlGn",
        )
    if not groups.empty and {"group_type", "group_value", "count"}.issubset(groups.columns):
        sector = groups[groups["group_type"].eq("sector")].copy()
        metric = "blended_score_mean" if "blended_score_mean" in sector.columns else "count"
        if not sector.empty:
            figures["finviz_groups_bar"] = px.bar(sector.sort_values(metric, ascending=False), x="group_value", y=metric, color=metric, title="Group Summary by Sector", template="plotly_white")
    return figures


def export_finviz_layer(outputs: FinvizLayerOutputs, namespace: Mapping[str, Any]) -> pd.DataFrame:
    root = _output_root(namespace)
    table_dir = root / "tables"
    config_dir = root / "config"
    payloads = {
        "FinvizArchitectureReview.csv": outputs.architecture_review,
        "FinvizDataDictionary.csv": outputs.data_dictionary,
        "FinvizSecuritiesMaster.csv": outputs.securities_master,
        "FinvizUniverseConstituents.csv": outputs.universe_constituents,
        "FinvizLatestCrossSection.csv": outputs.latest_cross_section,
        "FinvizGroupSummaries.csv": outputs.group_summaries,
        "FinvizMapPayload.csv": outputs.map_payload,
        "FinvizWatchlists.csv": outputs.watchlists,
        "FinvizEvents.csv": outputs.events,
        "FinvizMacroRisk.csv": outputs.macro_risk,
        "FinvizAlerts.csv": outputs.alerts,
        "FinvizQuickView.csv": outputs.quick_view,
    }
    rows = []
    for filename, df in payloads.items():
        path = table_dir / filename
        _safe_write_csv(df, path)
        rows.append({"artifact": filename, "path": str(path), "rows": len(df), "updated_at": utc_now()})
    config = {
        "blueprint": FINVIZ_MODULE_BLUEPRINT.to_dict("records"),
        "schema": DATA_MODEL_SCHEMA.to_dict("records"),
        "generated_at": utc_now(),
    }
    cfg_path = config_dir / "FinvizLayerConfig.json"
    _safe_write_json(config, cfg_path)
    rows.append({"artifact": "FinvizLayerConfig.json", "path": str(cfg_path), "rows": 1, "updated_at": utc_now()})
    manifest = pd.DataFrame(rows)
    _safe_write_csv(manifest, table_dir / "FinvizExportManifest.csv")
    return manifest


def run_finviz_platform_layer(namespace: dict[str, Any], screener_config: Mapping[str, Any] | None = None) -> FinvizLayerOutputs:
    if screener_config is not None or "screener_results" not in namespace:
        run_integrated_screener(namespace, screener_config or namespace.get("SCREENERCONFIG", None))
    base = build_screener_base_frame(namespace)
    if base.empty:
        base = _as_df(_first(namespace, "latestcrosssection", "latest_cross_section", "companyranking"))
    securities_master = build_securities_master(namespace, base)
    universe_constituents = build_universe_constituents(namespace, securities_master)
    group_summaries = build_group_summaries(base)
    map_payload = build_map_payload(base)
    watchlists = build_watchlists(namespace, base)
    events = build_events_layer(namespace, base)
    macro_risk = build_macro_risk_board(namespace)
    quick_view = build_quick_view(namespace, base)

    namespace["finviz_events"] = events
    alerts = build_alerts(namespace, base)
    outputs = FinvizLayerOutputs(
        architecture_review=build_architecture_review(),
        data_dictionary=DATA_MODEL_SCHEMA.copy(),
        securities_master=securities_master,
        universe_constituents=universe_constituents,
        latest_cross_section=base,
        group_summaries=group_summaries,
        map_payload=map_payload,
        watchlists=watchlists,
        events=events,
        macro_risk=macro_risk,
        alerts=alerts,
        quick_view=quick_view,
        export_manifest=pd.DataFrame(),
    )
    manifest = export_finviz_layer(outputs, namespace)
    outputs.export_manifest = manifest
    namespace.update({
        "finviz_architecture_review": outputs.architecture_review,
        "finviz_module_blueprint": FINVIZ_MODULE_BLUEPRINT.copy(),
        "finviz_data_model_schema": DATA_MODEL_SCHEMA.copy(),
        "finviz_securities_master": outputs.securities_master,
        "finviz_universe_constituents": outputs.universe_constituents,
        "finviz_latest_cross_section": outputs.latest_cross_section,
        "finviz_group_summaries": outputs.group_summaries,
        "finviz_map_payload": outputs.map_payload,
        "finviz_watchlists": outputs.watchlists,
        "finviz_events": outputs.events,
        "finviz_macro_risk": outputs.macro_risk,
        "finviz_alerts": outputs.alerts,
        "finviz_quick_view": outputs.quick_view,
        "finviz_export_manifest": outputs.export_manifest,
    })
    figures = build_finviz_plotly_figures(namespace)
    namespace["finviz_figures"] = figures
    existing = namespace.get("interactivefigures")
    if isinstance(existing, dict):
        existing.update(figures)
    else:
        namespace["interactivefigures"] = figures
    return outputs
