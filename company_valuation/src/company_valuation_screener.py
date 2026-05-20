"""Notebook-native FINVIZ-style screener core v2.

The screener is designed for the Company Valuation notebook, not as a generic
standalone screener. `latestcrosssection` is the master frame; ranking,
valuation, ML, diagnostics and scenario tables are auxiliary enrichments.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

try:
    from research_platform_core import (
        as_df as _core_as_df,
        first_available as _core_first,
        normalize_name,
        normalize_ticker,
        resolve_alias,
        safe_write_csv as _core_safe_write_csv,
        safe_write_json as _core_safe_write_json,
    )
except ModuleNotFoundError:
    from src.research_platform_core import (
        as_df as _core_as_df,
        first_available as _core_first,
        normalize_name,
        normalize_ticker,
        resolve_alias,
        safe_write_csv as _core_safe_write_csv,
        safe_write_json as _core_safe_write_json,
    )


UTC_NOW = lambda: datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _as_df(value: Any) -> pd.DataFrame:
    return _core_as_df(value)


def _first(namespace: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    return _core_first(namespace, *names, default=default)


FILTER_SCHEMA = {
    "descriptive": {
        "ticker": {"type": "text", "operators": ["equals", "in", "contains", "regex"]},
        "company_name": {"type": "text", "operators": ["contains", "regex", "equals"]},
        "country": {"type": "category", "operators": ["equals", "in"]},
        "region": {"type": "category", "operators": ["equals", "in"]},
        "exchange": {"type": "category", "operators": ["equals", "in"]},
        "sector": {"type": "category", "operators": ["equals", "in", "contains"]},
        "industry": {"type": "category", "operators": ["equals", "in", "contains"]},
        "index_membership": {"type": "category", "operators": ["equals", "in", "contains"]},
        "market_cap": {"type": "numeric", "operators": ["gt", "lt", "between", "top_pct"]},
        "market_cap_bucket": {"type": "category", "operators": ["equals", "in"]},
        "price": {"type": "numeric", "operators": ["gt", "lt", "between"]},
        "avg_volume": {"type": "numeric", "operators": ["gt", "lt", "between"]},
        "liquidity": {"type": "numeric", "operators": ["gt", "lt", "between"]},
        "security_type": {"type": "category", "operators": ["equals", "in"]},
        "is_etf": {"type": "boolean", "operators": ["is_true", "is_false"]},
        "is_adr": {"type": "boolean", "operators": ["is_true", "is_false"]},
    },
    "fundamental": {
        "pe_ratio": {"type": "numeric", "operators": ["gt", "lt", "between", "bottom_pct"]},
        "forward_pe": {"type": "numeric", "operators": ["gt", "lt", "between", "bottom_pct"]},
        "pb_ratio": {"type": "numeric", "operators": ["gt", "lt", "between", "bottom_pct"]},
        "ev_ebitda": {"type": "numeric", "operators": ["gt", "lt", "between", "bottom_pct"]},
        "ev_sales": {"type": "numeric", "operators": ["gt", "lt", "between", "bottom_pct"]},
        "roe": {"type": "numeric", "operators": ["gt", "lt", "between", "top_pct"]},
        "roa": {"type": "numeric", "operators": ["gt", "lt", "between", "top_pct"]},
        "gross_margin": {"type": "numeric", "operators": ["gt", "lt", "between"]},
        "operating_margin": {"type": "numeric", "operators": ["gt", "lt", "between"]},
        "net_margin": {"type": "numeric", "operators": ["gt", "lt", "between"]},
        "revenue_growth": {"type": "numeric", "operators": ["gt", "lt", "between"]},
        "earnings_growth": {"type": "numeric", "operators": ["gt", "lt", "between"]},
        "debt_equity": {"type": "numeric", "operators": ["gt", "lt", "between"]},
        "current_ratio": {"type": "numeric", "operators": ["gt", "lt", "between"]},
        "dividend_yield": {"type": "numeric", "operators": ["gt", "lt", "between"]},
        "payout_ratio": {"type": "numeric", "operators": ["gt", "lt", "between"]},
        "fcf_yield": {"type": "numeric", "operators": ["gt", "lt", "between", "top_pct"]},
        "valuation_score": {"type": "numeric", "operators": ["gt", "lt", "between", "top_pct"]},
        "quality_score": {"type": "numeric", "operators": ["gt", "lt", "between", "top_pct"]},
    },
    "technical": {
        "sma20_rel": {"type": "numeric", "operators": ["gt", "lt", "between"]},
        "sma50_rel": {"type": "numeric", "operators": ["gt", "lt", "between"]},
        "sma200_rel": {"type": "numeric", "operators": ["gt", "lt", "between"]},
        "ret_21d": {"type": "numeric", "operators": ["gt", "lt", "between", "top_pct"]},
        "ret_63d": {"type": "numeric", "operators": ["gt", "lt", "between", "top_pct"]},
        "ret_126d": {"type": "numeric", "operators": ["gt", "lt", "between", "top_pct"]},
        "ret_252d": {"type": "numeric", "operators": ["gt", "lt", "between", "top_pct"]},
        "volatility": {"type": "numeric", "operators": ["gt", "lt", "between", "bottom_pct"]},
        "atr": {"type": "numeric", "operators": ["gt", "lt", "between"]},
        "rsi": {"type": "numeric", "operators": ["gt", "lt", "between"]},
        "relative_volume": {"type": "numeric", "operators": ["gt", "lt", "between"]},
        "trend_regime": {"type": "category", "operators": ["equals", "in"]},
    },
    "risk_factor_macro": {
        "beta": {"type": "numeric", "operators": ["gt", "lt", "between"]},
        "leverage_risk": {"type": "numeric", "operators": ["gt", "lt", "between"]},
        "earnings_quality_flag": {"type": "boolean", "operators": ["is_true", "is_false"]},
        "factor_exposure": {"type": "numeric", "operators": ["gt", "lt", "between"]},
        "days_since_fundamental": {"type": "numeric", "operators": ["gt", "lt", "between"]},
        "fundamental_matched": {"type": "boolean", "operators": ["is_true", "is_false"]},
        "scenario_downside": {"type": "numeric", "operators": ["gt", "lt", "between"]},
        "macro_sensitivity": {"type": "numeric", "operators": ["gt", "lt", "between"]},
        "robustness_status": {"type": "category", "operators": ["equals", "in"]},
    },
    "internal": {
        "blended_score": {"type": "numeric", "operators": ["gt", "lt", "between", "top_pct"]},
        "valuation_score": {"type": "numeric", "operators": ["gt", "lt", "between", "top_pct"]},
        "quality_score": {"type": "numeric", "operators": ["gt", "lt", "between", "top_pct"]},
        "momentum_score": {"type": "numeric", "operators": ["gt", "lt", "between", "top_pct"]},
        "risk_score": {"type": "numeric", "operators": ["gt", "lt", "between", "top_pct"]},
        "sws_snowflake_score": {"type": "numeric", "operators": ["gt", "lt", "between", "top_pct"]},
        "upside_to_fair_value": {"type": "numeric", "operators": ["gt", "lt", "between", "top_pct"]},
        "ml_prediction_score": {"type": "numeric", "operators": ["gt", "lt", "between", "top_pct"]},
        "committee_verdict": {"type": "category", "operators": ["equals", "in", "contains"]},
    },
}


FIELD_ALIASES = {
    "ticker": ["symbol"],
    "company_name": ["name", "company", "security", "Security", "companyname", "company_name"],
    "price": ["price", "current_price", "currentprice", "adj_close", "adjclose", "close", "last_price", "lastprice"],
    "avg_volume": ["avg_volume", "avgvolume", "average_volume", "averagevolume", "avg_volume_3m", "volume"],
    "liquidity": ["liquidity", "dollar_volume", "dollarvolume", "avg_dollar_volume", "avgdollarvolume"],
    "market_cap": ["market_cap", "marketcap", "market_cap_est", "marketcapest", "mktcap"],
    "pe_ratio": ["pe_ratio", "peratio", "pe", "trailingpe"],
    "forward_pe": ["forward_pe", "forwardpe", "forwardPE"],
    "pb_ratio": ["pb_ratio", "pbratio", "pb", "pricebook"],
    "ev_ebitda": ["ev_ebitda", "evebitda", "evtoebitda"],
    "ev_sales": ["ev_sales", "evsales", "sales_multiple_proxy", "salesmultipleproxy"],
    "roe": ["roe", "return_on_equity", "returnonequity"],
    "roa": ["roa", "return_on_assets", "returnonassets"],
    "gross_margin": ["gross_margin", "grossmargin"],
    "operating_margin": ["operating_margin", "operatingmargin"],
    "net_margin": ["net_margin", "netmargin"],
    "revenue_growth": ["revenue_growth", "revenuegrowth", "sales_growth", "salesgrowth"],
    "earnings_growth": ["earnings_growth", "earningsgrowth", "eps_growth", "epsgrowth", "net_income_growth", "netincomegrowth"],
    "debt_equity": ["debt_equity", "debtequity", "debt_to_equity", "debttoequity", "total_debt_equity", "totaldebtequity"],
    "current_ratio": ["current_ratio", "currentratio"],
    "dividend_yield": ["dividend_yield", "dividendyield"],
    "payout_ratio": ["payout_ratio", "payoutratio"],
    "fcf_yield": ["fcf_yield", "fcfyield", "free_cash_flow_yield", "freecashflowyield"],
    "sma20_rel": ["sma20_rel", "sma20rel", "price_sma20", "pricesma20"],
    "sma50_rel": ["sma50_rel", "sma50rel", "price_sma50", "pricesma50"],
    "sma200_rel": ["sma200_rel", "sma200rel", "price_sma200", "pricesma200"],
    "ret_21d": ["ret_21d", "ret21d", "ret_1m", "ret1m"],
    "ret_63d": ["ret_63d", "ret63d", "ret_3m", "ret3m", "ret_126d", "ret126d"],
    "ret_126d": ["ret_126d", "ret126d", "ret_6m", "ret6m"],
    "ret_252d": ["ret_252d", "ret252d", "ret_12m", "ret12m", "annual_return", "annualreturn"],
    "volatility": ["volatility", "vol_63d", "vol63d", "vol_126d", "vol126d"],
    "beta": ["beta"],
    "days_since_fundamental": ["days_since_fundamental", "dayssincefundamental"],
    "fundamental_matched": ["fundamental_matched", "fundamentalmatched"],
    "effective_fundamental_date": ["effective_fundamental_date", "effectivefundamentaldate"],
    "robustness_status": ["robustness_status", "robustnessstatus", "qa_status", "qastatus", "status"],
    "scenario_downside": ["scenario_downside", "scenariodownside", "bear_downside", "beardownside"],
    "blended_score": ["blended_score", "blendedscore"],
    "valuation_score": ["valuation_score", "valuationscore"],
    "quality_score": ["quality_score", "qualityscore"],
    "momentum_score": ["momentum_score", "momentumscore"],
    "risk_score": ["risk_score", "riskscore"],
    "sws_snowflake_score": ["sws_snowflake_score", "swssnowflakescore"],
    "upside_to_fair_value": ["upside_to_fair_value", "upsidetofairvalue", "upside"],
    "ml_prediction_score": ["ml_prediction_score", "mlpredictionscore", "y_pred", "ypred", "prediction", "model_score", "modelscore"],
}


SCREENER_PRESETS = {
    "Quality compounders": {
        "description": "High quality, positive momentum, acceptable leverage.",
        "filters": [
            {"field": "quality_score", "op": "gt", "value": 0.60},
            {"field": "roe", "op": "gt", "value": 0.10},
            {"field": "debt_equity", "op": "lt", "value": 2.0},
            {"field": "ret_63d", "op": "gt", "value": -0.15},
        ],
        "ranking": {"mode": "weighted_score", "weights": {"quality_score": 0.40, "blended_score": 0.25, "momentum_score": 0.20, "valuation_score": 0.15}},
    },
    "Undervalued large caps": {
        "description": "Large/liquid companies with internal valuation upside.",
        "filters": [
            {"field": "market_cap", "op": "gt", "value": 10_000_000_000},
            {"field": "valuation_score", "op": "gt", "value": 0.55},
            {"field": "upside_to_fair_value", "op": "gt", "value": 0.05},
        ],
        "ranking": {"mode": "weighted_score", "weights": {"upside_to_fair_value": 0.45, "valuation_score": 0.35, "quality_score": 0.20}},
    },
    "Dividend quality": {
        "description": "Dividend payers with quality and payout discipline.",
        "filters": [
            {"field": "dividend_yield", "op": "gt", "value": 0.02},
            {"field": "payout_ratio", "op": "lt", "value": 0.80},
            {"field": "quality_score", "op": "gt", "value": 0.50},
        ],
        "ranking": {"mode": "weighted_score", "weights": {"dividend_yield": 0.35, "quality_score": 0.35, "risk_score": 0.30}},
    },
    "Growth at reasonable price": {
        "description": "Growth with valuation guardrails.",
        "filters": [
            {"field": "revenue_growth", "op": "gt", "value": 0.03},
            {"field": "pe_ratio", "op": "between", "value": [0, 35]},
            {"field": "quality_score", "op": "gt", "value": 0.45},
        ],
        "ranking": {"mode": "weighted_score", "weights": {"revenue_growth": 0.35, "quality_score": 0.30, "valuation_score": 0.20, "momentum_score": 0.15}},
    },
    "Deep value with quality floor": {
        "description": "Value names after removing weakest quality tail.",
        "filters": [
            {"field": "valuation_score", "op": "gt", "value": 0.65},
            {"field": "quality_score", "op": "gt", "value": 0.35},
            {"field": "debt_equity", "op": "lt", "value": 4.0},
        ],
        "ranking": {"mode": "weighted_score", "weights": {"valuation_score": 0.60, "quality_score": 0.25, "upside_to_fair_value": 0.15}},
    },
    "Momentum with fundamental support": {
        "description": "Price strength backed by fundamentals.",
        "filters": [
            {"field": "ret_63d", "op": "gt", "value": 0.00},
            {"field": "sma50_rel", "op": "gt", "value": 1.0},
            {"field": "quality_score", "op": "gt", "value": 0.45},
        ],
        "ranking": {"mode": "weighted_score", "weights": {"momentum_score": 0.45, "ret_63d": 0.25, "quality_score": 0.20, "risk_score": 0.10}},
    },
    "Italian banks / European financials": {
        "description": "Italy/Europe financials with internal ranking context.",
        "filters": [
            {"field": "sector", "op": "contains", "value": "financ|bank", "case": False},
            {"field": "country", "op": "in", "value": ["Italy", "IT", "Europe", "EU"]},
        ],
        "ranking": {"mode": "weighted_score", "weights": {"blended_score": 0.40, "valuation_score": 0.30, "quality_score": 0.30}},
    },
    "High-ROE, low-leverage": {
        "description": "Capital efficiency plus balance-sheet control.",
        "filters": [
            {"field": "roe", "op": "gt", "value": 0.12},
            {"field": "debt_equity", "op": "lt", "value": 1.5},
        ],
        "ranking": {"mode": "weighted_score", "weights": {"roe": 0.50, "quality_score": 0.30, "risk_score": 0.20}},
    },
    "Turnaround candidates": {
        "description": "Weak momentum but upside/valuation support.",
        "filters": [
            {"field": "ret_63d", "op": "lt", "value": 0.05},
            {"field": "upside_to_fair_value", "op": "gt", "value": 0.10},
            {"field": "quality_score", "op": "gt", "value": 0.30},
        ],
        "ranking": {"mode": "weighted_score", "weights": {"upside_to_fair_value": 0.50, "valuation_score": 0.30, "quality_score": 0.20}},
    },
    "Top-ranked internal model ideas": {
        "description": "Use notebook-native blendedscore/blended_score first.",
        "filters": [{"field": "blended_score", "op": "top_pct", "value": 0.25}],
        "ranking": {"mode": "raw_sort", "sort_by": "blended_score", "ascending": False},
    },
}


DEFAULT_RESULT_COLUMNS = [
    "screener_rank", "ticker", "company_name", "sector", "industry", "country", "exchange",
    "price", "market_cap", "market_cap_bucket", "avg_volume", "liquidity",
    "pe_ratio", "pb_ratio", "ev_ebitda", "roe", "revenue_growth", "debt_equity",
    "ret_21d", "ret_63d", "volatility", "days_since_fundamental", "fundamental_matched",
    "robustness_status", "scenario_downside", "blended_score", "valuation_score",
    "quality_score", "momentum_score", "risk_score", "sws_snowflake_score",
    "upside_to_fair_value", "ml_prediction_score", "screener_score",
]


@dataclass
class ScreenerResult:
    filtered: pd.DataFrame
    audit: pd.DataFrame
    progression: pd.DataFrame
    config: dict[str, Any]
    summary: pd.DataFrame
    data_dictionary: pd.DataFrame


def resolve_column(df: pd.DataFrame, field: str) -> str | None:
    return resolve_alias(df, field, FIELD_ALIASES)


def ensure_canonical_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for canonical in sorted({field for group in FILTER_SCHEMA.values() for field in group}):
        if canonical not in out.columns:
            col = resolve_column(out, canonical)
            if col is not None:
                out[canonical] = out[col]
    return out


def coerce_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin(["true", "1", "yes", "y", "pass", "matched", "available"])


def market_cap_bucket(value: Any) -> str:
    try:
        x = float(value)
    except Exception:
        return "unknown"
    if x >= 200_000_000_000:
        return "mega"
    if x >= 10_000_000_000:
        return "large"
    if x >= 2_000_000_000:
        return "mid"
    if x >= 300_000_000:
        return "small"
    if x >= 50_000_000:
        return "micro"
    return "nano"


def derive_core_fields(df: pd.DataFrame, namespace: Mapping[str, Any] | None = None) -> pd.DataFrame:
    out = ensure_canonical_columns(df)
    namespace = namespace or {}
    if "ticker" in out.columns:
        out["ticker"] = out["ticker"].map(normalize_ticker)
    if "company_name" not in out.columns:
        col = resolve_column(out, "company_name")
        out["company_name"] = out[col] if col else np.nan
    if "liquidity" not in out.columns and {"price", "avg_volume"}.issubset(out.columns):
        out["liquidity"] = pd.to_numeric(out["price"], errors="coerce") * pd.to_numeric(out["avg_volume"], errors="coerce")
    if "market_cap_bucket" not in out.columns and "market_cap" in out.columns:
        out["market_cap_bucket"] = out["market_cap"].map(market_cap_bucket)
    if "fundamental_matched" not in out.columns:
        eff = resolve_column(out, "effective_fundamental_date")
        days = resolve_column(out, "days_since_fundamental")
        if eff:
            out["fundamental_matched"] = out[eff].notna()
        elif days:
            out["fundamental_matched"] = pd.to_numeric(out[days], errors="coerce").notna()
        else:
            out["fundamental_matched"] = np.nan
    if "scenario_downside" not in out.columns:
        out["scenario_downside"] = derive_scenario_downside(out, namespace)
    if "robustness_status" not in out.columns:
        out["robustness_status"] = derive_robustness_status(out, namespace)
    if "is_etf" not in out.columns:
        stype = resolve_column(out, "security_type")
        out["is_etf"] = out[stype].astype(str).str.contains("ETF|FUND", case=False, na=False) if stype else False
    if "is_adr" not in out.columns:
        out["is_adr"] = out["company_name"].astype(str).str.contains("ADR", case=False, na=False) if "company_name" in out.columns else False
    return out


def derive_scenario_downside(base: pd.DataFrame, namespace: Mapping[str, Any]) -> pd.Series:
    scenario = _as_df(_first(namespace, "scenario_table", "scenariooutput", "scenario_output", "scenarioframework", "portfolio_scenarios"))
    if scenario.empty or "ticker" not in base.columns:
        return pd.Series(np.nan, index=base.index)
    scenario = scenario.copy()
    scenario["ticker"] = scenario["ticker"].map(normalize_ticker) if "ticker" in scenario.columns else ""
    downside_col = resolve_column(scenario, "scenario_downside")
    if downside_col is None:
        bear_col = next((c for c in scenario.columns if "bear" in normalize_name(c) and ("return" in normalize_name(c) or "upside" in normalize_name(c) or "downside" in normalize_name(c))), None)
        price_col = resolve_column(scenario, "price")
        if bear_col and price_col:
            scenario["scenario_downside"] = pd.to_numeric(scenario[bear_col], errors="coerce") / pd.to_numeric(scenario[price_col], errors="coerce") - 1
            downside_col = "scenario_downside"
        elif bear_col:
            scenario["scenario_downside"] = pd.to_numeric(scenario[bear_col], errors="coerce")
            downside_col = "scenario_downside"
    if downside_col is None:
        return pd.Series(np.nan, index=base.index)
    mapping = scenario.dropna(subset=["ticker"]).groupby("ticker")[downside_col].last()
    return base["ticker"].map(mapping)


def derive_robustness_status(base: pd.DataFrame, namespace: Mapping[str, Any]) -> pd.Series:
    status = pd.Series("PASS", index=base.index, dtype="object")
    days_col = resolve_column(base, "days_since_fundamental")
    if days_col:
        stale = pd.to_numeric(base[days_col], errors="coerce") > 540
        status.loc[stale.fillna(False)] = "WARN"
    matched_col = resolve_column(base, "fundamental_matched")
    if matched_col:
        unmatched = coerce_bool(base[matched_col]) == False
        status.loc[unmatched.fillna(False)] = "WARN"
    qa = _as_df(_first(namespace, "qadf", "qa_df", "qaextended", "qa_extended", "qa_report", "robustnesstable", "robustness_table"))
    if not qa.empty and "ticker" in qa.columns:
        qa = qa.copy()
        qa["ticker"] = qa["ticker"].map(normalize_ticker)
        qa_status_col = resolve_column(qa, "robustness_status") or resolve_column(qa, "status")
        if qa_status_col:
            qa_map = qa.groupby("ticker")[qa_status_col].last()
            mapped = base["ticker"].map(qa_map)
            status.loc[mapped.notna()] = mapped.loc[mapped.notna()].astype(str).str.upper()
    return status


def safe_auxiliary_join(base: pd.DataFrame, aux: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    if base.empty or aux.empty or "ticker" not in base.columns or "ticker" not in aux.columns:
        return base
    aux = aux.copy()
    aux["ticker"] = aux["ticker"].map(normalize_ticker)
    aux = aux.dropna(subset=["ticker"]).drop_duplicates("ticker", keep="last")
    add_cols = []
    normalized_existing = {normalize_name(c) for c in base.columns}
    for col in aux.columns:
        if col == "ticker":
            continue
        if col in base.columns or normalize_name(col) in normalized_existing:
            continue
        add_cols.append(col)
    if not add_cols:
        return base
    rename = {col: f"{prefix}{col}" for col in add_cols if prefix and col not in FIELD_ALIASES}
    joined = base.merge(aux[["ticker", *add_cols]].rename(columns=rename), on="ticker", how="left")
    return joined


def build_screener_base_frame(namespace: Mapping[str, Any]) -> pd.DataFrame:
    base = _as_df(_first(namespace, "latestcrosssection", "latest_cross_section"))
    if base.empty:
        base = _as_df(_first(namespace, "companyranking", "company_ranking"))
    if base.empty or "ticker" not in base.columns:
        return pd.DataFrame()
    base = base.copy()
    base["ticker"] = base["ticker"].map(normalize_ticker)
    for name in ["companyranking", "company_ranking", "valuationoutput", "valuation_output", "valuationoutput", "valuation_gap_table"]:
        base = safe_auxiliary_join(base, _as_df(namespace.get(name)))
    ml = _as_df(_first(namespace, "modelpredictions", "model_predictions", "ml_predictions"))
    if not ml.empty and "ticker" in ml.columns:
        pred_col = resolve_column(ml, "ml_prediction_score")
        if pred_col:
            pred = ml.copy()
            pred["ticker"] = pred["ticker"].map(normalize_ticker)
            pred = pred.groupby("ticker")[pred_col].mean().reset_index().rename(columns={pred_col: "ml_prediction_score"})
            base = safe_auxiliary_join(base, pred)
    return derive_core_fields(base, namespace)


def apply_single_filter(df: pd.DataFrame, spec: Mapping[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    before = len(df)
    field = str(spec.get("field", ""))
    op = str(spec.get("op", spec.get("operator", "equals"))).lower()
    value = spec.get("value")
    col = resolve_column(df, field)
    audit = {
        "field": field,
        "operator": op,
        "value": value,
        "resolved_column": col,
        "before": before,
        "after": before,
        "dropped": 0,
        "status": "SKIP",
        "detail": "",
        "updated_at": UTC_NOW(),
    }
    if before == 0:
        audit["detail"] = "No rows available"
        return df, audit
    if col is None:
        audit["detail"] = "Field unavailable after alias resolution"
        return df, audit
    series = df[col]
    try:
        if op in {"gt", "greater_than", ">"}:
            mask = pd.to_numeric(series, errors="coerce") > float(value)
        elif op in {"gte", ">="}:
            mask = pd.to_numeric(series, errors="coerce") >= float(value)
        elif op in {"lt", "less_than", "<"}:
            mask = pd.to_numeric(series, errors="coerce") < float(value)
        elif op in {"lte", "<="}:
            mask = pd.to_numeric(series, errors="coerce") <= float(value)
        elif op == "between":
            lo, hi = value
            nums = pd.to_numeric(series, errors="coerce")
            mask = nums.between(float(lo), float(hi), inclusive="both")
        elif op in {"equals", "eq", "=="}:
            mask = series.astype(str).str.upper() == str(value).upper()
        elif op in {"in", "in_list"}:
            values = [str(v).upper() for v in (value if isinstance(value, list) else [value])]
            mask = series.astype(str).str.upper().isin(values)
        elif op == "contains":
            mask = series.astype(str).str.contains(str(value), case=bool(spec.get("case", False)), regex=True, na=False)
        elif op == "regex":
            flags = 0 if spec.get("case", False) else re.IGNORECASE
            pattern = re.compile(str(value), flags=flags)
            mask = series.astype(str).map(lambda x: bool(pattern.search(x)))
        elif op == "is_true":
            mask = coerce_bool(series)
        elif op == "is_false":
            mask = ~coerce_bool(series)
        elif op in {"top_pct", "bottom_pct"}:
            nums = pd.to_numeric(series, errors="coerce")
            valid = nums.dropna()
            if valid.empty:
                audit["detail"] = "No numeric values for percentile filter"
                return df, audit
            pct = max(0.0, min(1.0, float(value)))
            n_keep = max(1, int(np.ceil(len(valid) * pct)))
            threshold = valid.nlargest(n_keep).min() if op == "top_pct" else valid.nsmallest(n_keep).max()
            mask = nums >= threshold if op == "top_pct" else nums <= threshold
        else:
            audit["detail"] = f"Unsupported operator: {op}"
            return df, audit
    except Exception as exc:
        audit["detail"] = f"Filter failed: {exc}"
        return df, audit
    out = df.loc[mask.fillna(False)].copy()
    audit.update({"after": len(out), "dropped": before - len(out), "status": "PASS", "detail": "Applied"})
    return out, audit


def percentile_score(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    nums = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if nums.notna().sum() == 0:
        return pd.Series(0.5, index=series.index)
    ranks = nums.rank(pct=True)
    return ranks if higher_is_better else 1 - ranks


def apply_ranking(df: pd.DataFrame, ranking: Mapping[str, Any] | None = None) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    ranking = ranking or {"mode": "raw_sort", "sort_by": "blended_score", "ascending": False}
    mode = str(ranking.get("mode", "raw_sort")).lower()
    skipped: list[str] = []
    if out.empty:
        out["screener_score"] = []
        out["screener_rank"] = []
        return out, skipped
    if mode == "raw_sort":
        sort_by = resolve_column(out, str(ranking.get("sort_by", "blended_score")))
        if sort_by:
            out["screener_score"] = pd.to_numeric(out[sort_by], errors="coerce")
            out = out.sort_values(sort_by, ascending=bool(ranking.get("ascending", False)), na_position="last")
        else:
            skipped.append(str(ranking.get("sort_by", "blended_score")))
            out["screener_score"] = 0.0
    elif mode in {"weighted_score", "composite"}:
        score = pd.Series(0.0, index=out.index)
        total_weight = 0.0
        for field, weight in (ranking.get("weights") or {}).items():
            col = resolve_column(out, field)
            if not col:
                skipped.append(field)
                continue
            lower = bool((ranking.get("lower_is_better") or {}).get(field, False))
            score += float(weight) * percentile_score(out[col], higher_is_better=not lower)
            total_weight += abs(float(weight))
        out["screener_score"] = score / total_weight if total_weight else 0.0
        out = out.sort_values("screener_score", ascending=False, na_position="last")
    elif mode == "sector_neutral":
        sort_by = resolve_column(out, str(ranking.get("sort_by", "blended_score")))
        sector = resolve_column(out, "sector")
        if sort_by and sector:
            out["screener_score"] = out.groupby(sector)[sort_by].rank(pct=True)
            out = out.sort_values("screener_score", ascending=False, na_position="last")
        else:
            skipped.extend([x for x, ok in [(str(ranking.get("sort_by", "blended_score")), sort_by), ("sector", sector)] if not ok])
            out["screener_score"] = 0.0
    elif mode == "quality_floor_value":
        quality = resolve_column(out, "quality_score")
        valuation = resolve_column(out, "valuation_score")
        if quality and valuation:
            out = out[pd.to_numeric(out[quality], errors="coerce") >= float(ranking.get("quality_floor", 0.4))].copy()
            out["screener_score"] = percentile_score(out[valuation], True)
            out = out.sort_values("screener_score", ascending=False)
        else:
            skipped.extend([x for x, ok in [("quality_score", quality), ("valuation_score", valuation)] if not ok])
            out["screener_score"] = 0.0
    else:
        skipped.append(f"unsupported_ranking_mode:{mode}")
        out["screener_score"] = 0.0
    out["screener_rank"] = np.arange(1, len(out) + 1)
    return out, skipped


def build_data_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group, fields in FILTER_SCHEMA.items():
        for field, spec in fields.items():
            col = resolve_column(df, field)
            rows.append({
                "group": group,
                "field": field,
                "resolved_column": col,
                "available": col is not None,
                "type": spec["type"],
                "non_null": int(df[col].notna().sum()) if col else 0,
                "aliases": ", ".join(FIELD_ALIASES.get(field, [])),
            })
    return pd.DataFrame(rows)


def run_screener(df: pd.DataFrame, config: Mapping[str, Any]) -> ScreenerResult:
    work = derive_core_fields(df)
    preset = config.get("preset")
    preset_config = SCREENER_PRESETS.get(preset, {}) if preset else {}
    filters = list(preset_config.get("filters", [])) + list(config.get("filters", []))
    ranking = config.get("ranking") or preset_config.get("ranking") or {"mode": "raw_sort", "sort_by": "blended_score", "ascending": False}
    audit_rows = []
    progression_rows = [{"step": "start", "field": "", "operator": "", "rows": len(work), "dropped": 0, "status": "START", "detail": ""}]
    for idx, spec in enumerate(filters, start=1):
        work, audit = apply_single_filter(work, spec)
        audit["step"] = idx
        audit_rows.append(audit)
        progression_rows.append({"step": idx, "field": audit["field"], "operator": audit["operator"], "rows": audit["after"], "dropped": audit["dropped"], "status": audit["status"], "detail": audit["detail"]})
    ranked, ranking_skips = apply_ranking(work, ranking)
    for field in ranking_skips:
        audit_rows.append({"field": field, "operator": "ranking", "value": ranking.get("mode"), "resolved_column": None, "before": len(work), "after": len(ranked), "dropped": 0, "status": "SKIP", "detail": "Ranking field unavailable", "updated_at": UTC_NOW(), "step": "ranking"})
    available_cols = [c for c in DEFAULT_RESULT_COLUMNS if c in ranked.columns]
    result = ranked[available_cols + [c for c in ranked.columns if c not in available_cols]].copy()
    audit = pd.DataFrame(audit_rows)
    summary = pd.DataFrame([{
        "preset": preset or "custom",
        "input_rows": len(df),
        "matched_rows": len(result),
        "filters_requested": len(filters),
        "filters_applied": int((audit["status"] == "PASS").sum()) if not audit.empty else 0,
        "filters_skipped": int((audit["status"] == "SKIP").sum()) if not audit.empty else 0,
        "ranking_mode": ranking.get("mode"),
        "ranking_skipped_fields": ", ".join(ranking_skips),
        "updated_at": UTC_NOW(),
    }])
    config_full = {"preset": preset, "filters": filters, "ranking": ranking, "description": preset_config.get("description", config.get("description", ""))}
    return ScreenerResult(result, audit, pd.DataFrame(progression_rows), config_full, summary, build_data_dictionary(df))


def safe_write_csv(df: pd.DataFrame, path: Path) -> None:
    _core_safe_write_csv(df, path)


def export_screener_outputs(result: ScreenerResult, output_root: str | Path) -> dict[str, Path]:
    root = Path(output_root)
    table_dir = root / "tables"
    config_dir = root / "config"
    paths = {
        "screener_results": table_dir / "ScreenerResults.csv",
        "screener_audit": table_dir / "ScreenerAudit.csv",
        "screener_progression": table_dir / "ScreenerProgression.csv",
        "screener_summary": table_dir / "ScreenerSummary.csv",
        "screener_schema": table_dir / "ScreenerSchema.csv",
        "screener_presets": table_dir / "ScreenerPresets.csv",
        "screener_data_dictionary": table_dir / "ScreenerDataDictionary.csv",
        "screener_config": config_dir / "ScreenerConfig.json",
    }
    safe_write_csv(result.filtered, paths["screener_results"])
    safe_write_csv(result.audit, paths["screener_audit"])
    safe_write_csv(result.progression, paths["screener_progression"])
    safe_write_csv(result.summary, paths["screener_summary"])
    safe_write_csv(screener_schema_frame(), paths["screener_schema"])
    safe_write_csv(screener_presets_frame(), paths["screener_presets"])
    safe_write_csv(result.data_dictionary, paths["screener_data_dictionary"])
    paths["screener_config"].parent.mkdir(parents=True, exist_ok=True)
    _core_safe_write_json(result.config, paths["screener_config"])
    return paths


def table_html(df: pd.DataFrame, title: str, max_rows: int = 30) -> str:
    if df is None or df.empty:
        return f"<section class='card'><h2>{escape(title)}</h2><p class='muted'>No data available.</p></section>"
    return f"<section class='card'><h2>{escape(title)}</h2>{df.head(max_rows).to_html(index=False, border=0, escape=True, classes='data-table')}</section>"


def build_screener_html_hooks(namespace: Mapping[str, Any]) -> dict[str, str]:
    results = _as_df(namespace.get("screener_results"))
    audit = _as_df(namespace.get("screener_audit"))
    summary = _as_df(namespace.get("screener_summary"))
    kpi = summary.iloc[0].to_dict() if not summary.empty else {}
    cards = "".join(
        f"<div class='card kpi'><div class='label'>{escape(label)}</div><div class='value'>{escape(str(value))}</div></div>"
        for label, value in [
            ("Preset", kpi.get("preset", "n/a")),
            ("Matches", kpi.get("matched_rows", 0)),
            ("Applied", kpi.get("filters_applied", 0)),
            ("Skipped", kpi.get("filters_skipped", 0)),
            ("Ranking", kpi.get("ranking_mode", "n/a")),
        ]
    )
    return {
        "screener_kpi_cards_html": f"<section class='grid kpi-grid'>{cards}</section>",
        "screener_top_table_html": table_html(results, "Top Screener Results", 50),
        "screener_preset_summary_html": table_html(summary, "Screener Summary", 10),
        "screener_audit_html": table_html(audit, "Screener Audit", 100),
    }


def run_integrated_screener(namespace: dict[str, Any], config: Mapping[str, Any] | None = None) -> ScreenerResult:
    config = dict(config or namespace.get("SCREENERCONFIG", {}) or {})
    if not config:
        config = {"preset": "Top-ranked internal model ideas", "filters": [], "ranking": {"mode": "raw_sort", "sort_by": "blended_score", "ascending": False}}
    base = build_screener_base_frame(namespace)
    result = run_screener(base, config)
    output_root = _first(namespace, "OUTPUTROOT", "OUTPUT_ROOT", "OUTPUT_DIR", "TABLESDIR", default=Path.cwd() / "output")
    output_root = Path(output_root)
    if output_root.name.lower() in {"tables", "figures", "logs"}:
        output_root = output_root.parent
    paths = export_screener_outputs(result, output_root)
    namespace["screener_base_frame"] = base
    namespace["screener_results"] = result.filtered
    namespace["screener_audit"] = result.audit
    namespace["screener_progression"] = result.progression
    namespace["screener_summary"] = result.summary
    namespace["screener_schema"] = screener_schema_frame()
    namespace["screener_presets"] = screener_presets_frame()
    namespace["screener_config"] = result.config
    namespace["screener_data_dictionary"] = result.data_dictionary
    namespace["screener_export_paths"] = paths
    namespace.update(build_screener_html_hooks(namespace))
    return result


def screener_schema_frame() -> pd.DataFrame:
    rows = []
    for group, fields in FILTER_SCHEMA.items():
        for field, spec in fields.items():
            rows.append({"group": group, "field": field, "type": spec["type"], "operators": ", ".join(spec["operators"]), "aliases": ", ".join(FIELD_ALIASES.get(field, []))})
    return pd.DataFrame(rows)


def screener_presets_frame() -> pd.DataFrame:
    return pd.DataFrame([
        {"preset": name, "description": cfg.get("description", ""), "filters": len(cfg.get("filters", [])), "ranking": cfg.get("ranking", {}).get("mode", "")}
        for name, cfg in SCREENER_PRESETS.items()
    ])
