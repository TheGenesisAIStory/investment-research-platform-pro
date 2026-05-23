"""Explainable smart-money analytics from normalized official-source tables."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schemas import SCORE_COLUMNS, empty_frame


def _score(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() == 0:
        return pd.Series([np.nan] * len(values), index=values.index)
    pct = values.rank(pct=True)
    return pct if higher_is_better else 1 - pct


def ownership_analytics(holdings: pd.DataFrame) -> pd.DataFrame:
    """Compute issuer-level 13F ownership signals."""
    if holdings.empty:
        return pd.DataFrame(columns=["ticker", "issuer_name", "new_13f_holders", "holder_count", "position_value_usd", "ownership_change_score"])
    df = holdings.copy()
    df["position_value_usd"] = pd.to_numeric(df["position_value_usd"], errors="coerce")
    group_keys = ["ticker", "issuer_name"]
    out = df.groupby(group_keys, dropna=False).agg(
        holder_count=("filer_name", "nunique"),
        position_value_usd=("position_value_usd", "sum"),
        report_dates=("report_date", "nunique"),
    ).reset_index()
    out["new_13f_holders"] = out["holder_count"]
    out["ownership_change_score"] = _score(out["position_value_usd"].fillna(0)) * 0.6 + _score(out["holder_count"].fillna(0)) * 0.4
    return out


def insider_analytics(insiders: pd.DataFrame) -> pd.DataFrame:
    """Compute explainable insider conviction score from Form 4-like records."""
    if insiders.empty:
        return pd.DataFrame(columns=["ticker", "issuer_name", "insider_net_value_usd", "insider_buy_count", "insider_conviction_score"])
    df = insiders.copy()
    code = df["transaction_code"].astype(str).str.upper()
    sign = np.where(code.isin(["P", "A"]), 1, np.where(code.isin(["S", "D"]), -1, 0))
    df["signed_value"] = pd.to_numeric(df["position_value_usd"], errors="coerce").fillna(0) * sign
    df["is_buy"] = sign > 0
    out = df.groupby(["ticker", "issuer_name"], dropna=False).agg(
        insider_net_value_usd=("signed_value", "sum"),
        insider_buy_count=("is_buy", "sum"),
    ).reset_index()
    out["insider_conviction_score"] = _score(out["insider_net_value_usd"].fillna(0)) * 0.7 + _score(out["insider_buy_count"].fillna(0)) * 0.3
    return out


def beneficial_ownership_analytics(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=["ticker", "issuer_name", "beneficial_event_count", "active_event_count", "activist_pressure_score"])
    df = events.copy()
    active = df["active_passive_flag"].astype(str).str.contains("active", case=False, na=False)
    out = df.groupby(["ticker", "issuer_name"], dropna=False).agg(
        beneficial_event_count=("source_form", "count"),
        active_event_count=("active_passive_flag", lambda x: int(x.astype(str).str.contains("active", case=False, na=False).sum())),
        max_ownership_percent=("ownership_percent", "max"),
    ).reset_index()
    out["activist_pressure_score"] = _score(out["active_event_count"].fillna(0)) * 0.6 + _score(out["max_ownership_percent"].fillna(0)) * 0.4
    return out


def government_spending_analytics(spending: pd.DataFrame) -> pd.DataFrame:
    if spending.empty:
        return pd.DataFrame(columns=["ticker", "issuer_name", "award_amount", "award_count", "government_demand_tailwind_score"])
    df = spending.copy()
    df["award_amount"] = pd.to_numeric(df["award_amount"], errors="coerce")
    out = df.groupby(["ticker", "issuer_name"], dropna=False).agg(
        award_amount=("award_amount", "sum"),
        award_count=("award_amount", "count"),
    ).reset_index()
    out["government_demand_tailwind_score"] = _score(out["award_amount"].fillna(0)) * 0.75 + _score(out["award_count"].fillna(0)) * 0.25
    return out


def macro_support_score(macro_positioning: pd.DataFrame, capital_flows: pd.DataFrame) -> tuple[float | np.floating | None, float | np.floating | None]:
    macro_score = None
    flow_score = None
    if not macro_positioning.empty and "net_position_percentile" in macro_positioning:
        macro_score = pd.to_numeric(macro_positioning["net_position_percentile"], errors="coerce").dropna().tail(20).mean()
    if not capital_flows.empty and "flow_percentile" in capital_flows:
        flow_score = pd.to_numeric(capital_flows["flow_percentile"], errors="coerce").dropna().tail(20).mean()
    return macro_score, flow_score


def smart_money_screener(
    holdings: pd.DataFrame,
    insiders: pd.DataFrame,
    beneficial_events: pd.DataFrame,
    government_spending: pd.DataFrame,
    macro_positioning: pd.DataFrame,
    capital_flows: pd.DataFrame,
) -> pd.DataFrame:
    """Build the explainable composite issuer ranking requested by the platform."""
    frames = [
        ownership_analytics(holdings),
        insider_analytics(insiders),
        beneficial_ownership_analytics(beneficial_events),
        government_spending_analytics(government_spending),
    ]
    base = frames[0]
    for frame in frames[1:]:
        if base.empty:
            base = frame
        elif not frame.empty:
            base = base.merge(frame, on=["ticker", "issuer_name"], how="outer")
    if base.empty:
        return empty_frame(SCORE_COLUMNS)

    macro_score, flow_score = macro_support_score(macro_positioning, capital_flows)
    base["macro_positioning_score"] = macro_score
    base["cross_border_flow_support_score"] = flow_score
    score_cols = [
        "ownership_change_score",
        "insider_conviction_score",
        "activist_pressure_score",
        "government_demand_tailwind_score",
        "cross_border_flow_support_score",
        "macro_positioning_score",
    ]
    for col in score_cols:
        if col not in base.columns:
            base[col] = np.nan
    base["coverage_count"] = base[score_cols].notna().sum(axis=1)
    base["composite_institutional_interest_score"] = base[score_cols].mean(axis=1, skipna=True)
    base["coverage_note"] = np.where(base["coverage_count"] >= 3, "multi_source", np.where(base["coverage_count"] > 0, "partial", "missing"))
    for col in SCORE_COLUMNS:
        if col not in base.columns:
            base[col] = pd.NA
    return base[SCORE_COLUMNS].sort_values("composite_institutional_interest_score", ascending=False, na_position="last").reset_index(drop=True)


def event_feed(insiders: pd.DataFrame, beneficial_events: pd.DataFrame, spending: pd.DataFrame, holdings: pd.DataFrame) -> pd.DataFrame:
    """Build chronological event feed across official-source signals."""
    rows: list[pd.DataFrame] = []
    if not beneficial_events.empty:
        rows.append(beneficial_events.assign(event_source="13D/13G", event_label=beneficial_events.get("event_type", "beneficial_ownership"), event_date=beneficial_events.get("filing_date")))
    if not insiders.empty:
        rows.append(insiders.assign(event_source="Form 4", event_label=insiders.get("transaction_code", "insider_transaction"), event_date=insiders.get("filing_date")))
    if not spending.empty:
        rows.append(spending.assign(event_source="USAspending", event_label=spending.get("award_type", "award"), event_date=spending.get("award_date")))
    if not holdings.empty:
        rows.append(holdings.assign(event_source="13F", event_label="holding_snapshot", event_date=holdings.get("filing_date")))
    if not rows:
        return pd.DataFrame(columns=["event_date", "event_source", "event_label", "ticker", "issuer_name", "filer_name", "source"])
    feed = pd.concat(rows, ignore_index=True, sort=False)
    cols = ["event_date", "event_source", "event_label", "ticker", "issuer_name", "filer_name", "insider_name", "position_value_usd", "award_amount", "source"]
    for col in cols:
        if col not in feed.columns:
            feed[col] = pd.NA
    return feed[cols].sort_values("event_date", ascending=False, na_position="last").reset_index(drop=True)


def issuer_deep_dive(ticker: str, datasets: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    ticker = str(ticker).upper()
    return {name: df[df.get("ticker", pd.Series(dtype=str)).astype(str).str.upper().eq(ticker)].copy() if not df.empty and "ticker" in df.columns else pd.DataFrame() for name, df in datasets.items()}


def manager_deep_dive(manager: str, holdings: pd.DataFrame) -> pd.DataFrame:
    if holdings.empty or "filer_name" not in holdings.columns:
        return pd.DataFrame()
    mask = holdings["filer_name"].astype(str).str.contains(str(manager), case=False, na=False)
    view = holdings[mask].copy()
    if view.empty:
        return view
    view["position_value_usd"] = pd.to_numeric(view["position_value_usd"], errors="coerce")
    return view.sort_values("position_value_usd", ascending=False, na_position="last")


def sector_theme_monitor(scores: pd.DataFrame) -> pd.DataFrame:
    if scores.empty or "sector" not in scores.columns:
        return pd.DataFrame(columns=["sector", "issuer_count", "avg_composite_score", "top_issuer"])
    return scores.groupby("sector", dropna=False).agg(
        issuer_count=("ticker", "count"),
        avg_composite_score=("composite_institutional_interest_score", "mean"),
        top_issuer=("issuer_name", "first"),
    ).reset_index().sort_values("avg_composite_score", ascending=False, na_position="last")
