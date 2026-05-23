"""Plotly chart builders for Smart Money Government Data Engine."""

from __future__ import annotations

import pandas as pd


def _px():
    try:
        import plotly.express as px
        return px
    except Exception:
        return None


def score_ranking_chart(scores: pd.DataFrame):
    px = _px()
    if px is None or scores.empty:
        return None
    y = "composite_institutional_interest_score"
    x = "ticker" if "ticker" in scores.columns else "issuer_name"
    return px.bar(scores.head(25), x=x, y=y, color="coverage_note" if "coverage_note" in scores.columns else None, title="Smart Money Composite Ranking", template="plotly_white")


def sector_heatmap(sector_summary: pd.DataFrame):
    px = _px()
    if px is None or sector_summary.empty:
        return None
    return px.treemap(sector_summary, path=["sector"], values="issuer_count", color="avg_composite_score", color_continuous_scale="RdYlGn", title="Sector Smart Money Heatmap")


def insider_bar(insiders: pd.DataFrame):
    px = _px()
    if px is None or insiders.empty:
        return None
    if "ticker" not in insiders.columns or "position_value_usd" not in insiders.columns:
        return None
    view = insiders.copy()
    view["position_value_usd"] = pd.to_numeric(view["position_value_usd"], errors="coerce")
    grouped = view.groupby("ticker", dropna=False)["position_value_usd"].sum().reset_index().sort_values("position_value_usd", ascending=False).head(25)
    return px.bar(grouped, x="ticker", y="position_value_usd", title="Insider Buy/Sell Value Proxy", template="plotly_white")


def government_exposure_treemap(spending: pd.DataFrame):
    px = _px()
    if px is None or spending.empty:
        return None
    if "award_amount" not in spending.columns:
        return None
    view = spending.copy()
    view["award_amount"] = pd.to_numeric(view["award_amount"], errors="coerce").fillna(0)
    path = [c for c in ["sector", "issuer_name", "recipient_name"] if c in view.columns]
    if not path:
        return None
    return px.treemap(view, path=path, values="award_amount", title="Government Spending Exposure", template="plotly_white")


def cot_percentile_chart(macro_positioning: pd.DataFrame):
    px = _px()
    if px is None or macro_positioning.empty:
        return None
    if not {"report_date", "net_position_percentile"}.issubset(macro_positioning.columns):
        return None
    return px.line(macro_positioning, x="report_date", y="net_position_percentile", color="market" if "market" in macro_positioning.columns else None, title="COT Net Positioning Percentile", template="plotly_white")


def tic_flow_chart(capital_flows: pd.DataFrame):
    px = _px()
    if px is None or capital_flows.empty:
        return None
    if not {"report_date", "flow_usd"}.issubset(capital_flows.columns):
        return None
    return px.line(capital_flows, x="report_date", y="flow_usd", color="country" if "country" in capital_flows.columns else None, title="TIC Cross-Border Flow Proxy", template="plotly_white")
