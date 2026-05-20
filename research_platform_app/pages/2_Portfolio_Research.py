from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import plotly.express as px
import streamlit as st

from support import configure_page, dataframe_with_download, load_portfolio_artifacts, metric_value, numeric_cols, show_empty, sidebar_roots


configure_page("Portfolio Research")

roots = sidebar_roots()
root = roots["portfolio"]
data = load_portfolio_artifacts(root)

st.title("Portfolio Research & Allocation")
st.caption("Allocation-aware research view over portfolio selection, weights, risk, performance and diagnostics artifacts.")

selection = data["selection_results"]
allocation = data["allocation"] if not data["allocation"].empty else selection
performance = data["performance"]
risk = data["risk"]

weight_col = "weight" if "weight" in allocation.columns else "engine_weight" if "engine_weight" in allocation.columns else None
turnover = metric_value(performance, ["turnover", "estimated_turnover", "portfolio_turnover"], "n/a")
volatility = metric_value(performance, ["volatility", "annual_volatility", "risk"], metric_value(risk, ["volatility", "portfolio_volatility"], "n/a"))

cols = st.columns(4)
cols[0].metric("Selected Names", len(selection) if not selection.empty else 0)
cols[1].metric("Expected Return", metric_value(performance, ["expected_return", "annual_return_net", "annual_return"], "n/a"))
cols[2].metric("Risk / Volatility", volatility)
cols[3].metric("Turnover", turnover)

with st.container(border=True):
    st.markdown("**Input origin**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("valuation_ranking", "present" if not selection.empty else "missing")
    c2.metric("portfolio_allocation", "present" if not data["allocation"].empty else "missing")
    c3.metric("engine_weights", "present" if not data["engine_weights"].empty else "missing")
    c4.metric("optimization_summary", "present" if not data["engine_metrics"].empty else "missing")

st.markdown(
    """
    <div class="rp-note">
    Portfolio selection is allocation-aware: it combines ranking, risk, weights, optimizer context and scenario diagnostics.
    It should not be interpreted as a standalone stock screener.
    </div>
    """,
    unsafe_allow_html=True,
)

tab_overview, tab_selection, tab_diagnostics = st.tabs(["Overview", "Selection Lab", "Diagnostics"])

with tab_overview:
    left, right = st.columns([2, 1])
    with left:
        if allocation.empty:
            show_empty("Allocation / weights", root)
        else:
            dataframe_with_download("Allocation / weights", allocation, "portfolio_allocation.csv")
    with right:
        if not allocation.empty and weight_col and "ticker" in allocation.columns:
            st.plotly_chart(px.pie(allocation.head(25), names="ticker", values=weight_col, title="Top Holdings", template="plotly_white"), width="stretch")
    dataframe_with_download("Performance summary", performance, "portfolio_performance.csv")

with tab_selection:
    dataframe_with_download("Portfolio selection summary", data["selection_summary"], "portfolio_selection_summary.csv")
    dataframe_with_download("Portfolio selection results", selection, "portfolio_selection_results.csv")
    if not selection.empty and "selection_score" in selection.columns:
        st.plotly_chart(px.bar(selection.head(30), x="ticker" if "ticker" in selection.columns else selection.index, y="selection_score", color="sector" if "sector" in selection.columns else None, title="Selection Score Ranking", template="plotly_white"), width="stretch")
    with st.expander("Ranking modes and active filters", expanded=False):
        dataframe_with_download("Selection presets", data["selection_presets"], "portfolio_selection_presets.csv")
        dataframe_with_download("Selection schema", data["selection_schema"], "portfolio_selection_schema.csv")

with tab_diagnostics:
    left, right = st.columns(2)
    with left:
        dataframe_with_download("Selection audit", data["selection_audit"], "portfolio_selection_audit.csv")
    with right:
        dataframe_with_download("Selection progression", data["selection_progression"], "portfolio_selection_progression.csv")
    dataframe_with_download("Risk dashboard", risk, "portfolio_risk_dashboard.csv")
    if not risk.empty and "score" in risk.columns:
        x_col = "risk_family" if "risk_family" in risk.columns else risk.columns[0]
        st.plotly_chart(px.bar(risk, x=x_col, y="score", color="status" if "status" in risk.columns else None, title="Risk Dashboard", template="plotly_white"), width="stretch")
    dataframe_with_download("Portfolio scenarios", data["scenarios"], "portfolio_scenarios.csv")
    scenarios = data["scenarios"]
    if not scenarios.empty and len(numeric_cols(scenarios)) > 0:
        y = st.selectbox("Scenario metric", numeric_cols(scenarios), key="portfolio_scenario_metric")
        x = "scenario" if "scenario" in scenarios.columns else scenarios.index
        st.plotly_chart(px.bar(scenarios, x=x, y=y, title=f"Scenario {y}", template="plotly_white"), width="stretch")
    with st.expander("Best-practice note", expanded=True):
        st.markdown("Use selection score, drawdown, volatility, scenario downside and concentration together. A high score with poor risk diagnostics needs manual review.")
