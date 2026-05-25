from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import plotly.express as px
import pandas as pd
import streamlit as st

from data_bootstrap import render_bootstrap_banner
from screener_workbench import normalize_ticker
from support import configure_page, dataframe_with_download, load_company_artifacts, load_portfolio_artifacts, load_smart_money_artifacts, load_ml_stock_lab_artifacts, metric_value, numeric_cols, render_context_bar, render_feature_metadata_expander, render_footer, render_metric_metadata_expander, render_page_header, render_page_intro, render_selected_ticker_context, safe_page_link, show_empty, sidebar_roots
from ui_ops import render_missing_data_cta


configure_page("Portfolio Research")

roots = sidebar_roots()
root = roots["portfolio"]
data = load_portfolio_artifacts(root)
company_data = load_company_artifacts(roots["company"])
smart_money = load_smart_money_artifacts(roots["workspace"])
ml_lab = load_ml_stock_lab_artifacts(roots["workspace"])

render_page_header(
    "Portfolio",
    "Allocation-aware research view over portfolio selection, holdings, weights, risk, performance and diagnostics artifacts.",
    "▣",
    module="RESEARCH",
    status="READY",
)
render_context_bar()
render_page_intro(
    "Inspect portfolio selection, holdings, risk/performance diagnostics and overlays from ML and Smart Money artifacts.",
    "Start from allocation overview, then drill into selected holdings or build a sandbox from Screener ideas.",
)
render_bootstrap_banner(roots, required=["company_screener", "ml_signals"])

context_ticker = normalize_ticker(st.session_state.get("selected_ticker", ""))
render_selected_ticker_context(roots, context_ticker, expanded=False)

selection = data["selection_results"]
allocation = data["allocation"] if not data["allocation"].empty else selection
performance = data["performance"]
risk = data["risk"]
smart_scores = smart_money["scores"]
ml_signals = ml_lab["signals"]
model_factor_frames = [
    company_data["dcf_model_factors"],
    company_data["ri_model_factors"],
    company_data["eva_model_factors"],
]
model_factors = pd.DataFrame()
for frame in model_factor_frames:
    if frame.empty or "ticker" not in frame.columns:
        continue
    clean = frame.copy()
    clean["ticker"] = clean["ticker"].astype(str).str.upper().str.strip()
    if model_factors.empty:
        model_factors = clean
    else:
        add_cols = [c for c in clean.columns if c != "ticker" and c not in model_factors.columns]
        if add_cols:
            model_factors = model_factors.merge(clean[["ticker", *add_cols]], on="ticker", how="outer")

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
    c4.metric("smart_money", "present" if not smart_scores.empty else "missing")

if allocation.empty and selection.empty:
    render_missing_data_cta(
        "Portfolio",
        job_id="portfolio_research_refresh",
        output_path=root / "tables" / "PortfolioSelectionResults.csv",
        cli_hint="python research_platform_app/scheduler.py --once --jobs portfolio_research_refresh",
    )

st.markdown(
    """
    <div class="rp-note">
    Portfolio selection is allocation-aware: it combines ranking, risk, weights, optimizer context and scenario diagnostics.
    It should not be interpreted as a standalone stock screener.
    </div>
    """,
    unsafe_allow_html=True,
)

if context_ticker:
    with st.expander(f"{context_ticker} · selected holding details", expanded=False):
        portfolio_row = allocation[allocation["ticker"].map(normalize_ticker).eq(context_ticker)] if not allocation.empty and "ticker" in allocation.columns else pd.DataFrame()
        if portfolio_row.empty:
            st.info("This ticker is not present in the current allocation/selection artifact.")
        else:
            st.dataframe(portfolio_row, width="stretch", hide_index=True)

tab_overview, tab_selection, tab_smart_money, tab_ml_lab, tab_uncertainty, tab_diagnostics = st.tabs(["Overview", "Selection Lab", "Smart Money Overlay", "ML Lab", "Valuation Uncertainty", "Diagnostics"])

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
    render_metric_metadata_expander(["sharpe", "turnover", "volatility", "max_drawdown", "tracking_error", "information_ratio"], "Portfolio metric glossary")

with tab_selection:
    dataframe_with_download("Portfolio selection summary", data["selection_summary"], "portfolio_selection_summary.csv")
    dataframe_with_download("Portfolio selection results", selection, "portfolio_selection_results.csv")
    if not selection.empty and "selection_score" in selection.columns:
        st.plotly_chart(px.bar(selection.head(30), x="ticker" if "ticker" in selection.columns else selection.index, y="selection_score", color="sector" if "sector" in selection.columns else None, title="Selection Score Ranking", template="plotly_white"), width="stretch")
    render_feature_metadata_expander(
        ["factor_composite_score", "ml_score", "score_composite", "valuation_signal_score", "smart_money_score", "quality_score", "risk_score"],
        "Portfolio selection score glossary",
    )
    with st.expander("Ranking modes and active filters", expanded=False):
        dataframe_with_download("Selection presets", data["selection_presets"], "portfolio_selection_presets.csv")
        dataframe_with_download("Selection schema", data["selection_schema"], "portfolio_selection_schema.csv")

with tab_smart_money:
    st.markdown(
        """
        Smart Money overlays help distinguish pure ranking candidates from names with institutional, insider,
        activist, macro-flow or government-demand confirmation. Treat missing coverage as a risk flag, not a neutral signal.
        """
    )
    if smart_scores.empty:
        st.info("Run `smart_money_government_refresh` from Run Notebooks to add official-source overlays.")
        safe_page_link("pages/6_🧪_Notebook_Runner.py", "Open Run Notebooks")
    else:
        overlay = allocation.copy() if not allocation.empty else selection.copy()
        if not overlay.empty and "ticker" in overlay.columns and "ticker" in smart_scores.columns:
            overlay = overlay.merge(
                smart_scores[[
                    c for c in [
                        "ticker",
                        "issuer_name",
                        "composite_institutional_interest_score",
                        "ownership_change_score",
                        "insider_conviction_score",
                        "activist_pressure_score",
                        "government_demand_tailwind_score",
                        "coverage_note",
                    ] if c in smart_scores.columns
                ]],
                on="ticker",
                how="left",
                suffixes=("", "_smart_money"),
            )
        else:
            overlay = smart_scores.copy()
        dataframe_with_download("Portfolio smart money overlay", overlay, "portfolio_smart_money_overlay.csv")
        if not overlay.empty and "composite_institutional_interest_score" in overlay.columns:
            x = "ticker" if "ticker" in overlay.columns else overlay.index
            st.plotly_chart(px.bar(overlay.head(30), x=x, y="composite_institutional_interest_score", color="coverage_note" if "coverage_note" in overlay.columns else None, title="Portfolio Candidates · Smart Money Confirmation", template="plotly_white"), width="stretch")
    with st.expander("Official-source coverage", expanded=False):
        dataframe_with_download("Smart Money coverage", smart_money["coverage"], "portfolio_smart_money_coverage.csv")

with tab_ml_lab:
    st.markdown("ML Stock Lab turns model-implied mispricing into ranking and quintile diagnostics for allocation review.")
    if ml_signals.empty:
        st.info("Run ML Stock Lab to add model-implied fair value, z-score and quintile outputs.")
        safe_page_link("pages/9_ML_Stock_Lab.py", "Open ML Stock Lab")
    else:
        overlay = allocation.copy() if not allocation.empty else selection.copy()
        if not overlay.empty and "ticker" in overlay.columns and "ticker" in ml_signals.columns:
            overlay = overlay.merge(
                ml_signals[[c for c in ["ticker", "fair_value_hat", "mispricing_rel", "zscore", "rank"] if c in ml_signals.columns]],
                on="ticker",
                how="left",
            )
        else:
            overlay = ml_signals.copy()
        dataframe_with_download("Portfolio ML Stock Lab overlay", overlay, "portfolio_ml_stock_lab_overlay.csv")
        if not overlay.empty and "zscore" in overlay.columns:
            x = "ticker" if "ticker" in overlay.columns else overlay.index
            st.plotly_chart(px.bar(overlay.sort_values("zscore", ascending=False).head(30), x=x, y="zscore", title="Portfolio Candidates · ML Mispricing", template="plotly_white"), width="stretch")
    dataframe_with_download("ML Stock Lab quintile metrics", ml_lab["quintile_metrics"], "portfolio_ml_stock_lab_quintile_metrics.csv")

with tab_uncertainty:
    st.markdown("Scenario uncertainty connects valuation-model dispersion to portfolio review. High upside with wide bands should receive lower conviction until assumptions are reviewed.")
    overlay = allocation.copy() if not allocation.empty else selection.copy()
    if overlay.empty or model_factors.empty or "ticker" not in overlay.columns:
        st.info("Run Company Valuation DCF/Residual Income/EVA scenario artifacts to add portfolio uncertainty overlays.")
        safe_page_link("pages/2_🔬_Valuation_Research.py", "Open Valuation Research")
    else:
        overlay = overlay.copy()
        overlay["ticker"] = overlay["ticker"].astype(str).str.upper().str.strip()
        uncertainty = overlay.merge(model_factors, on="ticker", how="left")
        dataframe_with_download("Portfolio valuation uncertainty overlay", uncertainty, "portfolio_valuation_uncertainty_overlay.csv")
        spread_cols = [c for c in ["dcf_scenario_spread", "residual_income_scenario_spread", "eva_scenario_spread"] if c in uncertainty.columns]
        gap_cols = [c for c in ["dcf_mispricing", "residual_income_mispricing", "eva_value_gap"] if c in uncertainty.columns]
        if spread_cols:
            spread_col = st.selectbox("Uncertainty metric", spread_cols, key="portfolio_uncertainty_metric")
            x_col = "ticker" if "ticker" in uncertainty.columns else uncertainty.index
            st.plotly_chart(px.bar(uncertainty.sort_values(spread_col, ascending=False).head(30), x=x_col, y=spread_col, title="Scenario uncertainty by holding", template="plotly_white"), width="stretch")
        if gap_cols:
            gap_col = st.selectbox("Model gap metric", gap_cols, key="portfolio_gap_metric")
            x_col = "ticker" if "ticker" in uncertainty.columns else uncertainty.index
            st.plotly_chart(px.bar(uncertainty.sort_values(gap_col, ascending=False).head(30), x=x_col, y=gap_col, title="Model-implied valuation gap by holding", template="plotly_white"), width="stretch")
        if weight_col and weight_col in uncertainty.columns and spread_cols:
            weights = pd.to_numeric(uncertainty[weight_col], errors="coerce").fillna(0.0)
            if weights.abs().sum() > 0:
                wavg = {}
                for col in spread_cols + gap_cols:
                    vals = pd.to_numeric(uncertainty[col], errors="coerce")
                    wavg[col] = float((vals.fillna(0.0) * weights).sum() / weights.abs().sum())
                st.json({"weighted_portfolio_model_risk": wavg})

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

render_footer()
