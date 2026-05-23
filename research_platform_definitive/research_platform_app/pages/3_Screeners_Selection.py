from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import plotly.express as px
import streamlit as st

from operations import generate_company_artifacts, generate_portfolio_artifacts
from support import configure_page, dataframe_with_download, load_company_artifacts, load_portfolio_artifacts, load_smart_money_artifacts, load_ml_stock_lab_artifacts, numeric_cols, sidebar_roots


configure_page("Screeners / Selection")

roots = sidebar_roots()
company = load_company_artifacts(roots["company"])
portfolio = load_portfolio_artifacts(roots["portfolio"])
smart_money = load_smart_money_artifacts(roots["workspace"])
ml_lab = load_ml_stock_lab_artifacts(roots["workspace"])

st.title("Screeners / Selection")
st.caption("Professional screening, portfolio-aware selection, audit trails, presets and schema coverage.")

with st.expander("Operational refresh", expanded=False):
    st.markdown("Rebuild lightweight screener/selection artifacts from already exported notebook tables.")
    left, right = st.columns(2)
    if left.button("Regenerate Company Screener Layer", width="stretch"):
        result = generate_company_artifacts(roots["company"])
        (st.success if result["ok"] else st.warning)(result["message"])
        if result.get("details"):
            st.json(result["details"])
        company = load_company_artifacts(roots["company"])
    if right.button("Regenerate Portfolio Selection Layer", width="stretch"):
        result = generate_portfolio_artifacts(roots["portfolio"], roots["workspace"])
        (st.success if result["ok"] else st.warning)(result["message"])
        if result.get("details"):
            st.json(result["details"])
        portfolio = load_portfolio_artifacts(roots["portfolio"])

screener = company["screener_results"]
selection = portfolio["selection_results"]
smart_scores = smart_money["scores"]
ml_signals = ml_lab["signals"]

with st.container(border=True):
    c1, c2, c3, c4 = st.columns(4)
    universe_options = ["All"]
    for col in ["index_membership", "index", "country", "sector"]:
        if col in screener.columns:
            universe_options += sorted(screener[col].dropna().astype(str).unique().tolist())
            break
    universe = c1.selectbox("Universe", sorted(set(universe_options)))
    score_cols = [c for c in numeric_cols(screener) if any(token in c.lower() for token in ["score", "return", "ret", "upside", "quality", "risk"])]
    feature_blocks = c2.multiselect("Feature Blocks", score_cols, default=score_cols[:4])
    ranking_mode = c3.selectbox("Ranking Mode", ["Notebook Rank", "Score Desc", "Quality First", "Risk Aware"])
    preset_options = ["Current Export"]
    if not company["screener_presets"].empty:
        preset_options += company["screener_presets"].iloc[:, 0].dropna().astype(str).unique().tolist()
    preset = c4.selectbox("Preset / Profile", preset_options)


def filter_universe(df):
    if universe == "All" or df.empty:
        return df
    for col in ["index_membership", "index", "country", "sector"]:
        if col in df.columns:
            return df[df[col].astype(str).eq(universe)]
    return df


company_view = filter_universe(screener)
if ranking_mode == "Score Desc":
    for col in ["screener_score", "blendedscore", "blended_score", "selection_score"]:
        if col in company_view.columns:
            company_view = company_view.sort_values(col, ascending=False)
            break
elif ranking_mode == "Quality First":
    for col in ["qualityscore", "quality_score"]:
        if col in company_view.columns:
            company_view = company_view.sort_values(col, ascending=False)
            break
elif ranking_mode == "Risk Aware":
    for col in ["riskscore", "risk_score"]:
        if col in company_view.columns:
            company_view = company_view.sort_values(col, ascending=True)
            break

cols = st.columns(4)
cols[0].metric("Company Rows", len(company_view))
cols[1].metric("Company Audit Rows", len(company["screener_audit"]))
cols[2].metric("Portfolio Rows", len(selection))
cols[3].metric("ML Signals", len(ml_signals))

company_tab, portfolio_tab, smart_tab, ml_tab = st.tabs(["Company Screener", "Portfolio Selection", "Smart Money Screener", "ML Lab Screener"])

with company_tab:
    st.subheader("Ranking and Signals")
    dataframe_with_download("Screener summary", company["screener_summary"], "company_screener_summary.csv")
    dataframe_with_download("Screener results", company_view, "company_screener_results.csv")
    chart_cols = feature_blocks or score_cols[:4]
    if not company_view.empty and chart_cols:
        metric = chart_cols[0]
        st.plotly_chart(px.histogram(company_view, x=metric, nbins=30, title=f"{metric} distribution", template="plotly_white"), width="stretch")
        corr_cols = [c for c in chart_cols if c in company_view.columns][:8]
        if len(corr_cols) >= 2:
            corr = company_view[corr_cols].corr(numeric_only=True)
            st.plotly_chart(px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", title="Feature block score heatmap"), width="stretch")
    with st.expander("Audit / Progression", expanded=False):
        dataframe_with_download("Screener audit", company["screener_audit"], "company_screener_audit.csv")
        dataframe_with_download("Screener progression", company["screener_progression"], "company_screener_progression.csv")

with portfolio_tab:
    st.subheader("Portfolio-Aware Selection")
    dataframe_with_download("Selection summary", portfolio["selection_summary"], "portfolio_selection_summary.csv")
    dataframe_with_download("Selection results", selection, "portfolio_selection_results.csv")
    if not selection.empty and "selection_score" in selection.columns:
        st.plotly_chart(px.histogram(selection, x="selection_score", nbins=25, title="Portfolio selection score distribution", template="plotly_white"), width="stretch")
    with st.expander("Audit / Progression", expanded=False):
        dataframe_with_download("Selection audit", portfolio["selection_audit"], "portfolio_selection_audit.csv")
        dataframe_with_download("Selection progression", portfolio["selection_progression"], "portfolio_selection_progression.csv")

with smart_tab:
    st.subheader("Official-Source Smart Money Screener")
    if smart_scores.empty:
        st.info("Smart Money screener is waiting for official-source files or a refresh run.")
        st.page_link("pages/8_Smart_Money_Gov_Data.py", label="Open Smart Money Government Data Engine")
    else:
        dataframe_with_download("Smart Money scores", smart_scores, "smart_money_scores.csv")
        if "composite_institutional_interest_score" in smart_scores.columns:
            st.plotly_chart(px.histogram(smart_scores, x="composite_institutional_interest_score", color="coverage_note" if "coverage_note" in smart_scores.columns else None, title="Smart Money score distribution", template="plotly_white"), width="stretch")
            score_cols_sm = [c for c in numeric_cols(smart_scores) if c.endswith("_score")]
            if len(score_cols_sm) >= 2:
                st.plotly_chart(px.imshow(smart_scores[score_cols_sm].corr(numeric_only=True), text_auto=True, color_continuous_scale="RdBu_r", title="Smart Money component correlation"), width="stretch")
    with st.expander("Coverage / event feed", expanded=False):
        dataframe_with_download("Smart Money coverage", smart_money["coverage"], "smart_money_coverage.csv")
        dataframe_with_download("Smart Money event feed", smart_money["events"], "smart_money_event_feed.csv")

with ml_tab:
    st.subheader("ML Fair-Value / Mispricing Screener")
    if ml_signals.empty:
        st.info("ML Stock Lab screener is waiting for generated artifacts.")
        st.page_link("pages/9_ML_Stock_Lab.py", label="Open ML Stock Lab")
    else:
        dataframe_with_download("ML Stock Lab signals", ml_signals, "ml_stock_lab_signals.csv")
        if "zscore" in ml_signals.columns:
            st.plotly_chart(px.histogram(ml_signals, x="zscore", nbins=30, title="ML z-score distribution", template="plotly_white"), width="stretch")
    with st.expander("Quintile diagnostics", expanded=False):
        dataframe_with_download("ML Stock Lab quintile returns", ml_lab["quintile_returns"], "ml_stock_lab_quintile_returns.csv")
        dataframe_with_download("ML Stock Lab quintile metrics", ml_lab["quintile_metrics"], "ml_stock_lab_quintile_metrics.csv")

with st.expander("Schemas, presets, maps and groups", expanded=False):
    left, right = st.columns(2)
    with left:
        st.markdown("### Company")
        dataframe_with_download("Company screener presets", company["screener_presets"], "company_screener_presets.csv")
        dataframe_with_download("Company screener schema", company["screener_schema"], "company_screener_schema.csv")
        dataframe_with_download("Finviz group summaries", company["finviz_groups"], "finviz_group_summaries.csv")
    with right:
        st.markdown("### Portfolio")
        dataframe_with_download("Portfolio selection presets", portfolio["selection_presets"], "portfolio_selection_presets.csv")
        dataframe_with_download("Portfolio selection schema", portfolio["selection_schema"], "portfolio_selection_schema.csv")
        dataframe_with_download("Finviz map payload", company["finviz_maps"], "finviz_map_payload.csv")

with st.expander("Interpretation note", expanded=True):
    st.markdown(
        """
        Filters and presets are auditable. Skipped filters usually mean source fields were unavailable after alias
        resolution, so they should be treated as coverage diagnostics rather than silent success.
        """
    )
