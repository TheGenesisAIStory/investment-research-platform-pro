from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import pandas as pd
import plotly.express as px
import streamlit as st

from support import configure_page, dataframe_with_download, load_company_artifacts, metric_value, numeric_cols, show_empty, sidebar_roots


configure_page("Valuation Research")

roots = sidebar_roots()
root = roots["company"]
data = load_company_artifacts(root)

st.title("Equity Valuation Research")
st.caption("Notebook-generated fair value, ranking, screener context, diagnostics and model evidence.")

valuation_gap = data["valuation_gap"]
extended = data["extended_valuation"]
screener = data["screener_results"]
summary = data["screener_summary"]

tickers: list[str] = []
for frame in [valuation_gap, extended, screener]:
    if not frame.empty and "ticker" in frame.columns:
        tickers.extend(frame["ticker"].dropna().astype(str).unique().tolist())
tickers = sorted(set(tickers))

control = st.container(border=True)
with control:
    c1, c2, c3 = st.columns(3)
    selected = c1.selectbox("Ticker", ["All", *tickers], index=0) if tickers else "All"
    universe_options = ["All"]
    for col in ["index_membership", "index", "country", "sector"]:
        if col in screener.columns:
            universe_options += sorted(screener[col].dropna().astype(str).unique().tolist())
            break
    universe = c2.selectbox("Universe / Segment", sorted(set(universe_options)))
    profile = c3.selectbox("Research Profile", ["Balanced", "Value", "Growth", "Quality", "Dividend"])


def filter_ticker(df: pd.DataFrame) -> pd.DataFrame:
    if selected == "All" or df.empty or "ticker" not in df.columns:
        return df
    return df[df["ticker"].astype(str).eq(selected)]


def filter_universe(df: pd.DataFrame) -> pd.DataFrame:
    if universe == "All" or df.empty:
        return df
    for col in ["index_membership", "index", "country", "sector"]:
        if col in df.columns:
            return df[df[col].astype(str).eq(universe)]
    return df


valuation_view = filter_ticker(filter_universe(valuation_gap if not valuation_gap.empty else extended))
screener_view = filter_ticker(filter_universe(screener))

cols = st.columns(4)
cols[0].metric("Ticker", selected if selected != "All" else "Universe")
cols[1].metric("Fair Value", metric_value(valuation_view, ["blended_fair_value", "fair_value", "target_price", "value"], "n/a"))
cols[2].metric("Upside", metric_value(valuation_view, ["upside", "upside_to_fair_value", "valuation_gap"], "n/a"))
cols[3].metric("Screener Matches", len(screener_view) if not screener_view.empty else 0)

if valuation_view.empty:
    st.warning("Run Company Valuation notebook to compute fair value and valuation outputs.")
    st.page_link("pages/5_Run_Notebooks.py", label="Open Run Notebooks")

status_box = st.container(border=True)
with status_box:
    st.markdown("**Data status**")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Coverage Rows", len(screener_view) if not screener_view.empty else len(valuation_view))
    s2.metric("Fundamental Age", metric_value(screener_view, ["dayssincefundamental", "days_since_fundamental"], "n/a"))
    s3.metric("Robustness", metric_value(screener_view, ["robustness_status", "data_quality_status"], "n/a"))
    s4.metric("Profile", profile)

st.markdown(
    """
    <div class="rp-note">
    Valuation ranking is thesis support, not a price target guarantee. Pair fair value with coverage,
    staleness, model assumptions, robustness and screener audit before relying on a shortlist.
    </div>
    """,
    unsafe_allow_html=True,
)

tab_overview, tab_models, tab_screener, tab_diagnostics = st.tabs(["Overview", "Models", "Screener Context", "Diagnostics & Caveats"])

with tab_overview:
    if valuation_view.empty:
        show_empty("Valuation summary", root)
    else:
        dataframe_with_download("Valuation summary", valuation_view, "valuation_summary.csv")
    if not valuation_view.empty and len(numeric_cols(valuation_view)) >= 1:
        value_col = st.selectbox("Chart metric", numeric_cols(valuation_view), key="valuation_metric")
        label_col = "ticker" if "ticker" in valuation_view.columns else valuation_view.index
        st.plotly_chart(px.bar(valuation_view.head(40), x=label_col, y=value_col, title=f"{value_col} by ticker", template="plotly_white"), width="stretch")

with tab_models:
    dataframe_with_download("Extended valuation results", filter_ticker(extended), "extended_valuation_results.csv")
    dataframe_with_download("Valuation model registry", data["valuation_models"], "valuation_model_registry.csv")
    dataframe_with_download("Valuation assumptions", data["valuation_assumptions"], "valuation_assumptions.csv")
    with st.expander("Model interpretation", expanded=True):
        st.markdown("DCF, residual income, multiples and scenario outputs should be compared as a range. Wide dispersion is a model-risk signal, not just noise.")

with tab_screener:
    dataframe_with_download("Screener summary", summary, "screener_summary.csv")
    dataframe_with_download("Screener results", screener_view, "screener_results.csv")
    if not screener_view.empty and {"valuation_score", "quality_score"}.issubset(screener_view.columns):
        color_col = "screener_score" if "screener_score" in screener_view.columns else None
        st.plotly_chart(px.scatter(screener_view, x="valuation_score", y="quality_score", color=color_col, hover_name="ticker" if "ticker" in screener_view.columns else None, title="Valuation vs Quality", template="plotly_white"), width="stretch")

with tab_diagnostics:
    dataframe_with_download("Refresh provenance", data["refresh_provenance"], "refresh_provenance.csv")
    dataframe_with_download("Provider registry", data["provider_registry"], "provider_registry.csv")
    dataframe_with_download("ML leaderboard", data["ml_leaderboard"], "ml_leaderboard.csv")
    with st.expander("Caveats", expanded=True):
        st.markdown(
            """
            - Missing provider data is surfaced through diagnostics instead of filled silently.
            - A high valuation score can still be a value trap if quality, leverage or freshness are weak.
            - ML outputs are overlays and should not override valuation assumptions without review.
            """
        )
