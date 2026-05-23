from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_DIR.parent
for candidate in [APP_DIR, PROJECT_ROOT]:
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import plotly.express as px
import streamlit as st

from support import configure_page, dataframe_with_download, load_smart_money_artifacts, sidebar_roots

try:
    from src.smart_money_engine import run_smart_money_engine
    from src.smart_money_engine.visualization import (
        cot_percentile_chart,
        government_exposure_treemap,
        insider_bar,
        score_ranking_chart,
        sector_heatmap,
        tic_flow_chart,
    )
except Exception:
    from smart_money_engine import run_smart_money_engine
    from smart_money_engine.visualization import (
        cot_percentile_chart,
        government_exposure_treemap,
        insider_bar,
        score_ranking_chart,
        sector_heatmap,
        tic_flow_chart,
    )


configure_page("Smart Money Gov Data")

roots = sidebar_roots()
smart_root = roots["workspace"] / "smart_money"
data = load_smart_money_artifacts(roots["workspace"])

st.title("Smart Money Government Data Engine")
st.caption("Official-source-first ownership, insider, activism, macro flow and government spending intelligence.")

with st.container(border=True):
    c1, c2, c3 = st.columns(3)
    max_files = c1.number_input("Max files per source", min_value=1, max_value=50, value=5, step=1)
    region = c2.selectbox("Region Lens", ["USA core", "EU / Italy proxy", "Global overview"])
    mode = c3.selectbox("Analyst Mode", ["Buy-side", "Macro", "Event-driven", "Government exposure"])
    if st.button("Run local Smart Money refresh", width="stretch"):
        outputs = run_smart_money_engine(roots["financial_db"], smart_root, max_files_per_source=int(max_files))
        st.success(f"Smart Money artifacts written to {smart_root}")
        st.json(outputs["manifest"])
        data = load_smart_money_artifacts(roots["workspace"])

scores = data["scores"]
events = data["events"]
coverage = data["coverage"]

cols = st.columns(4)
cols[0].metric("Ranked Issuers", len(scores))
cols[1].metric("Event Feed Rows", len(events))
cols[2].metric("Sources OK", int(coverage.get("status", []).astype(str).eq("OK").sum()) if not coverage.empty and "status" in coverage.columns else 0)
cols[3].metric("Lens", region)

st.markdown(
    """
    <div class="rp-note">
    This layer does not fabricate smart-money signals. If official files are missing locally,
    outputs stay schema-compliant and coverage tables show the gap.
    USA signals are more complete; EU/Italy is a federated proxy layer based on ESMA, ECB, TED and national registers.
    </div>
    """,
    unsafe_allow_html=True,
)

tabs = st.tabs([
    "Overview",
    "Smart Money Screener",
    "Issuer Deep Dive",
    "Manager Deep Dive",
    "Sector / Theme Monitor",
    "Macro & Flows",
    "Government Exposure",
    "Governance & Roadmap",
])

with tabs[0]:
    chart = score_ranking_chart(scores)
    if chart is not None:
        st.plotly_chart(chart, width="stretch")
    dataframe_with_download("Coverage", coverage, "smart_money_coverage.csv")
    dataframe_with_download("Source registry", data["source_registry"], "smart_money_source_registry.csv")

with tabs[1]:
    dataframe_with_download("Smart Money Screener", scores, "smart_money_scores.csv")
    if not scores.empty and "composite_institutional_interest_score" in scores.columns:
        st.plotly_chart(px.histogram(scores, x="composite_institutional_interest_score", color="coverage_note" if "coverage_note" in scores.columns else None, title="Composite score distribution", template="plotly_white"), width="stretch")
    with st.expander("Score formula", expanded=True):
        st.markdown(
            """
            CompositeInstitutionalInterestScore is the mean of available explainable components:
            13F ownership change, insider conviction, activist pressure, government demand tailwind,
            macro positioning support and cross-border flow support. Missing components reduce coverage,
            not silently imputed signal quality.
            """
        )

with tabs[2]:
    tickers = sorted(scores["ticker"].dropna().astype(str).unique().tolist()) if not scores.empty and "ticker" in scores.columns else []
    selected = st.selectbox("Issuer ticker", tickers) if tickers else ""
    if selected:
        st.dataframe(scores[scores["ticker"].astype(str).eq(selected)], width="stretch", hide_index=True)
        for key in ["holdings", "insiders", "beneficial_events", "government_spending"]:
            df = data[key]
            if not df.empty and "ticker" in df.columns:
                with st.expander(key.replace("_", " ").title(), expanded=False):
                    st.dataframe(df[df["ticker"].astype(str).eq(selected)], width="stretch", hide_index=True)
    else:
        st.info("No issuer-level official-source signals available yet.")

with tabs[3]:
    holdings = data["holdings"]
    managers = sorted(holdings["filer_name"].dropna().astype(str).unique().tolist()) if not holdings.empty and "filer_name" in holdings.columns else []
    manager = st.selectbox("Manager / filer", managers) if managers else ""
    if manager:
        view = holdings[holdings["filer_name"].astype(str).eq(manager)]
        dataframe_with_download("Manager holdings", view, "smart_money_manager_holdings.csv")
    else:
        st.info("Manager deep dive appears when SEC 13F holdings are available.")

with tabs[4]:
    dataframe_with_download("Sector monitor", data["sector_monitor"], "smart_money_sector_monitor.csv")
    chart = sector_heatmap(data["sector_monitor"])
    if chart is not None:
        st.plotly_chart(chart, width="stretch")

with tabs[5]:
    c1 = cot_percentile_chart(data["macro_positioning"])
    c2 = tic_flow_chart(data["capital_flows"])
    if c1 is not None:
        st.plotly_chart(c1, width="stretch")
    if c2 is not None:
        st.plotly_chart(c2, width="stretch")
    dataframe_with_download("CFTC COT positioning", data["macro_positioning"], "smart_money_cot.csv")
    dataframe_with_download("Treasury TIC flows", data["capital_flows"], "smart_money_tic.csv")

with tabs[6]:
    chart = government_exposure_treemap(data["government_spending"])
    if chart is not None:
        st.plotly_chart(chart, width="stretch")
    dataframe_with_download("Government spending exposure", data["government_spending"], "smart_money_government_spending.csv")

with tabs[7]:
    st.subheader("USA vs EU / Italy caveat")
    st.markdown(
        """
        **USA:** SEC, CFTC, Treasury and USAspending provide centralized official sources for ownership,
        insider, fund, macro positioning, cross-border flows and public spending.

        **EU / Italy:** coverage is federated. ESMA FIRDS helps instrument identity, ECB SDW gives macro context,
        TED supports procurement intelligence, while major holdings/insider disclosure depends on national regulators
        such as CONSOB, AMF, BaFin and CNMV. EU signals are marked as complete, partial, proxy or unavailable.
        """
    )
    dataframe_with_download("Government dataset matrix", data["government_dataset_matrix"], "smart_money_government_dataset_matrix.csv")
    dataframe_with_download("Open-source pattern matrix", data["open_source_pattern_matrix"], "smart_money_open_source_patterns.csv")
    dataframe_with_download("30-day MVP plan", data["mvp_30_day_plan"], "smart_money_mvp_30_day_plan.csv")
    dataframe_with_download("Dataset priority ranking", data["dataset_priority_ranking"], "smart_money_dataset_priority.csv")
    dataframe_with_download("Analyst quick wins", data["analyst_quick_wins"], "smart_money_quick_wins.csv")

with st.expander("Event Feed", expanded=False):
    dataframe_with_download("Event feed", events, "smart_money_event_feed.csv")
