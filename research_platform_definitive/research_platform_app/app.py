from __future__ import annotations

import pandas as pd
import streamlit as st

from operations import generate_all_artifacts, generate_company_artifacts, generate_portfolio_artifacts
from orchestration import build_freshness_table
from orchestration.scheduler import scheduler_tick
from orchestration.scheduler_process import scheduler_status, start_scheduler_process, stop_scheduler_process
from support import (
    add_artifact_usage,
    configure_page,
    find_artifacts,
    friendly_domain,
    latest_modified_label,
    run_history_frame,
    run_metrics,
    load_smart_money_artifacts,
    load_ml_stock_lab_artifacts,
    sidebar_roots,
    status_counts_label,
)


configure_page("Overview")

roots = sidebar_roots()
artifacts = add_artifact_usage(find_artifacts(roots))
freshness = build_freshness_table(roots)
runs = run_history_frame()
metrics = run_metrics(runs)
smart_money = load_smart_money_artifacts(roots["workspace"])
ml_lab = load_ml_stock_lab_artifacts(roots["workspace"])

st.title("Research Platform · Overview")
st.caption("Buy-side research workstation over notebook-generated valuation, portfolio, data platform and orchestration artifacts.")

st.markdown(
    """
    <div class="rp-note">
    <b>Operating model:</b> notebooks remain the analytical engine, local modules own reusable computation,
    Database Finanziario is the Drive-first data layer, and Streamlit is the operational research console.
    </div>
    """,
    unsafe_allow_html=True,
)

cards = st.columns(4)
cards[0].metric("Runs · 7d", metrics["last7"])
cards[1].metric("Success Rate", metrics["success_rate"])
cards[2].metric("Artifact Contracts", status_counts_label(freshness))
cards[3].metric("ML Lab Signals", len(ml_lab["signals"]))

st.subheader("Artifact Roots")
root_cols = st.columns(4)
for col, domain in zip(root_cols, ["company", "portfolio", "workspace", "financial_db"]):
    root = roots[domain]
    domain_rows = artifacts[artifacts["domain"].eq(domain)] if not artifacts.empty and "domain" in artifacts.columns else pd.DataFrame()
    col.markdown(f"**{friendly_domain(domain)}**")
    col.metric("Files", len(domain_rows), latest_modified_label(domain_rows if domain != "financial_db" else root))
    with col.expander("Path", expanded=False):
        st.code(str(root))

nav_cols = st.columns(4)
nav_cols[0].page_link("pages/1_Valuation_Research.py", label="Valuation Research")
nav_cols[1].page_link("pages/2_Portfolio_Research.py", label="Portfolio Research")
nav_cols[2].page_link("pages/4_Artifacts_Exports.py", label="Artifacts / Exports")
nav_cols[3].page_link("pages/7_Data_Platform.py", label="Data Platform")
st.page_link("pages/11_Data_API_Control_Center.py", label="Open Data/API Control Center")
st.page_link("pages/8_Smart_Money_Gov_Data.py", label="Open Smart Money Government Data Engine")
st.page_link("pages/9_ML_Stock_Lab.py", label="Open ML Stock Lab")
st.page_link("pages/12_Banking_Data_Lab.py", label="Open Banking Data Lab")

with st.container(border=True):
    st.markdown("**Smart Money Integration**")
    sm1, sm2, sm3, sm4 = st.columns(4)
    sm1.metric("Issuer Scores", len(smart_money["scores"]))
    sm2.metric("Events", len(smart_money["events"]))
    sm3.metric("Coverage Rows", len(smart_money["coverage"]))
    sm4.metric("Source Registry", len(smart_money["source_registry"]))

with st.container(border=True):
    st.markdown("**ML Stock Lab Integration**")
    ml1, ml2, ml3, ml4 = st.columns(4)
    ml1.metric("Panel Rows", len(ml_lab["panel"]))
    ml2.metric("Signals", len(ml_lab["signals"]))
    ml3.metric("Quintile Rows", len(ml_lab["quintile_returns"]))
    ml4.metric("Metrics", len(ml_lab["metrics"]))

with st.expander("Generate missing lightweight artifacts", expanded=False):
    st.markdown(
        """
        These actions rebuild low-cost screener/selection/platform layers from existing notebook exports.
        They do not execute full notebooks or trigger heavy provider refreshes.
        """
    )
    c1, c2, c3 = st.columns(3)
    if c1.button("Generate Company Layer", width="stretch"):
        result = generate_company_artifacts(roots["company"])
        (st.success if result["ok"] else st.warning)(result["message"])
        if result.get("details"):
            st.json(result["details"])
    if c2.button("Generate Portfolio Selection", width="stretch"):
        result = generate_portfolio_artifacts(roots["portfolio"], roots["workspace"])
        (st.success if result["ok"] else st.warning)(result["message"])
        if result.get("details"):
            st.json(result["details"])
    if c3.button("Generate All Lightweight Layers", width="stretch"):
        results = generate_all_artifacts(roots)
        for name, result in results.items():
            (st.success if result["ok"] else st.warning)(f"{name.title()}: {result['message']}")
            if result.get("details"):
                st.json(result["details"])

with st.expander("Keep notebooks and artifacts updated", expanded=False):
    st.markdown(
        """
        Safe scheduler ticks refresh Data Platform status and lightweight screener/selection artifacts when
        required outputs are missing or stale. Full notebook jobs should run from Run Notebooks, preferably out of hours.
        """
    )
    if st.button("Run safe freshness tick now", width="stretch"):
        st.json(scheduler_tick(roots, ["data_platform_status_refresh", "screener_refresh"]))
    status = scheduler_status()
    st.write(f"Scheduler status: `{'RUNNING' if status.get('alive') else 'STOPPED'}`")
    scol1, scol2 = st.columns(2)
    if scol1.button("Start safe local scheduler", width="stretch"):
        st.json(start_scheduler_process(roots, ["data_platform_status_refresh", "screener_refresh"], interval_seconds=900))
    if scol2.button("Stop local scheduler", width="stretch"):
        st.json(stop_scheduler_process())
    st.code("python research_platform_app/scheduler.py --interval-seconds 900 --jobs data_platform_status_refresh screener_refresh")

tab_artifacts, tab_freshness, tab_runs = st.tabs(["Artifact Availability", "Freshness", "Recent Runs"])

with tab_artifacts:
    if artifacts.empty:
        st.info("No exported artifacts found yet. Run the company valuation and/or portfolio notebooks, then refresh this app.")
    else:
        st.dataframe(artifacts.head(200), width="stretch", hide_index=True)

with tab_freshness:
    if freshness.empty:
        st.info("No artifact contracts found.")
    else:
        status_counts = freshness["status"].value_counts().to_dict()
        fcols = st.columns(4)
        for col, status_name in zip(fcols, ["OK", "STALE", "MISSING_REQUIRED", "MISSING_OPTIONAL"]):
            col.metric(status_name, status_counts.get(status_name, 0))
        st.dataframe(freshness, width="stretch", hide_index=True)

with tab_runs:
    if runs.empty:
        st.info("No run history yet. Launch a job from Run Notebooks.")
    else:
        cols = ["run_id", "job_id", "status", "created_at", "finished_at", "runner_type", "error_message"]
        st.dataframe(runs[[c for c in cols if c in runs.columns]].head(20), width="stretch", hide_index=True)

with st.expander("How to use this app", expanded=True):
    st.markdown(
        """
        1. Use **Data Platform** to confirm Drive data, provider health and freshness.
        2. Use **Run Notebooks** for controlled notebook or module refresh jobs.
        3. Use **Valuation**, **Portfolio** and **Screeners** to inspect the latest generated artifacts.
        4. Use **Artifacts / Exports** as the contract view for future API and dashboard evolution.
        """
    )
