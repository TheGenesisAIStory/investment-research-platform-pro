from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import pandas as pd
import streamlit as st

from orchestration import JobStore, build_freshness_table, get_job_registry, validate_expected_artifacts
from orchestration.background import launch_job_process
from support import add_artifact_usage, configure_page, find_artifacts, render_context_bar, render_footer, render_page_header, render_page_intro, safe_page_link, sidebar_roots
from ui_ops import job_status_board, render_job_board


configure_page("Export Center")

roots = sidebar_roots()
artifacts = add_artifact_usage(find_artifacts(roots))
freshness = build_freshness_table(roots)
store = JobStore()
registry = get_job_registry()
runs_df = pd.DataFrame([run.to_dict() for run in store.list_runs()])

render_page_header(
    "Export Center",
    "Contract-level visibility into CSV, JSON and HTML outputs consumed by the app, notebooks and future APIs.",
    "⇧",
    module="PLATFORM OPS",
    status="READY",
)
render_context_bar()
render_page_intro(
    "Inspect exported artifacts, freshness contracts and job outputs from one contract-level operations page.",
    "Filter artifacts, validate contracts, or retry the producing job with safe defaults.",
)

if artifacts.empty:
    st.warning("No exported artifacts found yet. Use the job board below to generate the first contracts.")

view = artifacts.copy()
if not view.empty and not freshness.empty and "path" in freshness.columns:
    status_cols = freshness[["path", "status", "modified", "age_hours", "required"]].rename(columns={"modified": "contract_modified"})
    view = view.merge(status_cols, on="path", how="left")
    view["status"] = view["status"].fillna("UNCONTRACTED")
elif not view.empty:
    view["status"] = "UNCONTRACTED"

namespace_options = sorted(view["domain"].dropna().unique()) if not view.empty and "domain" in view.columns else []
type_options = sorted(view["type"].dropna().unique()) if not view.empty and "type" in view.columns else []
status_options = sorted(view["status"].dropna().unique()) if not view.empty and "status" in view.columns else []

c1, c2, c3 = st.columns(3)
domains = c1.multiselect("Namespace / Job Domain", namespace_options, default=namespace_options)
file_types = c2.multiselect("Type", type_options, default=type_options)
statuses = c3.multiselect("Status", status_options, default=status_options)
query = st.text_input("Search name/path", "")

if view.empty:
    filtered = view
else:
    filtered = view[view["domain"].isin(domains) & view["type"].isin(file_types) & view["status"].isin(statuses)].copy()
if query:
    q = query.lower()
    filtered = filtered[filtered["name"].str.lower().str.contains(q, na=False) | filtered["path"].str.lower().str.contains(q, na=False)]

cols = st.columns(4)
cols[0].metric("Files", len(filtered))
cols[1].metric("CSV", int(filtered["type"].eq("csv").sum()) if "type" in filtered.columns else 0)
cols[2].metric("JSON", int(filtered["type"].eq("json").sum()) if "type" in filtered.columns else 0)
cols[3].metric("HTML", int(filtered["type"].eq("html").sum()) if "type" in filtered.columns else 0)

if filtered.empty:
    is_data_platform = is_smart_money = is_ml_lab = pd.Series(dtype=bool)
else:
    is_data_platform = filtered["name"].str.contains("DataPlatform_|data_platform", case=False, na=False) | filtered["path"].str.contains("api_contracts", case=False, na=False)
    is_smart_money = filtered["name"].str.contains("SmartMoney|smart_money", case=False, na=False) | filtered["path"].str.contains("smart_money", case=False, na=False)
    is_ml_lab = filtered["name"].str.contains("MLStockLab|ml_stock_lab", case=False, na=False) | filtered["path"].str.contains("ml_stock_lab", case=False, na=False)
notebook_artifacts = filtered[~is_data_platform & ~is_smart_money & ~is_ml_lab].copy()
data_platform_artifacts = filtered[is_data_platform].copy()
smart_money_artifacts = filtered[is_smart_money].copy()
ml_lab_artifacts = filtered[is_ml_lab].copy()

tab_jobs, tab_notebook, tab_data_platform, tab_smart_money, tab_ml_lab, tab_contract = st.tabs(["Job Contracts", "Notebook Artifacts", "Data Platform Artifacts", "Smart Money Artifacts", "ML Lab Artifacts", "Contract Freshness"])

display_cols = ["domain", "type", "name", "status", "modified", "age_hours", "used_by", "path", "size_kb"]

with tab_jobs:
    board = job_status_board(registry, runs_df)
    render_job_board(board, key="export_center_job_board", height=300)
    selected_job_id = st.selectbox("Job detail / retry", sorted(registry.keys()), key="export_job_detail")
    selected_job = registry[selected_job_id]
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Runner", selected_job.runner_type)
    d2.metric("Enabled", "yes" if selected_job.enabled else "no")
    d3.metric("Output", selected_job.output_domain)
    d4.metric("Expected artifacts", len(selected_job.expected_artifacts))
    st.write(selected_job.description)
    artifact_root = roots.get(selected_job.output_domain, roots["workspace"])
    artifact_df = pd.DataFrame(validate_expected_artifacts(selected_job, artifact_root))
    if artifact_df.empty:
        st.info("This job has no artifact contract rows.")
    else:
        st.dataframe(artifact_df, width="stretch", hide_index=True)
    if st.button("Run / Retry Job With Defaults", width="stretch", disabled=not selected_job.enabled):
        defaults = {spec.name: spec.default for spec in selected_job.parameters}
        with st.spinner("Starting background job..."):
            run = launch_job_process(selected_job_id, defaults, roots=roots, store=store)
        st.success(f"Job started: Run ID={run.run_id}")

with tab_notebook:
    if notebook_artifacts.empty:
        st.info("No notebook artifacts match the selected filters.")
    else:
        st.dataframe(notebook_artifacts[[c for c in display_cols if c in notebook_artifacts.columns]], width="stretch", hide_index=True)

with tab_data_platform:
    if data_platform_artifacts.empty:
        st.info("No Data Platform artifacts match the selected filters.")
    else:
        st.dataframe(data_platform_artifacts[[c for c in display_cols if c in data_platform_artifacts.columns]], width="stretch", hide_index=True)

with tab_smart_money:
    if smart_money_artifacts.empty:
        st.info("No Smart Money artifacts match the selected filters.")
    else:
        st.dataframe(smart_money_artifacts[[c for c in display_cols if c in smart_money_artifacts.columns]], width="stretch", hide_index=True)
    safe_page_link("pages/1_📡_Smart_Money_Macro.py", "Open Smart Money / Macro")

with tab_ml_lab:
    if ml_lab_artifacts.empty:
        st.info("No ML Lab artifacts match the selected filters.")
    else:
        st.dataframe(ml_lab_artifacts[[c for c in display_cols if c in ml_lab_artifacts.columns]], width="stretch", hide_index=True)
    safe_page_link("pages/9_ML_Stock_Lab.py", "Open ML Stock Lab")

with tab_contract:
    if freshness.empty:
        st.info("No artifact contracts configured.")
    else:
        st.dataframe(freshness, width="stretch", hide_index=True)

selected_path = st.selectbox("Select artifact for download", filtered["path"].tolist() if not filtered.empty and "path" in filtered.columns else [])
if selected_path:
    path = Path(selected_path)
    if path.exists():
        mime = "text/csv" if path.suffix.lower() == ".csv" else "application/json" if path.suffix.lower() == ".json" else "text/html"
        st.download_button("Download selected artifact", path.read_bytes(), file_name=path.name, mime=mime, width="content")
        if path.suffix.lower() == ".html":
            st.link_button("Open HTML artifact path", f"file://{path}")

with st.expander("Artifact contract", expanded=True):
    st.markdown(
        """
        - **Notebook Artifacts** are produced by valuation and portfolio notebooks or their lightweight module refreshes.
        - **Data Platform Artifacts** describe Database Finanziario inventory, provider state and API-ready contracts.
        - **Smart Money Artifacts** are official-source overlays for valuation, portfolio and screener workflows.
        - **ML Lab Artifacts** are fair-value, mispricing, z-score and quintile diagnostics generated by `ml_stock_lab`.
        - `OK`, `STALE` and `MISSING_*` are contract statuses, not investment conclusions.
        """
    )

render_footer()
