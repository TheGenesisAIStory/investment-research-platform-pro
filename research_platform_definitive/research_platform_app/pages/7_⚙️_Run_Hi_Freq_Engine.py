from __future__ import annotations

import json
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from orchestration import JobStore, JobStatus, get_job_registry
from orchestration.background import launch_job_process
from support import configure_page, render_context_bar, render_footer, render_page_intro, safe_page_link, sidebar_roots
from ui_ops import job_status_board, render_job_board, render_safe_log_preview

configure_page("Run Hi-Freq Engine")

import pandas as pd
import streamlit as st


roots = sidebar_roots()
store = JobStore()
registry = get_job_registry()
runs = store.list_runs()


st.title("Run Hi-Freq Engine")
st.caption("Operational run monitor for notebook jobs, lightweight refreshes and future high-frequency research tasks.")
render_context_bar()
render_page_intro(
    "Monitor active and historical runs, retry selected jobs and inspect logs without exposing raw stacktraces in the main UI.",
    "Use Refresh History first; retry only the job whose status or artifact freshness requires attention.",
)
st.markdown(
    """
    <div class="rp-note">
    This page keeps the existing run-history functionality and exposes it as the operational control room.
    High-frequency or microstructure jobs can be added to the same registry without changing the UX contract.
    </div>
    """,
    unsafe_allow_html=True,
)

top_left, top_right = st.columns(2)
if top_left.button("Refresh history", width="stretch"):
    st.rerun()

if not runs:
    st.info("No runs have been recorded yet. Launch a job from the Run Notebooks page.")
    render_footer()
    st.stop()

df = pd.DataFrame([run.to_dict() for run in runs])
created = pd.to_datetime(df.get("created_at"), errors="coerce", utc=True)
last7 = df[created >= (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=7))] if created.notna().any() else df.iloc[0:0]
success_rate = f"{round(100 * df['status'].astype(str).eq('SUCCESS').sum() / len(df), 1)}%" if len(df) else "n/a"
top_job = df["job_id"].value_counts().index[0] if "job_id" in df.columns and df["job_id"].notna().any() else "n/a"
latest_by_job = df.sort_values("created_at", ascending=False).drop_duplicates("job_id") if "job_id" in df.columns else pd.DataFrame()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Runs · 7d", len(last7))
m2.metric("Success Rate", success_rate)
m3.metric("Most Run Job", top_job)
m4.metric("Core Jobs", len(latest_by_job))

with st.container(border=True):
    st.markdown("**Core job monitor**")
    render_job_board(job_status_board(registry, df), key="hifreq_job_board", height=240)
    job_to_run = st.selectbox("Run / retry job", sorted(registry.keys()), help="Launches the selected job in a background worker with default parameters.")
    job_spec = registry[job_to_run]
    if st.button("Run / Retry Selected Job", width="stretch", disabled=not job_spec.enabled):
        defaults = {spec.name: spec.default for spec in job_spec.parameters}
        with st.spinner("Starting background job..."):
            new_run = launch_job_process(job_to_run, defaults, roots=roots, store=store)
        st.success(f"Job started: Run ID={new_run.run_id}")

left, mid, right = st.columns(3)
status_filter = left.multiselect("Status", [status.value for status in JobStatus], default=[])
job_filter = mid.multiselect("Job", sorted(df["job_id"].dropna().unique().tolist()), default=[])
date_window = right.selectbox("Time Window", ["All", "Last 24h", "Last 7d", "Last 30d"])

view = df.copy()
if status_filter:
    view = view[view["status"].isin(status_filter)]
if job_filter:
    view = view[view["job_id"].isin(job_filter)]
if date_window != "All" and "created_at" in view.columns:
    hours = {"Last 24h": 24, "Last 7d": 24 * 7, "Last 30d": 24 * 30}[date_window]
    created_view = pd.to_datetime(view["created_at"], errors="coerce", utc=True)
    view = view[created_view >= (pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=hours))]

display_cols = [
    "run_id", "job_id", "status", "created_at", "started_at", "finished_at",
    "runner_type", "notebook_path", "output_notebook_path", "log_path", "error_message",
]
st.dataframe(view[[c for c in display_cols if c in view.columns]], width="stretch", hide_index=True)

if view.empty:
    st.info("No runs match the selected filters.")
    render_footer()
    st.stop()

with st.expander("Latest run per core job", expanded=False):
    if latest_by_job.empty:
        st.info("No core job run summary available.")
    else:
        cols = ["job_id", "status", "created_at", "finished_at", "runner_type", "error_message"]
        st.dataframe(latest_by_job[[c for c in cols if c in latest_by_job.columns]], width="stretch", hide_index=True)

if view["status"].isin(["PENDING", "RUNNING"]).any():
    import time

    if top_right.checkbox("Auto-refresh active runs every 5 seconds", value=True):
        time.sleep(5)
        st.rerun()

selected_run_id = st.selectbox("Run details", view["run_id"].tolist())
selected = store.get_run(selected_run_id)

if selected:
    st.subheader(selected.run_id)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Job", selected.job_id)
    c2.metric("Status", selected.status.value)
    c3.metric("Runner", selected.runner_type)
    c4.metric("Finished", selected.finished_at or "n/a")

    with st.expander("Parameters", expanded=True):
        st.json(selected.parameters)

    with st.expander("Artifacts", expanded=True):
        if selected.artifacts_detected:
            artifact_df = pd.DataFrame(selected.artifacts_detected)
            st.dataframe(artifact_df, width="stretch", hide_index=True)
            if "path" in artifact_df.columns:
                st.caption("Related artifacts are also visible from Artifacts / Exports.")
                safe_page_link("pages/5_📤_Export_Center.py", "Open Artifacts / Exports")
        else:
            st.info("No artifact validation was recorded for this run.")

    with st.expander("Log", expanded=False):
        path = Path(selected.log_path)
        if path.exists():
            log_text = path.read_text(encoding="utf-8", errors="replace")[-16000:]
            l1, l2, l3 = st.columns(3)
            l1.metric("ERROR", log_text.upper().count("ERROR"))
            l2.metric("WARN", log_text.upper().count("WARN"))
            l3.metric("INFO", log_text.upper().count("INFO"))
            render_safe_log_preview(log_text)
            st.download_button("Download log", path.read_bytes(), file_name=path.name)
        else:
            st.warning("Log file is missing.")

    if selected.output_notebook_path:
        output_path = Path(selected.output_notebook_path)
        if output_path.exists():
            st.download_button("Download executed notebook/output", output_path.read_bytes(), file_name=output_path.name)

    rerun_col, json_col = st.columns(2)
    if rerun_col.button("Rerun With Same Parameters", width="stretch"):
        job = registry.get(selected.job_id)
        if job is None or not job.enabled:
            st.warning("The original job is no longer enabled.")
        else:
            new_run = launch_job_process(selected.job_id, selected.parameters, roots=roots, store=store)
            st.success(f"Job started: Run ID={new_run.run_id}")
    json_col.download_button(
        "Download run metadata JSON",
        json.dumps(selected.to_dict(), indent=2, default=str),
        file_name=f"{selected.run_id}.json",
        mime="application/json",
        width="stretch",
    )

render_footer()
