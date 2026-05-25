from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from orchestration import JobStore, build_freshness_table, get_job_registry, run_job, validate_expected_artifacts
from orchestration.background import launch_job_process
from orchestration.freshness import freshness_badge
from orchestration.notebook_parameters import ensure_parameters_cell, has_parameters_cell
from orchestration.scheduler import scheduler_tick
from orchestration.scheduler_process import scheduler_status, start_scheduler_process, stop_scheduler_process
from support import configure_page, render_context_bar, render_footer, render_page_intro, sidebar_roots
from ui_ops import job_status_board, render_job_board, render_safe_log_preview

configure_page("Notebook Runner")

import pandas as pd
import streamlit as st


def render_parameter(spec):
    help_text = spec.description or None
    if spec.type == "bool":
        return st.checkbox(spec.name, value=bool(spec.default), help=help_text)
    if spec.type == "int":
        min_value = int(spec.minimum) if spec.minimum is not None else None
        max_value = int(spec.maximum) if spec.maximum is not None else None
        return st.number_input(spec.name, value=int(spec.default or 0), min_value=min_value, max_value=max_value, step=1, help=help_text)
    if spec.type == "float":
        return st.number_input(spec.name, value=float(spec.default or 0.0), min_value=spec.minimum, max_value=spec.maximum, help=help_text)
    if spec.type == "choice":
        choices = spec.choices or [spec.default]
        index = choices.index(spec.default) if spec.default in choices else 0
        return st.selectbox(spec.name, choices, index=index, help=help_text)
    if spec.type == "date":
        return st.text_input(spec.name, value=str(spec.default or ""), help=help_text)
    return st.text_input(spec.name, value=str(spec.default or ""), help=help_text)


roots = sidebar_roots()
store = JobStore()
registry = get_job_registry()
runs_df = pd.DataFrame([run.to_dict() for run in store.list_runs()])

st.title("Notebook Runner")
st.caption("Operational launcher for notebook-safe jobs and lightweight artifact refreshes.")
render_context_bar()
render_page_intro(
    "Advanced operations live here: data bootstrap, notebook execution, ML training and artifact refresh jobs.",
    "Daily research should start from Home, Screener, Valuation or Data Platform; use this page when a run is required.",
)

st.markdown(
    """
    <div class="rp-note">
    <b>Advanced operations:</b> this page is not required for normal desk browsing. It launches supported jobs,
    stores executed notebook copies/logs, and validates exported artifacts afterward.
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("Platform job board", expanded=True):
    st.caption("One operating model for notebook runs, lightweight refreshes, data sync and model artifacts.")
    render_job_board(job_status_board(registry, runs_df), key="notebook_runner_job_board", height=260)

job_options = {f"{job.label} ({job.job_id})": job_id for job_id, job in registry.items()}
selected_label = st.selectbox("Job", list(job_options))
job = registry[job_options[selected_label]]

cols = st.columns(4)
cols[0].metric("Runner", job.runner_type)
cols[1].metric("Enabled", "yes" if job.enabled else "no")
cols[2].metric("Timeout", f"{job.timeout_seconds}s")
cols[3].metric("Output Domain", job.output_domain)

duration_hint = {
    "module": "light · usually seconds",
    "data_platform": "light · inventory only",
    "price_refresh": "medium · depends on stale symbols",
    "research_data_bootstrap": "heavy · dry-run is seconds, execute depends on provider scope",
    "ml_training_lab": "medium/heavy · depends on panel size and model list",
    "papermill": "heavy · notebook execution",
    "nbclient": "heavy · fallback notebook execution",
}.get(job.runner_type, "unknown")

with st.container(border=True):
    h1, h2, h3 = st.columns(3)
    h1.metric("Duration Estimate", duration_hint)
    h2.metric("Produces", f"{len(job.expected_artifacts)} artifacts")
    h3.metric("Tags", ", ".join(job.tags[:3]) if job.tags else "n/a")
    st.write(job.description)
    if job.runner_type in {"papermill", "nbclient"}:
        st.warning("This can block resources while the notebook runs. Prefer background mode and out-of-hours execution for heavy notebooks.")

job_runs = runs_df[runs_df["job_id"].eq(job.job_id)].copy() if not runs_df.empty and "job_id" in runs_df.columns else pd.DataFrame()
with st.expander("Last 3 runs for selected job", expanded=not job_runs.empty):
    if job_runs.empty:
        st.info("No previous runs for this job.")
    else:
        show_cols = ["run_id", "status", "created_at", "started_at", "finished_at", "runner_type", "error_message"]
        st.dataframe(job_runs[[c for c in show_cols if c in job_runs.columns]].head(3), width="stretch", hide_index=True)

if not job.enabled:
    st.warning(job.disabled_reason or "This job is disabled.")

with st.expander("Advanced job contract", expanded=False):
    st.write(job.description)
    st.code(str(job.notebook_path or "module-only job"))
    if job.notebook_path:
        has_params = has_parameters_cell(job.notebook_path)
        st.write(f"Papermill parameters cell: `{'present' if has_params else 'missing'}`")
        if not has_params:
            c1, c2 = st.columns(2)
            if c1.button("Dry-check parameters cell injection", width="stretch"):
                st.json(ensure_parameters_cell(job.notebook_path, [spec.name for spec in job.parameters], dry_run=True))
            if c2.button("Insert parameters cell with backup", width="stretch"):
                st.json(ensure_parameters_cell(job.notebook_path, [spec.name for spec in job.parameters], dry_run=False))
    artifact_root = roots.get(job.output_domain, roots["workspace"])
    artifact_rows = validate_expected_artifacts(job, artifact_root)
    artifact_df = pd.DataFrame(artifact_rows)
    if not artifact_df.empty:
        st.dataframe(artifact_df, width="stretch", hide_index=True)
        stale_or_missing = artifact_df[artifact_df["status"].astype(str).isin(["STALE", "MISSING_REQUIRED"])]
        if not stale_or_missing.empty:
            st.warning("Some expected artifacts are stale or missing. Confirm Data Platform freshness before launching a heavy run.")

with st.form("run_job_form"):
    st.subheader("Run Parameters")
    st.caption("Defaults are chosen for safe desk usage. Heavy notebook jobs should usually run in background mode.")
    values = {spec.name: render_parameter(spec) for spec in job.parameters}
    async_mode = st.checkbox("Run in background process", value=True, help="Keeps Streamlit responsive while notebook/module work runs in a detached worker.")
    submitted = st.form_submit_button("Run Job", disabled=not job.enabled, width="stretch")

if submitted:
    if async_mode:
        with st.spinner("Starting background job..."):
            run = launch_job_process(job.job_id, values, roots=roots, store=store)
        st.success(f"Job started: Run ID={run.run_id}")
        if run.log_path:
            st.caption(f"Log: {run.log_path}")
        st.info("Monitor status in the Latest Run panel or Run Hi-Freq Engine. The app remains usable while the worker runs.")
    else:
        with st.status("Running job...", expanded=True) as status:
            st.write("Creating run metadata")
            run = run_job(job.job_id, values, roots=roots, store=store)
            st.write(f"Run ID: `{run.run_id}`")
            st.write(f"Status: `{run.status.value}`")
            if run.error_message:
                st.error(run.error_message)
            status.update(label=f"Job finished: {run.status.value}", state="complete" if run.status.value == "SUCCESS" else "error")

latest = store.list_runs()
latest_for_job = next((run for run in latest if run.job_id == job.job_id), None)
if latest_for_job:
    st.subheader("Latest Run")
    c1, c2, c3 = st.columns(3)
    c1.metric("Run ID", latest_for_job.run_id)
    c2.metric("Status", latest_for_job.status.value)
    c3.metric("Finished", latest_for_job.finished_at or "n/a")
    if latest_for_job.log_path and Path(latest_for_job.log_path).exists():
        with st.expander("Run log", expanded=False):
            render_safe_log_preview(Path(latest_for_job.log_path).read_text(encoding="utf-8", errors="replace"))
    if latest_for_job.output_notebook_path:
        path = Path(latest_for_job.output_notebook_path)
        st.write(f"Executed notebook/output: `{path}`")
        if path.exists():
            st.download_button("Download executed notebook/log marker", path.read_bytes(), file_name=path.name)
    if latest_for_job.artifacts_detected:
        st.subheader("Artifacts Detected")
        st.dataframe(pd.DataFrame(latest_for_job.artifacts_detected), width="stretch", hide_index=True)
    if latest_for_job.status.value in {"PENDING", "RUNNING"}:
        import time

        if st.checkbox("Auto-refresh running status every 5 seconds", value=True):
            time.sleep(5)
            st.rerun()

with st.expander("Keep artifacts fresh", expanded=False):
    st.write("Runs a single scheduler tick. By default it only launches jobs whose required artifacts are missing or stale.")
    schedule_jobs = st.multiselect("Jobs to check", list(registry.keys()), default=["screener_refresh"])
    if st.button("Run scheduler tick now", width="stretch"):
        st.json(scheduler_tick(roots, schedule_jobs, store=store))
    scheduler_state = scheduler_status()
    st.write(f"Local scheduler: `{'RUNNING' if scheduler_state.get('alive') else 'STOPPED'}`")
    interval = st.number_input("Scheduler interval seconds", min_value=60, max_value=86400, value=900, step=60)
    off1, off2 = st.columns(2)
    off_start = off1.number_input("Notebook off-hour start", min_value=0, max_value=23, value=20, step=1)
    off_end = off2.number_input("Notebook off-hour end", min_value=0, max_value=23, value=7, step=1)
    start_col, stop_col = st.columns(2)
    if start_col.button("Start scheduler process", width="stretch"):
        st.json(start_scheduler_process(
            roots,
            schedule_jobs,
            interval_seconds=int(interval),
            notebook_offhour_start=int(off_start),
            notebook_offhour_end=int(off_end),
        ))
    if stop_col.button("Stop scheduler process", width="stretch"):
        st.json(stop_scheduler_process())
    st.code("python research_platform_app/scheduler.py --once --jobs screener_refresh")
    st.code("python research_platform_app/scheduler.py --interval-seconds 900 --jobs screener_refresh")

with st.expander("Freshness Monitor", expanded=False):
    freshness = build_freshness_table(roots)
    if freshness.empty:
        st.info("No artifact contracts available.")
    else:
        summary = freshness["status"].value_counts().reset_index()
        summary.columns = ["status", "count"]
        st.dataframe(summary, width="stretch", hide_index=True)
        for status_value in summary["status"].tolist():
            st.markdown(f"{freshness_badge(status_value)}", unsafe_allow_html=True)
        st.dataframe(freshness, width="stretch", hide_index=True)

render_footer()
