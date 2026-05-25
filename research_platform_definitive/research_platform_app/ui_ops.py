"""Shared Streamlit UI helpers for platform operations.

These helpers keep job/run/status microcopy consistent across Export Center,
Notebook Runner, Hi-Freq Engine and data consumer pages.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def latest_runs_by_job(runs_df: pd.DataFrame) -> pd.DataFrame:
    if runs_df.empty or "job_id" not in runs_df.columns:
        return pd.DataFrame()
    view = runs_df.copy()
    sort_col = "created_at" if "created_at" in view.columns else view.columns[0]
    return view.sort_values(sort_col, ascending=False).drop_duplicates("job_id")


def job_status_board(registry: dict[str, Any], runs_df: pd.DataFrame) -> pd.DataFrame:
    latest = latest_runs_by_job(runs_df)
    latest_lookup = latest.set_index("job_id").to_dict("index") if not latest.empty and "job_id" in latest.columns else {}
    rows: list[dict[str, Any]] = []
    for job_id, job in registry.items():
        run = latest_lookup.get(job_id, {})
        rows.append(
            {
                "job_id": job_id,
                "label": job.label,
                "area": ", ".join(job.tags[:3]) if job.tags else job.output_domain,
                "runner": job.runner_type,
                "enabled": bool(job.enabled),
                "last_status": run.get("status", "NEVER_RUN"),
                "last_finished": run.get("finished_at") or run.get("created_at") or "",
                "output_domain": job.output_domain,
                "expected_artifacts": len(job.expected_artifacts),
                "description": job.description,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    status_order = {"RUNNING": 0, "PENDING": 1, "FAILED": 2, "NEVER_RUN": 3, "SUCCESS": 4}
    frame["_status_order"] = frame["last_status"].map(status_order).fillna(5)
    return frame.sort_values(["_status_order", "area", "label"]).drop(columns="_status_order").reset_index(drop=True)


def render_safe_log_preview(log_text: str, *, limit: int = 12000) -> None:
    import streamlit as st

    text = str(log_text or "")
    if "Traceback (most recent call last)" in text:
        st.warning("Detailed Python traceback is hidden in the UI. Download the full log for engineering debug.")
        concise = [
            line
            for line in text.splitlines()
            if any(token in line.upper() for token in ["ERROR", "WARNING", "WARN", "FAILED", "EXCEPTION"])
        ][-30:]
        st.code("\n".join(concise) or "Failure detected. Full traceback available in the downloadable log.")
    else:
        st.code(text[-limit:])


def render_missing_data_cta(
    label: str,
    *,
    job_id: str,
    output_path: Path,
    cli_hint: str,
    page_link: str = "pages/6_🧪_Notebook_Runner.py",
) -> None:
    import streamlit as st

    with st.container(border=True):
        st.warning(f"{label} artifacts are unavailable or stale.")
        st.write("Use the platform job runner first; the page will read the exported artifacts after refresh.")
        c1, c2 = st.columns([1, 2])
        with c1:
            try:
                st.page_link(page_link, label=f"Open job: {job_id}")
            except Exception:
                st.caption(f"Open Notebook Runner and select `{job_id}`.")
        with c2:
            st.code(f"# Expected output\n{output_path}\n\n# CLI fallback\n{cli_hint}", language="bash")


def render_job_board(frame: pd.DataFrame, *, key: str, height: int = 320) -> None:
    import streamlit as st

    if frame.empty:
        st.info("No registered jobs are available in this session.")
        return
    display_cols = [
        "job_id",
        "label",
        "area",
        "runner",
        "enabled",
        "last_status",
        "last_finished",
        "output_domain",
        "expected_artifacts",
    ]
    st.dataframe(
        frame[[col for col in display_cols if col in frame.columns]],
        width="stretch",
        height=height,
        hide_index=True,
        key=key,
    )
