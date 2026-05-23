"""Freshness-driven local scheduler."""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

from .background import launch_job_process
from .freshness import build_freshness_table
from .job_store import JobStore
from .registry import get_job_registry
from .status import JobStatus


DEFAULT_SCHEDULED_JOBS = ["data_platform_status_refresh", "screener_refresh"]


def _has_active_run(store: JobStore, job_id: str) -> bool:
    return any(run.job_id == job_id and run.status in {JobStatus.PENDING, JobStatus.RUNNING} for run in store.list_runs())


def jobs_needing_refresh(roots: dict[str, Path], job_ids: list[str] | None = None) -> list[str]:
    registry = get_job_registry()
    candidates = job_ids or DEFAULT_SCHEDULED_JOBS
    freshness = build_freshness_table(roots)
    if freshness.empty:
        return []
    needed: list[str] = []
    for job_id in candidates:
        job = registry.get(job_id)
        if job is None or not job.enabled:
            continue
        rows = freshness[freshness["job_id"].eq(job_id)]
        if rows.empty:
            continue
        bad = rows["status"].isin(["STALE", "MISSING_REQUIRED"]).any()
        if bad:
            needed.append(job_id)
    return needed


def _inside_off_hours(start_hour: int, end_hour: int) -> bool:
    hour = datetime.now().hour
    if start_hour == end_hour:
        return True
    if start_hour < end_hour:
        return start_hour <= hour < end_hour
    return hour >= start_hour or hour < end_hour


def scheduler_tick(
    roots: dict[str, Path],
    job_ids: list[str] | None = None,
    store: JobStore | None = None,
    notebook_offhour_start: int = 20,
    notebook_offhour_end: int = 7,
) -> list[dict]:
    store = store or JobStore()
    launched: list[dict] = []
    registry = get_job_registry()
    for job_id in jobs_needing_refresh(roots, job_ids):
        job = registry.get(job_id)
        if job and job.runner_type in {"papermill", "nbclient"} and not _inside_off_hours(notebook_offhour_start, notebook_offhour_end):
            launched.append({"job_id": job_id, "status": "SKIP", "detail": f"outside notebook off-hour window {notebook_offhour_start}:00-{notebook_offhour_end}:00"})
            continue
        if _has_active_run(store, job_id):
            launched.append({"job_id": job_id, "status": "SKIP", "detail": "already running"})
            continue
        run = launch_job_process(job_id, {}, roots, store=store)
        launched.append({"job_id": job_id, "status": "LAUNCHED", "run_id": run.run_id})
    return launched


def scheduler_loop(
    roots: dict[str, Path],
    interval_seconds: int = 900,
    job_ids: list[str] | None = None,
    notebook_offhour_start: int = 20,
    notebook_offhour_end: int = 7,
) -> None:
    while True:
        scheduler_tick(roots, job_ids, notebook_offhour_start=notebook_offhour_start, notebook_offhour_end=notebook_offhour_end)
        time.sleep(interval_seconds)
