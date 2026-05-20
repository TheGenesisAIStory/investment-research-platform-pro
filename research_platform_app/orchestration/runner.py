"""Unified job runner interface."""

from __future__ import annotations

import traceback
import os
from pathlib import Path

try:
    from research_platform_app.operations import generate_all_artifacts
    from research_platform_app.support import default_roots
except Exception:
    from operations import generate_all_artifacts
    from support import default_roots
try:
    from src.research_platform_core.data_platform import refresh_europe_stoxx_prices_incremental, write_data_platform_status
except Exception:
    from research_platform_core.data_platform import refresh_europe_stoxx_prices_incremental, write_data_platform_status

from .artifact_contracts import validate_expected_artifacts
from .job_store import JobStore
from .models import JobRun, NotebookJob
from .nbclient_runner import run_with_nbclient
from .papermill_runner import run_with_papermill
from .registry import get_job
from .status import JobStatus
from .utils import RUNS_ROOT, append_log, ensure_dir, new_run_id, utc_now


def _output_notebook_path(job: NotebookJob, run_id: str) -> Path:
    if not job.output_notebook_template:
        return RUNS_ROOT / run_id / f"{job.job_id}.module-run.txt"
    return Path(str(job.output_notebook_template).format(run_id=run_id))


def _artifact_root(job: NotebookJob, roots: dict[str, Path]) -> Path:
    return roots.get(job.output_domain, roots.get("workspace", Path.cwd()))


def create_job_run(
    job_id: str,
    parameters: dict | None = None,
    store: JobStore | None = None,
) -> JobRun:
    store = store or JobStore()
    job = get_job(job_id)
    if not job.enabled:
        raise ValueError(job.disabled_reason or f"Job {job_id} is disabled")

    run_id = new_run_id(job.job_id)
    params = job.validate_parameters(parameters or {})
    output_notebook = _output_notebook_path(job, run_id)
    if not output_notebook.is_absolute():
        output_notebook = RUNS_ROOT.parents[0] / output_notebook
    log_path = RUNS_ROOT / run_id / "run.log"
    ensure_dir(log_path.parent)

    run = JobRun(
        run_id=run_id,
        job_id=job.job_id,
        status=JobStatus.PENDING,
        created_at=utc_now(),
        parameters=params,
        runner_type=job.runner_type,
        notebook_path=str(job.notebook_path or ""),
        output_notebook_path=str(output_notebook),
        log_path=str(log_path),
    )
    store.save_run(run)
    return run


def execute_run(
    run_id: str,
    roots: dict[str, Path] | None = None,
    store: JobStore | None = None,
    prefer_fallback: bool = True,
) -> JobRun:
    roots = roots or default_roots()
    if roots.get("financial_db"):
        os.environ["FINANCIAL_DB_ROOT"] = str(roots["financial_db"])
        os.environ["DB_BASE"] = str(roots["financial_db"])
        os.environ["DATA_PATH"] = str(roots["financial_db"])
    store = store or JobStore()
    run = store.get_run(run_id)
    if run is None:
        raise ValueError(f"Run not found: {run_id}")
    job = get_job(run.job_id)
    params = run.parameters
    output_notebook = Path(run.output_notebook_path)
    log_path = Path(run.log_path)
    ensure_dir(log_path.parent)

    run.status = JobStatus.RUNNING
    run.started_at = utc_now()
    store.save_run(run)
    append_log(log_path, f"Run started: {run.run_id}")
    append_log(log_path, f"Job: {job.job_id}")
    append_log(log_path, f"Parameters: {params}")

    try:
        if job.runner_type == "module":
            result = generate_all_artifacts(roots)
            append_log(log_path, str(result))
            output_notebook.write_text("Module-only job. No notebook executed.\n", encoding="utf-8")
        elif job.runner_type == "data_platform":
            max_files = int(params.get("top_n") or 5000)
            paths = write_data_platform_status(roots["financial_db"], roots["workspace"], max_files=max_files)
            append_log(log_path, f"Data platform status written: {paths}")
            output_notebook.write_text("Data-platform status job. No notebook executed.\n", encoding="utf-8")
        elif job.runner_type == "price_refresh":
            max_symbols = int(params.get("top_n") or 25)
            force = bool(params.get("refresh_cache", False))
            manifest = refresh_europe_stoxx_prices_incremental(
                roots["financial_db"],
                max_symbols=max_symbols,
                force=force,
            )
            append_log(log_path, manifest.to_string(index=False))
            paths = write_data_platform_status(roots["financial_db"], roots["workspace"], max_files=5000)
            append_log(log_path, f"Data platform status written after price refresh: {paths}")
            output_notebook.write_text("Price-refresh module job. No notebook executed.\n", encoding="utf-8")
        elif job.runner_type == "papermill":
            try:
                run_with_papermill(job, params, output_notebook, log_path)
            except Exception as exc:
                if not prefer_fallback:
                    raise
                append_log(log_path, f"Papermill failed, falling back to nbclient: {exc}")
                run.runner_type = "nbclient_fallback"
                run_with_nbclient(job, params, output_notebook, log_path)
        elif job.runner_type == "nbclient":
            run_with_nbclient(job, params, output_notebook, log_path)
        else:
            raise ValueError(f"Unsupported runner_type: {job.runner_type}")

        artifact_root = _artifact_root(job, roots)
        if job.runner_type == "price_refresh":
            workspace_specs = [spec for spec in job.expected_artifacts if not spec.relative_path.startswith("catalog/")]
            drive_specs = [spec for spec in job.expected_artifacts if spec.relative_path.startswith("catalog/")]
            from .artifact_contracts import validate_artifacts
            run.artifacts_detected = validate_artifacts(artifact_root, workspace_specs) + validate_artifacts(roots["financial_db"], drive_specs)
        else:
            run.artifacts_detected = validate_expected_artifacts(job, artifact_root)
        required_missing = [row for row in run.artifacts_detected if row["required"] and not row["exists"]]
        run.status = JobStatus.FAILED if required_missing else JobStatus.SUCCESS
        if required_missing:
            run.error_message = "Required artifacts missing after execution"
            append_log(log_path, run.error_message)
        else:
            append_log(log_path, "Run completed successfully")
    except Exception as exc:
        run.status = JobStatus.FAILED
        run.error_message = str(exc)
        append_log(log_path, traceback.format_exc())
    finally:
        run.finished_at = utc_now()
        store.save_run(run)
    return run


def run_job(
    job_id: str,
    parameters: dict | None = None,
    roots: dict[str, Path] | None = None,
    store: JobStore | None = None,
    prefer_fallback: bool = True,
) -> JobRun:
    store = store or JobStore()
    run = create_job_run(job_id, parameters, store=store)
    return execute_run(run.run_id, roots=roots, store=store, prefer_fallback=prefer_fallback)
