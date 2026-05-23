"""Smoke checks for the Streamlit orchestration layer."""

from __future__ import annotations

import tempfile
from pathlib import Path

from orchestration import JobStore, build_freshness_table, create_job_run, execute_run, get_job, get_job_registry, jobs_needing_refresh, run_job, validate_expected_artifacts
from orchestration.models import JobRun
from orchestration.status import JobStatus
from support import default_roots


def main() -> int:
    registry = get_job_registry()
    assert "valuation_research_refresh" in registry
    assert "portfolio_research_refresh" in registry
    assert "screener_refresh" in registry
    assert "data_platform_status_refresh" in registry

    screener = get_job("screener_refresh")
    params = screener.validate_parameters({"top_n": 10, "refresh_cache": False})
    assert params["top_n"] == 10

    roots = default_roots()
    contracts = validate_expected_artifacts(screener, roots["company"])
    assert contracts and {"label", "exists", "status"}.issubset(contracts[0])

    freshness = build_freshness_table(roots)
    assert "job_id" in freshness.columns

    with tempfile.TemporaryDirectory() as tmp:
        store = JobStore(Path(tmp) / "runs.json")
        run = JobRun(
            run_id="test",
            job_id="screener_refresh",
            status=JobStatus.PENDING,
            created_at="2026-01-01T00:00:00+00:00",
        )
        store.save_run(run)
        loaded = store.get_run("test")
        assert loaded is not None and loaded.status == JobStatus.PENDING
        created = create_job_run("screener_refresh", {"top_n": 5}, store=store)
        executed = execute_run(created.run_id, roots=roots, store=store)
        assert executed.status in {JobStatus.SUCCESS, JobStatus.FAILED}

    dry_store = JobStore(Path(tempfile.mkdtemp()) / "runs.json")
    module_run = run_job("screener_refresh", {"top_n": 5}, roots=roots, store=dry_store)
    assert module_run.status in {JobStatus.SUCCESS, JobStatus.FAILED}
    assert module_run.log_path
    assert isinstance(jobs_needing_refresh(roots, ["screener_refresh"]), list)
    data_run = run_job("data_platform_status_refresh", {"top_n": 500}, roots=roots, store=JobStore(Path(tempfile.mkdtemp()) / "runs.json"))
    assert data_run.status in {JobStatus.SUCCESS, JobStatus.FAILED}

    print("orchestration_smoke_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
