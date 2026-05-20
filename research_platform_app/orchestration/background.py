"""Background process helpers for non-blocking Streamlit job launches."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from .job_store import JobStore
from .models import JobRun
from .runner import create_job_run
from .utils import APP_ROOT


def launch_job_process(
    job_id: str,
    parameters: dict | None,
    roots: dict[str, Path],
    store: JobStore | None = None,
) -> JobRun:
    """Create a pending run and execute it in a detached Python subprocess."""
    store = store or JobStore()
    run = create_job_run(job_id, parameters, store=store)
    cmd = [
        sys.executable,
        str(APP_ROOT / "worker.py"),
        "--run-id",
        run.run_id,
        "--company-root",
        str(roots["company"]),
        "--portfolio-root",
        str(roots["portfolio"]),
        "--workspace-root",
        str(roots["workspace"]),
        "--financial-db-root",
        str(roots.get("financial_db", "")),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{APP_ROOT}:{APP_ROOT.parent}:{env.get('PYTHONPATH', '')}"
    env["FINANCIAL_DB_ROOT"] = str(roots.get("financial_db", ""))
    env["DB_BASE"] = str(roots.get("financial_db", ""))
    env["DATA_PATH"] = str(roots.get("financial_db", ""))
    subprocess.Popen(
        cmd,
        cwd=str(APP_ROOT.parent),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return run
