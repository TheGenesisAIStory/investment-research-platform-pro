"""Papermill notebook runner."""

from __future__ import annotations

import subprocess
import sys
import os
from pathlib import Path

from .models import NotebookJob
from .utils import PROJECT_ROOT, append_log, ensure_dir


def run_with_papermill(job: NotebookJob, parameters: dict, output_notebook: Path, log_path: Path) -> None:
    if job.notebook_path is None:
        raise ValueError("Papermill runner requires a notebook_path")
    ensure_dir(output_notebook.parent)
    cmd = [
        sys.executable,
        "-m",
        "papermill",
        str(job.notebook_path),
        str(output_notebook),
        "--cwd",
        str(PROJECT_ROOT),
        "--log-output",
    ]
    for key, value in parameters.items():
        cmd.extend(["-p", str(key), str(value)])
    append_log(log_path, "$ " + " ".join(cmd))
    env = os.environ.copy()
    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        text=True,
        capture_output=True,
        timeout=job.timeout_seconds,
    )
    if proc.stdout:
        append_log(log_path, proc.stdout)
    if proc.stderr:
        append_log(log_path, proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"Papermill failed with exit code {proc.returncode}. See {log_path}")
