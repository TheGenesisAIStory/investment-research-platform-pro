"""Artifact freshness monitoring."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .artifact_contracts import validate_artifacts
from .registry import get_job_registry


def build_freshness_table(roots: dict[str, Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for job in get_job_registry().values():
        root = roots.get(job.output_domain, roots.get("workspace", Path.cwd()))
        for row in validate_artifacts(root, job.expected_artifacts):
            row["job_id"] = job.job_id
            row["job_label"] = job.label
            row["output_domain"] = job.output_domain
            rows.append(row)
    return pd.DataFrame(rows)


def freshness_badge(status: str) -> str:
    palette = {
        "OK": "#067647",
        "STALE": "#b54708",
        "MISSING_REQUIRED": "#b42318",
        "MISSING_OPTIONAL": "#667085",
    }
    color = palette.get(status, "#667085")
    return f"<span style='background:{color};color:white;padding:3px 8px;border-radius:999px;font-size:12px'>{status}</span>"
