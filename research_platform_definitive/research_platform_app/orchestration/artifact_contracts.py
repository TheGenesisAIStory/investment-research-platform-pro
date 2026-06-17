"""Artifact contract validation for notebook jobs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .models import ArtifactSpec, NotebookJob


def validate_artifacts(root: Path, specs: list[ArtifactSpec]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    now = pd.Timestamp.now(tz="UTC")
    for spec in specs:
        path = root / spec.relative_path
        exists = path.exists()
        modified = pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC") if exists else pd.NaT
        age_hours = round((now - modified).total_seconds() / 3600, 2) if exists and pd.notna(modified) else None
        stale = bool(spec.freshness_hours is not None and age_hours is not None and age_hours > spec.freshness_hours)
        status = "OK" if exists and not stale else "STALE" if exists and stale else "MISSING_REQUIRED" if spec.required else "MISSING_OPTIONAL"
        rows.append({
            "label": spec.label,
            "relative_path": spec.relative_path,
            "path": str(path),
            "required": spec.required,
            "exists": exists,
            "status": status,
            "size_kb": round(path.stat().st_size / 1024, 2) if exists else 0.0,
            "modified": modified.isoformat() if exists else "",
            "age_hours": age_hours,
            "freshness_hours": spec.freshness_hours,
        })
    return rows


def validate_expected_artifacts(job: NotebookJob, output_root: Path) -> list[dict[str, Any]]:
    return validate_artifacts(output_root, job.expected_artifacts)


def artifact_summary_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows)
