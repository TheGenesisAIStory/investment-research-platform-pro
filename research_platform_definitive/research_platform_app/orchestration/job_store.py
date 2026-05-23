"""JSON-backed run history store."""

from __future__ import annotations

from pathlib import Path

from .models import JobRun
from .utils import STATE_ROOT, ensure_dir, read_json, write_json


class JobStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or STATE_ROOT / "job_runs.json"
        ensure_dir(self.path.parent)

    def list_runs(self) -> list[JobRun]:
        payload = read_json(self.path, [])
        return [JobRun.from_dict(item) for item in payload if isinstance(item, dict)]

    def save_run(self, run: JobRun) -> None:
        runs = [item for item in self.list_runs() if item.run_id != run.run_id]
        runs.append(run)
        runs = sorted(runs, key=lambda item: item.created_at, reverse=True)
        write_json(self.path, [item.to_dict() for item in runs])

    def get_run(self, run_id: str) -> JobRun | None:
        for run in self.list_runs():
            if run.run_id == run_id:
                return run
        return None

    def as_frame(self):
        import pandas as pd

        return pd.DataFrame([run.to_dict() for run in self.list_runs()])
