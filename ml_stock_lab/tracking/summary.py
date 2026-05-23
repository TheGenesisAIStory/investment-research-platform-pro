"""Experiment-card helpers for ML stock labs."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any
import json

import pandas as pd


def _to_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return value
    if isinstance(value, pd.Series):
        return value.to_dict()
    return {"value": value}


def build_experiment_summary(
    hyperparams: dict[str, Any],
    panel_validation: Any = None,
    metrics: dict[str, Any] | pd.Series | None = None,
    status: Any = None,
) -> dict[str, Any]:
    """Build a flat experiment-card dict from config, panel info and metrics."""
    panel = _to_dict(panel_validation)
    metric_values = _to_dict(metrics)
    status_values = _to_dict(status)
    summary = {f"param_{k}": v for k, v in (hyperparams or {}).items()}
    summary.update({f"panel_{k}": v for k, v in panel.items() if k != "extra"})
    summary.update({f"metric_{k}": v for k, v in metric_values.items()})
    summary.update({f"status_{k}": v for k, v in status_values.items()})
    return summary


def save_experiment_summary(summary: dict[str, Any], output_root: Path, filename: str = "MLStockLab_experiment_summary.json") -> Path:
    """Save an experiment summary as JSON and a companion one-row CSV."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / filename
    json_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    csv_path = output_root / filename.replace(".json", ".csv")
    pd.DataFrame([summary]).to_csv(csv_path, index=False)
    return json_path
