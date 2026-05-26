"""Model monitoring artifacts for ML Stock Lab.

The training layer writes predictions by date/ticker/model.  This module turns
those predictions into rolling IC diagnostics so deterioration across regimes
is visible without reopening notebooks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .data_platform import resolve_data_platform_roots, utc_now


PREDICTION_CANDIDATES: tuple[Path, ...] = (
    Path("ml_training_lab") / "tables" / "MLTraining_predictions.csv",
    Path("ml_stock_lab") / "tables" / "MLStockLab_predictions.csv",
)
MONITORING_REL = Path("ml_lab") / "model_monitoring"
SUMMARY_NAME = "model_monitoring_summary.csv"


def _read_csv(path: Path, **kwargs: object) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 1:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, **kwargs)
    except Exception:
        return pd.DataFrame()


def discover_prediction_path(output_root: str | Path | None = None) -> Path | None:
    roots = resolve_data_platform_roots(repo_output_root=output_root)
    for rel in PREDICTION_CANDIDATES:
        path = roots.repo_output / rel
        if path.exists() and path.stat().st_size > 1:
            return path
    return None


def _daily_ic(predictions: pd.DataFrame, model: str, target_col: str, prediction_col: str) -> pd.DataFrame:
    frame = predictions[predictions["model"].astype(str).eq(str(model))].copy() if "model" in predictions.columns else predictions.copy()
    if frame.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for date, group in frame.groupby("date"):
        if len(group) < 5:
            continue
        ic = group[prediction_col].corr(group[target_col], method="pearson")
        rank_ic = group[prediction_col].corr(group[target_col], method="spearman")
        rows.append({"date": date, "model": model, "ic": ic, "rank_ic": rank_ic, "names": int(len(group))})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out = out.dropna(subset=["date"]).sort_values("date")
    return out


def build_model_monitoring_artifacts(
    output_root: str | Path | None = None,
    *,
    predictions: pd.DataFrame | None = None,
    models: Iterable[str] | None = None,
    target_col: str = "forward_return",
    prediction_col: str = "expected_return",
    window: int = 252,
    write: bool = True,
) -> dict[str, pd.DataFrame]:
    """Compute rolling IC diagnostics and persist model-level parquet files."""
    roots = resolve_data_platform_roots(repo_output_root=output_root)
    if predictions is None:
        path = discover_prediction_path(roots.repo_output)
        predictions = _read_csv(path) if path else pd.DataFrame()
    frame = predictions.copy()
    if frame.empty or not {"date", target_col, prediction_col}.issubset(frame.columns):
        empty = pd.DataFrame(columns=["model", "observations", "latest_rank_ic_rolling", "latest_ic_rolling"])
        if write:
            root = roots.repo_output / MONITORING_REL
            root.mkdir(parents=True, exist_ok=True)
            empty.to_csv(root / SUMMARY_NAME, index=False)
        return {"summary": empty, "rolling": pd.DataFrame()}
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame[target_col] = pd.to_numeric(frame[target_col], errors="coerce")
    frame[prediction_col] = pd.to_numeric(frame[prediction_col], errors="coerce")
    frame = frame.dropna(subset=["date", target_col, prediction_col])
    model_list = list(models or (frame["model"].dropna().astype(str).unique().tolist() if "model" in frame.columns else ["model"]))
    rolling_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    root = roots.repo_output / MONITORING_REL
    if write:
        root.mkdir(parents=True, exist_ok=True)
    for model in model_list:
        daily = _daily_ic(frame, model, target_col, prediction_col)
        if daily.empty:
            continue
        min_periods = min(int(window), max(3, min(int(window), 60)))
        daily["ic_rolling_12m"] = daily["ic"].rolling(window, min_periods=min_periods).mean()
        daily["rank_ic_rolling_12m"] = daily["rank_ic"].rolling(window, min_periods=min_periods).mean()
        daily["generated_at"] = utc_now()
        rolling_frames.append(daily)
        latest = daily.tail(1).iloc[0]
        summary_rows.append(
            {
                "model": model,
                "observations": int(len(daily)),
                "latest_date": pd.to_datetime(latest["date"]).date().isoformat(),
                "latest_ic": float(latest["ic"]) if pd.notna(latest["ic"]) else np.nan,
                "latest_rank_ic": float(latest["rank_ic"]) if pd.notna(latest["rank_ic"]) else np.nan,
                "latest_ic_rolling": float(latest["ic_rolling_12m"]) if pd.notna(latest["ic_rolling_12m"]) else np.nan,
                "latest_rank_ic_rolling": float(latest["rank_ic_rolling_12m"]) if pd.notna(latest["rank_ic_rolling_12m"]) else np.nan,
                "target": target_col,
                "prediction": prediction_col,
                "window": int(window),
                "generated_at": utc_now(),
            }
        )
        if write:
            safe_model = "".join(ch if ch.isalnum() else "_" for ch in str(model))
            daily.to_parquet(root / f"{safe_model}_rolling_ic.parquet", index=False)
    summary = pd.DataFrame(summary_rows)
    rolling = pd.concat(rolling_frames, ignore_index=True) if rolling_frames else pd.DataFrame()
    if write:
        summary.to_csv(root / SUMMARY_NAME, index=False)
    return {"summary": summary, "rolling": rolling}


def load_model_monitoring_artifacts(output_root: str | Path | None = None) -> dict[str, pd.DataFrame]:
    roots = resolve_data_platform_roots(repo_output_root=output_root)
    root = roots.repo_output / MONITORING_REL
    summary = _read_csv(root / SUMMARY_NAME)
    frames: list[pd.DataFrame] = []
    for path in sorted(root.glob("*_rolling_ic.parquet")):
        try:
            frames.append(pd.read_parquet(path))
        except Exception:
            continue
    rolling = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return {"summary": summary, "rolling": rolling}
