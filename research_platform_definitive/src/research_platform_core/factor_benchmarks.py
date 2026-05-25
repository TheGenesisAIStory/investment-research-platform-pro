"""Factor benchmark portfolios for ML Stock Lab governance.

The ML models should be judged against simple, transparent factor baselines:
top-ranked value, quality, momentum, low-vol/risk and composite factor baskets.
This module computes lightweight long-only/top-bucket and long-short summaries
from the canonical FactorUniversePanel without depending on Streamlit.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .data_platform import resolve_data_platform_roots, utc_now


FACTOR_PANEL_CANDIDATES: tuple[Path, ...] = (
    Path("ml_training_lab") / "tables" / "FactorUniversePanel.csv",
    Path("tables") / "FactorUniversePanel.csv",
)
BENCHMARK_TABLE_REL = Path("ml_training_lab") / "tables" / "FactorBenchmarkSummary.csv"
BENCHMARK_MANIFEST_REL = Path("ml_training_lab") / "FactorBenchmarkManifest.json"
DEFAULT_FACTORS: tuple[str, ...] = (
    "value_score",
    "quality_score",
    "momentum_score",
    "risk_score",
    "size_score",
    "growth_score",
    "factor_composite_score",
)


def _read_csv(path: Path, **kwargs: object) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 1:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, **kwargs)
    except Exception:
        return pd.DataFrame()


def discover_factor_panel_path(output_root: str | Path | None = None) -> Path | None:
    """Return the first existing FactorUniversePanel path."""
    roots = resolve_data_platform_roots(repo_output_root=output_root)
    for rel in FACTOR_PANEL_CANDIDATES:
        candidate = roots.repo_output / rel
        if candidate.exists() and candidate.stat().st_size > 1:
            return candidate
    return None


def _target_horizon_days(target_col: str) -> int:
    match = re.search(r"(\d+)d", str(target_col))
    return int(match.group(1)) if match else 21


def _available_columns(path: Path) -> list[str]:
    frame = _read_csv(path, nrows=0)
    return frame.columns.tolist()


def _clean_panel(
    path: Path,
    factors: Iterable[str],
    target_col: str,
    *,
    max_rows: int | None = None,
) -> tuple[pd.DataFrame, str]:
    columns = _available_columns(path)
    if not columns:
        return pd.DataFrame(), target_col
    target = target_col if target_col in columns else "forward_return" if "forward_return" in columns else ""
    if not target:
        return pd.DataFrame(), target_col
    requested = ["date", target, *[factor for factor in factors if factor in columns]]
    if not requested or target not in requested:
        return pd.DataFrame(), target
    frame = _read_csv(path, usecols=lambda col: col in set(requested), nrows=max_rows)
    if frame.empty or target not in frame.columns:
        return pd.DataFrame(), target
    frame = frame.copy()
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    else:
        frame["date"] = "panel"
    frame[target] = pd.to_numeric(frame[target], errors="coerce")
    return frame, target


def _factor_summary(
    frame: pd.DataFrame,
    factor: str,
    target_col: str,
    *,
    top_quantile: float,
    bottom_quantile: float,
    horizon_days: int,
) -> dict[str, object]:
    work = frame[["date", factor, target_col]].copy()
    work[factor] = pd.to_numeric(work[factor], errors="coerce")
    work[target_col] = pd.to_numeric(work[target_col], errors="coerce")
    work = work.dropna(subset=[factor, target_col])
    if work.empty:
        return {
            "factor": factor,
            "target": target_col,
            "horizon_days": horizon_days,
            "status": "NO_OBSERVATIONS",
            "observations": 0,
            "dates": 0,
        }

    ranks = work.groupby("date", dropna=False)[factor].rank(pct=True, method="average")
    work["bucket"] = np.select(
        [ranks >= 1.0 - float(top_quantile), ranks <= float(bottom_quantile)],
        ["top", "bottom"],
        default="middle",
    )
    by_date = work.groupby(["date", "bucket"], dropna=False)[target_col].mean().unstack("bucket")
    top = by_date["top"] if "top" in by_date.columns else pd.Series(dtype=float)
    bottom = by_date["bottom"] if "bottom" in by_date.columns else pd.Series(dtype=float)
    long_short = (top - bottom).dropna()
    top = top.dropna()
    bottom = bottom.dropna()
    rank_ic = work.groupby("date", dropna=False)[[factor, target_col]].apply(
        lambda group: group[factor].corr(group[target_col], method="spearman") if len(group) >= 5 else np.nan
    ).dropna()
    annualization = float(np.sqrt(252.0 / max(int(horizon_days), 1)))
    top_std = float(top.std(ddof=1)) if len(top) > 1 else np.nan
    ls_std = float(long_short.std(ddof=1)) if len(long_short) > 1 else np.nan
    return {
        "factor": factor,
        "target": target_col,
        "horizon_days": horizon_days,
        "status": "OK",
        "observations": int(len(work)),
        "dates": int(work["date"].nunique()),
        "top_quantile": float(top_quantile),
        "bottom_quantile": float(bottom_quantile),
        "top_mean_return": float(top.mean()) if len(top) else np.nan,
        "bottom_mean_return": float(bottom.mean()) if len(bottom) else np.nan,
        "long_short_mean_return": float(long_short.mean()) if len(long_short) else np.nan,
        "top_hit_rate": float((top > 0).mean()) if len(top) else np.nan,
        "long_short_hit_rate": float((long_short > 0).mean()) if len(long_short) else np.nan,
        "top_long_sharpe": float(top.mean() / top_std * annualization) if top_std and pd.notna(top_std) and top_std > 0 else np.nan,
        "long_short_sharpe": float(long_short.mean() / ls_std * annualization) if ls_std and pd.notna(ls_std) and ls_std > 0 else np.nan,
        "rank_ic_mean": float(rank_ic.mean()) if len(rank_ic) else np.nan,
        "rank_ic_observations": int(len(rank_ic)),
    }


def compute_factor_benchmark_summary(
    output_root: str | Path | None = None,
    *,
    factors: Iterable[str] = DEFAULT_FACTORS,
    target_col: str = "forward_return_21d",
    top_quantile: float = 0.2,
    bottom_quantile: float = 0.2,
    max_rows: int | None = 250_000,
    write: bool = True,
) -> pd.DataFrame:
    """Compute factor baseline performance summaries from FactorUniversePanel.

    `max_rows` keeps the Streamlit action responsive; scheduled jobs can pass
    `None` to evaluate the full panel.
    """
    path = discover_factor_panel_path(output_root)
    factor_list = list(dict.fromkeys(str(factor) for factor in factors if str(factor).strip()))
    if path is None:
        return pd.DataFrame(columns=["factor", "target", "status", "observations", "dates"])
    frame, target = _clean_panel(path, factor_list, target_col, max_rows=max_rows)
    horizon = _target_horizon_days(target)
    rows: list[dict[str, object]] = []
    for factor in factor_list:
        if frame.empty or factor not in frame.columns:
            rows.append(
                {
                    "factor": factor,
                    "target": target,
                    "horizon_days": horizon,
                    "status": "MISSING_FACTOR",
                    "observations": 0,
                    "dates": 0,
                }
            )
            continue
        rows.append(
            _factor_summary(
                frame,
                factor,
                target,
                top_quantile=top_quantile,
                bottom_quantile=bottom_quantile,
                horizon_days=horizon,
            )
        )
    out = pd.DataFrame(rows)
    if write:
        write_factor_benchmark_artifacts(out, output_root, source_path=path, target_col=target, max_rows=max_rows)
    return out


def write_factor_benchmark_artifacts(
    summary: pd.DataFrame,
    output_root: str | Path | None = None,
    *,
    source_path: str | Path | None = None,
    target_col: str = "forward_return_21d",
    max_rows: int | None = 250_000,
) -> dict[str, str]:
    roots = resolve_data_platform_roots(repo_output_root=output_root)
    table_path = roots.repo_output / BENCHMARK_TABLE_REL
    manifest_path = roots.repo_output / BENCHMARK_MANIFEST_REL
    table_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(table_path, index=False)
    manifest = {
        "generated_at": utc_now(),
        "source_path": str(source_path or ""),
        "target_col": target_col,
        "max_rows": max_rows,
        "rows": int(len(summary)),
        "status": "OK" if not summary.empty else "EMPTY",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"summary": str(table_path), "manifest": str(manifest_path)}


def load_factor_benchmark_summary(
    output_root: str | Path | None = None,
    *,
    refresh_if_missing: bool = False,
    max_rows: int | None = 250_000,
) -> pd.DataFrame:
    roots = resolve_data_platform_roots(repo_output_root=output_root)
    path = roots.repo_output / BENCHMARK_TABLE_REL
    frame = _read_csv(path)
    if frame.empty and refresh_if_missing:
        frame = compute_factor_benchmark_summary(roots.repo_output, max_rows=max_rows, write=True)
    return frame
