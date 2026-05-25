"""Monthly factor portfolio baselines for ML model governance.

These are transparent controls for the ML Stock Lab: if a model cannot beat
simple value/quality/momentum factor baskets, the desk should see that plainly.
The implementation is intentionally conservative and uses only point-in-time
factor columns plus forward-return targets already present in FactorUniversePanel.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .data_platform import resolve_data_platform_roots, utc_now
from .factor_benchmarks import DEFAULT_FACTORS, discover_factor_panel_path


OUTPUT_REL = Path("factor_baselines")
METRICS_NAME = "baseline_portfolio_metrics.csv"
RETURNS_NAME = "baseline_portfolio_returns.csv"
DEFAULT_BASELINE_FACTORS: tuple[str, ...] = ("value_score", "quality_score", "momentum_score")


def _read_csv(path: Path, **kwargs: object) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 1:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, **kwargs)
    except Exception:
        return pd.DataFrame()


def _target_for_horizon(columns: Iterable[str], horizon: int) -> str:
    candidates = [f"forward_return_{int(horizon)}d", "forward_return"]
    available = set(columns)
    return next((col for col in candidates if col in available), "")


def _load_panel(
    output_root: str | Path | None = None,
    *,
    panel: pd.DataFrame | None = None,
    factors: Iterable[str] = DEFAULT_BASELINE_FACTORS,
    horizons: Iterable[int] = (21, 63, 252),
    max_rows: int | None = 500_000,
) -> pd.DataFrame:
    if panel is not None:
        frame = panel.copy()
    else:
        path = discover_factor_panel_path(output_root)
        if path is None:
            return pd.DataFrame()
        header = _read_csv(path, nrows=0).columns.tolist()
        target_cols = [col for horizon in horizons for col in [_target_for_horizon(header, int(horizon))] if col]
        usecols = set(["date", "ticker", *factors, *target_cols])
        frame = _read_csv(path, usecols=lambda col: col in usecols, nrows=max_rows)
    if frame.empty or "date" not in frame.columns or "ticker" not in frame.columns:
        return pd.DataFrame()
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame = frame.dropna(subset=["date", "ticker"])
    frame["month"] = frame["date"].dt.to_period("M").astype(str)
    return frame


def _max_drawdown(returns: pd.Series) -> float:
    clean = pd.to_numeric(returns, errors="coerce").dropna()
    if clean.empty:
        return np.nan
    equity = (1.0 + clean).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return float(drawdown.min())


def _sortino(returns: pd.Series, periods_per_year: int = 12) -> float:
    clean = pd.to_numeric(returns, errors="coerce").dropna()
    downside = clean[clean < 0]
    if clean.empty or len(downside) < 2:
        return np.nan
    downside_std = float(downside.std(ddof=1))
    if not downside_std:
        return np.nan
    return float(clean.mean() / downside_std * np.sqrt(periods_per_year))


def _sharpe(returns: pd.Series, periods_per_year: int = 12) -> float:
    clean = pd.to_numeric(returns, errors="coerce").dropna()
    if len(clean) < 2:
        return np.nan
    std = float(clean.std(ddof=1))
    if not std:
        return np.nan
    return float(clean.mean() / std * np.sqrt(periods_per_year))


def _monthly_turnover(selected_by_month: dict[str, set[str]]) -> float:
    turnovers: list[float] = []
    previous: set[str] | None = None
    for month in sorted(selected_by_month):
        current = selected_by_month[month]
        if previous is not None:
            union = current | previous
            turnovers.append(float(len(current.symmetric_difference(previous)) / max(len(union), 1)))
        previous = current
    return float(np.nanmean(turnovers)) if turnovers else np.nan


def _strategy_metrics(
    returns: pd.Series,
    *,
    factor: str,
    horizon: int,
    target_col: str,
    strategy: str,
    ic_by_month: pd.Series,
    turnover: float,
) -> dict[str, object]:
    clean = pd.to_numeric(returns, errors="coerce").dropna()
    return {
        "factor": factor,
        "horizon": int(horizon),
        "target": target_col,
        "strategy": strategy,
        "status": "OK",
        "rebalance": "monthly",
        "observations": int(len(clean)),
        "annual_return": float(clean.mean() * 12.0) if len(clean) else np.nan,
        "sharpe": _sharpe(clean),
        "sortino": _sortino(clean),
        "maxdd": _max_drawdown(clean),
        "ic_mean": float(ic_by_month.dropna().mean()) if len(ic_by_month.dropna()) else np.nan,
        "turnover": turnover,
        "generated_at": utc_now(),
    }


def compute_factor_portfolio_baselines(
    output_root: str | Path | None = None,
    *,
    panel: pd.DataFrame | None = None,
    factors: Iterable[str] = DEFAULT_BASELINE_FACTORS,
    horizons: Iterable[int] = (21, 63, 252),
    max_rows: int | None = 500_000,
    write: bool = True,
) -> dict[str, pd.DataFrame]:
    """Compute monthly long-only and long-short factor baseline metrics."""
    factor_list = [factor for factor in dict.fromkeys(factors) if factor]
    horizon_list = [int(horizon) for horizon in horizons]
    frame = _load_panel(output_root, panel=panel, factors=factor_list, horizons=horizon_list, max_rows=max_rows)
    if frame.empty:
        empty_metrics = pd.DataFrame(columns=["factor", "horizon", "strategy", "sharpe", "sortino", "maxdd", "ic_mean", "turnover"])
        empty_returns = pd.DataFrame(columns=["month", "factor", "horizon", "strategy", "return"])
        if write:
            write_factor_portfolio_baselines(empty_metrics, empty_returns, output_root)
        return {"metrics": empty_metrics, "returns": empty_returns}

    metrics_rows: list[dict[str, object]] = []
    return_rows: list[dict[str, object]] = []
    for horizon in horizon_list:
        target_col = _target_for_horizon(frame.columns, horizon)
        if not target_col:
            continue
        frame[target_col] = pd.to_numeric(frame[target_col], errors="coerce")
        for factor in factor_list:
            if factor not in frame.columns:
                for strategy in ["long_only", "long_short"]:
                    metrics_rows.append(
                        {
                            "factor": factor,
                            "horizon": horizon,
                            "target": target_col,
                            "strategy": strategy,
                            "status": "MISSING_FACTOR",
                            "rebalance": "monthly",
                            "observations": 0,
                            "annual_return": np.nan,
                            "sharpe": np.nan,
                            "sortino": np.nan,
                            "maxdd": np.nan,
                            "ic_mean": np.nan,
                            "turnover": np.nan,
                            "generated_at": utc_now(),
                        }
                    )
                continue
            work = frame[["month", "ticker", factor, target_col]].copy()
            work[factor] = pd.to_numeric(work[factor], errors="coerce")
            work = work.dropna(subset=[factor, target_col])
            if work.empty:
                for strategy in ["long_only", "long_short"]:
                    metrics_rows.append(
                        {
                            "factor": factor,
                            "horizon": horizon,
                            "target": target_col,
                            "strategy": strategy,
                            "status": "NO_OBSERVATIONS",
                            "rebalance": "monthly",
                            "observations": 0,
                            "annual_return": np.nan,
                            "sharpe": np.nan,
                            "sortino": np.nan,
                            "maxdd": np.nan,
                            "ic_mean": np.nan,
                            "turnover": np.nan,
                            "generated_at": utc_now(),
                        }
                    )
                continue
            ranks = work.groupby("month")[factor].rank(pct=True, method="average")
            work["rank_pct"] = ranks
            long_only_returns: dict[str, float] = {}
            long_short_returns: dict[str, float] = {}
            selected_top10: dict[str, set[str]] = {}
            selected_top20: dict[str, set[str]] = {}
            ic_by_month = work.groupby("month")[[factor, target_col]].apply(
                lambda group: group[factor].corr(group[target_col], method="spearman") if len(group) >= 5 else np.nan
            )
            for month, group in work.groupby("month"):
                top10 = group[group["rank_pct"] >= 0.90]
                top20 = group[group["rank_pct"] >= 0.80]
                bottom20 = group[group["rank_pct"] <= 0.20]
                selected_top10[str(month)] = set(top10["ticker"].astype(str))
                selected_top20[str(month)] = set(top20["ticker"].astype(str))
                if not top10.empty:
                    long_only_returns[str(month)] = float(top10[target_col].mean())
                if not top20.empty and not bottom20.empty:
                    long_short_returns[str(month)] = float(top20[target_col].mean() - bottom20[target_col].mean())
            for month, value in long_only_returns.items():
                return_rows.append({"month": month, "factor": factor, "horizon": horizon, "strategy": "long_only", "return": value})
            for month, value in long_short_returns.items():
                return_rows.append({"month": month, "factor": factor, "horizon": horizon, "strategy": "long_short", "return": value})
            metrics_rows.append(
                _strategy_metrics(
                    pd.Series(long_only_returns, dtype=float),
                    factor=factor,
                    horizon=horizon,
                    target_col=target_col,
                    strategy="long_only",
                    ic_by_month=ic_by_month,
                    turnover=_monthly_turnover(selected_top10),
                )
            )
            metrics_rows.append(
                _strategy_metrics(
                    pd.Series(long_short_returns, dtype=float),
                    factor=factor,
                    horizon=horizon,
                    target_col=target_col,
                    strategy="long_short",
                    ic_by_month=ic_by_month,
                    turnover=_monthly_turnover(selected_top20),
                )
            )

    metrics = pd.DataFrame(metrics_rows)
    returns = pd.DataFrame(return_rows)
    if write:
        write_factor_portfolio_baselines(metrics, returns, output_root)
    return {"metrics": metrics, "returns": returns}


def write_factor_portfolio_baselines(
    metrics: pd.DataFrame,
    returns: pd.DataFrame,
    output_root: str | Path | None = None,
) -> dict[str, str]:
    roots = resolve_data_platform_roots(repo_output_root=output_root)
    root = roots.repo_output / OUTPUT_REL
    root.mkdir(parents=True, exist_ok=True)
    metrics_path = root / METRICS_NAME
    returns_path = root / RETURNS_NAME
    metrics.to_csv(metrics_path, index=False)
    returns.to_csv(returns_path, index=False)
    return {"metrics": str(metrics_path), "returns": str(returns_path)}


def load_factor_portfolio_baselines(
    output_root: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    roots = resolve_data_platform_roots(repo_output_root=output_root)
    root = roots.repo_output / OUTPUT_REL
    return {
        "metrics": _read_csv(root / METRICS_NAME),
        "returns": _read_csv(root / RETURNS_NAME),
    }
