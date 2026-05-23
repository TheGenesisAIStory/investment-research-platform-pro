"""Evaluation metrics for ML stock valuation and portfolio experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd


def oos_r2(y_true: pd.Series, y_pred: pd.Series, benchmark: pd.Series | float | None = None) -> float:
    """R2_OS = 1 - SSE_model / SSE_benchmark."""
    y = pd.to_numeric(y_true, errors="coerce")
    p = pd.to_numeric(y_pred, errors="coerce")
    if benchmark is None:
        b = pd.Series(y.mean(), index=y.index)
    elif isinstance(benchmark, (int, float)):
        b = pd.Series(float(benchmark), index=y.index)
    else:
        b = pd.to_numeric(benchmark, errors="coerce")
    mask = y.notna() & p.notna() & b.notna()
    if mask.sum() == 0:
        return float("nan")
    denom = ((y[mask] - b[mask]) ** 2).sum()
    return float(1 - ((y[mask] - p[mask]) ** 2).sum() / denom) if denom else float("nan")


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    r = pd.to_numeric(returns, errors="coerce").dropna()
    if r.empty or r.std(ddof=0) == 0:
        return float("nan")
    return float((r.mean() / r.std(ddof=0)) * np.sqrt(periods_per_year))


def turnover(weights: pd.DataFrame, date_col: str = "date", ticker_col: str = "ticker", weight_col: str = "weight") -> float:
    if weights.empty or not {date_col, ticker_col, weight_col}.issubset(weights.columns):
        return float("nan")
    pivot = weights.pivot_table(index=date_col, columns=ticker_col, values=weight_col, aggfunc="sum").fillna(0)
    return float(pivot.diff().abs().sum(axis=1).mean())


def transaction_cost_adjusted_returns(returns: pd.Series, turnover_series: pd.Series | float, cost_bps: float = 10.0) -> pd.Series:
    r = pd.to_numeric(returns, errors="coerce")
    t = turnover_series if isinstance(turnover_series, pd.Series) else pd.Series(float(turnover_series), index=r.index)
    return (r - pd.to_numeric(t, errors="coerce").fillna(0) * cost_bps / 10000).rename("net_return")


def evaluate_quintile_backtest(qret: pd.DataFrame) -> pd.DataFrame:
    if qret.empty or "return" not in qret.columns:
        return pd.DataFrame(columns=["portfolio", "mean_return", "volatility", "sharpe", "observations", "avg_names"])
    rows = []
    for key, grp in qret.groupby("quantile"):
        r = pd.to_numeric(grp["return"], errors="coerce")
        names = pd.to_numeric(grp["name_count"], errors="coerce") if "name_count" in grp.columns else pd.Series(dtype=float)
        rows.append({
            "portfolio": str(key),
            "mean_return": r.mean(),
            "volatility": r.std(ddof=0),
            "sharpe": sharpe_ratio(r, periods_per_year=12 if len(r) < 80 else 252),
            "observations": r.notna().sum(),
            "avg_names": names.mean() if not names.empty else np.nan,
        })
    return pd.DataFrame(rows)


def factor_alpha(portfolio_returns: pd.Series, factor_data: pd.DataFrame | None = None) -> pd.DataFrame:
    """Small factor-alpha helper; returns intercept-only alpha if factors absent."""
    r = pd.to_numeric(portfolio_returns, errors="coerce").dropna()
    if r.empty:
        return pd.DataFrame(columns=["alpha", "beta_count", "r2"])
    if factor_data is None or factor_data.empty:
        return pd.DataFrame([{"alpha": r.mean(), "beta_count": 0, "r2": np.nan}])
    common = factor_data.loc[factor_data.index.intersection(r.index)].select_dtypes("number")
    y = r.loc[common.index]
    if common.empty:
        return pd.DataFrame([{"alpha": r.mean(), "beta_count": 0, "r2": np.nan}])
    try:
        from sklearn.linear_model import LinearRegression
        model = LinearRegression().fit(common, y)
        return pd.DataFrame([{"alpha": model.intercept_, "beta_count": common.shape[1], "r2": model.score(common, y)}])
    except Exception:
        return pd.DataFrame([{"alpha": r.mean(), "beta_count": 0, "r2": np.nan}])
