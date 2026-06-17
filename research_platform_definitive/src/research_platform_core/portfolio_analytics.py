"""Portfolio analytics and construction helpers for Gen.is.IA.

The functions are backend-only and Streamlit-independent. They operate on
periodic return series, benchmark returns and optional holdings metadata so the
Portfolio page can expose professional risk analytics without embedding finance
math in the UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize


TRADING_DAYS = 252


def _returns(series: Iterable[float] | pd.Series) -> pd.Series:
    values = pd.to_numeric(pd.Series(series), errors="coerce").dropna()
    return values.astype(float)


def _annualization(periods_per_year: int = TRADING_DAYS) -> float:
    return float(np.sqrt(periods_per_year))


def equity_curve(returns: Iterable[float] | pd.Series) -> pd.Series:
    clean = _returns(returns)
    if clean.empty:
        return pd.Series(dtype=float)
    return (1.0 + clean).cumprod()


def drawdown_series(returns: Iterable[float] | pd.Series) -> pd.Series:
    curve = equity_curve(returns)
    if curve.empty:
        return pd.Series(dtype=float)
    return curve / curve.cummax() - 1.0


def drawdown_durations(drawdowns: Iterable[float] | pd.Series) -> list[int]:
    durations: list[int] = []
    current = 0
    for value in pd.to_numeric(pd.Series(drawdowns), errors="coerce").fillna(0):
        if value < 0:
            current += 1
        elif current:
            durations.append(current)
            current = 0
    if current:
        durations.append(current)
    return durations


def max_drawdown(returns: Iterable[float] | pd.Series) -> float:
    dd = drawdown_series(returns)
    return float(dd.min()) if not dd.empty else np.nan


def sharpe_ratio(returns: Iterable[float] | pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = TRADING_DAYS) -> float:
    clean = _returns(returns)
    if len(clean) < 2:
        return np.nan
    excess = clean - risk_free_rate / periods_per_year
    std = float(excess.std(ddof=1))
    return float(excess.mean() / std * _annualization(periods_per_year)) if std else np.nan


def sortino_ratio(returns: Iterable[float] | pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = TRADING_DAYS) -> float:
    clean = _returns(returns)
    if clean.empty:
        return np.nan
    excess = clean - risk_free_rate / periods_per_year
    downside = excess[excess < 0]
    if len(downside) < 2:
        return np.nan
    down_std = float(downside.std(ddof=1))
    return float(excess.mean() / down_std * _annualization(periods_per_year)) if down_std else np.nan


def calmar_ratio(returns: Iterable[float] | pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    clean = _returns(returns)
    if clean.empty:
        return np.nan
    total = float((1.0 + clean).prod() - 1.0)
    years = max(len(clean) / periods_per_year, 1 / periods_per_year)
    cagr = (1.0 + total) ** (1.0 / years) - 1.0 if total > -1 else -1.0
    mdd = abs(max_drawdown(clean))
    return float(cagr / mdd) if mdd else np.nan


def omega_ratio(returns: Iterable[float] | pd.Series, threshold: float = 0.0) -> float:
    clean = _returns(returns)
    gains = (clean - threshold).clip(lower=0).sum()
    losses = (threshold - clean).clip(lower=0).sum()
    return float(gains / losses) if losses else np.nan


def kappa_ratio(returns: Iterable[float] | pd.Series, threshold: float = 0.0, order: int = 3, periods_per_year: int = TRADING_DAYS) -> float:
    clean = _returns(returns)
    if clean.empty:
        return np.nan
    excess_mean = float((clean - threshold).mean() * periods_per_year)
    lpm = float(np.mean(np.maximum(threshold - clean, 0) ** order))
    return float(excess_mean / (lpm ** (1.0 / order))) if lpm else np.nan


def tracking_error(portfolio_returns: Iterable[float], benchmark_returns: Iterable[float], periods_per_year: int = TRADING_DAYS) -> float:
    aligned = pd.concat([_returns(portfolio_returns), _returns(benchmark_returns)], axis=1).dropna()
    if len(aligned) < 2:
        return np.nan
    active = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    return float(active.std(ddof=1) * _annualization(periods_per_year))


def information_ratio(portfolio_returns: Iterable[float], benchmark_returns: Iterable[float], periods_per_year: int = TRADING_DAYS) -> float:
    aligned = pd.concat([_returns(portfolio_returns), _returns(benchmark_returns)], axis=1).dropna()
    if len(aligned) < 2:
        return np.nan
    active = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    te = float(active.std(ddof=1))
    return float(active.mean() / te * _annualization(periods_per_year)) if te else np.nan


def beta_to_benchmark(portfolio_returns: Iterable[float], benchmark_returns: Iterable[float]) -> float:
    aligned = pd.concat([_returns(portfolio_returns), _returns(benchmark_returns)], axis=1).dropna()
    if len(aligned) < 3:
        return np.nan
    var = float(aligned.iloc[:, 1].var(ddof=1))
    return float(aligned.iloc[:, 0].cov(aligned.iloc[:, 1]) / var) if var else np.nan


def treynor_ratio(portfolio_returns: Iterable[float], benchmark_returns: Iterable[float], risk_free_rate: float = 0.0, periods_per_year: int = TRADING_DAYS) -> float:
    clean = _returns(portfolio_returns)
    beta = beta_to_benchmark(portfolio_returns, benchmark_returns)
    annual_return = float(clean.mean() * periods_per_year) if not clean.empty else np.nan
    return float((annual_return - risk_free_rate) / beta) if beta and pd.notna(beta) else np.nan


def jensen_alpha(portfolio_returns: Iterable[float], benchmark_returns: Iterable[float], risk_free_rate: float = 0.0, periods_per_year: int = TRADING_DAYS) -> float:
    aligned = pd.concat([_returns(portfolio_returns), _returns(benchmark_returns)], axis=1).dropna()
    if aligned.empty:
        return np.nan
    beta = beta_to_benchmark(aligned.iloc[:, 0], aligned.iloc[:, 1])
    rp = float(aligned.iloc[:, 0].mean() * periods_per_year)
    rb = float(aligned.iloc[:, 1].mean() * periods_per_year)
    return float(rp - (risk_free_rate + beta * (rb - risk_free_rate))) if pd.notna(beta) else np.nan


def m2_measure(portfolio_returns: Iterable[float], benchmark_returns: Iterable[float], risk_free_rate: float = 0.0, periods_per_year: int = TRADING_DAYS) -> float:
    bench = _returns(benchmark_returns)
    bench_std = float(bench.std(ddof=1) * _annualization(periods_per_year)) if len(bench) > 1 else np.nan
    sharpe = sharpe_ratio(portfolio_returns, risk_free_rate, periods_per_year)
    return float(sharpe * bench_std + risk_free_rate) if pd.notna(sharpe) and pd.notna(bench_std) else np.nan


def value_at_risk(returns: Iterable[float], confidence: float = 0.95, method: str = "parametric") -> float:
    clean = _returns(returns)
    if clean.empty:
        return np.nan
    z = 1.645 if confidence == 0.95 else 2.326 if confidence == 0.99 else abs(float(pd.Series(clean).quantile(1 - confidence)))
    if method == "parametric":
        return float(clean.mean() - z * clean.std(ddof=1))
    return float(clean.quantile(1 - confidence))


def conditional_var(returns: Iterable[float], confidence: float = 0.95) -> float:
    clean = _returns(returns)
    if clean.empty:
        return np.nan
    var = value_at_risk(clean, confidence=confidence, method="historical")
    tail = clean[clean <= var]
    return float(tail.mean()) if not tail.empty else np.nan


def tail_ratio(returns: Iterable[float]) -> float:
    clean = _returns(returns)
    if clean.empty:
        return np.nan
    downside = abs(float(clean.quantile(0.05)))
    return float(abs(clean.quantile(0.95)) / downside) if downside else np.nan


def turnover_annual(weights: pd.DataFrame, periods_per_year: int = 12) -> float:
    if weights.empty:
        return np.nan
    numeric = weights.select_dtypes(include=[np.number])
    if numeric.empty or len(numeric) < 2:
        return np.nan
    periodic = numeric.diff().abs().sum(axis=1).dropna() / 2.0
    return float(periodic.mean() * periods_per_year) if not periodic.empty else np.nan


def compute_portfolio_performance_metrics(
    portfolio_returns: Iterable[float],
    benchmark_returns: Iterable[float] | None = None,
    *,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
    turnover: float | None = None,
    assumed_cost_bp: float = 10.0,
) -> dict[str, float]:
    clean = _returns(portfolio_returns)
    dd = drawdown_series(clean)
    durations = drawdown_durations(dd)
    benchmark = _returns(benchmark_returns) if benchmark_returns is not None else pd.Series(dtype=float)
    total_return = float((1.0 + clean).prod() - 1.0) if not clean.empty else np.nan
    annual_turnover = float(turnover) if turnover is not None and pd.notna(turnover) else np.nan
    estimated_cost = annual_turnover * assumed_cost_bp / 10000.0 if pd.notna(annual_turnover) else np.nan
    net_returns = clean - (estimated_cost / periods_per_year if pd.notna(estimated_cost) else 0.0)
    metrics = {
        "annual_return": float(clean.mean() * periods_per_year) if not clean.empty else np.nan,
        "annual_volatility": float(clean.std(ddof=1) * _annualization(periods_per_year)) if len(clean) > 1 else np.nan,
        "sharpe": sharpe_ratio(clean, risk_free_rate, periods_per_year),
        "sortino": sortino_ratio(clean, risk_free_rate, periods_per_year),
        "max_drawdown": max_drawdown(clean),
        "calmar_ratio": calmar_ratio(clean, periods_per_year),
        "omega_ratio": omega_ratio(clean),
        "kappa_ratio": kappa_ratio(clean, periods_per_year=periods_per_year),
        "avg_drawdown": float(dd[dd < 0].mean()) if not dd.empty and (dd < 0).any() else 0.0,
        "avg_drawdown_duration": float(np.mean(durations)) if durations else 0.0,
        "max_drawdown_duration": float(max(durations)) if durations else 0.0,
        "ulcer_index": float(np.sqrt(np.mean(np.square(dd.clip(upper=0))))) if not dd.empty else np.nan,
        "pain_index": float(dd.clip(upper=0).abs().mean()) if not dd.empty else np.nan,
        "recovery_factor": float(abs(total_return) / abs(max_drawdown(clean))) if pd.notna(total_return) and max_drawdown(clean) else np.nan,
        "skewness_returns": float(clean.skew()) if len(clean) > 2 else np.nan,
        "kurtosis_returns": float(clean.kurt()) if len(clean) > 3 else np.nan,
        "var_95": value_at_risk(clean, 0.95),
        "var_99": value_at_risk(clean, 0.99),
        "cvar_95": conditional_var(clean, 0.95),
        "cvar_99": conditional_var(clean, 0.99),
        "tail_ratio": tail_ratio(clean),
        "hit_rate": float((clean > 0).mean()) if not clean.empty else np.nan,
        "avg_up_month": float(clean[clean > 0].mean()) if (clean > 0).any() else np.nan,
        "avg_down_month": float(clean[clean < 0].mean()) if (clean < 0).any() else np.nan,
        "turnover_annual": annual_turnover,
        "avg_holding_period": float(periods_per_year / annual_turnover) if annual_turnover and annual_turnover > 0 else np.nan,
        "estimated_cost_bp": float(estimated_cost * 10000.0) if pd.notna(estimated_cost) else np.nan,
        "net_sharpe": sharpe_ratio(net_returns, risk_free_rate, periods_per_year),
    }
    if not benchmark.empty:
        metrics.update(
            {
                "information_ratio": information_ratio(clean, benchmark, periods_per_year),
                "tracking_error": tracking_error(clean, benchmark, periods_per_year),
                "beta_portfolio": beta_to_benchmark(clean, benchmark),
                "treynor_ratio": treynor_ratio(clean, benchmark, risk_free_rate, periods_per_year),
                "jensen_alpha": jensen_alpha(clean, benchmark, risk_free_rate, periods_per_year),
                "m2_measure": m2_measure(clean, benchmark, risk_free_rate, periods_per_year),
            }
        )
    return metrics


def _regularized_cov(returns_df: pd.DataFrame, lambda_reg: float = 1e-4) -> pd.DataFrame:
    returns = returns_df.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    cov = returns.cov().fillna(0.0) * TRADING_DAYS
    values = cov.to_numpy(dtype=float)
    values = values + float(lambda_reg) * np.eye(values.shape[0])
    return pd.DataFrame(values, index=cov.index, columns=cov.columns)


def _portfolio_stats(weights: np.ndarray, mean_returns: np.ndarray, cov: np.ndarray, risk_free_rate: float = 0.0) -> tuple[float, float, float]:
    ret = float(weights @ mean_returns)
    vol = float(np.sqrt(weights @ cov @ weights))
    sharpe = (ret - risk_free_rate) / vol if vol else np.nan
    return ret, vol, sharpe


def _default_bounds(n: int, constraints: dict | None = None) -> list[tuple[float, float]]:
    max_weight = float((constraints or {}).get("max_weight", 1.0))
    long_only = bool((constraints or {}).get("long_only", True))
    lower = 0.0 if long_only else -max_weight
    return [(lower, max_weight) for _ in range(n)]


def _optimize_weights(returns_df: pd.DataFrame, objective, constraints: dict | None = None) -> np.ndarray:
    n = returns_df.shape[1]
    x0 = np.repeat(1.0 / max(n, 1), n)
    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=_default_bounds(n, constraints),
        constraints=({"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0},),
        options={"maxiter": 500, "ftol": 1e-10},
    )
    return result.x if result.success else x0


def compute_risk_parity_weights(returns_df: pd.DataFrame, cov_matrix: pd.DataFrame | None = None, lambda_reg: float = 1e-4) -> tuple[dict[str, float], pd.DataFrame]:
    clean = returns_df.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    if clean.empty:
        return {}, pd.DataFrame()
    cov = cov_matrix if cov_matrix is not None else _regularized_cov(clean, lambda_reg)
    vols = np.sqrt(np.diag(cov.to_numpy(dtype=float)))
    inv_vol = np.divide(1.0, vols, out=np.zeros_like(vols), where=vols > 0)
    weights = inv_vol / inv_vol.sum() if inv_vol.sum() else np.repeat(1.0 / len(inv_vol), len(inv_vol))
    marginal = cov.to_numpy(dtype=float) @ weights
    portfolio_var = float(weights @ marginal)
    contribution = weights * marginal / portfolio_var if portfolio_var else np.zeros_like(weights)
    weight_dict = dict(zip(clean.columns.astype(str), weights.astype(float)))
    rc = pd.DataFrame({"asset": clean.columns.astype(str), "weight": weights, "risk_contribution": contribution})
    return weight_dict, rc


def compute_min_variance_weights(returns_df: pd.DataFrame, constraints: dict | None = None, lambda_reg: float = 1e-4) -> dict[str, float]:
    clean = returns_df.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    if clean.empty:
        return {}
    cov = _regularized_cov(clean, lambda_reg).to_numpy(dtype=float)
    weights = _optimize_weights(clean, lambda w: float(w @ cov @ w), constraints)
    return dict(zip(clean.columns.astype(str), weights.astype(float)))


def compute_max_sharpe_weights(returns_df: pd.DataFrame, risk_free_rate: float = 0.0, constraints: dict | None = None, lambda_reg: float = 1e-4) -> dict[str, float]:
    clean = returns_df.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    if clean.empty:
        return {}
    mean_returns = clean.mean().to_numpy(dtype=float) * TRADING_DAYS
    cov = _regularized_cov(clean, lambda_reg).to_numpy(dtype=float)
    weights = _optimize_weights(clean, lambda w: -_portfolio_stats(w, mean_returns, cov, risk_free_rate)[2], constraints)
    return dict(zip(clean.columns.astype(str), weights.astype(float)))


def compute_efficient_frontier(returns_df: pd.DataFrame, n_points: int = 50, constraints: dict | None = None, lambda_reg: float = 1e-4) -> pd.DataFrame:
    clean = returns_df.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    if clean.empty:
        return pd.DataFrame()
    mean_returns = clean.mean().to_numpy(dtype=float) * TRADING_DAYS
    cov = _regularized_cov(clean, lambda_reg).to_numpy(dtype=float)
    target_returns = np.linspace(float(np.nanmin(mean_returns)), float(np.nanmax(mean_returns)), int(n_points))
    rows: list[dict[str, float | str]] = []
    n = clean.shape[1]
    x0 = np.repeat(1.0 / n, n)
    for target in target_returns:
        result = minimize(
            lambda w: float(w @ cov @ w),
            x0,
            method="SLSQP",
            bounds=_default_bounds(n, constraints),
            constraints=(
                {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
                {"type": "eq", "fun": lambda w, target=target: float(w @ mean_returns) - target},
            ),
            options={"maxiter": 500, "ftol": 1e-10},
        )
        weights = result.x if result.success else x0
        ret, vol, sharpe = _portfolio_stats(weights, mean_returns, cov)
        rows.append({"expected_return": ret, "volatility": vol, "sharpe": sharpe, "weights": dict(zip(clean.columns.astype(str), weights.astype(float)))})
    return pd.DataFrame(rows)


def compute_brinson_attribution(portfolio: pd.DataFrame, benchmark: pd.DataFrame, sector_col: str = "sector", weight_col: str = "weight", return_col: str = "return") -> pd.DataFrame:
    if portfolio.empty or benchmark.empty:
        return pd.DataFrame()
    p = portfolio.copy()
    b = benchmark.copy()
    for frame in [p, b]:
        frame[weight_col] = pd.to_numeric(frame.get(weight_col, 0.0), errors="coerce").fillna(0.0)
        frame[return_col] = pd.to_numeric(frame.get(return_col, 0.0), errors="coerce").fillna(0.0)
    p_sec = p.groupby(sector_col).apply(lambda g: pd.Series({"w_p": g[weight_col].sum(), "r_p": np.average(g[return_col], weights=g[weight_col]) if g[weight_col].sum() else g[return_col].mean()}))
    b_sec = b.groupby(sector_col).apply(lambda g: pd.Series({"w_b": g[weight_col].sum(), "r_b": np.average(g[return_col], weights=g[weight_col]) if g[weight_col].sum() else g[return_col].mean()}))
    merged = p_sec.merge(b_sec, left_index=True, right_index=True, how="outer").fillna(0.0)
    r_b_total = float((merged["w_b"] * merged["r_b"]).sum())
    merged["allocation_effect"] = (merged["w_p"] - merged["w_b"]) * (merged["r_b"] - r_b_total)
    merged["selection_effect"] = merged["w_b"] * (merged["r_p"] - merged["r_b"])
    merged["interaction_effect"] = (merged["w_p"] - merged["w_b"]) * (merged["r_p"] - merged["r_b"])
    merged["active_return"] = merged["allocation_effect"] + merged["selection_effect"] + merged["interaction_effect"]
    return merged.reset_index().rename(columns={sector_col: "sector"})
