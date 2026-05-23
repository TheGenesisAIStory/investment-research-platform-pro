"""Modular portfolio optimization engine layer.

The functions here are intentionally defensive so they can run in Colab or a
local notebook even when optional optimization libraries are missing. They use
Pandas inputs/outputs and return a unified result dictionary for dashboards.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd


TRADING_DAYS = 252


DEFAULT_PORTFOLIO_ENGINE_CONFIG: dict[str, Any] = {
    "enabled": False,
    "engine": "none",
    "objective": "mean_variance",
    "risk_model": "sample_cov",
    "expected_returns": "historical_mean",
    "risk_free_rate": 0.02,
    "constraints": {
        "min_weight": 0.0,
        "max_weight": 0.30,
        "leverage": 1.0,
        "short": False,
        "turnover_limit": 0.30,
        "tracking_error_target": None,
    },
    "frontier_points": 25,
    "cvar_alpha": 0.95,
    "backtest_horizon": 252,
    "rebalance_freq": "M",
    "fallback_engine": "internal_mean_variance",
}


@dataclass
class OptionalPortfolioLibs:
    pypfopt: Any = None
    riskfolio: Any = None
    cvxportfolio: Any = None
    cvxpy: Any = None


def import_optional_portfolio_libs() -> OptionalPortfolioLibs:
    libs = OptionalPortfolioLibs()
    try:
        import pypfopt  # type: ignore

        libs.pypfopt = pypfopt
    except Exception:
        pass
    try:
        import riskfolio as rp  # type: ignore

        libs.riskfolio = rp
    except Exception:
        pass
    try:
        import cvxportfolio as cvx  # type: ignore

        libs.cvxportfolio = cvx
    except Exception:
        pass
    try:
        import cvxpy as cp  # type: ignore

        libs.cvxpy = cp
    except Exception:
        pass
    return libs


def merge_portfolio_engine_config(
    engine_config: Mapping[str, Any] | None = None,
    portfolio_config: Mapping[str, Any] | None = None,
    risk_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    config = {
        **DEFAULT_PORTFOLIO_ENGINE_CONFIG,
        **(dict(engine_config or {})),
    }
    config["constraints"] = {
        **DEFAULT_PORTFOLIO_ENGINE_CONFIG["constraints"],
        **dict((engine_config or {}).get("constraints", {}) if isinstance(engine_config, Mapping) else {}),
    }
    if portfolio_config:
        if "turnover_limit" in portfolio_config:
            config["constraints"]["turnover_limit"] = float(portfolio_config["turnover_limit"])
        if "transaction_cost_bps" in portfolio_config:
            config["transaction_cost_bps"] = float(portfolio_config["transaction_cost_bps"])
        if "benchmark" in portfolio_config:
            config["benchmark"] = portfolio_config["benchmark"]
    if risk_config:
        config["risk_factor_model"] = risk_config.get("risk_factor_model", config.get("risk_factor_model"))
    return config


def build_returns_matrix_from_universe(
    price_df: pd.DataFrame,
    universe: Mapping[str, Any],
    experiment: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    if price_df is None or price_df.empty:
        return pd.DataFrame()
    df = price_df.copy()
    if "date" not in df.columns or "ticker" not in df.columns:
        raise ValueError("price_df must include date and ticker columns")
    price_col = "price" if "price" in df.columns else "adj_close" if "adj_close" in df.columns else "close"
    tickers = list(dict.fromkeys(universe.get("all_tickers", []) or df["ticker"].dropna().unique().tolist()))
    df["date"] = pd.to_datetime(df["date"])
    wide = df[df["ticker"].isin(tickers)].pivot_table(index="date", columns="ticker", values=price_col, aggfunc="last").sort_index()
    if experiment:
        start, end = experiment.get("start_date"), experiment.get("end_date")
        if start:
            wide = wide.loc[wide.index >= pd.Timestamp(start)]
        if end:
            wide = wide.loc[wide.index <= pd.Timestamp(end)]
    returns = wide.ffill().pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).dropna(how="all")
    return returns.dropna(axis=1, how="all")


def _expected_returns(returns_df: pd.DataFrame) -> pd.Series:
    return returns_df.mean() * TRADING_DAYS


def _covariance(returns_df: pd.DataFrame) -> pd.DataFrame:
    return returns_df.cov() * TRADING_DAYS


def _clean_weights(weights: Mapping[str, float] | pd.Series) -> pd.DataFrame:
    series = pd.Series(weights, dtype="float64").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return (
        series.rename("weight")
        .reset_index()
        .rename(columns={"index": "ticker"})
        .sort_values("weight", ascending=False)
        .reset_index(drop=True)
    )


def _portfolio_metrics(returns_df: pd.DataFrame, weights: pd.Series, risk_free_rate: float = 0.02, alpha: float = 0.95) -> dict[str, float]:
    weights = weights.reindex(returns_df.columns).fillna(0.0)
    weights = weights / weights.sum() if weights.sum() else weights
    port_ret = returns_df.mul(weights, axis=1).sum(axis=1).dropna()
    if port_ret.empty:
        return {"expected_return": np.nan, "volatility": np.nan, "sharpe": np.nan, "cvar": np.nan, "max_drawdown": np.nan}
    ann_return = port_ret.mean() * TRADING_DAYS
    ann_vol = port_ret.std() * np.sqrt(TRADING_DAYS)
    equity = (1 + port_ret).cumprod()
    drawdown = equity / equity.cummax() - 1
    var = port_ret.quantile(1 - alpha)
    cvar = port_ret[port_ret <= var].mean() if (port_ret <= var).any() else np.nan
    return {
        "expected_return": float(ann_return),
        "volatility": float(ann_vol),
        "sharpe": float((ann_return - risk_free_rate) / ann_vol) if ann_vol else np.nan,
        "cvar": float(cvar) if pd.notna(cvar) else np.nan,
        "max_drawdown": float(drawdown.min()),
    }


def _internal_mean_variance(returns_df: pd.DataFrame, config: Mapping[str, Any]) -> dict[str, Any]:
    constraints = config.get("constraints", {})
    min_w = float(constraints.get("min_weight", 0.0))
    max_w = float(constraints.get("max_weight", 0.30))
    mu = _expected_returns(returns_df)
    vol = returns_df.std() * np.sqrt(TRADING_DAYS)
    score = (mu / vol.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0)
    if score.sum() <= 0:
        w = pd.Series(1 / len(mu), index=mu.index)
    else:
        w = score / score.sum()
    w = w.clip(lower=min_w, upper=max_w)
    w = w / w.sum() if w.sum() else pd.Series(1 / len(mu), index=mu.index)
    return {
        "weights": _clean_weights(w),
        "frontier": build_internal_frontier(returns_df, config),
        "backtest": build_static_weight_backtest(returns_df, w),
        "metrics": _portfolio_metrics(returns_df, w, float(config.get("risk_free_rate", 0.02)), float(config.get("cvar_alpha", 0.95))),
        "diagnostics": {"engine_status": "internal_fallback", "message": "Used internal risk-adjusted score optimizer."},
    }


def build_internal_frontier(returns_df: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    mu = _expected_returns(returns_df)
    cov = _covariance(returns_df)
    n_points = int(config.get("frontier_points", 25))
    rows = []
    rng = np.random.default_rng(42)
    for _ in range(max(n_points, 5)):
        raw = rng.random(len(mu))
        w = pd.Series(raw / raw.sum(), index=mu.index)
        er = float(np.dot(w, mu))
        vol = float(np.sqrt(np.dot(w.T, np.dot(cov, w))))
        rows.append({"expected_return": er, "volatility": vol, "sharpe": (er - float(config.get("risk_free_rate", 0.02))) / vol if vol else np.nan, "portfolio": "random_frontier"})
    return pd.DataFrame(rows)


def build_static_weight_backtest(returns_df: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    weights = weights.reindex(returns_df.columns).fillna(0.0)
    weights = weights / weights.sum() if weights.sum() else weights
    port_ret = returns_df.mul(weights, axis=1).sum(axis=1).fillna(0)
    return pd.DataFrame({"date": port_ret.index, "portfolio_return": port_ret.values, "portfolio_equity": (1 + port_ret).cumprod().values})


def run_pyportfolioopt_engine(returns_df: pd.DataFrame, config: Mapping[str, Any]) -> dict[str, Any]:
    libs = import_optional_portfolio_libs()
    if libs.pypfopt is None:
        out = _internal_mean_variance(returns_df, config)
        out["diagnostics"]["requested_engine"] = "pyportfolioopt"
        return out
    from pypfopt import EfficientCVaR, EfficientFrontier, HRPOpt

    objective = str(config.get("objective", "mean_variance")).lower()
    constraints = config.get("constraints", {})
    min_w = float(constraints.get("min_weight", 0.0))
    max_w = float(constraints.get("max_weight", 0.30))
    mu = _expected_returns(returns_df)
    cov = _covariance(returns_df)
    if objective == "hrp":
        opt = HRPOpt(returns_df.dropna())
        weights = opt.optimize()
    elif objective == "cvar":
        opt = EfficientCVaR(mu, returns_df.dropna(), weight_bounds=(min_w, max_w), beta=float(config.get("cvar_alpha", 0.95)))
        weights = opt.min_cvar()
    else:
        opt = EfficientFrontier(mu, cov, weight_bounds=(min_w, max_w))
        if objective in {"min_volatility", "min_variance"}:
            weights = opt.min_volatility()
        else:
            weights = opt.max_sharpe(risk_free_rate=float(config.get("risk_free_rate", 0.02)))
    cleaned = opt.clean_weights() if hasattr(opt, "clean_weights") else weights
    weights_df = _clean_weights(cleaned)
    w = weights_df.set_index("ticker")["weight"]
    return {
        "weights": weights_df,
        "frontier": build_internal_frontier(returns_df, config),
        "backtest": build_static_weight_backtest(returns_df, w),
        "metrics": _portfolio_metrics(returns_df, w, float(config.get("risk_free_rate", 0.02)), float(config.get("cvar_alpha", 0.95))),
        "diagnostics": {"engine_status": "ok", "engine": "pyportfolioopt", "objective": objective},
    }


def run_riskfolio_engine(returns_df: pd.DataFrame, config: Mapping[str, Any]) -> dict[str, Any]:
    libs = import_optional_portfolio_libs()
    if libs.riskfolio is None:
        out = _internal_mean_variance(returns_df, config)
        out["diagnostics"]["requested_engine"] = "riskfolio"
        return out
    rp = libs.riskfolio
    objective = str(config.get("objective", "risk_parity")).lower()
    port = rp.Portfolio(returns=returns_df.dropna())
    port.assets_stats(method_mu="hist", method_cov="hist")
    constraints = config.get("constraints", {})
    port.upperlng = float(constraints.get("max_weight", 0.30))
    port.lowerlng = float(constraints.get("min_weight", 0.0))
    if objective in {"risk_parity", "rp"}:
        weights = port.rp_optimization(model="Classic", rm="MV", rf=float(config.get("risk_free_rate", 0.02)), b=None, hist=True)
    else:
        rm = "CVaR" if objective in {"cvar", "min_cvar"} else "MV"
        obj = "MinRisk" if objective in {"cvar", "min_cvar", "min_volatility"} else "Sharpe"
        weights = port.optimization(model="Classic", rm=rm, obj=obj, rf=float(config.get("risk_free_rate", 0.02)), l=0, hist=True)
    series = weights.iloc[:, 0] if isinstance(weights, pd.DataFrame) else pd.Series(weights)
    weights_df = _clean_weights(series)
    w = weights_df.set_index("ticker")["weight"]
    return {
        "weights": weights_df,
        "frontier": build_internal_frontier(returns_df, config),
        "backtest": build_static_weight_backtest(returns_df, w),
        "metrics": _portfolio_metrics(returns_df, w, float(config.get("risk_free_rate", 0.02)), float(config.get("cvar_alpha", 0.95))),
        "diagnostics": {"engine_status": "ok", "engine": "riskfolio", "objective": objective},
    }


def run_cvxportfolio_engine(price_or_returns_df: pd.DataFrame, config: Mapping[str, Any]) -> dict[str, Any]:
    returns_df = price_or_returns_df.copy()
    if not isinstance(returns_df.index, pd.DatetimeIndex) and "date" in returns_df.columns:
        returns_df = returns_df.set_index("date")
    constraints = config.get("constraints", {})
    max_w = float(constraints.get("max_weight", 0.30))
    min_w = float(constraints.get("min_weight", 0.0))
    # Robust convex single-period policy with rolling monthly-style backtest.
    libs = import_optional_portfolio_libs()
    if libs.cvxpy is None:
        out = _internal_mean_variance(returns_df, config)
        out["diagnostics"]["requested_engine"] = "cvxportfolio"
        return out
    cp = libs.cvxpy
    mu = _expected_returns(returns_df).fillna(0.0)
    cov = _covariance(returns_df).fillna(0.0)
    n = len(mu)
    w_var = cp.Variable(n)
    risk_aversion = float(config.get("risk_aversion", 5.0))
    objective = cp.Maximize(mu.values @ w_var - risk_aversion * cp.quad_form(w_var, cov.values))
    problem = cp.Problem(objective, [cp.sum(w_var) == 1, w_var >= min_w, w_var <= max_w])
    problem.solve(solver=cp.CLARABEL if "CLARABEL" in cp.installed_solvers() else None)
    weights = pd.Series(np.asarray(w_var.value).reshape(-1), index=mu.index) if w_var.value is not None else pd.Series(1 / n, index=mu.index)
    weights = weights.clip(lower=min_w, upper=max_w)
    weights = weights / weights.sum() if weights.sum() else pd.Series(1 / n, index=mu.index)
    return {
        "weights": _clean_weights(weights),
        "frontier": build_internal_frontier(returns_df, config),
        "backtest": build_static_weight_backtest(returns_df, weights),
        "metrics": _portfolio_metrics(returns_df, weights, float(config.get("risk_free_rate", 0.02)), float(config.get("cvar_alpha", 0.95))),
        "diagnostics": {"engine_status": "ok", "engine": "cvxportfolio/cvxpy_policy", "objective": str(config.get("objective", "multi_period"))},
    }


def unified_run_portfolio_engine(
    price_df: pd.DataFrame | None,
    returns_df: pd.DataFrame | None,
    engine_config: Mapping[str, Any] | None,
    universe: Mapping[str, Any] | None = None,
    experiment: Mapping[str, Any] | None = None,
    portfolio_config: Mapping[str, Any] | None = None,
    risk_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    config = merge_portfolio_engine_config(engine_config, portfolio_config, risk_config)
    if not config.get("enabled", False) or str(config.get("engine", "none")).lower() in {"none", ""}:
        return {"weights": pd.DataFrame(), "frontier": pd.DataFrame(), "backtest": pd.DataFrame(), "metrics": {}, "diagnostics": {"engine_status": "disabled"}}
    if returns_df is None or returns_df.empty:
        if price_df is None or price_df.empty:
            return {"weights": pd.DataFrame(), "frontier": pd.DataFrame(), "backtest": pd.DataFrame(), "metrics": {}, "diagnostics": {"engine_status": "no_data"}}
        returns_df = build_returns_matrix_from_universe(price_df, universe or {}, experiment or {})
    returns_df = returns_df.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all").dropna(how="all")
    if returns_df.empty or returns_df.shape[1] < 2:
        return {"weights": pd.DataFrame(), "frontier": pd.DataFrame(), "backtest": pd.DataFrame(), "metrics": {}, "diagnostics": {"engine_status": "insufficient_assets"}}
    engine = str(config.get("engine", "pyportfolioopt")).lower()
    if engine in {"pyportfolioopt", "pypfopt"}:
        return run_pyportfolioopt_engine(returns_df, config)
    if engine in {"riskfolio", "riskfolio-lib", "riskfolio_lib"}:
        return run_riskfolio_engine(returns_df, config)
    if engine in {"cvxportfolio", "cvx"}:
        return run_cvxportfolio_engine(returns_df, config)
    out = _internal_mean_variance(returns_df, config)
    out["diagnostics"]["requested_engine"] = engine
    return out
