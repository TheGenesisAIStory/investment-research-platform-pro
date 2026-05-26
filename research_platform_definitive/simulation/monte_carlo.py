"""Vectorized Monte Carlo simulation helpers.

References: Glasserman (2003), Monte Carlo Methods in Financial Engineering.
The functions are deterministic by default (`seed=42`) and avoid any dependency
on portfolio-specific UI state.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _as_numeric_frame(frame: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    if isinstance(frame, pd.DataFrame):
        return frame.apply(pd.to_numeric, errors="coerce")
    return pd.DataFrame(np.asarray(frame, dtype=float))


def monte_carlo_returns(
    factor_scores: pd.DataFrame | np.ndarray,
    n_sim: int = 10_000,
    horizon: int = 252,
    seed: int = 42,
) -> dict[str, object]:
    """Simulate portfolio return paths from factor-implied returns and covariance.

    Parameters
    ----------
    factor_scores:
        Numeric matrix where rows are observations and columns are assets,
        factor portfolios or model score return proxies.
    n_sim:
        Number of paths.
    horizon:
        Number of simulated periods.
    seed:
        Reproducibility seed.

    Returns
    -------
    dict with percentile paths, terminal percentiles, VaR and CVaR metrics.
    """
    frame = _as_numeric_frame(factor_scores).dropna(how="all")
    if frame.empty:
        return {
            "paths": pd.DataFrame(),
            "terminal_percentiles": {},
            "var_95": np.nan,
            "cvar_95": np.nan,
            "var_99": np.nan,
            "cvar_99": np.nan,
        }
    looks_like_returns = frame.abs().quantile(0.95, numeric_only=True).max() < 1.0
    returns = frame.copy() if bool(looks_like_returns) else frame.pct_change().replace([np.inf, -np.inf], np.nan)
    if returns.notna().sum().sum() < max(5, frame.shape[1] + 1):
        returns = frame.diff().replace([np.inf, -np.inf], np.nan)
    returns = returns.dropna(how="all").fillna(0.0)
    if returns.empty:
        returns = pd.DataFrame(np.zeros((1, frame.shape[1])), columns=frame.columns)

    mu = returns.mean().to_numpy(dtype=float)
    cov = returns.cov().to_numpy(dtype=float)
    cov = np.atleast_2d(cov)
    if cov.shape != (len(mu), len(mu)):
        cov = np.eye(len(mu)) * float(np.nanvar(returns.to_numpy(dtype=float)))
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
    cov = cov + np.eye(len(mu)) * 1e-10

    rng = np.random.default_rng(seed)
    shocks = rng.multivariate_normal(mu, cov, size=(int(n_sim), int(horizon)), method="svd")
    equal_weight_returns = shocks.mean(axis=2)
    paths = np.cumprod(1.0 + equal_weight_returns, axis=1) - 1.0
    bands = np.percentile(paths, [5, 25, 50, 75, 95], axis=0)
    paths_df = pd.DataFrame(
        bands.T,
        columns=["p05", "p25", "p50", "p75", "p95"],
        index=pd.RangeIndex(1, int(horizon) + 1, name="period"),
    )
    terminal = paths[:, -1]
    var_95 = float(np.percentile(terminal, 5))
    var_99 = float(np.percentile(terminal, 1))
    cvar_95 = float(terminal[terminal <= var_95].mean()) if np.any(terminal <= var_95) else np.nan
    cvar_99 = float(terminal[terminal <= var_99].mean()) if np.any(terminal <= var_99) else np.nan
    return {
        "paths": paths_df,
        "terminal_percentiles": {
            "p05": float(np.percentile(terminal, 5)),
            "p25": float(np.percentile(terminal, 25)),
            "p50": float(np.percentile(terminal, 50)),
            "p75": float(np.percentile(terminal, 75)),
            "p95": float(np.percentile(terminal, 95)),
        },
        "var_95": var_95,
        "cvar_95": cvar_95,
        "var_99": var_99,
        "cvar_99": cvar_99,
    }


def monte_carlo_factor_uncertainty(
    factor_df: pd.DataFrame | np.ndarray,
    bootstrap: bool = True,
    n_sim: int = 10_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Estimate factor prediction intervals by resampling observations.

    If `bootstrap=True`, rows are sampled with replacement.  Otherwise a
    Gaussian approximation around each factor mean is used.  Output rows are
    factors and columns are 5/25/50/75/95 percentiles of the simulated mean.
    """
    frame = _as_numeric_frame(factor_df).replace([np.inf, -np.inf], np.nan).dropna(how="all")
    if frame.empty:
        return pd.DataFrame(columns=["p05", "p25", "p50", "p75", "p95"])
    frame = frame.fillna(frame.mean(numeric_only=True)).fillna(0.0)
    values = frame.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    if bootstrap:
        sample_idx = rng.integers(0, len(frame), size=(int(n_sim), len(frame)))
        simulated = values[sample_idx].mean(axis=1)
    else:
        mu = values.mean(axis=0)
        sigma = values.std(axis=0, ddof=1) / np.sqrt(max(len(frame), 1))
        simulated = rng.normal(mu, sigma, size=(int(n_sim), values.shape[1]))
    percentiles = np.percentile(simulated, [5, 25, 50, 75, 95], axis=0).T
    return pd.DataFrame(percentiles, index=frame.columns, columns=["p05", "p25", "p50", "p75", "p95"])


__all__ = ["monte_carlo_factor_uncertainty", "monte_carlo_returns"]
