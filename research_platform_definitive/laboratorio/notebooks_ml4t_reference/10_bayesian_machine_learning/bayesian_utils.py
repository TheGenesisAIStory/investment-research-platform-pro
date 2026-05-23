"""Modern Bayesian workflow helpers for chapter 10 notebooks.

The original notebooks relied on PyMC3/Theano and remote data readers that are
fragile on current Python versions. These helpers keep the chapter executable by
using conjugate updates, simulation, and deterministic numpy/pandas workflows.
They are intentionally lightweight and suitable for CI/offline execution.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CredibleInterval:
    mean: float
    lower: float
    upper: float


def ensure_output_dir(path: str | Path = "../data/bayesian_machine_learning") -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def simulate_strategy_returns(periods: int = 1_250, seed: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    market = rng.normal(0.00035, 0.011, periods)
    quality = 0.00018 + 0.55 * market + rng.normal(0, 0.0075, periods)
    momentum = 0.00028 + 0.85 * market + rng.normal(0, 0.0105, periods)
    defensive = 0.00012 + 0.30 * market + rng.normal(0, 0.0055, periods)
    end = pd.Timestamp.today().normalize()
    if end.dayofweek >= 5:
        end = end - pd.offsets.BDay(1)
    idx = pd.bdate_range(end=end, periods=periods)
    return pd.DataFrame(
        {
            "market": market,
            "quality": quality,
            "momentum": momentum,
            "defensive": defensive,
        },
        index=idx,
    )


def beta_binomial_updates(observations: Iterable[int], alpha: float = 1.0, beta: float = 1.0) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    a, b = float(alpha), float(beta)
    for step, obs in enumerate(observations, start=1):
        if obs not in (0, 1):
            raise ValueError("observations must contain 0/1 outcomes")
        a += obs
        b += 1 - obs
        rows.append(
            {
                "step": step,
                "observation": int(obs),
                "alpha": a,
                "beta": b,
                "posterior_mean": a / (a + b),
                "posterior_std": np.sqrt((a * b) / (((a + b) ** 2) * (a + b + 1))),
            }
        )
    return pd.DataFrame(rows)


def normal_mean_posterior(
    values: pd.Series | np.ndarray,
    prior_mean: float = 0.0,
    prior_std: float = 0.02,
    observation_std: float | None = None,
) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError("values must contain at least one finite observation")
    sigma = float(observation_std or arr.std(ddof=1) or 1e-6)
    prior_var = prior_std**2
    obs_var = sigma**2
    posterior_var = 1.0 / (1.0 / prior_var + arr.size / obs_var)
    posterior_mean = posterior_var * (prior_mean / prior_var + arr.sum() / obs_var)
    return {
        "posterior_mean": float(posterior_mean),
        "posterior_std": float(np.sqrt(posterior_var)),
        "sample_mean": float(arr.mean()),
        "sample_std": sigma,
        "n": int(arr.size),
    }


def sample_posterior_normal(summary: dict[str, float], draws: int = 20_000, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(summary["posterior_mean"], summary["posterior_std"], draws)


def credible_interval(samples: np.ndarray, level: float = 0.95) -> CredibleInterval:
    alpha = (1 - level) / 2
    return CredibleInterval(
        mean=float(np.mean(samples)),
        lower=float(np.quantile(samples, alpha)),
        upper=float(np.quantile(samples, 1 - alpha)),
    )


def bayesian_sharpe_samples(
    returns: pd.Series,
    draws: int = 20_000,
    seed: int = 21,
    annualization: int = 252,
) -> np.ndarray:
    arr = returns.dropna().to_numpy(dtype=float)
    if arr.size < 5:
        raise ValueError("returns need at least five observations")
    rng = np.random.default_rng(seed)
    mean_samples = rng.normal(arr.mean(), arr.std(ddof=1) / np.sqrt(arr.size), draws)
    # Inverse-chi-square style volatility approximation with finite fallback.
    vol_samples = arr.std(ddof=1) * np.sqrt((arr.size - 1) / rng.chisquare(max(arr.size - 1, 1), draws))
    return np.sqrt(annualization) * mean_samples / np.maximum(vol_samples, 1e-10)


def compare_bayesian_sharpe(returns: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    selected = columns or list(returns.columns)
    rows = []
    for i, column in enumerate(selected):
        samples = bayesian_sharpe_samples(returns[column], seed=100 + i)
        ci = credible_interval(samples)
        rows.append({"strategy": column, "mean": ci.mean, "lower_95": ci.lower, "upper_95": ci.upper})
    return pd.DataFrame(rows).sort_values("mean", ascending=False).reset_index(drop=True)


def rolling_regression(y: pd.Series, x: pd.Series, window: int = 126) -> pd.DataFrame:
    aligned = pd.concat({"y": y, "x": x}, axis=1).dropna()
    rows: list[dict[str, float | pd.Timestamp]] = []
    for end in range(window, len(aligned) + 1):
        frame = aligned.iloc[end - window : end]
        X = np.column_stack([np.ones(len(frame)), frame["x"].to_numpy()])
        coef = np.linalg.pinv(X.T @ X) @ X.T @ frame["y"].to_numpy()
        residual = frame["y"].to_numpy() - X @ coef
        rows.append(
            {
                "date": frame.index[-1],
                "alpha": float(coef[0]),
                "beta": float(coef[1]),
                "residual_vol": float(residual.std(ddof=1)),
            }
        )
    return pd.DataFrame(rows).set_index("date")


def stochastic_volatility_proxy(returns: pd.Series, span: int = 30) -> pd.DataFrame:
    clean = returns.dropna()
    realized = clean.rolling(21).std() * np.sqrt(252)
    latent = clean.pow(2).ewm(span=span, adjust=False).mean().pow(0.5) * np.sqrt(252)
    shock = clean.abs().rolling(21).mean()
    return pd.DataFrame({"return": clean, "realized_vol": realized, "latent_vol_proxy": latent, "shock_proxy": shock})
