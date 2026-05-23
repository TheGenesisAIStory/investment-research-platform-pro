"""Portable linear-model helpers for the ML4T linear models chapter."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression


def regression_data(n_samples: int = 900, n_features: int = 10, seed: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    x, y = make_regression(n_samples=n_samples, n_features=n_features, n_informative=5, noise=10.0, random_state=seed)
    return pd.DataFrame(x, columns=[f"factor_{i:02d}" for i in range(n_features)]), pd.Series(y, name="target")


def classification_data(n_samples: int = 900, n_features: int = 10, seed: int = 21) -> tuple[pd.DataFrame, pd.Series]:
    x, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=5, n_redundant=2, random_state=seed)
    return pd.DataFrame(x, columns=[f"factor_{i:02d}" for i in range(n_features)]), pd.Series(y, name="target")


def panel_factor_data(periods: int = 160, assets: int = 30, factors: int = 5, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=periods)
    rows = []
    betas = rng.normal(0, 0.4, size=(assets, factors))
    for date in dates:
        factor_values = rng.normal(0, 1, factors)
        noise = rng.normal(0, 0.03, assets)
        returns = betas @ factor_values * 0.01 + noise
        for asset in range(assets):
            row = {"date": date, "ticker": f"stk_{asset:03d}", "return_1d": returns[asset]}
            row.update({f"factor_{i:02d}": betas[asset, i] + rng.normal(0, 0.05) for i in range(factors)})
            rows.append(row)
    return pd.DataFrame(rows)


def ols_numpy(x: pd.DataFrame, y: pd.Series) -> pd.Series:
    design = np.column_stack([np.ones(len(x)), x.to_numpy()])
    coefs = np.linalg.pinv(design) @ y.to_numpy()
    return pd.Series(coefs, index=["intercept", *x.columns], name="coef")


def rank_ic(panel: pd.DataFrame, factor: str, target: str = "return_1d") -> pd.DataFrame:
    rows = []
    for date, group in panel.groupby("date"):
        subset = group[[factor, target]].dropna()
        if len(subset) >= 5:
            rows.append({"date": date, "factor": factor, "ic": subset[factor].corr(subset[target], method="spearman")})
    return pd.DataFrame(rows)
