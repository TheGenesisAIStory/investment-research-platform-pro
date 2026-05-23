"""Portable time-series helpers without statsmodels/pandas-datareader."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_time_series(periods: int = 1000, seed: int = 91) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=periods)
    shock = rng.normal(0, 0.01, periods)
    ar = np.zeros(periods)
    for i in range(1, periods):
        ar[i] = 0.08 + 0.82 * ar[i - 1] + shock[i]
    random_walk = 100 + np.cumsum(rng.normal(0.02, 1.0, periods))
    trend = 50 + np.linspace(0, 8, periods) + rng.normal(0, 0.5, periods)
    return pd.DataFrame({"date": dates, "stationary_ar": ar, "random_walk": random_walk, "trend": trend})


def make_pair_series(periods: int = 1000, seed: int = 17) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=periods)
    x = 100 + np.cumsum(rng.normal(0.02, 1.0, periods))
    spread = rng.normal(0, 1.0, periods)
    y = 5 + 1.4 * x + spread
    return pd.DataFrame({"date": dates, "asset_x": x, "asset_y": y, "spread": y - 1.4 * x})


def stationarity_summary(series: pd.Series) -> dict[str, float]:
    diff = series.diff().dropna()
    return {"mean": float(series.mean()), "std": float(series.std()), "lag1_autocorr": float(series.autocorr(1)), "diff_std_ratio": float(diff.std() / series.std()) if series.std() else 0.0}


def fit_ar(series: pd.Series, lags: int = 3) -> tuple[pd.Series, pd.Series]:
    data = pd.concat({f"lag_{i}": series.shift(i) for i in range(1, lags + 1)}, axis=1).dropna()
    y = series.loc[data.index]
    design = np.column_stack([np.ones(len(data)), data.to_numpy()])
    coef = np.linalg.pinv(design) @ y.to_numpy()
    fitted = pd.Series(design @ coef, index=data.index, name="fitted")
    return pd.Series(coef, index=["intercept", *data.columns], name="coef"), fitted


def ewma_volatility(returns: pd.Series, span: int = 30) -> pd.Series:
    return returns.pow(2).ewm(span=span).mean().pow(0.5).rename("ewma_volatility")


def fit_var_two(series: pd.DataFrame, lags: int = 1) -> pd.DataFrame:
    cols = list(series.columns)
    frames = []
    for target in cols:
        x = pd.concat({f"{col}_lag{lag}": series[col].shift(lag) for col in cols for lag in range(1, lags + 1)}, axis=1).dropna()
        y = series[target].loc[x.index]
        design = np.column_stack([np.ones(len(x)), x.to_numpy()])
        coef = np.linalg.pinv(design) @ y.to_numpy()
        frames.append(pd.Series(coef, index=["intercept", *x.columns], name=target))
    return pd.concat(frames, axis=1)


def hedge_ratio(x: pd.Series, y: pd.Series) -> float:
    design = np.column_stack([np.ones(len(x)), x.to_numpy()])
    coef = np.linalg.pinv(design) @ y.to_numpy()
    return float(coef[1])


def pairs_backtest(pair: pd.DataFrame, entry_z: float = 1.5, exit_z: float = 0.25) -> pd.DataFrame:
    beta = hedge_ratio(pair["asset_x"], pair["asset_y"])
    spread = pair["asset_y"] - beta * pair["asset_x"]
    z = (spread - spread.rolling(60).mean()) / spread.rolling(60).std()
    position = pd.Series(0.0, index=pair.index)
    active = 0.0
    for i, value in enumerate(z.fillna(0)):
        if active == 0 and value > entry_z:
            active = -1
        elif active == 0 and value < -entry_z:
            active = 1
        elif active != 0 and abs(value) < exit_z:
            active = 0
        position.iloc[i] = active
    spread_return = spread.diff().fillna(0)
    strategy_return = position.shift(1).fillna(0) * spread_return
    return pd.DataFrame({"date": pair["date"], "spread": spread, "z_score": z, "position": position, "strategy_return": strategy_return})
