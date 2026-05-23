"""Portable ML4T workflow helpers without legacy backtesting engines."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def make_workflow_panel(periods: int = 756, assets: int = 12, seed: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=periods)
    rows = []
    for asset in range(assets):
        returns = rng.normal(0.0002, 0.012 + asset * 0.0005, periods)
        price = 100 * np.exp(np.cumsum(returns))
        for i, date in enumerate(dates):
            rows.append({"date": date, "ticker": f"asset_{asset:02d}", "close": price[i], "return_1d": returns[i]})
    df = pd.DataFrame(rows)
    group = df.groupby("ticker", group_keys=False)
    df["momentum_21"] = group["close"].pct_change(21)
    df["volatility_21"] = group["return_1d"].rolling(21).std().reset_index(level=0, drop=True)
    df["target_5d"] = group["close"].shift(-5).div(df["close"]).sub(1)
    return df.dropna().reset_index(drop=True)


def vectorized_signal_backtest(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy()
    df["signal"] = df.groupby("date")["momentum_21"].rank(pct=True).gt(0.7).astype(float)
    df["strategy_return"] = df.groupby("ticker")["signal"].shift(1).fillna(0).mul(df["return_1d"])
    return df.groupby("date")["strategy_return"].mean().rename("strategy_return").reset_index()


def walk_forward_ml(panel: pd.DataFrame, train_days: int = 252, test_days: int = 63) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = ["momentum_21", "volatility_21"]
    dates = pd.Index(sorted(panel["date"].unique()))
    predictions = []
    metrics = []
    start = 0
    split = 1
    while start + train_days + test_days <= len(dates):
        train_dates = dates[start : start + train_days]
        test_dates = dates[start + train_days : start + train_days + test_days]
        train = panel[panel["date"].isin(train_dates)]
        test = panel[panel["date"].isin(test_dates)]
        model = RandomForestRegressor(n_estimators=80, min_samples_leaf=10, random_state=split, n_jobs=-1)
        model.fit(train[features], train["target_5d"])
        pred = test[["date", "ticker", "target_5d"] + features].copy()
        pred["prediction"] = model.predict(test[features])
        pred["split"] = split
        predictions.append(pred)
        rmse = float(np.sqrt(mean_squared_error(test["target_5d"], pred["prediction"])))
        metrics.append({"split": split, "rmse": rmse, "train_start": train_dates[0], "test_start": test_dates[0]})
        start += test_days
        split += 1
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(metrics)


def prediction_backtest(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for date, group in predictions.groupby("date"):
        high = group["prediction"].quantile(0.8)
        low = group["prediction"].quantile(0.2)
        rows.append({"date": date, "long_short_return": group.loc[group["prediction"] >= high, "target_5d"].mean() - group.loc[group["prediction"] <= low, "target_5d"].mean()})
    return pd.DataFrame(rows)
