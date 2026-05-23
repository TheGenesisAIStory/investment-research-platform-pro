"""Modern tree-based model helpers for chapter 11 notebooks."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def ensure_output_dir(path: str | Path = "../data/tree_models") -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def make_equity_panel(n_assets: int = 80, periods: int = 756, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp("2026-05-08"), periods=periods)
    tickers = [f"EQ{i:03d}" for i in range(n_assets)]
    rows = []
    market = rng.normal(0.00025, 0.010, periods)
    sector_beta = rng.normal(1.0, 0.25, n_assets)
    quality = rng.normal(0, 1, n_assets)
    for i, ticker in enumerate(tickers):
        noise = rng.normal(0, 0.012, periods)
        ret = 0.00005 + sector_beta[i] * market + 0.00018 * quality[i] + noise
        price = 50 * np.exp(np.cumsum(ret))
        df = pd.DataFrame({"date": dates, "ticker": ticker, "return_1d": ret, "close": price})
        df["sector_beta"] = sector_beta[i]
        df["quality"] = quality[i]
        rows.append(df)
    panel = pd.concat(rows, ignore_index=True).sort_values(["ticker", "date"])
    panel["momentum_21"] = panel.groupby("ticker")["close"].pct_change(21)
    panel["momentum_63"] = panel.groupby("ticker")["close"].pct_change(63)
    panel["volatility_21"] = panel.groupby("ticker")["return_1d"].rolling(21).std().reset_index(level=0, drop=True)
    panel["reversal_5"] = -panel.groupby("ticker")["close"].pct_change(5)
    panel["target_5d"] = panel.groupby("ticker")["close"].pct_change(5).shift(-5)
    panel["target_positive"] = (panel["target_5d"] > 0).astype(int)
    return panel.dropna().reset_index(drop=True)


FEATURES = ["momentum_21", "momentum_63", "volatility_21", "reversal_5", "sector_beta", "quality"]


def rmse(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def time_split(panel: pd.DataFrame, train_fraction: float = 0.75) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = panel["date"].quantile(train_fraction)
    train = panel[panel["date"] <= cutoff].copy()
    test = panel[panel["date"] > cutoff].copy()
    return train, test


def fit_decision_tree(panel: pd.DataFrame) -> tuple[DecisionTreeRegressor, dict[str, float]]:
    train, test = time_split(panel)
    model = DecisionTreeRegressor(max_depth=4, min_samples_leaf=50, random_state=42)
    model.fit(train[FEATURES], train["target_5d"])
    pred = model.predict(test[FEATURES])
    metrics = {"test_rmse": rmse(test["target_5d"], pred)}
    return model, metrics


def compare_bagging(panel: pd.DataFrame) -> pd.DataFrame:
    train, test = time_split(panel)
    base = DecisionTreeRegressor(max_depth=None, min_samples_leaf=20, random_state=42)
    bag = BaggingRegressor(estimator=base, n_estimators=50, random_state=42, n_jobs=-1)
    shallow = DecisionTreeRegressor(max_depth=4, min_samples_leaf=50, random_state=42)
    rows = []
    for name, model in {"single_tree": shallow, "bagging": bag}.items():
        model.fit(train[FEATURES], train["target_5d"])
        pred = model.predict(test[FEATURES])
        rows.append({"model": name, "rmse": rmse(test["target_5d"], pred)})
    return pd.DataFrame(rows).sort_values("rmse")


def tune_random_forest(panel: pd.DataFrame) -> pd.DataFrame:
    train, _ = time_split(panel)
    configs = []
    tscv = TimeSeriesSplit(n_splits=3)
    for depth in [3, 5, None]:
        for leaf in [25, 75]:
            model = RandomForestRegressor(
                n_estimators=80,
                max_depth=depth,
                min_samples_leaf=leaf,
                random_state=42,
                n_jobs=-1,
            )
            scores = -cross_val_score(
                model, train[FEATURES], train["target_5d"], cv=tscv, scoring="neg_root_mean_squared_error", n_jobs=-1
            )
            configs.append({"max_depth": str(depth), "min_samples_leaf": leaf, "cv_rmse": scores.mean()})
    return pd.DataFrame(configs).sort_values("cv_rmse")


def fit_signal_model(panel: pd.DataFrame) -> tuple[RandomForestClassifier, pd.DataFrame, pd.DataFrame]:
    train, test = time_split(panel)
    model = RandomForestClassifier(
        n_estimators=120,
        max_depth=5,
        min_samples_leaf=50,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    model.fit(train[FEATURES], train["target_positive"])
    scored = test.copy()
    scored["score"] = model.predict_proba(test[FEATURES])[:, 1]
    scored["prediction"] = (scored["score"] > 0.5).astype(int)
    diagnostics = pd.DataFrame(
        {
            "metric": ["accuracy", "roc_auc"],
            "value": [
                accuracy_score(scored["target_positive"], scored["prediction"]),
                roc_auc_score(scored["target_positive"], scored["score"]),
            ],
        }
    )
    importance = pd.DataFrame({"feature": FEATURES, "importance": model.feature_importances_}).sort_values(
        "importance", ascending=False
    )
    return model, scored, diagnostics


def signal_quality(scored: pd.DataFrame, buckets: int = 5) -> pd.DataFrame:
    frame = scored.copy()
    frame["bucket"] = frame.groupby("date")["score"].transform(
        lambda s: pd.qcut(s.rank(method="first"), buckets, labels=False, duplicates="drop") + 1
    )
    return frame.groupby("bucket")["target_5d"].agg(["mean", "std", "count"]).reset_index()


def long_short_backtest(scored: pd.DataFrame, quantile: float = 0.2) -> pd.DataFrame:
    rows = []
    for date, frame in scored.groupby("date"):
        low = frame["score"].quantile(quantile)
        high = frame["score"].quantile(1 - quantile)
        long_ret = frame.loc[frame["score"] >= high, "target_5d"].mean()
        short_ret = frame.loc[frame["score"] <= low, "target_5d"].mean()
        rows.append({"date": date, "strategy_return": long_ret - short_ret})
    result = pd.DataFrame(rows).dropna().set_index("date")
    result["equity_curve"] = (1 + result["strategy_return"]).cumprod()
    return result
