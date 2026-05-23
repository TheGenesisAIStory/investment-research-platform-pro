"""Modern gradient boosting helpers for chapter 12 notebooks."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["momentum_10", "momentum_21", "momentum_63", "volatility_21", "volume_z", "quality", "value"]


def ensure_output_dir(path: str | Path = "../data/gradient_boosting") -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def make_boosting_panel(n_assets: int = 100, periods: int = 756, seed: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp("2026-05-08"), periods=periods)
    market = rng.normal(0.0002, 0.009, periods)
    rows = []
    for i in range(n_assets):
        ticker = f"GBM{i:03d}"
        quality = rng.normal()
        value = rng.normal()
        beta = rng.normal(1.0, 0.2)
        volume = np.exp(rng.normal(12, 0.35, periods))
        eps = rng.normal(0, 0.012, periods)
        ret = 0.00005 + beta * market + 0.00015 * quality - 0.00010 * value + eps
        close = 40 * np.exp(np.cumsum(ret))
        frame = pd.DataFrame({"date": dates, "ticker": ticker, "close": close, "return_1d": ret, "volume": volume})
        frame["quality"] = quality
        frame["value"] = value
        rows.append(frame)
    panel = pd.concat(rows, ignore_index=True).sort_values(["ticker", "date"])
    group = panel.groupby("ticker")
    panel["momentum_10"] = group["close"].pct_change(10)
    panel["momentum_21"] = group["close"].pct_change(21)
    panel["momentum_63"] = group["close"].pct_change(63)
    panel["volatility_21"] = group["return_1d"].rolling(21).std().reset_index(level=0, drop=True)
    panel["volume_z"] = group["volume"].transform(lambda s: (s - s.rolling(21).mean()) / s.rolling(21).std())
    panel["target_5d"] = group["close"].pct_change(5).shift(-5)
    panel["target_positive"] = (panel["target_5d"] > 0).astype(int)
    return panel.dropna().reset_index(drop=True)


def time_split(panel: pd.DataFrame, train_fraction: float = 0.75) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = panel["date"].quantile(train_fraction)
    return panel[panel["date"] <= cutoff].copy(), panel[panel["date"] > cutoff].copy()


def rmse(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def baseline_models(panel: pd.DataFrame) -> pd.DataFrame:
    train, test = time_split(panel)
    models = {
        "dummy": DummyClassifier(strategy="most_frequent"),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=80, learning_rate=0.04, max_depth=2, random_state=42),
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=120, learning_rate=0.04, max_leaf_nodes=15, random_state=42),
    }
    rows = []
    for name, model in models.items():
        model.fit(train[FEATURES], train["target_positive"])
        proba = model.predict_proba(test[FEATURES])[:, 1] if hasattr(model, "predict_proba") else model.predict(test[FEATURES])
        pred = (proba > 0.5).astype(int)
        auc = roc_auc_score(test["target_positive"], proba) if len(np.unique(test["target_positive"])) > 1 else np.nan
        rows.append({"model": name, "accuracy": accuracy_score(test["target_positive"], pred), "roc_auc": auc})
    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False)


def tune_gbm(panel: pd.DataFrame) -> pd.DataFrame:
    train, _ = time_split(panel)
    tscv = TimeSeriesSplit(n_splits=3)
    rows = []
    for lr in [0.03, 0.06]:
        for depth in [1, 2, 3]:
            model = GradientBoostingClassifier(n_estimators=80, learning_rate=lr, max_depth=depth, random_state=42)
            scores = cross_val_score(model, train[FEATURES], train["target_positive"], cv=tscv, scoring="roc_auc")
            rows.append({"learning_rate": lr, "max_depth": depth, "cv_roc_auc": scores.mean()})
    return pd.DataFrame(rows).sort_values("cv_roc_auc", ascending=False)


def fit_best_classifier(panel: pd.DataFrame) -> tuple[GradientBoostingClassifier, pd.DataFrame, pd.DataFrame]:
    train, test = time_split(panel)
    model = GradientBoostingClassifier(n_estimators=120, learning_rate=0.04, max_depth=2, random_state=42)
    model.fit(train[FEATURES], train["target_positive"])
    scored = test.copy()
    scored["score"] = model.predict_proba(test[FEATURES])[:, 1]
    scored["prediction"] = (scored["score"] > 0.5).astype(int)
    metrics = pd.DataFrame(
        {"metric": ["accuracy", "roc_auc"], "value": [accuracy_score(scored["target_positive"], scored["prediction"]), roc_auc_score(scored["target_positive"], scored["score"])]}
    )
    return model, scored, metrics


def fit_regressor(panel: pd.DataFrame) -> tuple[GradientBoostingRegressor, pd.DataFrame, float]:
    train, test = time_split(panel)
    model = GradientBoostingRegressor(n_estimators=120, learning_rate=0.04, max_depth=2, random_state=42)
    model.fit(train[FEATURES], train["target_5d"])
    scored = test.copy()
    scored["prediction"] = model.predict(test[FEATURES])
    return model, scored, rmse(scored["target_5d"], scored["prediction"])


def score_quantiles(scored: pd.DataFrame, score_col: str = "score", buckets: int = 5) -> pd.DataFrame:
    frame = scored.copy()
    frame["bucket"] = frame.groupby("date")[score_col].transform(lambda s: pd.qcut(s.rank(method="first"), buckets, labels=False, duplicates="drop") + 1)
    return frame.groupby("bucket")["target_5d"].agg(["mean", "std", "count"]).reset_index()


def long_short_backtest(scored: pd.DataFrame, score_col: str = "score", quantile: float = 0.2) -> pd.DataFrame:
    rows = []
    for date, frame in scored.groupby("date"):
        low = frame[score_col].quantile(quantile)
        high = frame[score_col].quantile(1 - quantile)
        rows.append({"date": date, "strategy_return": frame.loc[frame[score_col] >= high, "target_5d"].mean() - frame.loc[frame[score_col] <= low, "target_5d"].mean()})
    out = pd.DataFrame(rows).dropna().set_index("date")
    out["equity_curve"] = (1 + out["strategy_return"]).cumprod()
    return out


def model_interpretation(model: GradientBoostingClassifier, panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    _, test = time_split(panel)
    perm = permutation_importance(model, test[FEATURES], test["target_positive"], scoring="roc_auc", n_repeats=5, random_state=42)
    importance = pd.DataFrame({"feature": FEATURES, "importance": perm.importances_mean}).sort_values("importance", ascending=False)
    pd_result = partial_dependence(model, test[FEATURES], [FEATURES[0]], grid_resolution=20)
    partial = pd.DataFrame({"feature_value": pd_result["grid_values"][0], "partial_dependence": pd_result["average"][0]})
    return importance, partial


def make_intraday_features(periods: int = 780, seed: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-05 09:30", periods=periods, freq="5min")
    ret = rng.normal(0.00002, 0.0015, periods)
    price = 100 * np.exp(np.cumsum(ret))
    frame = pd.DataFrame({"timestamp": idx, "price": price, "return_5m": ret})
    frame["momentum_6"] = frame["price"].pct_change(6)
    frame["momentum_18"] = frame["price"].pct_change(18)
    frame["volatility_18"] = frame["return_5m"].rolling(18).std()
    frame["target_30m"] = frame["price"].pct_change(6).shift(-6)
    frame["target_positive"] = (frame["target_30m"] > 0).astype(int)
    return frame.dropna().reset_index(drop=True)
