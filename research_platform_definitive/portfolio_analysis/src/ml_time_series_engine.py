"""ML and time-series engine for the Investment Research Platform Pro.

The module borrows architectural ideas from public time-series/stock/crypto ML
projects: explicit configs, windowed datasets, model comparison hooks,
forecasting, anomaly flags, regime detection and dashboard-ready DataFrames.
It does not vendor external repository code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd


DEFAULT_TIME_SERIES_ENGINE_CONFIG: dict[str, Any] = {
    "enabled": False,
    "model": "ridge_light",
    "task": "forecasting",
    "lookback": 60,
    "horizon": 5,
    "min_history": 120,
    "test_size": 40,
    "anomaly_z": 2.5,
    "regime_window": 63,
    "random_state": 42,
}

DEFAULT_DEEP_STOCK_ENGINE_CONFIG: dict[str, Any] = {
    "enabled": False,
    "model": "lstm_basic",
    "lookback": 60,
    "horizon": 5,
    "epochs": 8,
    "batch_size": 32,
    "hidden_units": 32,
    "dropout": 0.10,
    "min_history": 160,
    "fallback_model": "gradient_boosting_light",
    "random_state": 42,
}

DEFAULT_CRYPTO_ENGINE_CONFIG: dict[str, Any] = {
    "enabled": False,
    "model": "gradient_boosting_light",
    "lookback": 48,
    "horizon": 3,
    "buy_threshold": 0.02,
    "sell_threshold": -0.02,
    "min_history": 120,
    "random_state": 42,
}


def import_optional_deep_learning() -> dict[str, Any]:
    out: dict[str, Any] = {"tensorflow": None, "torch": None}
    try:
        import tensorflow as tf  # type: ignore

        out["tensorflow"] = tf
    except Exception:
        pass
    try:
        import torch  # type: ignore

        out["torch"] = torch
    except Exception:
        pass
    return out


def merge_engine_config(defaults: Mapping[str, Any], override: Mapping[str, Any] | None = None) -> dict[str, Any]:
    config = dict(defaults)
    config.update(dict(override or {}))
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
        raise ValueError("price_df must include date and ticker")
    price_col = "price" if "price" in df.columns else "adj_close" if "adj_close" in df.columns else "close"
    tickers = list(dict.fromkeys(universe.get("all_tickers", []) or df["ticker"].dropna().unique().tolist()))
    df["date"] = pd.to_datetime(df["date"])
    wide = df[df["ticker"].isin(tickers)].pivot_table(index="date", columns="ticker", values=price_col, aggfunc="last").sort_index().ffill()
    if experiment:
        if experiment.get("start_date"):
            wide = wide.loc[wide.index >= pd.Timestamp(experiment["start_date"])]
        if experiment.get("end_date"):
            wide = wide.loc[wide.index <= pd.Timestamp(experiment["end_date"])]
    return wide.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).dropna(how="all")


def build_price_panel_from_long(price_df: pd.DataFrame, universe: Mapping[str, Any] | None = None) -> pd.DataFrame:
    if price_df is None or price_df.empty:
        return pd.DataFrame()
    df = price_df.copy()
    price_col = "price" if "price" in df.columns else "adj_close" if "adj_close" in df.columns else "close"
    tickers = list(dict.fromkeys((universe or {}).get("all_tickers", []) or df["ticker"].dropna().unique().tolist()))
    df["date"] = pd.to_datetime(df["date"])
    return df[df["ticker"].isin(tickers)].pivot_table(index="date", columns="ticker", values=price_col, aggfunc="last").sort_index().ffill()


def make_supervised_windows(series: pd.Series, lookback: int, horizon: int) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    values = clean.values.astype("float64")
    X, y, dates = [], [], []
    for i in range(lookback, len(values) - horizon + 1):
        X.append(values[i - lookback : i])
        y.append(np.prod(1 + values[i : i + horizon]) - 1)
        dates.append(clean.index[i + horizon - 1])
    if not X:
        return np.empty((0, lookback)), np.empty((0,)), pd.DatetimeIndex([])
    return np.asarray(X), np.asarray(y), pd.DatetimeIndex(dates)


def _fit_predict_light_model(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, model_name: str, random_state: int = 42) -> np.ndarray:
    if len(X_train) == 0 or len(X_test) == 0:
        return np.asarray([])
    try:
        if "gradient" in model_name or "gbm" in model_name:
            from sklearn.ensemble import GradientBoostingRegressor

            model = GradientBoostingRegressor(random_state=random_state, n_estimators=120, max_depth=2)
        else:
            from sklearn.linear_model import Ridge

            model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        return np.asarray(model.predict(X_test), dtype="float64")
    except Exception:
        return np.full(len(X_test), float(np.nanmean(y_train)) if len(y_train) else 0.0)


def _fit_predict_lstm(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, config: Mapping[str, Any]) -> tuple[np.ndarray, str]:
    libs = import_optional_deep_learning()
    tf = libs.get("tensorflow")
    if tf is None or len(X_train) < 20:
        return _fit_predict_light_model(X_train, y_train, X_test, str(config.get("fallback_model", "gradient_boosting_light")), int(config.get("random_state", 42))), "fallback_light"
    try:
        tf.keras.utils.set_random_seed(int(config.get("random_state", 42)))
        lookback = X_train.shape[1]
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(lookback, 1)),
                tf.keras.layers.LSTM(int(config.get("hidden_units", 32))),
                tf.keras.layers.Dropout(float(config.get("dropout", 0.10))),
                tf.keras.layers.Dense(1),
            ]
        )
        model.compile(optimizer="adam", loss="mse")
        model.fit(
            X_train.reshape((-1, lookback, 1)),
            y_train,
            epochs=int(config.get("epochs", 8)),
            batch_size=int(config.get("batch_size", 32)),
            verbose=0,
        )
        pred = model.predict(X_test.reshape((-1, lookback, 1)), verbose=0).reshape(-1)
        return pred.astype("float64"), "tensorflow_lstm"
    except Exception:
        return _fit_predict_light_model(X_train, y_train, X_test, str(config.get("fallback_model", "gradient_boosting_light")), int(config.get("random_state", 42))), "fallback_light"


def _forecast_one_series(series: pd.Series, config: Mapping[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    lookback = int(config.get("lookback", 60))
    horizon = int(config.get("horizon", 5))
    test_size = int(config.get("test_size", 40))
    X, y, dates = make_supervised_windows(series, lookback, horizon)
    if len(y) < max(10, test_size // 2):
        return pd.DataFrame(), {"status": "insufficient_history", "n_samples": int(len(y))}
    split = max(1, len(y) - min(test_size, len(y) // 3))
    X_train, y_train, X_test, y_test = X[:split], y[:split], X[split:], y[split:]
    model_name = str(config.get("model", "ridge_light"))
    if "lstm" in model_name:
        pred, model_used = _fit_predict_lstm(X_train, y_train, X_test, config)
    else:
        pred = _fit_predict_light_model(X_train, y_train, X_test, model_name, int(config.get("random_state", 42)))
        model_used = model_name
    idx = dates[split:]
    forecast = pd.DataFrame({"date": idx, "actual": y_test, "forecast": pred})
    err = forecast["forecast"] - forecast["actual"]
    diagnostics = {
        "status": "ok",
        "model": model_used,
        "model_used": model_used,
        "n_samples": int(len(y)),
        "mae": float(err.abs().mean()) if not err.empty else np.nan,
        "rmse": float(np.sqrt((err**2).mean())) if not err.empty else np.nan,
        "directional_accuracy": float((np.sign(forecast["forecast"]) == np.sign(forecast["actual"])).mean()) if not forecast.empty else np.nan,
    }
    return forecast, diagnostics


def run_time_series_lab(returns_df: pd.DataFrame, config: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
    config = merge_engine_config(DEFAULT_TIME_SERIES_ENGINE_CONFIG, config)
    if not config.get("enabled", False) or returns_df is None or returns_df.empty:
        return {"forecast": pd.DataFrame(), "diagnostics": pd.DataFrame(), "anomalies": pd.DataFrame(), "regimes": pd.DataFrame()}
    forecasts, diagnostics = [], []
    for ticker in returns_df.columns:
        forecast, diag = _forecast_one_series(returns_df[ticker], config)
        diag["ticker"] = ticker
        if not forecast.empty:
            forecast.insert(0, "ticker", ticker)
            forecasts.append(forecast)
        diagnostics.append(diag)
    anomalies = build_anomaly_flags(returns_df, float(config.get("anomaly_z", 2.5)))
    regimes = build_regime_table(returns_df, int(config.get("regime_window", 63)))
    return {"forecast": pd.concat(forecasts, ignore_index=True) if forecasts else pd.DataFrame(), "diagnostics": pd.DataFrame(diagnostics), "anomalies": anomalies, "regimes": regimes}


def build_anomaly_flags(returns_df: pd.DataFrame, z_threshold: float = 2.5) -> pd.DataFrame:
    rows = []
    for ticker in returns_df.columns:
        s = pd.to_numeric(returns_df[ticker], errors="coerce").dropna()
        if s.empty:
            continue
        z = (s - s.rolling(63, min_periods=20).mean()) / s.rolling(63, min_periods=20).std()
        hits = z.abs() >= z_threshold
        for date, value in z[hits].tail(20).items():
            rows.append({"date": date, "ticker": ticker, "z_score": value, "return": s.loc[date], "flag": "anomaly"})
    return pd.DataFrame(rows)


def build_regime_table(returns_df: pd.DataFrame, regime_window: int = 63) -> pd.DataFrame:
    if returns_df is None or returns_df.empty:
        return pd.DataFrame()
    port = returns_df.mean(axis=1).dropna()
    vol = port.rolling(regime_window, min_periods=max(20, regime_window // 3)).std() * np.sqrt(252)
    trend = port.rolling(regime_window, min_periods=max(20, regime_window // 3)).mean() * 252
    out = pd.DataFrame({"date": port.index, "rolling_vol": vol.values, "rolling_return": trend.values})
    vol_hi = out["rolling_vol"].quantile(0.67)
    ret_hi = out["rolling_return"].quantile(0.50)
    out["regime"] = np.where(out["rolling_vol"] >= vol_hi, "high_vol", np.where(out["rolling_return"] >= ret_hi, "risk_on", "risk_off"))
    return out.dropna().reset_index(drop=True)


def run_deep_stock_model(price_panel: pd.DataFrame, config: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
    config = merge_engine_config(DEFAULT_DEEP_STOCK_ENGINE_CONFIG, config)
    if not config.get("enabled", False) or price_panel is None or price_panel.empty:
        return {"signals": pd.DataFrame(), "forecast": pd.DataFrame(), "diagnostics": pd.DataFrame()}
    returns = price_panel.ffill().pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    cfg = {**config, "model": config.get("model", "lstm_basic"), "test_size": min(30, int(config.get("min_history", 160)) // 4)}
    forecasts, diagnostics, signals = [], [], []
    for ticker in returns.columns:
        forecast, diag = _forecast_one_series(returns[ticker], cfg)
        diag["ticker"] = ticker
        diagnostics.append(diag)
        if forecast.empty:
            continue
        forecast.insert(0, "ticker", ticker)
        forecasts.append(forecast)
        latest = forecast.tail(1).iloc[0]
        score = float(latest["forecast"])
        signals.append({"ticker": ticker, "ml_score": score, "signal": "buy" if score > 0 else "sell" if score < 0 else "hold", "model_used": diag.get("model_used"), "forecast_horizon": cfg.get("horizon")})
    return {"signals": pd.DataFrame(signals).sort_values("ml_score", ascending=False) if signals else pd.DataFrame(), "forecast": pd.concat(forecasts, ignore_index=True) if forecasts else pd.DataFrame(), "diagnostics": pd.DataFrame(diagnostics)}


def prepare_crypto_features(crypto_price_df: pd.DataFrame) -> pd.DataFrame:
    if crypto_price_df is None or crypto_price_df.empty:
        return pd.DataFrame()
    df = crypto_price_df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["ticker", "date"]) if "ticker" in df.columns else df.sort_values("date")
    price_col = "price" if "price" in df.columns else "close" if "close" in df.columns else "adj_close"
    if "ticker" not in df.columns:
        df["ticker"] = "CRYPTO"
    df["return"] = df.groupby("ticker")[price_col].pct_change(fill_method=None)
    df["vol_24"] = df.groupby("ticker")["return"].transform(lambda s: s.rolling(24, min_periods=8).std())
    df["mom_24"] = df.groupby("ticker")[price_col].pct_change(24)
    df["mom_72"] = df.groupby("ticker")[price_col].pct_change(72)
    return df.replace([np.inf, -np.inf], np.nan)


def run_crypto_signal_engine(crypto_price_df: pd.DataFrame, config: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
    config = merge_engine_config(DEFAULT_CRYPTO_ENGINE_CONFIG, config)
    if not config.get("enabled", False) or crypto_price_df is None or crypto_price_df.empty:
        return {"signals": pd.DataFrame(), "forecast": pd.DataFrame(), "diagnostics": pd.DataFrame()}
    features = prepare_crypto_features(crypto_price_df)
    if features.empty:
        return {"signals": pd.DataFrame(), "forecast": pd.DataFrame(), "diagnostics": pd.DataFrame([{"status": "no_crypto_data"}])}
    forecasts, diagnostics, signals = [], [], []
    for ticker, part in features.groupby("ticker"):
        series = part.set_index("date")["return"] if "date" in part.columns else part["return"]
        forecast, diag = _forecast_one_series(series, config)
        diag["ticker"] = ticker
        diagnostics.append(diag)
        if forecast.empty:
            continue
        forecast.insert(0, "ticker", ticker)
        forecasts.append(forecast)
        pred = float(forecast.tail(1)["forecast"].iloc[0])
        signal = "buy" if pred >= float(config.get("buy_threshold", 0.02)) else "sell" if pred <= float(config.get("sell_threshold", -0.02)) else "hold"
        signals.append({"ticker": ticker, "forecast_return": pred, "signal": signal, "model_used": diag.get("model_used")})
    return {"signals": pd.DataFrame(signals), "forecast": pd.concat(forecasts, ignore_index=True) if forecasts else pd.DataFrame(), "diagnostics": pd.DataFrame(diagnostics)}


def unified_run_ml_time_series_engine(
    price_df: pd.DataFrame | None,
    returns_df: pd.DataFrame | None,
    universe: Mapping[str, Any],
    experiment: Mapping[str, Any],
    time_series_config: Mapping[str, Any] | None = None,
    deep_stock_config: Mapping[str, Any] | None = None,
    crypto_config: Mapping[str, Any] | None = None,
    crypto_price_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    if returns_df is None or returns_df.empty:
        returns_df = build_returns_matrix_from_universe(price_df if price_df is not None else pd.DataFrame(), universe, experiment)
    price_panel = build_price_panel_from_long(price_df if price_df is not None else pd.DataFrame(), universe)
    ts = run_time_series_lab(returns_df, time_series_config or {})
    deep = run_deep_stock_model(price_panel, deep_stock_config or {})
    crypto = run_crypto_signal_engine(crypto_price_df if crypto_price_df is not None else pd.DataFrame(), crypto_config or {})
    return {"time_series": ts, "deep_stock": deep, "crypto": crypto}
