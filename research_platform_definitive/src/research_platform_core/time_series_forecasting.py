"""Time-series forecasting helpers for the Gen.is.IA research workstation.

This module complements the cross-sectional ML Stock Lab.  The existing ML
lab predicts forward returns across many securities; this layer forecasts one
series at a time using strictly lagged/rolling features.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .data_explorer import load_ticker_ohlcv, list_available_tickers
from .data_platform import resolve_data_platform_roots, utc_now
from .macro_market import macro_asset_catalog


TIME_SERIES_TABLE_ROOT = Path("time_series_lab") / "tables"
DEFAULT_HORIZONS = (5, 21, 63)
DEFAULT_MODELS = ("naive", "ols", "gbrt")


@dataclass(frozen=True)
class TimeSeriesForecastConfig:
    """Configuration for a single-series forecast run."""

    source: str = "macro"
    symbol: str = "SPY"
    horizons: tuple[int, ...] = DEFAULT_HORIZONS
    models: tuple[str, ...] = DEFAULT_MODELS
    train_end_year: int = 2018
    test_start_year: int = 2019
    max_rows: int | None = None
    write_artifacts: bool = True


@dataclass
class TimeSeriesForecastResult:
    """In-memory result returned by the forecasting layer."""

    config: TimeSeriesForecastConfig
    source_frame: pd.DataFrame
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    latest_forecasts: pd.DataFrame
    feature_schema: pd.DataFrame
    manifest: dict[str, Any]


def _safe_symbol(symbol: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(symbol or "").strip().upper())


def _read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 1:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, **kwargs)
    except Exception:
        return pd.DataFrame()


def _normalize_history(frame: pd.DataFrame, symbol: str, source: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["date", "symbol", "source", "close"])
    out = frame.copy()
    out.columns = [str(col).strip().lower().replace(" ", "_") for col in out.columns]
    if "adj_close" in out.columns and "adjclose" not in out.columns:
        out["adjclose"] = out["adj_close"]
    if "close" not in out.columns and "adjclose" in out.columns:
        out["close"] = out["adjclose"]
    if "date" not in out.columns:
        return pd.DataFrame(columns=["date", "symbol", "source", "close"])
    out["date"] = pd.to_datetime(out["date"], errors="coerce", utc=True).dt.tz_convert(None)
    out["close"] = pd.to_numeric(out.get("close"), errors="coerce")
    out = out.dropna(subset=["date", "close"]).sort_values("date")
    out["symbol"] = str(symbol or out.get("symbol", "")).upper()
    out["source"] = source
    keep = [
        col
        for col in [
            "date",
            "symbol",
            "provider_symbol",
            "name",
            "asset_class",
            "region",
            "country",
            "category",
            "exposure",
            "currency",
            "source",
            "open",
            "high",
            "low",
            "close",
            "adjclose",
            "volume",
            "ticker",
            "source_path",
        ]
        if col in out.columns
    ]
    return out[keep].reset_index(drop=True)


def available_time_series_assets(
    financial_db_root: str | Path | None = None,
    output_root: str | Path | None = None,
    equity_limit: int = 5000,
) -> pd.DataFrame:
    """Return macro assets and broad equity tickers that can feed the TS Lab."""
    roots = resolve_data_platform_roots(financial_db_root=financial_db_root, repo_output_root=output_root)
    macro = macro_asset_catalog()
    macro = macro.assign(source_type="macro", display_label=macro["symbol"] + " · " + macro["name"])

    equities = list_available_tickers(roots.financial_db, roots.repo_output, limit=equity_limit)
    if not equities.empty:
        equities = equities.rename(columns={"ticker": "symbol"})
        equities["source_type"] = "equity"
        equities["name"] = equities["symbol"]
        equities["asset_class"] = "equity"
        equities["region"] = equities.get("universe", "")
        equities["display_label"] = equities["symbol"] + " · Equity OHLCV"
        keep_equity = ["symbol", "name", "asset_class", "region", "source_type", "display_label"]
        equities = equities[[col for col in keep_equity if col in equities.columns]]
    else:
        equities = pd.DataFrame(columns=["symbol", "name", "asset_class", "region", "source_type", "display_label"])

    keep_macro = ["symbol", "name", "asset_class", "region", "source_type", "display_label"]
    out = pd.concat([macro[keep_macro], equities[keep_macro]], ignore_index=True, sort=False)
    return out.drop_duplicates(["source_type", "symbol"]).reset_index(drop=True)


def load_macro_series(
    symbol: str,
    financial_db_root: str | Path | None = None,
    output_root: str | Path | None = None,
    tail_rows: int | None = None,
) -> pd.DataFrame:
    """Load one macro/multi-asset series from the Macro DB or sample artifact."""
    roots = resolve_data_platform_roots(financial_db_root=financial_db_root, repo_output_root=output_root)
    clean = str(symbol or "").strip().upper()
    if not clean:
        return pd.DataFrame()

    manifest_paths = [
        roots.repo_output / "macro_market" / "tables" / "MacroAssetManifest.csv",
        roots.financial_db / "MarketData" / "Macro" / "MacroAssetManifest.csv",
    ]
    for manifest_path in manifest_paths:
        manifest = _read_csv(manifest_path)
        if manifest.empty or "symbol" not in manifest.columns:
            continue
        matches = manifest[manifest["symbol"].astype(str).str.upper().eq(clean)]
        if matches.empty:
            continue
        target = str(matches.iloc[0].get("target_path", "") or "")
        if target:
            path = Path(target).expanduser()
            if path.exists():
                try:
                    frame = pd.read_parquet(path)
                    return _normalize_history(frame.tail(tail_rows) if tail_rows else frame, clean, "macro")
                except Exception:
                    pass

    catalog = macro_asset_catalog()
    matches = catalog[catalog["symbol"].astype(str).str.upper().eq(clean)]
    if not matches.empty:
        asset_class = str(matches.iloc[0].get("asset_class", "macro"))
        candidate = roots.financial_db / "MarketData" / "Macro" / asset_class / f"{_safe_symbol(clean)}.parquet"
        if candidate.exists():
            try:
                frame = pd.read_parquet(candidate)
                return _normalize_history(frame.tail(tail_rows) if tail_rows else frame, clean, "macro")
            except Exception:
                pass

    sample = _read_csv(roots.repo_output / "macro_market" / "tables" / "MacroHistorySample.csv")
    if not sample.empty and "symbol" in sample.columns:
        sample = sample[sample["symbol"].astype(str).str.upper().eq(clean)]
        if not sample.empty:
            return _normalize_history(sample.tail(tail_rows) if tail_rows else sample, clean, "macro_sample")
    return pd.DataFrame()


def load_equity_series(
    ticker: str,
    financial_db_root: str | Path | None = None,
    output_root: str | Path | None = None,
    tail_rows: int | None = None,
) -> pd.DataFrame:
    """Load one equity OHLCV series from the shared OHLCV parquet store."""
    rows = int(tail_rows) if tail_rows else 1_000_000
    frame = load_ticker_ohlcv(ticker, financial_db_root=financial_db_root, output_root=output_root, tail_rows=rows)
    return _normalize_history(frame, ticker, "equity")


def prepare_time_series_frame(frame: pd.DataFrame, symbol: str | None = None) -> pd.DataFrame:
    """Normalize a raw price frame into the canonical TS Lab input."""
    if frame.empty:
        return pd.DataFrame(columns=["date", "symbol", "close", "return_1d"])
    clean = _normalize_history(frame, symbol or str(frame.get("symbol", "").iloc[0] if "symbol" in frame.columns and len(frame) else ""), "time_series")
    clean["return_1d"] = clean["close"].pct_change()
    clean["log_return_1d"] = np.log(clean["close"]).diff()
    return clean.drop_duplicates("date").reset_index(drop=True)


def make_time_series_features(
    frame: pd.DataFrame,
    horizon: int = 21,
    *,
    lags: Iterable[int] = (1, 2, 5, 21, 63),
    rolling_windows: Iterable[int] = (5, 21, 63),
) -> pd.DataFrame:
    """Create strictly lagged/rolling features and one forward-return target."""
    base = prepare_time_series_frame(frame)
    if base.empty:
        return pd.DataFrame()
    out = base[["date", "symbol", "close", "return_1d", "log_return_1d"]].copy()
    close = out["close"]
    returns = out["return_1d"]

    for lag in sorted({int(item) for item in lags if int(item) > 0}):
        out[f"return_lag_{lag}d"] = close.pct_change(lag)
        out[f"return_1d_lag_{lag}"] = returns.shift(lag)

    for window in sorted({int(item) for item in rolling_windows if int(item) > 1}):
        roll = returns.rolling(window=window, min_periods=max(3, min(window, 10)))
        out[f"rolling_mean_return_{window}d"] = roll.mean()
        out[f"rolling_volatility_{window}d"] = roll.std()
        out[f"momentum_{window}d"] = close.pct_change(window)
        out[f"moving_average_gap_{window}d"] = close / close.rolling(window=window, min_periods=max(3, min(window, 10))).mean() - 1.0
        rolling_max = close.rolling(window=window, min_periods=max(3, min(window, 10))).max()
        out[f"rolling_drawdown_{window}d"] = close / rolling_max - 1.0

    out[f"forward_return_{int(horizon)}d"] = close.shift(-int(horizon)) / close - 1.0
    out["horizon_days"] = int(horizon)
    return out.replace([np.inf, -np.inf], np.nan).reset_index(drop=True)


def _feature_columns(frame: pd.DataFrame, target_col: str) -> list[str]:
    excluded = {"date", "symbol", "close", "horizon_days", target_col}
    excluded.update({col for col in frame.columns if col.startswith("forward_return_")})
    numeric = frame.select_dtypes(include=[np.number]).columns
    return [col for col in numeric if col not in excluded]


def _split_train_test(
    frame: pd.DataFrame,
    target_col: str,
    train_end_year: int,
    test_start_year: int,
    *,
    min_train_rows: int = 40,
    min_test_rows: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    usable = frame.dropna(subset=[target_col]).copy()
    if usable.empty:
        return usable.iloc[0:0], usable.iloc[0:0]
    years = pd.to_datetime(usable["date"], errors="coerce").dt.year
    train = usable[years <= int(train_end_year)]
    test = usable[years >= int(test_start_year)]
    if len(train) >= min_train_rows and len(test) >= min_test_rows:
        return train, test

    split = max(min_train_rows, int(len(usable) * 0.75))
    if len(usable) - split < min_test_rows:
        split = max(1, len(usable) - min_test_rows)
    return usable.iloc[:split].copy(), usable.iloc[split:].copy()


def _metric_row(
    *,
    symbol: str,
    model: str,
    horizon: int,
    y_true: pd.Series,
    y_pred: pd.Series,
    train_rows: int,
    test_rows: int,
    feature_count: int,
) -> dict[str, Any]:
    valid = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).dropna()
    if valid.empty:
        return {
            "symbol": symbol,
            "model": model,
            "horizon_days": horizon,
            "train_rows": train_rows,
            "test_rows": test_rows,
            "feature_count": feature_count,
            "mae": np.nan,
            "rmse": np.nan,
            "mape": np.nan,
            "directional_accuracy": np.nan,
            "hit_ratio": np.nan,
        }
    err = valid["y_pred"] - valid["y_true"]
    denom = valid["y_true"].abs().clip(lower=1e-6)
    return {
        "symbol": symbol,
        "model": model,
        "horizon_days": horizon,
        "train_rows": int(train_rows),
        "test_rows": int(test_rows),
        "feature_count": int(feature_count),
        "mae": float(err.abs().mean()),
        "rmse": float(math.sqrt(float((err**2).mean()))),
        "mape": float((err.abs() / denom).mean()),
        "directional_accuracy": float((np.sign(valid["y_pred"]) == np.sign(valid["y_true"])).mean()),
        "hit_ratio": float((valid["y_true"] * valid["y_pred"] > 0).mean()),
    }


def _fit_sklearn_model(model_name: str, train: pd.DataFrame, test: pd.DataFrame, latest: pd.DataFrame, features: list[str], target_col: str) -> tuple[pd.Series, float | None]:
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.linear_model import LinearRegression
    except Exception:
        return pd.Series(index=test.index, dtype=float), None

    if len(train) < 20 or not features:
        return pd.Series(index=test.index, dtype=float), None
    model_name = model_name.lower()
    if model_name == "ols":
        model = LinearRegression()
    elif model_name == "gbrt":
        model = GradientBoostingRegressor(random_state=7, n_estimators=120, max_depth=2, learning_rate=0.04)
    else:
        return pd.Series(index=test.index, dtype=float), None

    x_train = train[features].replace([np.inf, -np.inf], np.nan)
    medians = x_train.median(numeric_only=True)
    x_train = x_train.fillna(medians).fillna(0.0)
    y_train = pd.to_numeric(train[target_col], errors="coerce")
    valid = y_train.notna()
    if valid.sum() < 20:
        return pd.Series(index=test.index, dtype=float), None
    model.fit(x_train.loc[valid], y_train.loc[valid])

    x_test = test[features].replace([np.inf, -np.inf], np.nan).fillna(medians).fillna(0.0)
    test_pred = pd.Series(model.predict(x_test), index=test.index, dtype=float)
    x_latest = latest[features].replace([np.inf, -np.inf], np.nan).fillna(medians).fillna(0.0)
    latest_pred = float(model.predict(x_latest)[0]) if not x_latest.empty else None
    return test_pred, latest_pred


def fit_time_series_forecasts(
    history: pd.DataFrame,
    *,
    symbol: str,
    source: str,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    models: Iterable[str] = DEFAULT_MODELS,
    train_end_year: int = 2018,
    test_start_year: int = 2019,
) -> TimeSeriesForecastResult:
    """Fit baseline/ML forecasts for a single price series."""
    source_frame = prepare_time_series_frame(history, symbol)
    config = TimeSeriesForecastConfig(
        source=source,
        symbol=str(symbol).upper(),
        horizons=tuple(int(h) for h in horizons),
        models=tuple(str(model).lower() for model in models),
        train_end_year=int(train_end_year),
        test_start_year=int(test_start_year),
        write_artifacts=False,
    )
    if source_frame.empty:
        empty = pd.DataFrame()
        return TimeSeriesForecastResult(config, source_frame, empty, empty, empty, empty, {"status": "NO_DATA", "symbol": symbol})

    metrics_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    latest_rows: list[dict[str, Any]] = []
    schema_rows: list[dict[str, Any]] = []
    last_close = float(source_frame["close"].dropna().iloc[-1])
    last_date = pd.to_datetime(source_frame["date"], errors="coerce").max()

    for horizon in config.horizons:
        target_col = f"forward_return_{horizon}d"
        feature_frame = make_time_series_features(source_frame, horizon=horizon)
        if feature_frame.empty or target_col not in feature_frame.columns:
            continue
        train, test = _split_train_test(feature_frame, target_col, config.train_end_year, config.test_start_year)
        features = _feature_columns(feature_frame, target_col)
        latest = feature_frame.tail(1)
        for feature in features:
            schema_rows.append({"symbol": config.symbol, "horizon_days": horizon, "feature": feature, "source": "lagged_or_rolling"})

        for model in config.models:
            model = model.lower()
            if model == "naive":
                preferred = f"return_lag_{horizon}d"
                fallback = "return_lag_1d"
                pred_col = preferred if preferred in test.columns else fallback
                test_pred = pd.to_numeric(test.get(pred_col), errors="coerce") if pred_col in test.columns else pd.Series(0.0, index=test.index)
                latest_pred = latest[pred_col].iloc[0] if pred_col in latest.columns else 0.0
            elif model in {"ols", "gbrt"}:
                test_pred, latest_pred = _fit_sklearn_model(model, train, test, latest, features, target_col)
            else:
                continue

            y_true = pd.to_numeric(test.get(target_col), errors="coerce")
            residuals = pd.DataFrame({"y_true": y_true, "y_pred": test_pred}).dropna()
            forecast_error_std = float((residuals["y_pred"] - residuals["y_true"]).std()) if len(residuals) >= 3 else np.nan
            metrics_rows.append(
                {
                    **_metric_row(
                        symbol=config.symbol,
                        model=model,
                        horizon=horizon,
                        y_true=y_true,
                        y_pred=test_pred,
                        train_rows=len(train),
                        test_rows=len(test),
                        feature_count=len(features),
                    ),
                    "forecast_error_std": forecast_error_std,
                }
            )
            predictions = test[["date", "symbol", "close", target_col]].copy()
            predictions = predictions.rename(columns={target_col: "realized_forward_return"})
            predictions["model"] = model
            predictions["horizon_days"] = horizon
            predictions["predicted_forward_return"] = test_pred.values if len(test_pred) == len(predictions) else np.nan
            prediction_frames.append(predictions)

            forecast_return = float(latest_pred) if latest_pred is not None and pd.notna(latest_pred) else np.nan
            interval_low = forecast_return - forecast_error_std if pd.notna(forecast_return) and pd.notna(forecast_error_std) else np.nan
            interval_high = forecast_return + forecast_error_std if pd.notna(forecast_return) and pd.notna(forecast_error_std) else np.nan
            latest_rows.append(
                {
                    "symbol": config.symbol,
                    "source": source,
                    "last_date": last_date.date().isoformat() if pd.notna(last_date) else "",
                    "last_close": last_close,
                    "model": model,
                    "horizon_days": horizon,
                    "forecast_return": forecast_return,
                    "forecast_error_std": forecast_error_std,
                    "forecast_return_low": interval_low,
                    "forecast_return_high": interval_high,
                    "forecast_level": last_close * (1.0 + forecast_return) if pd.notna(forecast_return) else np.nan,
                    "forecast_level_low": last_close * (1.0 + interval_low) if pd.notna(interval_low) else np.nan,
                    "forecast_level_high": last_close * (1.0 + interval_high) if pd.notna(interval_high) else np.nan,
                    "train_rows": len(train),
                    "test_rows": len(test),
                    "feature_count": len(features),
                }
            )

    metrics = pd.DataFrame(metrics_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True, sort=False) if prediction_frames else pd.DataFrame()
    latest_forecasts = pd.DataFrame(latest_rows)
    feature_schema = pd.DataFrame(schema_rows).drop_duplicates().reset_index(drop=True) if schema_rows else pd.DataFrame()
    manifest = {
        "status": "OK" if not metrics.empty else "NO_MODEL_RESULTS",
        "source": source,
        "symbol": config.symbol,
        "rows": int(len(source_frame)),
        "start_date": source_frame["date"].min().date().isoformat() if len(source_frame) else "",
        "end_date": source_frame["date"].max().date().isoformat() if len(source_frame) else "",
        "horizons": list(config.horizons),
        "models": list(config.models),
        "train_end_year": config.train_end_year,
        "test_start_year": config.test_start_year,
        "updated_at": utc_now(),
        "method_note": "Single-series forecasting using past lag/rolling features only; complements cross-sectional ML Stock Lab.",
    }
    return TimeSeriesForecastResult(config, source_frame, metrics, predictions, latest_forecasts, feature_schema, manifest)


def write_time_series_forecast_artifacts(
    result: TimeSeriesForecastResult,
    output_root: str | Path | None = None,
) -> dict[str, Path]:
    """Persist TS Lab artifacts under output/time_series_lab/tables."""
    roots = resolve_data_platform_roots(repo_output_root=output_root)
    table_root = roots.repo_output / TIME_SERIES_TABLE_ROOT
    table_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "metrics": table_root / "TimeSeriesForecast_metrics.csv",
        "predictions": table_root / "TimeSeriesForecast_predictions.csv",
        "latest": table_root / "TimeSeriesForecast_latest.csv",
        "feature_schema": table_root / "TimeSeriesForecast_feature_schema.csv",
        "manifest": table_root / "TimeSeriesForecast_manifest.json",
    }
    result.metrics.to_csv(paths["metrics"], index=False)
    result.predictions.to_csv(paths["predictions"], index=False)
    result.latest_forecasts.to_csv(paths["latest"], index=False)
    result.feature_schema.to_csv(paths["feature_schema"], index=False)
    paths["manifest"].write_text(json.dumps(result.manifest, indent=2), encoding="utf-8")
    return paths


def run_time_series_forecast(
    config: TimeSeriesForecastConfig,
    financial_db_root: str | Path | None = None,
    output_root: str | Path | None = None,
) -> TimeSeriesForecastResult:
    """Load one source series, fit forecasts and optionally write artifacts."""
    source = config.source.lower()
    if source == "macro":
        history = load_macro_series(config.symbol, financial_db_root=financial_db_root, output_root=output_root, tail_rows=config.max_rows)
    elif source == "equity":
        history = load_equity_series(config.symbol, financial_db_root=financial_db_root, output_root=output_root, tail_rows=config.max_rows)
    else:
        history = pd.DataFrame()
    result = fit_time_series_forecasts(
        history,
        symbol=config.symbol,
        source=source,
        horizons=config.horizons,
        models=config.models,
        train_end_year=config.train_end_year,
        test_start_year=config.test_start_year,
    )
    if config.write_artifacts:
        paths = write_time_series_forecast_artifacts(result, output_root=output_root)
        result.manifest["artifact_paths"] = {key: str(path) for key, path in paths.items()}
        paths["manifest"].write_text(json.dumps(result.manifest, indent=2), encoding="utf-8")
    return result


def load_time_series_forecast_artifacts(output_root: str | Path | None = None) -> dict[str, Any]:
    """Load latest Time Series Lab artifacts for Streamlit viewers."""
    roots = resolve_data_platform_roots(repo_output_root=output_root)
    table_root = roots.repo_output / TIME_SERIES_TABLE_ROOT
    manifest_path = table_root / "TimeSeriesForecast_manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}
    return {
        "metrics": _read_csv(table_root / "TimeSeriesForecast_metrics.csv"),
        "predictions": _read_csv(table_root / "TimeSeriesForecast_predictions.csv"),
        "latest": _read_csv(table_root / "TimeSeriesForecast_latest.csv"),
        "feature_schema": _read_csv(table_root / "TimeSeriesForecast_feature_schema.csv"),
        "manifest": manifest,
        "table_root": table_root,
    }


def summarize_time_series_forecast_context(
    symbols: Iterable[str],
    output_root: str | Path | None = None,
    *,
    preferred_model: str = "gbrt",
) -> pd.DataFrame:
    """Return compact latest forecast rows for Macro/Portfolio context panels."""
    wanted = {str(symbol or "").strip().upper() for symbol in symbols if str(symbol or "").strip()}
    artifacts = load_time_series_forecast_artifacts(output_root)
    latest = artifacts.get("latest", pd.DataFrame())
    if latest.empty or not wanted or "symbol" not in latest.columns:
        return pd.DataFrame()
    view = latest.copy()
    view["symbol"] = view["symbol"].astype(str).str.upper()
    view = view[view["symbol"].isin(wanted)]
    if view.empty:
        return pd.DataFrame()
    view["_model_rank"] = view["model"].astype(str).str.lower().ne(str(preferred_model).lower()).astype(int) if "model" in view.columns else 0
    sort_cols = ["symbol", "horizon_days", "_model_rank"]
    view = view.sort_values([col for col in sort_cols if col in view.columns])
    dedupe = [col for col in ["symbol", "horizon_days"] if col in view.columns]
    if dedupe:
        view = view.drop_duplicates(dedupe, keep="first")
    keep = [
        "symbol",
        "source",
        "last_date",
        "last_close",
        "model",
        "horizon_days",
        "forecast_return",
        "forecast_error_std",
        "forecast_return_low",
        "forecast_return_high",
        "forecast_level",
        "forecast_level_low",
        "forecast_level_high",
        "test_rows",
        "feature_count",
    ]
    return view[[col for col in keep if col in view.columns]].reset_index(drop=True)
