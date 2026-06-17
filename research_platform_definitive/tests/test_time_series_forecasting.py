from __future__ import annotations

import json

import numpy as np
import pandas as pd

from research_platform_core import (
    TimeSeriesForecastConfig,
    fit_time_series_forecasts,
    load_macro_series,
    make_time_series_features,
    run_time_series_forecast,
)
from research_platform_core.time_series_forecasting import write_time_series_forecast_artifacts


def _synthetic_history(rows: int = 360) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-01", periods=rows)
    returns = 0.0002 + rng.normal(0, 0.01, rows)
    close = 100.0 * (1.0 + pd.Series(returns)).cumprod()
    return pd.DataFrame({"date": dates, "close": close})


def test_make_time_series_features_uses_forward_target_without_future_features() -> None:
    history = _synthetic_history(90)
    features = make_time_series_features(history, horizon=5)

    assert "forward_return_5d" in features.columns
    assert "return_lag_5d" in features.columns
    assert "rolling_volatility_21d" in features.columns

    first_valid_target = features["forward_return_5d"].first_valid_index()
    expected = history.loc[first_valid_target + 5, "close"] / history.loc[first_valid_target, "close"] - 1.0
    assert np.isclose(features.loc[first_valid_target, "forward_return_5d"], expected)
    assert not any(col.startswith("forward_return_") and col != "forward_return_5d" for col in features.columns)


def test_fit_time_series_forecasts_returns_metrics_predictions_and_latest() -> None:
    result = fit_time_series_forecasts(
        _synthetic_history(420),
        symbol="TEST",
        source="synthetic",
        horizons=(5, 21),
        models=("naive", "ols"),
        train_end_year=2020,
        test_start_year=2021,
    )

    assert result.manifest["status"] == "OK"
    assert set(result.metrics["model"]) == {"naive", "ols"}
    assert set(result.metrics["horizon_days"]) == {5, 21}
    assert {"mae", "rmse", "mape", "directional_accuracy"}.issubset(result.metrics.columns)
    assert {"forecast_error_std"}.issubset(result.metrics.columns)
    assert {"forecast_return_low", "forecast_return_high", "forecast_level_low", "forecast_level_high"}.issubset(result.latest_forecasts.columns)
    assert not result.predictions.empty
    assert not result.latest_forecasts.empty


def test_write_time_series_forecast_artifacts(tmp_path) -> None:
    result = fit_time_series_forecasts(
        _synthetic_history(320),
        symbol="TEST",
        source="synthetic",
        horizons=(5,),
        models=("naive", "ols"),
        train_end_year=2020,
        test_start_year=2021,
    )

    paths = write_time_series_forecast_artifacts(result, output_root=tmp_path)

    assert paths["metrics"].exists()
    assert paths["predictions"].exists()
    assert paths["latest"].exists()
    assert paths["feature_schema"].exists()
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["symbol"] == "TEST"


def test_run_time_series_forecast_loads_macro_series_from_manifest(tmp_path) -> None:
    financial_db = tmp_path / "financial_db"
    output = tmp_path / "output"
    macro_path = financial_db / "MarketData" / "Macro" / "equity_index_etf" / "SPY.parquet"
    macro_path.parent.mkdir(parents=True, exist_ok=True)
    _synthetic_history(340).assign(symbol="SPY").to_parquet(macro_path, index=False)
    manifest_root = output / "macro_market" / "tables"
    manifest_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "symbol": "SPY",
                "asset_class": "equity_index_etf",
                "status": "OK",
                "target_path": str(macro_path),
                "rows": 340,
            }
        ]
    ).to_csv(manifest_root / "MacroAssetManifest.csv", index=False)

    loaded = load_macro_series("SPY", financial_db_root=financial_db, output_root=output)
    assert len(loaded) == 340

    result = run_time_series_forecast(
        TimeSeriesForecastConfig(source="macro", symbol="SPY", horizons=(5,), models=("naive",), write_artifacts=True),
        financial_db_root=financial_db,
        output_root=output,
    )
    assert result.manifest["status"] == "OK"
    assert (output / "time_series_lab" / "tables" / "TimeSeriesForecast_latest.csv").exists()
