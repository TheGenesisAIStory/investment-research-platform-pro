from __future__ import annotations

from pathlib import Path

import pandas as pd

from research_platform_core.data_explorer import (
    get_single_ticker_snapshot,
    list_available_tickers,
    load_data_explorer_preview,
)


def _seed_ticker_artifacts(tmp_path: Path) -> tuple[Path, Path]:
    db = tmp_path / "Database Finanziario"
    output = tmp_path / "output"
    parquet_root = output / "local_financial_db" / "MarketData" / "OHLCV"
    daily_dir = parquet_root / "daily" / "NASDAQ"
    daily_dir.mkdir(parents=True)
    prices = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-02", periods=3, freq="D"),
            "open": [10.0, 10.5, 11.0],
            "high": [10.5, 11.0, 11.5],
            "low": [9.8, 10.2, 10.8],
            "close": [10.2, 10.9, 11.1],
            "adjclose": [10.2, 10.9, 11.1],
            "volume": [1000, 1200, 1300],
        }
    )
    prices.to_parquet(daily_dir / "AAA.parquet", index=False)
    (output / "tables").mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "provider_symbol": "AAA",
                "exchange": "NASDAQ",
                "universe": "sp500",
                "coverage_status": "OK",
                "status": "downloaded",
                "last_price_date": "2024-01-04",
                "parquet_root": str(parquet_root),
            }
        ]
    ).to_csv(output / "tables" / "OHLCV_daily_manifest.csv", index=False)
    factor_path = output / "ml_training_lab" / "tables" / "FactorUniversePanel.csv"
    factor_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "date": "2024-01-02",
                "ticker": "AAA",
                "price": 10.2,
                "value_score": 60,
                "quality_score": 70,
                "momentum_score": 80,
                "risk_score": 55,
                "factor_composite_score": 66,
                "forward_return_21d": 0.03,
            },
            {
                "date": "2024-02-02",
                "ticker": "AAA",
                "price": 11.1,
                "value_score": 62,
                "quality_score": 72,
                "momentum_score": 82,
                "risk_score": 57,
                "factor_composite_score": 68,
                "forward_return_21d": 0.01,
            },
        ]
    ).to_csv(factor_path, index=False)
    ml_path = output / "ml_stock_lab" / "tables" / "MLStockLab_trained_model_signals.csv"
    ml_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "date": "2024-02-02",
                "ticker": "AAA",
                "score_ols": 55,
                "score_rf": 65,
                "score_gbrt": 60,
                "score_composite": 61,
                "fair_value_hat": 12.5,
                "valuation_signal_score": 74,
            }
        ]
    ).to_csv(ml_path, index=False)
    fundamentals = db / "Equities" / "us" / "sp500" / "fundamentals"
    fundamentals.mkdir(parents=True)
    pd.DataFrame([{"asOfDate": "2023-12-31", "periodType": "12M", "revenue": 100.0}]).to_parquet(fundamentals / "AAA_income.parquet", index=False)
    return db, output


def test_single_ticker_snapshot_collects_data_layers(tmp_path: Path) -> None:
    db, output = _seed_ticker_artifacts(tmp_path)

    snapshot = get_single_ticker_snapshot("AAA", db, output)
    availability = snapshot["availability"]

    assert availability.query("domain == 'OHLCV'")["status"].iloc[0] == "OK"
    assert availability.query("domain == 'Fundamentals'")["status"].iloc[0] == "OK"
    assert availability.query("domain == 'Factor Panel'")["status"].iloc[0] == "OK"
    assert availability.query("domain == 'ML Signals'")["status"].iloc[0] == "OK"
    assert not snapshot["ohlcv"].empty
    assert not snapshot["factors"].empty
    assert not snapshot["ml_signals"].empty


def test_list_available_tickers_uses_coverage_manifest(tmp_path: Path) -> None:
    db, output = _seed_ticker_artifacts(tmp_path)

    tickers = list_available_tickers(db, output)

    assert "AAA" in tickers["ticker"].tolist()
    assert tickers.loc[tickers["ticker"].eq("AAA"), "coverage_status"].iloc[0] == "OK"


def test_data_explorer_preview_filters_factor_panel_by_date(tmp_path: Path) -> None:
    db, output = _seed_ticker_artifacts(tmp_path)

    preview = load_data_explorer_preview(
        "factor_panel",
        db,
        output,
        universe="All",
        start_date="2024-02-01",
        end_date="2024-12-31",
        max_rows=100,
    )

    assert len(preview) == 1
    assert preview["date"].iloc[0] == "2024-02-02"
