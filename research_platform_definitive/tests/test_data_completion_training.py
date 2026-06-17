from __future__ import annotations

from pathlib import Path

import pandas as pd

from research_platform_core.data_completion import CompletionConfig, ResearchDataBootstrapper, validate_completion_coverage
from ml_stock_lab.factor_registry import add_factor_scores, feature_columns_for_blocks, is_leakage_feature
from ml_stock_lab.training import train_ml_model_suite


def test_data_completion_dry_run_writes_manifests(tmp_path: Path) -> None:
    financial_db = tmp_path / "Database Finanziario"
    output = tmp_path / "output"
    config = CompletionConfig(
        start_year=2000,
        end_year=2026,
        execute=False,
        max_assets=2,
        max_symbols=2,
        include_smart_money=False,
        include_banking=False,
    )
    result = ResearchDataBootstrapper(financial_db, output, config).run_all()

    assert "equity_prices" in result
    assert (output / "data_completion" / "full_completion_manifest.json").exists()
    assert (output / "data_completion" / "DataCompletion_source_map.csv").exists()

    report = validate_completion_coverage(financial_db, output, strict=False)
    assert not report.empty
    assert report["passes"].astype(bool).any()


def test_ml_training_suite_trains_on_factor_panel(tmp_path: Path) -> None:
    output = tmp_path / "output"
    db = tmp_path / "Database Finanziario"
    panel_path = output / "ml_training_lab" / "tables" / "FactorUniversePanel.csv"
    panel_path.parent.mkdir(parents=True)

    rows = []
    dates = pd.date_range("2017-01-31", "2020-12-31", freq="ME")
    tickers = ["AAA", "BBB", "CCC", "DDD"]
    for ticker_id, ticker in enumerate(tickers):
        for idx, date in enumerate(dates):
            price = 50 + ticker_id * 5 + idx * (1 + ticker_id * 0.1)
            rows.append(
                {
                    "date": date.date().isoformat(),
                    "ticker": ticker,
                    "price": price,
                    "market_value": price * 100,
                    "forward_return": 0.01 * (ticker_id + 1) + 0.0005 * idx,
                    "quality_score": 40 + ticker_id * 10,
                    "momentum_score": 30 + idx,
                    "risk_score": 80 - ticker_id * 5,
                    "valuation_score": 60 - ticker_id * 3,
                }
            )
    pd.DataFrame(rows).to_csv(panel_path, index=False)

    result = train_ml_model_suite(
        output_root=output,
        financial_db_root=db,
        start_year=2017,
        end_year=2020,
        train_end_year=2018,
        test_start_year=2019,
        models=["ols"],
        max_rows=None,
    )

    assert result["status"] == "OK"
    assert result["metrics"]["status"].eq("OK").any()
    assert (output / "ml_training_lab" / "tables" / "MLTraining_predictions.csv").exists()
    assert (output / "ml_training_lab" / "tables" / "MLTraining_predictions_wide.csv").exists()
    assert (output / "ml_training_lab" / "tables" / "MLTraining_model_cards.csv").exists()
    assert (output / "ml_training_lab" / "MLTraining_manifest.json").exists()
    assert (output / "ml_stock_lab" / "tables" / "MLStockLab_model_comparison.csv").exists()
    assert "rank_ic" in result["metrics"].columns
    assert "target_horizon_days" in result["metrics"].columns


def test_factor_registry_scores_and_leakage_controls() -> None:
    panel = pd.DataFrame(
        {
            "date": ["2024-01-31"] * 4,
            "ticker": ["AAA", "BBB", "CCC", "DDD"],
            "pe": [10, 20, 30, 40],
            "pb": [1, 2, 3, 4],
            "roe": [20, 15, 10, 5],
            "debt_to_equity": [10, 20, 30, 40],
            "ret252d": [0.40, 0.30, 0.10, -0.10],
            "ret21d": [0.03, 0.02, 0.01, 0.00],
            "vol63d": [0.10, 0.20, 0.30, 0.40],
            "forward_return": [0.02, 0.01, 0.00, -0.01],
            "prediction": [1, 2, 3, 4],
        }
    )

    scored = add_factor_scores(panel)
    features = feature_columns_for_blocks(scored, ["value", "quality", "momentum", "risk"], target="forward_return", min_non_null=1)

    assert scored.loc[scored["ticker"].eq("AAA"), "value_score"].iloc[0] > scored.loc[scored["ticker"].eq("DDD"), "value_score"].iloc[0]
    assert "momentum_12_1" in scored.columns
    assert "forward_return" not in features
    assert "prediction" not in features
    assert is_leakage_feature("future_return")
