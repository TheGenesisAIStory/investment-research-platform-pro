from __future__ import annotations

from research_platform_core import (
    category_for_feature,
    metadata_for_feature,
    metadata_frame,
    metadata_help,
    metadata_for_metric,
    metric_help,
    metrics_metadata_frame,
)
from ml_stock_lab.factor_registry import RAW_FACTOR_COLUMNS


def test_feature_metadata_lookup_and_aliases() -> None:
    meta = metadata_for_feature("momentum_12_1")
    assert meta is not None
    assert meta.category == "momentum"
    assert "ret252d" in meta.formula

    alias = metadata_for_feature("ret252d")
    assert alias is not None
    assert alias.id == "ret252d"
    assert category_for_feature("score_composite") == "ml"


def test_feature_metadata_frame_is_filtered() -> None:
    frame = metadata_frame(["score_composite", "ml_score", "unknown_column"])
    assert set(frame["id"]) == {"score_composite", "ml_score"}
    assert "Interpretation" not in frame.columns
    assert "Formula:" in metadata_help("ml_score")


def test_metric_metadata_lookup_and_aliases() -> None:
    meta = metadata_for_metric("rank_information_coefficient")
    assert meta is not None
    assert meta.id == "rank_ic"
    assert "Spearman" in meta.definition
    assert "Formula:" in metric_help("sharpe")


def test_metrics_metadata_frame_is_filtered() -> None:
    frame = metrics_metadata_frame(["r2_os", "rank_ic", "unknown_metric"])
    assert set(frame["id"]) == {"r2_os", "rank_ic"}
    assert {"definition", "formula", "interpretation"}.issubset(frame.columns)


def test_all_factor_registry_columns_have_metadata() -> None:
    missing = [column for column in RAW_FACTOR_COLUMNS if metadata_for_feature(column) is None]
    assert missing == []


def test_real_artifact_columns_have_metadata() -> None:
    feature_columns = [
        "ret21d",
        "ret63d",
        "ret126d",
        "ret252d",
        "vol63d",
        "vol126d",
        "vol252d",
        "forward_return_21d",
        "target_horizon_days",
        "screener_score",
        "selection_score",
        "composite_score",
        "annual_return",
        "scenario_downside",
        "percentile_rank",
        "prediction",
        "actual",
    ]
    missing_features = [column for column in feature_columns if metadata_for_feature(column) is None]
    assert missing_features == []

    metric_columns = [
        "r2_os",
        "ic",
        "rank_ic",
        "sharpe_long_short",
        "sharpe_long_short_net_cost",
        "volatility_long_short",
        "avg_names_long_short",
        "feature_count",
        "train_rows",
        "test_rows",
        "prediction_rows",
        "panel_rows",
        "mae",
        "rmse",
        "mape",
        "directional_accuracy",
        "forecast_return",
        "forecast_level",
    ]
    missing_metrics = [column for column in metric_columns if metadata_for_metric(column) is None]
    assert missing_metrics == []
