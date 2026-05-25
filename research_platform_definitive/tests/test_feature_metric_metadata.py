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


def test_feature_metadata_lookup_and_aliases() -> None:
    meta = metadata_for_feature("momentum_12_1")
    assert meta is not None
    assert meta.category == "momentum"
    assert "ret252d" in meta.formula

    alias = metadata_for_feature("ret252d")
    assert alias is not None
    assert alias.id == "return_1y"
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

