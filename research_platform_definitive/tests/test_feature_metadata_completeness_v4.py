from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


VALID_CATEGORIES = {
    "value",
    "momentum",
    "profitability",
    "investment",
    "intangibles",
    "trading_frictions",
    "macro",
    "risk",
    "quality",
    "smart_money",
    "ml_derived",
    "target",
    "composite",
    "technical",
    "size",
    "factor_alpha",
}


def test_188_features_registered():
    from research_platform_core.feature_metadata import FEATURE_METADATA

    assert len(FEATURE_METADATA) >= 188


def test_all_features_have_academic_metadata():
    from research_platform_core.feature_metadata import FEATURE_METADATA

    for feature_id, meta in FEATURE_METADATA.items():
        assert meta.formula
        assert meta.formula_latex, feature_id
        assert meta.source_paper, feature_id
        assert meta.source_doi, feature_id
        assert meta.factor_zoo_category, feature_id
        assert meta.economic_rationale, feature_id


def test_factor_zoo_categories_valid():
    from research_platform_core.feature_metadata import FEATURE_METADATA

    for feature_id, meta in FEATURE_METADATA.items():
        assert meta.factor_zoo_category in VALID_CATEGORIES, feature_id


def test_factor_alpha_and_investment_present():
    from research_platform_core.feature_metadata import FEATURE_METADATA

    for feature_id in ["alpha_1y", "alpha_5factor", "tracking_error_1y", "asset_growth", "net_stock_issues"]:
        assert feature_id in FEATURE_METADATA
    assert FEATURE_METADATA["alpha_1y"].category == "factor_alpha"
    assert FEATURE_METADATA["asset_growth"].factor_zoo_category == "investment"


def test_all_targets_not_leaking():
    from research_platform_core.feature_metadata import FEATURE_METADATA

    for feature_id, meta in FEATURE_METADATA.items():
        if meta.category == "target" or "forward_return" in feature_id:
            assert meta.leakage_risk is False
            assert meta.point_in_time_safe is True


def test_alpha_factors_computed():
    from research_platform_core.equity_feature_engineering import compute_alpha_factors

    rng = np.random.default_rng(42)
    n_assets, n_days = 5, 320
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    prices = pd.DataFrame(
        np.cumprod(1 + rng.normal(0.0003, 0.01, size=(n_days, n_assets)), axis=0) * 100,
        index=dates,
        columns=[f"T{i}" for i in range(n_assets)],
    )
    benchmark = pd.Series(np.cumprod(1 + rng.normal(0.0002, 0.009, size=n_days)) * 100, index=dates)
    result = compute_alpha_factors(prices, pd.DataFrame(), benchmark, windows=(252,))
    assert "alpha_1y" in result.columns
    assert "tracking_error_1y" in result.columns
    assert len(result) == n_assets


def test_investment_factors_lag_safety():
    from research_platform_core.equity_feature_engineering import compute_investment_factors

    frame = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "AAA"],
            "report_date": pd.to_datetime(["2023-03-31", "2023-06-30", "2023-09-30"]),
            "total_assets": [100.0, 110.0, 121.0],
            "capex": [5.0, 6.0, 7.0],
            "shares_outstanding": [10.0, 10.5, 10.5],
            "total_debt": [30.0, 33.0, 31.0],
            "total_equity": [70.0, 77.0, 90.0],
            "net_income": [7.0, 8.0, 9.0],
            "revenue": [50.0, 55.0, 61.0],
        }
    )
    result = compute_investment_factors(frame)
    assert "asset_growth" in result.columns
    assert pd.isna(result.loc[1, "asset_growth"])
    assert result.loc[2, "asset_growth"] == pytest.approx(0.10)
    assert result.loc[2, "pit_lag_days"] == 45
