from __future__ import annotations

import numpy as np
import pandas as pd

from research_platform_core.equity_feature_engineering import (
    compute_advanced_fundamental_features,
    compute_advanced_technical_features,
    compute_piotroski_f_score,
)
from research_platform_core.portfolio_analytics import (
    compute_efficient_frontier,
    compute_max_sharpe_weights,
    compute_min_variance_weights,
    compute_portfolio_performance_metrics,
    compute_risk_parity_weights,
)
from research_platform_core.valuation_analytics import (
    compute_asset_based_valuation,
    compute_comps_valuation,
    compute_dcf_valuation,
    compute_ddm_valuation,
    compute_eva_residual_income,
    compute_market_multiples,
    compute_wacc,
)


def test_piotroski_f_score_reaches_strong_when_all_criteria_improve() -> None:
    frame = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA"],
            "date": pd.to_datetime(["2025-12-31", "2026-12-31"]),
            "roa": [0.05, 0.08],
            "operating_cash_flow": [10.0, 12.0],
            "accruals_ratio": [0.05, -0.02],
            "debt_to_equity": [0.7, 0.5],
            "current_ratio": [1.1, 1.4],
            "shares_outstanding": [100.0, 100.0],
            "gross_margin": [0.31, 0.36],
            "asset_turnover": [0.8, 0.95],
        }
    )
    score = compute_piotroski_f_score(frame)
    assert score.iloc[-1] == 9.0


def test_advanced_technical_features_add_course_style_columns() -> None:
    dates = pd.date_range("2024-01-01", periods=280, freq="B")
    price = pd.Series(np.linspace(100, 150, len(dates)) + np.sin(np.arange(len(dates))), index=dates)
    frame = pd.DataFrame(
        {
            "date": dates,
            "price": price.to_numpy(),
            "high": price.to_numpy() * 1.01,
            "low": price.to_numpy() * 0.99,
            "volume": np.linspace(1_000_000, 1_500_000, len(dates)),
            "shares_outstanding": 100_000_000,
        }
    )
    out = compute_advanced_technical_features(frame)
    for column in ["momentum_12m_1m", "vol_ratio", "rsi_14", "macd_signal", "bb_position", "golden_cross"]:
        assert column in out.columns
    assert out["price_to_sma_200"].notna().any()


def test_advanced_fundamental_features_include_quality_value_growth() -> None:
    frame = pd.DataFrame(
        {
            "ticker": ["AAA"] * 800,
            "revenue_ttm": np.linspace(1_000, 1_600, 800),
            "net_income": np.linspace(80, 140, 800),
            "total_assets": np.linspace(2_000, 2_400, 800),
            "total_equity": np.linspace(900, 1_100, 800),
            "total_debt": 300.0,
            "cash": 120.0,
            "ebit": 180.0,
            "ebitda": 220.0,
            "gross_profit": np.linspace(450, 650, 800),
            "operating_income": 170.0,
            "free_cash_flow": np.linspace(90, 130, 800),
            "operating_cash_flow": np.linspace(100, 150, 800),
            "current_assets": 600.0,
            "current_liabilities": 300.0,
            "interest_expense": 20.0,
            "market_cap": 3_000.0,
            "price": 30.0,
            "shares_outstanding": 100.0,
        }
    )
    out = compute_advanced_fundamental_features(frame)
    for column in ["roic", "fcf_margin", "interest_coverage", "fcf_yield", "ev_ebitda", "revenue_growth_3y", "piotroski_f_score"]:
        assert column in out.columns
    assert out["interest_coverage"].iloc[-1] > 0


def test_wacc_and_dcf_return_positive_intrinsic_value() -> None:
    payload = {
        "market_cap": 1_000.0,
        "total_debt": 250.0,
        "cash": 100.0,
        "free_cash_flow": 80.0,
        "shares_outstanding": 100.0,
        "current_price": 9.0,
        "beta": 1.1,
        "interest_expense": 12.5,
        "pretax_income": 100.0,
        "tax_expense": 21.0,
    }
    wacc = compute_wacc(payload)
    dcf = compute_dcf_valuation(payload, wacc=wacc["wacc"], growth_stage1=0.03, terminal_growth=0.02, n_years=5)
    assert 0 < wacc["wacc"] < 0.2
    assert dcf["intrinsic_price_dcf"].dropna().iloc[-1] > 0


def test_valuation_multiples_eva_ddm_and_asset_value_are_available() -> None:
    payload = {
        "price": 20.0,
        "market_cap": 2_000.0,
        "enterprise_value": 2_200.0,
        "revenue_ttm": 1_000.0,
        "ebitda": 250.0,
        "ebit": 180.0,
        "free_cash_flow": 120.0,
        "operating_cash_flow": 150.0,
        "eps_ttm": 2.0,
        "total_equity": 900.0,
        "total_debt": 300.0,
        "cash": 100.0,
        "shares_outstanding": 100.0,
        "annual_dividend": 0.8,
        "net_income": 200.0,
        "total_assets": 2_500.0,
        "total_liabilities": 1_600.0,
        "current_assets": 800.0,
        "ppe": 1_000.0,
        "beta": 1.0,
    }
    multiples = compute_market_multiples(payload)
    eva = compute_eva_residual_income(payload, wacc=0.09)
    ddm = compute_ddm_valuation(payload, cost_of_equity=0.10)
    asset = compute_asset_based_valuation(payload)
    assert multiples["ev_ebitda"] > 0
    assert "wacc_spread" in eva
    assert ddm["intrinsic_price_ddm"] > 0
    assert asset["book_value_per_share"] > 0


def test_comps_valuation_uses_sector_medians() -> None:
    company = {"ticker": "AAA", "sector": "Tech", "price": 20.0, "eps_ttm": 2.0, "ebitda": 200.0, "total_debt": 100.0, "cash": 50.0, "shares_outstanding": 100.0}
    peers = pd.DataFrame(
        [
            {"ticker": "BBB", "sector": "Tech", "price": 30.0, "eps_ttm": 3.0, "ebitda": 300.0, "enterprise_value": 2_400.0, "market_cap": 2_000.0},
            {"ticker": "CCC", "sector": "Tech", "price": 40.0, "eps_ttm": 4.0, "ebitda": 400.0, "enterprise_value": 3_600.0, "market_cap": 3_000.0},
        ]
    )
    comps = compute_comps_valuation(company, peers)
    assert comps["sector_pe_median"] > 0
    assert "implied_price_pe" in comps


def test_portfolio_performance_metrics_include_advanced_risk() -> None:
    returns = pd.Series([0.01, -0.02, 0.015, 0.005, -0.01, 0.02] * 20)
    benchmark = pd.Series([0.006, -0.01, 0.01, 0.004, -0.008, 0.015] * 20)
    metrics = compute_portfolio_performance_metrics(returns, benchmark, turnover=1.2, assumed_cost_bp=10)
    for key in ["calmar_ratio", "omega_ratio", "ulcer_index", "tail_ratio", "information_ratio", "net_sharpe"]:
        assert key in metrics


def test_portfolio_optimizers_return_valid_weights_and_frontier() -> None:
    rng = np.random.default_rng(7)
    returns = pd.DataFrame(rng.normal(0.0005, 0.01, size=(252, 4)), columns=["A", "B", "C", "D"])
    min_var = compute_min_variance_weights(returns)
    max_sharpe = compute_max_sharpe_weights(returns)
    rp_weights, rp_contrib = compute_risk_parity_weights(returns)
    frontier = compute_efficient_frontier(returns, n_points=8)
    assert abs(sum(min_var.values()) - 1.0) < 1e-6
    assert abs(sum(max_sharpe.values()) - 1.0) < 1e-6
    assert abs(sum(rp_weights.values()) - 1.0) < 1e-6
    assert not rp_contrib.empty
    assert len(frontier) == 8
