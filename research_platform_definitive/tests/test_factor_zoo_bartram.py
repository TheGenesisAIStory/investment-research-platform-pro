from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml_stock_lab.factor_registry import FACTOR_BLOCKS, get_factors_by_asset_class, get_factors_by_category
from research_platform_core.commodity_factors import build_commodity_factors
from research_platform_core.cross_asset_factors import build_cross_asset_momentum, cross_asset_value, global_risk_factor, liquidity_factor
from research_platform_core.feature_metadata import metadata_for_feature
from research_platform_core.fi_factors import build_fi_factors
from research_platform_core.fx_factors import build_fx_factors
from research_platform_core.institutional_factors import compute_institutional_equity_factors, idiosyncratic_volatility
from research_platform_core.smart_money import build_cot_hedging_pressure


def _macro_history(n_days: int = 330) -> pd.DataFrame:
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    symbols = [
        "EURUSD=X",
        "GBPUSD=X",
        "USDJPY=X",
        "DX-Y.NYB",
        "TLT",
        "SHY",
        "HYG",
        "LQD",
        "TIP",
        "CL=F",
        "BZ=F",
        "GC=F",
        "HG=F",
        "SPY",
        "QQQ",
        "BTC-USD",
    ]
    records: list[dict[str, object]] = []
    t = np.arange(n_days, dtype=float)
    for i, symbol in enumerate(symbols):
        drift = 0.00015 * (i + 1)
        close = 100.0 * np.exp(drift * t + 0.02 * np.sin(t / 23.0 + i))
        records.extend({"date": date, "symbol": symbol, "close": value} for date, value in zip(dates, close, strict=True))
    return pd.DataFrame(records)


def test_institutional_equity_factors_smoke() -> None:
    dates = pd.bdate_range("2023-01-02", periods=80)
    panel = pd.DataFrame(
        {
            "ticker": ["AAA"] * len(dates) + ["BBB"] * len(dates),
            "date": list(dates) * 2,
            "close": np.r_[np.linspace(100, 130, len(dates)), np.linspace(80, 76, len(dates))],
            "revenue": 200.0,
            "cogs": 120.0,
            "total_assets": np.r_[np.linspace(100, 120, len(dates)), np.linspace(90, 95, len(dates))],
            "cash_from_operations": 18.0,
            "net_income": 12.0,
            "dividends_paid": 2.0,
            "buybacks": 1.0,
            "stock_issuance": 0.2,
            "market_cap": 500.0,
            "current_assets": 40.0,
            "current_liabilities": 25.0,
            "retained_earnings": 35.0,
            "ebit": 20.0,
            "total_liabilities": 45.0,
        }
    )
    result = compute_institutional_equity_factors(panel)
    expected = {
        "short_term_reversal",
        "gross_profitability",
        "investment_factor",
        "accruals",
        "cash_profitability",
        "earnings_quality",
        "net_payout_yield",
        "distress_risk",
    }
    assert expected.issubset(result.columns)
    assert result["gross_profitability"].dropna().iloc[-1] > 0
    assert result["short_term_reversal"].iloc[:21].isna().all()


def test_idiosyncratic_volatility_lagged_shape() -> None:
    dates = pd.bdate_range("2023-01-02", periods=90)
    market = pd.Series(np.sin(np.arange(90) / 11.0) * 0.005, index=dates)
    returns = pd.DataFrame(
        {
            "AAA": market * 1.2 + 0.001 * np.cos(np.arange(90) / 5.0),
            "BBB": market * 0.7 + 0.0015 * np.sin(np.arange(90) / 7.0),
        },
        index=dates,
    )
    result = idiosyncratic_volatility(returns, market, window=30)
    assert result.shape == returns.shape
    assert result.iloc[:20].isna().all().all()
    assert result.iloc[40:].notna().any().any()


@pytest.mark.parametrize(
    ("builder", "expected_column"),
    [
        (build_fx_factors, "fx_mom_eurusd"),
        (build_fi_factors, "fi_credit_carry_hyg_lqd"),
        (build_commodity_factors, "commodity_mom_brent"),
        (build_cross_asset_momentum, "xasset_mom_spy"),
    ],
)
def test_cross_asset_factor_builders_smoke(builder, expected_column: str) -> None:
    result = builder(_macro_history())
    assert "date" in result.columns
    assert expected_column in result.columns
    assert result[expected_column].iloc[:21].isna().any()
    assert result[expected_column].iloc[260:].notna().any()


def test_cross_asset_value_combines_lagged_value_signals() -> None:
    dates = pd.date_range("2018-01-31", periods=90, freq="ME")
    t = np.arange(len(dates), dtype=float)
    value_signals = {
        "equity": pd.DataFrame({"hml": np.sin(t / 7.0)}, index=dates),
        "fx": pd.DataFrame({"ppp": np.cos(t / 9.0)}, index=dates),
        "fi": pd.DataFrame({"yield_reversion": np.sin(t / 11.0)}, index=dates),
        "commodity": pd.DataFrame({"gold_value": np.cos(t / 13.0)}, index=dates),
    }
    result = cross_asset_value(value_signals, zscore_window=24)
    assert "date" in result.columns
    assert "xasset_value_score" in result.columns
    assert "xasset_value_equity_hml" in result.columns
    assert result["xasset_value_score"].iloc[:12].isna().any()
    assert result["xasset_value_score"].iloc[30:].notna().any()


def test_global_risk_factor_rolling_pca_outputs() -> None:
    dates = pd.date_range("2015-01-31", periods=96, freq="ME")
    t = np.arange(len(dates), dtype=float)
    equity = pd.DataFrame({"spy": 0.01 * np.sin(t / 5.0) + 0.002}, index=dates)
    fx = pd.DataFrame({"dxy": 0.004 * np.cos(t / 7.0)}, index=dates)
    fi = pd.DataFrame({"tlt": -0.006 * np.sin(t / 5.0) + 0.001}, index=dates)
    commodity = pd.DataFrame({"gold": 0.005 * np.cos(t / 8.0)}, index=dates)
    result = global_risk_factor({"equity": equity, "fx": fx, "fi": fi, "commodity": commodity}, window=36)
    assert "date" in result.columns
    assert "global_risk_factor" in result.columns
    assert "global_risk_explained_variance" in result.columns
    assert result["global_risk_factor"].iloc[:12].isna().any()
    assert result["global_risk_factor"].iloc[45:].notna().any()
    assert result["global_risk_explained_variance"].dropna().between(0, 1).all()


def test_liquidity_factor_amihud_and_spread_fallback() -> None:
    dates = pd.bdate_range("2022-01-03", periods=80)
    t = np.arange(len(dates), dtype=float)
    returns = pd.DataFrame(
        {
            "SPY": 0.001 * np.sin(t / 5.0),
            "TLT": 0.0015 * np.cos(t / 7.0),
            "GLD": 0.0012 * np.sin(t / 9.0),
        },
        index=dates,
    )
    dollar_volume = pd.DataFrame(
        {
            "SPY": 1_000_000 + t * 1_000,
            "TLT": 750_000 + t * 800,
            "GLD": 500_000 + t * 500,
        },
        index=dates,
    )
    result = liquidity_factor(returns, dollar_volume, window=21)
    assert "date" in result.columns
    assert "xasset_liquidity_score" in result.columns
    assert "xasset_liquidity_spy" in result.columns
    assert result["xasset_liquidity_score"].iloc[:5].isna().any()
    assert result["xasset_liquidity_score"].iloc[30:].notna().any()

    alternating = pd.DataFrame(
        {
            "SPY": ((-1.0) ** np.arange(len(dates))) * 0.002,
            "TLT": ((-1.0) ** np.arange(len(dates))) * 0.0015,
            "GLD": ((-1.0) ** np.arange(len(dates))) * 0.001,
        },
        index=dates,
    )
    fallback = liquidity_factor(alternating, None, window=21)
    assert "xasset_liquidity_score" in fallback.columns
    assert fallback["xasset_liquidity_score"].iloc[30:].notna().any()


def test_factor_builders_degrade_when_proxy_missing() -> None:
    frame = _macro_history()
    missing_fx = frame[~frame["symbol"].eq("EURUSD=X")]
    result = build_fx_factors(missing_fx)
    assert "date" in result.columns
    assert "fx_mom_eurusd" not in result.columns


def test_cot_hedging_pressure_features() -> None:
    dates = pd.date_range("2022-01-04", periods=70, freq="W-TUE")
    rows: list[dict[str, object]] = []
    for i, date in enumerate(dates):
        rows.append(
            {
                "market_and_exchange_names": "GOLD - COMMODITY EXCHANGE INC.",
                "report_date_as_yyyy_mm_dd": date.isoformat(),
                "noncomm_positions_long_all": 1000 + i * 5,
                "noncomm_positions_short_all": 700 + i * 3,
                "comm_positions_long_all": 800 + i,
                "comm_positions_short_all": 900 + i * 2,
            }
        )
    result = build_cot_hedging_pressure(pd.DataFrame(rows))
    assert "cot_net_noncomm_gold" in result.columns
    assert "cot_hedging_pressure_gold" in result.columns
    assert "cot_z_gold" in result.columns
    assert result["cot_net_noncomm_gold"].iloc[0] != result["cot_net_noncomm_gold"].iloc[-1]


def test_factor_zoo_registry_and_metadata() -> None:
    assert "fx_factors" in FACTOR_BLOCKS
    assert "cross_asset_momentum" in FACTOR_BLOCKS
    assert "cross_asset_value" in FACTOR_BLOCKS
    assert "global_risk_factor" in FACTOR_BLOCKS
    assert "liquidity_factor" in FACTOR_BLOCKS
    assert get_factors_by_asset_class("fx")
    assert get_factors_by_category("momentum")
    for feature_id in [
        "short_term_reversal",
        "gross_profitability",
        "fx_carry_eurusd",
        "fi_term_carry_tlt_shy",
        "commodity_mom_gold",
        "cot_hedging_pressure_gold",
        "xasset_vol_adj_mom_spy",
        "xasset_value_score",
        "global_risk_factor",
        "global_risk_explained_variance",
        "xasset_liquidity_score",
    ]:
        meta = metadata_for_feature(feature_id)
        assert meta is not None, feature_id
        assert meta.source_paper
        assert meta.factor_zoo_category
