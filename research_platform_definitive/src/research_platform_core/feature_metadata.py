"""Readable feature and factor metadata for the Gen.is.IA workstation.

The registry is intentionally lightweight: it does not compute any signal and
does not change model behavior.  It gives Streamlit pages a single vocabulary
for tooltips, compact explainers and analyst-facing glossary tables.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class FeatureMetadata:
    id: str
    name: str
    category: str
    description: str
    formula: str
    interpretation: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


FEATURE_METADATA: dict[str, FeatureMetadata] = {
    "value_score": FeatureMetadata(
        "value_score",
        "Value score",
        "value",
        "Cross-sectional value composite based on valuation multiples and shareholder yield where available.",
        "percentile(low P/E, low P/B, low EV/EBITDA, high dividend yield)",
        "Higher values indicate cheaper names relative to the current universe.",
    ),
    "valuation_score": FeatureMetadata(
        "valuation_score",
        "Valuation score",
        "value",
        "Model or rule-based valuation percentile used by the Screener and valuation overlays.",
        "cross-sectional percentile of available valuation inputs",
        "Higher values indicate a more attractive valuation profile.",
    ),
    "quality_score": FeatureMetadata(
        "quality_score",
        "Quality score",
        "quality",
        "Composite of profitability, capital efficiency, margin strength and balance-sheet discipline.",
        "percentile(ROE, ROIC, margins, inverse leverage)",
        "Higher values indicate stronger business quality and cleaner financial risk.",
    ),
    "momentum_score": FeatureMetadata(
        "momentum_score",
        "Momentum score",
        "momentum",
        "Intermediate-term price momentum score, preferring 12-1 month momentum when available.",
        "percentile(momentum_12_1 or trailing return)",
        "Higher values indicate stronger relative price trend.",
    ),
    "momentum_12_1": FeatureMetadata(
        "momentum_12_1",
        "12-1M momentum",
        "momentum",
        "Twelve-month return excluding the most recent month, a standard academic momentum proxy.",
        "(1 + ret252d) / (1 + ret21d) - 1",
        "Higher values indicate persistent medium-term trend after excluding near-term reversal noise.",
    ),
    "momentum_12_1_score": FeatureMetadata(
        "momentum_12_1_score",
        "12-1M momentum score",
        "momentum",
        "Cross-sectional percentile rank of 12-1 month momentum.",
        "percentile(momentum_12_1)",
        "Higher values identify stronger relative trend names.",
    ),
    "risk_score": FeatureMetadata(
        "risk_score",
        "Low-volatility / risk score",
        "risk",
        "Composite proxy for lower realized volatility, lower beta and smaller drawdowns.",
        "percentile(inverse volatility, inverse beta, inverse drawdown)",
        "Higher values indicate a lower-risk profile versus peers.",
    ),
    "size_score": FeatureMetadata(
        "size_score",
        "Size / liquidity score",
        "size",
        "Market-cap and liquidity proxy used to stabilize investability screens and model universes.",
        "percentile(log market value or market cap)",
        "Higher values indicate larger and typically more liquid securities.",
    ),
    "growth_score": FeatureMetadata(
        "growth_score",
        "Growth score",
        "growth",
        "Revenue or earnings growth composite where point-in-time fundamentals are available.",
        "percentile(revenue growth, revenue CAGR, EPS growth)",
        "Higher values indicate stronger growth evidence.",
    ),
    "factor_composite_score": FeatureMetadata(
        "factor_composite_score",
        "Factor composite",
        "composite",
        "Desk-level blend of canonical factor blocks such as value, quality, momentum, risk, size and growth.",
        "weighted average of enabled factor scores",
        "Higher values indicate stronger multi-factor conviction.",
    ),
    "ml_score": FeatureMetadata(
        "ml_score",
        "ML score",
        "ml",
        "Normalized machine-learning score used by the Screener and ML Stock Lab.",
        "model prediction transformed to 0-100 rank/score",
        "Higher values indicate stronger model-implied attractiveness, not a standalone recommendation.",
    ),
    "score_composite": FeatureMetadata(
        "score_composite",
        "Composite model score",
        "ml",
        "Weighted blend of active ML models selected in Model Settings.",
        "sum(weight_i * score_model_i)",
        "Higher values indicate stronger ensemble conviction across enabled models.",
    ),
    "score_ols": FeatureMetadata(
        "score_ols",
        "OLS score",
        "ml",
        "Linear baseline score from the interpretable OLS-style model.",
        "normalized OLS prediction",
        "Useful as an interpretable benchmark against non-linear models.",
    ),
    "score_rf": FeatureMetadata(
        "score_rf",
        "Random Forest score",
        "ml",
        "Tree ensemble score that can capture non-linear interactions across factor blocks.",
        "normalized Random Forest prediction",
        "Higher values indicate stronger non-linear model signal; inspect feature importance for drivers.",
    ),
    "score_gbrt": FeatureMetadata(
        "score_gbrt",
        "GBRT score",
        "ml",
        "Gradient-boosted tree score for cross-sectional return or ranking prediction.",
        "normalized gradient boosting prediction",
        "Higher values indicate stronger boosting-model signal; monitor overfit and stability.",
    ),
    "fair_value_hat": FeatureMetadata(
        "fair_value_hat",
        "Estimated fair value",
        "valuation",
        "Model-implied fair value from valuation or ML valuation artifacts.",
        "valuation model output",
        "Compare with market price; use with assumptions and uncertainty bands.",
    ),
    "mispricing_rel": FeatureMetadata(
        "mispricing_rel",
        "Relative mispricing",
        "valuation",
        "Relative gap between estimated fair value and current market value.",
        "(fair_value_hat / market_value) - 1",
        "Positive values indicate estimated upside; negative values indicate estimated downside.",
    ),
    "zscore": FeatureMetadata(
        "zscore",
        "Mispricing z-score",
        "valuation",
        "Standardized valuation gap used for ranking and anomaly checks.",
        "(mispricing - universe_mean) / universe_std",
        "Higher absolute values indicate more unusual valuation gaps.",
    ),
    "valuation_signal_score": FeatureMetadata(
        "valuation_signal_score",
        "Valuation signal score",
        "valuation",
        "Normalized valuation attractiveness score used in Screener and Portfolio overlays.",
        "percentile(valuation gap or valuation composite)",
        "Higher values indicate stronger valuation support.",
    ),
    "smart_money_score": FeatureMetadata(
        "smart_money_score",
        "Smart Money score",
        "smart_money",
        "Issuer-level evidence score from ownership, insider, government, flow or positioning artifacts.",
        "weighted evidence score from Smart Money artifacts",
        "Higher values indicate stronger institutional/event evidence; inspect event details before use.",
    ),
    "composite_institutional_interest_score": FeatureMetadata(
        "composite_institutional_interest_score",
        "Institutional interest composite",
        "smart_money",
        "Composite Smart Money score focused on ownership and event confirmation.",
        "weighted ownership/event percentile blend",
        "Higher values indicate stronger institutional confirmation.",
    ),
    "return_1m": FeatureMetadata(
        "return_1m",
        "1M return",
        "price",
        "Recent one-month price return.",
        "price_t / price_t-21 - 1",
        "Useful for near-term trend checks; can be noisy.",
    ),
    "return_3m": FeatureMetadata(
        "return_3m",
        "3M return",
        "price",
        "Recent three-month price return.",
        "price_t / price_t-63 - 1",
        "Useful for medium-term confirmation and macro/sector pulse.",
    ),
    "return_1y": FeatureMetadata(
        "return_1y",
        "1Y return",
        "price",
        "Trailing one-year price return.",
        "price_t / price_t-252 - 1",
        "Useful for trend context but overlaps with momentum features unless lagged.",
    ),
    "vol63d": FeatureMetadata(
        "vol63d",
        "63D realized volatility",
        "risk",
        "Three-month realized volatility proxy.",
        "std(daily returns, 63 days) * sqrt(252)",
        "Higher values indicate more short-term risk.",
    ),
    "vol252d": FeatureMetadata(
        "vol252d",
        "252D realized volatility",
        "risk",
        "One-year realized volatility proxy.",
        "std(daily returns, 252 days) * sqrt(252)",
        "Higher values indicate more long-term realized risk.",
    ),
    "pe": FeatureMetadata(
        "pe",
        "P/E",
        "value",
        "Price to earnings multiple.",
        "market price / earnings per share",
        "Lower values are cheaper if earnings quality is comparable.",
    ),
    "pb": FeatureMetadata(
        "pb",
        "P/B",
        "value",
        "Price to book multiple.",
        "market capitalization / book equity",
        "Lower values can indicate value, but financials and distressed firms need caution.",
    ),
    "ev_ebitda": FeatureMetadata(
        "ev_ebitda",
        "EV/EBITDA",
        "value",
        "Enterprise value relative to EBITDA.",
        "enterprise value / EBITDA",
        "Lower values are cheaper on operating cash-flow proxy, subject to capex and debt quality.",
    ),
    "roe": FeatureMetadata(
        "roe",
        "ROE",
        "quality",
        "Return on equity.",
        "net income / book equity",
        "Higher values indicate stronger profitability, but can be boosted by leverage.",
    ),
    "debt_to_equity": FeatureMetadata(
        "debt_to_equity",
        "Debt / equity",
        "risk",
        "Balance-sheet leverage proxy.",
        "total debt / book equity",
        "Higher values indicate more financial leverage and usually more risk.",
    ),
}

FEATURE_METADATA.update(
    {
        "ev_ebit": FeatureMetadata(
            "ev_ebit",
            "EV/EBIT",
            "value",
            "Enterprise value relative to operating profit before interest and tax.",
            "enterprise value / EBIT",
            "Lower values are cheaper on operating earnings, but cyclical margins require caution.",
        ),
        "dividend_yield": FeatureMetadata(
            "dividend_yield",
            "Dividend yield",
            "value",
            "Cash dividend yield relative to current market price.",
            "annual dividends per share / market price",
            "Higher values support shareholder yield, but can also flag distress if unsustainable.",
        ),
        "roic": FeatureMetadata(
            "roic",
            "ROIC",
            "quality",
            "Return on invested capital, a capital-efficiency measure less leverage-sensitive than ROE.",
            "NOPAT / invested capital",
            "Higher values indicate stronger economic profitability versus capital employed.",
        ),
        "gross_margin": FeatureMetadata(
            "gross_margin",
            "Gross margin",
            "quality",
            "Share of revenue retained after cost of goods sold.",
            "gross profit / revenue",
            "Higher values suggest pricing power or cost advantage, subject to industry mix.",
        ),
        "operating_margin": FeatureMetadata(
            "operating_margin",
            "Operating margin",
            "quality",
            "Operating profit margin after core operating expenses.",
            "operating income / revenue",
            "Higher values suggest better operating efficiency and business quality.",
        ),
        "price": FeatureMetadata(
            "price",
            "Price",
            "price",
            "Latest adjusted or close price used by the factor panel.",
            "latest close or adjusted close",
            "Used as market context; not a predictive feature by itself unless transformed.",
        ),
        "ret21d": FeatureMetadata(
            "ret21d",
            "21D return",
            "momentum",
            "Approximate one-month trailing return.",
            "price_t / price_t-21 - 1",
            "Higher values indicate positive near-term momentum, but can be reversal-prone.",
        ),
        "ret63d": FeatureMetadata(
            "ret63d",
            "63D return",
            "momentum",
            "Approximate three-month trailing return.",
            "price_t / price_t-63 - 1",
            "Higher values indicate medium-term price strength.",
        ),
        "ret126d": FeatureMetadata(
            "ret126d",
            "126D return",
            "momentum",
            "Approximate six-month trailing return.",
            "price_t / price_t-126 - 1",
            "Higher values indicate sustained trend over a half-year horizon.",
        ),
        "ret252d": FeatureMetadata(
            "ret252d",
            "252D return",
            "momentum",
            "Approximate one-year trailing return.",
            "price_t / price_t-252 - 1",
            "Higher values indicate strong long-horizon trend; avoid mixing with forward targets without lag discipline.",
        ),
        "vol126d": FeatureMetadata(
            "vol126d",
            "126D realized volatility",
            "risk",
            "Six-month realized volatility proxy.",
            "std(daily returns, 126 days) * sqrt(252)",
            "Higher values indicate more medium-term price risk.",
        ),
        "beta": FeatureMetadata(
            "beta",
            "Beta",
            "risk",
            "Sensitivity to the selected benchmark or market factor.",
            "cov(asset_return, benchmark_return) / var(benchmark_return)",
            "Higher values indicate stronger market exposure.",
        ),
        "max_drawdown": FeatureMetadata(
            "max_drawdown",
            "Maximum drawdown",
            "risk",
            "Largest peak-to-trough price loss over the measurement window.",
            "min(price / running_max(price) - 1)",
            "More negative values indicate worse downside path risk.",
        ),
        "market_value": FeatureMetadata(
            "market_value",
            "Market value",
            "size",
            "Market capitalization or market-value proxy used for investability.",
            "shares outstanding * market price",
            "Higher values usually indicate larger, more liquid and more investable names.",
        ),
        "log_market_value": FeatureMetadata(
            "log_market_value",
            "Log market value",
            "size",
            "Log-transformed market value used to stabilize size effects.",
            "ln(market_value)",
            "Higher values indicate larger companies while reducing extreme scale effects.",
        ),
        "market_cap": FeatureMetadata(
            "market_cap",
            "Market capitalization",
            "size",
            "Equity market value of the company.",
            "shares outstanding * market price",
            "Higher values indicate larger companies and usually better capacity/liquidity.",
        ),
        "marketcap": FeatureMetadata(
            "marketcap",
            "Market capitalization",
            "size",
            "Provider-specific market-cap column mapped into the size factor family.",
            "shares outstanding * market price",
            "Higher values indicate larger companies and usually better capacity/liquidity.",
        ),
        "revenue_growth": FeatureMetadata(
            "revenue_growth",
            "Revenue growth",
            "growth",
            "Recent revenue growth proxy from fundamentals.",
            "revenue_t / revenue_t-1 - 1",
            "Higher values indicate stronger top-line expansion.",
        ),
        "revenue_cagr": FeatureMetadata(
            "revenue_cagr",
            "Revenue CAGR",
            "growth",
            "Compounded annual revenue growth over a multi-year window.",
            "(revenue_end / revenue_start)^(1 / years) - 1",
            "Higher values indicate more durable growth if not driven by one-off effects.",
        ),
        "sales_cagr": FeatureMetadata(
            "sales_cagr",
            "Sales CAGR",
            "growth",
            "Compounded annual sales growth over a multi-year window.",
            "(sales_end / sales_start)^(1 / years) - 1",
            "Higher values indicate stronger historical sales compounding.",
        ),
        "eps_growth": FeatureMetadata(
            "eps_growth",
            "EPS growth",
            "growth",
            "Earnings-per-share growth proxy.",
            "EPS_t / EPS_t-1 - 1",
            "Higher values indicate stronger earnings growth, subject to buybacks and cyclicality.",
        ),
        "model_mispricing": FeatureMetadata(
            "model_mispricing",
            "Model mispricing",
            "model_based",
            "Generic valuation-model gap exported by model-based valuation artifacts.",
            "model_value / market_value - 1",
            "Positive values indicate estimated upside from the model; inspect assumptions first.",
        ),
        "model_mispricing_zscore": FeatureMetadata(
            "model_mispricing_zscore",
            "Model mispricing z-score",
            "model_based",
            "Standardized model mispricing relative to the current cross-section.",
            "(model_mispricing - mean_t) / std_t",
            "Higher absolute values indicate more unusual model gaps.",
        ),
        "model_mispricing_rank": FeatureMetadata(
            "model_mispricing_rank",
            "Model mispricing rank",
            "model_based",
            "Cross-sectional rank of model-implied mispricing.",
            "rank(model_mispricing)",
            "Higher ranks indicate stronger model-implied upside versus peers.",
        ),
        "dcf_mispricing": FeatureMetadata(
            "dcf_mispricing",
            "DCF mispricing",
            "model_based",
            "Relative value gap from discounted cash-flow artifacts.",
            "DCF fair value / market value - 1",
            "Positive values indicate DCF-implied upside, sensitive to WACC and terminal growth.",
        ),
        "residual_income_mispricing": FeatureMetadata(
            "residual_income_mispricing",
            "Residual income mispricing",
            "model_based",
            "Relative value gap from residual-income valuation artifacts.",
            "residual income fair value / market value - 1",
            "Positive values indicate upside under book value, ROE and cost-of-equity assumptions.",
        ),
        "eva_mispricing": FeatureMetadata(
            "eva_mispricing",
            "EVA mispricing",
            "model_based",
            "Relative value gap from economic-value-added valuation artifacts.",
            "EVA fair value / market value - 1",
            "Positive values indicate upside under economic-profit assumptions.",
        ),
        "forward_return": FeatureMetadata(
            "forward_return",
            "Forward return target",
            "target",
            "Model target return over the configured forecast horizon.",
            "price_t+h / price_t - 1",
            "This is a target/label, not an input feature; it must remain leakage-protected.",
        ),
        "forward_return_21d": FeatureMetadata(
            "forward_return_21d",
            "21D forward return",
            "target",
            "One-month forward return target.",
            "price_t+21 / price_t - 1",
            "Used for training/validation only; never as an input feature at time t.",
        ),
        "forward_return_63d": FeatureMetadata(
            "forward_return_63d",
            "63D forward return",
            "target",
            "Three-month forward return target.",
            "price_t+63 / price_t - 1",
            "Used for training/validation only; never as an input feature at time t.",
        ),
        "forward_return_252d": FeatureMetadata(
            "forward_return_252d",
            "252D forward return",
            "target",
            "One-year forward return target.",
            "price_t+252 / price_t - 1",
            "Used for training/validation only; never as an input feature at time t.",
        ),
        "target_horizon_days": FeatureMetadata(
            "target_horizon_days",
            "Target horizon",
            "target",
            "Forecast horizon, in trading days, used for the selected target.",
            "h in forward_return_h",
            "Must be visible in UI/model cards so users do not mix short-term and long-term signals.",
        ),
        "prediction": FeatureMetadata(
            "prediction",
            "Model prediction",
            "ml",
            "Raw model prediction before ranking or normalization.",
            "f_t(X)",
            "Interpret relative to target definition and calibration; not a recommendation by itself.",
        ),
        "actual": FeatureMetadata(
            "actual",
            "Realized target",
            "target",
            "Realized outcome used to evaluate model predictions.",
            "observed forward_return_h",
            "Validation-only field; should not enter feature selection.",
        ),
        "prediction_rank": FeatureMetadata(
            "prediction_rank",
            "Prediction rank",
            "ml",
            "Cross-sectional rank of model predictions.",
            "rank(prediction) within date/universe",
            "Higher ranks indicate better model-implied position in the cross-section.",
        ),
        "percentile_rank": FeatureMetadata(
            "percentile_rank",
            "Percentile rank",
            "composite",
            "0-100 percentile ranking used by score tables.",
            "rank(value, pct=True) * 100",
            "Higher values indicate stronger relative standing for the chosen signal.",
        ),
        "signal": FeatureMetadata(
            "signal",
            "Signal",
            "composite",
            "Generic directional signal emitted by an artifact.",
            "artifact-specific score or rank",
            "Interpret using the source artifact and supporting score columns.",
        ),
        "score": FeatureMetadata(
            "score",
            "Generic score",
            "composite",
            "Artifact-level score when a more specific score name is not available.",
            "artifact-specific normalized score",
            "Use the source table to identify whether this is ML, valuation, screening or Smart Money.",
        ),
        "screener_score": FeatureMetadata(
            "screener_score",
            "Screener score",
            "composite",
            "Rule-based score produced by the company screener artifact.",
            "weighted ranking of screener criteria",
            "Higher values indicate stronger match to the configured screener logic.",
        ),
        "selection_score": FeatureMetadata(
            "selection_score",
            "Portfolio selection score",
            "portfolio",
            "Ranking score used by portfolio selection artifacts.",
            "weighted blend of ranking, risk and constraint-aware inputs",
            "Higher values indicate stronger candidate status for portfolio construction.",
        ),
        "composite_score": FeatureMetadata(
            "composite_score",
            "Composite score",
            "composite",
            "General combined score used by portfolio or screening artifacts.",
            "weighted average of enabled source scores",
            "Higher values indicate stronger combined conviction, subject to the artifact recipe.",
        ),
        "annual_return": FeatureMetadata(
            "annual_return",
            "Annual return",
            "portfolio",
            "Annualized return estimate or realized annual return from portfolio artifacts.",
            "periodic_return annualized",
            "Higher values are attractive only after risk, turnover and drawdown review.",
        ),
        "scenario_downside": FeatureMetadata(
            "scenario_downside",
            "Scenario downside",
            "valuation",
            "Downside estimate under adverse valuation or stress assumptions.",
            "bear_case_value / current_value - 1",
            "More negative values indicate greater modeled downside risk.",
        ),
        "quality_flag": FeatureMetadata(
            "quality_flag",
            "Quality flag",
            "quality",
            "Boolean or categorical quality-screen indicator.",
            "artifact-specific pass/fail quality rule",
            "Use as a quick diagnostic, then inspect the underlying quality metrics.",
        ),
        "macro_spy_ret21d": FeatureMetadata("macro_spy_ret21d", "SPY 21d macro momentum", "macro_context", "Lagged 21-trading-day return of SPY from Macro DB.", "SPY_t / SPY_t-21 - 1, lagged one row", "Positive values indicate supportive US equity momentum."),
        "macro_spy_ret63d": FeatureMetadata("macro_spy_ret63d", "SPY 63d macro momentum", "macro_context", "Lagged 63-trading-day return of SPY from Macro DB.", "SPY_t / SPY_t-63 - 1, lagged one row", "Positive values indicate supportive medium-term risk appetite."),
        "macro_dxy_ret21d": FeatureMetadata("macro_dxy_ret21d", "DXY 21d move", "macro_context", "Lagged 21-day move in the US Dollar Index proxy.", "DXY_t / DXY_t-21 - 1, lagged one row", "Sharp positive values can signal USD tightening pressure."),
        "macro_dxy_ret63d": FeatureMetadata("macro_dxy_ret63d", "DXY 63d move", "macro_context", "Lagged 63-day move in the US Dollar Index proxy.", "DXY_t / DXY_t-63 - 1, lagged one row", "Positive values can be a headwind for global risk assets and EM exposures."),
        "macro_brent_ret21d": FeatureMetadata("macro_brent_ret21d", "Brent 21d momentum", "macro_context", "Lagged 21-day return of Brent crude future proxy.", "Brent_t / Brent_t-21 - 1, lagged one row", "Captures energy price impulse for macro context."),
        "macro_brent_ret63d": FeatureMetadata("macro_brent_ret63d", "Brent 63d momentum", "macro_context", "Lagged 63-day return of Brent crude future proxy.", "Brent_t / Brent_t-63 - 1, lagged one row", "Captures medium-term energy cycle pressure."),
        "macro_wti_ret21d": FeatureMetadata("macro_wti_ret21d", "WTI 21d momentum", "macro_context", "Lagged 21-day return of WTI crude future proxy.", "WTI_t / WTI_t-21 - 1, lagged one row", "Captures energy beta context."),
        "macro_wti_ret63d": FeatureMetadata("macro_wti_ret63d", "WTI 63d momentum", "macro_context", "Lagged 63-day return of WTI crude future proxy.", "WTI_t / WTI_t-63 - 1, lagged one row", "Captures medium-term oil trend context."),
        "macro_tlt_ret21d": FeatureMetadata("macro_tlt_ret21d", "TLT 21d move", "macro_context", "Lagged 21-day return of long-duration US Treasury ETF proxy.", "TLT_t / TLT_t-21 - 1, lagged one row", "Positive values can indicate falling long rates or flight-to-quality demand."),
        "macro_tlt_ret63d": FeatureMetadata("macro_tlt_ret63d", "TLT 63d move", "macro_context", "Lagged 63-day return of long-duration US Treasury ETF proxy.", "TLT_t / TLT_t-63 - 1, lagged one row", "Captures duration/rates regime context."),
        "macro_gld_ret21d": FeatureMetadata("macro_gld_ret21d", "Gold 21d momentum", "macro_context", "Lagged 21-day return of GLD gold ETF proxy.", "GLD_t / GLD_t-21 - 1, lagged one row", "Positive values can indicate defensive, inflation or USD-sensitive demand."),
        "macro_gld_ret63d": FeatureMetadata("macro_gld_ret63d", "Gold 63d momentum", "macro_context", "Lagged 63-day return of GLD gold ETF proxy.", "GLD_t / GLD_t-63 - 1, lagged one row", "Captures medium-term precious metal context."),
        "macro_btc_ret21d": FeatureMetadata("macro_btc_ret21d", "Bitcoin 21d momentum", "macro_context", "Lagged 21-day return of Bitcoin proxy.", "BTC_t / BTC_t-21 - 1, lagged one row", "High values indicate crypto risk appetite, with high noise."),
        "macro_btc_ret63d": FeatureMetadata("macro_btc_ret63d", "Bitcoin 63d momentum", "macro_context", "Lagged 63-day return of Bitcoin proxy.", "BTC_t / BTC_t-63 - 1, lagged one row", "Medium-term crypto risk appetite proxy."),
        "macro_credit_hyg_tlt_ret63d": FeatureMetadata("macro_credit_hyg_tlt_ret63d", "Credit risk-on spread", "macro_context", "Lagged 63-day relative return of HYG versus TLT.", "return_63d(HYG) - return_63d(TLT), lagged one row", "Positive values indicate credit outperforming duration, a risk-on credit signal."),
        "macro_credit_lqd_tlt_ret63d": FeatureMetadata("macro_credit_lqd_tlt_ret63d", "IG credit duration spread", "macro_context", "Lagged 63-day relative return of LQD versus TLT.", "return_63d(LQD) - return_63d(TLT), lagged one row", "Positive values indicate investment-grade credit outperforming long Treasuries."),
        "macro_curve_tnx_irx": FeatureMetadata("macro_curve_tnx_irx", "US curve proxy", "macro_context", "Difference between US 10Y yield proxy and front-end Treasury yield proxy.", "TNX - IRX, lagged one row", "Higher values indicate a steeper yield-curve proxy."),
        "macro_curve_tlt_shy_ret63d": FeatureMetadata("macro_curve_tlt_shy_ret63d", "Duration curve proxy", "macro_context", "Fallback curve proxy using relative TLT versus SHY returns.", "return_63d(TLT) - return_63d(SHY), lagged one row", "Positive values indicate long duration outperforming short duration."),
        "macro_vix_level": FeatureMetadata("macro_vix_level", "VIX level", "macro_context", "Lagged level of the VIX implied-volatility proxy.", "VIX_t, lagged one row", "Higher values indicate higher equity-market stress."),
        "macro_vix_change21d": FeatureMetadata("macro_vix_change21d", "VIX 21d change", "macro_context", "Lagged 21-day point change in VIX.", "VIX_t - VIX_t-21, lagged one row", "Positive values indicate rising stress."),
        "macro_risk_on_score": FeatureMetadata("macro_risk_on_score", "Macro risk-on score", "macro_context", "Simple 0-100 composite of lagged equity, credit, dollar and volatility conditions.", "mean(pass conditions) * 100", "Higher values indicate a more supportive risk regime."),
        "macro_context_score": FeatureMetadata("macro_context_score", "Macro context score", "macro_context", "Current macro_context composite used as an optional experimental ML feature.", "macro_risk_on_score", "Higher values indicate a more supportive macro context."),
        "net_noncommercial": FeatureMetadata("net_noncommercial", "Net non-commercial COT", "smart_money", "CFTC COT long minus short positions for the selected speculative/non-commercial trader group.", "noncommercial_long - noncommercial_short", "Positive values indicate net long speculative positioning."),
        "net_noncommercial_oi_pct": FeatureMetadata("net_noncommercial_oi_pct", "Net non-commercial % OI", "smart_money", "Net non-commercial positioning scaled by total open interest.", "(noncommercial_long - noncommercial_short) / open_interest", "Positive values indicate net long positioning relative to market size."),
        "weekly_change_net": FeatureMetadata("weekly_change_net", "Weekly net COT change", "smart_money", "Week-over-week change in net non-commercial positioning.", "net_noncommercial_t - net_noncommercial_t-1", "Positive values indicate increasing net long positioning."),
        "net_position_percentile_3y": FeatureMetadata("net_position_percentile_3y", "3Y COT percentile", "smart_money", "Rolling three-year percentile of current net non-commercial positioning.", "percentile_rank(net_noncommercial over 156 weeks)", "High values indicate crowded long positioning versus recent history."),
    }
)


ALIASES = {
    "return_1m": "ret21d",
    "return_3m": "ret63d",
    "return_6m": "ret126d",
    "return_1y": "ret252d",
    "volatility": "vol252d",
    "expected_return": "prediction",
    "screening_score": "screener_score",
    "marketvalue": "market_value",
    "mktcap": "market_cap",
}


def normalize_feature_id(feature: str) -> str:
    return str(feature or "").strip().lower().replace(" ", "_").replace("-", "_")


def metadata_for_feature(feature: str) -> FeatureMetadata | None:
    key = normalize_feature_id(feature)
    key = ALIASES.get(key, key)
    return FEATURE_METADATA.get(key)


def metadata_help(feature: str) -> str:
    meta = metadata_for_feature(feature)
    if meta is None:
        return "No glossary entry yet. Treat this field as an artifact-specific column and inspect its source table."
    return f"{meta.description} Formula: {meta.formula}. Interpretation: {meta.interpretation}"


def metadata_frame(features: Iterable[str] | None = None) -> pd.DataFrame:
    keys = [normalize_feature_id(feature) for feature in features] if features is not None else list(FEATURE_METADATA)
    rows = []
    seen: set[str] = set()
    for key in keys:
        resolved = ALIASES.get(key, key)
        if resolved in seen:
            continue
        seen.add(resolved)
        meta = FEATURE_METADATA.get(resolved)
        if meta is not None:
            rows.append(meta.to_dict())
    return pd.DataFrame(rows)


def category_for_feature(feature: str) -> str:
    meta = metadata_for_feature(feature)
    return meta.category if meta else "artifact"
