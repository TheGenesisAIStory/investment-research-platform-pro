"""Readable feature and factor metadata for the Gen.is.IA workstation.

The registry is intentionally lightweight: it does not compute any signal and
does not change model behavior.  It gives Streamlit pages a single vocabulary
for tooltips, compact explainers and analyst-facing glossary tables.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
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
    sub_category: str = ""
    leakage_risk: bool = False
    data_requirement: tuple[str, ...] = ()
    source_paper: str = ""
    alias: tuple[str, ...] = ()
    asset_class: str = "equity"
    formula_latex: str = ""
    computation_window: str = ""
    normalization: str = ""
    source_doi: str = ""
    economic_rationale: str = ""
    factor_zoo_category: str = ""
    point_in_time_safe: bool = True
    lag_required: str = "none"
    winsorize_bounds: tuple[float, float] = (0.01, 0.99)
    direction: str = "neutral"
    typical_range: str = ""
    implementation_module: str = ""

    def to_dict(self) -> dict[str, object]:
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
        "regime_spy_trend_sign": FeatureMetadata("regime_spy_trend_sign", "SPY trend sign", "macro_regime", "Lagged sign of the 63-day SPY return.", "sign(return_63d(SPY)), lagged one row", "Positive values indicate an equity-friendly trend regime."),
        "regime_vix_regime": FeatureMetadata("regime_vix_regime", "VIX regime flag", "macro_regime", "Discrete stress flag from VIX level.", "1 if VIX > 25; -1 if VIX < 15; else 0, lagged one row", "Higher values indicate higher volatility stress."),
        "regime_yield_curve_slope": FeatureMetadata("regime_yield_curve_slope", "Yield curve slope proxy", "macro_regime", "Lagged 10Y minus front-end yield proxy when available.", "TNX - IRX, lagged one row", "Higher values indicate a steeper curve; negative values flag inversion pressure."),
        "regime_dxy_trend": FeatureMetadata("regime_dxy_trend", "DXY short trend", "macro_regime", "Lagged 21-day return of the dollar index proxy.", "return_21d(DXY), lagged one row", "Positive values indicate dollar strength, often a tightening/liquidity headwind."),
        "regime_gold_trend": FeatureMetadata("regime_gold_trend", "Gold short trend", "macro_regime", "Lagged 21-day return of the gold ETF/future proxy.", "return_21d(GLD), lagged one row", "Positive values can indicate defensive, inflation or real-rate sensitivity."),
        "flow_1w_proxy": FeatureMetadata("flow_1w_proxy", "ETF 1W flow proxy", "smart_money", "Public-data proxy for one-week ETF asset movement from price/AUM metadata.", "delta(close * estimated shares, 5d)", "Directional only; use as a proxy until licensed creations/redemptions are attached."),
        "flow_1m_proxy": FeatureMetadata("flow_1m_proxy", "ETF 1M flow proxy", "smart_money", "Public-data proxy for one-month ETF asset movement from price/AUM metadata.", "delta(close * estimated shares, 21d)", "Positive values suggest assets moving with/into the ETF proxy, subject to price effects."),
        "put_call_ratio_oi": FeatureMetadata("put_call_ratio_oi", "Put/call ratio by OI", "smart_money", "Options put open interest divided by call open interest for the selected ETF/index proxy.", "sum(put open interest) / sum(call open interest)", "Higher values indicate more put-heavy option positioning."),
        "net_noncommercial": FeatureMetadata("net_noncommercial", "Net non-commercial COT", "smart_money", "CFTC COT long minus short positions for the selected speculative/non-commercial trader group.", "noncommercial_long - noncommercial_short", "Positive values indicate net long speculative positioning."),
        "net_noncommercial_oi_pct": FeatureMetadata("net_noncommercial_oi_pct", "Net non-commercial % OI", "smart_money", "Net non-commercial positioning scaled by total open interest.", "(noncommercial_long - noncommercial_short) / open_interest", "Positive values indicate net long positioning relative to market size."),
        "weekly_change_net": FeatureMetadata("weekly_change_net", "Weekly net COT change", "smart_money", "Week-over-week change in net non-commercial positioning.", "net_noncommercial_t - net_noncommercial_t-1", "Positive values indicate increasing net long positioning."),
        "net_position_percentile_3y": FeatureMetadata("net_position_percentile_3y", "3Y COT percentile", "smart_money", "Rolling three-year percentile of current net non-commercial positioning.", "percentile_rank(net_noncommercial over 156 weeks)", "High values indicate crowded long positioning versus recent history."),
    }
)


FEATURE_METADATA.update(
    {
        "ret_1d": FeatureMetadata("ret_1d", "1D return", "price", "One-day simple return.", "close_t / close_t-1 - 1", "Short-horizon price move; noisy and mainly useful for diagnostics.", "returns"),
        "ret_5d": FeatureMetadata("ret_5d", "5D return", "price", "One-week simple return.", "close_t / close_t-5 - 1", "Captures very short-term trend or reversal context.", "returns"),
        "ret_21d": FeatureMetadata("ret_21d", "21D return", "price", "Approximate one-month simple return.", "close_t / close_t-21 - 1", "Positive values indicate short-term momentum.", "returns"),
        "ret_63d": FeatureMetadata("ret_63d", "63D return", "price", "Approximate three-month simple return.", "close_t / close_t-63 - 1", "Positive values indicate medium-term momentum.", "returns"),
        "ret_126d": FeatureMetadata("ret_126d", "126D return", "price", "Approximate six-month simple return.", "close_t / close_t-126 - 1", "Useful as a medium-term trend feature.", "returns"),
        "ret_252d": FeatureMetadata("ret_252d", "252D return", "price", "Approximate one-year simple return.", "close_t / close_t-252 - 1", "One-year trailing trend; use 12-1M momentum for predictive tests.", "returns"),
        "ret_52w_high_proximity": FeatureMetadata("ret_52w_high_proximity", "Distance from 52W high", "momentum", "Percentage distance from the trailing 52-week high.", "(price - high_52w) / high_52w", "Values near zero indicate proximity to the yearly high.", "52_week"),
        "ret_52w_low_proximity": FeatureMetadata("ret_52w_low_proximity", "Distance from 52W low", "risk", "Percentage distance from the trailing 52-week low.", "(price - low_52w) / low_52w", "Low or negative values can flag recent stress.", "52_week"),
        "price_to_52w_high": FeatureMetadata("price_to_52w_high", "Price / 52W high", "momentum", "Ratio of current price to trailing 52-week high.", "price / high_52w", "Values close to 1 indicate strong price trend or breakout proximity.", "52_week"),
        "momentum_reversal_1m": FeatureMetadata("momentum_reversal_1m", "1M reversal", "momentum", "Short-term reversal proxy that penalizes the latest month return.", "-ret_21d", "Higher values indicate weaker recent one-month return, often used as reversal control.", "reversal", False, ("price",), "Jegadeesh 1990"),
        "momentum_12m_1m": FeatureMetadata("momentum_12m_1m", "12-1M momentum", "momentum", "One-year momentum excluding the latest month.", "ret_252d - ret_21d", "Higher values indicate persistent medium-term trend after excluding reversal noise.", "academic_momentum", False, ("price",), "Jegadeesh and Titman 1993"),
        "idiosyncratic_momentum": FeatureMetadata("idiosyncratic_momentum", "Idiosyncratic momentum", "momentum", "Momentum adjusted for broad market momentum exposure.", "stock 12-1M momentum - benchmark momentum proxy", "Higher values indicate stock-specific trend beyond market beta.", "residual_momentum"),
        "vol_21d": FeatureMetadata("vol_21d", "21D realized volatility", "risk", "Annualized one-month realized volatility.", "std(daily returns, 21d) * sqrt(252)", "Higher values indicate elevated short-term risk.", "volatility"),
        "vol_63d": FeatureMetadata("vol_63d", "63D realized volatility", "risk", "Annualized three-month realized volatility.", "std(daily returns, 63d) * sqrt(252)", "Higher values indicate elevated medium-term risk.", "volatility"),
        "vol_126d": FeatureMetadata("vol_126d", "126D realized volatility", "risk", "Annualized six-month realized volatility.", "std(daily returns, 126d) * sqrt(252)", "Higher values indicate higher realized risk.", "volatility"),
        "vol_252d": FeatureMetadata("vol_252d", "252D realized volatility", "risk", "Annualized one-year realized volatility.", "std(daily returns, 252d) * sqrt(252)", "Higher values indicate higher long-term realized risk.", "volatility"),
        "realized_vol_daily": FeatureMetadata("realized_vol_daily", "Realized volatility", "risk", "Annualized realized volatility from daily returns.", "std(daily returns) * sqrt(252)", "Higher values indicate more variable returns.", "volatility"),
        "vol_ratio": FeatureMetadata("vol_ratio", "Short / long vol ratio", "risk", "Ratio of short-term to long-term realized volatility.", "vol_21d / vol_252d", "Values above 1 indicate volatility is rising versus its yearly baseline.", "volatility"),
        "idiosyncratic_vol": FeatureMetadata("idiosyncratic_vol", "Idiosyncratic volatility", "risk", "Annualized residual volatility from rolling market-model residuals.", "std(asset_return - beta * market_return, 252d) * sqrt(252)", "Higher values indicate stock-specific noise/risk.", "residual_risk"),
        "downside_vol_21d": FeatureMetadata("downside_vol_21d", "21D downside volatility", "risk", "Annualized volatility using only negative daily returns.", "std(min(return, 0), 21d) * sqrt(252)", "Higher values indicate larger downside variability.", "downside_risk"),
        "max_drawdown_1y": FeatureMetadata("max_drawdown_1y", "1Y max drawdown", "risk", "Largest peak-to-trough loss over the trailing year.", "min(price / rolling_max(price, 252d) - 1)", "More negative values indicate deeper recent drawdown.", "drawdown"),
        "avg_true_range_21d": FeatureMetadata("avg_true_range_21d", "21D ATR / price", "risk", "Average true range normalized by price.", "mean(true_range, 21d) / close", "Higher values indicate larger intraday/trading-range risk.", "technical"),
        "avg_volume_21d": FeatureMetadata("avg_volume_21d", "21D average volume", "liquidity", "Average daily share volume over the last month.", "mean(volume, 21d)", "Higher values indicate better trading liquidity.", "liquidity"),
        "avg_volume_252d": FeatureMetadata("avg_volume_252d", "252D average volume", "liquidity", "Average daily share volume over the last year.", "mean(volume, 252d)", "Higher values indicate structural liquidity.", "liquidity"),
        "volume_ratio": FeatureMetadata("volume_ratio", "Volume ratio", "liquidity", "Short-term volume relative to long-term volume.", "avg_volume_21d / avg_volume_252d", "Values above 1 indicate elevated trading activity.", "liquidity"),
        "amihud_illiquidity": FeatureMetadata("amihud_illiquidity", "Amihud illiquidity", "liquidity", "Price impact proxy from absolute return per dollar volume.", "mean(abs(return) / dollar_volume, 21d) * 1e6", "Higher values indicate lower liquidity and higher market-impact risk.", "liquidity", False, ("price", "volume"), "Amihud 2002"),
        "turnover_ratio_21d": FeatureMetadata("turnover_ratio_21d", "21D turnover ratio", "liquidity", "Average volume scaled by shares outstanding.", "mean(volume / shares_outstanding, 21d)", "Higher values indicate stronger trading intensity.", "liquidity"),
        "rsi_14": FeatureMetadata("rsi_14", "RSI 14", "technical", "Relative Strength Index over 14 trading days.", "100 - 100/(1 + avg_gain_14 / avg_loss_14)", "High values can indicate overbought trend; low values can indicate oversold conditions.", "oscillator"),
        "macd_signal": FeatureMetadata("macd_signal", "MACD signal spread", "technical", "MACD line minus its 9-day signal line.", "EMA12 - EMA26 - EMA9(MACD)", "Positive values indicate improving trend momentum.", "trend"),
        "bb_position": FeatureMetadata("bb_position", "Bollinger band position", "technical", "Current price location inside Bollinger bands.", "(price - lower_band) / (upper_band - lower_band)", "Values near 1 sit near the upper band; near 0 sit near the lower band.", "bands"),
        "price_to_sma_50": FeatureMetadata("price_to_sma_50", "Price / SMA50", "technical", "Price relative to 50-day moving average.", "price / SMA_50", "Values above 1 indicate price above intermediate trend.", "trend"),
        "price_to_sma_200": FeatureMetadata("price_to_sma_200", "Price / SMA200", "technical", "Price relative to 200-day moving average.", "price / SMA_200", "Values above 1 indicate price above long-term trend.", "trend"),
        "golden_cross": FeatureMetadata("golden_cross", "Golden cross flag", "technical", "Binary flag for SMA50 above SMA200.", "1 if SMA_50 > SMA_200 else 0", "One indicates intermediate trend above long-term trend.", "trend"),
        "roa": FeatureMetadata("roa", "ROA", "quality", "Return on assets.", "net_income / total_assets", "Higher values indicate more profitable asset utilization.", "profitability"),
        "net_margin": FeatureMetadata("net_margin", "Net margin", "quality", "Net income as a share of revenue.", "net_income / revenue", "Higher values indicate stronger bottom-line profitability.", "profitability"),
        "ebitda_margin": FeatureMetadata("ebitda_margin", "EBITDA margin", "quality", "EBITDA as a share of revenue.", "EBITDA / revenue", "Higher values indicate stronger operating cash-profit margin.", "profitability"),
        "fcf_margin": FeatureMetadata("fcf_margin", "FCF margin", "quality", "Free cash flow as a share of revenue.", "free_cash_flow / revenue", "Higher values indicate stronger cash conversion.", "cash_quality"),
        "asset_turnover": FeatureMetadata("asset_turnover", "Asset turnover", "quality", "Revenue generated per unit of assets.", "revenue / total_assets", "Higher values indicate more efficient asset use.", "efficiency"),
        "current_ratio": FeatureMetadata("current_ratio", "Current ratio", "quality", "Short-term assets relative to liabilities.", "current_assets / current_liabilities", "Higher values indicate stronger short-term liquidity.", "financial_health"),
        "quick_ratio": FeatureMetadata("quick_ratio", "Quick ratio", "quality", "Liquid current assets relative to current liabilities.", "(current_assets - inventory) / current_liabilities", "Higher values indicate stronger near-cash liquidity.", "financial_health"),
        "cash_ratio": FeatureMetadata("cash_ratio", "Cash ratio", "quality", "Cash relative to current liabilities.", "cash / current_liabilities", "Higher values indicate more conservative liquidity.", "financial_health"),
        "debt_to_ebitda": FeatureMetadata("debt_to_ebitda", "Debt / EBITDA", "quality", "Gross debt scaled by EBITDA.", "total_debt / EBITDA", "Lower values indicate lower leverage burden.", "leverage"),
        "net_debt_to_ebitda": FeatureMetadata("net_debt_to_ebitda", "Net debt / EBITDA", "quality", "Debt net of cash scaled by EBITDA.", "(total_debt - cash) / EBITDA", "Lower values indicate cleaner balance-sheet risk.", "leverage"),
        "interest_coverage": FeatureMetadata("interest_coverage", "Interest coverage", "quality", "EBIT coverage of interest expense.", "EBIT / interest_expense", "Higher values indicate greater ability to service debt.", "leverage"),
        "altman_z_score": FeatureMetadata("altman_z_score", "Altman Z-score", "quality", "Distress score using the revised non-manufacturing formulation.", "6.56*WC/TA + 3.26*RE/TA + 6.72*EBIT/TA + 1.05*Equity/Liabilities", "Higher values indicate lower distress risk.", "distress", False, ("balance_sheet", "income_statement"), "Altman 1995"),
        "piotroski_f_score": FeatureMetadata("piotroski_f_score", "Piotroski F-Score", "quality", "Nine-point binary score covering profitability, leverage/liquidity and operating efficiency.", "sum(F1..F9 binary accounting tests)", "0-2 weak, 3-6 neutral, 7-9 strong fundamental quality.", "quality_score", False, ("fundamentals",), "Piotroski 2000"),
        "accruals_ratio": FeatureMetadata("accruals_ratio", "Accruals ratio", "quality", "Accrual component of earnings scaled by assets.", "(net_income - operating_cash_flow) / total_assets", "Lower values indicate cleaner cash-backed earnings.", "earnings_quality", False, ("income_statement", "cash_flow"), "Sloan 1996"),
        "book_to_market": FeatureMetadata("book_to_market", "Book-to-market", "value", "Book equity scaled by market capitalization.", "book_equity / market_cap", "Higher values indicate cheaper accounting valuation.", "valuation", False, ("book_equity", "market_cap"), "Fama-French 1993"),
        "earnings_yield": FeatureMetadata("earnings_yield", "Earnings yield", "value", "Earnings scaled by market capitalization.", "net_income / market_cap", "Higher values indicate cheaper earnings valuation.", "valuation"),
        "fcf_yield": FeatureMetadata("fcf_yield", "FCF yield", "value", "Free cash flow scaled by market capitalization.", "free_cash_flow / market_cap", "Higher values indicate more free cash flow per dollar of market value.", "valuation"),
        "ebitda_yield": FeatureMetadata("ebitda_yield", "EBITDA yield", "value", "EBITDA scaled by enterprise value.", "EBITDA / enterprise_value", "Higher values indicate cheaper enterprise valuation.", "valuation"),
        "sales_to_price": FeatureMetadata("sales_to_price", "Sales-to-price", "value", "Revenue scaled by market capitalization.", "revenue / market_cap", "Higher values indicate cheaper sales valuation.", "valuation"),
        "ev_ebit": FeatureMetadata("ev_ebit", "EV/EBIT", "valuation", "Enterprise value relative to operating profit.", "enterprise_value / EBIT", "Lower values are cheaper if EBIT quality is comparable.", "ev_multiples"),
        "ev_sales": FeatureMetadata("ev_sales", "EV/Sales", "valuation", "Enterprise value relative to revenue.", "enterprise_value / revenue", "Lower values are cheaper, but margins and growth matter.", "ev_multiples"),
        "ev_fcf": FeatureMetadata("ev_fcf", "EV/FCF", "valuation", "Enterprise value relative to free cash flow.", "enterprise_value / free_cash_flow", "Lower values indicate cheaper free-cash-flow valuation.", "ev_multiples"),
        "pcf_ratio": FeatureMetadata("pcf_ratio", "P/CF", "valuation", "Market capitalization relative to operating cash flow.", "market_cap / operating_cash_flow", "Lower values indicate cheaper cash-flow valuation.", "price_multiples"),
        "peg_ratio": FeatureMetadata("peg_ratio", "PEG ratio", "valuation", "P/E adjusted for expected earnings growth.", "P/E / earnings_growth", "Lower values indicate cheaper growth-adjusted valuation.", "growth_adjusted"),
        "intrinsic_pb": FeatureMetadata("intrinsic_pb", "Intrinsic P/B", "valuation", "Gordon-style fair price-to-book from ROE, WACC and growth.", "1 + (ROE - WACC) / (WACC - g)", "Higher values indicate justified premium to book when returns exceed cost of capital.", "residual_income"),
        "wacc_spread": FeatureMetadata("wacc_spread", "ROIC - WACC spread", "valuation", "Capital efficiency spread over cost of capital.", "ROIC - WACC", "Positive values indicate value creation.", "eva"),
        "eva": FeatureMetadata("eva", "Economic Value Added", "valuation", "Profit after charging capital at WACC.", "NOPAT - WACC * invested_capital", "Positive EVA indicates value creation after capital cost.", "eva"),
        "revenue_growth_3y": FeatureMetadata("revenue_growth_3y", "3Y revenue CAGR", "growth", "Three-year compound revenue growth.", "(revenue_t / revenue_t-3y)^(1/3) - 1", "Higher values indicate stronger historical top-line growth.", "growth"),
        "earnings_growth_3y": FeatureMetadata("earnings_growth_3y", "3Y EPS CAGR", "growth", "Three-year compound earnings-per-share growth.", "(EPS_t / EPS_t-3y)^(1/3) - 1", "Higher values indicate stronger earnings growth.", "growth"),
        "fcf_growth_1y": FeatureMetadata("fcf_growth_1y", "1Y FCF growth", "growth", "Year-over-year change in free cash flow.", "FCF_t / FCF_t-1y - 1", "Higher values indicate improving cash generation.", "cash_growth"),
        "capex_intensity": FeatureMetadata("capex_intensity", "Capex intensity", "growth", "Capital expenditure as a share of revenue.", "capex / revenue", "Higher values can indicate reinvestment, but also capital intensity.", "reinvestment"),
        "rd_intensity": FeatureMetadata("rd_intensity", "R&D intensity", "growth", "Research and development expense as a share of revenue.", "R&D / revenue", "Higher values can indicate innovation investment.", "reinvestment"),
        "asset_growth": FeatureMetadata("asset_growth", "Asset growth", "growth", "Year-over-year growth in total assets.", "total_assets_t / total_assets_t-1 - 1", "Very high values can indicate expansion or balance-sheet bloat.", "growth", False, ("balance_sheet",), "Cooper, Gulen and Schill 2008"),
        "log_market_cap": FeatureMetadata("log_market_cap", "Log market cap", "size", "Natural log of market capitalization.", "ln(market_cap)", "Higher values indicate larger firms.", "size"),
        "log_total_assets": FeatureMetadata("log_total_assets", "Log total assets", "size", "Natural log of total assets.", "ln(total_assets)", "Higher values indicate larger balance sheets.", "size"),
        "log_revenue": FeatureMetadata("log_revenue", "Log revenue", "size", "Natural log of revenue.", "ln(revenue)", "Higher values indicate larger business scale.", "size"),
        "intrinsic_price_dcf": FeatureMetadata("intrinsic_price_dcf", "DCF intrinsic price", "valuation", "Per-share intrinsic value from discounted cash-flow model.", "equity_value_dcf / shares_outstanding", "Compare with current price and assumptions.", "dcf"),
        "upside_dcf": FeatureMetadata("upside_dcf", "DCF upside", "valuation", "Relative gap between DCF intrinsic price and market price.", "intrinsic_price_dcf / current_price - 1", "Positive values indicate estimated DCF upside.", "dcf"),
        "intrinsic_price_ddm": FeatureMetadata("intrinsic_price_ddm", "DDM intrinsic price", "valuation", "Gordon dividend-discount fair value.", "D1 / (cost_of_equity - dividend_growth)", "Useful only for dividend-paying firms with stable payout policy.", "ddm"),
        "intrinsic_price_ddm_2stage": FeatureMetadata("intrinsic_price_ddm_2stage", "Two-stage DDM price", "valuation", "Present value of high-growth dividends plus terminal Gordon value.", "PV(stage-1 dividends) + PV(terminal DDM)", "Useful for dividend payers with transition growth assumptions.", "ddm"),
        "book_value_per_share": FeatureMetadata("book_value_per_share", "Book value / share", "valuation", "Accounting equity per share.", "total_equity / shares_outstanding", "Higher values support asset-backed valuation checks.", "asset_based"),
        "tangible_book_per_share": FeatureMetadata("tangible_book_per_share", "Tangible book / share", "valuation", "Book value excluding intangibles and goodwill per share.", "(equity - intangibles - goodwill) / shares", "Especially useful for banks and asset-heavy firms.", "asset_based"),
        "net_asset_value": FeatureMetadata("net_asset_value", "Net asset value", "valuation", "Assets net of liabilities, intangibles and goodwill.", "assets - liabilities - intangibles - goodwill", "Asset-backed floor proxy, not a liquidation guarantee.", "asset_based"),
        "liquidation_value_approx": FeatureMetadata("liquidation_value_approx", "Approx. liquidation value", "valuation", "Conservative liquidation proxy using haircuts.", "0.8*current_assets + 0.5*PPE - liabilities", "Rough downside anchor for distressed cases.", "asset_based"),
        "implied_price_pe": FeatureMetadata("implied_price_pe", "Implied price from sector P/E", "valuation", "Price implied by applying sector median P/E to EPS.", "EPS * sector_median_PE", "Above current price implies discount to sector earnings multiple.", "comps"),
        "implied_price_ev_ebitda": FeatureMetadata("implied_price_ev_ebitda", "Implied price from sector EV/EBITDA", "valuation", "Equity price implied by sector EV/EBITDA multiple.", "(EBITDA * sector_median_EV_EBITDA - net_debt) / shares", "Above current price implies discount to sector enterprise multiple.", "comps"),
        "premium_discount_to_sector": FeatureMetadata("premium_discount_to_sector", "Premium / discount to sector", "valuation", "Market price relative to comparable-company implied value.", "current_price / implied_price - 1", "Negative values indicate discount to sector-implied price.", "comps"),
    }
)

VALID_FACTOR_ZOO_CATEGORIES: set[str] = {
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


_PAPER_DOI = {
    "Gen.is.IA internal": "https://genisia.local/methodology/internal",
    "Jensen 1968": "10.2307/2325404",
    "Fama-French 1993": "10.1016/0304-405X(93)90023-5",
    "Fama-French 2015": "10.1016/j.jfineco.2014.10.010",
    "Carhart 1997": "10.1111/j.1540-6261.1997.tb03808.x",
    "Hou-Xue-Zhang 2015": "10.1093/rfs/hhu068",
    "Ang-Hodrick-Xing-Zhang 2006": "10.1111/j.1540-6261.2006.00836.x",
    "Cooper-Gulen-Schill 2008": "10.1111/j.1540-6261.2008.01378.x",
    "Titman-Wei-Xie 2004": "10.2307/3694915",
    "Xing 2008": "10.1093/rfs/hhm028",
    "Pontiff-Woodgate 2008": "10.1111/j.1540-6261.2008.01319.x",
    "Spiess-Affleck-Graves 1999": "10.1016/S0304-405X(99)00029-4",
    "Novy-Marx 2013": "10.1016/j.jfineco.2013.01.003",
    "Desai-Rajgopal-Venkatachalam 2004": "10.2307/3666231",
    "Asness-Frazzini-Pedersen 2019": "10.2139/ssrn.2312432",
    "Stambaugh-Yu-Yuan 2015": "10.1111/jofi.12270",
    "Jegadeesh-Titman 1993": "10.1111/j.1540-6261.1993.tb04702.x",
    "George-Hwang 2004": "10.1111/j.1540-6261.2004.00710.x",
    "Bali-Cakici-Whitelaw 2011": "10.1017/S0022109011000421",
    "Gutierrez-Pirinsky 2007": "10.1111/j.1540-6261.2007.01247.x",
    "Pastor-Stambaugh 2003": "10.1086/374184",
    "Roll 1984": "10.1111/j.1540-6261.1984.tb03646.x",
    "Lesmond-Ogden-Trzcinka 1999": "10.1093/rfs/12.5.1113",
    "Amihud 2002": "10.1016/S0304-405X(01)00024-6",
    "Piotroski 2000": "10.1111/1475-679X.00033",
    "Altman 1995": "https://pages.stern.nyu.edu/~ealtman/Zscores.pdf",
    "Sloan 1996": "10.2307/2491046",
    "Kakushadze 2015": "https://arxiv.org/abs/1601.00991",
}


_CATEGORY_TO_ZOO = {
    "value": "value",
    "valuation": "value",
    "momentum": "momentum",
    "quality": "quality",
    "growth": "investment",
    "size": "size",
    "risk": "risk",
    "volatility": "risk",
    "price": "technical",
    "technical": "technical",
    "liquidity": "trading_frictions",
    "macro": "macro",
    "macro_context": "macro",
    "macro_regime": "macro",
    "smart_money": "smart_money",
    "sentiment": "smart_money",
    "ml": "ml_derived",
    "model": "ml_derived",
    "target": "target",
    "composite": "composite",
    "factor_alpha": "factor_alpha",
    "factor_investment": "investment",
    "factor_profitability": "profitability",
    "factor_zoo_academic": "intangibles",
    "alpha101": "technical",
}


_CATEGORY_DEFAULT_PAPER = {
    "value": "Fama-French 1993",
    "valuation": "Fama-French 1993",
    "momentum": "Jegadeesh-Titman 1993",
    "quality": "Fama-French 2015",
    "growth": "Fama-French 2015",
    "size": "Fama-French 1993",
    "risk": "Ang-Hodrick-Xing-Zhang 2006",
    "volatility": "Ang-Hodrick-Xing-Zhang 2006",
    "price": "Jegadeesh-Titman 1993",
    "technical": "Gen.is.IA internal",
    "liquidity": "Amihud 2002",
    "macro": "Gen.is.IA internal",
    "macro_context": "Gen.is.IA internal",
    "macro_regime": "Gen.is.IA internal",
    "smart_money": "Gen.is.IA internal",
    "sentiment": "Gen.is.IA internal",
    "ml": "Gen.is.IA internal",
    "model": "Gen.is.IA internal",
    "target": "Gen.is.IA internal",
    "composite": "Gen.is.IA internal",
    "factor_alpha": "Jensen 1968",
    "factor_investment": "Fama-French 2015",
    "factor_profitability": "Fama-French 2015",
    "factor_zoo_academic": "Hou-Xue-Zhang 2015",
    "alpha101": "Kakushadze 2015",
}


_ZOO_RATIONALE = {
    "value": "Cheap securities may earn premia when prices overreact to weak fundamentals or when distress risk is overcompensated.",
    "momentum": "Return continuation captures underreaction, slow information diffusion and investor herding before eventual reversal.",
    "profitability": "More profitable firms generate higher cash flows per unit of capital and can sustain stronger future returns.",
    "investment": "Conservative investment policies avoid overexpansion and historically command higher expected returns than aggressive investment.",
    "intangibles": "Intangible investment can create durable assets that are imperfectly captured by accounting book values.",
    "trading_frictions": "Illiquidity and trading constraints require compensation and can reveal limits-to-arbitrage effects.",
    "macro": "Macro and regime variables condition factor payoffs by capturing risk appetite, rates, funding and growth states.",
    "risk": "Risk measures quantify systematic, idiosyncratic and tail exposure that investors may require compensation to bear.",
    "quality": "High-quality balance sheets and cash-backed earnings reduce distress and earnings-manipulation risk.",
    "smart_money": "Positioning, ownership and flow indicators summarize informed or constrained investor demand.",
    "ml_derived": "Model-derived scores compress many weak signals into a calibrated cross-sectional ranking.",
    "target": "Targets define realized future outcomes used only for training/evaluation, never as live predictors.",
    "composite": "Composite signals diversify single-factor noise by blending economically related predictors.",
    "technical": "Technical variables capture short-horizon trend, reversal and trading-state information from prices and volume.",
    "size": "Firm size proxies investability, liquidity and the historical small-firm return premium.",
    "factor_alpha": "Residual alpha measures performance unexplained by standard risk factors and highlights abnormal return persistence.",
}


def _academic_feature(
    feature_id: str,
    name: str,
    category: str,
    sub_category: str,
    description: str,
    formula: str,
    formula_latex: str,
    interpretation: str,
    source_paper: str,
    factor_zoo_category: str,
    *,
    direction: str = "neutral",
    data_requirement: tuple[str, ...] = (),
    lag_required: str = "none",
    computation_window: str = "",
    implementation_module: str = "equity_feature_engineering.py",
    asset_class: str = "equity",
    typical_range: str = "",
) -> FeatureMetadata:
    return FeatureMetadata(
        id=feature_id,
        name=name,
        category=category,
        sub_category=sub_category,
        description=description,
        formula=formula,
        formula_latex=formula_latex,
        interpretation=interpretation,
        source_paper=source_paper,
        source_doi=_PAPER_DOI.get(source_paper, "https://genisia.local/methodology/internal"),
        factor_zoo_category=factor_zoo_category,
        economic_rationale=_ZOO_RATIONALE.get(factor_zoo_category, _ZOO_RATIONALE["composite"]),
        direction=direction,
        data_requirement=data_requirement,
        lag_required=lag_required,
        computation_window=computation_window,
        implementation_module=implementation_module,
        asset_class=asset_class,
        typical_range=typical_range,
        point_in_time_safe=True,
        leakage_risk=False,
    )


FEATURE_METADATA.update(
    {
        item.id: item
        for item in (
            _academic_feature("alpha_1y", "Alpha 1Y vs benchmark", "factor_alpha", "capm_alpha", "Jensen alpha from rolling 252-day CAPM regression versus selected benchmark.", "mean(excess_return) - beta * mean(market_excess_return)", r"\hat{\alpha}=\bar{r}_i-\hat{\beta}_i\bar{r}_m", "Positive values indicate return above market-risk-adjusted expectation.", "Jensen 1968", "factor_alpha", direction="positive", data_requirement=("price", "benchmark_return"), computation_window="252d rolling"),
            _academic_feature("alpha_3y", "Alpha 3Y vs benchmark", "factor_alpha", "capm_alpha", "Jensen alpha from rolling 756-day CAPM regression versus selected benchmark.", "mean(excess_return) - beta * mean(market_excess_return)", r"\hat{\alpha}_{3y}=\bar{r}_i-\hat{\beta}_i\bar{r}_m", "Positive values indicate persistent benchmark-adjusted abnormal return.", "Jensen 1968", "factor_alpha", direction="positive", data_requirement=("price", "benchmark_return"), computation_window="756d rolling"),
            _academic_feature("alpha_3factor", "FF3 alpha", "factor_alpha", "multi_factor_alpha", "Residual alpha from the Fama-French three-factor model.", "r_i - (r_f + b_mkt*MKT + b_smb*SMB + b_hml*HML)", r"\hat{\alpha}^{FF3}=r_i-(r_f+\hat{b}_1MKT+\hat{b}_2SMB+\hat{b}_3HML)", "Positive values indicate return unexplained by market, size and value exposure.", "Fama-French 1993", "factor_alpha", direction="positive", data_requirement=("returns", "ff3_factors")),
            _academic_feature("alpha_5factor", "FF5 alpha", "factor_alpha", "multi_factor_alpha", "Residual alpha from the Fama-French five-factor model.", "r_i - (r_f + b1*MKT + b2*SMB + b3*HML + b4*RMW + b5*CMA)", r"\hat{\alpha}^{FF5}=r_i-(r_f+\hat{b}_1MKT+\hat{b}_2SMB+\hat{b}_3HML+\hat{b}_4RMW+\hat{b}_5CMA)", "Positive values indicate return unexplained by market, size, value, profitability and investment exposure.", "Fama-French 2015", "factor_alpha", direction="positive", data_requirement=("returns", "ff5_factors")),
            _academic_feature("alpha_carhart4", "Carhart 4F alpha", "factor_alpha", "multi_factor_alpha", "Residual alpha from the Carhart four-factor model including momentum.", "r_i - (r_f + b1*MKT + b2*SMB + b3*HML + b4*WML)", r"\hat{\alpha}^{4F}=r_i-(r_f+\hat{b}_1MKT+\hat{b}_2SMB+\hat{b}_3HML+\hat{b}_4WML)", "Positive values indicate return not explained by standard factor and momentum exposures.", "Carhart 1997", "factor_alpha", direction="positive", data_requirement=("returns", "carhart_factors")),
            _academic_feature("alpha_q5", "q5 alpha", "factor_alpha", "multi_factor_alpha", "Residual alpha from the Hou-Xue-Zhang q-factor model.", "r_i - (r_f + b_mkt*MKT + b_me*ME + b_ia*IA + b_roe*ROE + b_eg*EG)", r"\hat{\alpha}^{q5}=r_i-(r_f+\hat{b}_1MKT+\hat{b}_2ME+\hat{b}_3IA+\hat{b}_4ROE+\hat{b}_5EG)", "Positive values indicate return unexplained by q-model investment and profitability risks.", "Hou-Xue-Zhang 2015", "factor_alpha", direction="positive", data_requirement=("returns", "q_factors")),
            _academic_feature("idiosyncratic_return_1y", "Idiosyncratic return 1Y", "factor_alpha", "residual_return", "Annualized residual return from a factor model.", "sum(residual_return, 252d)", r"\sum_{d=t-251}^{t}\hat{\epsilon}_{i,d}", "Positive values indicate stock-specific return after factor adjustment.", "Ang-Hodrick-Xing-Zhang 2006", "factor_alpha", direction="positive", data_requirement=("returns", "factor_residuals"), computation_window="252d rolling"),
            _academic_feature("tracking_error_1y", "Tracking error 1Y", "factor_alpha", "active_risk", "Annualized volatility of active returns versus benchmark.", "std(r_i - r_b, 252d) * sqrt(252)", r"\sigma(r_i-r_b)\sqrt{252}", "Higher values indicate more benchmark-relative active risk.", "Gen.is.IA internal", "factor_alpha", data_requirement=("returns", "benchmark_return"), computation_window="252d rolling"),
            _academic_feature("information_ratio_1y", "Information ratio 1Y", "factor_alpha", "active_efficiency", "Active return per unit of tracking error.", "mean(active_return) / tracking_error", r"\frac{\bar{r}_i-\bar{r}_b}{\sigma(r_i-r_b)}", "Higher values indicate more efficient benchmark-relative alpha.", "Gen.is.IA internal", "factor_alpha", direction="positive", data_requirement=("returns", "benchmark_return"), computation_window="252d rolling"),
            _academic_feature("treynor_ratio_1y", "Treynor ratio 1Y", "factor_alpha", "systematic_efficiency", "Excess return per unit of market beta.", "(return - risk_free) / beta", r"\frac{R_i-R_f}{\beta_i}", "Higher values indicate better compensation for systematic risk.", "Jensen 1968", "factor_alpha", direction="positive", data_requirement=("returns", "benchmark_return"), computation_window="252d rolling"),
            _academic_feature("m2_measure_1y", "M-squared 1Y", "factor_alpha", "risk_adjusted_return", "Sharpe transformed into benchmark-volatility return units.", "sharpe_i * vol_benchmark + risk_free", r"M^2=SR_i\sigma_m+r_f", "Higher values indicate better risk-adjusted return in comparable return units.", "Gen.is.IA internal", "factor_alpha", direction="positive"),
            _academic_feature("upside_capture_ratio", "Upside capture ratio", "factor_alpha", "capture", "Portfolio or stock participation in benchmark up periods.", "mean(r_i | r_b>0) / mean(r_b | r_b>0)", r"\frac{E[r_i|r_b>0]}{E[r_b|r_b>0]}", "Values above 1 indicate stronger participation when the benchmark rises.", "Gen.is.IA internal", "factor_alpha", direction="positive"),
            _academic_feature("downside_capture_ratio", "Downside capture ratio", "factor_alpha", "capture", "Portfolio or stock participation in benchmark down periods.", "mean(r_i | r_b<0) / mean(r_b | r_b<0)", r"\frac{E[r_i|r_b<0]}{E[r_b|r_b<0]}", "Lower values indicate less participation in benchmark losses.", "Gen.is.IA internal", "factor_alpha", direction="negative"),
            _academic_feature("batting_average_1y", "Batting average 1Y", "factor_alpha", "active_hit_rate", "Share of months outperforming benchmark.", "count(active_return>0) / count(months)", r"\frac{1}{T}\sum 1_{\{r_i-r_b>0\}}", "Higher values indicate more frequent benchmark outperformance.", "Gen.is.IA internal", "factor_alpha", direction="positive", computation_window="12 monthly observations"),
            _academic_feature("omega_ratio_1y", "Omega ratio 1Y", "factor_alpha", "partial_moments", "Upside payoff divided by downside payoff around a threshold.", "sum(max(r-threshold,0)) / abs(sum(min(r-threshold,0)))", r"\frac{\int_{\tau}^{\infty}(1-F(r))dr}{\int_{-\infty}^{\tau}F(r)dr}", "Values above 1 indicate more upside than downside payoff.", "Gen.is.IA internal", "factor_alpha", direction="positive"),
            _academic_feature("asset_growth", "Asset growth", "factor_investment", "aqr_investment", "Year-over-year change in total assets, the primary investment factor signal.", "(total_assets_t - total_assets_t-1) / total_assets_t-1", r"\frac{TA_t-TA_{t-1}}{TA_{t-1}}", "Lower values indicate conservative investment policy, historically associated with higher returns.", "Cooper-Gulen-Schill 2008", "investment", direction="negative", lag_required="fiscal_quarter_lag", data_requirement=("balance_sheet",), computation_window="YoY"),
            _academic_feature("capex_to_assets", "Capex to assets", "factor_investment", "capital_investment", "Capital expenditure scaled by total assets.", "capex / total_assets", r"\frac{CAPEX_t}{TA_t}", "Lower values indicate more conservative capital investment intensity.", "Titman-Wei-Xie 2004", "investment", direction="negative", lag_required="fiscal_quarter_lag", data_requirement=("cash_flow", "balance_sheet")),
            _academic_feature("capex_growth", "Capex growth", "factor_investment", "capital_investment", "Growth rate of capital expenditure.", "(capex_t - capex_t-1) / abs(capex_t-1)", r"\frac{CAPEX_t-CAPEX_{t-1}}{|CAPEX_{t-1}|}", "Very high values can signal aggressive investment and lower future returns.", "Xing 2008", "investment", direction="negative", lag_required="fiscal_quarter_lag"),
            _academic_feature("net_stock_issues", "Net stock issues", "factor_investment", "external_financing", "Change in split-adjusted shares outstanding.", "ln(shares_t / shares_t-1)", r"\ln(SHR_t/SHR_{t-1})", "Lower values indicate less equity issuance and dilution.", "Pontiff-Woodgate 2008", "investment", direction="negative", lag_required="fiscal_quarter_lag"),
            _academic_feature("net_debt_issues", "Net debt issues", "factor_investment", "external_financing", "Change in total debt scaled by assets.", "(debt_t - debt_t-1) / total_assets_t-1", r"\frac{D_t-D_{t-1}}{TA_{t-1}}", "Higher debt issuance can flag external financing pressure.", "Spiess-Affleck-Graves 1999", "investment", direction="negative", lag_required="fiscal_quarter_lag"),
            _academic_feature("investment_to_assets", "Investment to assets", "factor_investment", "cma_proxy", "CMA-style investment intensity proxy.", "change(total_assets) / lag(total_assets)", r"\frac{\Delta TA_t}{TA_{t-1}}", "Lower values map to conservative investment exposure.", "Fama-French 2015", "investment", direction="negative", lag_required="fiscal_quarter_lag"),
            _academic_feature("roe_growth", "ROE growth", "factor_investment", "q5_eg", "Growth in return on equity, proxy for q-factor expected growth.", "roe_t - roe_t-1", r"ROE_t-ROE_{t-1}", "Higher values indicate improving profitability growth.", "Hou-Xue-Zhang 2015", "investment", direction="positive", lag_required="fiscal_quarter_lag"),
            _academic_feature("earnings_announcement_return", "Earnings announcement return", "factor_investment", "earnings_event", "Abnormal return around earnings announcement window.", "return[-1,+1] - benchmark_return[-1,+1]", r"CAR_{[-1,+1]}", "Positive values indicate favorable earnings-event surprise.", "Gen.is.IA internal", "momentum", direction="positive", lag_required="event_lag"),
            _academic_feature("sue_score", "Standardized unexpected earnings", "factor_investment", "earnings_momentum", "Standardized unexpected earnings surprise.", "(EPS_t - EPS_t-4) / std(EPS surprise, 8q)", r"\frac{EPS_t-EPS_{t-4}}{\sigma(\Delta EPS)}", "Higher values indicate positive earnings surprise momentum.", "Gen.is.IA internal", "momentum", direction="positive", lag_required="earnings_release_lag"),
            _academic_feature("analyst_revision_score", "Analyst revision score", "factor_investment", "analyst_revisions", "Consensus estimate revision momentum.", "net_upward_revisions / analyst_count", r"\frac{UpRevisions-DownRevisions}{N_{analysts}}", "Higher values indicate improving analyst expectations.", "Gen.is.IA internal", "smart_money", direction="positive", lag_required="provider_timestamp"),
            _academic_feature("earnings_surprise_3m", "Earnings surprise 3M", "factor_investment", "earnings_momentum", "Rolling three-month earnings surprise proxy.", "mean(SUE, 3m)", r"\overline{SUE}_{3m}", "Higher values indicate repeated positive earnings surprise.", "Gen.is.IA internal", "momentum", direction="positive", lag_required="earnings_release_lag"),
            _academic_feature("external_financing_ratio", "External financing ratio", "factor_investment", "external_financing", "Equity plus debt issuance scaled by assets.", "(net_stock_issues + net_debt_issues) / assets", r"\frac{\Delta Equity+\Delta Debt}{TA}", "Lower values indicate less dependence on external financing.", "Gen.is.IA internal", "investment", direction="negative", lag_required="fiscal_quarter_lag"),
            _academic_feature("rmw_proxy", "RMW proxy", "factor_profitability", "operating_profitability", "Operating profitability proxy for Fama-French RMW.", "(revenue - COGS - SGA - interest) / book_equity", r"\frac{Rev-COGS-SGA-IntExp}{BE_t}", "Higher values indicate robust operating profitability.", "Fama-French 2015", "profitability", direction="positive", lag_required="fiscal_quarter_lag"),
            _academic_feature("cash_earnings_to_price", "Cash earnings to price", "factor_profitability", "cash_profitability", "Cash flow from operations scaled by price or market value.", "CFO / market_cap", r"\frac{CFO_t}{ME_t}", "Higher values indicate more cash earnings per dollar of equity value.", "Desai-Rajgopal-Venkatachalam 2004", "value", direction="positive", lag_required="fiscal_quarter_lag"),
            _academic_feature("gross_profitability", "Gross profitability", "factor_profitability", "profitability", "Gross profit scaled by total assets.", "(revenue - COGS) / total_assets", r"\frac{Rev_t-COGS_t}{TA_t}", "Higher values indicate more productive assets.", "Novy-Marx 2013", "profitability", direction="positive", lag_required="fiscal_quarter_lag"),
            _academic_feature("mispricing_score_stambaugh", "Stambaugh mispricing score", "factor_profitability", "composite_mispricing", "Composite anomaly score inspired by Stambaugh-Yu-Yuan mispricing factors.", "average anomaly rank across selected mispricing features", r"\frac{1}{K}\sum_{k=1}^{K}rank(z_{k})", "Higher values indicate greater estimated overpricing/underpricing depending construction.", "Stambaugh-Yu-Yuan 2015", "composite", direction="neutral"),
            _academic_feature("noa_ratio", "Net operating assets", "factor_profitability", "accruals", "Net operating assets scaled by lagged total assets.", "net_operating_assets / lag(total_assets)", r"\frac{NOA_t}{TA_{t-1}}", "Lower values indicate less accrual-heavy operating balance sheet.", "Sloan 1996", "quality", direction="negative", lag_required="fiscal_quarter_lag"),
            _academic_feature("operating_leverage", "Operating leverage", "factor_profitability", "cost_structure", "Fixed-cost intensity proxy.", "fixed_costs / total_costs", r"\frac{FixedCosts_t}{TotalCosts_t}", "Higher values can amplify earnings sensitivity to sales changes.", "Gen.is.IA internal", "profitability", direction="neutral", lag_required="fiscal_quarter_lag"),
            _academic_feature("financial_leverage_change", "Financial leverage change", "factor_profitability", "leverage", "Change in financial leverage.", "debt_to_equity_t - debt_to_equity_t-1", r"LEV_t-LEV_{t-1}", "Lower values indicate deleveraging and safer balance-sheet trend.", "Gen.is.IA internal", "quality", direction="negative", lag_required="fiscal_quarter_lag"),
            _academic_feature("effective_tax_rate", "Effective tax rate", "factor_profitability", "tax", "Income tax expense divided by pretax income.", "tax_expense / pretax_income", r"\frac{TaxExp_t}{PretaxIncome_t}", "Stable values support quality checks; extremes can be one-off.", "Gen.is.IA internal", "quality", direction="neutral", lag_required="fiscal_quarter_lag"),
            _academic_feature("r_and_d_to_market", "R&D to market", "factor_profitability", "intangibles", "Research and development expense scaled by market value.", "R&D / market_cap", r"\frac{R\&D_t}{ME_t}", "Higher values can indicate intangible investment not captured on the balance sheet.", "Gen.is.IA internal", "intangibles", direction="positive", lag_required="fiscal_quarter_lag"),
            _academic_feature("org_capital_to_assets", "Organization capital to assets", "factor_profitability", "intangibles", "Organization capital proxy scaled by total assets.", "capitalized_SGA / total_assets", r"\frac{OrgCap_t}{TA_t}", "Higher values can indicate durable intangible operating assets.", "Gen.is.IA internal", "intangibles", direction="positive", lag_required="fiscal_quarter_lag"),
            _academic_feature("earnings_momentum_sue", "Earnings momentum SUE", "momentum", "earnings_momentum", "SUE-based post-earnings-announcement drift signal.", "(EPS_t - EPS_t-4) / std(EPS surprise, 8q)", r"\frac{EPS_t-EPS_{t-4}}{\sigma(\Delta EPS)}", "Higher values indicate positive earnings momentum.", "Jegadeesh-Titman 1993", "momentum", direction="positive", lag_required="earnings_release_lag"),
            _academic_feature("price_momentum_6_1", "6-1M price momentum", "momentum", "price_momentum", "Six-month return excluding the most recent month.", "(P_t-21 - P_t-126) / P_t-126", r"\frac{P_{t-21}-P_{t-126}}{P_{t-126}}", "Higher values indicate medium-term trend excluding short-term reversal.", "Jegadeesh-Titman 1993", "momentum", direction="positive"),
            _academic_feature("industry_momentum", "Industry momentum", "momentum", "industry_relative", "Industry-level momentum assigned to constituent stocks.", "industry_return_12_1", r"MOM^{industry}_{t}", "Higher values indicate a strong peer-industry trend.", "Jegadeesh-Titman 1993", "momentum", direction="positive"),
            _academic_feature("residual_momentum_252d", "Residual momentum 252D", "momentum", "residual_momentum", "Momentum of factor-model residual returns.", "sum(residual_return, 252d)", r"\sum_{d=t-251}^{t}\hat{\epsilon}_{i,d}", "Higher values indicate stock-specific trend after factor adjustment.", "Gutierrez-Pirinsky 2007", "momentum", direction="positive"),
            _academic_feature("momentum_acceleration_3m", "Momentum acceleration 3M", "momentum", "price_momentum", "Change in short-horizon momentum versus longer-horizon momentum.", "ret_63d - ret_126d", r"r_{63d}-r_{126d}", "Positive values indicate strengthening recent momentum.", "Gen.is.IA internal", "momentum", direction="positive"),
            _academic_feature("volume_weighted_momentum", "Volume-weighted momentum", "momentum", "volume_momentum", "Price momentum weighted by trading volume intensity.", "momentum_12m_1m * volume_ratio", r"MOM_{12,1}\times VolumeRatio", "Higher values indicate momentum confirmed by volume.", "Gen.is.IA internal", "momentum", direction="positive"),
            _academic_feature("52w_high_momentum", "52-week high momentum", "momentum", "price_anchor", "Closeness to the 52-week high.", "price / high_52w", r"\frac{P_t}{High_{52w}}", "Values near 1 indicate price near annual high, a documented momentum anchor.", "George-Hwang 2004", "momentum", direction="positive"),
            _academic_feature("max_return_1m", "MAX return 1M", "momentum", "lottery", "Maximum daily return over the past month.", "max(daily_return, 21d)", r"\max_{d\in[1,21]} r_{i,d}", "Higher values can indicate lottery-like payoff and lower future returns.", "Bali-Cakici-Whitelaw 2011", "momentum", direction="negative"),
            _academic_feature("pastor_stambaugh_liquidity", "Pastor-Stambaugh liquidity beta", "liquidity", "liquidity_beta", "Sensitivity to aggregate liquidity innovations.", "beta(asset_return, PS_liquidity_factor)", r"\beta_{PS}", "Higher exposure indicates stronger sensitivity to market liquidity conditions.", "Pastor-Stambaugh 2003", "trading_frictions", direction="neutral"),
            _academic_feature("bid_ask_spread_proxy", "Roll spread proxy", "liquidity", "spread", "Implicit bid-ask spread from return autocovariance.", "2 * sqrt(-cov(r_t, r_t-1))", r"2\sqrt{-Cov(r_t,r_{t-1})}", "Higher values indicate wider estimated trading spreads.", "Roll 1984", "trading_frictions", direction="negative"),
            _academic_feature("zero_trading_days_ratio", "Zero-trading-days ratio", "liquidity", "illiquidity", "Share of days with zero returns or zero volume.", "count(zero_return_or_volume) / count(days)", r"\frac{N_{zero}}{T}", "Higher values indicate stale prices and lower liquidity.", "Lesmond-Ogden-Trzcinka 1999", "trading_frictions", direction="negative"),
            _academic_feature("price_impact_21d", "Price impact 21D", "liquidity", "amihud_variant", "Short-window Amihud price-impact proxy.", "mean(abs(return)/dollar_volume, 21d)", r"\frac{1}{21}\sum\frac{|r_d|}{DVOL_d}", "Higher values indicate stronger price impact per dollar traded.", "Amihud 2002", "trading_frictions", direction="negative"),
            _academic_feature("float_ratio", "Float ratio", "liquidity", "ownership_float", "Free float shares as a share of total shares outstanding.", "float_shares / shares_outstanding", r"\frac{FloatShares}{SharesOut}", "Higher values indicate more tradable supply.", "Gen.is.IA internal", "trading_frictions", direction="positive"),
            _academic_feature("regime_credit_spread", "Credit spread regime", "macro_regime", "credit", "High-yield versus investment-grade credit stress proxy.", "HYG relative return - LQD relative return", r"Trend(HYG)-Trend(LQD)", "Higher values indicate stronger credit risk appetite when constructed as HY outperformance.", "Gen.is.IA internal", "macro", asset_class="multi_asset", implementation_module="macro_features.py"),
            _academic_feature("regime_em_stress", "EM stress regime", "macro_regime", "em_stress", "Emerging-market stress from relative equity and volatility proxies.", "vol(EEM) - vol(VEA)", r"\sigma(EEM)-\sigma(VEA)", "Higher values indicate more emerging-market stress.", "Gen.is.IA internal", "macro", direction="negative", asset_class="multi_asset", implementation_module="macro_features.py"),
            _academic_feature("regime_eu_sovereign_stress", "EU sovereign stress", "macro_regime", "rates", "BTP-Bund sovereign spread proxy.", "Italy_10Y - Germany_10Y", r"y^{IT}_{10Y}-y^{DE}_{10Y}", "Higher values indicate more Italian/EU sovereign stress.", "Gen.is.IA internal", "macro", direction="negative", asset_class="macro", implementation_module="macro_features.py"),
            _academic_feature("regime_breakeven_inflation", "Breakeven inflation regime", "macro_regime", "inflation", "Nominal minus real yield proxy.", "nominal_10y - real_10y", r"y^{nom}_{10Y}-y^{real}_{10Y}", "Higher values indicate higher market-implied inflation compensation.", "Gen.is.IA internal", "macro", asset_class="macro", implementation_module="macro_features.py"),
            _academic_feature("regime_global_growth_proxy", "Global growth proxy", "macro_regime", "growth", "Global equity trend proxy.", "return(global_equity_index, 63d)", r"r^{World}_{63d}", "Higher values indicate stronger global growth/risk appetite backdrop.", "Gen.is.IA internal", "macro", direction="positive", asset_class="multi_asset", implementation_module="macro_features.py"),
            _academic_feature("regime_commodities_trend", "Commodities trend regime", "macro_regime", "commodities", "Broad commodity trend proxy.", "return(commodity_index, 63d)", r"r^{Comm}_{63d}", "Higher values indicate stronger commodity cycle momentum.", "Gen.is.IA internal", "macro", direction="positive", asset_class="multi_asset", implementation_module="macro_features.py"),
            _academic_feature("regime_funding_liquidity", "Funding liquidity regime", "macro_regime", "funding", "Short-rate spread proxy for funding stress.", "short_rate_stress_proxy", r"FundingSpread_t", "Higher values indicate tighter funding liquidity.", "Gen.is.IA internal", "macro", direction="negative", asset_class="macro", implementation_module="macro_features.py"),
            _academic_feature("regime_risk_parity_signal", "Risk parity regime signal", "macro_regime", "cross_asset_vol", "Composite signal from equity, bond and commodity volatility.", "weighted zscore(vol_equity, vol_bonds, vol_commodities)", r"\sum w_a z(\sigma_a)", "Higher values indicate cross-asset volatility stress.", "Gen.is.IA internal", "macro", direction="negative", asset_class="multi_asset", implementation_module="macro_features.py"),
            _academic_feature("short_interest_ratio", "Short interest ratio", "sentiment", "short_interest", "Short interest scaled by average daily volume.", "short_interest / avg_daily_volume", r"\frac{ShortInterest}{ADV}", "Higher values indicate more crowded short positioning.", "Gen.is.IA internal", "smart_money", direction="negative", data_requirement=("FMP short_interest",), lag_required="provider_timestamp", implementation_module="equity_feature_engineering.py:compute_sentiment_alternative_features"),
            _academic_feature("institutional_ownership_pct", "Institutional ownership %", "sentiment", "ownership", "Percent of shares held by institutions.", "institutional_shares / shares_outstanding", r"\frac{InstShares}{SharesOut}", "Higher values indicate stronger institutional sponsorship, but can also imply crowding.", "Gen.is.IA internal", "smart_money", direction="neutral", lag_required="provider_timestamp", implementation_module="equity_feature_engineering.py:compute_sentiment_alternative_features"),
            _academic_feature("insider_net_buying", "Insider net buying", "sentiment", "insider_trading", "Net insider buy transactions over a recent window.", "insider_buys - insider_sells", r"Buys_{insider}-Sells_{insider}", "Positive values indicate insider accumulation.", "Gen.is.IA internal", "smart_money", direction="positive", lag_required="provider_timestamp", implementation_module="equity_feature_engineering.py:compute_sentiment_alternative_features"),
            _academic_feature("analyst_coverage_count", "Analyst coverage count", "sentiment", "analyst", "Number of sell-side analysts covering the company.", "count(analysts)", r"N_{analysts}", "Higher values indicate more analyst attention and usually lower information opacity.", "Gen.is.IA internal", "smart_money", direction="neutral", lag_required="provider_timestamp", implementation_module="equity_feature_engineering.py:compute_sentiment_alternative_features"),
            _academic_feature("earnings_estimate_dispersion", "Earnings estimate dispersion", "sentiment", "analyst", "Dispersion of analyst EPS estimates scaled by mean estimate.", "std(EPS_estimates) / abs(mean(EPS_estimates))", r"\frac{\sigma(EPS^{est})}{|E[EPS^{est}]|}", "Higher values indicate more uncertainty/disagreement about earnings.", "Gen.is.IA internal", "smart_money", direction="negative", lag_required="provider_timestamp", implementation_module="equity_feature_engineering.py:compute_sentiment_alternative_features"),
            _academic_feature("hiring_rate", "Hiring rate", "factor_zoo_academic", "labor", "Change in employees scaled by lagged employees.", "(employees_t - employees_t-1) / employees_t-1", r"\frac{Emp_t-Emp_{t-1}}{Emp_{t-1}}", "High hiring can proxy aggressive expansion and future operating leverage.", "Hou-Xue-Zhang 2015", "investment", direction="neutral", lag_required="fiscal_quarter_lag"),
            _academic_feature("patent_intensity", "Patent intensity", "factor_zoo_academic", "intangibles", "Patents granted scaled by total assets.", "patents / total_assets", r"\frac{Patents_t}{TA_t}", "Higher values can indicate innovation intensity not fully captured by book assets.", "Gen.is.IA internal", "intangibles", direction="positive", lag_required="provider_timestamp"),
            _academic_feature("advertising_to_sales", "Advertising to sales", "factor_zoo_academic", "intangibles", "Advertising expense scaled by revenue.", "advertising_expense / revenue", r"\frac{Adv_t}{Sales_t}", "Higher values can indicate brand-building intangible investment.", "Gen.is.IA internal", "intangibles", direction="positive", lag_required="fiscal_quarter_lag"),
            _academic_feature("pp_and_e_growth", "PP&E growth", "factor_zoo_academic", "investment", "Growth in net property, plant and equipment.", "(ppe_t - ppe_t-1) / ppe_t-1", r"\frac{PPE_t-PPE_{t-1}}{PPE_{t-1}}", "High values indicate capital expansion and potential investment-factor exposure.", "Fama-French 2015", "investment", direction="negative", lag_required="fiscal_quarter_lag"),
            _academic_feature("working_capital_change", "Working capital change", "factor_zoo_academic", "investment", "Change in working capital scaled by lagged assets.", "change(current_assets - current_liabilities) / lag(total_assets)", r"\frac{\Delta WC_t}{TA_{t-1}}", "Large increases can indicate investment in operating capacity or accrual pressure.", "Fama-French 2015", "investment", direction="neutral", lag_required="fiscal_quarter_lag"),
        )
    }
)

try:
    from .alpha101 import Alpha101Suite

    FEATURE_METADATA.update(
        {
            alpha_id: FeatureMetadata(
                id=alpha_id,
                name=f"WQ Alpha {alpha_id[-3:]}",
                category="alpha101",
                sub_category="worldquant_formulaic",
                description=str(meta["formula_text"]),
                formula=str(meta["formula_text"]),
                formula_latex=rf"\text{{WorldQuant {alpha_id[-3:]} formula; see Kakushadze (2015) Appendix A}}",
                interpretation="Cross-sectionally ranked formulaic price-volume alpha. Higher ranks indicate stronger signal according to the formula ordering.",
                leakage_risk=False,
                data_requirement=tuple(meta["inputs"]),
                source_paper=str(meta["paper"]),
                alias=(f"alpha_{alpha_id[-3:]}",),
                asset_class="equity",
                source_doi=f"https://arxiv.org/abs/{meta['arxiv']}",
                economic_rationale=str(meta["economic_rationale"]),
                factor_zoo_category=str(meta["factor_zoo_category"]),
                point_in_time_safe=True,
                lag_required="none",
                direction="neutral",
                typical_range="0-1 cross-sectional rank",
                implementation_module="alpha101.py",
                normalization="cross-sectional rank",
            )
            for alpha_id, meta in Alpha101Suite.METADATA.items()
        }
    )
except Exception:
    # Metadata must remain importable even if optional Alpha101 dependencies are
    # unavailable in a constrained environment.
    pass


def _latex_from_formula(formula: str) -> str:
    text = str(formula or "").strip()
    if not text:
        return r"\text{See implementation notes}"
    safe = text.replace("_", r"\_")
    return rf"\text{{{safe[:180]}}}"


def _complete_feature_metadata(meta: FeatureMetadata) -> FeatureMetadata:
    category_key = str(meta.category or "").strip().lower()
    feature_key = str(meta.id or "").strip().lower().replace(" ", "_").replace("-", "_")
    zoo = meta.factor_zoo_category or _CATEGORY_TO_ZOO.get(category_key, "composite")
    if "forward_return" in feature_key or category_key == "target":
        zoo = "target"
    if zoo not in VALID_FACTOR_ZOO_CATEGORIES:
        zoo = _CATEGORY_TO_ZOO.get(zoo, "composite")
    source_paper = meta.source_paper or _CATEGORY_DEFAULT_PAPER.get(category_key, "Gen.is.IA internal")
    source_doi = meta.source_doi or _PAPER_DOI.get(source_paper, "https://genisia.local/methodology/internal")
    formula_latex = meta.formula_latex or _latex_from_formula(meta.formula)
    rationale = meta.economic_rationale or _ZOO_RATIONALE.get(zoo, _ZOO_RATIONALE["composite"])
    computation_window = meta.computation_window or ("forward horizon" if zoo == "target" else "point-in-time latest or rolling window")
    normalization = meta.normalization or ("cross-sectional z-score" if zoo not in {"target", "macro"} else "none")
    direction = meta.direction or ("positive" if zoo in {"value", "momentum", "profitability", "quality", "factor_alpha"} else "neutral")
    typical_range = meta.typical_range or ("model/data dependent" if zoo != "target" else "realized return in decimal units")
    implementation_module = meta.implementation_module or (
        "equity_feature_engineering.py" if meta.asset_class == "equity" else "macro_features.py"
    )
    return replace(
        meta,
        formula_latex=formula_latex,
        source_paper=source_paper,
        source_doi=source_doi,
        economic_rationale=rationale,
        factor_zoo_category=zoo,
        computation_window=computation_window,
        normalization=normalization,
        direction=direction,
        typical_range=typical_range,
        implementation_module=implementation_module,
        leakage_risk=False if zoo == "target" else meta.leakage_risk,
        point_in_time_safe=True if zoo == "target" or meta.point_in_time_safe else meta.point_in_time_safe,
    )


FEATURE_METADATA = {key: _complete_feature_metadata(meta) for key, meta in FEATURE_METADATA.items()}


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
