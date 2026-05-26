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


_SOURCE_BY_ID = {
    "momentum_12_1": "Jegadeesh and Titman (1993)",
    "momentum_12m_1m": "Jegadeesh and Titman (1993)",
    "momentum_reversal_1m": "Jegadeesh (1990)",
    "short_term_reversal": "Jegadeesh (1990)",
    "long_term_reversal": "De Bondt and Thaler (1985)",
    "idiosyncratic_momentum": "Blitz, Huij and Martens (2011)",
    "book_to_market": "Fama and French (1992, 1993)",
    "earnings_yield": "Basu (1977)",
    "fcf_yield": "Lakonishok, Shleifer and Vishny (1994)",
    "ev_ebitda": "Loughran and Wellman (2011)",
    "gross_profitability": "Novy-Marx (2013)",
    "roe": "Piotroski (2000)",
    "roic": "Penman (2001)",
    "accruals_ratio": "Sloan (1996)",
    "piotroski_f_score": "Piotroski (2000)",
    "altman_z_score": "Altman (1968, 1995 revised)",
    "log_market_cap": "Banz (1981); Fama and French (1992)",
    "beta": "Sharpe (1964)",
    "beta_market": "Sharpe (1964)",
    "idiosyncratic_vol": "Ang et al. (2006)",
    "bab_score": "Frazzini and Pedersen (2014)",
    "amihud_illiquidity": "Amihud (2002)",
    "turnover_ratio_21d": "Datar, Naik and Radcliffe (1998)",
    "asset_growth": "Cooper, Gulen and Schill (2008)",
    "capex_growth": "Fama and French (2015)",
    "net_issuance": "Ritter (1991); Loughran and Ritter (1995)",
}

_SOURCE_BY_CATEGORY = {
    "momentum": "Jegadeesh and Titman (1993); Carhart (1997)",
    "value": "Fama and French (1992, 1993)",
    "valuation": "Damodaran (2012); Koller, Goedhart and Wessels (2015)",
    "quality": "Novy-Marx (2013); Piotroski (2000)",
    "size": "Banz (1981); Fama and French (1992)",
    "risk": "Sharpe (1964); Ang et al. (2006)",
    "liquidity": "Amihud (2002); Pastor and Stambaugh (2003)",
    "growth": "Fama and French (2015); Cooper, Gulen and Schill (2008)",
    "technical": "Murphy (1999), Technical Analysis of the Financial Markets",
    "macro_context": "Gen.is.IA internal macro context methodology",
    "macro_regime": "Gen.is.IA internal macro regime methodology",
    "smart_money": "CFTC Commitments of Traders public methodology; Gen.is.IA internal",
    "ml": "Gen.is.IA internal ML governance methodology",
    "model": "Gen.is.IA internal model output methodology",
    "portfolio": "Grinold and Kahn (2000); Gen.is.IA internal",
    "price": "Gen.is.IA internal OHLCV feature methodology",
}

_RATIONALE_BY_CATEGORY = {
    "momentum": "Momentum features capture medium-term return continuation while controlling for short-term reversal noise.",
    "value": "Value features proxy expected-return compensation for cheap fundamentals relative to market price.",
    "valuation": "Valuation features compare market price with cash-flow, book-value or peer-implied anchors.",
    "quality": "Quality features capture profitability, balance-sheet strength and earnings quality.",
    "size": "Size features control for market-cap and investability effects.",
    "risk": "Risk features measure volatility, beta, drawdown and downside exposure.",
    "liquidity": "Liquidity features proxy trading capacity and market-impact risk.",
    "growth": "Growth features capture historical expansion and reinvestment intensity.",
    "technical": "Technical features summarize trend, range and oscillator state from past prices only.",
    "macro_context": "Macro context features expose broad risk appetite, rates, credit and commodity regimes to equity models.",
    "macro_regime": "Regime features summarize market state for interpretation and conditional analysis.",
    "smart_money": "Smart-money features summarize positioning or flow evidence as context, not as standalone trade instructions.",
    "ml": "ML output fields are model governance artifacts used for ranking, diagnostics and ensemble comparison.",
}


def _latex_from_formula(formula: str) -> str:
    text = str(formula or "").strip()
    if not text:
        return r"\text{defined in source artifact}"
    return text.replace("*", r"\cdot ").replace("_", r"\_")


def _complete_feature_metadata(meta: FeatureMetadata) -> FeatureMetadata:
    source = meta.source_paper or _SOURCE_BY_ID.get(meta.id) or _SOURCE_BY_CATEGORY.get(meta.category, "Gen.is.IA internal")
    rationale = meta.economic_rationale or _RATIONALE_BY_CATEGORY.get(meta.category, "Operational feature used by the Gen.is.IA research platform.")
    formula_latex = meta.formula_latex or _latex_from_formula(meta.formula)
    implementation = meta.implementation_module or "research_platform_core data/model artifact pipeline"
    window = meta.computation_window or ("point-in-time or trailing window as defined by formula" if meta.formula else "artifact-defined")
    normalization = meta.normalization or ("cross-sectional rank/z-score where used in factor models" if meta.category in {"value", "quality", "momentum", "risk", "size", "growth"} else "none")
    direction = meta.direction
    if direction == "neutral" and meta.interpretation:
        lower = meta.interpretation.lower()
        if "higher" in lower or "positive" in lower:
            direction = "higher_is_better"
        elif "lower" in lower or "negative" in lower:
            direction = "lower_is_better"
    return replace(
        meta,
        source_paper=source,
        formula_latex=formula_latex,
        economic_rationale=rationale,
        implementation_module=implementation,
        computation_window=window,
        normalization=normalization,
        direction=direction,
        factor_zoo_category=meta.factor_zoo_category or meta.category,
        typical_range=meta.typical_range or "asset/universe dependent",
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
