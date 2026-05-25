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


ALIASES = {
    "market_value": "size_score",
    "marketcap": "size_score",
    "market_cap": "size_score",
    "ret21d": "return_1m",
    "ret63d": "return_3m",
    "ret252d": "return_1y",
    "volatility": "vol252d",
    "prediction": "ml_score",
    "expected_return": "ml_score",
    "screening_score": "factor_composite_score",
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

