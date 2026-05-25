"""Readable metric metadata for model, portfolio and valuation dashboards."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class MetricMetadata:
    id: str
    name: str
    category: str
    definition: str
    formula: str
    interpretation: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


METRICS_METADATA: dict[str, MetricMetadata] = {
    "r2_os": MetricMetadata(
        "r2_os",
        "Out-of-sample R²",
        "model_validation",
        "Predictive improvement versus a benchmark forecast on unseen data.",
        "1 - SSE_model / SSE_benchmark",
        "Positive values indicate the model beats the benchmark out of sample.",
    ),
    "ic": MetricMetadata(
        "ic",
        "Information Coefficient",
        "model_validation",
        "Cross-sectional correlation between predicted scores and realized returns.",
        "corr(score, forward_return)",
        "Higher positive values indicate better signal direction.",
    ),
    "rank_ic": MetricMetadata(
        "rank_ic",
        "Rank Information Coefficient",
        "model_validation",
        "Spearman rank correlation between predicted ranks and realized return ranks.",
        "spearman_rank_corr(score, forward_return)",
        "A core stock-ranking metric; positive and stable is better.",
    ),
    "hit_ratio": MetricMetadata(
        "hit_ratio",
        "Hit ratio",
        "model_validation",
        "Share of observations where the model direction or selection was correct.",
        "correct_predictions / observations",
        "Higher is better, but inspect base rate and payoff asymmetry.",
    ),
    "sharpe": MetricMetadata(
        "sharpe",
        "Sharpe ratio",
        "portfolio",
        "Annualized excess return per unit of realized volatility.",
        "mean(return) / std(return) * sqrt(periods_per_year)",
        "Higher values indicate better risk-adjusted returns.",
    ),
    "sharpe_long_short": MetricMetadata(
        "sharpe_long_short",
        "Long-short Sharpe",
        "portfolio",
        "Risk-adjusted return of the long high-score / short low-score spread.",
        "Sharpe(return_top_quantile - return_bottom_quantile)",
        "Useful for checking whether the score ranks stocks economically.",
    ),
    "max_drawdown": MetricMetadata(
        "max_drawdown",
        "Maximum drawdown",
        "risk",
        "Largest peak-to-trough loss over the evaluated period.",
        "min(cumulative_return / running_max - 1)",
        "Lower absolute drawdown is safer; large drawdowns can invalidate good average returns.",
    ),
    "volatility": MetricMetadata(
        "volatility",
        "Volatility",
        "risk",
        "Annualized standard deviation of returns.",
        "std(periodic_returns) * sqrt(periods_per_year)",
        "Higher values indicate more return variability.",
    ),
    "turnover": MetricMetadata(
        "turnover",
        "Turnover",
        "implementation",
        "Average absolute change in portfolio weights between rebalance dates.",
        "mean(sum(abs(weight_t - weight_t-1)))",
        "Higher turnover raises transaction-cost and capacity risk.",
    ),
    "cvar": MetricMetadata(
        "cvar",
        "CVaR",
        "risk",
        "Expected loss conditional on being in the worst tail of returns.",
        "mean(returns | returns <= VaR_alpha)",
        "More negative values indicate worse tail risk.",
    ),
    "alpha": MetricMetadata(
        "alpha",
        "Alpha",
        "portfolio",
        "Return unexplained by selected benchmark or factor exposures.",
        "intercept from return regression",
        "Positive alpha is desirable if robust after costs and out of sample.",
    ),
    "beta": MetricMetadata(
        "beta",
        "Beta",
        "risk",
        "Sensitivity to a benchmark or risk factor.",
        "cov(asset, benchmark) / var(benchmark)",
        "Higher values indicate stronger benchmark exposure.",
    ),
    "tracking_error": MetricMetadata(
        "tracking_error",
        "Tracking error",
        "risk",
        "Volatility of active return versus a benchmark.",
        "std(portfolio_return - benchmark_return) * sqrt(periods_per_year)",
        "Higher values indicate more active risk.",
    ),
    "information_ratio": MetricMetadata(
        "information_ratio",
        "Information ratio",
        "portfolio",
        "Active return per unit of tracking error.",
        "mean(active_return) / std(active_return)",
        "Higher values indicate better benchmark-relative efficiency.",
    ),
}


ALIASES = {
    "rank_information_coefficient": "rank_ic",
    "information_coefficient": "ic",
    "mean_return": "alpha",
    "avg_return": "alpha",
    "drawdown": "max_drawdown",
    "vol": "volatility",
    "ir": "information_ratio",
}


def normalize_metric_id(metric: str) -> str:
    return str(metric or "").strip().lower().replace(" ", "_").replace("-", "_")


def metadata_for_metric(metric: str) -> MetricMetadata | None:
    key = normalize_metric_id(metric)
    key = ALIASES.get(key, key)
    return METRICS_METADATA.get(key)


def metric_help(metric: str) -> str:
    meta = metadata_for_metric(metric)
    if meta is None:
        return "No glossary entry yet. Inspect the source artifact or methodology document for this metric."
    return f"{meta.definition} Formula: {meta.formula}. Interpretation: {meta.interpretation}"


def metrics_metadata_frame(metrics: Iterable[str] | None = None) -> pd.DataFrame:
    keys = [normalize_metric_id(metric) for metric in metrics] if metrics is not None else list(METRICS_METADATA)
    rows = []
    seen: set[str] = set()
    for key in keys:
        resolved = ALIASES.get(key, key)
        if resolved in seen:
            continue
        seen.add(resolved)
        meta = METRICS_METADATA.get(resolved)
        if meta is not None:
            rows.append(meta.to_dict())
    return pd.DataFrame(rows)

