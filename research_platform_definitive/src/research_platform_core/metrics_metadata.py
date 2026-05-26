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
    formula_latex: str = ""
    paper: str = ""
    typical_range: str = ""
    annualized: bool = False

    def to_dict(self) -> dict[str, object]:
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

METRICS_METADATA.update(
    {
        "sortino": MetricMetadata(
            "sortino",
            "Sortino ratio",
            "portfolio",
            "Return per unit of downside volatility.",
            "mean(return) / std(min(return, 0)) * sqrt(periods_per_year)",
            "Higher values indicate better downside-risk-adjusted returns.",
        ),
        "volatility_long_short": MetricMetadata(
            "volatility_long_short",
            "Long-short volatility",
            "risk",
            "Realized volatility of the long high-score / short low-score spread.",
            "std(return_top_quantile - return_bottom_quantile) * sqrt(periods_per_year)",
            "Lower values indicate a smoother spread, but should be considered with expected return.",
        ),
        "sharpe_long_short_net_cost": MetricMetadata(
            "sharpe_long_short_net_cost",
            "Net-cost long-short Sharpe",
            "portfolio",
            "Long-short Sharpe after transaction-cost adjustment.",
            "Sharpe(long_short_return - turnover * cost_bps / 10000)",
            "Higher values indicate stronger implementable signal quality after costs.",
        ),
        "avg_names_long_short": MetricMetadata(
            "avg_names_long_short",
            "Average long-short names",
            "implementation",
            "Average number of names in the long-short portfolio legs.",
            "mean(name_count_top + name_count_bottom)",
            "Higher values usually indicate better diversification and less single-name concentration.",
        ),
        "rows": MetricMetadata(
            "rows",
            "Rows",
            "data_quality",
            "Number of observations available in an artifact or training dataset.",
            "count(rows)",
            "Low values can invalidate model or score interpretation.",
        ),
        "panel_rows": MetricMetadata(
            "panel_rows",
            "Panel rows",
            "data_quality",
            "Number of ticker-date observations in the modeling panel.",
            "count(ticker, date observations)",
            "Higher values generally support more stable training, subject to coverage quality.",
        ),
        "ticker_count": MetricMetadata(
            "ticker_count",
            "Ticker count",
            "data_quality",
            "Number of distinct tickers represented in an artifact.",
            "n_unique(ticker)",
            "Higher values indicate broader cross-sectional coverage.",
        ),
        "date_count": MetricMetadata(
            "date_count",
            "Date count",
            "data_quality",
            "Number of distinct dates represented in an artifact.",
            "n_unique(date)",
            "Higher values indicate broader time-series coverage.",
        ),
        "feature_count": MetricMetadata(
            "feature_count",
            "Feature count",
            "model_governance",
            "Number of input features used by a model run.",
            "count(model_features)",
            "Too few features can underfit; too many weak features can raise overfit and governance risk.",
        ),
        "prediction_rows": MetricMetadata(
            "prediction_rows",
            "Prediction rows",
            "model_validation",
            "Number of prediction observations generated by a model.",
            "count(model predictions)",
            "Low values reduce confidence in validation metrics.",
        ),
        "train_rows": MetricMetadata(
            "train_rows",
            "Training rows",
            "model_validation",
            "Number of observations used for model fitting.",
            "count(train observations)",
            "Must be large enough relative to feature count and model complexity.",
        ),
        "test_rows": MetricMetadata(
            "test_rows",
            "Test rows",
            "model_validation",
            "Number of out-of-sample observations used for validation.",
            "count(test observations)",
            "Low values make OOS metrics unstable.",
        ),
        "train_date_count": MetricMetadata(
            "train_date_count",
            "Training dates",
            "model_validation",
            "Number of distinct dates in the training window.",
            "n_unique(train date)",
            "Shows whether training uses a real time span or a narrow sample.",
        ),
        "test_date_count": MetricMetadata(
            "test_date_count",
            "Test dates",
            "model_validation",
            "Number of distinct dates in the out-of-sample test window.",
            "n_unique(test date)",
            "Shows whether validation spans enough market regimes.",
        ),
        "observations": MetricMetadata(
            "observations",
            "Observations",
            "data_quality",
            "Number of observations contributing to a metric.",
            "count(non-null metric observations)",
            "Metrics with few observations should be treated as preliminary.",
        ),
        "avg_names": MetricMetadata(
            "avg_names",
            "Average names",
            "implementation",
            "Average number of securities in a portfolio or quantile bucket.",
            "mean(name_count)",
            "Higher values generally indicate more diversified evidence.",
        ),
        "mean_return": MetricMetadata(
            "mean_return",
            "Mean return",
            "portfolio",
            "Average return over the evaluated period.",
            "mean(periodic_return)",
            "Higher values are better only if robust after volatility, drawdown and turnover.",
        ),
        "top_mean_return": MetricMetadata(
            "top_mean_return",
            "Top-bucket mean return",
            "factor_benchmark",
            "Average forward return of the highest-ranked factor bucket.",
            "mean(return | factor_rank >= 1 - top_quantile)",
            "A transparent long-only factor baseline for comparing ML model selections.",
        ),
        "bottom_mean_return": MetricMetadata(
            "bottom_mean_return",
            "Bottom-bucket mean return",
            "factor_benchmark",
            "Average forward return of the lowest-ranked factor bucket.",
            "mean(return | factor_rank <= bottom_quantile)",
            "Used with top-bucket return to evaluate the economic spread of a factor.",
        ),
        "long_short_mean_return": MetricMetadata(
            "long_short_mean_return",
            "Long-short mean return",
            "factor_benchmark",
            "Average return of the high-factor bucket minus the low-factor bucket.",
            "mean(return_top_bucket - return_bottom_bucket)",
            "Positive values indicate the factor ranking is directionally aligned with realized returns.",
        ),
        "top_long_sharpe": MetricMetadata(
            "top_long_sharpe",
            "Top-bucket Sharpe",
            "factor_benchmark",
            "Risk-adjusted return of the long-only top factor bucket.",
            "mean(return_top) / std(return_top) * sqrt(252 / horizon_days)",
            "Useful as a simple investable factor baseline.",
        ),
        "long_short_sharpe": MetricMetadata(
            "long_short_sharpe",
            "Long-short factor Sharpe",
            "factor_benchmark",
            "Risk-adjusted spread between high-factor and low-factor buckets.",
            "mean(return_top - return_bottom) / std(return_top - return_bottom) * sqrt(252 / horizon_days)",
            "Higher values indicate cleaner factor separation before implementation constraints.",
        ),
        "regime_score": MetricMetadata(
            "regime_score",
            "Market regime score",
            "macro_context",
            "Explainable 0-100 score summarizing risk-on macro conditions.",
            "mean(equity_momentum, credit_relative_strength, dollar_pressure, volatility_condition) * 100",
            "Higher values indicate a more supportive risk-on macro backdrop.",
        ),
        "mae": MetricMetadata(
            "mae",
            "Mean absolute error",
            "time_series_forecast",
            "Average absolute forecast error on the out-of-sample window.",
            "mean(abs(predicted_return - realized_return))",
            "Lower is better; compare it across models on the same series and horizon.",
        ),
        "rmse": MetricMetadata(
            "rmse",
            "Root mean squared error",
            "time_series_forecast",
            "Square-root of the average squared forecast error.",
            "sqrt(mean((predicted_return - realized_return)^2))",
            "Lower is better and large misses are penalized more than in MAE.",
        ),
        "mape": MetricMetadata(
            "mape",
            "Mean absolute percentage error",
            "time_series_forecast",
            "Average absolute error scaled by the realized target magnitude.",
            "mean(abs(error) / max(abs(realized_return), epsilon))",
            "Useful as a scale-free diagnostic, but unstable when realized returns are near zero.",
        ),
        "directional_accuracy": MetricMetadata(
            "directional_accuracy",
            "Directional accuracy",
            "time_series_forecast",
            "Share of out-of-sample observations where forecast and realized return have the same sign.",
            "mean(sign(predicted_return) == sign(realized_return))",
            "Values above 50% can indicate useful directional signal, subject to costs and regimes.",
        ),
        "forecast_return": MetricMetadata(
            "forecast_return",
            "Forecast return",
            "time_series_forecast",
            "Predicted forward return for the selected horizon.",
            "E[price_t+h / price_t - 1 | lagged features]",
            "Treat as scenario context, not as a standalone trading instruction.",
        ),
        "forecast_level": MetricMetadata(
            "forecast_level",
            "Forecast level",
            "time_series_forecast",
            "Implied future price level from the latest close and forecast return.",
            "last_close * (1 + forecast_return)",
            "Useful for visual scenario framing; uncertainty bands should be added before production use.",
        ),
        "forecast_error_std": MetricMetadata(
            "forecast_error_std",
            "Forecast error standard deviation",
            "time_series_forecast",
            "Standard deviation of out-of-sample forecast residuals for a model and horizon.",
            "std(predicted_return - realized_return)",
            "Used as a simple uncertainty band around the latest forecast.",
        ),
        "annualized_volatility": MetricMetadata(
            "annualized_volatility",
            "Annualized volatility",
            "risk",
            "Standard deviation of daily returns scaled to a trading-year basis.",
            "std(daily_return) * sqrt(252)",
            "Higher values indicate larger historical price variability.",
        ),
        "annualized_variance": MetricMetadata(
            "annualized_variance",
            "Annualized variance",
            "risk",
            "Variance corresponding to annualized volatility.",
            "annualized_volatility ^ 2",
            "Useful as a basic risk input; higher values mean wider return dispersion.",
        ),
        "beta_to_benchmark": MetricMetadata(
            "beta_to_benchmark",
            "Beta to benchmark",
            "risk",
            "Sensitivity of the asset return to benchmark return over the selected window.",
            "cov(asset_return, benchmark_return) / var(benchmark_return)",
            "Values above 1 indicate more benchmark sensitivity than the benchmark itself.",
        ),
        "correlation_to_benchmark": MetricMetadata(
            "correlation_to_benchmark",
            "Correlation to benchmark",
            "risk",
            "Linear correlation between asset and benchmark daily returns.",
            "corr(asset_return, benchmark_return)",
            "Higher positive values indicate stronger co-movement with the benchmark.",
        ),
        "avg_pairwise_corr": MetricMetadata(
            "avg_pairwise_corr",
            "Average pairwise correlation",
            "portfolio",
            "Average off-diagonal correlation among selected assets.",
            "mean(corr_i,j for i != j)",
            "Higher values indicate less diversification in the selected basket.",
        ),
    }
)


METRICS_METADATA.update(
    {
        "calmar_ratio": MetricMetadata("calmar_ratio", "Calmar ratio", "portfolio", "CAGR per unit of maximum drawdown.", "CAGR / abs(max_drawdown)", "Higher values indicate better return per drawdown unit."),
        "omega_ratio": MetricMetadata("omega_ratio", "Omega ratio", "portfolio", "Gain/loss payoff ratio around a threshold return.", "sum(max(r-threshold,0)) / abs(sum(min(r-threshold,0)))", "Values above 1 indicate more upside payoff than downside payoff."),
        "kappa_ratio": MetricMetadata("kappa_ratio", "Kappa ratio", "portfolio", "Excess return scaled by lower partial moment.", "mean(r-threshold) / LPM_n^(1/n)", "Higher values indicate better compensation for downside moments."),
        "treynor_ratio": MetricMetadata("treynor_ratio", "Treynor ratio", "portfolio", "Excess return per unit of benchmark beta.", "(return - risk_free) / beta", "Higher values indicate better systematic-risk-adjusted return."),
        "jensen_alpha": MetricMetadata("jensen_alpha", "Jensen alpha", "portfolio", "CAPM abnormal return versus benchmark-implied return.", "return_p - [rf + beta*(return_m-rf)]", "Positive values indicate excess return after market beta adjustment."),
        "m2_measure": MetricMetadata("m2_measure", "M-squared", "portfolio", "Sharpe ratio expressed in benchmark-volatility return units.", "Sharpe_p * vol_market + risk_free", "Makes Sharpe-like performance comparable in return units."),
        "avg_drawdown": MetricMetadata("avg_drawdown", "Average drawdown", "risk", "Mean drawdown level across the evaluation period.", "mean(drawdown_t)", "More negative values indicate more persistent underwater periods."),
        "avg_drawdown_duration": MetricMetadata("avg_drawdown_duration", "Average drawdown duration", "risk", "Average length of drawdown episodes in periods.", "mean(drawdown episode lengths)", "Longer durations indicate slower recovery."),
        "max_drawdown_duration": MetricMetadata("max_drawdown_duration", "Max drawdown duration", "risk", "Longest underwater episode in periods.", "max(drawdown episode lengths)", "Long values indicate extended capital impairment."),
        "ulcer_index": MetricMetadata("ulcer_index", "Ulcer index", "risk", "Root mean square drawdown.", "sqrt(mean(drawdown_pct^2))", "Higher values indicate deeper and/or more persistent drawdowns."),
        "pain_index": MetricMetadata("pain_index", "Pain index", "risk", "Average absolute drawdown.", "mean(abs(drawdown_pct))", "Higher values indicate more time spent below prior highs."),
        "recovery_factor": MetricMetadata("recovery_factor", "Recovery factor", "portfolio", "Total return per unit of maximum drawdown.", "abs(total_return) / abs(max_drawdown)", "Higher values indicate stronger recovery relative to peak loss."),
        "skewness_returns": MetricMetadata("skewness_returns", "Return skewness", "distribution", "Asymmetry of periodic return distribution.", "skew(returns)", "Negative values indicate left-tail asymmetry."),
        "kurtosis_returns": MetricMetadata("kurtosis_returns", "Return kurtosis", "distribution", "Excess tail thickness of return distribution.", "kurtosis(returns)", "High values indicate fat tails."),
        "var_95": MetricMetadata("var_95", "VaR 95%", "risk", "Parametric 95% value at risk.", "mean(return) - 1.645*std(return)", "More negative values indicate larger expected tail loss."),
        "var_99": MetricMetadata("var_99", "VaR 99%", "risk", "Parametric 99% value at risk.", "mean(return) - 2.326*std(return)", "More negative values indicate larger severe-tail loss."),
        "cvar_95": MetricMetadata("cvar_95", "CVaR 95%", "risk", "Expected shortfall below 95% VaR threshold.", "mean(returns <= VaR_95)", "More negative values indicate worse tail-loss severity."),
        "cvar_99": MetricMetadata("cvar_99", "CVaR 99%", "risk", "Expected shortfall below 99% VaR threshold.", "mean(returns <= VaR_99)", "More negative values indicate worse extreme-tail severity."),
        "tail_ratio": MetricMetadata("tail_ratio", "Tail ratio", "distribution", "Upside tail divided by downside tail magnitude.", "abs(P95 return) / abs(P5 return)", "Values above 1 indicate more upside than downside tail."),
        "avg_up_month": MetricMetadata("avg_up_month", "Average up period", "distribution", "Average return in positive periods.", "mean(return | return > 0)", "Higher values indicate stronger upside payoff."),
        "avg_down_month": MetricMetadata("avg_down_month", "Average down period", "distribution", "Average return in negative periods.", "mean(return | return < 0)", "More negative values indicate harsher downside payoff."),
        "up_down_capture": MetricMetadata("up_down_capture", "Up/down capture", "portfolio", "Relative participation in benchmark up periods divided by down periods.", "up_capture / down_capture", "Higher values indicate better upside participation versus downside participation."),
        "turnover_annual": MetricMetadata("turnover_annual", "Annual turnover", "implementation", "Annualized one-way turnover from portfolio weight changes.", "sum(abs(delta weights))/2 annualized", "Higher values raise cost and capacity risk."),
        "avg_holding_period": MetricMetadata("avg_holding_period", "Average holding period", "implementation", "Approximate days implied by annual turnover.", "252 / turnover_annual", "Longer holding periods generally imply lower implementation burden."),
        "estimated_cost_bp": MetricMetadata("estimated_cost_bp", "Estimated cost", "implementation", "Annualized transaction cost estimate in basis points.", "turnover_annual * cost_per_turn_bp", "Higher costs reduce implementable performance."),
        "net_sharpe": MetricMetadata("net_sharpe", "Net Sharpe", "portfolio", "Sharpe ratio after estimated transaction costs.", "Sharpe(return - cost_drag)", "Use this for implementable comparisons when costs are material."),
        "allocation_effect": MetricMetadata("allocation_effect", "Allocation effect", "attribution", "Brinson effect from over/underweighting sectors.", "(w_p - w_b) * (r_b_sector - r_b_total)", "Positive values indicate beneficial sector allocation."),
        "selection_effect": MetricMetadata("selection_effect", "Selection effect", "attribution", "Brinson effect from stock selection within sectors.", "w_b * (r_p_sector - r_b_sector)", "Positive values indicate beneficial security selection."),
        "interaction_effect": MetricMetadata("interaction_effect", "Interaction effect", "attribution", "Residual Brinson interaction between allocation and selection.", "(w_p - w_b) * (r_p_sector - r_b_sector)", "Positive values indicate aligned allocation and selection."),
        "active_return": MetricMetadata("active_return", "Active return", "portfolio", "Portfolio return minus benchmark return.", "return_portfolio - return_benchmark", "Positive values indicate outperformance before attribution split."),
    }
)


ALIASES = {
    "rank_information_coefficient": "rank_ic",
    "information_coefficient": "ic",
    "avg_return": "alpha",
    "drawdown": "max_drawdown",
    "vol": "volatility",
    "ir": "information_ratio",
    "r2": "r2_os",
    "r2_oos": "r2_os",
    "sharpe_ratio": "sharpe",
    "factor_long_short_sharpe": "long_short_sharpe",
    "ls_sharpe": "long_short_sharpe",
    "ls_mean_return": "long_short_mean_return",
    "top_bucket_return": "top_mean_return",
    "mean_absolute_error": "mae",
    "root_mean_squared_error": "rmse",
    "dir_acc": "directional_accuracy",
    "forecasted_return": "forecast_return",
    "forecasted_level": "forecast_level",
    "std": "annualized_volatility",
    "variance": "annualized_variance",
    "beta_vs_benchmark": "beta_to_benchmark",
    "corr_to_benchmark": "correlation_to_benchmark",
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
