"""Readable metric metadata for model, portfolio and valuation dashboards."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
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
    source_paper: str = ""
    source_doi: str = ""
    interpretation_range: str = ""
    is_higher_better: bool = True
    metric_family: str = ""

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

METRICS_METADATA.update(
    {
        "sterling_ratio": MetricMetadata("sterling_ratio", "Sterling ratio", "portfolio", "Annualized return divided by average annual drawdown.", "annualized_return / abs(avg_annual_drawdown)", "Higher values indicate stronger return per average drawdown unit."),
        "burke_ratio": MetricMetadata("burke_ratio", "Burke ratio", "portfolio", "Annualized return divided by the square root of squared drawdowns.", "annualized_return / sqrt(sum(drawdown_i^2))", "Higher values indicate better return relative to multiple drawdown episodes."),
        "martin_ratio": MetricMetadata("martin_ratio", "Martin ratio", "portfolio", "Annualized return divided by Ulcer index.", "annualized_return / ulcer_index", "Higher values indicate better compensation for persistent drawdowns."),
        "gain_to_pain_ratio": MetricMetadata("gain_to_pain_ratio", "Gain-to-pain ratio", "portfolio", "Total gains divided by absolute total losses.", "sum(gains) / abs(sum(losses))", "Values above 1 indicate more aggregate gains than losses."),
        "kappa_3_ratio": MetricMetadata("kappa_3_ratio", "Kappa 3 ratio", "portfolio", "Excess return divided by third lower partial moment.", "mean(r-threshold) / LPM_3^(1/3)", "Higher values indicate better compensation for downside skew and tail risk."),
        "historical_var_95": MetricMetadata("historical_var_95", "Historical VaR 95%", "risk", "Empirical fifth percentile of return distribution.", "quantile(returns, 5%)", "More negative values indicate worse historical tail loss."),
        "historical_var_99": MetricMetadata("historical_var_99", "Historical VaR 99%", "risk", "Empirical first percentile of return distribution.", "quantile(returns, 1%)", "More negative values indicate worse extreme historical tail loss."),
        "parametric_var_99": MetricMetadata("parametric_var_99", "Parametric VaR 99%", "risk", "Gaussian one-percent value at risk.", "mean(return) - 2.326 * std(return)", "More negative values indicate larger Gaussian tail-loss estimate."),
        "common_sense_ratio": MetricMetadata("common_sense_ratio", "Common sense ratio", "risk", "Tail ratio multiplied by gain-to-pain style payoff quality.", "tail_ratio * gain_to_pain_ratio", "Higher values indicate attractive upside/downside payoff quality."),
        "ff3_r_squared": MetricMetadata("ff3_r_squared", "FF3 R-squared", "factor", "Regression explanatory power of Fama-French three factors.", "1 - SSE_FF3 / SST", "Higher values indicate stronger explanation by market, size and value factors."),
        "ff5_r_squared": MetricMetadata("ff5_r_squared", "FF5 R-squared", "factor", "Regression explanatory power of Fama-French five factors.", "1 - SSE_FF5 / SST", "Higher values indicate stronger explanation by FF5 factors."),
        "ff3_mkt_beta": MetricMetadata("ff3_mkt_beta", "FF3 market beta", "factor", "Market loading from FF3 regression.", "coefficient(MKT_RF)", "Higher values indicate stronger market exposure."),
        "ff3_smb_beta": MetricMetadata("ff3_smb_beta", "FF3 SMB beta", "factor", "Size-factor loading from FF3 regression.", "coefficient(SMB)", "Positive values indicate small-cap factor exposure."),
        "ff3_hml_beta": MetricMetadata("ff3_hml_beta", "FF3 HML beta", "factor", "Value-factor loading from FF3 regression.", "coefficient(HML)", "Positive values indicate value factor exposure."),
        "ff5_rmw_beta": MetricMetadata("ff5_rmw_beta", "FF5 RMW beta", "factor", "Profitability-factor loading from FF5 regression.", "coefficient(RMW)", "Positive values indicate robust-profitability exposure."),
        "ff5_cma_beta": MetricMetadata("ff5_cma_beta", "FF5 CMA beta", "factor", "Investment-factor loading from FF5 regression.", "coefficient(CMA)", "Positive values indicate conservative-investment exposure."),
        "ic_mean_21d": MetricMetadata("ic_mean_21d", "Mean IC 21D", "model_validation", "Average information coefficient for 21-day horizon.", "mean(corr(prediction, realized_return_21d))", "Higher positive values indicate stronger short-horizon rank signal."),
        "ic_mean_63d": MetricMetadata("ic_mean_63d", "Mean IC 63D", "model_validation", "Average information coefficient for 63-day horizon.", "mean(corr(prediction, realized_return_63d))", "Higher positive values indicate stronger medium-horizon rank signal."),
        "ic_ir_63d": MetricMetadata("ic_ir_63d", "IC IR 63D", "model_validation", "Information ratio of 63-day IC over time.", "mean(IC_63d) / std(IC_63d)", "Higher values indicate more stable predictive information."),
        "rank_ic_63d": MetricMetadata("rank_ic_63d", "Rank IC 63D", "model_validation", "Spearman rank IC for 63-day horizon.", "spearman(predicted_rank, realized_return_63d_rank)", "Higher positive values indicate better rank ordering."),
        "cot_net_speculator_zscore": MetricMetadata("cot_net_speculator_zscore", "COT net speculator z-score", "smart_money", "Z-score of net non-commercial COT positioning.", "(net_speculator - mean_52w) / std_52w", "Positive values indicate crowded speculative long positioning versus one-year history."),
        "etf_flow_momentum_score": MetricMetadata("etf_flow_momentum_score", "ETF flow momentum score", "smart_money", "Normalized ETF flow momentum composite.", "zscore(flow_1m) + zscore(flow_3m)", "Higher values indicate stronger recent ETF demand."),
        "dark_pool_activity_proxy": MetricMetadata("dark_pool_activity_proxy", "Dark pool activity proxy", "smart_money", "Proxy for off-exchange or block trading activity.", "block_or_off_exchange_volume / total_volume", "Higher values indicate more opaque institutional trading activity."),
    }
)

METRICS_METADATA.update(
    {
        "regime_label": MetricMetadata(
            "regime_label",
            "Market regime label",
            "macro_context",
            "Discrete market-regime classification used for research context.",
            "rule_based_label(equity_momentum, credit_spread, VIX, yield_slope, commodities)",
            "Labels are contextual diagnostics, not direct trading signals.",
        ),
        "regime_confidence": MetricMetadata(
            "regime_confidence",
            "Regime confidence",
            "macro_context",
            "Availability- and rule-strength-adjusted confidence for the regime label.",
            "base_rule_confidence * signal_availability_adjustment",
            "Higher values indicate more complete and internally consistent regime evidence.",
        ),
        "equity_momentum_21d": MetricMetadata(
            "equity_momentum_21d",
            "Equity momentum 21D",
            "macro_context",
            "Twenty-one trading day return of the equity market proxy.",
            "SPY_t / SPY_t-21 - 1",
            "Negative values below -2% contribute to risk-off classification.",
        ),
        "credit_spread": MetricMetadata(
            "credit_spread",
            "Credit spread proxy",
            "macro_context",
            "Relative performance of high-yield credit versus long-duration Treasuries.",
            "return_21d(HYG) - return_21d(TLT)",
            "Negative values indicate credit stress or defensive duration leadership.",
        ),
        "vix_proxy": MetricMetadata(
            "vix_proxy",
            "VIX proxy",
            "macro_context",
            "Raw or percentile-normalized volatility-stress proxy.",
            "VIX level or rank_pct(VIX, 252d)",
            "High values flag risk-off or crisis conditions.",
        ),
        "yield_slope": MetricMetadata(
            "yield_slope",
            "Yield slope proxy",
            "macro_context",
            "Long-rate minus front-end-rate proxy.",
            "10Y yield proxy - 2Y/front-end yield proxy",
            "Flat or inverted slopes are treated as macro warnings.",
        ),
        "commodity_momentum": MetricMetadata(
            "commodity_momentum",
            "Commodity momentum",
            "macro_context",
            "Twenty-one day commodity trend proxy, preferring Brent then gold.",
            "commodity_proxy_t / commodity_proxy_t-21 - 1",
            "Used as context for reflation or defensive commodity regimes.",
        ),
    }
)


_METRIC_PAPER_DOI = {
    "Gen.is.IA internal": "https://genisia.local/methodology/internal",
    "Sharpe 1966": "10.1086/294846",
    "Sortino-Price 1994": "https://www.pm-research.com/content/iijinvest/3/3/59",
    "Young 1991": "https://www.tandfonline.com/doi/abs/10.1080/09603109100000023",
    "Artzner-Delbaen-Eber-Heath 1999": "10.1111/1467-9965.00068",
    "Fama-French 2015": "10.1016/j.jfineco.2014.10.010",
    "Grinold-Kahn 2000": "https://www.mheducation.com/highered/product/active-portfolio-management-grinold-kahn/M9780070248823.html",
    "Jensen 1968": "10.2307/2325404",
}


_METRIC_CATEGORY_DEFAULTS = {
    "portfolio": ("Sharpe 1966", "performance", True, "Higher is better when robust after costs."),
    "risk": ("Artzner-Delbaen-Eber-Heath 1999", "risk", False, "Lower risk magnitudes are usually better, conditional on expected return."),
    "distribution": ("Artzner-Delbaen-Eber-Heath 1999", "risk", True, "Interpret with skew, tail and sample-size context."),
    "model_validation": ("Grinold-Kahn 2000", "model", True, "Positive and stable values are better."),
    "factor": ("Fama-French 2015", "factor", True, "Exposure is not inherently good or bad; use with target mandate."),
    "smart_money": ("Gen.is.IA internal", "smart_money", True, "Directional interpretation depends on source and crowding context."),
    "implementation": ("Gen.is.IA internal", "portfolio", False, "Lower values usually mean easier implementation."),
    "attribution": ("Gen.is.IA internal", "portfolio", True, "Positive values contribute to active return."),
    "data_quality": ("Gen.is.IA internal", "model", True, "Higher coverage is generally better."),
    "model_governance": ("Gen.is.IA internal", "model", True, "Use with governance thresholds, not in isolation."),
}


_METRIC_LATEX = {
    "information_ratio": r"IR=\frac{R_p-R_b}{\sigma(R_p-R_b)}",
    "rank_ic": r"\rho_s(\hat{r}_{t+h},r_{t+h})",
    "rank_ic_63d": r"\rho_s(\hat{r}_{t+63},r_{t+63})",
    "cvar_95": r"ES_{95}=E[R\mid R\le VaR_{95}]",
    "cvar_99": r"ES_{99}=E[R\mid R\le VaR_{99}]",
    "historical_var_95": r"VaR_{95}=Q_{0.05}(R)",
    "historical_var_99": r"VaR_{99}=Q_{0.01}(R)",
    "parametric_var_99": r"\mu_R-2.326\sigma_R",
    "calmar_ratio": r"\frac{\bar{r}_{ann}}{|MDD|}",
    "ulcer_index": r"\sqrt{\frac{1}{T}\sum DD_t^2}",
}


def _metric_latex_from_formula(formula: str) -> str:
    safe = str(formula or "see implementation").replace("_", r"\_")
    return rf"\text{{{safe[:180]}}}"


def _complete_metric_metadata(meta: MetricMetadata) -> MetricMetadata:
    category_key = str(meta.category or "").strip().lower()
    paper, family, higher_better, default_range = _METRIC_CATEGORY_DEFAULTS.get(
        category_key,
        ("Gen.is.IA internal", category_key or "model", True, "Interpret in strategy and sample context."),
    )
    if meta.id in {"sharpe", "sharpe_long_short", "sharpe_long_short_net_cost", "net_sharpe"}:
        paper, family, higher_better, default_range = ("Sharpe 1966", "performance", True, ">1 good, >2 strong, sample dependent.")
    if meta.id in {"sortino", "kappa_ratio", "kappa_3_ratio", "omega_ratio", "gain_to_pain_ratio"}:
        paper, family, higher_better, default_range = ("Sortino-Price 1994", "performance", True, ">1 typically preferred, strategy dependent.")
    if meta.id in {"calmar_ratio", "sterling_ratio", "burke_ratio", "martin_ratio"}:
        paper, family, higher_better, default_range = ("Young 1991", "performance", True, ">1 strong, 0.5-1 acceptable, <0.5 weak.")
    if meta.id.startswith("ff3_") or meta.id.startswith("ff5_"):
        paper, family, higher_better, default_range = ("Fama-French 2015", "factor", True, "Exposure metric; sign depends on mandate.")
    if meta.id.startswith("ic_") or meta.id.startswith("rank_ic") or meta.id in {"ic", "rank_ic"}:
        paper, family, higher_better, default_range = ("Grinold-Kahn 2000", "model", True, "Positive and stable is preferred.")
    if "cvar" in meta.id or "var" in meta.id:
        paper, family, higher_better, default_range = ("Artzner-Delbaen-Eber-Heath 1999", "risk", False, "Less negative/lower loss magnitude is preferred.")
    source_paper = meta.source_paper or meta.paper or paper
    source_doi = meta.source_doi or _METRIC_PAPER_DOI.get(source_paper, "https://genisia.local/methodology/internal")
    formula_latex = meta.formula_latex or _METRIC_LATEX.get(meta.id, _metric_latex_from_formula(meta.formula))
    interpretation_range = meta.interpretation_range or meta.typical_range or default_range
    typical_range = meta.typical_range or interpretation_range
    return replace(
        meta,
        formula_latex=formula_latex,
        paper=meta.paper or source_paper,
        source_paper=source_paper,
        source_doi=source_doi,
        interpretation_range=interpretation_range,
        typical_range=typical_range,
        is_higher_better=higher_better,
        metric_family=meta.metric_family or family,
    )


METRICS_METADATA = {key: _complete_metric_metadata(meta) for key, meta in METRICS_METADATA.items()}


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
