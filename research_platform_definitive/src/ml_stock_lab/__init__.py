"""ml_stock_lab: ML equity valuation, screening and portfolio research tools."""

from .datasets import FundamentalDatasetBuilder, load_aqr_factor_panel, load_artifact_panel, load_financial_db_panel, normalize_panel, validate_panel_coverage
from .features import add_basic_features, add_model_based_mispricing_features, make_forward_returns, select_numeric_features
from .valuation import PeerImpliedValuator, RollingCrossSectionalValuator
from .signals import EnsembleMispricingSignal, compute_absolute_mispricing, compute_relative_mispricing, cross_sectional_zscore
from .screening import StockScreener, assign_quantiles, rank_scores, top_bottom
from .prediction import ExpectedReturnModel, describe_temporal_split, temporal_train_test_split
from .portfolio import long_short_weights, make_quantile_portfolios, mean_variance_from_forecasts
from .evaluation import evaluate_quintile_backtest, factor_alpha, oos_r2, sharpe_ratio, transaction_cost_adjusted_returns, turnover
from .lab import run_ml_stock_lab_experiment

__all__ = [
    "FundamentalDatasetBuilder",
    "load_artifact_panel",
    "load_aqr_factor_panel",
    "load_financial_db_panel",
    "normalize_panel",
    "validate_panel_coverage",
    "add_basic_features",
    "add_model_based_mispricing_features",
    "make_forward_returns",
    "select_numeric_features",
    "PeerImpliedValuator",
    "RollingCrossSectionalValuator",
    "EnsembleMispricingSignal",
    "compute_absolute_mispricing",
    "compute_relative_mispricing",
    "cross_sectional_zscore",
    "StockScreener",
    "assign_quantiles",
    "rank_scores",
    "top_bottom",
    "ExpectedReturnModel",
    "describe_temporal_split",
    "temporal_train_test_split",
    "long_short_weights",
    "make_quantile_portfolios",
    "mean_variance_from_forecasts",
    "evaluate_quintile_backtest",
    "factor_alpha",
    "oos_r2",
    "sharpe_ratio",
    "transaction_cost_adjusted_returns",
    "turnover",
    "run_ml_stock_lab_experiment",
]
