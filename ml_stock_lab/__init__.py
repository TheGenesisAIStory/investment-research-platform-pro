"""ml_stock_lab: reusable ML equity valuation, signals and diagnostics tools."""

from .datasets import (
    FundamentalDatasetBuilder,
    add_basic_features,
    build_company_selection_panel,
    build_company_selection_widget,
    company_source_paths,
    filter_company_selection_panel,
    load_financial_db_panel,
    load_panel_csv,
    make_forward_returns,
    normalize_panel,
    select_numeric_features,
)
from .evaluation import (
    ExperimentStatus,
    FactorAlphaEvaluator,
    PanelValidationResult,
    PerformanceMetrics,
    annualized_volatility,
    build_status_from_signals,
    max_drawdown,
    plot_cumulative_returns,
    quintile_risk_report,
    save_status_file,
    sharpe_ratio,
    should_run_experiment,
    turnover,
    validate_panel,
)
from .prediction import FundamentalPredictorRF, ExpectedReturnModel, describe_temporal_split, oos_r2, temporal_train_test_split
from .signals import (
    EnsembleMispricingSignal,
    MispricingSignal,
    assign_quantiles,
    compute_absolute_mispricing,
    compute_relative_mispricing,
    cross_sectional_zscore,
    make_quantile_portfolios,
    quintile_returns_wide,
    rank_scores,
    top_bottom,
)
from .tracking import build_experiment_summary, save_experiment_summary
from .valuation import (
    PeerImpliedGBRT,
    PeerImpliedLasso,
    PeerImpliedOLS,
    PeerImpliedRF,
    PeerImpliedValuator,
    PeerOLSValuator,
    RollingPeerValuator,
)

# Compatibility bridge: the canonical v1 workstation package lives under
# research_platform_definitive/src/ml_stock_lab and exposes the app/job entry
# points. When Python is launched from the repo root, this legacy package can
# shadow the editable install; load only the missing canonical entry points
# under a private module name so old imports keep working.
try:
    run_ml_stock_lab_experiment
except NameError:
    import importlib.util
    import sys
    from pathlib import Path

    canonical_init = Path(__file__).resolve().parents[1] / "research_platform_definitive" / "src" / "ml_stock_lab" / "__init__.py"
    if canonical_init.exists():
        spec = importlib.util.spec_from_file_location(
            "_research_platform_definitive_ml_stock_lab",
            canonical_init,
            submodule_search_locations=[str(canonical_init.parent)],
        )
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            run_ml_stock_lab_experiment = module.run_ml_stock_lab_experiment
            train_ml_model_suite = module.train_ml_model_suite
            load_training_panel = module.load_training_panel
            summarize_training_with_ollama = module.summarize_training_with_ollama

__all__ = [
    "EnsembleMispricingSignal",
    "ExperimentStatus",
    "ExpectedReturnModel",
    "FactorAlphaEvaluator",
    "FundamentalPredictorRF",
    "FundamentalDatasetBuilder",
    "MispricingSignal",
    "PanelValidationResult",
    "PeerImpliedGBRT",
    "PeerImpliedLasso",
    "PeerImpliedOLS",
    "PeerImpliedRF",
    "PeerImpliedValuator",
    "PeerOLSValuator",
    "PerformanceMetrics",
    "RollingPeerValuator",
    "add_basic_features",
    "annualized_volatility",
    "assign_quantiles",
    "build_company_selection_panel",
    "build_company_selection_widget",
    "build_experiment_summary",
    "build_status_from_signals",
    "company_source_paths",
    "compute_absolute_mispricing",
    "compute_relative_mispricing",
    "cross_sectional_zscore",
    "describe_temporal_split",
    "filter_company_selection_panel",
    "load_financial_db_panel",
    "load_panel_csv",
    "make_forward_returns",
    "make_quantile_portfolios",
    "max_drawdown",
    "normalize_panel",
    "oos_r2",
    "plot_cumulative_returns",
    "quintile_returns_wide",
    "quintile_risk_report",
    "rank_scores",
    "save_experiment_summary",
    "save_status_file",
    "select_numeric_features",
    "sharpe_ratio",
    "should_run_experiment",
    "temporal_train_test_split",
    "top_bottom",
    "turnover",
    "validate_panel",
    "run_ml_stock_lab_experiment",
    "load_training_panel",
    "summarize_training_with_ollama",
    "train_ml_model_suite",
]
