"""ml_stock_lab: reusable ML equity valuation, signals and diagnostics tools.

This top-level package is kept for the original notebooks and integrations.
The Streamlit workstation now also ships a richer implementation under
``research_platform_definitive/src/ml_stock_lab``.  When both are present in
the same checkout, Python can resolve this legacy package first; the small
compatibility bridge at the bottom exposes the modern workstation runners
without disturbing the legacy subpackages.
"""

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


def _load_workstation_ml_stock_lab():
    """Load the workstation package under a private alias.

    Loading it as ``ml_stock_lab`` would collide with this legacy package and
    its notebook-facing submodules.  A private alias lets modern modules use
    their own relative imports while we expose selected public entry points
    below.
    """
    import importlib
    import importlib.util
    import sys
    from pathlib import Path

    alias = "_genisia_workstation_ml_stock_lab"
    if alias in sys.modules:
        return sys.modules[alias]

    package_dir = Path(__file__).resolve().parents[1] / "research_platform_definitive" / "src" / "ml_stock_lab"
    init_file = package_dir / "__init__.py"
    if not init_file.exists():
        return None

    spec = importlib.util.spec_from_file_location(alias, init_file, submodule_search_locations=[str(package_dir)])
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)

    for submodule in ("factor_registry", "lab", "training"):
        try:
            sys.modules.setdefault(f"{__name__}.{submodule}", importlib.import_module(f"{alias}.{submodule}"))
        except Exception:
            continue
    return module


_workstation_ml_stock_lab = None
try:
    _workstation_ml_stock_lab = _load_workstation_ml_stock_lab()
except Exception:
    _workstation_ml_stock_lab = None

if _workstation_ml_stock_lab is not None:
    for _name in (
        "run_ml_stock_lab_experiment",
        "load_training_panel",
        "summarize_training_with_ollama",
        "train_ml_model_suite",
    ):
        if hasattr(_workstation_ml_stock_lab, _name):
            globals()[_name] = getattr(_workstation_ml_stock_lab, _name)

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
]

for _name in (
    "run_ml_stock_lab_experiment",
    "load_training_panel",
    "summarize_training_with_ollama",
    "train_ml_model_suite",
):
    if _name in globals() and _name not in __all__:
        __all__.append(_name)
