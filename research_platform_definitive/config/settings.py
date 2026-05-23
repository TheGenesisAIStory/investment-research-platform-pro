# ============================================================
# CORE SETTINGS & CONFIGURATIONS
# ============================================================

VALUATION_CONFIG = {
    "dcf": {
        "cash_flow_definition": "fcff",
        "discounting_approach": "wacc",
        "terminal_value_method": "gordon_growth",
        "use_midyear_convention": True,
        "normalize_margins": True,
        "normalize_working_capital": True,
        "normalize_capex": True,
    },
    "relative_valuation": {
        "multiple_set": ["pe", "ev_ebitda", "ev_sales", "pbv"],
        "aggregation_method": "median",
        "outlier_filtering": True,
        "winsorize_percentile": 0.05,
    },
    "residual_income": {
        "book_value_anchor": True,
        "cost_of_equity_method": "capm",
    },
    "dividend_discount_model": {
        "enabled_only_for_dividend_payers": True,
        "dividend_growth_method": "historical_blended",
    },
    "economic_profit": {
        "capital_charge_method": "wacc_times_invested_capital",
    },
}

ML_CONFIG = {
    "feature_set_type": "fundamental_plus_market",
    "feature_selection_method": "hybrid",
    "scaling_method": "robust",
    "train_window_months": 60,
    "validation_window_months": 12,
    "test_window_months": 12,
    "cross_validation_folds": 5,
    "time_series_split": True,
    "hyperparameter_tuning": True,
    "performance_metric_regression": "rmse",
    "performance_metric_classification": "f1",
    "classification_labels": ["SELL", "HOLD", "BUY"],
    "prediction_horizon_months": 12,
}

GOVERNANCE_CONFIG = {
    "random_seed": 42,
    "run_sensitivity_analysis": True,
    "sensitivity_discount_rate_bps": [-200, -100, 0, 100, 200],
    "sensitivity_terminal_growth_bps": [-100, 0, 100],
    "run_scenario_analysis": True,
    "scenario_names": ["bear", "base", "bull"],
    "generate_investment_view": True,
    "generate_model_comparison_table": True,
    "generate_peer_comparison_table": True,
    "generate_explainability_outputs": True,
    "compare_train_vs_test_metrics": True,
    "require_out_of_sample_validation": True,
    "store_model_ranking": True,
}
