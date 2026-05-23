import logging
from pathlib import Path
from .presets import PRESET_MAP
from .settings import VALUATION_CONFIG, ML_CONFIG, GOVERNANCE_CONFIG

def build_master_request(mr_dict):
    if not mr_dict:
        raise ValueError("\n🚨 CRITICAL: MASTER_REQUEST is empty. Please apply selection.\n")
    return mr_dict

def sync_all_configs_from_master_request(mr, experiment_base):
    selected_profile = mr.get('user_profile', 'cfa_balanced')
    preset_values = PRESET_MAP.get(selected_profile, PRESET_MAP['cfa_balanced'])

    experiment = experiment_base.copy()
    experiment.update(preset_values)

    experiment['selected_company'] = mr['ticker']
    experiment['start_date'] = mr['start_date']
    experiment['end_date'] = mr['end_date']
    experiment['market_focus'] = mr['market']
    experiment['reporting_currency'] = mr['currency']
    experiment['peer_selection_method'] = mr['peer_method']
    experiment['n_peers'] = mr['n_peers']
    experiment['manual_peers'] = mr['manual_peers']
    experiment['dcf_horizon_years'] = mr['dcf_horizon']
    experiment['run_valuation_models'] = mr['models']

    experiment["analysis_mode"] = "company_analysis_with_ml_overlay"
    experiment["primary_objective"] = "worth_investing_decision"
    experiment["target"] = "target_mispricing_score"
    experiment["secondary_target"] = "future_excess_return_12m"
    experiment["feature_blocks"] = ["dcf_inputs", "quality_ratios", "valuation_outputs", "market_overlay"]

    experiment["VALUATION_CONFIG"] = VALUATION_CONFIG
    experiment["ML_CONFIG"] = ML_CONFIG
    experiment["GOVERNANCE_CONFIG"] = GOVERNANCE_CONFIG

    return experiment

def validate_master_request(experiment, mr):
    warnings = []
    ticker = experiment.get('selected_company', '')

    if not ticker or ticker.islower() or "_" in ticker:
        warnings.append(f"⚠️ FORMAT MISMATCH: Ticker '{ticker}' is not canonical.")
    if not experiment.get('start_date') or not experiment.get('end_date'):
        warnings.append("⚠️ DATE ERROR: Analysis dates are missing.")

    if warnings:
        for w in warnings:
            print(w)
        raise ValueError("Strict diagnostics failed: Downstream configuration mismatch detected.")

    print("✅ All settings perfectly consistent and fully integrated from MASTER_REQUEST.")
    return True
