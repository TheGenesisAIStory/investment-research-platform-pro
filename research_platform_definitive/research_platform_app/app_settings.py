"""Persistent UI/settings artifacts for the Streamlit workstation.

The settings file is intentionally lightweight: it stores user-facing
preferences and model routing choices, while the canonical data contracts
remain in research_platform_core and the Database Finanziario.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


SETTINGS_VERSION = "1.0"

DEFAULT_MODEL_REGISTRY: list[dict[str, Any]] = [
    {
        "id": "ols",
        "name": "OLS Fair Value",
        "type": "regressor",
        "universe": "shared equity panel",
        "training_window": "configurable 2000-2026",
        "features": "fundamental value/quality/risk",
        "primary_metric": "R2_OS",
    },
    {
        "id": "lasso",
        "name": "Lasso Fair Value",
        "type": "regularized regressor",
        "universe": "shared equity panel",
        "training_window": "configurable 2000-2026",
        "features": "sparse fundamental factor set",
        "primary_metric": "R2_OS",
    },
    {
        "id": "rf",
        "name": "Random Forest",
        "type": "nonlinear regressor",
        "universe": "shared equity panel",
        "training_window": "configurable 2000-2026",
        "features": "fundamental + factor interactions",
        "primary_metric": "quintile spread",
    },
    {
        "id": "gbrt",
        "name": "Gradient Boosted Trees",
        "type": "nonlinear regressor",
        "universe": "shared equity panel",
        "training_window": "configurable 2000-2026",
        "features": "fundamental + momentum/risk factors",
        "primary_metric": "rank IC / quintile spread",
    },
    {
        "id": "ensemble",
        "name": "Research Ensemble",
        "type": "composite",
        "universe": "shared equity panel",
        "training_window": "latest validated model stack",
        "features": "weighted model scores",
        "primary_metric": "composite conviction",
    },
]

DEFAULT_SETTINGS: dict[str, Any] = {
    "version": SETTINGS_VERSION,
    "screener": {
        "default_template": "Custom",
        "custom_universes": [],
        "default_columns": [],
        "enabled_model_scores": ["ols"],
        "composite_model": "research_composite",
    },
    "models": {
        "registry": DEFAULT_MODEL_REGISTRY,
        "active_models": ["ols"],
        "composite_weights": {"ols": 1.0},
        "screener_enabled": True,
    },
    "data": {
        "refresh_policy": "daily_incremental",
        "coverage_strictness": "non-strict",
        "coverage_watch_threshold": 80,
        "limited_history_ok": True,
        "network_timeout_watch_threshold": 10,
        "ohlcv_parquet_root": "",
    },
    "smart_money": {
        "min_score_watch": 50.0,
        "require_recent_event_default": False,
        "recent_event_window_days": 45,
    },
    "banking": {
        "min_bank_score": 0.0,
        "include_market_panel": True,
    },
}


def settings_path(workspace_root: Path) -> Path:
    return Path(workspace_root) / "config" / "platform_settings.json"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_platform_settings(workspace_root: Path) -> dict[str, Any]:
    path = settings_path(workspace_root)
    if not path.exists():
        return copy.deepcopy(DEFAULT_SETTINGS)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return copy.deepcopy(DEFAULT_SETTINGS)
    return _deep_merge(DEFAULT_SETTINGS, payload if isinstance(payload, dict) else {})


def save_platform_settings(workspace_root: Path, settings: dict[str, Any]) -> Path:
    path = settings_path(workspace_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _deep_merge(DEFAULT_SETTINGS, settings)
    payload["version"] = SETTINGS_VERSION
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def update_platform_settings(workspace_root: Path, section: str, values: dict[str, Any]) -> Path:
    settings = load_platform_settings(workspace_root)
    current = settings.get(section, {})
    settings[section] = _deep_merge(current if isinstance(current, dict) else {}, values)
    return save_platform_settings(workspace_root, settings)


def model_registry_frame(settings: dict[str, Any]):
    import pandas as pd

    registry = settings.get("models", {}).get("registry") or DEFAULT_MODEL_REGISTRY
    return pd.DataFrame(registry)


def build_model_score_view(signals, active_models: list[str], weights: dict[str, float]):
    """Return signals with side-by-side model score columns when artifacts exist.

    Current artifacts often expose one canonical score. This helper is forward
    compatible with score_<model_id> columns produced by future training runs.
    """
    import pandas as pd

    if signals is None or signals.empty:
        return pd.DataFrame()
    view = signals.copy()
    score_cols: list[str] = []
    for model_id in active_models:
        candidate = f"score_{model_id}"
        if candidate in view.columns:
            score_cols.append(candidate)
        elif model_id in view.columns:
            view[f"score_{model_id}"] = pd.to_numeric(view[model_id], errors="coerce")
            score_cols.append(f"score_{model_id}")
        elif "score" in view.columns and len(active_models) == 1:
            view[f"score_{model_id}"] = pd.to_numeric(view["score"], errors="coerce")
            score_cols.append(f"score_{model_id}")
    if score_cols:
        weighted = []
        total_weight = 0.0
        for col in score_cols:
            model_id = col.replace("score_", "", 1)
            weight = float(weights.get(model_id, 0.0))
            if weight <= 0:
                continue
            weighted.append(pd.to_numeric(view[col], errors="coerce") * weight)
            total_weight += weight
        if weighted and total_weight > 0:
            view["score_composite"] = sum(weighted) / total_weight
    return view
