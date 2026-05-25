"""LLM advisor helpers for model selection, stock-pick explanation and audit."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from .llm_client import LlmResponse, OllamaClient
from .llm_prompts import PROMPT_REGISTRY, SYSTEM_POLICY_IT


def _compact_records(frame: pd.DataFrame | None, columns: list[str] | None = None, limit: int = 12) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    view = frame.copy()
    if columns:
        view = view[[col for col in columns if col in view.columns]]
    return json.loads(view.head(limit).to_json(orient="records", date_format="iso"))


def _call_prompt(
    prompt_id: str,
    payload: dict[str, Any],
    *,
    client: OllamaClient | None = None,
    model: str | None = None,
    temperature: float = 0.2,
) -> dict[str, Any]:
    prompt_template = PROMPT_REGISTRY[prompt_id]
    prompt = prompt_template.format(payload=json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    llm = client or OllamaClient(model=model)
    response: LlmResponse = llm.generate_completion(prompt, system=SYSTEM_POLICY_IT, temperature=temperature)
    return {
        "status": response.status,
        "prompt_id": prompt_id,
        "model": response.model,
        "content": response.content,
        "error": response.error,
        "payload": payload,
    }


def suggest_screener_config(
    user_request: str,
    *,
    available_context: dict[str, Any],
    client: OllamaClient | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Translate natural-language screener intent into a JSON config draft."""
    prompt = PROMPT_REGISTRY["screener_config_it"].format(
        user_request=user_request,
        available_context=json.dumps(available_context, ensure_ascii=False, indent=2, default=str),
    )
    llm = client or OllamaClient(model=model)
    response = llm.generate_completion(prompt, system=SYSTEM_POLICY_IT, temperature=0.1, response_format="json")
    return {
        "status": response.status,
        "prompt_id": "screener_config_it",
        "model": response.model,
        "content": response.content,
        "error": response.error,
        "payload": {"user_request": user_request, "available_context": available_context},
    }


def advise_model_configuration(
    *,
    universe: str,
    target_horizon_days: int,
    feature_blocks: list[str],
    available_models: list[str],
    active_models: list[str],
    metrics: pd.DataFrame | None = None,
    constraints: dict[str, Any] | None = None,
    client: OllamaClient | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Ask the LLM to propose a governed model stack for a use case."""
    payload = {
        "universe": universe,
        "target_horizon_days": int(target_horizon_days),
        "feature_blocks_available": feature_blocks,
        "available_models": available_models,
        "active_models": active_models,
        "constraints": constraints or {},
        "recent_metrics": _compact_records(metrics, limit=8),
    }
    return _call_prompt("model_advisor_it", payload, client=client, model=model)


def explain_stock_picks(
    picks: pd.DataFrame,
    *,
    client: OllamaClient | None = None,
    model: str | None = None,
    top_n: int = 8,
) -> dict[str, Any]:
    """Explain an already-computed ranking without changing the selection."""
    columns = [
        "ticker",
        "company_name",
        "sector",
        "country",
        "composite_conviction_score",
        "ml_score",
        "valuation_signal_score",
        "quality_proxy",
        "smart_money_score",
        "pe",
        "pb",
        "roe",
        "debt_to_equity",
        "volatility",
        "mispricing_rel",
        "ml_quintile",
        "screening_quintile",
    ]
    payload = {"top_n": int(top_n), "picks": _compact_records(picks, columns=columns, limit=top_n)}
    return _call_prompt("stock_pick_explainer_it", payload, client=client, model=model)


def advise_forecast_horizon(
    *,
    use_case: str,
    current_horizon_days: int,
    turnover_hint: str = "",
    cost_bps: float = 10.0,
    feature_blocks: list[str] | None = None,
    metrics: pd.DataFrame | None = None,
    client: OllamaClient | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Ask the LLM to reason about 21/63/252 day target horizons."""
    payload = {
        "use_case": use_case,
        "current_horizon_days": int(current_horizon_days),
        "candidate_horizons_days": [21, 63, 252],
        "turnover_hint": turnover_hint,
        "transaction_cost_bps": float(cost_bps),
        "feature_blocks_available": feature_blocks or [],
        "recent_metrics": _compact_records(metrics, limit=8),
    }
    return _call_prompt("forecast_horizon_advisor_it", payload, client=client, model=model)


def audit_model_governance(
    *,
    model_cards: pd.DataFrame | None = None,
    metrics: pd.DataFrame | None = None,
    feature_importance: pd.DataFrame | None = None,
    client: OllamaClient | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Generate a model-governance review from model artifacts."""
    payload = {
        "model_cards": _compact_records(model_cards, limit=12),
        "metrics": _compact_records(metrics, limit=12),
        "top_feature_importance": _compact_records(feature_importance, limit=20),
    }
    return _call_prompt("model_governance_audit_it", payload, client=client, model=model)
