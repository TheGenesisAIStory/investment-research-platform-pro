from __future__ import annotations

from research_platform_core.llm_client import OllamaClient
from research_platform_core.llm_prompts import PROMPT_REGISTRY
from research_platform_core.llm_advisors import advise_forecast_horizon, advise_model_configuration, audit_model_governance, explain_stock_picks, suggest_screener_config
import pandas as pd


class FakeOllamaClient(OllamaClient):
    def _post(self, endpoint, payload):
        if endpoint == "/api/chat":
            return {"message": {"content": "chat ok"}, "model": payload["model"], "done": True}
        return {"response": "generate ok", "model": payload["model"], "done": True}


def test_ollama_client_generate_and_chat_parse_responses() -> None:
    client = FakeOllamaClient(base_url="http://example.test", model="test-model", retries=0)

    generated = client.generate_completion("hello")
    chatted = client.chat([{"role": "user", "content": "hello"}])

    assert generated.status == "OK"
    assert generated.content == "generate ok"
    assert chatted.status == "OK"
    assert chatted.content == "chat ok"


def test_prompt_registry_contains_project_prompts() -> None:
    assert "screener_config_it" in PROMPT_REGISTRY
    assert "ml_signal_explainer_it" in PROMPT_REGISTRY
    assert "feature_audit_it" in PROMPT_REGISTRY
    assert "model_advisor_it" in PROMPT_REGISTRY
    assert "stock_pick_explainer_it" in PROMPT_REGISTRY
    assert "forecast_horizon_advisor_it" in PROMPT_REGISTRY


def test_llm_advisors_build_payloads_without_network() -> None:
    client = FakeOllamaClient(base_url="http://example.test", model="test-model", retries=0)
    metrics = pd.DataFrame([{"model": "rf", "rank_ic": 0.04, "r2_os": 0.01}])
    picks = pd.DataFrame([{"ticker": "AAA", "ml_score": 88, "valuation_signal_score": 70}])

    model_advice = advise_model_configuration(
        universe="US large cap",
        target_horizon_days=21,
        feature_blocks=["value", "momentum"],
        available_models=["ols", "rf"],
        active_models=["ols"],
        metrics=metrics,
        client=client,
    )
    pick_explanation = explain_stock_picks(picks, client=client)
    horizon_advice = advise_forecast_horizon(
        use_case="medium-term stock picking",
        current_horizon_days=21,
        feature_blocks=["value", "quality"],
        metrics=metrics,
        client=client,
    )
    audit = audit_model_governance(model_cards=metrics, metrics=metrics, client=client)
    screener = suggest_screener_config("quality US large cap", available_context={"universes": ["sp500"]}, client=client)

    assert model_advice["status"] == "OK"
    assert model_advice["prompt_id"] == "model_advisor_it"
    assert pick_explanation["payload"]["picks"][0]["ticker"] == "AAA"
    assert horizon_advice["payload"]["candidate_horizons_days"] == [21, 63, 252]
    assert audit["prompt_id"] == "model_governance_audit_it"
    assert screener["prompt_id"] == "screener_config_it"
