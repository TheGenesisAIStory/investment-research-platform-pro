"""Offline-first LLM Lab utilities for vibe coding and vibe trading.

The lab renders Italian prompt packets and provider configuration without
requiring API calls. Real LLM calls can be layered on top by reading the
provider registry and sending the rendered prompt to the chosen SDK.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
from string import Template
from typing import Any

import pandas as pd

from .research_database import ResearchDatabase, default_research_database_path, utc_now_iso


@dataclass(frozen=True, slots=True)
class ProviderConfig:
    """LLM provider configuration discovered from environment variables."""

    provider: str
    model: str
    api_key_env: str
    configured: bool
    base_url: str | None = None
    notes: str = ""


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    """Named prompt template with required context fields."""

    name: str
    description_it: str
    template: str
    required_fields: tuple[str, ...]

    def render(self, context: dict[str, Any]) -> str:
        missing = [field for field in self.required_fields if field not in context or context[field] in {None, ""}]
        if missing:
            raise ValueError(f"Missing prompt context fields for {self.name}: {', '.join(missing)}")
        safe_context = {key: _stringify(value) for key, value in context.items()}
        return Template(self.template).safe_substitute(safe_context).strip() + "\n"


PROMPT_TEMPLATES: dict[str, PromptTemplate] = {
    "strategy_review_it": PromptTemplate(
        name="strategy_review_it",
        description_it="Review di una strategia con formule, ipotesi, rischi e prossimi esperimenti.",
        required_fields=("strategy_name", "universe", "signals", "backtests"),
        template="""
Sei il copilota di ricerca della Investment Research Platform Pro.

Strategia da rivedere: $strategy_name
Universo: $universe

Segnali disponibili:
$signals

Backtest disponibili:
$backtests

Scrivi una nota in italiano semplice con:
1. tesi della strategia;
2. formula del segnale in pseudocodice;
3. cosa dicono metriche, drawdown e turnover;
4. rischi di data leakage, overfitting e regime;
5. prossimi 3 esperimenti da lanciare.

Non proporre live trading. Se mancano dati, dichiaralo in modo chiaro e gentile.
""",
    ),
    "vibe_coding_strategy_it": PromptTemplate(
        name="vibe_coding_strategy_it",
        description_it="Prompt per generare una StrategyBase compatibile con Vibe-Trading.",
        required_fields=("idea", "universe", "start_date", "end_date"),
        template="""
Genera una strategia Python compatibile con:

from integrations.vibe_trading_bridge.strategy_adapter import StrategyBase, StrategyContext

Idea utente:
$idea

Universo: $universe
Finestra backtest: $start_date -> $end_date

Requisiti:
- codice semplice e leggibile;
- nessun path locale hard-coded;
- usa solo dati disponibili fino a context.date;
- spiega formula, parametri, filtri, ribilanciamento e risk management nella docstring;
- aggiungi uno snippet di test con run_backtest;
- output finale: codice, assunzioni, limiti e metriche attese.
""",
    ),
    "backtest_diagnostic_it": PromptTemplate(
        name="backtest_diagnostic_it",
        description_it="Diagnostica di un backtest gia' eseguito.",
        required_fields=("strategy_name", "metrics", "equity_curve_note"),
        template="""
Analizza questo backtest della Research Platform.

Strategia: $strategy_name
Metriche:
$metrics

Nota equity curve:
$equity_curve_note

Produci:
1. lettura delle metriche in linguaggio umano;
2. segnali di robustezza o fragilita';
3. controlli mancanti;
4. esperimenti prioritari;
5. messaggio breve da mostrare in app mobile.
""",
    ),
    "data_quality_brief_it": PromptTemplate(
        name="data_quality_brief_it",
        description_it="Brief Data Center per copertura, freschezza e query utili.",
        required_fields=("catalog", "table_counts"),
        template="""
Sei il responsabile Data Center.

Catalogo dati:
$catalog

Conteggi tabelle:
$table_counts

Scrivi un brief operativo in italiano:
- cosa e' pronto;
- quali dati sono sintetici/sample;
- quali tabelle alimentano Research Platform, LLM Lab e mobile;
- come aggiungere una nuova fonte dati;
- quali controlli fare prima di fidarsi dei risultati.
""",
    ),
    "quantdinger_mobile_summary_it": PromptTemplate(
        name="quantdinger_mobile_summary_it",
        description_it="Sintesi friendly per dashboard mobile QuantDinger-style.",
        required_fields=("signals", "backtests"),
        template="""
Prepara una sintesi mobile breve e chiara.

Segnali:
$signals

Backtest:
$backtests

Output richiesto:
- 3 bullet massimo;
- una frase di rischio;
- una CTA prudente, ad esempio 'apri il laboratorio' o 'rivedi il backtest';
- niente promessa di rendimento.
""",
    ),
}


def provider_registry() -> list[ProviderConfig]:
    """Discover local provider readiness from environment variables."""

    return [
        ProviderConfig("openai", os.environ.get("OPENAI_MODEL", "gpt-4.1"), "OPENAI_API_KEY", bool(os.environ.get("OPENAI_API_KEY")), notes="Provider cloud per coding/reasoning."),
        ProviderConfig("anthropic", os.environ.get("ANTHROPIC_MODEL", "claude-3-7-sonnet-latest"), "ANTHROPIC_API_KEY", bool(os.environ.get("ANTHROPIC_API_KEY")), notes="Provider alternativo per review lunghe."),
        ProviderConfig("ollama", os.environ.get("OLLAMA_MODEL", "llama3.1"), "OLLAMA_BASE_URL", bool(os.environ.get("OLLAMA_BASE_URL")), base_url=os.environ.get("OLLAMA_BASE_URL"), notes="Provider locale quando disponibile."),
        ProviderConfig("vibe_trading", "local_bridge", "VIBE_TRADING_ROOT", bool(os.environ.get("VIBE_TRADING_ROOT") or Path("vibe_trading").exists()), notes="Bridge locale per generazione strategia e backtest."),
    ]


def provider_registry_frame() -> pd.DataFrame:
    """Return provider readiness as a DataFrame for Streamlit/docs."""

    return pd.DataFrame([asdict(provider) for provider in provider_registry()])


def available_templates() -> pd.DataFrame:
    """Return prompt-template metadata."""

    return pd.DataFrame(
        [
            {
                "name": template.name,
                "description_it": template.description_it,
                "required_fields": ", ".join(template.required_fields),
            }
            for template in PROMPT_TEMPLATES.values()
        ]
    )


def render_prompt(template_name: str, context: dict[str, Any]) -> str:
    """Render one named prompt template."""

    try:
        template = PROMPT_TEMPLATES[template_name]
    except KeyError as exc:
        raise ValueError(f"Unknown prompt template: {template_name}") from exc
    return template.render(context)


def build_context_from_database(db_path: Path | str | None = None, strategy_name: str = "risk_balanced_blend") -> dict[str, Any]:
    """Build a compact prompt context from the canonical research DB."""

    db = ResearchDatabase(db_path or default_research_database_path())
    counts = db.table_counts()
    strategies = db.list_strategies()
    signals = db.latest_signals(limit=8)
    backtests = db.latest_backtests(limit=5)
    catalog = db.read_table("data_catalog_entries")
    universe = ""
    if not backtests.empty:
        universe = str(backtests["universe"].iloc[0])
    return {
        "strategy_name": strategy_name,
        "universe": universe or "sample universe",
        "signals": _frame_to_markdown(signals),
        "backtests": _frame_to_markdown(backtests),
        "metrics": _frame_to_markdown(backtests[["strategy_name", "total_return", "sharpe", "max_drawdown", "avg_turnover"]] if not backtests.empty else backtests),
        "equity_curve_note": "Equity curve salvata in backtest_equity; aprire dashboard o query SQLite per il dettaglio giornaliero.",
        "catalog": _frame_to_markdown(catalog),
        "table_counts": json.dumps(counts, indent=2, sort_keys=True),
        "idea": "Migliora il blend momentum/z-score con controllo volatilita' e costi realistici.",
        "start_date": str(backtests["start_date"].iloc[0]) if not backtests.empty else "2021-01-04",
        "end_date": str(backtests["end_date"].iloc[0]) if not backtests.empty else "2026-05-22",
        "strategies": _frame_to_markdown(strategies),
    }


def save_prompt_packet(
    template_names: list[str] | None = None,
    output_dir: Path | str | None = None,
    db_path: Path | str | None = None,
) -> dict[str, str]:
    """Render prompt files to disk and return their paths."""

    names = template_names or list(PROMPT_TEMPLATES)
    out = Path(output_dir).expanduser() if output_dir else default_research_database_path().parents[1] / "llm_lab"
    out.mkdir(parents=True, exist_ok=True)
    context = build_context_from_database(db_path)
    paths: dict[str, str] = {}
    for name in names:
        prompt = render_prompt(name, context)
        path = out / f"{name}.md"
        header = f"# {name}\n\nGenerated: {utc_now_iso()}\n\n"
        path.write_text(header + prompt, encoding="utf-8")
        paths[name] = str(path)
    manifest = {
        "generated_at": utc_now_iso(),
        "templates": names,
        "paths": paths,
        "providers": [asdict(provider) for provider in provider_registry()],
    }
    manifest_path = out / "llm_lab_prompt_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    paths["manifest"] = str(manifest_path)
    return paths


def _frame_to_markdown(frame: pd.DataFrame, max_rows: int = 10) -> str:
    if frame is None or frame.empty:
        return "_Nessun dato disponibile._"
    view = frame.head(max_rows).copy()
    try:
        return view.to_markdown(index=False)
    except Exception:
        return view.to_csv(index=False)


def _stringify(value: Any) -> str:
    if isinstance(value, pd.DataFrame):
        return _frame_to_markdown(value)
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, indent=2, sort_keys=True, default=str)
    text = str(value)
    return re.sub(r"\n{3,}", "\n\n", text)
