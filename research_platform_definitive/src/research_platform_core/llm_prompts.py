"""Versioned prompt templates for LLM-assisted investment research."""

from __future__ import annotations


SYSTEM_POLICY_IT = """Sei un assistente di ricerca buy-side.
Usa solo i dati forniti nel payload. Distingui sempre tra dati calcolati,
inferenze e ipotesi. Non dare raccomandazioni operative di acquisto/vendita.
Quando l'evidenza e' debole, dichiaralo esplicitamente e proponi controlli."""


SCREENER_CONFIG_IT = """Trasforma la richiesta dell'utente in una configurazione Screener JSON.
Campi ammessi: universes, sectors, countries, max_pe, max_pb, min_roe,
max_debt_to_equity, min_ml_score, min_smart_money_score, ml_quintiles,
require_smart_events, sort_by. Rispondi solo con JSON valido.

Richiesta utente:
{user_request}

Contesto disponibile:
{available_context}
"""


ML_SIGNAL_EXPLAINER_IT = """Spiega il segnale ML per il ticker {ticker}.
Output richiesto:
1. Upshot in 1-2 frasi.
2. Driver principali con valore, confronto universo e direzione.
3. Posizionamento per value, quality, momentum, risk e Smart Money.
4. Limiti del segnale e controlli consigliati.

Dati:
{payload}
"""


FEATURE_AUDIT_IT = """Fai una review metodologica del feature set seguente.
Controlla leakage, mismatch di horizon, ridondanza, interpretabilita' e
allineamento con value, quality, momentum, size e low-vol.

Feature set e metadata:
{payload}
"""


DATA_HEALTH_MEMO_IT = """Riassumi lo stato dati per un analista non tecnico.
Evidenzia coverage, failure critici, limited history e azioni consigliate.
Non mostrare stacktrace; cita run_id e stage quando presenti.

Data Health payload:
{payload}
"""


MODEL_ADVISOR_PROMPT_IT = """Suggerisci una configurazione di training ML per stock picking.
Devi ragionare come model governance advisor: proponi modelli, feature blocks,
horizon e controlli, ma non generare raccomandazioni operative di trading.

Output richiesto:
1. Configurazione consigliata in JSON con chiavi: models, feature_blocks,
   target_horizon_days, rationale, governance_checks.
2. Trade-off interpretabilita' vs complessita'.
3. Rischi di overfitting/leakage e controlli minimi.

Contesto:
{payload}
"""


STOCK_PICK_EXPLAINER_PROMPT_IT = """Spiega una lista di stock picks gia' selezionata dai modelli quantitativi.
Non modificare il ranking e non inventare dati. Usa solo il payload.

Output richiesto:
1. Sintesi generale della lista.
2. Per ogni ticker: perche' e' entrato, driver factor/ML/valuation,
   rischi principali e dati mancanti.
3. Note di cautela: research-only, nessuna raccomandazione operativa.

Payload:
{payload}
"""


FORECAST_HORIZON_ADVISOR_PROMPT_IT = """Valuta quale horizon di forecasting usare per questo use case.
Confronta 21, 63 e 252 trading days in termini di turnover, rumore,
stabilita' del segnale, costi e tipo di strategia.

Output richiesto:
1. Horizon primario consigliato.
2. Horizon secondario se utile.
3. Modelli e feature blocks coerenti con ciascun horizon.
4. Governance checks prima di usare il modello.

Payload:
{payload}
"""


MODEL_GOVERNANCE_AUDIT_PROMPT_IT = """Fai una peer review sintetica del model stack.
Controlla model cards, feature blocks, target horizon, metriche, IC/RankIC,
baseline mancanti, leakage potenziale, stabilita' e interpretabilita'.

Output richiesto:
1. Verdict: OK / Watch / Blocker.
2. Finding prioritizzati.
3. Azioni consigliate prima del prossimo retraining.

Payload:
{payload}
"""


PROMPT_REGISTRY = {
    "system_policy_it": SYSTEM_POLICY_IT,
    "screener_config_it": SCREENER_CONFIG_IT,
    "ml_signal_explainer_it": ML_SIGNAL_EXPLAINER_IT,
    "feature_audit_it": FEATURE_AUDIT_IT,
    "data_health_memo_it": DATA_HEALTH_MEMO_IT,
    "model_advisor_it": MODEL_ADVISOR_PROMPT_IT,
    "stock_pick_explainer_it": STOCK_PICK_EXPLAINER_PROMPT_IT,
    "forecast_horizon_advisor_it": FORECAST_HORIZON_ADVISOR_PROMPT_IT,
    "model_governance_audit_it": MODEL_GOVERNANCE_AUDIT_PROMPT_IT,
}
