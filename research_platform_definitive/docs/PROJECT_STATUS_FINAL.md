# Project Status Final

Aggiornato: 2026-05-23

Repository standalone: `https://github.com/TheGenesisAIStory/investment-research-platform-pro`

## Sintesi

Il progetto e' organizzato intorno a quattro superfici operative:

1. Research Platform canonica in `research_platform_definitive/`.
2. Data Center con schema SQLite, sample data e pipeline rigenerabile.
3. Laboratorio LLM/Vibe Trading con prompt italiani e provider registry.
4. Bridge QuantDinger/mobile con endpoint per strategie, segnali e backtest summary.

Il bundle resta notebook-first, ma ora ha uno strato dati condiviso che rende piu' facile testare app, LLM Lab e mobile senza aspettare provider esterni.

## Stato componenti

| Componente | Percorso | Stato | Come si usa |
|---|---|---|---|
| Research Platform | `research_platform_definitive/research_platform_app/app.py` | pronta | `streamlit run research_platform_definitive/research_platform_app/app.py` |
| Data/API Control | `research_platform_definitive/data_api_app.py` | pronta | `streamlit run research_platform_definitive/data_api_app.py` |
| Data Center DB | `src/research_platform_core/research_database.py` | pronto | `python3 research_platform_definitive/scripts/populate_research_database.py` |
| Sample datasets | `research_platform_definitive/data/sample/` | pronti | CSV leggibili e summary JSON |
| LLM Lab | `src/research_platform_core/llm_lab.py` | pronto | `python3 research_platform_definitive/scripts/llm_lab_cli.py packet` |
| Vibe bridge | `integrations/vibe_trading_bridge/` | operativo fallback | `python3 scripts/vibe_cli.py ... --no-vibe` |
| QuantDinger sidecar | `integrations/quantdinger_bridge/ml_service.py` | pronto | `uvicorn integrations.quantdinger_bridge.ml_service:app --port 8000` |
| Mobile Vue | `integrations/quantdinger_bridge/vue/mobile/` | iniziale reale | segnali, strategie, backtest on demand e storico |

## Cosa e' stato chiuso in questo ciclo

- Creato schema DB finale per strumenti, OHLCV, feature/label, segnali, backtest, esperimenti, catalogo dati e ingestion runs.
- Aggiunto popolamento sintetico realistico con piu' di 11k barre OHLCV, oltre 10k righe feature e circa 500 segnali.
- Salvati CSV sample tracciabili in `data/sample/`.
- Aggiunti prompt italiani per review strategia, vibe coding, diagnostica backtest, data quality e sintesi mobile.
- Esteso il sidecar FastAPI con endpoint dashboard-safe:
  - `/api/v1/ml/strategies`
  - `/api/v1/ml/signals/snapshot`
  - `/api/v1/ml/backtests/summary`
- Aggiornati componenti mobile per vedere strategie e storico backtest, con fallback ai segnali snapshot.
- Aggiunta documentazione su Data Center, formule, LLM Lab e handoff mobile.

## Comandi rapidi

```bash
python3 research_platform_definitive/scripts/populate_research_database.py
python3 research_platform_definitive/scripts/llm_lab_cli.py packet
python3 -m pytest tests/test_quantdinger_bridge research_platform_definitive/tests -q
python3 research_platform_definitive/scripts/validate_definitive_bundle.py
python3 research_platform_definitive/scripts/validate_notebooks.py
python3 research_platform_definitive/scripts/validate_artifacts.py
```

## Notebook finali

I notebook fonte restano questi:

- `company_valuation/notebooks/Company_Valuation_Final_Version.ipynb`
- `portfolio_analysis/notebooks/Portfolio-Analysis-Model_RESEARCH_PLATFORM_PRO.ipynb`
- `machine_learning_lab/notebooks/ML_Stock_Lab_Experiments.ipynb`
- `data_api_management/notebooks/Data_API_Management_Colab.ipynb`

La mappa friendly e' in `notebooks/README.md`. I notebook devono importare logica dai package `src/` e non duplicare calcoli core.

## Nice to have non bloccanti

- Sostituire tutti i warning `pd.Timestamp.utcnow()` con `pd.Timestamp.now(tz="UTC")`.
- Collegare provider reali al nuovo `ResearchDatabase` con job incrementali.
- Aggiungere grafici mobile nativi al posto della mini equity curve HTML/CSS.
- Creare una pagina Streamlit dedicata al DB finale con query pronte.
- Eseguire un backtest walk-forward su dati reali prima di promuovere qualunque strategia.

## Nota prudente

Il sample e' utile per sviluppo e demo, non per decisioni finanziarie. La pipeline e' pronta per dati reali, ma le strategie devono essere validate con dati point-in-time, costi realistici e review umana.
