# Project Review Totale - 2026-05-25

## Stato Backfill

- Nessun processo `bootstrap_research_data_2000_2026.py` o training ML risulta
  attivo dopo la rigenerazione del factor panel e il retraining bounded.
- Nessun lock stage e' presente in `output/locks` dopo i run.
- `validate_research_data_coverage.py` passa in modalita' non-strict.
- `equity_fundamentals`: `OK`.
- `equity_prices`: `PARTIAL` in Data Health per presenza di titoli
  `LIMITED_HISTORY`, `DELISTED` e `NO_PRICE_DATA`, ma senza failure critiche.
- Manifest OHLCV corrente:
  - `OK`: 10.074 strumenti.
  - `LIMITED_HISTORY`: 3.432 strumenti.
  - `NO_PRICE_DATA`: 126 strumenti.
  - `DELISTED`: 19 strumenti.
  - parquet locali indicizzati: 12.098.

Interpretazione: il backfill prezzi e' utilizzabile per ricerca e Screener in
modalita' non-strict. I nomi con storia corta devono restare disponibili, ma
marcati come `LIMITED_HISTORY`.

## Factor Panel e Training Eseguiti

- `factor_universe_panel` e' stato rigenerato da parquet OHLCV locali:
  - righe: 2.173.286.
  - ticker: 1.000 nel run bounded validato.
  - output locale: `output/ml_training_lab/tables/FactorUniversePanel.csv`.
  - mirror Drive: `Database Finanziario/Factors/EquityFactorPanel.csv`.
- Il primo smoke training ha evidenziato un problema metodologico: `--max-rows`
  prendeva `head(n)` e schiacciava la finestra temporale al 2000-2001.
- Il loader e' stato corretto con campionamento chunked lungo tutto il file per
  mantenere il test 2019-2026 anche nei run bounded.
- Retraining bounded eseguito con OLS, RF, GBRT ed ensemble:
  - panel training: 27.500 righe, 908 ticker, date 2000-01-03 -> 2026-05-22.
  - split: train fino a 2018-12-31, test da 2019-01-02.
  - artifact: model cards, metrics, predictions long/wide, feature importance,
    segnali consumabili da ML Stock Lab/Screener.

## Correzione Trovata Durante Review

`get_data_health_summary()` falliva su manifest reali quando il breakdown
coverage conteneva chiavi miste stringa/float e veniva serializzato con
`sort_keys=True`. La correzione normalizza sempre le chiavi prima del JSON.

## Contenuto Quantitativo Implementato

- Aggiunto vocabolario fattoriale canonico: value, quality, momentum, risk,
  size, growth, model-based.
- Aggiunto momentum 12-1 e target forward return a 21/63/252 giorni.
- Aggiunta policy anti-leakage per la selezione feature.
- Aggiunte metriche IC e RankIC per il monitoraggio ranking.
- Training ML aggiornato con predictions wide multi-model e model cards.
- Screener allineato semanticamente:
  - `ml_score` = expected-return/composite model.
  - `valuation_signal_score` = valuation/mispricing percentile.
  - `fair_value_hat` non e' piu' trattato come ML score.

## Ollama / LLM Governance

Ollama e' integrato come assistente di governance e spiegazione, non come
decisore di trade:

- Model Advisor in ML Stock Lab.
- Forecast Horizon Advisor in ML Stock Lab.
- Model Governance Audit in ML Stock Lab.
- Screener Assistant in Screener Builder.
- Stock Pick Explainer in Screener Builder.

Tutti i prompt sono versionati in `research_platform_core.llm_prompts`.

## Problemi Residui

- Il factor panel completo va esteso oltre il run bounded da 1.000 ticker quando
  si vuole una release dati finale su tutto l'universo.
- Le metriche ML correnti sono validate su run bounded e non vanno ancora
  interpretate come performance economica definitiva.
- `macro_fx`, `factor_libraries`, `smart_money`, `banking` sono ancora
  `PARTIAL/PLANNED` nel summary dati e vanno promossi gradualmente a run
  operativi.
- La sync Drive va mantenuta come pubblicazione/archivio, non come target per
  scritture batch lunghe.

## Quality Gate Eseguiti

- Compileall su `src`, app e scripts.
- Test mirati LLM/Data Health/Screener.
- AppTest su Screener, ML Stock Lab e Data Platform.
- `factor_universe_panel` bounded: completato.
- `train_ml_models_2000_2026.py` bounded: completato, `status=OK`.

Il prossimo gate prima di una release taggata e':

```bash
.venv/bin/pytest research_platform_definitive/tests -q
.venv/bin/python research_platform_definitive/scripts/validate_definitive_bundle.py
.venv/bin/python research_platform_definitive/scripts/validate_notebooks.py
.venv/bin/python research_platform_definitive/scripts/validate_artifacts.py
.venv/bin/python research_platform_definitive/scripts/validate_research_data_coverage.py
```
