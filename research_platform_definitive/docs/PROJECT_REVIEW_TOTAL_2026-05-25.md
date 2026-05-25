# Project Review Totale - 2026-05-25

## Stato Backfill

- Nessun processo `bootstrap_research_data_2000_2026.py` o training ML risulta
  attivo dopo la rigenerazione del factor panel e il retraining bounded.
- Nessun lock stage e' presente in `output/locks` dopo i run.
- `validate_research_data_coverage.py` passa in modalita' non-strict.
- `equity_fundamentals`: `OK`.
- `equity_prices`: `OK` in coverage validation non-strict, con categorie
  esplicite per titoli a storia corta o non scaricabili.
- Manifest OHLCV corrente:
  - `OK`: 10.074 strumenti.
  - `LIMITED_HISTORY`: 3.432 strumenti.
  - `NO_PRICE_DATA`: 126 strumenti.
  - `DELISTED`: 19 strumenti.
  - parquet locali indicizzati: 12.537.

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
- Aggiunti glossari core in `research_platform_core.feature_metadata` e
  `research_platform_core.metrics_metadata`, usati dalla UI Gen.is.IA:
  - copertura completa delle colonne dichiarate in `RAW_FACTOR_COLUMNS`;
  - copertura delle colonne reali principali viste in ML/Screener/Portfolio;
  - copertura delle metriche modello/portfolio/data-quality principali.
- Aggiunto `research_platform_core.time_series_forecasting` per coprire il
  gap di forecasting single-series:
  - input da Macro DB o OHLCV equity;
  - feature lag/rolling senza leakage;
  - modelli `naive`, `ols`, `gbrt`;
  - metriche `mae`, `rmse`, `mape`, `directional_accuracy`;
  - artifact in `output/time_series_lab/tables`.

## App / UI Gen.is.IA

- Branding globale: `Gen.is.IA Investment Research Workstation`.
- Header condiviso `render_page_header(...)` su Home, Data Platform, Macro,
  Smart Money, Screener, ML Stock Lab, Valuation, Portfolio, Banking, Library e
  Tools.
- Home trasformata in Command Center con quick actions, ticker search, workflow
  cards e platform status.
- Data Platform estesa con Single Ticker Explorer, Data Explorer, Domain Status,
  Data Health & Restart, Failures Panel e Run Monitor.
- Macro View separata da Smart Money, con 40 asset/proxy tra global, USA, EU,
  Italy, crypto, FX, commodity, ETF e fixed income.
- Time Series Lab aggiunto sotto LABS come modulo di scenario per indici,
  macro proxy, FX, commodity, crypto e singole equity.
- Macro View mostra la disponibilita' dei forecast Time Series Lab e apre il
  Lab con la serie preselezionata.
- Portfolio include `Macro & TS Context` con forecast di scenario e matrice di
  correlazione holdings.
- Screener include `Risk & Correlation` per correlazioni rapide sulla selezione.
- Screener, ML Lab, Valuation e Portfolio condividono `selected_ticker` e
  mostrano context panel e glossari inline.

## Ollama / LLM Governance

Ollama e' integrato come assistente di governance e spiegazione, non come
decisore di trade:

- Model Advisor in ML Stock Lab.
- Forecast Horizon Advisor in ML Stock Lab.
- Model Governance Audit in ML Stock Lab.
- Screener Assistant in Screener Builder.
- Stock Pick Explainer in Screener Builder.

Tutti i prompt sono versionati in `research_platform_core.llm_prompts`.

## Multi-Asset, Smart Money e Baseline Fattoriali

- Macro DB e' stato esteso a 92 proxy: FX, commodities, crypto, ETF equity,
  ETF fixed income, ETF commodity, ETF crypto, fixed-income proxies e indici.
- `MultiAssetUniverseManifest.csv` rende visibile per ogni dominio lo stato
  `OK/PARTIAL/PLANNED`, numero strumenti e ultimo aggiornamento; Data Platform
  e Macro View lo consumano senza scansioni Drive pesanti.
- `SmartMoneySourceManifest.csv` esplicita CFTC COT, ETF flows, options
  positioning e issuer-event evidence come fonti `OK`, `READY_OPTIONAL`,
  `PARTIAL` o `PLANNED`; nessun pannello Smart Money resta vuoto senza stato.
- `research_platform_core.factor_benchmarks` aggiunge benchmark pure-factor per
  value, quality, momentum, risk, size, growth e composite factor.
- ML Stock Lab ha una tab `Factor Baselines` per confrontare i modelli ML con
  top-bucket e top-minus-bottom factor portfolios.

## Problemi Residui

- Il factor panel corrente e' ampio e coerente, ma il run ML validato resta
  bounded su 1.000 ticker. Per una release dati finale serve un training full
  su tutto l'universo disponibile.
- Le metriche ML correnti servono come smoke/validation quantitativa, non come
  performance economica definitiva.
- Time Series Lab v1 non include ancora ARIMA/ETS, intervalli di previsione o
  modelli multivariati macro; e' una base feature-based pronta per estensione.
- `factor_libraries` e `banking` restano domini da completare a livello di
  storico full 2000-2026. Macro/Multi-Asset e Smart Money hanno ora manifest
  applicativi espliciti, ma alcune fonti restano `PLANNED` finche' non vengono
  abilitate o licenziate.
- La sync Drive va mantenuta come pubblicazione/archivio, non come target per
  scritture batch lunghe.

## Quality Gate Eseguiti

- Compileall su `src`, app e scripts.
- Test suite completa: 73 test passati nel repo locale; 77 test passati nel
  worktree GitHub separato.
- Validator: definitive bundle, artifacts, research data coverage e notebooks OK.
- AppTest: Home e Time Series Lab con `exceptions 0` dopo il nuovo modulo;
  nel ciclo precedente anche Screener, Data Platform, ML Stock Lab, Valuation e
  Portfolio erano verdi.
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
