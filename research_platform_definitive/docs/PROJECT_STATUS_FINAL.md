# Project Status Final - Gen.is.IA Investment Research Workstation

Data: 2026-05-26

## Stato sintetico

La piattaforma e' ora in stato `v1-ready` per uso locale/desk research:

- core Python pacchettizzato e importabile;
- Database Finanziario condiviso fra core, app, ML Lab, notebook e script;
- Data Platform con health, restart, failures, run monitor, ticker explorer e
  data explorer;
- Home Command Center, Screener, ML Stock Lab, Valuation, Portfolio, Macro View
  e Smart Money integrati da `selected_ticker`;
- branding Gen.is.IA uniforme e glossari inline per fattori, feature e metriche;
- OHLCV locali su root stabile, non su scrittura batch diretta Drive;
- test, validator e smoke UI verdi.

## Dati

- `equity_fundamentals`: `OK`.
- `equity_prices`: `OK` in validation non-strict.
- OHLCV manifest:
  - `OK`: 10.074 strumenti.
  - `LIMITED_HISTORY`: 3.432 strumenti.
  - `NO_PRICE_DATA`: 126 strumenti.
  - `DELISTED`: 19 strumenti.
  - parquet locali indicizzati: 12.537.
- Factor panel:
  - righe: 2.173.286.
  - ticker nel panel validato: 1.000.
  - data range: 2000-01-03 -> 2026-05-22.
- Macro View:
  - multi-asset universe esteso con 140 proxy fra global, USA, EU, Italy,
    crypto, FX, commodities, ETF e fixed income dove disponibili.

## ML / contenuto quantitativo

- Modelli training presenti: OLS, RF, GBRT, ensemble.
- Target standard: `forward_return` a 21 trading days; supporto 21/63/252 giorni
  nel training pipeline.
- Forecasting time-series: nuovo `Time Series Lab` con forecast single-series
  su Macro DB/OHLCV equity, orizzonti 5/21/63/126 giorni, baseline `naive` e
  modelli feature-based `ols`/`gbrt`, con intervalli derivati dai residui
  out-of-sample.
- Statistica base condivisa: volatilita', varianza, beta benchmark,
  correlazioni benchmark e matrici di correlazione per Screener/Portfolio.
- Factor vocabulary: value, quality, momentum, risk, size, growth,
  model-based.
- Regime detection: `OK`; quattro stati (`risk_on`, `risk_off`, `crisis`,
  `recovery`) derivati da equity momentum, credit spread proxy, VIX, yield
  slope e commodity momentum.
- Macro context features: `OK / experimental`; blocco opzionale
  `macro_context` con lag minimo di una osservazione e as-of join.
- Alpha101: `OK / experimental`; 101 formule WorldQuant/Kakushadze registrate
  e selezionabili in ML Stock Lab. Il retraining locale Alpha101 e' completato
  con 4/4 modelli `OK` su split 2000-2018 / 2019-2026 e artifact `_alpha101`
  sotto `output/ml_training_lab`.
- Leakage policy attiva tramite `feature_columns_for_blocks()`.
- Metriche: `r2_os`, `ic`, `rank_ic`, `sharpe_long_short`,
  `sharpe_long_short_net_cost`, data/model coverage metrics.
- Glossari:
  - `research_platform_core.feature_metadata`;
  - `research_platform_core.metrics_metadata`.

## UI / app

- Branding globale: `Gen.is.IA Investment Research Workstation`.
- Macro menu:
  - RESEARCH: Home, Screener, Valuation, Portfolio, Macro View, Smart Money.
  - LABS: ML Stock Lab, Time Series Lab, Banking Data Lab, Research Library.
  - PLATFORM OPS: Data Platform, Notebook Runner, Export Center, Hi-Freq Engine,
    Data/API Control Center.
- Ogni pagina principale usa context bar e header Gen.is.IA.
- Screener, ML Lab, Valuation e Portfolio condividono
  `st.session_state.selected_ticker`.
- Data Platform e Home sono entrypoint data-centric, non solo job-centric.
- Macro View e Portfolio leggono gli artifact Time Series Lab come scenario
  context, senza alterare ranking o pesi di portafoglio.

## Quality gate eseguiti

```bash
.venv/bin/python -m compileall -q research_platform_definitive/src/research_platform_core research_platform_definitive/research_platform_app
.venv/bin/python -m pytest research_platform_definitive/tests -q
.venv/bin/python research_platform_definitive/scripts/validate_definitive_bundle.py
.venv/bin/python research_platform_definitive/scripts/validate_artifacts.py
.venv/bin/python research_platform_definitive/scripts/validate_research_data_coverage.py
.venv/bin/python research_platform_definitive/scripts/validate_notebooks.py
```

Risultato corrente:

- locale: 73 test passati;
- worktree GitHub separato: 133 test passati;
- validator: OK;
- AppTest Home, Macro View, ML Lab e Portfolio: `exceptions 0`.

## Limiti dichiarati

- Il training ML full su tutto l'universo va ancora lanciato per passare da
  smoke/bounded validation a performance economica definitiva; il run Alpha101
  corrente e' bounded/local e usa fallback proxy per il factor panel privo di
  OHLCV completo.
- Banking fundamentals ufficiali e issuer/fund flows restano `PLANNED`/
  `PARTIAL`; Smart Money pubblico e macro context sono presenti come layer di
  contesto.
- Il validator coverage puo' indicare `MISSING` per i CSV pesanti
  `output/data_completion/*` nei worktree temporanei quando gli artifact locali
  non sono sincronizzati.
- Drive resta archivio/sync; i batch parquet devono scrivere prima su filesystem
  locale stabile.

## Stato release

Stato: `v1-ready / prod-like local workstation`.

Prossimo tag consigliato dopo eventuale review manuale UI:

```bash
git tag v1.0.0
git push origin v1.0.0
```
