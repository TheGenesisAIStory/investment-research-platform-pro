# Project Status Final - Gen.is.IA Investment Research Workstation

Data: 2026-05-25

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
  - 40 asset/proxy compilati fra global, USA, EU, Italy, crypto, FX,
    commodities, ETF e fixed income.

## ML / contenuto quantitativo

- Modelli training presenti: OLS, RF, GBRT, ensemble.
- Target standard: `forward_return` a 21 trading days; supporto 21/63/252 giorni
  nel training pipeline.
- Factor vocabulary: value, quality, momentum, risk, size, growth,
  model-based.
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
  - LABS: ML Stock Lab, Banking Data Lab, Research Library.
  - PLATFORM OPS: Data Platform, Notebook Runner, Export Center, Hi-Freq Engine,
    Data/API Control Center.
- Ogni pagina principale usa context bar e header Gen.is.IA.
- Screener, ML Lab, Valuation e Portfolio condividono
  `st.session_state.selected_ticker`.
- Data Platform e Home sono entrypoint data-centric, non solo job-centric.

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

- locale: 69 test passati;
- worktree GitHub separato: 73 test passati;
- validator: OK;
- AppTest principali: `exceptions 0`.

## Limiti dichiarati

- Il training ML full su tutto l'universo va ancora lanciato per passare da
  smoke/bounded validation a performance economica definitiva.
- `macro_fx`, `factor_libraries`, `smart_money`, `banking` sono ancora `PLANNED`
  nel validator completion 2000-2026, anche se esistono artifact/app parziali o
  dedicati.
- Drive resta archivio/sync; i batch parquet devono scrivere prima su filesystem
  locale stabile.

## Stato release

Stato: `v1-ready / prod-like local workstation`.

Prossimo tag consigliato dopo eventuale review manuale UI:

```bash
git tag v1.0.0
git push origin v1.0.0
```

