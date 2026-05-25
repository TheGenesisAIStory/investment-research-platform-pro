# Data Bootstrap

Questo documento descrive il bootstrap minimo del Database Finanziario usato dalla Streamlit workstation.

## Scopo

La piattaforma deve evitare schermate vuote quando un notebook non è ancora stato eseguito. Il bootstrap leggero crea la struttura attesa, copia eventuali seed locali già presenti e scrive un manifest di audit. Non scarica automaticamente grandi dataset senza una scelta esplicita dell'utente.

## Artifact minimi

| Dataset | Path atteso | Uso |
| --- | --- | --- |
| Equity metadata | `Database Finanziario/data/us_equities_meta_data.csv` | Identità issuer e universo Screener |
| Company screener | `company_valuation/output/tables/ScreenerResults.csv` | Valuation e ranking base |
| ML signals | `output/ml_stock_lab/tables/MLStockLab_signals.csv` | Reasoning ML, quintili e score |
| Smart Money scores | `output/smart_money/tables/SmartMoney_smart_money_scores.csv` | Overlay official-source |
| Banking universe | `output/banks_pipeline/banks_universe.csv` | Banking Data Lab |

## Bootstrap dalla UI

Le pagine Screener, Valuation, Portfolio, Smart Money, ML Stock Lab e Banking Data Lab mostrano un banner giallo quando mancano artifact critici. Il pulsante `Run light bootstrap`:

- crea le directory `data/`, `data/prices/`, `data/fundamentals/`, `data/mappings/`, `data/catalog/` e `logs/`;
- copia `archive/local_databases_not_on_drive/data/us_equities_meta_data.csv` se disponibile;
- scrive `output/bootstrap/data_bootstrap_manifest.json`.

## Refresh CLI consigliato

```bash
python scripts/initial_setup.py --execute --max-items 5
python scripts/sync_prices.py --execute --universes sp500,ftsemib --max-symbols 25
python scripts/sync_fundamentals.py --execute --universes sp500,ftsemib --max-symbols 10
python scripts/sync_official_macro.py --execute --build-features --start 2010-01
python scripts/validate_data.py
```

## Backfill storico 2000-2026

Per portare la workstation verso la copertura completa richiesta dal progetto, usare l'orchestratore storico. Di default genera solo piano e manifest, senza chiamate provider:

```bash
python scripts/bootstrap_research_data_2000_2026.py --max-assets 25 --max-symbols 10
python scripts/validate_research_data_coverage.py
```

Per eseguire davvero download/refresh su provider configurati:

```bash
python scripts/bootstrap_research_data_2000_2026.py \
  --execute \
  --start-year 2000 \
  --end-year 2026 \
  --universes sp500,nasdaq100,eurostoxx50,ftsemib \
  --markets us_all,europe_major,japan_major,global_etfs \
  --max-assets 250 \
  --max-symbols 25
```

Per un backfill completo senza cap, usare `--max-assets 0 --max-symbols 0` sapendo che tempi, rate limit e copertura reale dipendono dai provider e dalle licenze disponibili.

Il training ML usa il pannello fattoriale prodotto dal backfill:

```bash
python scripts/train_ml_models_2000_2026.py --models ols,rf --train-end-year 2018 --test-start-year 2019
python scripts/validate_research_data_coverage.py --strict
```

Artifact principali:

- `output/data_completion/full_completion_manifest.json`
- `output/data_completion/DataCompletion_source_map.csv`
- `output/ml_training_lab/tables/FactorUniversePanel.csv`
- `output/ml_training_lab/tables/MLTraining_metrics.csv`
- `output/ml_training_lab/MLTraining_manifest.json`
- `output/ml_stock_lab/tables/MLStockLab_model_comparison.csv`

## Frequenza operativa

- Metadata universi: settimanale o quando cambia l'universo.
- Prezzi OHLCV: giornaliero/incrementale.
- Fondamentali: settimanale o post earnings.
- ML Stock Lab: dopo refresh fondamentali/prezzi.
- Smart Money / Gov Data: settimanale o prima di una review tematica.
- Banking Data Lab: on demand, con refresh market panel se serve.

## Policy

I dati sintetici o sample devono essere marcati nel nome file o nelle colonne di provenance. I provider esterni vanno usati solo dopo un controllo di freshness Drive-first, così la piattaforma resta riproducibile e non consuma crediti/API senza necessità.
