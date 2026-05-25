# Developer Setup

Questo bundle usa un packaging editable standard. I package condivisi sono:

- `research_platform_core`: data layer, Database Finanziario, OHLCV, coverage, banking, macro e orchestrazione dati.
- `ml_stock_lab`: training, scoring, segnali e model artifacts.
- `smart_money_engine`: Smart Money / Gov Data.

La app Streamlit rimane in `research_platform_app/`, ma importa sempre questi package installati.

## Setup locale

```bash
cd research_platform_definitive
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -c "import research_platform_core, ml_stock_lab, smart_money_engine"
```

## Root condivisi

Il Database Finanziario e' unico. Cambia solo `FINANCIAL_DB_ROOT` per spostare il data lake usato da core, app, script, Notebook Runner e ML Lab.

```bash
export FINANCIAL_DB_ROOT="/path/to/Database Finanziario"
export RESEARCH_PLATFORM_OUTPUT_ROOT="$(pwd)/output"
export RESEARCH_PLATFORM_LOCAL_CACHE="$(pwd)/output/data_cache"
```

Se `FINANCIAL_DB_ROOT` non e' impostata, il resolver prova i path Drive/Colab canonici definiti in `research_platform_core.storage_policy`.

## Avvio app

```bash
cd research_platform_definitive
streamlit run research_platform_app/app.py --server.port 8503
```

Oppure:

```bash
./RUN_STREAMLIT_APP.sh
```

Lo script controlla gli import dei package e installa editable se il venv non e' ancora pronto.

## Verifiche minime

```bash
python -c "import research_platform_core, ml_stock_lab, smart_money_engine"
python -m compileall -q src research_platform_app scripts
pytest tests -q
python scripts/validate_definitive_bundle.py
python scripts/validate_research_data_coverage.py
curl -I --max-time 5 http://localhost:8503
```

## Routine dati

Daily refresh leggero:

```bash
python scripts/sync_ohlcv_prices.py --execute --mode incremental --max-assets 0
python scripts/validate_research_data_coverage.py
```

Backfill storico completo:

```bash
PYTHONUNBUFFERED=1 python -u scripts/bootstrap_research_data_2000_2026.py \
  --execute --full --refresh \
  --start-year 2000 --end-year 2026 \
  --max-assets 0 --max-symbols 0 --max-factor-datasets 0
```

Training ML dopo refresh materiale:

```bash
python scripts/train_ml_models_2000_2026.py \
  --start-year 2000 --end-year 2026 \
  --train-end-year 2018 --test-start-year 2019 \
  --models ols,rf
```
