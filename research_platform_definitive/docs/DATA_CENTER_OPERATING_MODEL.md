# Data Center Operating Model

Questo documento descrive lo strato dati finale della Research Platform. L'idea e' semplice: i notebook, la Streamlit app, il Laboratorio LLM e il bridge QuantDinger devono leggere dallo stesso contratto dati, senza duplicare logica e senza dipendere sempre da API live.

## Avvio rapido

Da root repository:

```bash
python3 research_platform_definitive/scripts/populate_research_database.py
```

Output principali:

- database locale: `research_platform_definitive/output/data_cache/research_platform.sqlite`
- sample CSV tracciati: `research_platform_definitive/data/sample/`
- summary JSON: `research_platform_definitive/data/sample/research_database_summary.json`

Il database generato e' locale e ignorato da Git. I CSV sample sono piccoli e servono come fixture leggibili, esempi Colab e fallback per review.

## Schema canonico

| Tabella | Cosa contiene | Chi la usa |
|---|---|---|
| `instruments` | anagrafica strumenti, ticker, valuta, mercato, settore | Data Center, mobile, notebook |
| `ohlcv_daily` | open/high/low/close/adj close/volume giornalieri | Research Platform, feature builder |
| `features_labels` | return, momentum, volatilita', z-score e label forward | ML Stock Lab, LLM Lab |
| `ml_signals` | segnali, score, confidence, target weight, reasoning | mobile, QuantDinger bridge |
| `backtest_results` | metriche aggregate di backtest | dashboard e documentazione |
| `backtest_equity` | curva equity giornaliera | chart e diagnostica |
| `experiment_logs` | log esperimenti ML, parametri e note | governance ricerca |
| `strategy_registry` | strategie, formule, assunzioni e limiti | LLM Lab, mobile, docs |
| `ingestion_runs` | audit di ogni popolamento/refresh | Data Center |
| `data_catalog_entries` | catalogo operativo dei dataset | Data/API Control Center |

## Come aggiungere una nuova fonte dati

1. Aggiungi la sorgente in `config/data_sources.yaml` o nel loader appropriato sotto `src/research_platform_core/loaders/`.
2. Normalizza gli strumenti in un DataFrame con almeno `symbol`, `name`, `exchange`, `country`, `currency`.
3. Scrivi prezzi o dati derivati usando `ResearchDatabase`:

```python
from research_platform_core.research_database import ResearchDatabase

db = ResearchDatabase()
db.upsert_instruments(instruments_df)
db.write_ohlcv(ohlcv_df, source="provider_name")
db.write_features_labels(features_df, feature_set="final_v1")
```

4. Registra il refresh in `ingestion_runs` con `write_ingestion_run`.
5. Aggiorna `data_catalog_entries` se nasce un nuovo dataset stabile.

## Query utili

```sql
SELECT symbol, name, exchange, sector
FROM instruments
ORDER BY symbol;
```

```sql
SELECT i.symbol, p.date, p.close, p.volume
FROM ohlcv_daily p
JOIN instruments i ON i.instrument_id = p.instrument_id
ORDER BY p.date DESC
LIMIT 20;
```

```sql
SELECT i.symbol, s.date, s.strategy_name, s.signal, s.target_weight, s.confidence
FROM ml_signals s
JOIN instruments i ON i.instrument_id = s.instrument_id
ORDER BY s.date DESC, ABS(s.target_weight) DESC
LIMIT 20;
```

## Dati sample vs dati reali

Il dataset creato da `populate_research_database.py` e' sintetico ma market-shaped: prezzi con drift, volatilita', fattore comune, volumi lognormali, feature e backtest coerenti. Serve per:

- demo locale senza provider;
- test ripetibili;
- mobile snapshot;
- prompt LLM con contesto concreto.

Quando sono disponibili dati reali, la pipeline resta la stessa: sostituisci la fonte, mantieni schema e provenance.

## Regola di qualita'

Prima di fidarti di un risultato, controlla:

- copertura strumenti e date;
- assenza di buchi lunghi in `ohlcv_daily`;
- feature calcolate solo con dati disponibili nel passato;
- costi e turnover nel backtest;
- confronto con baseline semplice, non solo con il modello migliore.
