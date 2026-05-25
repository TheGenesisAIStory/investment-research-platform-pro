# Data Completion 2000-2026

Questo documento è il runbook operativo per completare Database Finanziario, fattori e training ML della Investment Research Platform Pro sul periodo 2000-2026.

Setup sviluppatore e packaging editable: vedere anche `docs/DEV_SETUP.md`.

## Scope

L'orchestratore storico coordina solo sorgenti già coerenti con il progetto:

| Dominio | Loader / provider policy | Artifact principali |
| --- | --- | --- |
| Equity prices | `OhlcvIngestJob` con yfinance, Stooq fallback e seed configurati | `Database Finanziario/MarketData/OHLCV` |
| Equity fundamentals | `EquityUniverseManager` e statement/multipli provider-available | `Database Finanziario/Equities/.../fundamentals` |
| Macro / FX / risk | yfinance overlays, FRED/CSV, ECB, Banca d'Italia | `FX`, `Commodities`, `RiskFactors`, `OfficialMacro` |
| Factor libraries | Fama-French, AQR, fattori derivati da OHLCV locale | `Factors`, `output/ml_training_lab` |
| Smart Money / Gov Data | `smart_money_engine` su fonti ufficiali/locali | `output/smart_money` |
| Banking Data Lab | `run_banks_data_pipeline` | `output/banks_pipeline` |

La copertura 2000-2026 è un obiettivo di backfill. Quando un provider non espone dati così lunghi, il manifest deve dichiarare la copertura effettiva; non si inventano fondamentali o eventi ufficiali mancanti.

## Root unico condiviso

Il Database Finanziario e' uno solo. Core, app Streamlit, ML Stock Lab, Notebook Runner e script risolvono lo stesso root tramite `research_platform_core.storage_policy` e `research_platform_core.data_platform`.

Priorita':

1. `FINANCIAL_DB_ROOT`, se impostata.
2. Path Drive/Colab canonici (`/content/drive/...` o `~/Library/CloudStorage/.../Database Finanziario`).
3. Fallback locale solo per smoke test e ambienti non montati.

Gli output operativi leggeri vanno in `RESEARCH_PLATFORM_OUTPUT_ROOT` (`output/` del bundle per default); gli artifact pesanti e storici stanno nel Database Finanziario.

## Comandi

Dry-run sicuro:

```bash
python scripts/bootstrap_research_data_2000_2026.py --max-assets 25 --max-symbols 10
python scripts/validate_research_data_coverage.py
```

Backfill reale con cap da scrivania:

```bash
python scripts/bootstrap_research_data_2000_2026.py \
  --execute \
  --start-year 2000 \
  --end-year 2026 \
  --max-assets 250 \
  --max-symbols 25
```

Backfill full-universe:

```bash
PYTHONUNBUFFERED=1 python -u scripts/bootstrap_research_data_2000_2026.py \
  --execute \
  --full \
  --start-year 2000 \
  --end-year 2026 \
  --max-assets 0 \
  --max-symbols 0 \
  --refresh
```

Training ML:

```bash
python scripts/train_ml_models_2000_2026.py \
  --start-year 2000 \
  --end-year 2026 \
  --train-end-year 2018 \
  --test-start-year 2019 \
  --models ols,rf
```

Training con sintesi Ollama locale:

```bash
ollama serve
python scripts/train_ml_models_2000_2026.py --models ols,rf --use-ollama --ollama-model llama3.1
```

Validation release:

```bash
python scripts/validate_research_data_coverage.py --strict
python scripts/validate_artifacts.py
python scripts/validate_definitive_bundle.py
pytest tests -q
```

## Routine giornaliera

Per l'uso desk-day, evitare full backfill continui. La routine consigliata e':

```bash
python scripts/sync_ohlcv_prices.py --execute --mode incremental --max-assets 0
python scripts/validate_research_data_coverage.py
```

Policy:

- OHLCV: daily incremental.
- Fundamentals: weekly o post reporting season.
- Smart Money / Gov Data: weekly o on-demand.
- Banking Data: on-demand.
- ML training: weekly/mensile o dopo refresh materiale del pannello fattoriale.

La pagina Data Platform mostra il contratto del Database Finanziario condiviso, i coverage status OHLCV e i log dei run recenti.

## Output attesi

- `output/data_completion/DataCompletion_source_map.csv`: mappa fonte, target e policy per dominio.
- `output/data_completion/*_2000_2026.csv`: manifest per dominio.
- `output/data_completion/full_completion_manifest.json`: riepilogo run.
- `output/data_completion/ResearchDataCoverageValidation.csv`: controllo PASS/FAIL non ambiguo.
- `output/ml_training_lab/tables/FactorUniversePanel.csv`: pannello fattoriale point-in-time-safe derivato da dati locali.
- `output/ml_training_lab/tables/MLTraining_metrics.csv`: metriche per modello, split e finestra.
- `output/ml_training_lab/tables/MLTraining_predictions.csv`: predizioni OOS per ticker/data.
- `output/ml_training_lab/models/*.pkl`: modelli serializzati con model card.
- `output/ml_stock_lab/tables/MLStockLab_model_comparison.csv`: vista consumabile dalla pagina ML Stock Lab.

## Regole metodologiche

- Il target ML standard è `forward_return`; viene calcolato solo da prezzi già presenti localmente.
- Gli split sono temporali: default train `2000-2018`, test `2019-2026`.
- I fattori derivati da prezzi usano solo informazione passata (`ret21d`, `ret63d`, `ret252d`, `vol63d`).
- Fondamentali, Smart Money e Banking mantengono provenance e copertura reale; se la storia 2000-2026 non esiste nel provider, il limite resta visibile nel manifest.
- La modalità dry-run è accettabile per smoke test, ma una release v1.0 dati-richiede `--execute` e validazione strict.

## Stati OHLCV speciali

La richiesta storica `2000-2026` non implica che ogni ticker debba avere prezzi dal 2000. La pipeline distingue:

- `OK`: storia coerente con la finestra richiesta o con la copertura provider.
- `LIMITED_HISTORY`: ticker attivo con listing recente; i prezzi iniziano dopo il 2000 e la finestra pre-listing resta vuota.
- `DELISTED`: nessun dato recente o ultimo prezzo troppo lontano dall'end date effettiva.
- `NETWORK_TIMEOUT`: timeout, `curl 28`, connessione o rate-limit temporaneo.
- `NO_PRICE_DATA` / `PROVIDER_ERROR`: risposta vuota o errore non classificabile.

Esempio: `FBYD` e' quotato su Nasdaq ma ha storia disponibile da ottobre 2023. Se un provider segnala `possibly delisted` su una variante come `FBYDP`, la pipeline prova un rescue recente e, quando trova la storia di `FBYD`, scrive `LIMITED_HISTORY` con `resolved_provider_symbol=FBYD`. Questo dato e' valido per Screener/ML/Portfolio dal primo giorno disponibile; non vengono forzati prezzi inesistenti prima del listing.

## Guardrail Google Drive

Durante i backfill full la pipeline evita di leggere alla cieca cache `constituents_current.csv` minuscole o header-only dentro `CloudStorage/GoogleDrive`. Questi file possono essere placeholder o non ancora idratati da Drive e bloccare `pandas.read_csv` prima del primo log utile. In quel caso il loader li considera cache non affidabili, ricalcola i constituents dal provider quando possibile e scrive progress log immediati per ogni stage.

Il coverage finale del backfill usa il catalogo target canonico invece di una scansione ricorsiva completa del Drive. La scansione globale rimane disponibile nelle pagine Data Platform, ma non deve bloccare un job storico lungo.

### Scrittura parquet OHLCV

Per i parquet giornalieri OHLCV, Google Drive non e' considerato una destinazione affidabile per scritture batch lunghe. Se `FINANCIAL_DB_ROOT` punta a un mount Drive/CloudStorage, `OhlcvIngestJob` usa automaticamente uno staging locale:

```text
output/data_cache/financial_db_mirror/MarketData/OHLCV/
```

Puoi forzare una root locale esplicita:

```bash
python scripts/bootstrap_research_data_2000_2026.py \
  --execute \
  --stage equity_prices \
  --full \
  --refresh \
  --start-year 2000 \
  --end-year 2026 \
  --markets us_all,europe_major,japan_major,global_etfs \
  --max-assets 0 \
  --ohlcv-parquet-root "$PWD/output/local_financial_db/MarketData/OHLCV"
```

La pipeline scrive prima un parquet temporaneo locale, poi fa replace/copy con retry. Se una scrittura parquet fallisce comunque, il job non si interrompe: i prezzi gia' inseriti nello store SQL restano validi, la riga manifest viene marcata `downloaded_db_only`, e il dettaglio appare in:

```text
output/tables/OHLCV_write_failures.csv
```

La modalita' legacy direct-to-Drive e' disponibile solo se impostata esplicitamente:

```bash
RESEARCH_PLATFORM_OHLCV_WRITE_MODE=drive python scripts/bootstrap_research_data_2000_2026.py --execute --stage equity_prices
```

Per un full backfill, la policy consigliata e': scrivere parquet localmente, validare coverage/manifest, poi usare un job separato di sync/archivio verso Drive.

### Fonte unica OHLCV e Data Health

La root effettiva dei parquet OHLCV e' risolta in un solo punto del core:

```python
from research_platform_core import get_ohlcv_parquet_root_info

info = get_ohlcv_parquet_root_info(financial_db_root, output_root)
```

L'ordine di precedenza e':

1. parametro esplicito (`parquet_root` / `--ohlcv-parquet-root`);
2. `RESEARCH_PLATFORM_OHLCV_PARQUET_ROOT`;
3. setting persistente `output/config/platform_settings.json -> data.ohlcv_parquet_root`;
4. `parquet_root` registrata nell'ultimo `output/tables/OHLCV_daily_manifest.csv`;
5. root locale automatica se il Database Finanziario e' su Drive/CloudStorage;
6. `Database Finanziario/MarketData/OHLCV` per filesystem locali stabili.

I loader che indicizzano i parquet giornalieri usano `get_ohlcv_daily_search_roots(...)`: la prima root e' sempre quella canonica, mentre le directory legacy restano solo come fallback di lettura durante la migrazione.

Per alimentare la pagina Data Platform senza logica Streamlit nel core sono disponibili:

```python
from research_platform_core import (
    get_data_health_summary,
    list_ohlcv_failures,
    restart_equity_prices,
)
```

- `get_data_health_summary(...)` produce righe per stage (`equity_fundamentals`, `equity_prices`, `macro_fx`, ecc.) con stato, asset coperti, file parquet e failure count.
- `list_ohlcv_failures(...)` legge `output/tables/OHLCV_write_failures.csv` e restituisce una tabella retry-ready con ticker, exchange, target path, errore sintetico e timestamp.
- `restart_equity_prices({"failed_only": True, ...})` rilancia programmaticamente lo stage prezzi solo sugli asset falliti; con `dry_run=True` valida la selezione senza chiamare provider.

## Frequenza

- OHLCV: giornaliero incrementale.
- Fondamentali: settimanale e dopo reporting season.
- Macro/FX/risk: giornaliero o settimanale secondo fonte.
- Smart Money / Gov Data: settimanale o on-demand per review tematiche.
- Banking: on-demand, con market panel quando serve.
- ML training: dopo refresh materiale di prezzi/fattori/fondamentali o prima di release.
