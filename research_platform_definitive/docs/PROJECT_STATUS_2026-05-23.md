# Stato progetto - 2026-05-23

Aggiornato: 2026-05-23 19:46 CEST  
Repository locale: `/Users/itsgennymac/GitHub/machine-learning-for-trading`  
Branch: `codex/research-platform-workstation`  
Remote GitHub: `https://github.com/TheGenesisAIStory/ml-trading-thesis-bot.git`

## Sintesi

Il progetto operativo e' ora centrato su `research_platform_definitive/`. La vecchia struttura estesa del repository e' stata sostituita da un bundle canonico con notebook, app Streamlit, package riusabili, documentazione, validatori e integrazioni.

Lo stato funzionale verificato in questo passaggio e' positivo: test mirati, validatori, smoke check e compilazione Python passano. Rimane da trattare con attenzione la dimensione della migrazione Git: il working tree contiene molte rimozioni intenzionali della vecchia struttura e il nuovo bundle deve essere salvato come migrazione coerente.

## Verifiche eseguite

| Area | Comando | Esito |
|---|---|---|
| Bundle canonico | `.venv/bin/python research_platform_definitive/scripts/validate_definitive_bundle.py` | PASS |
| Notebook canonici | `.venv/bin/python research_platform_definitive/scripts/validate_notebooks.py` | PASS |
| Artifact contract | `.venv/bin/python research_platform_definitive/scripts/validate_artifacts.py` | PASS |
| Laboratorio reference | `.venv/bin/python research_platform_definitive/scripts/validate_laboratorio.py` | PASS, 129 file, 75 notebook |
| Test Python | `.venv/bin/python -m pytest tests/test_quantdinger_bridge research_platform_definitive/tests -q` | PASS, 27 test |
| Compile check | `.venv/bin/python -m compileall -q ml_stock_lab integrations research_platform_definitive/tests research_platform_definitive/src research_platform_definitive/research_platform_app research_platform_definitive/scripts` | PASS |
| App orchestration | `.venv/bin/python research_platform_definitive/research_platform_app/smoke_checks.py` | PASS |
| Data/API control | `.venv/bin/python research_platform_definitive/scripts/smoke_data_api_control.py` | PASS |
| Data Center | `.venv/bin/python research_platform_definitive/scripts/smoke_data_center_enhancement.py` | PASS |
| ML Stock Lab | `.venv/bin/python research_platform_definitive/scripts/smoke_ml_stock_lab.py` | PASS |
| Smart Money engine | `.venv/bin/python research_platform_definitive/scripts/smoke_smart_money_engine.py` | PASS |
| OHLCV pipeline | `.venv/bin/python research_platform_definitive/scripts/smoke_ohlcv_pipeline.py` | PASS |

Nota: i test passano con soli warning di deprecazione `Pandas4Warning` su `pd.Timestamp.utcnow()`. Non bloccano il funzionamento, ma vanno pianificati per pulizia futura.

## Stato per componente

| Componente | Percorso canonico | Stato | Note operative |
|---|---|---|---|
| Root repository | `README.md`, `.env.example`, `.gitignore` | Funzionante | La root punta al bundle definitivo. `.env.example` include variabili per storage e cache. |
| Bundle definitivo | `research_platform_definitive/` | Funzionante | Validato con `validate_definitive_bundle.py`. E' la fonte di verita' operativa. |
| Notebook canonici | `company_valuation/notebooks`, `portfolio_analysis/notebooks`, `machine_learning_lab/notebooks`, `data_api_management/notebooks` | Funzionanti | I notebook principali sono presenti e validi JSON notebook. |
| Streamlit workstation | `research_platform_definitive/research_platform_app/` | Funzionante | Smoke orchestration OK. Avvio: `streamlit run research_platform_definitive/research_platform_app/app.py`. |
| Data/API control center | `research_platform_definitive/data_api_app.py`, `research_platform_definitive/research_platform_app/pages/11_Data_API_Control_Center.py` | Funzionante | Smoke OK. Gestisce provider, credential status, export batch e contratti dati. |
| Core data platform | `research_platform_definitive/src/research_platform_core/` | Funzionante | Test e smoke OK. Include cataloghi, cache, API management, batch export, OHLCV e macro loaders. |
| Data Center enhancement | `research_platform_definitive/src/research_platform_core/loaders/` | Funzionante | Smoke OK. Copertura locale rilevata: factor data 38.9%, OHLCV 60%, altre famiglie non ancora popolate. |
| OHLCV pipeline | `ohlcv_ingest.py`, `ohlcv_store.py`, `scripts/smoke_ohlcv_pipeline.py` | Funzionante | Test upsert SQLite, import Kaggle seed e ingest fake OK. |
| Official macro | `loaders/ecb_client.py`, `loaders/bditalia_client.py`, `macro_features.py` | Funzionante | Test inclusi nella suite. Serve popolamento dati/API per copertura reale. |
| ML Stock Lab | `research_platform_definitive/src/ml_stock_lab/` e `ml_stock_lab/` | Funzionante | Smoke OK. Il package root supporta integrazioni esterne; il package nel bundle e' canonico per la piattaforma. |
| Smart Money engine | `research_platform_definitive/src/smart_money_engine/` | Funzionante | Smoke OK. Gli artifact demo sono validati. |
| Company Valuation | `research_platform_definitive/company_valuation/` | Funzionante come bundle | Moduli e notebook canonici presenti. Da validare con dati/provider reali quando si esegue analisi live. |
| Portfolio Analysis | `research_platform_definitive/portfolio_analysis/` | Funzionante come bundle | Moduli e notebook canonici presenti. Da validare con portafogli reali quando disponibili. |
| Laboratorio | `research_platform_definitive/laboratorio/` | Reference OK | Libreria di ricerca curata, non fonte di produzione. Validati 129 file e 75 notebook. |
| Vibe-Trading bridge | `integrations/vibe_trading_bridge/`, `vibe_trading/` | Compila | Dipendenza opzionale via submodule. Requirements in `requirements-vibe.txt`. |
| QuantDinger bridge | `integrations/quantdinger_bridge/`, `external/QuantDinger*` | Testato | Test adapter/config/db/service inclusi nei 27 test. Submodule puliti localmente. |
| Drive sync | `scripts/sync_repo_to_google_drive.sh` | Pronto per sync | Target locale Google Drive: `MyDrive/GitHub/machine-learning-for-trading/`. Esclude `.git`, venv, cache e DB locali. |
| GitHub sync | branch `codex/research-platform-workstation` | Pronto per commit/push | Da salvare come migrazione unica con documentazione e fix test harness. |

## Intervento di armonizzazione fatto

E' stato aggiunto `research_platform_definitive/tests/conftest.py` per rendere importabili i package canonici durante `pytest` senza dover impostare manualmente `PYTHONPATH`. Prima di questa correzione, i test del bundle fallivano in collection su `ModuleNotFoundError: No module named 'research_platform_core'`.

## Stato Git prima del salvataggio finale

La migrazione e' ampia:

- 934 file tracciati risultavano rimossi dalla vecchia struttura.
- 3 file tracciati risultavano modificati: `.env.example`, `.gitignore`, `README.md`.
- 20 nuove entry principali risultavano non tracciate, tra cui `research_platform_definitive/`, `ml_stock_lab/`, `integrations/`, `tests/`, `external/`, `vibe_trading/`, `docs/` e script di sync.

Questa forma e' coerente con una riorganizzazione: vecchi materiali fuori standard sono stati rimossi o spostati, mentre il bundle definitivo diventa il punto di manutenzione.

## Salvataggi

Locale:

- Documento di stato: `research_platform_definitive/docs/PROJECT_STATUS_2026-05-23.md`
- Fix test harness: `research_platform_definitive/tests/conftest.py`
- Indici aggiornati: `research_platform_definitive/MANIFEST.csv`, `research_platform_definitive/REVIEW_SUMMARY.md`

Drive:

- Script operativo: `./scripts/sync_repo_to_google_drive.sh`
- Destinazione: `/Users/itsgennymac/Library/CloudStorage/GoogleDrive-sfn.gns@gmail.com/Il mio Drive/GitHub/machine-learning-for-trading`
- Modalita' standard: `SYNC_MODE=operational`, senza `.git`, virtualenv, cache runtime, DB locali e archive pesanti.

GitHub:

- Remote: `origin`
- Branch: `codex/research-platform-workstation`
- Il salvataggio deve includere la migrazione completa, questo documento e il fix dei test.

## Rischi residui e prossimi passi

1. La copertura dati reale non e' completa in locale: `smoke_data_center_enhancement.py` segnala 0% per equity universe, FX, official macro, risk factors e commodities. Non e' un errore di codice, ma richiede popolamento dati/API.
2. I warning `pd.Timestamp.utcnow()` vanno sostituiti con `pd.Timestamp.now("UTC")` prima di un upgrade Pandas maggiore.
3. I submodule `vibe_trading`, `external/QuantDinger` ed `external/QuantDinger-Mobile` risultano puliti localmente, ma vanno mantenuti tramite `.gitmodules` e commit gitlink.
4. Le analisi live di Company Valuation e Portfolio Analysis richiedono dati/provider reali per una verifica end-to-end oltre gli smoke test.
5. Per evitare nuovo drift, ogni nuovo notebook deve essere promosso tramite `MANIFEST.csv`, `notebooks/README.md` e i validator del bundle.

## Comandi rapidi

Validazione completa essenziale:

```bash
.venv/bin/python -m pytest tests/test_quantdinger_bridge research_platform_definitive/tests -q
.venv/bin/python research_platform_definitive/scripts/validate_definitive_bundle.py
.venv/bin/python research_platform_definitive/scripts/validate_notebooks.py
.venv/bin/python research_platform_definitive/scripts/validate_artifacts.py
.venv/bin/python research_platform_definitive/scripts/validate_laboratorio.py
.venv/bin/python research_platform_definitive/research_platform_app/smoke_checks.py
.venv/bin/python research_platform_definitive/scripts/smoke_data_api_control.py
.venv/bin/python research_platform_definitive/scripts/smoke_data_center_enhancement.py
.venv/bin/python research_platform_definitive/scripts/smoke_ml_stock_lab.py
.venv/bin/python research_platform_definitive/scripts/smoke_smart_money_engine.py
.venv/bin/python research_platform_definitive/scripts/smoke_ohlcv_pipeline.py
```

Avvio app:

```bash
streamlit run research_platform_definitive/research_platform_app/app.py
streamlit run research_platform_definitive/data_api_app.py
```

Sync Drive:

```bash
./scripts/sync_repo_to_google_drive.sh
```
