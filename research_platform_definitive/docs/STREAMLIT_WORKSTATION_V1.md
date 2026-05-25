# Streamlit Workstation v1

Questo documento descrive il layer applicativo della Investment Research Platform come console quotidiana per un desk buy-side.

## Menu operativo

La navigazione e divisa in tre blocchi:

- **RESEARCH**: Home / Overview, Screening & Research, Valuation, Portfolio, Smart Money & Macro. Sono le pagine per l'uso quotidiano di analisti e PM.
- **LABS**: ML Stock Lab & Models, Banking Data Lab, Research Library. Sono ambienti di ricerca avanzata e validazione.
- **PLATFORM OPS**: Data Platform, Notebook Runner, Export Center, Hi-Freq Engine, Data/API Control Center. Sono pagine operative per dati, job, export e troubleshooting.

Ogni pagina principale usa la context bar globale con reporting currency, FX proxy, benchmark, overlays, selected ticker e screener attivo.

## Configurazione persistente

Le preferenze utente e i parametri di routing applicativo sono salvati in:

```text
research_platform_definitive/output/config/platform_settings.json
```

Il file e creato/aggiornato dalla app e contiene:

- `screener`: template default, colonne default e custom watchlist.
- `models`: registry modelli, modelli attivi, pesi del composite score.
- `data`: policy refresh, strictness coverage, soglie watch.
- `smart_money`: soglie e default per eventi recenti.
- `banking`: preferenze leggere per Banking Data Lab.

Il Database Finanziario resta il contratto dati canonico. `platform_settings.json` non duplica dataset, ma controlla come la UI li presenta e quali model stack usare.

## Uso senza notebook

L'utente basico puo:

1. Aprire **Home / Overview** per leggere salute dati, run recenti e coverage.
2. Usare **Screening & Research** per costruire screeners, salvare config e aprire ticker nelle pagine Research.
3. Usare **Valuation**, **Portfolio**, **Smart Money & Macro** e **ML Stock Lab** sui dati/artifacts esistenti.
4. Avviare refresh leggeri dalle pagine dedicate quando gli artifacts sono mancanti o stale.

I notebook restano disponibili tramite **Notebook Runner** per sviluppo, manutenzione avanzata e job batch pesanti.

## Multi-modello ML

Il model routing della UI e gestito in **ML Stock Lab & Models**:

- selezione dei modelli attivi;
- pesi del composite score;
- tabella registry con tipo, universo, finestra training e metrica primaria.

Quando gli artifacts espongono colonne `score_<model_id>`, la app le mostra affiancate e calcola `score_composite`. Se e disponibile solo lo score canonico, la UI lo conserva come fonte validata senza fabbricare predizioni inesistenti.

## Data Platform

**Data Platform** mostra:

- **Single Ticker Explorer**: search box per un ticker e mappa immediata di
  OHLCV, fundamentals, factor panel, ML signals, valuation proxy e Smart Money.
- **Data Explorer**: preview tabellare e grafica leggera su manifest OHLCV,
  fundamentals manifest, factor panel, segnali ML e Smart Money.
- **Domain Status**: stato esplicito per Equity, FX/Macro, Factor Libraries,
  Smart Money e Banking. Le sezioni non popolate dichiarano se sono planned,
  parziali o in attesa di backfill.
- path del Database Finanziario condiviso;
- contratto domini dati;
- manifest OHLCV con `OK`, `LIMITED_HISTORY`, `DELISTED`, `NETWORK_TIMEOUT`, `NO_PRICE_DATA`;
- log run recenti;
- Data Settings per strictness e policy refresh.

`LIMITED_HISTORY` e accettabile in modalita non-strict: indica listing recente/storia corta, non un errore critico.
