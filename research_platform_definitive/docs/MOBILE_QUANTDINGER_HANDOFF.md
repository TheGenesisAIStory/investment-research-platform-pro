# Mobile QuantDinger Handoff

La parte mobile usa i componenti Vue/Vant in:

```text
integrations/quantdinger_bridge/vue/mobile/
```

L'obiettivo di questo ciclo e' renderli utili anche senza backend QuantDinger completo: possono leggere strategie, segnali snapshot e summary backtest dal sidecar ML.

## Endpoint sidecar

| Endpoint | Scopo |
|---|---|
| `GET /api/v1/ml/models` | modelli disponibili |
| `GET /api/v1/ml/strategies` | registry strategie con formule e limiti |
| `GET /api/v1/ml/signals` | segnali live da QuantDinger OHLCV |
| `GET /api/v1/ml/signals/snapshot` | segnali salvati nel database Research Platform |
| `POST /api/v1/ml/backtest` | backtest on demand |
| `GET /api/v1/ml/backtests/summary` | storico backtest per dashboard |

## Configurazione app

In sviluppo:

```bash
export VITE_ML_API_URL=http://localhost:8000
```

In produzione si puo' usare un reverse proxy e lasciare `VITE_ML_API_URL` vuoto, cosi' le chiamate vanno verso la stessa origin.

## Esperienza mobile

`MLSignalsList.vue`:

- mostra strategie disponibili come tag;
- prova i segnali live;
- se il backend live fallisce, usa `/signals/snapshot`;
- mostra confidence, target e stop.

`MLBacktestSummary.vue`:

- lancia un backtest on demand;
- mostra Sharpe, return e drawdown;
- visualizza una equity curve compatta;
- carica gli ultimi backtest salvati con `/backtests/summary`.

## Flow consigliato

1. Popola il database sample:

```bash
python3 research_platform_definitive/scripts/populate_research_database.py
```

2. Avvia sidecar:

```bash
uvicorn integrations.quantdinger_bridge.ml_service:app --host 0.0.0.0 --port 8000
```

3. Avvia QuantDinger-Mobile puntando `VITE_ML_API_URL` al sidecar.

4. Apri schermate ML Signals e ML Backtest.

## Nota UX

I messaggi devono essere prudenti: "paper", "snapshot", "sample" e "backtest" sono parole importanti. La UI non deve suggerire che un segnale sia un ordine o una raccomandazione operativa.
