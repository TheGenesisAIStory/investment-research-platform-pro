# Gen.is.IA Time Series Forecasting Layer

## 1. Forecasting gia coperto dalla piattaforma

La piattaforma copre gia una forma importante di forecasting: la **predictability cross-sectionale dei rendimenti equity**. Il factor panel e il ML Stock Lab costruiscono target come `forward_return_21d`, `forward_return_63d` e `forward_return_252d`, cioe rendimenti futuri su orizzonti di circa 1, 3 e 12 mesi di mercato.

Questi target sono usati per stimare e confrontare modelli di selezione titoli. Il problema non e prevedere una singola serie nel tempo, ma ordinare un universo di titoli in base al ritorno atteso relativo. Per questo le metriche principali sono:

- **IC**: correlazione tra score previsto e rendimento futuro.
- **RankIC**: correlazione tra ranking previsto e ranking dei rendimenti realizzati.
- **Long-short Sharpe**: qualita economica del portafoglio long top-score e short bottom-score.
- **Out-of-sample R2**: miglioramento predittivo rispetto a un benchmark su dati non visti.

Questa parte e coerente con il ML equity research e con il factor investing: misura se value, quality, momentum, risk, size, growth e segnali model-based aiutano a distinguere titoli migliori e peggiori su orizzonti futuri.

## 2. Cosa mancava

Prima di questo layer mancava un modulo dedicato al **forecasting time-series** in senso classico:

- nessun forecast del livello o rendimento di una singola serie, come SPY, EURUSD, BTC, BTP proxy o AAPL;
- nessuna pipeline standard di feature univariate basata su lag, rolling mean, rolling volatility, trend, drawdown e moving-average gap;
- nessuna tabella di metriche time-series come MAE, RMSE, MAPE e directional accuracy;
- nessuna UI separata che distinguesse chiaramente il problema “forecast one series” dal problema “rank many stocks”.

Il nuovo layer non sostituisce il ML Stock Lab. Lo completa con una vista di scenario per indici, macro proxy, FX, commodity, crypto e singole equity.

## 3. Time Series Lab v1

Il modulo `research_platform_core.time_series_forecasting` introduce un Time Series Lab minimale ma serio.

### Input

Il Lab puo usare:

- **Macro DB**: serie da `MarketData/Macro`, popolate dalla Macro View con proxy globali, USA, EU, Italy, FX, commodity, fixed income e crypto.
- **Equity OHLCV**: parquet locali letti tramite la stessa root condivisa usata da Screener, Data Platform e ML Stock Lab.

### Feature engineering

Per ogni serie il layer costruisce solo feature disponibili alla data di forecast:

- rendimenti laggati su 1, 2, 5, 21 e 63 giorni;
- medie mobili dei rendimenti;
- volatilita rolling;
- momentum rolling;
- gap rispetto alla media mobile;
- drawdown rolling.

Il target e `forward_return_{horizon}d`, calcolato come:

```text
close[t + horizon] / close[t] - 1
```

Questa colonna non viene mai inclusa tra le feature.

### Modelli v1

La prima versione implementa:

- `naive`: baseline di persistenza basata sul rendimento trailing dello stesso orizzonte;
- `ols`: regressione lineare su lag e rolling features;
- `gbrt`: Gradient Boosting Regressor su lag e rolling features.

ARIMA/ETS, forecast intervals, modelli multivariati e LSTM restano estensioni pianificate. La v1 evita di aggiungerli come placeholder per mantenere il codice facile da verificare.

### Orizzonti

Gli orizzonti default sono:

- 5 giorni: scenario tattico breve;
- 21 giorni: circa un mese di mercato;
- 63 giorni: circa un trimestre.

Il Lab consente anche 126 giorni come opzione UI.

### Metriche

Le metriche prodotte sono:

- **MAE**: errore assoluto medio sul rendimento forecastato;
- **RMSE**: errore quadratico medio con penalizzazione degli errori grandi;
- **MAPE**: errore assoluto scalato per la dimensione del ritorno realizzato;
- **Directional accuracy**: quota di osservazioni in cui forecast e rendimento realizzato hanno lo stesso segno;
- **Hit ratio**: quota di previsioni con segno e payoff coerenti.

## 4. UI e workflow

La pagina `Time Series Lab` vive sotto **LABS**.

La UI contiene:

- selezione sorgente: Macro DB oppure Equity OHLCV;
- selezione serie: proxy macro/market o ticker equity;
- selezione orizzonti e modelli;
- opzioni avanzate per train/test split e massimo numero di righe;
- grafico storico interattivo;
- tabella dei forecast piu recenti;
- grafico forecast vs realized;
- tabella metriche;
- schema delle feature usate.

Gli artifact sono salvati in:

```text
output/time_series_lab/tables/
  TimeSeriesForecast_metrics.csv
  TimeSeriesForecast_predictions.csv
  TimeSeriesForecast_latest.csv
  TimeSeriesForecast_feature_schema.csv
  TimeSeriesForecast_manifest.json
```

## 5. Collegamenti con Macro e Portfolio

Il Time Series Lab deve essere usato come **contesto di scenario**, non come motore automatico di trading.

Uso consigliato:

- Macro View: mostrare forecast su SPY, QQQ, FEZ, EWI, EURUSD, DXY, TLT, GLD, BTC e altri proxy;
- Portfolio: usare forecast di indici e tassi come indicatore di regime, stress o scenario;
- ML Stock Lab: mantenere separato il ranking cross-sectionale, ma confrontare i segnali equity con il contesto macro forecastato.

## 6. Estensioni v1.5/v2

Prossimi passi consigliati:

1. Aggiungere baseline ARIMA/ETS opzionali se `statsmodels` e disponibile.
2. Aggiungere intervalli di previsione usando residui out-of-sample o conformal intervals.
3. Introdurre modelli multivariati con feature macro laggate.
4. Salvare model cards per ogni forecast run.
5. Collegare Macro View e Portfolio a `TimeSeriesForecast_latest.csv`.
6. Valutare LSTM/transformer solo come laboratorio sperimentale, non come default.
