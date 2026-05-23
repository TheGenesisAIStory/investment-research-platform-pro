# data_quality_brief_it

Generated: 2026-05-23T18:20:59.279408+00:00

Sei il responsabile Data Center.

Catalogo dati:
dataset_id,domain,table_name,title_it,description_it,frequency,primary_key,storage,owner,freshness_rule,sample_query,updated_at
instrument_master,reference_data,instruments,Anagrafica strumenti,"Ticker, mercato, valuta, settore e simbolo provider usati da app e notebook.",on_change,symbol,sqlite:research_platform.sqlite,Data Center,Aggiorna quando cambia universo o provider.,"SELECT symbol, name, exchange, sector FROM instruments ORDER BY symbol;",2026-05-23T18:18:54.525973+00:00
ohlcv_daily,market_data,ohlcv_daily,Prezzi OHLCV giornalieri,"Open, high, low, close, adjusted close e volumi con tracciamento fonte.",daily,"instrument_id,date,source",sqlite:research_platform.sqlite + parquet lake opzionale,Data Center,Refresh giornaliero per strumenti attivi; API solo se cache stale.,SELECT * FROM ohlcv_daily ORDER BY date DESC LIMIT 20;,2026-05-23T18:18:54.525973+00:00
features_labels_final_v1,ml_features,features_labels,Feature e label ML,"Rendimenti, momentum, volatilita', z-score e label forward per esperimenti ML.",daily_after_prices,"instrument_id,date,feature_set",sqlite:research_platform.sqlite,Research Platform,Rigenera dopo ogni refresh OHLCV o cambio formula.,SELECT * FROM features_labels WHERE feature_set='final_v1' LIMIT 20;,2026-05-23T18:18:54.525973+00:00
ml_signals_latest,signals,ml_signals,Segnali ML e target weight,"Segnali pronti per dashboard, mobile e bridge QuantDinger.",on_backtest_or_refresh,"instrument_id,date,strategy_name,model_name,horizon",sqlite:research_platform.sqlite,LLM Lab / Research Platform,Aggiorna a fine run; invalida se dati o modello cambiano.,SELECT * FROM ml_signals ORDER BY date DESC LIMIT 20;,2026-05-23T18:18:54.525973+00:00
backtest_results,backtests,backtest_results,Storico backtest,"Metriche aggregate, universo, finestra e configurazione modello.",on_run,backtest_id,sqlite:research_platform.sqlite,Research Platform,Append-only: ogni esperimento produce un nuovo record.,"SELECT strategy_name, total_return, sharpe, max_drawdown FROM backtest_results;",2026-05-23T18:18:54.525973+00:00


Conteggi tabelle:
{
  "backtest_equity": 1405,
  "backtest_results": 1,
  "data_catalog_entries": 5,
  "experiment_logs": 3,
  "features_labels": 10736,
  "ingestion_runs": 1,
  "instruments": 8,
  "ml_signals": 496,
  "ohlcv_daily": 11240,
  "strategy_registry": 3
}

Scrivi un brief operativo in italiano:
- cosa e' pronto;
- quali dati sono sintetici/sample;
- quali tabelle alimentano Research Platform, LLM Lab e mobile;
- come aggiungere una nuova fonte dati;
- quali controlli fare prima di fidarsi dei risultati.
