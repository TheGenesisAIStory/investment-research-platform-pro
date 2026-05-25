# ML Content Layer v2

Questa nota documenta il passaggio dal ML Lab come prototipo di notebook/artifact
a un layer quantitativo condiviso fra ML Stock Lab e Screener.

## Factor vocabulary

Il vocabolario canonico vive in `src/ml_stock_lab/factor_registry.py`.
Ogni score e' espresso 0-100, dove valori piu' alti sono migliori dal punto di
vista della ricerca:

- `value`: P/E, P/B, EV/EBIT, EV/EBITDA piu' bassi e dividend yield piu' alto.
- `quality`: ROE, ROIC, margini piu' alti e leva piu' bassa.
- `momentum`: preferisce `momentum_12_1`; in fallback usa `ret252d`.
- `risk`: volatilita', beta e drawdown piu' bassi.
- `size`: market cap/liquidita' come proxy di investibilita'.
- `growth`: revenue/earnings growth quando disponibili.
- `model_based`: mispricing da DCF, residual income, EVA o altri artifact.

Il factor panel prodotto da `research_platform_core.data_completion` aggiunge
`ret21d`, `ret63d`, `ret126d`, `ret252d`, `vol63d`, `vol126d`, `vol252d`,
`forward_return_21d`, `forward_return_63d`, `forward_return_252d` e gli score
canonici calcolabili dai dati disponibili.

## Feature metadata e glossario UI

Il glossario operativo vive in `src/research_platform_core/feature_metadata.py`.
Questo layer non modifica i calcoli: serve a rendere leggibili in app le colonne
usate da Screener, ML Stock Lab, Valuation e Portfolio.

Ogni entry contiene:

- `id`: nome tecnico della colonna.
- `name`: label leggibile in UI.
- `category`: famiglia (`value`, `quality`, `momentum`, `risk`, `size`,
  `growth`, `model_based`, `ml`, `valuation`, `portfolio`, `target`).
- `description`: cosa misura.
- `formula`: pseudo-formula o definizione operativa.
- `interpretation`: cosa significa un valore alto/basso.

Copertura consolidata:

- `value`: `pe`, `pb`, `ev_ebit`, `ev_ebitda`, `dividend_yield`,
  `valuation_score`, `value_score`.
- `quality`: `roe`, `roic`, `gross_margin`, `operating_margin`,
  `debt_to_equity`, `quality_score`, `quality_flag`.
- `momentum`: `ret21d`, `ret63d`, `ret126d`, `ret252d`,
  `momentum_12_1`, `momentum_12_1_score`, `momentum_score`.
- `risk`: `vol63d`, `vol126d`, `vol252d`, `beta`, `max_drawdown`,
  `risk_score`.
- `size`: `market_value`, `market_cap`, `marketcap`, `log_market_value`,
  `size_score`.
- `growth`: `revenue_growth`, `revenue_cagr`, `sales_cagr`, `eps_growth`,
  `growth_score`.
- `model_based`: `model_mispricing`, `model_mispricing_zscore`,
  `model_mispricing_rank`, `dcf_mispricing`,
  `residual_income_mispricing`, `eva_mispricing`.
- `target/leakage`: `forward_return`, `forward_return_21d`,
  `forward_return_63d`, `forward_return_252d`, `actual`,
  `target_horizon_days`.

Nota importante: le colonne `forward_return_*` e `actual` sono documentate nel
glossario per trasparenza, ma restano target/validation fields. La selezione
feature continua a escluderle tramite la leakage policy.

## Target e leakage policy

Il target expected-return standard e' `forward_return` a 21 trading days.
`train_ml_model_suite(..., target_horizon_days=...)` puo' usare 21, 63 o 252
giorni se il panel contiene le colonne corrispondenti.

La selezione feature passa da `factor_registry.feature_columns_for_blocks()` e
da una denylist anti-leakage. Sono esclusi target, forward returns, prediction,
actual, selected/top-k e score post-modello. Se non esistono feature fattoriali
sufficienti, il runner puo' usare un fallback numerico controllato per smoke
test, ma il model card lo rende visibile.

Nei run bounded `--max-rows`, il loader campiona il panel in chunk lungo tutto
l'artifact invece di usare `head(n)`. Questo mantiene la validazione temporale
2000-2018 / 2019-2026 anche quando il training viene eseguito in modalita'
rapida per smoke test o debug.

## Artifact multi-model

Il training scrive:

- `MLTraining_metrics.csv`: metriche per modello, incluse `r2_os`, `ic`,
  `rank_ic`, `sharpe_long_short`, `sharpe_long_short_net_cost`.
- `MLTraining_predictions.csv`: formato long, una riga per modello/ticker/data.
- `MLTraining_predictions_wide.csv`: formato wide con `score_ols`, `score_rf`,
  ecc. e `score_composite`.
- `MLTraining_model_cards.csv`: metadati modello, feature block, target horizon
  e split.
- `MLStockLab_trained_model_signals.csv`: segnali wide consumabili dallo
  Screener.

## Metrics metadata

Le definizioni delle metriche vivono in
`src/research_platform_core/metrics_metadata.py`. La UI usa questo registro per
spiegare metriche di validazione, portfolio e data quality senza duplicare
testo nelle pagine.

Metriche coperte:

- Validazione modello: `r2_os`, `ic`, `rank_ic`, `hit_ratio`,
  `prediction_rows`, `train_rows`, `test_rows`, `train_date_count`,
  `test_date_count`.
- Portfolio/risk: `sharpe`, `sharpe_long_short`,
  `sharpe_long_short_net_cost`, `sortino`, `volatility`,
  `volatility_long_short`, `max_drawdown`, `cvar`, `turnover`, `alpha`,
  `beta`, `tracking_error`, `information_ratio`.
- Data/model governance: `rows`, `panel_rows`, `ticker_count`,
  `date_count`, `feature_count`, `avg_names`, `avg_names_long_short`.

I test `test_feature_metric_metadata.py` verificano che:

- tutte le colonne dichiarate in `RAW_FACTOR_COLUMNS` abbiano una definizione;
- le colonne reali viste negli artifact ML/Screener/Portfolio principali
  abbiano una definizione o un alias;
- le metriche principali dei training artifacts siano spiegabili in UI.

## Screener semantics

Lo Screener separa ora:

- `ml_score`: score expected-return / composite model.
- `valuation_signal_score`: percentile del mispricing/valuation gap.
- `fair_value_hat`: output valuation, non piu' trattato come ML score.

Questo riduce la collisione concettuale tra modelli di rendimento atteso,
fair-value model e valuation pre-read.

## LLM/Ollama

Il client condiviso e' `research_platform_core.llm_client.OllamaClient`.
Supporta endpoint locale o remoto tramite `OLLAMA_BASE_URL`, `generate_completion`
e `chat`. I prompt di progetto vivono in `research_platform_core.llm_prompts` e
coprono configurazione Screener, spiegazione segnali ML, audit feature set e
memo Data Health.

## Model cycle con Ollama

Ollama entra nel ciclo modelli come assistente di governance, non come motore di
trade. I ranking restano prodotti da fattori, modelli ML, valuation e Smart
Money; il LLM legge output e configurazioni e produce testo ispezionabile.

Funzioni backend:

- `advise_model_configuration()`: propone modelli, feature blocks e horizon per
  uno use case dato, con trade-off interpretabilita'/complessita'.
- `advise_forecast_horizon()`: confronta 21/63/252 trading days rispetto a
  turnover, costi, rumore e stabilita' del segnale.
- `audit_model_governance()`: rivede model cards, metriche, feature importance e
  segnala leakage, baseline mancanti o rischi metodologici.
- `suggest_screener_config()`: traduce una richiesta in linguaggio naturale in
  una bozza JSON di filtri Screener.
- `explain_stock_picks()`: spiega una lista gia' selezionata dal ranking
  quantitativo senza modificarla.

In UI:

- ML Stock Lab espone `LLM Model Advisor` nella tab Models & Settings.
- Screener Builder espone `LLM Screener Assistant` e `Explain current top picks
  with Ollama`.

Policy: ogni output LLM e' research-only, deve dichiarare incertezza e non deve
essere usato come raccomandazione operativa.
