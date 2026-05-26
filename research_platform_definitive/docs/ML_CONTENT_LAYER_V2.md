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
- `fx_factors`, `fi_factors`, `commodity_factors`: blocchi sperimentali
  Bartram-style costruiti dal Macro DB 140 proxy.
- `smart_money_factors`: COT hedging pressure e positioning z-score da Smart
  Money/CFTC.
- `cross_asset_momentum`: momentum-everywhere su equity, FX, FI, commodity e
  crypto proxy.

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
  `net_payout_yield`, `valuation_score`, `value_score`.
- `quality`: `roe`, `roic`, `gross_margin`, `operating_margin`,
  `debt_to_equity`, `gross_profitability`, `investment_factor`, `accruals`,
  `cash_profitability`, `earnings_quality`, `quality_score`, `quality_flag`.
- `momentum`: `ret21d`, `ret63d`, `ret126d`, `ret252d`,
  `momentum_12_1`, `short_term_reversal`, `momentum_12_1_score`,
  `momentum_score`.
- `risk`: `vol63d`, `vol126d`, `vol252d`, `beta`, `max_drawdown`,
  `idiosyncratic_vol`, `distress_risk`, `risk_score`.
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

## Factor baseline portfolios

Per evitare di confrontare i modelli ML contro il vuoto, il core include ora
`research_platform_core.factor_benchmarks`. Il modulo legge il
`FactorUniversePanel` e produce benchmark trasparenti per:

- `value_score`
- `quality_score`
- `momentum_score`
- `risk_score`
- `size_score`
- `growth_score`
- `factor_composite_score`

Per ciascun fattore calcola un bucket top, un bucket bottom e lo spread
top-minus-bottom rispetto a `forward_return_21d`/`63d`/`252d`. Le metriche
principali sono `top_mean_return`, `long_short_mean_return`,
`long_short_sharpe`, `top_long_sharpe` e `rank_ic_mean`.

Gli artifact sono:

- `output/ml_training_lab/tables/FactorBenchmarkSummary.csv`
- `output/ml_training_lab/FactorBenchmarkManifest.json`

In UI, ML Stock Lab espone la tab `Factor Baselines`: il ricercatore puo'
calcolare una versione interattiva campionata e confrontare immediatamente ML vs
fattori tradizionali. Per run full-panel/scheduled, il limite `max_rows` puo'
essere disattivato lato job.

## Macro context feature block

Il Macro DB multi-asset alimenta ora piu' blocchi sperimentali opzionali:

```text
macro_context
alpha101
fx_factors
fi_factors
commodity_factors
smart_money_factors
cross_asset_momentum
```

`macro_context` vive in `ml_stock_lab.factor_registry` ma le feature sono
costruite in `research_platform_core.macro_context` e `ml_stock_lab.macro_features`.
Le colonne principali includono momentum macro su SPY/DXY/Brent/WTI/TLT,
credit spread proxy `HYG - TLT`, curve proxy `TNX - IRX`, VIX percentile
e `macro_regime_encoded`.

Regola anti-leakage: il `MacroContextPanel.csv` e' laggato di una osservazione
prima dell'as-of join con il factor panel equity. Il blocco non entra nei
modelli di default; va selezionato esplicitamente in ML Stock Lab per testare
se migliora IC/RankIC/Sharpe rispetto ai fattori equity core.

`alpha101` espone le 101 formule WorldQuant/Kakushadze (2016,
arXiv:1601.00991), calcolate da OHLCV come segnali cross-sectionally ranked.
Anche questo blocco e' disattivato di default: serve per confrontare la
famiglia di alpha formulaici con i fattori accademici canonici, non per
sostituirli.

| Block | Status | N feature | Source | Default |
| --- | --- | ---: | --- | --- |
| `macro_context` | experimental | 9 core + alias legacy | Macro DB 140 proxy | off |
| `alpha101` | experimental | 101 | Kakushadze (2016), arXiv:1601.00991 | off |
| `fx_factors` | experimental | 44 | Bartram et al. (2021), Menkhoff et al. (2012) | off |
| `fi_factors` | experimental | 4 | Fama-Bliss (1987), Elton et al. (2001), AMP (2013) | off |
| `commodity_factors` | experimental | 16 | Gorton-Rouwenhorst (2006), Bartram et al. (2021) | off |
| `smart_money_factors` | experimental | 18 | De Roon et al. (2000), CFTC COT | off |
| `cross_asset_momentum` | experimental | 30 | Asness, Moskowitz and Pedersen (2013) | off |

Il nuovo set Bartram/AMP e' implementato come layer additivo: non sovrascrive
feature gia' migliori nel factor panel, ma registra colonne opzionali e builder
indipendenti (`institutional_factors.py`, `fx_factors.py`, `fi_factors.py`,
`commodity_factors.py`, `cross_asset_factors.py`, `smart_money.py`). Tutti i
builder usano almeno uno shift di una osservazione; le feature fondamentali
richiedono lag fiscale prima dell'uso live.

Risultati retraining Alpha101: gli artifact `_alpha101` vengono prodotti da
`scripts/train_ml_models_2000_2026.py --use-alpha101 True`. Il run locale del
2026-05-26 e' completato con 4/4 modelli `OK`, split temporale 2000-2018 /
2019-2026, target 21d e 121 feature complessive. Poiche' il factor panel storico
contiene `price` e `market_value` ma non OHLCV completo, il retraining usa il
fallback proxy rank-based per il blocco Alpha101; il modulo
`research_platform_core.alpha101` resta la fonte completa delle 101 formule
WorldQuant quando un panel OHLCV e' disponibile.

| Model | IC | RankIC | Net LS Sharpe | Fit rows | Prediction rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| OLS | 0.0175 | -0.0451 | 0.6208 | 137,780 | 137,220 |
| RF | -0.0102 | -0.0085 | 0.6199 | 15,000 | 137,220 |
| GBRT | -0.0117 | -0.0185 | 0.5966 | 15,000 | 137,220 |
| Ensemble | -0.0057 | -0.0272 | 0.6210 | 15,000 | 137,220 |

Gli artifact completi sono disponibili localmente in
`output/ml_training_lab/*_alpha101.*` e
`output/ml_stock_lab/tables/*_alpha101.csv`; i file panel/prediction pesanti non
sono pensati per essere versionati in Git.

La metodologia e' documentata in `docs/MACRO_CONTEXT_LAYER.md`.

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
- Time-series forecasting: `mae`, `rmse`, `mape`,
  `directional_accuracy`, `forecast_return`, `forecast_level`,
  `forecast_error_std`.
- Basic market statistics: `annualized_volatility`, `annualized_variance`,
  `beta_to_benchmark`, `correlation_to_benchmark`, `avg_pairwise_corr`.
- Factor benchmark: `top_mean_return`, `bottom_mean_return`,
  `long_short_mean_return`, `top_long_sharpe`, `long_short_sharpe`.

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

## Time Series Forecasting Layer

Il layer cross-sectionale rimane il cuore dello stock picking: ordina molti
titoli usando target come `forward_return_21d`, `forward_return_63d` e
`forward_return_252d`. Per coprire anche il forecasting classico di una singola
serie, e' stato aggiunto `research_platform_core.time_series_forecasting`.

La pagina LABS `Time Series Lab` permette di:

- scegliere una serie da Macro DB o OHLCV equity;
- stimare forecast su 5/21/63/126 giorni;
- confrontare baseline `naive`, `ols` e `gbrt`;
- salvare metriche, predizioni, forecast latest e feature schema sotto
  `output/time_series_lab/tables`.

La metodologia e' documentata in `docs/TIME_SERIES_FORECASTING_LAYER.md`.
Questo modulo e' pensato come contesto di scenario per Macro View e Portfolio,
non come sostituto dei ranking cross-sectionali dello Stock Lab.

## Multi-asset and Smart Money coverage

La parte non-equity e' ora resa esplicita da due manifest leggeri:

- `research_platform_core.multi_asset_universe` compila
  `MultiAssetUniverseManifest.csv` da Macro DB e puo' anche fare ingestion
  dedicata in `output/multi_asset_universe/ohlcv`, distinguendo FX,
  commodities, crypto, ETF equity, ETF fixed income, ETF commodity e
  fixed-income proxies.
- `research_platform_core.smart_money` compila `SmartMoneySourceManifest.csv`
  e `SmartMoneyAssetCatalog.csv` per CFTC COT, ETF flows proxy, options PCR e
  issuer-event evidence.

Data Platform mostra questi manifest in `Domain Status`; Macro View usa lo
stesso catalogo per far vedere asset scaricati e asset pianificati. Le fonti
Smart Money non ancora disponibili non sono mostrate come pannelli vuoti:
restano `PLANNED` o `READY_OPTIONAL` finche' un job/fonte reale non le popola.

Home espone ora due workflow trasversali:

- **Multi-Asset Context**: Data Platform -> Macro View -> Time Series Lab.
- **Smart Money Check**: Smart Money -> Macro View -> Portfolio Context.

## Factor baselines and monitoring

Il ML Stock Lab ha due layer di governance quantitativa:

- `research_platform_core.factor_portfolio_baselines` calcola portafogli
  baseline mensili long-only top decile e long-short top/bottom per value,
  quality e momentum, salvando metriche in
  `output/factor_baselines/baseline_portfolio_metrics.csv`.
- `research_platform_core.model_monitoring` calcola IC/RankIC rolling per
  modello da `MLTraining_predictions.csv`, salvando gli artifact in
  `output/ml_lab/model_monitoring`.

Dettaglio operativo: `docs/MULTI_ASSET_SMART_MONEY_ML_V2.md`.
