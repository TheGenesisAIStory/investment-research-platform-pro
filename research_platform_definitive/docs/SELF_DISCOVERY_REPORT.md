# Gen.is.IA Self Discovery Report


_Generated from local repository inspection on 2026-05-26._


## 1. Executive Summary


- Files under `research_platform_definitive/`: 14,503 in the artifact-rich source workspace; the integration branch intentionally excludes generated `output/`.
- Core Python modules inspected: 43 top-level `research_platform_core` modules, plus loader subpackage modules.
- Streamlit/app page entries inspected: 16.
- Pytest files inspected: 27 after the v4 academic metadata tests; current suite verification is 112 passed.
- Docs files: 33.
- Output/artifact files: 13,426
- Feature metadata entries: 188
- Metric metadata entries: 79


**State:** v2-ready research workstation after guided merge. Core equity, multi-asset, FF regional, Smart Money v1, ML, valuation, portfolio and Time Series Lab layers are wired; remaining needs-work items are Italy local academic factors, AQR parser robustness, issuer/fund-flow feeds and audited banking fundamentals.


**Top strengths:** shared data platform + local OHLCV root, broad ML/factor/valuation/portfolio content layer, and strong Streamlit product shell with Data Health/selected ticker context.


**Top critical issues:** keep source commits separate from generated outputs, preserve the local-OHLCV/Drive-archive policy, and treat `PARTIAL` domains transparently in Data Health rather than hiding residual data gaps.


## 2. Git Evolution


| Commit | Message | Feature introduced | Current stability |
| --- | --- | --- | --- |
| c5a7e76 | Initial standalone research platform | bootstrap app/core skeleton | stabile base storica |
| 0e2b252 | Finalize research platform operating layer | operational layer, runners, artifacts | stabile ma superata da packaging later |
| 70d9dd0 | Add academic methods and v1 recovery plan | docs/metodi accademici | stabile come riferimento |
| 108139b | Add research workstation data health and ML content layer | data health + ML layer | stabile, poi esteso |
| 71d452e | Add data-centric Data Platform explorer | Data Platform data explorer | stabile |
| 5bbf142 | Add shared selected ticker context | selected_ticker cross-page | stabile |
| e23e2c6 | Refactor Home into research command center | Home command center | stabile, UI iterata |
| d2963dc | Add macro view and sentiment data layer | macro/sentiment | parziale: sentiment/social ancora da rafforzare |
| 0923043 | Add Gen.is.IA branding and metadata glossaries | branding + glossaries | stabile |
| 8a07096 | Expand ML factor and metric metadata coverage | metadata coverage | stabile ma ancora gap formule/paper |
| b479b41 | Finalize project status documentation | status docs | stabile |
| 3043fc3 | Connect forecasting context and market statistics | forecast context + market stats | stabile v1 |
| 58b2f0b | Add time series forecasting lab | TS Lab | stabile v1, ARIMA/ETS/deep planned |
| 71ddae9 | Add multi-asset coverage and factor baselines | multi-asset + baselines | stabile dopo ultimo backfill |
| c10ea16 | Add macro context regime and COT layers | regime + COT | parziale: COT/source ops optional |
| 0fe1c7a | Add multi-asset smart money ML governance layers | smart money + governance | parziale ma usable context layer |
| d1f1a4f | Add Home workflow shortcuts | workflow shortcuts | stabile |
| 95b5752 | Add quant content completeness v3 | portfolio/valuation/features v3 | recente: richiede test/doc gap review |


## 3. Directory Inventory


| Top directory | file count |
| --- | --- |
| . | 8 |
| .pytest_cache | 5 |
| .streamlit | 1 |
| __pycache__ | 1 |
| archive | 431 |
| company_valuation | 81 |
| config | 14 |
| data_api_management | 2 |
| docs | 31 |
| machine_learning_lab | 3 |
| notebooks | 31 |
| output | 13426 |
| portfolio_analysis | 26 |
| research_platform_app | 180 |
| scripts | 45 |
| src | 151 |
| tests | 63 |
| utils | 4 |


## 4. Core Modules — Complete Status Table


| Module | LOC | Purpose | Classes | Functions | UI pages | Tests | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| __init__.py | 422 | Small shared core for notebook-first research platform modules. | 0 | 0 | non esposto direttamente | nessun test diretto | completo |
| api_management.py | 305 | Secret-safe API governance helpers for the research platform. | 1 | 14 | non esposto direttamente | test_data_api_control.py | completo |
| api_orchestrator.py | 429 | Provider fallback, rate limiting and audit logging for data acquisition. | 8 | 25 | non esposto direttamente | test_ohlcv_pipeline.py | parziale/TODO |
| aqr_factors.py | 492 | AQR Data Library factor provider. | 3 | 25 | non esposto direttamente | nessun test diretto | completo |
| banking_data.py | 704 | Open banking data pipeline helpers for Italian and euro-area bank research. | 0 | 26 | non esposto direttamente | nessun test diretto | completo |
| batch_download.py | 289 | Batch export helpers for Drive-first dataset inventories. | 0 | 11 | non esposto direttamente | test_data_api_control.py | completo |
| batch_downloader.py | 52 | Parallel batch downloader with conservative rate limiting. | 1 | 3 | non esposto direttamente | nessun test diretto | completo |
| cache_manager.py | 67 | Small disk cache with TTL metadata for API/data refreshes. | 1 | 6 | non esposto direttamente | test_data_center_enhancement.py | completo |
| data_bridge.py | 70 | Bridge selected datasets from the Data/API platform into ml_stock_lab. | 1 | 3 | non esposto direttamente | test_data_api_control.py | completo |
| data_center_catalog.py | 217 | Canonical target catalog for Data Center coverage. | 0 | 3 | non esposto direttamente | test_data_center_enhancement.py, test_official_macro.py | completo |
| data_completion.py | 731 | End-to-end data completion orchestration for the research workstation. | 2 | 30 | non esposto direttamente | test_data_completion_training.py, test_data_health.py | completo |
| data_explorer.py | 648 | Data-centric exploration helpers for the Streamlit Data Platform. | 0 | 21 | 8_🗄️_Data_Platform.py, home_command_center.py | test_data_explorer.py | completo |
| data_health.py | 885 | Backend data-health helpers for Data Platform and restart workflows. | 2 | 23 | 4_🔍_Screener_Builder.py, 8_🗄️_Data_Platform.py, 9_ML_Stock_Lab.py, home_command_center.py | test_data_health.py, test_ohlcv_pipeline.py | completo |
| data_platform.py | 779 | Drive-first data platform helpers for the research notebooks and app. | 2 | 30 | 8_🗄️_Data_Platform.py | test_context_bootstrap.py, test_data_health.py | completo |
| data_utils.py | 69 | Shared dataframe and alias helpers for notebook-first platform modules. | 0 | 5 | non esposto direttamente | nessun test diretto | completo |
| equity_feature_engineering.py | 255 | Advanced equity feature engineering for optional factor blocks. | 0 | 9 | non esposto direttamente | nessun test diretto | completo |
| export_utils.py | 52 | Shared export helpers with notebook-safe fallbacks. | 0 | 3 | non esposto direttamente | nessun test diretto | completo |
| factor_benchmarks.py | 247 | Factor benchmark portfolios for ML Stock Lab governance. | 0 | 9 | 9_ML_Stock_Lab.py | test_factor_benchmarks.py | parziale/TODO |
| factor_portfolio_baselines.py | 291 | Monthly factor portfolio baselines for ML model governance. | 0 | 11 | 9_ML_Stock_Lab.py | test_multi_asset_smart_money_ml_v2.py | completo |
| feature_metadata.py | 720 | Readable feature and factor metadata for the Gen.is.IA workstation. | 1 | 6 | non esposto direttamente | test_feature_metric_metadata.py, test_official_macro.py | completo |
| html_utils.py | 24 | Small HTML helpers shared by notebook dashboard hooks. | 0 | 1 | non esposto direttamente | nessun test diretto | completo |
| llm_advisors.py | 164 | LLM advisor helpers for model selection, stock-pick explanation and audit. | 0 | 7 | 4_🔍_Screener_Builder.py, 9_ML_Stock_Lab.py | test_llm_client.py | completo |
| llm_client.py | 115 | Small Ollama client for local or private-cloud research assistance. | 2 | 4 | non esposto direttamente | test_llm_client.py | completo |
| llm_prompts.py | 124 | Versioned prompt templates for LLM-assisted investment research. | 0 | 0 | non esposto direttamente | test_llm_client.py | completo |
| __init__.py | 39 | Data acquisition loaders for the Data/API Control Center. | 0 | 0 | non esposto direttamente | nessun test diretto | completo |
| aqr_data.py | 39 | AQR loader wrapper for the Data Center enhancement. | 1 | 4 | non esposto direttamente | nessun test diretto | completo |
| bditalia_client.py | 459 | Banca d'Italia BDS/Infostat public export client. | 2 | 18 | non esposto direttamente | test_official_macro.py | completo |
| ecb_client.py | 345 | ECB Data Portal SDMX 2.1 REST client. | 2 | 15 | non esposto direttamente | test_official_macro.py | completo |
| equity_universe.py | 268 | Global equity universe and fundamentals loaders. | 1 | 16 | non esposto direttamente | test_data_center_enhancement.py | completo |
| fama_french.py | 179 | Fama-French Data Library loader. | 2 | 8 | non esposto direttamente | test_data_center_enhancement.py | completo |
| fx_commodities.py | 125 | FX and commodities loaders using Drive-first storage and yfinance fallback. | 1 | 7 | non esposto direttamente | nessun test diretto | completo |
| kaggle_seed_loader.py | 500 | Import local Kaggle market-data dumps as static OHLCV seeds. | 2 | 20 | non esposto direttamente | test_ohlcv_pipeline.py | completo |
| market_universe.py | 294 | Market universe discovery for broad OHLCV ingestion. | 1 | 12 | non esposto direttamente | nessun test diretto | completo |
| ohlcv_client.py | 752 | OHLCV provider client built on the shared APIOrchestrator. | 5 | 37 | non esposto direttamente | test_ohlcv_pipeline.py | parziale/TODO |
| risk_factors.py | 131 | Risk factor loaders for credit, rates and volatility proxies. | 1 | 5 | non esposto direttamente | test_data_center_enhancement.py | completo |
| macro_context.py | 198 | Macro context features for equity ML models. | 0 | 6 | non esposto direttamente | test_macro_context_regime.py | completo |
| macro_features.py | 259 | Macro feature engineering for ECB and Banca d'Italia official data. | 0 | 11 | non esposto direttamente | test_official_macro.py | parziale/TODO |
| macro_market.py | 329 | Macro market metadata, snapshots and lightweight database compiler. | 1 | 8 | 13_🌍_Macro_View.py | test_macro_context_regime.py, test_macro_sentiment_layer.py, test_multi_asset_smart_money_manifests.py | completo |
| market_statistics.py | 150 | Basic market statistics for ticker, screener and portfolio views. | 0 | 5 | non esposto direttamente | test_market_statistics.py | completo |
| metrics_metadata.py | 481 | Readable metric metadata for model, portfolio and valuation dashboards. | 1 | 5 | non esposto direttamente | test_feature_metric_metadata.py | completo |
| model_monitoring.py | 143 | Model monitoring artifacts for ML Stock Lab. | 0 | 5 | 9_ML_Stock_Lab.py | test_multi_asset_smart_money_ml_v2.py | completo |
| multi_asset_universe.py | 499 | Multi-asset universe and coverage manifest helpers. | 0 | 15 | 8_🗄️_Data_Platform.py | test_multi_asset_smart_money_manifests.py, test_multi_asset_smart_money_ml_v2.py | completo |
| ohlcv_ingest.py | 684 | Historical and incremental OHLCV ingestion jobs. | 1 | 20 | non esposto direttamente | test_ohlcv_pipeline.py | parziale/TODO |
| ohlcv_store.py | 360 | SQLite/Postgres-compatible OHLCV storage helpers. | 1 | 10 | non esposto direttamente | test_ohlcv_pipeline.py | completo |
| portfolio_analytics.py | 376 | Portfolio analytics and construction helpers for Gen.is.IA. | 0 | 31 | non esposto direttamente | nessun test diretto | completo |
| regime_detection.py | 130 | Simple, explainable market regime detection from Macro DB features. | 0 | 4 | home_command_center.py | test_macro_context_regime.py | parziale/TODO |
| run_lock.py | 167 | Stage-level lock files for long-running data jobs. | 2 | 9 | 8_🗄️_Data_Platform.py | test_data_health.py, test_run_lock.py | parziale/TODO |
| sentiment_analysis.py | 211 | Lightweight social sentiment connectors and scoring utilities. | 0 | 7 | 13_🌍_Macro_View.py | test_macro_sentiment_layer.py | completo |
| smart_money.py | 625 | Smart Money source catalog and lightweight coverage manifests. | 1 | 16 | 1_📡_Smart_Money_Macro.py, 8_🗄️_Data_Platform.py | test_context_bootstrap.py, test_cot_smart_money.py, test_data_completion_training.py | completo |
| storage_policy.py | 120 | Project storage policy shared by notebooks, Streamlit and scripts. | 1 | 5 | non esposto direttamente | nessun test diretto | completo |
| time_series_forecasting.py | 596 | Time-series forecasting helpers for the Gen.is.IA research workstation. | 2 | 17 | non esposto direttamente | test_time_series_forecasting.py | parziale/TODO |
| valuation_analytics.py | 246 | Valuation analytics helpers for DCF, WACC, multiples, EVA/DDM and comps. | 0 | 10 | non esposto direttamente | nessun test diretto | completo |


## 5. Streamlit Pages / App Entries


| Page | Core imports | st.tabs calls | Session state keys | TODO/placeholder markers |
| --- | --- | --- | --- | --- |
| app.py | - | 0 | - | 0 |
| home_command_center.py | data_explorer, data_health, regime_detection | 1 | get, active_screener_name, home_focus, selected_ticker | 10 |
| 0_🏠_Home.py | - | 0 | - | 0 |
| 10_Laboratorio_Research_Library.py | - | 0 | - | 1 |
| 11_Data_API_Control_Center.py | - | 0 | - | 0 |
| 12_Banking_Data_Lab.py | core | 1 | selected_ticker | 1 |
| 13_🌍_Macro_View.py | core, macro_market, sentiment_analysis | 2 | get, ts_lab_source, ts_lab_symbol | 8 |
| 14_⏱️_Time_Series_Lab.py | core | 1 | get, selected_ticker, ts_lab_source, ts_lab_symbol | 2 |
| 1_📡_Smart_Money_Macro.py | smart_money | 0 | get, selected_ticker | 3 |
| 2_🔬_Valuation_Research.py | - | 2 | get, selected_ticker | 0 |
| 3_📁_Portfolio_Research.py | core | 1 | get, ts_lab_source, ts_lab_symbol | 0 |
| 4_🔍_Screener_Builder.py | core, data_health, llm_advisors | 1 | get, active_explain_tab, active_screener_config, active_screener_name, last_screening_result_count, preferred_screening_columns, selected_ticker | 0 |
| 5_📤_Export_Center.py | - | 1 | - | 0 |
| 6_🧪_Notebook_Runner.py | - | 0 | - | 0 |
| 7_⚙️_Run_Hi_Freq_Engine.py | - | 0 | - | 1 |
| 8_🗄️_Data_Platform.py | data_explorer, data_health, data_platform, multi_asset_universe, run_lock, smart_money | 2 | get, data_platform_preview_request, data_platform_ticker, data_retry_tickers, last_data_restart_run, selected_ticker | 8 |
| 9_ML_Stock_Lab.py | data_health, factor_benchmarks, factor_portfolio_baselines, llm_advisors, model_monitoring | 0 | get, selected_ticker | 0 |


## 6. Test Inventory


| Test file | n_tests | Core modules referenced | Coverage type |
| --- | --- | --- | --- |
| test_context_bootstrap.py | 7 | data_platform, smart_money | unit/integration smoke |
| test_cot_smart_money.py | 2 | smart_money | unit/integration smoke |
| test_data_api_control.py | 3 | api_management, batch_download, data_bridge | unit/integration smoke |
| test_data_center_enhancement.py | 3 | cache_manager, data_center_catalog, equity_universe, fama_french, risk_factors | unit/integration smoke |
| test_data_completion_training.py | 3 | data_completion, smart_money | unit/integration smoke |
| test_data_explorer.py | 4 | data_explorer | unit/integration smoke |
| test_data_health.py | 12 | data_completion, data_health, data_platform, run_lock | unit/integration smoke |
| test_factor_benchmarks.py | 1 | factor_benchmarks | unit/integration smoke |
| test_feature_metric_metadata.py | 6 | feature_metadata, metrics_metadata | unit/integration smoke |
| test_llm_client.py | 3 | llm_advisors, llm_client, llm_prompts | unit/integration smoke |
| test_macro_context_regime.py | 2 | macro_context, macro_market, regime_detection | unit/integration smoke |
| test_macro_sentiment_layer.py | 4 | macro_market, sentiment_analysis | unit/integration smoke |
| test_market_statistics.py | 1 | market_statistics | unit/integration smoke |
| test_multi_asset_smart_money_manifests.py | 2 | macro_market, multi_asset_universe, smart_money | unit/integration smoke |
| test_multi_asset_smart_money_ml_v2.py | 4 | factor_portfolio_baselines, model_monitoring, multi_asset_universe, smart_money | unit/integration smoke |
| test_official_macro.py | 4 | bditalia_client, data_center_catalog, ecb_client, feature_metadata, macro_features | unit/integration smoke |
| test_ohlcv_pipeline.py | 14 | api_orchestrator, data_health, kaggle_seed_loader, ohlcv_client, ohlcv_ingest, ohlcv_store | unit/integration smoke |
| test_run_lock.py | 2 | run_lock | unit/integration smoke |
| test_screener_workbench.py | 4 | smart_money | unit/integration smoke |
| test_time_series_forecasting.py | 4 | macro_market, time_series_forecasting | unit/integration smoke |


**Core modules without direct test references:** __init__.py, aqr_data.py, aqr_factors.py, banking_data.py, batch_downloader.py, data_utils.py, equity_feature_engineering.py, export_utils.py, fx_commodities.py, html_utils.py, market_universe.py, portfolio_analytics.py, storage_policy.py, valuation_analytics.py



## 7. Docs and Config


| Doc | Bytes | Heading/first line |
| --- | --- | --- |
| API_USAGE.md | 1624 | # API Usage |
| AQR_FACTOR_LIBRARY.md | 1915 | # AQR Factor Library Integration |
| BANKING_DATA_PIPELINE.md | 2108 | # Banking Data Pipeline |
| CODEX_CANONICAL_MAINTENANCE_GUIDE.md | 12325 | # Codex Canonical Maintenance Guide |
| COLAB_LOCAL_DRIVE_WORKFLOW.md | 2901 | # Colab, Local and Google Drive Workflow |
| DATA_BOOTSTRAP.md | 3983 | # Data Bootstrap |
| DATA_CATALOG.md | 1608 | # Data Catalog |
| DATA_COMPLETION_2000_2026.md | 11967 | # Data Completion 2000-2026 |
| DATA_PLATFORM_INTEGRATION.md | 6144 | # Data Platform Integration |
| DCF_MONTE_CARLO_EXTENSION.md | 2619 | # Model-Based Valuation Uncertainty Extension |
| DEV_SETUP.md | 2386 | # Developer Setup |
| KAGGLE_SEED_INTEGRATION.md | 2262 | # Kaggle Seed Integration |
| MACRO_CONTEXT_LAYER.md | 2160 | # Macro Context Layer |
| MACRO_SENTIMENT_DATA_LAYER.md | 1493 | # Macro & Sentiment Data Layer |
| MAINTENANCE.md | 1114 | # Maintenance |
| ML_CONTENT_LAYER_V2.md | 12026 | # ML Content Layer v2 |
| ML_STOCK_LAB_OVERVIEW.md | 3272 | # ML Stock Lab |
| MULTI_ASSET_SMART_MONEY_ML_V2.md | 3215 | # Gen.is.IA Multi-Asset, Smart Money and ML Governance v2 |
| OFFICIAL_MACRO_DATA.md | 2504 | # Official Macro Data Integration |
| OHLCV_API_ORCHESTRATION.md | 3787 | # OHLCV API Orchestration |
| OHLCV_PRICE_PIPELINE.md | 3461 | # Global OHLCV Price Pipeline |
| PROJECT_REVIEW_TOTAL_2026-05-25.md | 8173 | # Project Review Totale - 2026-05-25 |
| PROJECT_STATUS_2026-05-23.md | 9912 | # Stato progetto - 2026-05-23 |
| PROJECT_STATUS_FINAL.md | 4245 | # Project Status Final - Gen.is.IA Investment Research Workstation |
| RESEARCH_PLATFORM_DEFINITIVE_REVIEW.md | 6502 | # Research Platform Definitive Review |
| RESEARCH_PLATFORM_INDEX.md | 1476 | # Research Platform Index |
| RESEARCH_PLATFORM_ORCHESTRATION.md | 3176 | # Research Platform Orchestration |
| SCREENER_BUILDER_WORKSTATION.md | 3045 | # Screener Builder Workstation |
| SMART_MONEY_GOVERNMENT_DATA_ENGINE.md | 4396 | # Smart Money Government Data Engine |
| STREAMLIT_WORKSTATION_V1.md | 3419 | # Streamlit Workstation v1 |
| TIME_SERIES_FORECASTING_LAYER.md | 6592 | # Gen.is.IA Time Series Forecasting Layer |




| Config file | Bytes |
| --- | --- |
| __init__.py | 0 |
| core.py | 2322 |
| data_api_control.yaml | 1062 |
| data_sources.yaml | 3248 |
| official_macro_sources.yaml | 1360 |
| ohlcv_data_sources.yaml | 4746 |
| presets.py | 1805 |
| rate_limits.yaml | 632 |
| scheduler_config.yaml | 1037 |
| settings.py | 2149 |


## 8. Output Manifest Reality Check


| Manifest/status artifact | Rows | Key counts | Bytes |
| --- | --- | --- | --- |
| output/PLATFORM_STATUS.json |  | multi_asset 140/140 OK; FF US/EU/JP/APAC/EM/World OK; IT local PARTIAL |  |
| output/data_completion/ResearchDataCoverageValidation.csv | 11 | {'OK': 11} in operational Drive mirror |  |
| output/macro_market/tables/MacroAssetManifest.csv | 95 | {'OK': 95} | 28357 |
| output/macro_market/tables/MultiAssetUniverseManifest.csv | 95 | {'OK': 95} legacy macro snapshot | 29828 |
| output/ml_stock_lab/MLStockLab_status.json |  |  | 111 |
| output/multi_asset_universe/MultiAssetUniverseManifest.csv | 140 | {'OK': 140} after incremental sync |  |
| output/ff_factors/FFRegionalManifest.csv | 7 | {'OK': 6, 'PARTIAL': 1} |  |
| output/review/Artifact_Manifest.csv | 38 |  | 5108 |
| output/smart_money/tables/SmartMoneySourceManifest.csv | 5 | {'READY_OPTIONAL': 2, 'PLANNED': 2, 'PARTIAL': 1} | 1705 |
| output/smart_money/tables/SmartMoney_ingestion_manifest.csv | 8 | {'MISSING': 4, 'EMPTY_OR_UNREADABLE': 4} | 982 |
| output/tables/Kaggle_seed_manifest.csv | 7 | {'missing_root': 7} | 1216 |
| output/tables/OHLCV_daily_manifest.csv | 261 | {'downloaded': 164, 'limited_history': 97} | 136764 |


## 9. Cross-Reference Findings


- Metadata features referenced in calculation/registry files: 87 / 109


- Sample metadata entries not observed in calculation files: none


- Sample calculation/registry tokens without metadata match: OfficialMacro, accruals_ratio, altman_z_score, asset_growth, avg_volume_21d, avg_volume_252d, bdi_public_debt, bditalia_public_debt, calendar_policy, cash_ratio, credit_growth, current_ratio, deposit_volumes, downside_vol_21d, earnings_yield, ebitda_margin, ebitda_yield, ecb_hicp, ecb_policy_rates, equity_prices, equity_prices_2000_2026, fcf_margin, fcf_yield, forward_return_, growth, hicp_euro_area_yoy, ic_mean, idiosyncratic_momentum, idiosyncratic_vol, max_drawdown_1y, metrics, missing_policy, momentum, momentum_12m_1m, momentum_reversal_1m, net_margin, official_macro_features, official_rates_bdi, ohlcv_critical_provider_failures, piotroski_f_score, policy_rate_deposit, policy_rate_mro, price_to_52w_high, price_to_sma_200, price_to_sma_50, public_debt, quick_ratio, realized_vol_daily, ret1d, ret_1d, ret_21d, ret_252d, ret_52w_high_proximity, ret_52w_low_proximity, retained_earnings, return, returns, sales_to_price, sharpe, strict


- Modules with no direct UI exposure include: __init__, api_management, api_orchestrator, aqr_factors, banking_data, batch_download, batch_downloader, cache_manager, data_bridge, data_center_catalog, data_completion, data_utils, equity_feature_engineering, export_utils, feature_metadata, html_utils, llm_client, llm_prompts, macro_context, macro_features, market_statistics, metrics_metadata, ohlcv_ingest, ohlcv_store, portfolio_analytics, storage_policy, time_series_forecasting, valuation_analytics, __init__, aqr_data, bditalia_client, ecb_client, equity_universe, fama_french, fx_commodities, kaggle_seed_loader, market_universe, ohlcv_client, risk_factors



## 10. Recommended Next Work


1. Normalize review/implementation on branch `codex/research-platform-v2-ml-ollama` or consciously merge the current workstation branch; do not mix generated output with source commits.
2. Fix Asia single-name universes: Nikkei/Hang Seng parsers currently need stricter table selection before single-name backfill.
3. Add academic factor regional loaders (FF US/EU/JP/APAC/EM/World; Italy local proxy) and expose them in Academic Reference.
4. Convert feature metadata audit into a failing-but-actionable quality gate after metadata is complete.
5. Expand UI exposure for implemented but hidden modules: portfolio optimizers, valuation residual/DDM/comps, FF/AQR browser, model monitoring.
