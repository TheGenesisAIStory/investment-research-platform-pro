# Full Project Review v4 — Universe Expansion + Academic Feature Documentation


_Generated after self-discovery; source state: local workspace plus latest output manifests._


## 0.1 Module Coverage Map: Core -> UI


| Modulo | Pagine UI che lo usano | Funzioni gap (non esposte, sample) |
| --- | --- | --- |
| __init__.py | - | - |
| api_management.py | - | discover_api_credentials_root, resolve_api_control_roots, mask_secret, load_api_folder_inventory, credential_master_sections, accepted_api_env_vars, parse_env_text, apply_env_text_to_session, build_credential_status, api_control_status, bui |
| api_orchestrator.py | - | can_call, record_call, remaining_day_budget, from_config, capable, rank_providers, provider_order, reserve, wait, fetch_with_fallback |
| aqr_factors.py | - | slugify, discover_aqr_dataset_pages, discover_aqr_datasets, parse_aqr_excel, get_aqr_factor_panel, get_all_factors_panel, refresh_aqr_factor_library, handle_starttag, handle_data, handle_endtag, catalog_path, discover |
| banking_data.py | - | utc_now, slugify, normalize_bank_name, cached_get, fetch_wikipedia_italian_banks, download_ecb_supervised_entities_pdf, parse_ecb_supervised_entities_pdf, ecb_get_series, build_ecb_bsi_macro_panel, load_bancaditalia_bds_file, fetch_bancadit |
| batch_download.py | - | slugify, dataset_id, infer_provider, enrich_inventory_for_export, estimate_batch_size, create_batch_download |
| batch_downloader.py | - | batch_download |
| cache_manager.py | - | stable_cache_key, get_or_fetch |
| data_bridge.py | - | discover_ml_stock_lab_notebooks, push_manifest |
| data_center_catalog.py | - | build_target_catalog, summarize_target_catalog |
| data_completion.py | - | data_source_map, run_full_data_completion, validate_completion_coverage, write_plan, write_coverage_report, run_all, run_stage |
| data_explorer.py | 8_🗄️_Data_Platform.py, home_command_center.py | find_ohlcv_parquet, load_ticker_ohlcv, load_factor_rows_for_ticker, load_ml_signal_rows_for_ticker, load_fundamental_rows_for_ticker, load_smart_money_rows_for_ticker, load_company_rows_for_ticker, load_portfolio_rows_for_ticker, get_ticker |
| data_health.py | 4_🔍_Screener_Builder.py, 8_🗄️_Data_Platform.py, 9_ML_Stock_Lab.py, home_command_center.py | restart_equity_prices_failed_only |
| data_platform.py | 8_🗄️_Data_Platform.py | utc_now, is_cloud_backed_path, get_ohlcv_parquet_root_info, get_ohlcv_parquet_root, get_ohlcv_daily_search_roots, discover_financial_database_root, infer_dataset_role, build_dataset_inventory, summarize_inventory, should_refresh, resolve_da |
| data_utils.py | - | as_df, first_available, normalize_name, resolve_alias |
| equity_feature_engineering.py | - | compute_advanced_technical_features, add_idiosyncratic_features, compute_advanced_fundamental_features, compute_piotroski_f_score, add_advanced_equity_features, residual_std |
| export_utils.py | - | output_root_from_namespace, safe_write_csv, safe_write_json |
| factor_benchmarks.py | 9_ML_Stock_Lab.py | discover_factor_panel_path, write_factor_benchmark_artifacts |
| factor_portfolio_baselines.py | 9_ML_Stock_Lab.py | write_factor_portfolio_baselines |
| feature_metadata.py | - | normalize_feature_id, metadata_for_feature, metadata_help, metadata_frame, category_for_feature |
| html_utils.py | - | table_html |
| llm_advisors.py | 4_🔍_Screener_Builder.py, 9_ML_Stock_Lab.py | - |
| llm_client.py | - | generate_completion, chat |
| llm_prompts.py | - | - |
| __init__.py | - | - |
| aqr_data.py | - | download_all, get_dataset |
| bditalia_client.py | - | normalize_bditalia_frame, specs, build_url, download_export, get_dataset, get_preset, get_credit_series, get_deposits_series, get_public_debt_series, release_frequency, save_dataset, sync_presets |
| ecb_client.py | - | normalize_ecb_frame, parse_sdmx_json, specs, build_url, get_series, get_preset, get_hicp_euro_area, get_policy_rates, get_mfi_balance_sheet, save_series, sync_presets |
| equity_universe.py | - | universe_catalog, get_current_constituents, write_constituents, sync_prices, sync_fundamentals |
| fama_french.py | - | parse_fama_french_csv, specs, download_dataset, download_all |
| fx_commodities.py | - | normalize_yfinance_frame, sync_fx, sync_commodities |
| kaggle_seed_loader.py | - | import_kaggle_seeds, discover_files, import_dataset, import_seeds |
| market_universe.py | - | fetch_us_listings, load_manual_exchange_files, fetch_index_universes, build_universe |
| ohlcv_client.py | - | normalize_ohlcv_frame, parse_yfinance_bulk, classify_provider_error, classify_history_window, is_variant_sensitive_symbol, is_low_priority_structured_symbol, provider_symbol_variants, merge_ohlcv_frames, get_ohlcv_daily, get_ohlcv_daily_bul |
| risk_factors.py | - | sync_fred, sync_volatility, build_derived_risk_factors |
| macro_context.py | - | load_macro_close_panel, build_macro_context_panel, load_macro_context_panel, add_macro_context_features |
| macro_features.py | - | normalize_datetime_column, to_month_end, series_to_wide, align_macro_frames, add_standard_transformations, build_inflation_nowcasting_dataset, build_credit_risk_macro_dataset, save_macro_feature_panel |
| macro_market.py | 13_🌍_Macro_View.py | ret |
| market_statistics.py | - | load_price_return_matrix, compute_ticker_market_statistics |
| metrics_metadata.py | - | normalize_metric_id, metadata_for_metric, metric_help, metrics_metadata_frame |
| model_monitoring.py | 9_ML_Stock_Lab.py | discover_prediction_path |
| multi_asset_universe.py | 8_🗄️_Data_Platform.py | multi_asset_universe_catalog, compile_multi_asset_universe_manifest, download_fn |
| ohlcv_ingest.py | - | validate_provider_symbol, summarize_ohlcv_manifest, build_assets, run_daily, run_intraday_5m |
| ohlcv_store.py | - | default_database_url, connect, init_schema, upsert_assets, asset_id_map, latest_daily_dates, latest_daily_dates_by_provider_symbol, upsert_daily_prices, upsert_intraday_5m |
| portfolio_analytics.py | - | drawdown_series, drawdown_durations, sharpe_ratio, sortino_ratio, calmar_ratio, omega_ratio, kappa_ratio, treynor_ratio, jensen_alpha, m2_measure, value_at_risk, conditional_var |
| regime_detection.py | home_command_center.py | classify_market_regime, build_market_regime_history, load_market_regime_history |
| run_lock.py | 8_🗄️_Data_Platform.py | acquire_stage_lock, release_stage_lock, list_stage_locks, lock_payload |
| sentiment_analysis.py | 13_🌍_Macro_View.py | score_text, score_sentiment_frame, fetch_stocktwits_messages, fetch_reddit_mentions, fetch_x_recent_mentions |
| smart_money.py | 1_📡_Smart_Money_Macro.py, 8_🗄️_Data_Platform.py | smart_money_source_catalog, normalize_cot_data, refresh_cftc_cot_snapshot |
| storage_policy.py | - | has_project_sentinel, first_existing, resolve_project_root, resolve_storage_roots, export_storage_env |
| time_series_forecasting.py | - | prepare_time_series_frame, make_time_series_features, fit_time_series_forecasts, write_time_series_forecast_artifacts |
| valuation_analytics.py | - | compute_wacc, compute_dcf_valuation, compute_market_multiples, compute_eva_residual_income, compute_ddm_valuation, compute_asset_based_valuation, compute_comps_valuation, fair_value_summary |


## 0.2 Equity Feature Inventory


| feature_name | category | sub_category | implementata? | documentata? | nel factor panel? | paper/source |
| --- | --- | --- | --- | --- | --- | --- |


**Orphan calculation-token sample:** OfficialMacro, accruals_ratio, altman_z_score, asset_growth, avg_volume_21d, avg_volume_252d, bdi_public_debt, bditalia_public_debt, calendar_policy, cash_ratio, credit_growth, current_ratio, deposit_volumes, downside_vol_21d, earnings_yield, ebitda_margin, ebitda_yield, ecb_hicp, ecb_policy_rates, equity_prices, equity_prices_2000_2026, fcf_margin, fcf_yield, forward_return_, growth, hicp_euro_area_yoy, ic_mean, idiosyncratic_momentum, idiosyncratic_vol, max_drawdown_1y, metrics, missing_policy, momentum, momentum_12m_1m, momentum_reversal_1m, net_margin, official_macro_features, official_rates_bdi, ohlcv_critical_provider_failures, piotroski_f_score, policy_rate_deposit, policy_rate_mro, price_to_52w_high, price_to_sma_200, price_to_sma_50, public_debt, quick_ratio, realized_vol_daily, ret1d, ret_1d, ret_21d, ret_252d, ret_52w_high_proximity, ret_52w_low_proximity, retained_earnings, return, returns, sales_to_price, sharpe, strict, ticker, timezone_policy, turnover_ratio_21d, vol_126d, vol_21d, vol_252d, vol_63d, vol_ratio, volatility, volume, volume_ratio



## 0.3 Multi-Asset Universe State


| asset_class | n_simboli_catalogo | n_simboli_dati_ok | copertura_regioni | gap |
| --- | --- | --- | --- | --- |
| commodity | 11 | 11 | global | OK |
| crypto | 15 | 15 | global, us | OK |
| etf_commodity | 12 | 12 | global | OK |
| etf_equity | 57 | 57 | apac, em, eu, global, it, jp, us | OK |
| etf_fi | 21 | 21 | eu, global, it, us | OK |
| fx | 20 | 20 | em, eu, global | OK |
| yield_proxy | 4 | 4 | it, us | OK |


## 0.4 Academic Factors State


| factor | source | US | EU | IT | JP | EM | World | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MKT_RF | Ken French Data Library | OK | OK | PARTIAL local construction | OK | OK | OK | OK except IT local factor panel metadata gap |
| SMB | Ken French Data Library | OK | OK | PARTIAL local construction | OK | OK | OK | OK except IT local factor panel metadata gap |
| HML | Ken French Data Library | OK | OK | PARTIAL local construction | OK | OK | OK | OK except IT local factor panel metadata gap |
| RMW | Ken French Data Library | OK | OK | PARTIAL local construction | OK | available via EM 5-factor source | gap for World proxy | PARTIAL by regional availability |
| CMA | Ken French Data Library | OK | OK | PARTIAL local construction | OK | available via EM 5-factor source | gap for World proxy | PARTIAL by regional availability |
| MOM | Ken French Data Library | OK | OK | PARTIAL local construction | OK | gap | gap | PARTIAL by regional availability |
| QMJ | AQR Data Library discovery/cache | discovered | discovered | proxy/gap | discovered | discovered | discovered | PARTIAL: parser/source durability still needs hardening |
| BAB | AQR Data Library discovery/cache | discovered | discovered | proxy/gap | discovered | discovered | discovered | PARTIAL: parser/source durability still needs hardening |
| HML_DEVIL | AQR Data Library discovery/cache | discovered | discovered | proxy/gap | discovered | discovered | discovered | PARTIAL: parser/source durability still needs hardening |


## 0.5 Streamlit Page Inventory


| pagina | moduli core usati | tab presenti | funzioni mancanti/TODO markers |
| --- | --- | --- | --- |
| app.py | - | 0 | 0 |
| home_command_center.py | data_explorer, data_health, regime_detection | 1 | 10 |
| 0_🏠_Home.py | - | 0 | 0 |
| 10_Laboratorio_Research_Library.py | - | 0 | 1 |
| 11_Data_API_Control_Center.py | - | 0 | 0 |
| 12_Banking_Data_Lab.py | core | 1 | 1 |
| 13_🌍_Macro_View.py | core, macro_market, sentiment_analysis | 2 | 8 |
| 14_⏱️_Time_Series_Lab.py | core | 1 | 2 |
| 1_📡_Smart_Money_Macro.py | smart_money | 0 | 3 |
| 2_🔬_Valuation_Research.py | - | 2 | 0 |
| 3_📁_Portfolio_Research.py | core | 1 | 0 |
| 4_🔍_Screener_Builder.py | core, data_health, llm_advisors | 1 | 0 |
| 5_📤_Export_Center.py | - | 1 | 0 |
| 6_🧪_Notebook_Runner.py | - | 0 | 0 |
| 7_⚙️_Run_Hi_Freq_Engine.py | - | 0 | 1 |
| 8_🗄️_Data_Platform.py | data_explorer, data_health, data_platform, multi_asset_universe, run_lock, smart_money | 2 | 8 |
| 9_ML_Stock_Lab.py | data_health, factor_benchmarks, factor_portfolio_baselines, llm_advisors, model_monitoring | 0 | 0 |


## 0.6 Pytest Coverage


| test_file | n_tests | modulo testato | coverage stimata |
| --- | --- | --- | --- |
| test_academic_reference_metadata.py | 4 | feature_metadata, metrics_metadata, Academic Reference page wiring | unit/docs smoke |
| test_banking_data.py | 2 | banking_data | unit smoke |
| test_context_bootstrap.py | 8 | data_platform, smart_money, ml_stock_lab import bridge | unit/integration smoke |
| test_cot_smart_money.py | 2 | smart_money | unit/integration smoke |
| test_data_api_control.py | 3 | api_management, batch_download, data_bridge | unit/integration smoke |
| test_data_center_enhancement.py | 3 | cache_manager, data_center_catalog, equity_universe, fama_french, risk_factors | unit/integration smoke |
| test_data_completion_training.py | 3 | data_completion, smart_money | unit/integration smoke |
| test_data_explorer.py | 4 | data_explorer | unit/integration smoke |
| test_data_health.py | 12 | data_completion, data_health, data_platform, run_lock | unit/integration smoke |
| test_factor_benchmarks.py | 1 | factor_benchmarks | unit/integration smoke |
| test_feature_portfolio_valuation_v3.py | 10 | equity_feature_engineering, portfolio_analytics, valuation_analytics | unit smoke |
| test_feature_metric_metadata.py | 6 | feature_metadata, metrics_metadata | unit/integration smoke |
| test_ff_factors.py | 3 | aqr_factors FF regional downloader | unit smoke with mocked fallback |
| test_llm_client.py | 3 | llm_advisors, llm_client, llm_prompts | unit/integration smoke |
| test_llm_lab.py | 3 | llm_lab | unit smoke |
| test_macro_context_regime.py | 2 | macro_context, macro_market, regime_detection | unit/integration smoke |
| test_macro_sentiment_layer.py | 4 | macro_market, sentiment_analysis | unit/integration smoke |
| test_market_statistics.py | 1 | market_statistics | unit/integration smoke |
| test_multi_asset_smart_money_manifests.py | 2 | macro_market, multi_asset_universe, smart_money | unit/integration smoke |
| test_multi_asset_smart_money_ml_v2.py | 4 | factor_portfolio_baselines, model_monitoring, multi_asset_universe, smart_money | unit/integration smoke |
| test_official_macro.py | 4 | bditalia_client, data_center_catalog, ecb_client, feature_metadata, macro_features | unit/integration smoke |
| test_ohlcv_pipeline.py | 14 | api_orchestrator, data_health, kaggle_seed_loader, ohlcv_client, ohlcv_ingest, ohlcv_store | unit/integration smoke |
| test_research_database.py | 2 | research_database | unit smoke |
| test_run_lock.py | 2 | run_lock | unit/integration smoke |
| test_screener_workbench.py | 4 | smart_money | unit/integration smoke |
| test_time_series_forecasting.py | 4 | macro_market, time_series_forecasting | unit/integration smoke |
| test_universe_completeness.py | 4 | multi_asset_universe | unit smoke |


**Test gaps:** aqr_data.py, batch_downloader.py, data_utils.py, export_utils.py, fx_commodities.py, html_utils.py, market_universe.py, storage_policy.py. These are mostly thin wrappers/utilities; core v4 content modules now have at least smoke-level coverage.



## 0.7 Priorities


### 1. Critical Bugs / Integrity
- Keep source commits separate from generated `output/`; current main workspace is dirty and should not be bulk-committed.
- Asia constituents: `nikkei225` parser can emit page metadata tokens; Hang Seng returned empty in this workspace.
- The latest EU/IT price backfill works after provider-symbol normalization; keep this patch and add tests for `.MI/.DE/.PA/.MC/.L`.


### 2. Data Gaps
- FF regional factors are now first-class cached artifacts for US/EU/JP/APAC/EM/World; Italy local factors remain `PARTIAL` until the factor panel carries reliable Italy country/exchange metadata.
- Italy academic factors require local construction from factor panel, not a Ken French direct dataset.
- Smart Money is present as framework/artifacts, but source durability and freshness policy need a validator equivalent to OHLCV.


### 3. UI Gaps
- Academic Reference is present as `15_📚_Academic_Reference.py` and reads cached FF factor parquet files; next UI improvement is to add IC-by-feature overlays when ML Lab exports those artifacts.
- Several analytics functions in portfolio/valuation/model monitoring exist but are not uniformly exposed as interactive controls.
- Data Platform is strong operationally; it should surface source freshness for FF/AQR/smart money in the same Data Health vocabulary.


### 4. Documentation Gaps
- Feature and metric metadata now receive deterministic academic/internal defaults for `formula_latex`, source paper/methodology and economic rationale; remaining work is hand-curating DOI fields for thesis-grade bibliography.
- `FEATURE_ACADEMIC_REFERENCE.md` should become the canonical paper/formula reference and feed the UI.


### 5. Test Gaps
- Provider suffix normalization, FF factor fallback/cache, local Italy factor construction, banking fallback, universe completeness, metadata completeness and Academic Reference wiring are covered by pytest smoke/unit tests.



## Implementation Gate Before Fase 1


- Commit only docs/review and source patches on `codex/research-platform-v2-ml-ollama`; exclude output artifacts.
- Run targeted tests for OHLCV provider normalization and metadata parsing before adding FF/AQR downloads.
