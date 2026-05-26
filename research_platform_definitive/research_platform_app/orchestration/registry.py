"""Central registry of supported notebook and lightweight jobs."""

from __future__ import annotations

from pathlib import Path

from .models import ArtifactSpec, NotebookJob, ParameterSpec
from .utils import PROJECT_ROOT


def _exists(path: Path) -> bool:
    return path.exists() and path.is_file()


COMPANY_NOTEBOOK = PROJECT_ROOT / "company_valuation" / "notebooks" / "Company_Valuation_Final_Version.ipynb"
PORTFOLIO_NOTEBOOK = PROJECT_ROOT / "portfolio_analysis" / "notebooks" / "Portfolio-Analysis-Model_RESEARCH_PLATFORM_PRO.ipynb"
ML_STOCK_LAB_NOTEBOOK = PROJECT_ROOT / "machine_learning_lab" / "notebooks" / "ML_Stock_Lab_Experiments.ipynb"


def get_job_registry() -> dict[str, NotebookJob]:
    return {
        "valuation_research_refresh": NotebookJob(
            job_id="valuation_research_refresh",
            label="Equity Valuation Notebook Refresh",
            notebook_path=COMPANY_NOTEBOOK,
            runner_type="papermill",
            output_notebook_template="runs/{run_id}/Company_Valuation_Final_Version.executed.ipynb",
            timeout_seconds=60 * 60,
            parameters=[
                ParameterSpec("ticker", "str", "ENEL.MI", "Primary ticker for notebook parameter cells."),
                ParameterSpec("top_n", "int", 25, "Number of top names for ranking exports.", minimum=1, maximum=500),
                ParameterSpec("refresh_cache", "bool", False, "Allow notebook cache refresh where supported."),
                ParameterSpec("rerun_exports_only", "bool", True, "Prefer export-only/light sections when the notebook supports it."),
            ],
            expected_artifacts=[
                ArtifactSpec("Screener results", "tables/ScreenerResults.csv", True, freshness_hours=72),
                ArtifactSpec("Screener summary", "tables/ScreenerSummary.csv", True, freshness_hours=72),
                ArtifactSpec("Finviz map payload", "tables/FinvizMapPayload.csv", False, freshness_hours=72),
                ArtifactSpec("Company dashboard", "dashboard/company_valuation_navigable_dashboard.html", False, freshness_hours=168),
            ],
            tags=["company", "valuation", "notebook"],
            enabled=_exists(COMPANY_NOTEBOOK),
            disabled_reason="" if _exists(COMPANY_NOTEBOOK) else "Company valuation notebook not found.",
            output_domain="company",
            description="Runs the company valuation research notebook and validates its exported artifacts.",
        ),
        "portfolio_research_refresh": NotebookJob(
            job_id="portfolio_research_refresh",
            label="Portfolio Research Notebook Refresh",
            notebook_path=PORTFOLIO_NOTEBOOK,
            runner_type="papermill",
            output_notebook_template="runs/{run_id}/Portfolio-Analysis-Model_RESEARCH_PLATFORM_PRO.executed.ipynb",
            timeout_seconds=60 * 45,
            parameters=[
                ParameterSpec("portfolio_name", "str", "Research Portfolio", "Portfolio label used by parameterized cells when available."),
                ParameterSpec("benchmark", "str", "SPY", "Benchmark ticker or label."),
                ParameterSpec("risk_profile", "choice", "balanced", "Risk profile for supported notebook config.", choices=["defensive", "balanced", "growth"]),
                ParameterSpec("refresh_cache", "bool", False, "Allow notebook cache refresh where supported."),
                ParameterSpec("rerun_exports_only", "bool", True, "Prefer export-only/light sections when the notebook supports it."),
            ],
            expected_artifacts=[
                ArtifactSpec("Portfolio selection results", "tables/PortfolioSelectionResults.csv", True, freshness_hours=72),
                ArtifactSpec("Portfolio selection summary", "tables/PortfolioSelectionSummary.csv", True, freshness_hours=72),
                ArtifactSpec("Portfolio allocation", "tables/PortfolioAllocation.csv", False, freshness_hours=72),
                ArtifactSpec("Portfolio dashboard", "dashboard/portfolio_research_dashboard.html", False, freshness_hours=168),
            ],
            tags=["portfolio", "allocation", "notebook"],
            enabled=_exists(PORTFOLIO_NOTEBOOK),
            disabled_reason="" if _exists(PORTFOLIO_NOTEBOOK) else "Portfolio research notebook not found.",
            output_domain="portfolio",
            description="Runs the portfolio research notebook and validates portfolio selection/allocation artifacts.",
        ),
        "screener_refresh": NotebookJob(
            job_id="screener_refresh",
            label="Lightweight Screener / Selection Refresh",
            notebook_path=None,
            runner_type="module",
            output_notebook_template="",
            timeout_seconds=60 * 5,
            parameters=[
                ParameterSpec("top_n", "int", 25, "Reserved for future UI-level result limiting.", minimum=1, maximum=500),
                ParameterSpec("refresh_cache", "bool", False, "No API/cache refresh is performed by this lightweight job."),
            ],
            expected_artifacts=[
                ArtifactSpec("Company screener results", "tables/ScreenerResults.csv", True, freshness_hours=72),
                ArtifactSpec("Finviz map payload", "tables/FinvizMapPayload.csv", False, freshness_hours=72),
            ],
            tags=["company", "portfolio", "screener", "module"],
            enabled=True,
            output_domain="company",
            description="Runs only low-cost module layers over existing exports. No notebook execution.",
        ),
        "data_platform_status_refresh": NotebookJob(
            job_id="data_platform_status_refresh",
            label="Database Finanziario Status Refresh",
            notebook_path=None,
            runner_type="data_platform",
            output_notebook_template="",
            timeout_seconds=60 * 5,
            parameters=[
                ParameterSpec("top_n", "int", 5000, "Maximum number of files to inventory.", minimum=500, maximum=20000),
                ParameterSpec("refresh_cache", "bool", False, "No API refresh is performed by this status job."),
            ],
            expected_artifacts=[
                ArtifactSpec("Data platform inventory", "tables/DataPlatform_inventory.csv", True, freshness_hours=24 * 7),
                ArtifactSpec("Data platform summary", "tables/DataPlatform_summary.csv", True, freshness_hours=24 * 7),
                ArtifactSpec("Provider registry", "tables/DataPlatform_api_providers.csv", False, freshness_hours=24 * 30),
                ArtifactSpec("Data platform API contract", "api_contracts/data_platform_contract.json", True, freshness_hours=24 * 7),
                ArtifactSpec("API control credential status", "tables/DataAPI_credential_status.csv", False, freshness_hours=24 * 7),
                ArtifactSpec("API control contract", "api_contracts/data_api_control_contract.json", False, freshness_hours=24 * 7),
            ],
            tags=["data", "drive", "catalog", "module"],
            enabled=True,
            output_domain="workspace",
            description="Inventories Database Finanziario and writes freshness/catalog status tables. No API calls.",
        ),
        "europe_prices_incremental_refresh": NotebookJob(
            job_id="europe_prices_incremental_refresh",
            label="Europe/STOXX Incremental Price Refresh",
            notebook_path=None,
            runner_type="price_refresh",
            output_notebook_template="",
            timeout_seconds=60 * 15,
            parameters=[
                ParameterSpec("top_n", "int", 25, "Maximum stale symbols to inspect/refresh.", minimum=1, maximum=250),
                ParameterSpec("refresh_cache", "bool", False, "Force refresh even if files are fresh."),
            ],
            expected_artifacts=[
                ArtifactSpec("Europe price refresh manifest", "catalog/europe_stoxx_incremental_price_refresh.csv", False, freshness_hours=24 * 7),
                ArtifactSpec("Data platform inventory", "tables/DataPlatform_inventory.csv", True, freshness_hours=24 * 7),
            ],
            tags=["data", "prices", "drive", "api", "module"],
            enabled=True,
            output_domain="workspace",
            description="Refreshes stale Europe/STOXX price parquet files incrementally. API calls only for stale/missing files.",
        ),
        "ohlcv_daily_incremental_refresh": NotebookJob(
            job_id="ohlcv_daily_incremental_refresh",
            label="Global OHLCV Daily Incremental Refresh",
            notebook_path=None,
            runner_type="ohlcv_daily",
            output_notebook_template="",
            timeout_seconds=60 * 60,
            parameters=[
                ParameterSpec("top_n", "int", 250, "Maximum assets to refresh. Use 0 for all discovered assets.", minimum=0, maximum=20000),
                ParameterSpec("refresh_cache", "bool", False, "Use full-history mode instead of incremental mode."),
            ],
            expected_artifacts=[
                ArtifactSpec("OHLCV daily manifest", "MarketData/OHLCV/manifests/ohlcv_daily_manifest.csv", False, freshness_hours=24 * 3),
                ArtifactSpec("OHLCV asset universe", "MarketData/OHLCV/asset_master_candidates.csv", False, freshness_hours=24 * 7),
                ArtifactSpec("OHLCV SQLite mirror", "MarketData/ohlcv.sqlite", False, freshness_hours=24 * 7),
            ],
            tags=["data", "ohlcv", "prices", "module"],
            enabled=True,
            output_domain="workspace",
            description="Builds/updates global OHLCV daily prices with yfinance bulk plus configured fallbacks and DB upserts.",
        ),
        "aqr_factor_library_refresh": NotebookJob(
            job_id="aqr_factor_library_refresh",
            label="AQR Factor Library Refresh",
            notebook_path=None,
            runner_type="aqr_factors",
            output_notebook_template="",
            timeout_seconds=60 * 20,
            parameters=[
                ParameterSpec("top_n", "int", 0, "Maximum AQR datasets to refresh. Use 0 for all discovered datasets.", minimum=0, maximum=100),
                ParameterSpec("refresh_cache", "bool", False, "Force redownload/reparse instead of using cached Excel/CSV files."),
            ],
            expected_artifacts=[
                ArtifactSpec("AQR factor discovery", "aqr_factors/tables/AQRFactorDiscovery.csv", True, freshness_hours=24 * 7),
                ArtifactSpec("AQR factor refresh log", "aqr_factors/tables/AQRFactorRefreshLog.csv", True, freshness_hours=24 * 7),
                ArtifactSpec("AQR factor panel", "aqr_factors/tables/AQRFactorPanel.csv", False, freshness_hours=24 * 7),
                ArtifactSpec("AQR factor manifest", "aqr_factors/tables/AQRFactorManifest.json", True, freshness_hours=24 * 7),
            ],
            tags=["data", "factors", "aqr", "module"],
            enabled=True,
            output_domain="workspace",
            description="Discovers AQR Data Library Excel files, caches them Drive-first, parses factor sheets, and writes a regression-ready wide panel.",
        ),
        "fama_french_factor_refresh": NotebookJob(
            job_id="fama_french_factor_refresh",
            label="Fama-French Factor Refresh",
            notebook_path=None,
            runner_type="fama_french",
            output_notebook_template="",
            timeout_seconds=60 * 10,
            parameters=[
                ParameterSpec("top_n", "int", 4, "Maximum Fama-French datasets to refresh. Use 0 for all configured datasets.", minimum=0, maximum=25),
                ParameterSpec("refresh_cache", "bool", False, "Force redownload instead of using cached CSV files."),
            ],
            expected_artifacts=[
                ArtifactSpec("Fama-French manifest", "Factors/FamaFrench/FamaFrench_manifest.csv", False, freshness_hours=24 * 31),
                ArtifactSpec("Data Center target catalog", "data_quality/DataCenter_target_catalog.csv", False, freshness_hours=24 * 7),
            ],
            tags=["data", "factors", "fama_french", "module"],
            enabled=True,
            output_domain="workspace",
            description="Downloads configured Ken French Data Library ZIP/CSV factors and writes normalized factor CSVs to Database Finanziario.",
        ),
        "data_center_validation": NotebookJob(
            job_id="data_center_validation",
            label="Data Center Coverage Validation",
            notebook_path=None,
            runner_type="data_center_validation",
            output_notebook_template="",
            timeout_seconds=60 * 5,
            parameters=[
                ParameterSpec("top_n", "int", 10000, "Maximum files to inventory for stale-data checks.", minimum=500, maximum=50000),
                ParameterSpec("refresh_cache", "bool", False, "No downloads are performed by validation."),
            ],
            expected_artifacts=[
                ArtifactSpec("Data Center target catalog", "data_quality/DataCenter_target_catalog.csv", True, freshness_hours=24 * 7),
                ArtifactSpec("Data Center target summary", "data_quality/DataCenter_target_summary.csv", True, freshness_hours=24 * 7),
            ],
            tags=["data", "quality", "catalog", "module"],
            enabled=True,
            output_domain="workspace",
            description="Writes strategic Data Center coverage and stale-inventory reports without provider calls.",
        ),
        "research_data_2000_2026_bootstrap": NotebookJob(
            job_id="research_data_2000_2026_bootstrap",
            label="Research Data Bootstrap 2000-2026",
            notebook_path=None,
            runner_type="research_data_bootstrap",
            output_notebook_template="",
            timeout_seconds=60 * 90,
            parameters=[
                ParameterSpec("start_year", "int", 2000, "Historical start year for all data families.", minimum=1990, maximum=2026),
                ParameterSpec("end_year", "int", 2026, "Historical end year for all data families.", minimum=2000, maximum=2026),
                ParameterSpec("top_n", "int", 250, "Maximum OHLCV assets. Use 0 for all discovered assets.", minimum=0, maximum=20000),
                ParameterSpec("max_symbols", "int", 25, "Maximum fundamental symbols per universe. Use 0 for all constituents.", minimum=0, maximum=2000),
                ParameterSpec("execute", "bool", False, "Run provider calls. Leave off for dry-run/manifest planning."),
                ParameterSpec("refresh_cache", "bool", False, "Force cache refresh where supported."),
            ],
            expected_artifacts=[
                ArtifactSpec("Data completion manifest", "data_completion/full_completion_manifest.json", True, freshness_hours=24 * 7),
                ArtifactSpec("Data source map", "data_completion/DataCompletion_source_map.csv", True, freshness_hours=24 * 30),
                ArtifactSpec("Equity prices manifest", "data_completion/equity_prices_2000_2026.csv", True, freshness_hours=24 * 7),
                ArtifactSpec("Factor universe panel", "ml_training_lab/tables/FactorUniversePanel.csv", False, freshness_hours=24 * 7),
                ArtifactSpec("Coverage validation", "data_completion/DataCompletion_coverage_validation.csv", False, freshness_hours=24 * 7),
            ],
            tags=["data", "bootstrap", "2000_2026", "module"],
            enabled=True,
            output_domain="workspace",
            description="Coordinates equities, OHLCV, macro/FX, factor libraries, Smart Money and Banking Data Lab completion for 2000-2026.",
        ),
        "ml_training_lab_2000_2026": NotebookJob(
            job_id="ml_training_lab_2000_2026",
            label="ML Training Lab 2000-2026",
            notebook_path=None,
            runner_type="ml_training_lab",
            output_notebook_template="",
            timeout_seconds=60 * 60,
            parameters=[
                ParameterSpec("start_year", "int", 2000, "Training panel start year.", minimum=1990, maximum=2026),
                ParameterSpec("end_year", "int", 2026, "Training panel end year.", minimum=2000, maximum=2026),
                ParameterSpec("train_end_year", "int", 2018, "Last year included in training split.", minimum=2000, maximum=2025),
                ParameterSpec("test_start_year", "int", 2019, "First year included in test split.", minimum=2001, maximum=2026),
                ParameterSpec("top_n", "int", 0, "Maximum rows for training smoke. Use 0 for full panel.", minimum=0, maximum=5000000),
                ParameterSpec("model_list", "str", "ols,rf", "Comma-separated model families: ols,lasso,rf,gbrt,ensemble."),
                ParameterSpec("use_ollama", "bool", False, "Generate a local Ollama summary if the server is running."),
            ],
            expected_artifacts=[
                ArtifactSpec("ML training metrics", "ml_training_lab/tables/MLTraining_metrics.csv", True, freshness_hours=24 * 7),
                ArtifactSpec("ML training predictions", "ml_training_lab/tables/MLTraining_predictions.csv", False, freshness_hours=24 * 7),
                ArtifactSpec("ML training manifest", "ml_training_lab/MLTraining_manifest.json", True, freshness_hours=24 * 7),
                ArtifactSpec("ML Stock Lab model comparison", "ml_stock_lab/tables/MLStockLab_model_comparison.csv", False, freshness_hours=24 * 7),
            ],
            tags=["ml_stock_lab", "training", "2000_2026", "module"],
            enabled=True,
            output_domain="workspace",
            description="Trains expected-return models on the completed factor panel and writes model cards, predictions and metrics.",
        ),
        "official_macro_refresh": NotebookJob(
            job_id="official_macro_refresh",
            label="ECB / Banca d'Italia Official Macro Refresh",
            notebook_path=None,
            runner_type="official_macro",
            output_notebook_template="",
            timeout_seconds=60 * 20,
            parameters=[
                ParameterSpec("top_n", "int", 0, "Reserved for future preset limiting. Use 0 for configured official macro presets.", minimum=0, maximum=25),
                ParameterSpec("refresh_cache", "bool", False, "Force redownload instead of using existing official macro CSV files."),
            ],
            expected_artifacts=[
                ArtifactSpec("ECB official macro manifest", "OfficialMacro/ECB/ecb_official_macro_manifest.csv", False, freshness_hours=24 * 31),
                ArtifactSpec("Banca d'Italia official macro manifest", "OfficialMacro/BancaItalia/bditalia_official_macro_manifest.csv", False, freshness_hours=24 * 31),
                ArtifactSpec("Inflation nowcasting panel", "OfficialMacro/features/inflation_nowcasting_panel.csv", False, freshness_hours=24 * 31),
                ArtifactSpec("Credit risk macro panel", "OfficialMacro/features/credit_risk_macro_panel.csv", False, freshness_hours=24 * 31),
            ],
            tags=["data", "macro", "ecb", "bancaditalia", "module"],
            enabled=True,
            output_domain="workspace",
            description="Downloads official ECB/Banca d'Italia macro datasets and writes ML-ready feature panels.",
        ),
        "smart_money_government_refresh": NotebookJob(
            job_id="smart_money_government_refresh",
            label="Smart Money Government Data Refresh",
            notebook_path=None,
            runner_type="smart_money",
            output_notebook_template="",
            timeout_seconds=60 * 10,
            parameters=[
                ParameterSpec("top_n", "int", 5, "Maximum local files to inspect per official source.", minimum=1, maximum=50),
                ParameterSpec("refresh_cache", "bool", False, "Reserved for future provider download; current job is local official-source scan only."),
            ],
            expected_artifacts=[
                ArtifactSpec("Smart Money scores", "smart_money/tables/SmartMoney_smart_money_scores.csv", True, freshness_hours=24 * 7),
                ArtifactSpec("Smart Money event feed", "smart_money/tables/SmartMoney_event_feed.csv", True, freshness_hours=24 * 7),
                ArtifactSpec("Smart Money coverage", "smart_money/tables/SmartMoney_coverage.csv", True, freshness_hours=24 * 7),
                ArtifactSpec("Smart Money report", "smart_money/reports/smart_money_government_data_report.html", False, freshness_hours=24 * 7),
            ],
            tags=["smart_money", "government_data", "ownership", "module"],
            enabled=True,
            output_domain="workspace",
            description="Runs the local official-source smart-money pipeline over SEC/CFTC/Treasury/USAspending/EU proxy files found in Database Finanziario.",
        ),
        "ml_stock_lab_experiments_refresh": NotebookJob(
            job_id="ml_stock_lab_experiments_refresh",
            label="ML Stock Lab Experiments Refresh",
            notebook_path=ML_STOCK_LAB_NOTEBOOK,
            runner_type="ml_stock_lab",
            output_notebook_template="runs/{run_id}/ML_Stock_Lab_Experiments.executed.ipynb",
            timeout_seconds=60 * 20,
            parameters=[
                ParameterSpec("top_n", "int", 2000, "Maximum rows for lightweight ML lab artifact generation.", minimum=100, maximum=10000),
                ParameterSpec("risk_profile", "choice", "ols", "Model family for fair-value estimation.", choices=["ols", "lasso", "rf", "gbrt", "ensemble"]),
                ParameterSpec("refresh_cache", "bool", False, "Reserved for future data refresh; current job reuses artifacts/Drive data."),
            ],
            expected_artifacts=[
                ArtifactSpec("ML Stock Lab panel", "ml_stock_lab/tables/MLStockLab_panel.csv", True, freshness_hours=24 * 7),
                ArtifactSpec("ML Stock Lab signals", "ml_stock_lab/tables/MLStockLab_signals.csv", True, freshness_hours=24 * 7),
                ArtifactSpec("ML Stock Lab metrics", "ml_stock_lab/tables/MLStockLab_metrics.csv", True, freshness_hours=24 * 7),
                ArtifactSpec("ML Stock Lab quintiles", "ml_stock_lab/tables/MLStockLab_quintile_returns.csv", False, freshness_hours=24 * 7),
            ],
            tags=["ml_stock_lab", "valuation", "screening", "module"],
            enabled=True,
            output_domain="workspace",
            description="Runs lightweight ml_stock_lab artifacts for fair value, mispricing, z-score screening and quintile diagnostics.",
        ),
        "banking_data_refresh": NotebookJob(
            job_id="banking_data_refresh",
            label="Banking Data Lab Refresh",
            notebook_path=None,
            runner_type="banking_data",
            output_notebook_template="",
            timeout_seconds=60 * 15,
            parameters=[
                ParameterSpec("market_start", "date", "2015-01-01", "Start date for listed-bank market panel."),
                ParameterSpec("include_market", "bool", False, "Refresh listed-bank yfinance market panel."),
                ParameterSpec("include_ecb_macro", "bool", False, "Call ECB BSI macro endpoints during refresh."),
            ],
            expected_artifacts=[
                ArtifactSpec("Banking universe", "banks_pipeline/banks_universe.csv", True, freshness_hours=24 * 14),
                ArtifactSpec("Banking macro/regulatory panel", "banks_pipeline/banks_macro_regulatory.csv", False, freshness_hours=24 * 31),
                ArtifactSpec("Banking market panel", "banks_pipeline/banks_market_panel.csv", False, freshness_hours=24 * 7),
                ArtifactSpec("Banking SQLite mirror", "banks_pipeline/banks_data.sqlite", False, freshness_hours=24 * 14),
            ],
            tags=["banking", "data", "official_sources", "module"],
            enabled=True,
            output_domain="workspace",
            description="Builds the Banking Data Lab universe and optional market/macro panels with clean artifact contracts.",
        ),
    }


def get_job(job_id: str) -> NotebookJob:
    return get_job_registry()[job_id]


def get_enabled_jobs() -> dict[str, NotebookJob]:
    return {job_id: job for job_id, job in get_job_registry().items() if job.enabled}
