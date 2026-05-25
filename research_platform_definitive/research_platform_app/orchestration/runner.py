"""Unified job runner interface."""

from __future__ import annotations

import traceback
import os
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if SRC_ROOT.exists() and str(SRC_ROOT) not in sys.path:
    # Prefer the canonical editable package over legacy top-level folders with
    # the same package names when Streamlit/AppTest runs from the repo root.
    sys.path.insert(0, str(SRC_ROOT))

try:
    from research_platform_app.operations import generate_all_artifacts
    from research_platform_app.support import default_roots
except Exception:
    from operations import generate_all_artifacts
    from support import default_roots
from research_platform_core import refresh_aqr_factor_library, run_banks_data_pipeline
from research_platform_core.api_management import write_api_control_status
from research_platform_core.data_center_catalog import build_target_catalog, summarize_target_catalog
from research_platform_core.data_completion import CompletionConfig, ResearchDataBootstrapper, validate_completion_coverage
from research_platform_core.data_platform import refresh_europe_stoxx_prices_incremental, write_data_platform_status
from research_platform_core.loaders.bditalia_client import BancaDItaliaClient
from research_platform_core.loaders.ecb_client import EcbClient
from research_platform_core.loaders.fama_french import FamaFrenchLoader
from research_platform_core.macro_features import build_credit_risk_macro_dataset, build_inflation_nowcasting_dataset, save_macro_feature_panel
from research_platform_core.ohlcv_ingest import OhlcvIngestJob, summarize_ohlcv_manifest
from smart_money_engine import run_smart_money_engine
from ml_stock_lab import run_ml_stock_lab_experiment, train_ml_model_suite

from .artifact_contracts import validate_expected_artifacts
from .job_store import JobStore
from .models import JobRun, NotebookJob
from .nbclient_runner import run_with_nbclient
from .papermill_runner import run_with_papermill
from .registry import get_job
from .status import JobStatus
from .utils import RUNS_ROOT, append_log, ensure_dir, new_run_id, utc_now


def _output_notebook_path(job: NotebookJob, run_id: str) -> Path:
    if not job.output_notebook_template:
        return RUNS_ROOT / run_id / f"{job.job_id}.module-run.txt"
    return Path(str(job.output_notebook_template).format(run_id=run_id))


def _artifact_root(job: NotebookJob, roots: dict[str, Path]) -> Path:
    return roots.get(job.output_domain, roots.get("workspace", Path.cwd()))


def create_job_run(
    job_id: str,
    parameters: dict | None = None,
    store: JobStore | None = None,
) -> JobRun:
    store = store or JobStore()
    job = get_job(job_id)
    if not job.enabled:
        raise ValueError(job.disabled_reason or f"Job {job_id} is disabled")

    run_id = new_run_id(job.job_id)
    params = job.validate_parameters(parameters or {})
    output_notebook = _output_notebook_path(job, run_id)
    if not output_notebook.is_absolute():
        output_notebook = RUNS_ROOT.parents[0] / output_notebook
    log_path = RUNS_ROOT / run_id / "run.log"
    ensure_dir(log_path.parent)

    run = JobRun(
        run_id=run_id,
        job_id=job.job_id,
        status=JobStatus.PENDING,
        created_at=utc_now(),
        parameters=params,
        runner_type=job.runner_type,
        notebook_path=str(job.notebook_path or ""),
        output_notebook_path=str(output_notebook),
        log_path=str(log_path),
    )
    store.save_run(run)
    return run


def execute_run(
    run_id: str,
    roots: dict[str, Path] | None = None,
    store: JobStore | None = None,
    prefer_fallback: bool = True,
) -> JobRun:
    roots = roots or default_roots()
    if roots.get("financial_db"):
        os.environ["FINANCIAL_DB_ROOT"] = str(roots["financial_db"])
        os.environ["DB_BASE"] = str(roots["financial_db"])
        os.environ["DATA_PATH"] = str(roots["financial_db"])
    store = store or JobStore()
    run = store.get_run(run_id)
    if run is None:
        raise ValueError(f"Run not found: {run_id}")
    job = get_job(run.job_id)
    params = run.parameters
    output_notebook = Path(run.output_notebook_path)
    log_path = Path(run.log_path)
    ensure_dir(log_path.parent)

    run.status = JobStatus.RUNNING
    run.started_at = utc_now()
    store.save_run(run)
    append_log(log_path, f"Run started: {run.run_id}")
    append_log(log_path, f"Job: {job.job_id}")
    append_log(log_path, f"Parameters: {params}")

    try:
        if job.runner_type == "module":
            result = generate_all_artifacts(roots)
            append_log(log_path, str(result))
            output_notebook.write_text("Module-only job. No notebook executed.\n", encoding="utf-8")
        elif job.runner_type == "data_platform":
            max_files = int(params.get("top_n") or 5000)
            paths = write_data_platform_status(roots["financial_db"], roots["workspace"], max_files=max_files)
            append_log(log_path, f"Data platform status written: {paths}")
            api_paths = write_api_control_status(roots["financial_db"], roots["workspace"])
            append_log(log_path, f"API control status written: {api_paths}")
            output_notebook.write_text("Data-platform status job. No notebook executed.\n", encoding="utf-8")
        elif job.runner_type == "price_refresh":
            max_symbols = int(params.get("top_n") or 25)
            force = bool(params.get("refresh_cache", False))
            manifest = refresh_europe_stoxx_prices_incremental(
                roots["financial_db"],
                max_symbols=max_symbols,
                force=force,
            )
            append_log(log_path, manifest.to_string(index=False))
            paths = write_data_platform_status(roots["financial_db"], roots["workspace"], max_files=5000)
            append_log(log_path, f"Data platform status written after price refresh: {paths}")
            output_notebook.write_text("Price-refresh module job. No notebook executed.\n", encoding="utf-8")
        elif job.runner_type == "ohlcv_daily":
            max_raw = int(params.get("top_n") or 250)
            max_assets = None if max_raw == 0 else max_raw
            mode = "full" if bool(params.get("refresh_cache", False)) else "incremental"
            manifest = OhlcvIngestJob(roots["financial_db"], roots["workspace"]).run_daily(
                markets=["us_all", "europe_major", "japan_major", "global_etfs"],
                start_date="2000-01-01",
                mode=mode,
                max_assets=max_assets,
                batch_size=80,
                dry_run=False,
            )
            append_log(log_path, summarize_ohlcv_manifest(manifest).to_string(index=False))
            output_notebook.write_text("OHLCV daily module job. No notebook executed.\n", encoding="utf-8")
        elif job.runner_type == "aqr_factors":
            max_datasets_raw = int(params.get("top_n") or 0)
            max_datasets = max_datasets_raw if max_datasets_raw > 0 else None
            force = bool(params.get("refresh_cache", False))
            paths = refresh_aqr_factor_library(
                financial_db_root=roots.get("financial_db"),
                output_root=roots.get("workspace"),
                refresh=force,
                max_datasets=max_datasets,
            )
            append_log(log_path, f"AQR factor library refreshed: {paths}")
            output_notebook.write_text("AQR factor-library module job. No notebook executed.\n", encoding="utf-8")
        elif job.runner_type == "fama_french":
            max_raw = int(params.get("top_n") or 0)
            max_datasets = max_raw if max_raw > 0 else None
            force = bool(params.get("refresh_cache", False))
            manifest = FamaFrenchLoader(roots["financial_db"], roots["workspace"]).download_all(refresh=force, max_datasets=max_datasets)
            append_log(log_path, manifest.to_string(index=False))
            output_notebook.write_text("Fama-French factor module job. No notebook executed.\n", encoding="utf-8")
        elif job.runner_type == "data_center_validation":
            max_files = int(params.get("top_n") or 10000)
            out_dir = roots["workspace"] / "data_quality"
            out_dir.mkdir(parents=True, exist_ok=True)
            catalog = build_target_catalog(roots["financial_db"])
            summary = summarize_target_catalog(catalog)
            catalog.to_csv(out_dir / "DataCenter_target_catalog.csv", index=False)
            summary.to_csv(out_dir / "DataCenter_target_summary.csv", index=False)
            append_log(log_path, f"Data Center validation written to {out_dir}; max_files={max_files}")
            output_notebook.write_text("Data Center validation module job. No notebook executed.\n", encoding="utf-8")
        elif job.runner_type == "research_data_bootstrap":
            max_assets_raw = int(params.get("top_n") or 0)
            max_symbols_raw = int(params.get("max_symbols") or 0)
            config = CompletionConfig(
                start_year=int(params.get("start_year") or 2000),
                end_year=int(params.get("end_year") or 2026),
                execute=bool(params.get("execute", False)),
                incremental=not bool(params.get("refresh_cache", False)),
                refresh=bool(params.get("refresh_cache", False)),
                max_assets=None if max_assets_raw == 0 else max_assets_raw,
                max_symbols=None if max_symbols_raw == 0 else max_symbols_raw,
            )
            bootstrapper = ResearchDataBootstrapper(roots["financial_db"], roots["workspace"], config)
            result = bootstrapper.run_all()
            coverage = validate_completion_coverage(roots["financial_db"], roots["workspace"], config.start_year, config.end_year, strict=False)
            append_log(log_path, f"Research data bootstrap config: {config}")
            for name, frame in result.items():
                append_log(log_path, f"{name}: rows={len(frame)}")
            append_log(log_path, coverage.to_string(index=False))
            output_notebook.write_text("Research data bootstrap module job. No notebook executed.\n", encoding="utf-8")
        elif job.runner_type == "ml_training_lab":
            max_rows_raw = int(params.get("top_n") or 0)
            model_list = [item.strip() for item in str(params.get("model_list") or "ols,rf").split(",") if item.strip()]
            result = train_ml_model_suite(
                output_root=roots["workspace"],
                financial_db_root=roots.get("financial_db"),
                start_year=int(params.get("start_year") or 2000),
                end_year=int(params.get("end_year") or 2026),
                train_end_year=int(params.get("train_end_year") or 2018),
                test_start_year=int(params.get("test_start_year") or 2019),
                models=model_list,
                max_rows=None if max_rows_raw == 0 else max_rows_raw,
                use_ollama=bool(params.get("use_ollama", False)),
                target_horizon_days=int(params.get("target_horizon_days") or 21),
                feature_blocks=[item.strip() for item in str(params.get("feature_blocks") or "value,quality,momentum,risk,size,growth,model_based").split(",") if item.strip()],
                cost_bps=float(params.get("cost_bps") or 10.0),
            )
            append_log(log_path, f"ML Training Lab result: {result.get('status')}")
            append_log(log_path, result["metrics"].to_string(index=False) if not result["metrics"].empty else "No metrics produced")
            append_log(log_path, f"Paths: {result.get('paths')}")
            output_notebook.write_text("ML training lab module job. No notebook executed.\n", encoding="utf-8")
        elif job.runner_type == "official_macro":
            force = bool(params.get("refresh_cache", False))
            ecb = EcbClient(roots["financial_db"], roots["workspace"])
            bdi = BancaDItaliaClient(roots["financial_db"], roots["workspace"])
            ecb_manifest = ecb.sync_presets(
                ["hicp_euro_area_yoy", "policy_rate_mro", "policy_rate_deposit", "m2_notional_stock_index", "m3_notional_stock_index"],
                start="2010-01",
                refresh=force,
            )
            bdi_manifest = bdi.sync_presets(["credit_growth", "deposit_volumes", "public_debt", "official_rates_bdi"], start="2010-01", refresh=force)
            feature_dir = roots["financial_db"] / "OfficialMacro" / "features"
            inflation = build_inflation_nowcasting_dataset(ecb_client=ecb, bditalia_client=bdi, start="2010-01")
            credit_risk = build_credit_risk_macro_dataset(ecb_client=ecb, bditalia_client=bdi, start="2010-01")
            save_macro_feature_panel(inflation, feature_dir / "inflation_nowcasting_panel.csv", fmt="csv")
            save_macro_feature_panel(credit_risk, feature_dir / "credit_risk_macro_panel.csv", fmt="csv")
            append_log(log_path, ecb_manifest.to_string(index=False))
            append_log(log_path, bdi_manifest.to_string(index=False))
            append_log(log_path, f"Official macro panels written to {feature_dir}")
            output_notebook.write_text("Official macro module job. No notebook executed.\n", encoding="utf-8")
        elif job.runner_type == "smart_money":
            max_files = int(params.get("top_n") or 5)
            result = run_smart_money_engine(
                roots["financial_db"],
                roots["workspace"] / "smart_money",
                max_files_per_source=max_files,
            )
            append_log(log_path, f"Smart Money manifest: {result['manifest']}")
            output_notebook.write_text("Smart-money government-data module job. No notebook executed.\n", encoding="utf-8")
        elif job.runner_type == "ml_stock_lab":
            max_rows = int(params.get("top_n") or 2000)
            model = str(params.get("risk_profile") or "ols")
            result = run_ml_stock_lab_experiment(
                roots["workspace"] / "ml_stock_lab",
                roots.get("financial_db"),
                model=model,
                max_rows=max_rows,
            )
            append_log(log_path, f"ML Stock Lab result: {result.get('status')}")
            append_log(log_path, f"ML Stock Lab paths: {result.get('paths')}")
            output_notebook.write_text("ML Stock Lab module job. No notebook executed.\n", encoding="utf-8")
        elif job.runner_type == "banking_data":
            output_dir = roots["workspace"] / "banks_pipeline"
            cache_dir = output_dir / "_cache"
            result = run_banks_data_pipeline(
                output_dir=output_dir,
                cache_dir=cache_dir,
                market_start=str(params.get("market_start") or "2015-01-01"),
                include_market=bool(params.get("include_market", False)),
                include_ecb_macro=bool(params.get("include_ecb_macro", False)),
            )
            append_log(log_path, f"Banking Data Lab artifacts: {{name: len(df) for name, df in result.items()}}")
            append_log(log_path, str({name: len(df) for name, df in result.items()}))
            output_notebook.write_text("Banking Data Lab module job. No notebook executed.\n", encoding="utf-8")
        elif job.runner_type == "papermill":
            try:
                run_with_papermill(job, params, output_notebook, log_path)
            except Exception as exc:
                if not prefer_fallback:
                    raise
                append_log(log_path, f"Papermill failed, falling back to nbclient: {exc}")
                run.runner_type = "nbclient_fallback"
                run_with_nbclient(job, params, output_notebook, log_path)
        elif job.runner_type == "nbclient":
            run_with_nbclient(job, params, output_notebook, log_path)
        else:
            raise ValueError(f"Unsupported runner_type: {job.runner_type}")

        artifact_root = _artifact_root(job, roots)
        if job.runner_type == "price_refresh":
            workspace_specs = [spec for spec in job.expected_artifacts if not spec.relative_path.startswith("catalog/")]
            drive_specs = [spec for spec in job.expected_artifacts if spec.relative_path.startswith("catalog/")]
            from .artifact_contracts import validate_artifacts
            run.artifacts_detected = validate_artifacts(artifact_root, workspace_specs) + validate_artifacts(roots["financial_db"], drive_specs)
        else:
            run.artifacts_detected = validate_expected_artifacts(job, artifact_root)
        required_missing = [row for row in run.artifacts_detected if row["required"] and not row["exists"]]
        run.status = JobStatus.FAILED if required_missing else JobStatus.SUCCESS
        if required_missing:
            run.error_message = "Required artifacts missing after execution"
            append_log(log_path, run.error_message)
        else:
            append_log(log_path, "Run completed successfully")
    except Exception as exc:
        run.status = JobStatus.FAILED
        run.error_message = str(exc)
        append_log(log_path, traceback.format_exc())
    finally:
        run.finished_at = utc_now()
        store.save_run(run)
    return run


def run_job(
    job_id: str,
    parameters: dict | None = None,
    roots: dict[str, Path] | None = None,
    store: JobStore | None = None,
    prefer_fallback: bool = True,
) -> JobRun:
    store = store or JobStore()
    run = create_job_run(job_id, parameters, store=store)
    return execute_run(run.run_id, roots=roots, store=store, prefer_fallback=prefer_fallback)
