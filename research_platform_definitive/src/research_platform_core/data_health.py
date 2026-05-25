"""Backend data-health helpers for Data Platform and restart workflows.

The functions in this module are intentionally Streamlit-free. They read the
same manifests used by validators and provide simple tables/controllers that
the UI can render without duplicating data orchestration logic.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data_platform import get_ohlcv_daily_search_roots, get_ohlcv_parquet_root_info, resolve_data_platform_roots, utc_now
from .ohlcv_ingest import OhlcvIngestJob, summarize_ohlcv_manifest
from .run_lock import acquire_stage_lock, list_stage_locks, read_stage_lock, release_stage_lock


DATA_COMPLETION_STAGES = [
    "equity_fundamentals",
    "equity_prices",
    "macro_fx",
    "factor_libraries",
    "smart_money",
    "banking",
    "factor_universe_panel",
]

CRITICAL_OHLCV_FAILURES = {"NETWORK_TIMEOUT", "PROVIDER_ERROR", "INVALID_SYMBOL", "WRITE_FAILED"}
WATCH_OHLCV_FAILURES = {"NO_PRICE_DATA", "DELISTED", "LIMITED_HISTORY"}


@dataclass(frozen=True)
class StageHealth:
    stage: str
    status: str
    coverage_ratio: float
    covered_assets: int
    total_assets: int
    failure_count: int
    critical_failure_count: int
    warning_failure_count: int
    last_run_at: str
    run_id: str = ""
    pid: str = ""
    log_path: str = ""
    lock_path: str = ""


@dataclass(frozen=True)
class TickerDataStatus:
    ticker: str
    overall_status: str
    fundamentals_status: str
    prices_status: str
    price_coverage_status: str = ""
    provider_error_type: str = ""
    has_price_parquet: bool = False
    last_price_date: str = ""
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 1:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _json_safe_mapping(mapping: dict[Any, Any]) -> dict[str, Any]:
    """Normalize manifest-derived dictionaries before JSON serialization."""
    clean: dict[str, Any] = {}
    for key, value in (mapping or {}).items():
        if pd.isna(key):
            safe_key = "UNKNOWN"
        else:
            safe_key = str(key)
        if isinstance(value, (np.integer,)):
            safe_value: Any = int(value)
        elif isinstance(value, (np.floating,)):
            safe_value = float(value)
        else:
            safe_value = value
        clean[safe_key] = safe_value
    return clean


def _pid_alive(pid: str | int | None) -> bool:
    if pid in {None, ""}:
        return False
    try:
        os.kill(int(str(pid).strip()), 0)
        return True
    except Exception:
        return False


def _stage_from_log(log_path: Path) -> str:
    if not log_path.exists():
        return ""
    current = ""
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return ""
    for line in lines[-300:]:
        start_match = re.search(r"stage=([a-z_]+) start", line)
        if start_match:
            current = start_match.group(1)
            continue
        simple_start = re.search(r"\b(equity_prices|equity_fundamentals|macro_fx|factor_libraries|smart_money|banking|factor_universe_panel) start\b", line)
        if simple_start:
            current = simple_start.group(1)
            continue
        done_match = re.search(r"stage=([a-z_]+).* done|([a-z_]+) done\b", line)
        done_stage = next((group for group in (done_match.groups() if done_match else []) if group), "") if done_match else ""
        if done_stage and done_stage == current:
            current = ""
    return current


def _running_stages(output_root: Path) -> dict[str, dict[str, Any]]:
    runs_root = output_root / "runs"
    running: dict[str, dict[str, Any]] = {}
    for info in list_stage_locks(output_root):
        if _pid_alive(info.pid):
            running[info.stage] = {
                "run_id": info.run_id,
                "pid": str(info.pid),
                "log_path": "",
                "lock_path": info.lock_path,
                "locked_by": info.run_id,
                "lock_acquired_at": info.started_at,
            }
    if not runs_root.exists():
        return running
    for run_dir in sorted([p for p in runs_root.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)[:30]:
        pid_path = run_dir / "backfill.pid"
        if not pid_path.exists():
            continue
        pid = pid_path.read_text(encoding="utf-8", errors="replace").strip()
        if not _pid_alive(pid):
            continue
        log_path = run_dir / "backfill.log"
        if not log_path.exists():
            log_path = run_dir / "run.log"
        stage = _stage_from_log(log_path)
        if not stage:
            name = run_dir.name.lower()
            if "equity_prices" in name or "ohlcv" in name:
                stage = "equity_prices"
            elif "equity_fundamentals" in name or "fundamental" in name:
                stage = "equity_fundamentals"
        if stage:
            running.setdefault(stage, {}).update({"run_id": run_dir.name, "pid": pid, "log_path": str(log_path)})
    return running


def _stage_status_from_manifest(frame: pd.DataFrame, strict: bool = False) -> str:
    if frame.empty:
        return "MISSING"
    values = set(frame.get("status", pd.Series(dtype=object)).fillna("").astype(str).str.upper())
    if values & {"FAILED", "ERROR"}:
        return "FAILED"
    if values & {"DONE", "OK"}:
        return "OK"
    if "PLANNED" in values:
        return "FAILED" if strict else "PLANNED"
    if "SKIPPED" in values:
        return "FAILED" if strict else "SKIPPED"
    if "EMPTY" in values:
        return "FAILED" if strict else "EMPTY"
    return ";".join(sorted(v for v in values if v)) or "UNKNOWN"


def list_ohlcv_failures(
    financial_db_root: str | Path | None = None,
    output_root: str | Path | None = None,
) -> pd.DataFrame:
    """Load OHLCV parquet-write failures as a retry-ready table."""
    roots = resolve_data_platform_roots(financial_db_root=financial_db_root, repo_output_root=output_root)
    candidates = [
        roots.repo_output / "tables" / "OHLCV_write_failures.csv",
        roots.financial_db / "MarketData" / "OHLCV" / "manifests" / "OHLCV_write_failures.csv",
    ]
    frame = next((df for df in (_read_csv(path) for path in candidates) if not df.empty), pd.DataFrame())
    if frame.empty:
        manifest = _read_csv(roots.repo_output / "tables" / "OHLCV_daily_manifest.csv")
        if not manifest.empty and "write_status" in manifest.columns:
            frame = manifest[manifest["write_status"].astype(str).str.upper().eq("FAILED")].copy()
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "asset",
                "exchange",
                "universe",
                "provider_symbol",
                "target_path",
                "error_summary",
                "last_failure_at",
                "retry_candidate",
            ]
        )
    out = frame.copy()
    out["ticker"] = out.get("ticker", out.get("provider_symbol", pd.Series("", index=out.index))).fillna("").astype(str).str.upper()
    out["asset"] = out["ticker"].where(out["ticker"].ne(""), out.get("provider_symbol", pd.Series("", index=out.index)).astype(str))
    out["exchange"] = out.get("exchange", pd.Series("", index=out.index)).fillna("").astype(str)
    out["universe"] = out.get("universe", out["exchange"]).fillna("").astype(str)
    out["provider_symbol"] = out.get("provider_symbol", out["ticker"]).fillna(out["ticker"]).astype(str)
    out["target_path"] = out.get("target_path", pd.Series("", index=out.index)).fillna("").astype(str)
    if "parquet_root" in out.columns:
        missing_target = out["target_path"].str.len().eq(0)
        if missing_target.any():
            asset_name = out["ticker"].where(out["ticker"].str.len().gt(0), out["provider_symbol"]).astype(str).str.replace("/", "_", regex=False)
            inferred = (
                out["parquet_root"].fillna("").astype(str).str.rstrip("/")
                + "/daily/"
                + out["exchange"].replace("", "unknown").fillna("unknown").astype(str)
                + "/"
                + asset_name
                + ".parquet"
            )
            out.loc[missing_target, "target_path"] = inferred[missing_target]
    out["error_summary"] = out.get("write_error", out.get("error", pd.Series("", index=out.index))).fillna("").astype(str)
    time_source = out.get("updated_at", out.get("ingestion_ts", pd.Series("", index=out.index))).fillna("").astype(str)
    if not time_source.str.len().gt(0).any():
        latest = max((path.stat().st_mtime for path in candidates if path.exists()), default=0)
        time_source = pd.Series(pd.Timestamp(latest, unit="s", tz="UTC").isoformat() if latest else "", index=out.index)
    out["last_failure_at"] = time_source
    out["retry_candidate"] = out["provider_symbol"].astype(str).str.len().gt(0)
    columns = ["ticker", "asset", "exchange", "universe", "provider_symbol", "target_path", "error_summary", "last_failure_at", "retry_candidate"]
    return out[[col for col in columns if col in out.columns]].reset_index(drop=True)


def list_ohlcv_provider_failures(
    financial_db_root: str | Path | None = None,
    output_root: str | Path | None = None,
) -> pd.DataFrame:
    """Load provider-side OHLCV failures/observations as a normalized table."""
    roots = resolve_data_platform_roots(financial_db_root=financial_db_root, repo_output_root=output_root)
    candidates = [
        roots.repo_output / "tables" / "OHLCV_provider_failures.csv",
        roots.financial_db / "MarketData" / "OHLCV" / "manifests" / "OHLCV_provider_failures.csv",
    ]
    frame = next((df for df in (_read_csv(path) for path in candidates) if not df.empty), pd.DataFrame())
    if frame.empty:
        manifest = _read_csv(roots.repo_output / "tables" / "OHLCV_daily_manifest.csv")
        if not manifest.empty:
            coverage = manifest.get("coverage_status", pd.Series("", index=manifest.index)).fillna("").astype(str).str.upper()
            status = manifest.get("status", pd.Series("", index=manifest.index)).fillna("").astype(str).str.lower()
            mask = coverage.isin(CRITICAL_OHLCV_FAILURES | WATCH_OHLCV_FAILURES) | status.isin(
                {"invalid_symbol", "failed", "bulk_empty", "network_timeout", "no_price_data", "limited_history", "delisted"}
            )
            frame = manifest[mask].copy()
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "provider_symbol",
                "exchange",
                "universe",
                "category",
                "provider",
                "error_type",
                "coverage_status",
                "error_summary",
                "attempts",
                "last_seen",
                "retry_candidate",
            ]
        )
    out = frame.copy()
    out["ticker"] = out.get("ticker", out.get("provider_symbol", pd.Series("", index=out.index))).fillna("").astype(str).str.upper()
    out["provider_symbol"] = out.get("provider_symbol", out["ticker"]).fillna(out["ticker"]).astype(str)
    out["exchange"] = out.get("exchange", pd.Series("", index=out.index)).fillna("").astype(str)
    out["universe"] = out.get("universe", out["exchange"]).fillna("").astype(str)
    out["category"] = out.get("category", pd.Series("common", index=out.index)).replace("", "common").fillna("common").astype(str)
    out["provider"] = out.get("provider", pd.Series("", index=out.index)).fillna("").astype(str)
    out["coverage_status"] = out.get("coverage_status", pd.Series("", index=out.index)).fillna("").astype(str).str.upper()
    out["error_type"] = out.get("error_type", out["coverage_status"]).fillna(out["coverage_status"]).astype(str).str.upper()
    out.loc[out["error_type"].str.len().eq(0), "error_type"] = out.loc[out["error_type"].str.len().eq(0), "coverage_status"]
    out["error_summary"] = out.get("coverage_reason", out.get("error", pd.Series("", index=out.index))).fillna("").astype(str)
    out["attempts"] = pd.to_numeric(out.get("attempts", pd.Series(1, index=out.index)), errors="coerce").fillna(1).astype(int)
    out["last_seen"] = out.get("last_seen", out.get("updated_at", pd.Series("", index=out.index))).fillna("").astype(str)
    out["retry_candidate"] = out["error_type"].isin(["NETWORK_TIMEOUT", "PROVIDER_ERROR", "NO_PRICE_DATA"])
    columns = [
        "ticker",
        "provider_symbol",
        "exchange",
        "universe",
        "category",
        "provider",
        "error_type",
        "coverage_status",
        "error_summary",
        "attempts",
        "last_seen",
        "retry_candidate",
    ]
    return out[[col for col in columns if col in out.columns]].reset_index(drop=True)


def load_run_events(run_dir: str | Path, limit: int | None = 500) -> pd.DataFrame:
    """Load structured progress events for a run directory."""
    path = Path(run_dir).expanduser()
    if path.is_dir():
        path = path / "progress.jsonl"
    if not path.exists():
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return pd.DataFrame()
    if limit is not None:
        lines = lines[-int(limit) :]
    for line in lines:
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return pd.DataFrame(rows)


def _latest_timestamp_from_frame(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    for column in ["updated_at", "last_seen", "last_failure_at", "created_at", "finished_at"]:
        if column in frame.columns:
            values = pd.to_datetime(frame[column], errors="coerce", utc=True).dropna()
            if not values.empty:
                return values.max().isoformat()
    return ""


def _derive_equity_prices_health(
    manifest: pd.DataFrame,
    write_failures: pd.DataFrame,
    provider_failures: pd.DataFrame,
    running: dict[str, Any] | None = None,
    strict: bool = False,
) -> StageHealth:
    if running:
        return StageHealth(
            stage="equity_prices",
            status="RUNNING",
            coverage_ratio=0.0,
            covered_assets=0,
            total_assets=0,
            failure_count=len(write_failures) + len(provider_failures),
            critical_failure_count=0,
            warning_failure_count=0,
            last_run_at="",
            run_id=str(running.get("run_id", "")),
            pid=str(running.get("pid", "")),
            log_path=str(running.get("log_path", "")),
            lock_path=str(running.get("lock_path", "")),
        )
    if manifest.empty:
        return StageHealth("equity_prices", "MISSING", 0.0, 0, 0, len(write_failures) + len(provider_failures), 0, 0, "")

    total = int(manifest.get("ticker", manifest.get("provider_symbol", pd.Series(dtype=object))).fillna("").astype(str).str.len().gt(0).sum())
    total = max(total, len(manifest))
    coverage = manifest.get("coverage_status", pd.Series("", index=manifest.index)).fillna("").astype(str).str.upper()
    write_failed = int(len(write_failures))
    provider_error_type = provider_failures.get("error_type", pd.Series(dtype=object)).fillna("").astype(str).str.upper()
    critical = int(provider_error_type.isin(CRITICAL_OHLCV_FAILURES).sum()) + write_failed
    warning = int(provider_error_type.isin(WATCH_OHLCV_FAILURES).sum())
    usable_mask = ~coverage.isin(CRITICAL_OHLCV_FAILURES | {"NO_PRICE_DATA"})
    covered = int(usable_mask.sum()) if len(usable_mask) else 0
    ratio = round(covered / max(total, 1), 4)
    if critical / max(total, 1) >= 0.10 or (strict and critical > 0):
        status = "FAILED"
    elif ratio < 0.80:
        status = "FAILED"
    elif critical > 0 or ratio < 0.99 or warning / max(total, 1) > 0.05:
        status = "PARTIAL"
    else:
        status = "OK"
    return StageHealth(
        stage="equity_prices",
        status=status,
        coverage_ratio=ratio,
        covered_assets=covered,
        total_assets=total,
        failure_count=int(len(write_failures) + len(provider_failures)),
        critical_failure_count=critical,
        warning_failure_count=warning,
        last_run_at=_latest_timestamp_from_frame(manifest),
    )


def _derive_generic_health(stage: str, frame: pd.DataFrame, running: dict[str, Any] | None = None, strict: bool = False) -> StageHealth:
    if running:
        return StageHealth(
            stage=stage,
            status="RUNNING",
            coverage_ratio=0.0,
            covered_assets=0,
            total_assets=0,
            failure_count=0,
            critical_failure_count=0,
            warning_failure_count=0,
            last_run_at="",
            run_id=str(running.get("run_id", "")),
            pid=str(running.get("pid", "")),
            log_path=str(running.get("log_path", "")),
            lock_path=str(running.get("lock_path", "")),
        )
    status = _stage_status_from_manifest(frame, strict=strict)
    total = len(frame)
    failures = 0
    if not frame.empty and "status" in frame.columns:
        values = frame["status"].fillna("").astype(str).str.upper()
        failures = int(values.isin({"FAILED", "ERROR", "PROVIDER_LIMITED"}).sum())
    covered = int(pd.to_numeric(frame.get("rows", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not frame.empty else 0
    if failures and (failures / max(total, 1) >= 0.2 or strict):
        status = "FAILED"
    elif failures or status in {"PLANNED", "SKIPPED", "EMPTY"}:
        status = "PARTIAL" if status not in {"MISSING"} else status
    return StageHealth(
        stage=stage,
        status=status,
        coverage_ratio=1.0 if status == "OK" else 0.0,
        covered_assets=covered,
        total_assets=total,
        failure_count=failures,
        critical_failure_count=failures,
        warning_failure_count=0,
        last_run_at=_latest_timestamp_from_frame(frame),
    )


def get_data_health_summary(
    financial_db_root: str | Path | None = None,
    output_root: str | Path | None = None,
    start_year: int = 2000,
    end_year: int = 2026,
    strict: bool = False,
) -> pd.DataFrame:
    """Return stage-level health rows for Data Platform cards/tables."""
    roots = resolve_data_platform_roots(financial_db_root=financial_db_root, repo_output_root=output_root)
    running = _running_stages(roots.repo_output)
    failures = list_ohlcv_failures(roots.financial_db, roots.repo_output)
    provider_failures = list_ohlcv_provider_failures(roots.financial_db, roots.repo_output)
    parquet_info = get_ohlcv_parquet_root_info(roots.financial_db, roots.repo_output)
    parquet_files = list((parquet_info.path / "daily").glob("*/*.parquet")) if (parquet_info.path / "daily").exists() else []
    completion_dir = roots.repo_output / "data_completion"
    rows: list[dict[str, Any]] = []
    for stage in DATA_COMPLETION_STAGES:
        path = completion_dir / f"{stage}_2000_2026.csv"
        frame = _read_csv(path)
        if stage == "equity_prices":
            manifest_for_health = _read_csv(roots.repo_output / "tables" / "OHLCV_daily_manifest.csv")
            health = _derive_equity_prices_health(
                manifest_for_health,
                failures,
                provider_failures,
                running=running.get(stage),
                strict=strict,
            )
        else:
            health = _derive_generic_health(stage, frame, running=running.get(stage), strict=strict)
        status = health.status
        covered_assets = 0
        coverage_breakdown: dict[str, Any] = {}
        if stage == "equity_prices":
            manifest = _read_csv(roots.repo_output / "tables" / "OHLCV_daily_manifest.csv")
            if not manifest.empty:
                covered_assets = int(manifest.get("ticker", pd.Series(dtype=object)).dropna().astype(str).str.upper().nunique())
                group_col = "universe" if "universe" in manifest.columns else "exchange" if "exchange" in manifest.columns else "coverage_status" if "coverage_status" in manifest.columns else "status"
                if group_col in manifest.columns:
                    coverage_breakdown = manifest.groupby(group_col, dropna=False)["ticker"].nunique().to_dict() if "ticker" in manifest.columns else manifest[group_col].value_counts().to_dict()
            if parquet_files:
                parquet_breakdown = pd.Series([path.parent.name for path in parquet_files]).value_counts().to_dict()
                covered_assets = max(covered_assets, len(parquet_files))
                if not coverage_breakdown or sum(int(value) for value in coverage_breakdown.values() if pd.notna(value)) < len(parquet_files):
                    coverage_breakdown = parquet_breakdown
            covered_assets = max(covered_assets, int(health.covered_assets or 0))
        elif stage == "equity_fundamentals" and not frame.empty:
            covered_assets = int(pd.to_numeric(frame.get("rows", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
            if "universe" in frame.columns:
                coverage_breakdown = frame.set_index("universe")["rows"].to_dict() if "rows" in frame.columns else frame["universe"].value_counts().to_dict()
        elif not frame.empty:
            covered_assets = int(pd.to_numeric(frame.get("rows", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        rows.append(
            {
                "stage": stage,
                "status": status,
                "rows": len(frame),
                "covered_assets": covered_assets,
                "total_assets": health.total_assets,
                "coverage_ratio": health.coverage_ratio,
                "coverage_breakdown": json.dumps(_json_safe_mapping(coverage_breakdown), default=str, sort_keys=True),
                "failure_count": health.failure_count,
                "critical_failure_count": health.critical_failure_count,
                "warning_failure_count": health.warning_failure_count,
                "provider_failure_count": len(provider_failures) if stage == "equity_prices" else 0,
                "write_failure_count": len(failures) if stage == "equity_prices" else 0,
                "ohlcv_parquet_root": str(parquet_info.path) if stage == "equity_prices" else "",
                "ohlcv_parquet_storage_mode": parquet_info.storage_mode if stage == "equity_prices" else "",
                "ohlcv_parquet_files": len(parquet_files) if stage == "equity_prices" else pd.NA,
                "run_id": running.get(stage, {}).get("run_id", ""),
                "pid": running.get(stage, {}).get("pid", ""),
                "log_path": running.get(stage, {}).get("log_path", ""),
                "lock_path": running.get(stage, {}).get("lock_path", ""),
                "locked_by": running.get(stage, {}).get("locked_by", ""),
                "lock_acquired_at": running.get(stage, {}).get("lock_acquired_at", ""),
                "manifest_path": str(path),
                "start_year": start_year,
                "end_year": end_year,
                "last_run_at": health.last_run_at,
                "updated_at": utc_now(),
            }
        )
    return pd.DataFrame(rows)


def _normalize_ticker(value: Any) -> str:
    return str(value or "").strip().upper()


def _status_from_counts(total: int, critical: int, warnings: int, covered: int, running: bool = False) -> str:
    if running:
        return "RUNNING"
    if total <= 0:
        return "MISSING"
    coverage_ratio = covered / max(total, 1)
    if critical / max(total, 1) >= 0.10 or coverage_ratio < 0.80:
        return "FAILED"
    if critical > 0 or warnings > 0 or coverage_ratio < 0.99:
        return "PARTIAL"
    return "OK"


def get_stage_health_for_universes(
    stage: str,
    universes: list[str] | tuple[str, ...] | None = None,
    financial_db_root: str | Path | None = None,
    output_root: str | Path | None = None,
    strict: bool = False,
) -> pd.DataFrame:
    """Return stage health scoped to selected universes when manifests expose it.

    If the stage manifest is not universe-aware, the function returns the global
    StageHealth row with a note so app pages can still show a truthful banner.
    """
    roots = resolve_data_platform_roots(financial_db_root=financial_db_root, repo_output_root=output_root)
    stage = str(stage)
    requested = [str(item).strip() for item in (universes or []) if str(item).strip()]
    global_health = get_data_health_summary(roots.financial_db, roots.repo_output, strict=strict)
    global_row = global_health[global_health["stage"].astype(str).eq(stage)]
    global_payload = global_row.iloc[0].to_dict() if not global_row.empty else {"stage": stage, "status": "MISSING"}
    running = str(global_payload.get("status", "")).upper() == "RUNNING"

    if not requested:
        out = pd.DataFrame([{**global_payload, "universe": "all", "requested_universe": ""}])
        return out.reset_index(drop=True)

    rows: list[dict[str, Any]] = []
    if stage == "equity_prices":
        manifest = _read_csv(roots.repo_output / "tables" / "OHLCV_daily_manifest.csv")
        if manifest.empty:
            return pd.DataFrame([{**global_payload, "universe": ",".join(requested), "requested_universe": ",".join(requested), "note": "OHLCV manifest missing"}])
        candidate_cols = [col for col in ["universe", "market", "exchange", "primary_source", "provider"] if col in manifest.columns]
        for universe in requested:
            mask = pd.Series(False, index=manifest.index)
            for col in candidate_cols:
                mask = mask | manifest[col].fillna("").astype(str).str.lower().eq(universe.lower())
            frame = manifest[mask].copy()
            note = ""
            if frame.empty:
                frame = manifest.copy()
                note = "No universe-level manifest match; using global OHLCV coverage."
            coverage = frame.get("coverage_status", pd.Series("", index=frame.index)).fillna("").astype(str).str.upper()
            total = int(frame.get("ticker", frame.get("provider_symbol", pd.Series(dtype=object))).fillna("").astype(str).str.len().gt(0).sum())
            total = max(total, len(frame))
            critical = int(coverage.isin(CRITICAL_OHLCV_FAILURES).sum())
            warnings = int(coverage.isin(WATCH_OHLCV_FAILURES).sum())
            covered = int((~coverage.isin(CRITICAL_OHLCV_FAILURES | {"NO_PRICE_DATA"})).sum())
            rows.append(
                {
                    "stage": stage,
                    "universe": universe,
                    "requested_universe": universe,
                    "status": _status_from_counts(total, critical, warnings, covered, running=running),
                    "covered_assets": covered,
                    "total_assets": total,
                    "coverage_ratio": round(covered / max(total, 1), 4),
                    "failure_count": critical + warnings,
                    "critical_failure_count": critical,
                    "warning_failure_count": warnings,
                    "run_id": global_payload.get("run_id", ""),
                    "pid": global_payload.get("pid", ""),
                    "last_run_at": global_payload.get("last_run_at", ""),
                    "note": note,
                }
            )
        return pd.DataFrame(rows).reset_index(drop=True)

    path = roots.repo_output / "data_completion" / f"{stage}_2000_2026.csv"
    frame = _read_csv(path)
    if frame.empty or "universe" not in frame.columns:
        return pd.DataFrame([{**global_payload, "universe": ",".join(requested), "requested_universe": ",".join(requested), "note": "Stage manifest is not universe-aware"}])
    for universe in requested:
        view = frame[frame["universe"].fillna("").astype(str).str.lower().eq(universe.lower())].copy()
        note = ""
        if view.empty:
            note = "Universe is not present in stage manifest."
        statuses = view.get("status", pd.Series(dtype=object)).fillna("").astype(str).str.upper()
        total = len(view)
        critical = int(statuses.isin({"FAILED", "ERROR"}).sum())
        warnings = int(statuses.isin({"PROVIDER_LIMITED", "EMPTY", "SKIPPED", "PLANNED"}).sum())
        covered = int(pd.to_numeric(view.get("rows", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not view.empty else 0
        status = "MISSING" if view.empty else _status_from_counts(max(total, 1), critical, warnings, max(total - critical, 0), running=running)
        rows.append(
            {
                "stage": stage,
                "universe": universe,
                "requested_universe": universe,
                "status": status,
                "covered_assets": covered,
                "total_assets": total,
                "coverage_ratio": 1.0 if status == "OK" else 0.0,
                "failure_count": critical + warnings,
                "critical_failure_count": critical,
                "warning_failure_count": warnings,
                "run_id": global_payload.get("run_id", ""),
                "pid": global_payload.get("pid", ""),
                "last_run_at": _latest_timestamp_from_frame(view) or global_payload.get("last_run_at", ""),
                "note": note,
            }
        )
    return pd.DataFrame(rows).reset_index(drop=True)


def _price_parquet_exists(ticker: str, financial_db_root: Path, output_root: Path) -> bool:
    clean = _normalize_ticker(ticker).replace("/", "_")
    if not clean:
        return False
    candidates = {clean, clean.replace(".", "_").replace("-", "_")}
    for daily_root in get_ohlcv_daily_search_roots(financial_db_root, output_root):
        if not daily_root.exists():
            continue
        for name in candidates:
            try:
                if next(daily_root.glob(f"*/{name}.parquet"), None) is not None:
                    return True
            except Exception:
                continue
    return False


def _fundamentals_exist(ticker: str, financial_db_root: Path) -> str:
    clean = _normalize_ticker(ticker)
    if not clean:
        return "MISSING"
    equities = financial_db_root / "Equities"
    if not equities.exists():
        return "UNKNOWN"
    safe_names = {clean.replace(".", "_").replace("-", "_"), clean}
    for safe in safe_names:
        try:
            if next(equities.glob(f"*/*/fundamentals/{safe}_*.parquet"), None) is not None:
                return "OK"
        except Exception:
            return "UNKNOWN"
    return "MISSING"


def get_data_status_for_tickers(
    tickers: list[str] | tuple[str, ...],
    financial_db_root: str | Path | None = None,
    output_root: str | Path | None = None,
) -> dict[str, TickerDataStatus]:
    """Return ticker-level data readiness from manifests and lightweight file checks."""
    roots = resolve_data_platform_roots(financial_db_root=financial_db_root, repo_output_root=output_root)
    requested = list(dict.fromkeys(_normalize_ticker(ticker) for ticker in tickers if _normalize_ticker(ticker)))
    if not requested:
        return {}
    manifest = _read_csv(roots.repo_output / "tables" / "OHLCV_daily_manifest.csv")
    provider_failures = list_ohlcv_provider_failures(roots.financial_db, roots.repo_output)
    if not manifest.empty:
        for col in ["ticker", "provider_symbol"]:
            if col not in manifest.columns:
                manifest[col] = ""
        manifest["_ticker_norm"] = manifest["ticker"].fillna("").astype(str).str.upper()
        manifest["_provider_norm"] = manifest["provider_symbol"].fillna("").astype(str).str.upper()
    if not provider_failures.empty:
        provider_failures["_ticker_norm"] = provider_failures["ticker"].fillna("").astype(str).str.upper()
        provider_failures["_provider_norm"] = provider_failures["provider_symbol"].fillna("").astype(str).str.upper()

    out: dict[str, TickerDataStatus] = {}
    for ticker in requested:
        price_rows = pd.DataFrame()
        if not manifest.empty:
            price_rows = manifest[(manifest["_ticker_norm"].eq(ticker)) | (manifest["_provider_norm"].eq(ticker))].copy()
        failure_rows = pd.DataFrame()
        if not provider_failures.empty:
            failure_rows = provider_failures[(provider_failures["_ticker_norm"].eq(ticker)) | (provider_failures["_provider_norm"].eq(ticker))].copy()
        has_parquet = _price_parquet_exists(ticker, roots.financial_db, roots.repo_output)
        coverage_status = ""
        last_price = ""
        if not price_rows.empty:
            coverage_values = price_rows.get("coverage_status", pd.Series("", index=price_rows.index)).fillna("").astype(str).str.upper()
            coverage_status = next((value for value in coverage_values if value), "")
            last_values = pd.to_datetime(price_rows.get("last_price_date", pd.Series(dtype=object)), errors="coerce").dropna()
            if not last_values.empty:
                last_price = last_values.max().date().isoformat()
        provider_error = ""
        if not failure_rows.empty and "error_type" in failure_rows.columns:
            provider_error = str(failure_rows["error_type"].fillna("").astype(str).iloc[0]).upper()
        if has_parquet or coverage_status in {"OK", "LIMITED_HISTORY", "DELISTED"}:
            prices_status = "PARTIAL" if coverage_status in {"LIMITED_HISTORY", "DELISTED"} else "OK"
        elif coverage_status in {"NETWORK_TIMEOUT", "PROVIDER_ERROR", "INVALID_SYMBOL"} or provider_error in CRITICAL_OHLCV_FAILURES:
            prices_status = "FAILED"
        elif coverage_status in {"NO_PRICE_DATA"} or provider_error in {"NO_PRICE_DATA"}:
            prices_status = "MISSING"
        else:
            prices_status = "MISSING"
        fundamentals_status = _fundamentals_exist(ticker, roots.financial_db)
        if prices_status == "FAILED":
            overall = "FAILED"
        elif prices_status == "MISSING" and fundamentals_status in {"MISSING", "UNKNOWN"}:
            overall = "FAILED"
        elif prices_status == "OK" and fundamentals_status == "OK":
            overall = "OK"
        elif prices_status == "OK" and fundamentals_status == "UNKNOWN":
            overall = "PARTIAL"
        elif prices_status in {"OK", "PARTIAL"}:
            overall = "PARTIAL"
        else:
            overall = "FAILED"
        issues = []
        if fundamentals_status != "OK":
            issues.append(f"fundamentals {fundamentals_status.lower()}")
        if prices_status != "OK":
            issues.append(f"prices {prices_status.lower()}")
        if provider_error:
            issues.append(provider_error)
        message = "OK" if not issues else "; ".join(issues)
        out[ticker] = TickerDataStatus(
            ticker=ticker,
            overall_status=overall,
            fundamentals_status=fundamentals_status,
            prices_status=prices_status,
            price_coverage_status=coverage_status,
            provider_error_type=provider_error,
            has_price_parquet=has_parquet,
            last_price_date=last_price,
            message=message,
        )
    return out


def _assets_from_failures(failures: pd.DataFrame) -> pd.DataFrame:
    if failures.empty:
        return pd.DataFrame()
    frame = failures[failures.get("retry_candidate", pd.Series(True, index=failures.index)).astype(bool)].copy()
    if frame.empty:
        return pd.DataFrame()
    assets = pd.DataFrame(
        {
            "ticker": frame["ticker"].fillna(frame["provider_symbol"]).astype(str).str.upper(),
            "provider_symbol": frame["provider_symbol"].fillna(frame["ticker"]).astype(str),
            "exchange": frame.get("exchange", pd.Series("unknown", index=frame.index)).replace("", "unknown").fillna("unknown"),
            "country": frame.get("country", pd.Series("", index=frame.index)).fillna(""),
            "type": frame.get("type", pd.Series("stock", index=frame.index)).replace("", "stock").fillna("stock"),
            "name": frame.get("name", frame["ticker"]).fillna(frame["ticker"]).astype(str),
            "primary_source": frame.get("primary_source", pd.Series("retry_failed", index=frame.index)).replace("", "retry_failed").fillna("retry_failed"),
            "active_flag": True,
        }
    )
    return assets.drop_duplicates(["ticker", "provider_symbol", "exchange"]).reset_index(drop=True)


def restart_equity_prices(stage_config: dict[str, Any] | None = None, failed_only: bool | None = None) -> pd.DataFrame:
    """Programmatic restart for the `equity_prices` stage.

    Parameters are passed through `stage_config` to keep this easy to call from
    Streamlit, notebooks or tests. Supported keys include financial_db_root,
    output_root, parquet_root, markets, universes, start_year, end_year,
    max_assets, batch_size, mode, failed_only and dry_run.
    """
    cfg = dict(stage_config or {})
    if failed_only is None:
        failed_only = bool(cfg.get("failed_only", False))
    financial_db_root = cfg.get("financial_db_root")
    output_root = cfg.get("output_root")
    roots = resolve_data_platform_roots(financial_db_root=financial_db_root, repo_output_root=output_root)
    parquet_root = cfg.get("parquet_root") or cfg.get("ohlcv_parquet_root")
    start_year = int(cfg.get("start_year", 2000))
    end_year = int(cfg.get("end_year", 2026))
    markets = cfg.get("markets") or cfg.get("universes") or ["us_all", "europe_major", "japan_major", "global_etfs"]
    if isinstance(markets, str):
        markets = [item.strip() for item in markets.split(",") if item.strip()]
    max_assets = cfg.get("max_assets")
    max_assets = None if max_assets in {None, "", 0, "0"} else int(max_assets)
    batch_size = int(cfg.get("batch_size", 80))
    mode = str(cfg.get("mode", "full"))
    dry_run = bool(cfg.get("dry_run", False))
    run_id = str(cfg.get("run_id") or os.environ.get("RESEARCH_PLATFORM_RUN_ID") or "")
    run_events_path = cfg.get("run_events_path")
    lock_root_for_metadata = parquet_root or get_ohlcv_parquet_root_info(roots.financial_db, roots.repo_output).path

    assets_override = None
    if failed_only:
        failures = list_ohlcv_failures(roots.financial_db, roots.repo_output)
        tickers = cfg.get("tickers") or cfg.get("assets") or cfg.get("symbols")
        if isinstance(tickers, str):
            tickers = [item.strip().upper() for item in tickers.split(",") if item.strip()]
        if tickers:
            ticker_set = {str(item).upper() for item in tickers}
            failures = failures[
                failures["ticker"].astype(str).str.upper().isin(ticker_set)
                | failures["provider_symbol"].astype(str).str.upper().isin(ticker_set)
            ].copy()
        assets_override = _assets_from_failures(failures)
        if assets_override.empty:
            return pd.DataFrame([{"stage": "equity_prices", "status": "NO_FAILURES_TO_RETRY", "updated_at": utc_now()}])

    lock = acquire_stage_lock("equity_prices", roots.repo_output, run_id=run_id or None, protected_root=lock_root_for_metadata)
    if lock is None:
        existing = read_stage_lock("equity_prices", roots.repo_output)
        return pd.DataFrame(
            [
                {
                    "stage": "equity_prices",
                    "status": "LOCKED_RUNNING",
                    "locked_by": existing.run_id if existing else "",
                    "lock_path": existing.lock_path if existing else "",
                    "pid": existing.pid if existing else "",
                    "updated_at": utc_now(),
                }
            ]
        )
    try:
        job = OhlcvIngestJob(
            financial_db_root=roots.financial_db,
            output_root=roots.repo_output,
            parquet_root=parquet_root,
            run_id=lock.run_id,
            run_events_path=run_events_path,
        )
        manifest = job.run_daily(
            markets=list(markets),
            start_date=f"{start_year}-01-01",
            end_date=f"{end_year}-12-31",
            mode=mode,
            max_assets=max_assets,
            batch_size=batch_size,
            dry_run=dry_run,
            assets_override=assets_override,
        )
    finally:
        release_stage_lock("equity_prices", roots.repo_output, lock_info=lock)
    summary = summarize_ohlcv_manifest(manifest)
    out_dir = roots.repo_output / "data_completion"
    out_dir.mkdir(parents=True, exist_ok=True)
    restart_path = out_dir / f"equity_prices_restart_{pd.Timestamp.now(tz='UTC').strftime('%Y%m%d_%H%M%S')}.csv"
    summary.to_csv(restart_path, index=False)
    return manifest


def restart_equity_prices_failed_only(stage_config: dict[str, Any] | None = None) -> pd.DataFrame:
    cfg = dict(stage_config or {})
    cfg["failed_only"] = True
    return restart_equity_prices(cfg)
