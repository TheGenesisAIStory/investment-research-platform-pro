"""Batch export helpers for Drive-first dataset inventories."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from .data_platform import read_dataset, utc_now


EXPORT_FORMATS = ["csv", "parquet", "json", "sqlite", "excel", "original"]
TABULAR_SUFFIXES = {".csv", ".parquet", ".xlsx", ".xls", ".json"}
PROVIDER_KEYWORDS = {
    "yfinance": ["yfinance", "yf_", "europe_stoxx_companies", "local_market_data"],
    "alpha_vantage": ["alpha_vantage", "alphavantage"],
    "tiingo": ["tiingo"],
    "fmp": ["financial_modeling_prep", "fmp"],
    "polygon": ["polygon"],
    "intrinio": ["intrinio"],
    "fred": ["fred"],
    "bls": ["bls"],
    "cftc": ["cftc", "cot"],
    "congress_gov": ["congress", "policy"],
    "econdb": ["econdb"],
    "eia": ["eia", "energy"],
    "nasdaq_data_link": ["nasdaq", "quandl"],
    "aqr": ["aqr", "factor_library"],
    "drive": ["analysis_outputs", "notebook_exports", "catalog"],
}


def slugify(value: object, fallback: str = "item") -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._-")
    return text[:120] or fallback


def dataset_id(path: object) -> str:
    text = str(path or "")
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def infer_provider(path: object) -> str:
    text = str(path or "").lower()
    for provider, tokens in PROVIDER_KEYWORDS.items():
        if any(token in text for token in tokens):
            return provider
    parts = [p for p in re.split(r"[\\/]", text) if p]
    return slugify(parts[0], "unknown") if parts else "unknown"


def enrich_inventory_for_export(inventory: pd.DataFrame) -> pd.DataFrame:
    if inventory.empty:
        return inventory.copy()
    out = inventory.copy()
    path_col = "path" if "path" in out.columns else "relative_path"
    out["dataset_id"] = out[path_col].map(dataset_id)
    out["provider"] = out.get("provider", out[path_col].map(infer_provider))
    if "role" not in out.columns:
        out["role"] = "other"
    if "size_mb" in out.columns:
        out["size_mb"] = pd.to_numeric(out["size_mb"], errors="coerce").fillna(0.0)
    else:
        out["size_mb"] = 0.0
    if "age_hours" in out.columns:
        age = pd.to_numeric(out["age_hours"], errors="coerce")
        out["freshness_bucket"] = pd.cut(
            age,
            bins=[-1, 48, 168, 720, float("inf")],
            labels=["fresh", "weekly", "stale", "old"],
        ).astype(str)
    else:
        out["freshness_bucket"] = "unknown"
    return out


def estimate_batch_size(inventory: pd.DataFrame) -> dict[str, Any]:
    if inventory.empty:
        return {"files": 0, "size_mb": 0.0, "roles": 0, "providers": 0}
    return {
        "files": int(len(inventory)),
        "size_mb": round(float(pd.to_numeric(inventory.get("size_mb", 0), errors="coerce").fillna(0).sum()), 3),
        "roles": int(inventory["role"].nunique()) if "role" in inventory else 0,
        "providers": int(inventory["provider"].nunique()) if "provider" in inventory else 0,
    }


def _safe_read_tabular(path: Path) -> pd.DataFrame:
    try:
        return read_dataset(path)
    except Exception:
        return pd.DataFrame()


def _write_frame(df: pd.DataFrame, target: Path, export_format: str) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if export_format == "csv":
        df.to_csv(target.with_suffix(".csv"), index=False)
        return str(target.with_suffix(".csv"))
    if export_format == "parquet":
        df.to_parquet(target.with_suffix(".parquet"), index=False)
        return str(target.with_suffix(".parquet"))
    if export_format == "json":
        df.to_json(target.with_suffix(".json"), orient="records", indent=2, date_format="iso")
        return str(target.with_suffix(".json"))
    raise ValueError(f"Unsupported frame export format: {export_format}")


def _copy_or_link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    try:
        os.link(source, target)
    except Exception:
        shutil.copy2(source, target)


def _write_sqlite_frame(conn: sqlite3.Connection, df: pd.DataFrame, table_name: str) -> None:
    if df.empty:
        return
    df.to_sql(table_name, conn, if_exists="replace", index=False)


def _write_excel_groups(rows: list[dict[str, Any]], export_root: Path) -> list[dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    role_groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        role_groups.setdefault(slugify(row.get("role"), "other"), []).append(row)
    for role, role_rows in role_groups.items():
        workbook = export_root / "by_role" / role / f"{role}_export.xlsx"
        workbook.parent.mkdir(parents=True, exist_ok=True)
        try:
            with pd.ExcelWriter(workbook) as writer:
                manifest_rows = []
                for idx, row in enumerate(role_rows[:60], start=1):
                    source = Path(str(row.get("path", "")))
                    df = _safe_read_tabular(source) if source.suffix.lower() in TABULAR_SUFFIXES else pd.DataFrame()
                    if df.empty:
                        continue
                    sheet = slugify(row.get("file") or source.stem, f"sheet_{idx}")[:31]
                    df.head(50000).to_excel(writer, sheet_name=sheet, index=False)
                    manifest_rows.append({**row, "excel_sheet": sheet})
                pd.DataFrame(manifest_rows or role_rows).to_excel(writer, sheet_name="_manifest", index=False)
            manifests.append({"role": role, "path": str(workbook), "status": "written"})
        except Exception as exc:
            manifests.append({"role": role, "path": str(workbook), "status": "failed", "error": str(exc)})
    return manifests


def create_batch_download(
    inventory: pd.DataFrame,
    output_root: Path | str,
    export_format: str = "csv",
    selected_ids: Iterable[str] | None = None,
    max_files: int | None = None,
    run_label: str | None = None,
    progress_callback: Callable[[int, int, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Export selected inventory rows into a timestamped batch_downloads folder."""
    if export_format not in EXPORT_FORMATS:
        raise ValueError(f"export_format must be one of {EXPORT_FORMATS}")

    data = enrich_inventory_for_export(inventory)
    if selected_ids is not None:
        selected = set(selected_ids)
        data = data[data["dataset_id"].isin(selected)]
    if max_files is not None and max_files > 0:
        data = data.head(max_files)

    output_root = Path(output_root).expanduser()
    timestamp = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
    label = slugify(run_label or f"{timestamp}_{export_format}_export")
    export_root = output_root / "batch_downloads" / label
    by_role_root = export_root / "by_role"
    by_provider_root = export_root / "by_provider"
    by_role_root.mkdir(parents=True, exist_ok=True)
    by_provider_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    started = time.time()

    if export_format == "sqlite":
        sqlite_path = export_root / "batch_export.sqlite"
        conn = sqlite3.connect(sqlite_path)
    else:
        conn = None

    try:
        total = len(data)
        if export_format == "excel":
            excel_manifest = _write_excel_groups(data.to_dict(orient="records"), export_root)
            rows.extend(excel_manifest)
            total = max(total, 1)
            if progress_callback:
                progress_callback(total, total, {"status": "excel_written"})
        else:
            for current, (_, item) in enumerate(data.iterrows(), start=1):
                source = Path(str(item.get("path", ""))).expanduser()
                role = slugify(item.get("role"), "other")
                provider = slugify(item.get("provider"), "unknown")
                name = slugify(source.stem or item.get("file") or item.get("dataset_id"))
                status = "pending"
                target_path = ""
                error = ""
                rows_existing = 0
                try:
                    if not source.exists() or not source.is_file():
                        status = "missing_source"
                    elif export_format == "original":
                        target = by_role_root / role / source.name
                        shutil.copy2(source, target)
                        _copy_or_link(target, by_provider_root / provider / source.name)
                        target_path = str(target)
                        status = "copied_original"
                    elif export_format == "sqlite":
                        df = _safe_read_tabular(source) if source.suffix.lower() in TABULAR_SUFFIXES else pd.DataFrame()
                        if df.empty:
                            status = "skipped_non_tabular"
                        else:
                            table = f"{provider}_{role}_{name}_{item.get('dataset_id')}"
                            table = slugify(table).replace(".", "_")[:63]
                            _write_sqlite_frame(conn, df, table)  # type: ignore[arg-type]
                            target_path = str(sqlite_path)
                            rows_existing = len(df)
                            status = "written_sqlite"
                    else:
                        df = _safe_read_tabular(source) if source.suffix.lower() in TABULAR_SUFFIXES else pd.DataFrame()
                        if df.empty:
                            target = by_role_root / role / source.name
                            shutil.copy2(source, target)
                            _copy_or_link(target, by_provider_root / provider / source.name)
                            target_path = str(target)
                            status = "copied_raw"
                        else:
                            base = by_role_root / role / f"{name}_{item.get('dataset_id')}"
                            target_path = _write_frame(df, base, export_format)
                            rows_existing = len(df)
                            provider_target = by_provider_root / provider / Path(target_path).name
                            _copy_or_link(Path(target_path), provider_target)
                            status = f"written_{export_format}"
                except Exception as exc:
                    status = "failed"
                    error = str(exc)

                manifest_row = {
                    **item.to_dict(),
                    "source_path": str(source),
                    "export_path": target_path,
                    "export_format": export_format,
                    "status": status,
                    "rows": rows_existing,
                    "error": error,
                    "updated_at": utc_now(),
                }
                rows.append(manifest_row)
                if progress_callback:
                    progress_callback(current, total, manifest_row)
    finally:
        if conn is not None:
            conn.close()

    manifest = pd.DataFrame(rows)
    manifest_path = export_root / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    metadata = {
        "generated_at": utc_now(),
        "run_label": label,
        "export_root": str(export_root),
        "export_format": export_format,
        "summary": estimate_batch_size(data),
        "elapsed_seconds": round(time.time() - started, 2),
        "status_counts": manifest["status"].value_counts().to_dict() if "status" in manifest else {},
        "manifest": str(manifest_path),
    }
    (export_root / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return {"metadata": metadata, "manifest": manifest, "export_root": export_root, "manifest_path": manifest_path}
