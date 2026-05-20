"""Drive-first data platform helpers for the research notebooks and app."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


DEFAULT_FINANCIAL_DB_CANDIDATES = [
    Path("/content/drive/MyDrive/Database Finanziario"),
    Path("/content/drive/MyDrive/GitHub/Database Finanziario"),
    Path.home() / "Library/CloudStorage/GoogleDrive-sfn.gns@gmail.com/Il mio Drive/Database Finanziario",
]

ROLE_KEYWORDS = {
    "prices": ["price", "prices", "ohlcv", "market", "stoxx", "equity", "_1d"],
    "fundamentals": ["fundamental", "financial", "income", "balance", "cashflow", "ratio", "valuation"],
    "macro_risk": ["macro", "risk", "factor", "fama", "fred", "rate", "yield", "fx", "currency", "commodity"],
    "alternative": ["alternative", "stocktwits", "polymarket", "news", "sentiment", "earnings"],
    "portfolio": ["portfolio", "allocation", "backtest", "risk"],
    "screener": ["screener", "screening", "universe"],
    "provider_registry": ["provider", "credential", "api"],
    "notebook_export": ["notebook_exports", "analysis_outputs", "dashboard", "report"],
}

PROVIDER_FALLBACKS = {
    "prices": ["drive:europe_stoxx_companies", "drive:local_market_data", "cache:repo", "api:yfinance", "api:stooq"],
    "fundamentals": ["drive:fundamentals", "drive:company_valuation_platform", "cache:repo", "api:yfinance", "api:fmp", "api:finnhub", "api:alpha_vantage"],
    "macro_risk": ["drive:ml_trading_risk_factors_csv", "drive:alpha_factor_library", "cache:repo", "api:fred", "api:yfinance"],
    "alternative": ["drive:alternative_data", "cache:repo"],
    "provider_registry": ["drive:catalog", "cache:repo"],
}


@dataclass(frozen=True)
class DataPlatformRoots:
    financial_db: Path
    local_cache: Path
    repo_output: Path
    available: bool
    source: str


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def discover_financial_database_root(extra_candidates: Iterable[Path | str] | None = None) -> tuple[Path, str, bool]:
    """Resolve the canonical Financial Database root across local Mac and Colab."""
    env_keys = ["FINANCIAL_DB_ROOT", "GOOGLE_DRIVE_DB_ROOT", "ML_TRADING_DB_BASE", "DB_BASE", "DATA_PATH"]
    for key in env_keys:
        value = os.environ.get(key)
        if value:
            path = Path(value).expanduser()
            return path, f"env:{key}", path.exists()
    candidates = [Path(p).expanduser() for p in (extra_candidates or [])]
    candidates.extend(DEFAULT_FINANCIAL_DB_CANDIDATES)
    for path in candidates:
        if path.exists():
            return path, "discovered", True
    fallback = candidates[0] if candidates else Path.cwd() / "Database Finanziario"
    return fallback, "fallback_missing", False


def resolve_data_platform_roots(
    financial_db_root: Path | str | None = None,
    local_cache_root: Path | str | None = None,
    repo_output_root: Path | str | None = None,
) -> DataPlatformRoots:
    if financial_db_root is None:
        root, source, available = discover_financial_database_root()
    else:
        root = Path(financial_db_root).expanduser()
        source = "explicit"
        available = root.exists()
    local_cache = Path(local_cache_root or os.environ.get("RESEARCH_PLATFORM_LOCAL_CACHE", Path.cwd() / "output" / "data_cache")).expanduser()
    repo_output = Path(repo_output_root or os.environ.get("RESEARCH_PLATFORM_OUTPUT_ROOT", Path.cwd() / "output")).expanduser()
    local_cache.mkdir(parents=True, exist_ok=True)
    repo_output.mkdir(parents=True, exist_ok=True)
    return DataPlatformRoots(root, local_cache, repo_output, available, source)


def infer_dataset_role(path: Path) -> str:
    text = str(path).lower()
    for role, words in ROLE_KEYWORDS.items():
        if any(word in text for word in words):
            return role
    return "other"


def _safe_relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except Exception:
        return str(path)


def build_dataset_inventory(root: Path, max_files: int | None = None) -> pd.DataFrame:
    """Inventory real files under the Financial Database without assuming schema."""
    root = Path(root).expanduser()
    if not root.exists():
        return pd.DataFrame([{"path": str(root), "exists": False, "role": "missing_root", "updated_at": utc_now()}])
    rows: list[dict[str, Any]] = []
    suffixes = {".csv", ".parquet", ".json", ".xlsx", ".xls", ".html", ".md", ".sqlite", ".rdata", ".rda"}
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in suffixes and ".Rproj.user" not in str(p)]
    files = sorted(files, key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    if max_files is not None:
        files = files[:max_files]
    for path in files:
        stat = path.stat()
        modified = pd.Timestamp(stat.st_mtime, unit="s", tz="UTC")
        rows.append({
            "relative_path": _safe_relative(path, root),
            "path": str(path),
            "file": path.name,
            "suffix": path.suffix.lower(),
            "role": infer_dataset_role(path),
            "size_mb": round(stat.st_size / (1024 * 1024), 4),
            "modified_utc": modified.isoformat(),
            "age_hours": round((pd.Timestamp.utcnow() - modified).total_seconds() / 3600, 2),
            "parent": _safe_relative(path.parent, root),
            "exists": True,
        })
    return pd.DataFrame(rows)


def load_catalog_tables(root: Path) -> dict[str, pd.DataFrame]:
    catalog = Path(root) / "catalog"
    names = {
        "data_inventory": "data_inventory.csv",
        "data_domains": "data_domains.csv",
        "api_providers": "api_providers.csv",
        "api_provider_health_checks": "api_provider_health_checks.csv",
        "credential_status_masked": "credential_status_masked.csv",
    }
    out: dict[str, pd.DataFrame] = {}
    for key, name in names.items():
        path = catalog / name
        if path.exists():
            try:
                out[key] = pd.read_csv(path)
            except Exception:
                out[key] = pd.DataFrame()
        else:
            out[key] = pd.DataFrame()
    return out


def summarize_inventory(inventory: pd.DataFrame) -> pd.DataFrame:
    if inventory.empty or "role" not in inventory.columns:
        return pd.DataFrame()
    summary = (
        inventory.groupby("role", dropna=False)
        .agg(files=("file", "count"), size_mb=("size_mb", "sum"), newest_age_hours=("age_hours", "min"), oldest_age_hours=("age_hours", "max"))
        .reset_index()
        .sort_values(["files", "size_mb"], ascending=False)
    )
    summary["freshness_status"] = pd.cut(
        pd.to_numeric(summary["newest_age_hours"], errors="coerce"),
        bins=[-1, 24 * 2, 24 * 7, 24 * 30, float("inf")],
        labels=["fresh", "weekly", "stale", "old"],
    ).astype(str)
    return summary


def dataset_status(root: Path, max_files: int | None = 5000) -> dict[str, pd.DataFrame]:
    inventory = build_dataset_inventory(root, max_files=max_files)
    catalog = load_catalog_tables(root)
    summary = summarize_inventory(inventory)
    return {"inventory": inventory, "summary": summary, **catalog}


def provider_fallback_plan(root: Path) -> pd.DataFrame:
    """Build an explicit provider/source fallback plan from the real Drive catalog."""
    catalog = load_catalog_tables(root)
    providers = catalog.get("api_providers", pd.DataFrame())
    credentials = catalog.get("credential_status_masked", pd.DataFrame())
    health = catalog.get("api_provider_health_checks", pd.DataFrame())
    rows: list[dict[str, Any]] = []
    for domain, chain in PROVIDER_FALLBACKS.items():
        for priority, source in enumerate(chain, start=1):
            source_type, _, name = source.partition(":")
            configured = None
            reachable = None
            free_tier = ""
            env_var = ""
            if source_type == "api" and not providers.empty:
                match = providers[providers["name"].astype(str).str.lower().str.replace(" ", "_").str.contains(name, na=False)]
                if match.empty:
                    match = providers[providers["name"].astype(str).str.lower().str.contains(name.replace("_", " "), na=False)]
                if not match.empty:
                    row = match.iloc[0]
                    env_var = str(row.get("env_var", ""))
                    free_tier = str(row.get("free_tier", ""))
                    if not credentials.empty and env_var:
                        cred = credentials[credentials["env_var"].astype(str).eq(env_var)]
                        configured = bool(cred["configured"].iloc[0]) if not cred.empty else bool(os.environ.get(env_var))
                    else:
                        configured = bool(os.environ.get(env_var))
                    if not health.empty:
                        h = health[health["name"].astype(str).str.lower().eq(str(row.get("name", "")).lower())]
                        reachable = bool(h["reachable"].iloc[0]) if not h.empty and "reachable" in h else None
            rows.append({
                "domain": domain,
                "priority": priority,
                "source": source,
                "source_type": source_type,
                "name": name,
                "configured": configured,
                "reachable": reachable,
                "free_tier": free_tier,
                "env_var": env_var,
                "policy": "read_first" if source_type in {"drive", "cache"} else "fallback_on_stale_or_missing",
            })
    return pd.DataFrame(rows)


def should_refresh(path: Path, max_age_hours: int = 24 * 7, min_rows: int | None = None) -> bool:
    path = Path(path)
    if not path.exists() or path.stat().st_size <= 1:
        return True
    age_hours = (pd.Timestamp.utcnow() - pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC")).total_seconds() / 3600
    if age_hours > max_age_hours:
        return True
    if min_rows is not None and path.suffix.lower() in {".csv", ".parquet"}:
        try:
            df = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
            return len(df) < min_rows
        except Exception:
            return True
    return False


def resolve_dataset_path(root: Path, domain: str, identifier: str | None = None) -> Path | None:
    """Resolve known Drive-first datasets without forcing a rigid schema."""
    root = Path(root)
    if domain == "prices" and identifier:
        token = identifier.lower().replace(".", "_").replace("-", "_")
        candidates = [
            root / "europe_stoxx_companies" / f"{token}_1d.parquet",
            root / "data" / "local_market_data" / f"{token}_1d.parquet",
            root / "raw_cache" / "prices" / f"{token}.parquet",
        ]
        for path in candidates:
            if path.exists():
                return path
        return candidates[0]
    if domain == "provider_registry":
        path = root / "catalog" / "api_providers.csv"
        return path if path.exists() else None
    return None


def read_dataset(path: Path, nrows: int | None = None) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, nrows=nrows)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, nrows=nrows)
    return pd.DataFrame()


def read_dataset_drive_first(
    root: Path,
    domain: str,
    identifier: str | None = None,
    max_age_hours: int = 24 * 7,
    allow_stale: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = resolve_dataset_path(root, domain, identifier)
    meta = {
        "domain": domain,
        "identifier": identifier,
        "path": str(path) if path else "",
        "source": "drive",
        "status": "missing",
        "should_refresh": True,
        "updated_at": utc_now(),
    }
    if path is None or not path.exists():
        return pd.DataFrame(), meta
    refresh = should_refresh(path, max_age_hours=max_age_hours)
    meta["should_refresh"] = refresh
    meta["status"] = "stale" if refresh else "fresh"
    df = read_dataset(path)
    if refresh and not allow_stale:
        return pd.DataFrame(), meta
    return df, meta


def incremental_merge(existing: pd.DataFrame, new: pd.DataFrame, key_cols: list[str], date_col: str | None = None) -> pd.DataFrame:
    frames = [df for df in [existing, new] if isinstance(df, pd.DataFrame) and not df.empty]
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    for col in key_cols:
        if col in out.columns:
            out[col] = out[col].astype(str).str.strip().str.upper()
    sort_cols = [c for c in ([date_col] if date_col else []) if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols)
    subset = [c for c in key_cols if c in out.columns]
    if date_col and date_col in out.columns:
        subset.append(date_col)
    if subset:
        out = out.drop_duplicates(subset=subset, keep="last")
    return out.reset_index(drop=True)


def write_dataset_incremental(
    new_data: pd.DataFrame,
    target_path: Path,
    key_cols: list[str],
    date_col: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    existing = read_dataset(target_path) if target_path.exists() else pd.DataFrame()
    merged = incremental_merge(existing, new_data, key_cols=key_cols, date_col=date_col)
    if target_path.suffix.lower() == ".parquet":
        merged.to_parquet(target_path, index=False)
    else:
        merged.to_csv(target_path, index=False)
    meta = {
        "target_path": str(target_path),
        "rows_existing": len(existing),
        "rows_new": len(new_data) if isinstance(new_data, pd.DataFrame) else 0,
        "rows_written": len(merged),
        "key_cols": key_cols,
        "date_col": date_col,
        "updated_at": utc_now(),
        **(metadata or {}),
    }
    target_path.with_suffix(target_path.suffix + ".metadata.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return meta


def _ticker_from_stoxx_file(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_1d"):
        stem = stem[:-3]
    parts = stem.split("_")
    if len(parts) >= 2:
        return f"{parts[0].upper()}.{parts[1].upper()}"
    return stem.upper()


def _normalise_yfinance_price_frame(raw: pd.DataFrame, ticker: str, template: pd.DataFrame | None = None) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    df = raw.reset_index().copy()
    rename = {
        "Date": "date_time",
        "Datetime": "date_time",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
        "Adj Close": "adjclose",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    if "adjclose" not in df.columns and "close" in df.columns:
        df["adjclose"] = df["close"]
    df["symbol"] = ticker.upper()
    for col in ["name", "country", "sector", "currency", "exchange", "instrument_type"]:
        if template is not None and not template.empty and col in template.columns and template[col].notna().any():
            df[col] = template[col].dropna().iloc[-1]
    if "instrument_type" not in df.columns:
        df["instrument_type"] = "EQUITY"
    wanted = ["date_time", "open", "high", "low", "close", "volume", "adjclose", "symbol", "name", "country", "sector", "currency", "exchange", "instrument_type"]
    for col in wanted:
        if col not in df.columns:
            df[col] = np.nan
    return df[wanted]


def refresh_europe_stoxx_prices_incremental(
    root: Path,
    max_symbols: int = 25,
    stale_hours: int = 24 * 7,
    force: bool = False,
    provider: str = "yfinance",
) -> pd.DataFrame:
    """Incrementally refresh stale Europe/STOXX daily parquet files.

    API calls are made only for files that are stale/missing unless force=True.
    """
    root = Path(root)
    price_dir = root / "europe_stoxx_companies"
    rows: list[dict[str, Any]] = []
    if not price_dir.exists():
        return pd.DataFrame([{"status": "missing_price_dir", "path": str(price_dir), "updated_at": utc_now()}])
    files = sorted(price_dir.glob("*_1d.parquet"))
    targets = files[:max_symbols]
    if provider != "yfinance":
        return pd.DataFrame([{"status": "unsupported_provider", "provider": provider, "updated_at": utc_now()}])
    try:
        import yfinance as yf
    except Exception as exc:
        return pd.DataFrame([{"status": "provider_unavailable", "provider": provider, "error": str(exc), "updated_at": utc_now()}])
    for path in targets:
        ticker = _ticker_from_stoxx_file(path)
        existing = read_dataset(path)
        needs = force or should_refresh(path, max_age_hours=stale_hours, min_rows=10)
        if not needs:
            rows.append({"ticker": ticker, "path": str(path), "status": "fresh_skip", "rows_existing": len(existing), "rows_new": 0, "updated_at": utc_now()})
            continue
        start = None
        if not existing.empty and "date_time" in existing.columns:
            max_date = pd.to_datetime(existing["date_time"], errors="coerce").max()
            if pd.notna(max_date):
                start = (max_date - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
        try:
            raw = yf.download(ticker, start=start, progress=False, auto_adjust=False, threads=False)
            new = _normalise_yfinance_price_frame(raw, ticker, existing)
            if new.empty:
                rows.append({"ticker": ticker, "path": str(path), "status": "empty_response", "rows_existing": len(existing), "rows_new": 0, "updated_at": utc_now()})
                continue
            meta = write_dataset_incremental(
                new,
                path,
                key_cols=["symbol"],
                date_col="date_time",
                metadata={"provider": provider, "domain": "prices", "refresh_mode": "incremental", "start": start},
            )
            rows.append({"ticker": ticker, "path": str(path), "status": "refreshed", **meta})
        except Exception as exc:
            rows.append({"ticker": ticker, "path": str(path), "status": "failed", "error": str(exc), "rows_existing": len(existing), "rows_new": 0, "updated_at": utc_now()})
    manifest = pd.DataFrame(rows)
    manifest_path = root / "catalog" / "europe_stoxx_incremental_price_refresh.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    return manifest


def publish_artifacts_to_financial_db(output_root: Path, financial_db_root: Path, domain: str) -> pd.DataFrame:
    output_root = Path(output_root)
    financial_db_root = Path(financial_db_root)
    target_root = financial_db_root / "analysis_outputs" / domain
    rows: list[dict[str, Any]] = []
    for sub in ["tables", "dashboard", "reports", "manifests", "config"]:
        source = output_root / sub
        if not source.exists():
            rows.append({"source": str(source), "target": "", "status": "missing_source", "updated_at": utc_now()})
            continue
        target = target_root / sub
        target.mkdir(parents=True, exist_ok=True)
        copied = 0
        for path in source.rglob("*"):
            if path.is_file():
                dest = target / path.relative_to(source)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, dest)
                copied += 1
        rows.append({"source": str(source), "target": str(target), "status": "synced", "files": copied, "updated_at": utc_now()})
    manifest = pd.DataFrame(rows)
    target_root.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(target_root / "sync_manifest.csv", index=False)
    return manifest


def write_data_platform_status(root: Path, output_root: Path, max_files: int | None = 5000) -> dict[str, Path]:
    status = dataset_status(root, max_files=max_files)
    output_root = Path(output_root)
    table_dir = output_root / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, df in status.items():
        path = table_dir / f"DataPlatform_{name}.csv"
        df.to_csv(path, index=False)
        paths[name] = path
    fallback = provider_fallback_plan(root)
    fallback_path = table_dir / "DataPlatform_provider_fallback_plan.csv"
    fallback.to_csv(fallback_path, index=False)
    paths["provider_fallback_plan"] = fallback_path
    contracts = {
        "generated_at": utc_now(),
        "financial_db_root": str(root),
        "tables": {name: str(path) for name, path in paths.items()},
        "domains": status.get("summary", pd.DataFrame()).to_dict(orient="records"),
        "provider_fallback_plan": fallback.to_dict(orient="records"),
    }
    api_dir = output_root / "api_contracts"
    api_dir.mkdir(parents=True, exist_ok=True)
    contract_path = api_dir / "data_platform_contract.json"
    contract_path.write_text(json.dumps(contracts, indent=2, default=str), encoding="utf-8")
    paths["api_contract"] = contract_path
    return paths
