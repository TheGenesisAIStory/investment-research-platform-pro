"""Data catalog helpers for local ML4T datasets and API provider metadata."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class DataDomain:
    name: str
    path: str
    description: str
    primary_sources: str
    publish_to_git: bool
    notes: str


DATA_DOMAINS: tuple[DataDomain, ...] = (
    DataDomain("api_registry", "data/data_providers/api_registry", "Provider metadata, masked credential status, and health checks", "Alpha Vantage, Tiingo, FMP, Polygon, Intrinio, FRED, BLS, CFTC, Congress.gov, EconDB, EIA, Nasdaq Data Link", True, "Safe only if credential files remain masked."),
    DataDomain("private_api_checks", "data/private_api_checks", "Local credentialed API smoke-check outputs", "Configured API services from .env.local or .env", False, "Private diagnostics; ignored by git."),
    DataDomain("europe_stoxx_companies", "data/europe_stoxx_companies", "Cached Europe/STOXX OHLCV and benchmark data", "Yahoo Chart public endpoint or configured provider APIs", False, "Research cache."),
    DataDomain("alternative_data", "data/alternative_data", "Alternative-data snapshots and engineered features", "Polymarket, Stocktwits, GDELT, Wikipedia, Google Trends, OpenInsider", False, "Keep raw/license-sensitive data local."),
    DataDomain("alpha_factor_research", "data/alpha_factor_research", "Alpha factors, IC diagnostics, ML predictions, and backtest returns", "Derived from market/fundamental/alternative inputs", False, "Generated research artifacts."),
    DataDomain("sec_edgar", "data/sec_edgar", "SEC submissions and companyfacts examples", "SEC EDGAR APIs", False, "Respect SEC fair-access policy."),
    DataDomain("storage_benchmark", "data/storage_benchmark", "Local storage benchmark outputs", "Synthetic benchmark data", False, "Generated outputs."),
    DataDomain("open_market_data", "data/open_market_data", "Open market-data examples used by notebooks", "Public sample datasets", False, "Verify provenance before publishing."),
)


def domains_to_dataframe() -> pd.DataFrame:
    return pd.DataFrame([asdict(domain) for domain in DATA_DOMAINS])


def folder_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def inventory_path(path: Path, base: Path) -> dict[str, Any]:
    files = [file for file in path.rglob("*") if file.is_file()] if path.exists() else []
    suffix_counts: dict[str, int] = {}
    for file in files:
        suffix = file.suffix.lower() or "<none>"
        suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1
    last_modified = max((file.stat().st_mtime for file in files), default=0)
    return {
        "path": str(path.relative_to(base) if path.exists() and path.is_relative_to(base) else path),
        "exists": path.exists(),
        "files": len(files),
        "size_mb": round(folder_size(path) / (1024**2), 3),
        "suffix_counts": json.dumps(suffix_counts, sort_keys=True),
        "last_modified_utc": datetime.fromtimestamp(last_modified, timezone.utc).isoformat() if files else "",
    }


def build_data_inventory(repo_root: Path) -> pd.DataFrame:
    rows = []
    for domain in DATA_DOMAINS:
        row = inventory_path(repo_root / domain.path, repo_root)
        row.update({"domain": domain.name, "publish_to_git": domain.publish_to_git, "description": domain.description})
        rows.append(row)
    return pd.DataFrame(rows)


def export_catalog(repo_root: Path, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    domains = domains_to_dataframe()
    inventory = build_data_inventory(repo_root)
    outputs = {
        "domains_csv": output_dir / "data_domains.csv",
        "domains_parquet": output_dir / "data_domains.parquet",
        "inventory_csv": output_dir / "data_inventory.csv",
        "inventory_parquet": output_dir / "data_inventory.parquet",
    }
    domains.to_csv(outputs["domains_csv"], index=False)
    domains.to_parquet(outputs["domains_parquet"], index=False)
    inventory.to_csv(outputs["inventory_csv"], index=False)
    inventory.to_parquet(outputs["inventory_parquet"], index=False)
    return {key: str(value) for key, value in outputs.items()}
