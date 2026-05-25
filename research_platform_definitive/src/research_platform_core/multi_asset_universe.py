"""Multi-asset universe and coverage manifest helpers.

The equity universe has a dedicated OHLCV manifest.  This module gives the
non-equity side (FX, commodities, ETF, fixed income, crypto) an equivalent,
lightweight manifest derived from the Macro DB catalog/manifest.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .data_platform import resolve_data_platform_roots, utc_now
from .macro_market import macro_asset_catalog


TABLE_REL = Path("macro_market") / "tables" / "MultiAssetUniverseManifest.csv"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 1:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _domain_for_asset_class(asset_class: str) -> str:
    value = str(asset_class or "").lower()
    if value == "fx":
        return "FX"
    if value == "commodity_etf":
        return "ETF Commodity"
    if value.startswith("commodity"):
        return "Commodities"
    if value == "crypto_etf":
        return "ETF Crypto"
    if "crypto" in value:
        return "Crypto"
    if value == "fixed_income_etf":
        return "ETF Fixed Income"
    if "fixed_income" in value:
        return "Fixed Income"
    if value == "rate_index":
        return "Fixed Income"
    if value == "volatility_index":
        return "Volatility"
    if value in {"equity_index_etf", "sector_etf"}:
        return "ETF Equity"
    if value.endswith("_etf") or "etf" in value:
        return "ETF"
    if "equity_index" in value:
        return "Equity Index / ETF"
    return "Other"


def compile_multi_asset_universe_manifest(
    financial_db_root: str | Path | None = None,
    output_root: str | Path | None = None,
) -> pd.DataFrame:
    """Build a manifest for non-equity/macro tradable proxies."""
    roots = resolve_data_platform_roots(financial_db_root=financial_db_root, repo_output_root=output_root)
    table_root = roots.repo_output / TABLE_REL.parent
    table_root.mkdir(parents=True, exist_ok=True)
    catalog = macro_asset_catalog()
    catalog.to_csv(table_root / "MacroAssetCatalog.csv", index=False)
    manifest_paths = [
        roots.repo_output / "macro_market" / "tables" / "MacroAssetManifest.csv",
        roots.financial_db / "MarketData" / "Macro" / "MacroAssetManifest.csv",
    ]
    macro_manifest = next((frame for frame in (_read_csv(path) for path in manifest_paths) if not frame.empty), pd.DataFrame())
    if not macro_manifest.empty and "symbol" in macro_manifest.columns:
        macro_manifest = macro_manifest.copy()
        macro_manifest["symbol"] = macro_manifest["symbol"].astype(str).str.upper()
        catalog = catalog.merge(
            macro_manifest[
                [
                    col
                    for col in [
                        "symbol",
                        "status",
                        "rows",
                        "last_date",
                        "target_path",
                        "updated_at",
                    ]
                    if col in macro_manifest.columns
                ]
            ],
            on="symbol",
            how="left",
            suffixes=("", "_manifest"),
        )
    else:
        catalog["status"] = "PLANNED"
        catalog["rows"] = 0
        catalog["last_date"] = ""
        catalog["target_path"] = ""
        catalog["updated_at"] = ""

    out = catalog.copy()
    out["symbol"] = out["symbol"].astype(str).str.upper()
    out["domain"] = out["asset_class"].map(_domain_for_asset_class)
    status_source = (
        out["status_manifest"]
        if "status_manifest" in out.columns
        else out["status"]
        if "status" in out.columns
        else pd.Series("PLANNED", index=out.index)
    )
    out["data_status"] = status_source.fillna("PLANNED").astype(str).str.upper()
    out["data_status"] = out["data_status"].replace({"READY": "PLANNED", "PROXY": "PLANNED"})
    out["rows"] = pd.to_numeric(out.get("rows", 0), errors="coerce").fillna(0).astype(int)
    out["coverage_label"] = out.apply(
        lambda row: "OK" if str(row["data_status"]).upper() == "OK" and int(row["rows"]) > 0 else str(row["data_status"]).upper(),
        axis=1,
    )
    out["first_date"] = ""
    last_date = out["last_date"] if "last_date" in out.columns else pd.Series("", index=out.index)
    out["last_date"] = last_date.fillna("").astype(str)
    out["manifest_updated_at"] = utc_now()
    columns = [
        "domain",
        "symbol",
        "provider_symbol",
        "name",
        "asset_class",
        "region",
        "country",
        "category",
        "exposure",
        "currency",
        "source",
        "coverage_label",
        "data_status",
        "rows",
        "first_date",
        "last_date",
        "target_path",
        "manifest_updated_at",
    ]
    out = out[[col for col in columns if col in out.columns]].sort_values(["domain", "region", "asset_class", "symbol"]).reset_index(drop=True)
    out.to_csv(table_root / TABLE_REL.name, index=False)
    return out


def load_multi_asset_universe_manifest(
    financial_db_root: str | Path | None = None,
    output_root: str | Path | None = None,
    *,
    refresh_if_missing: bool = True,
) -> pd.DataFrame:
    roots = resolve_data_platform_roots(financial_db_root=financial_db_root, repo_output_root=output_root)
    path = roots.repo_output / TABLE_REL
    frame = _read_csv(path)
    if frame.empty and refresh_if_missing:
        frame = compile_multi_asset_universe_manifest(roots.financial_db, roots.repo_output)
    return frame


def summarize_multi_asset_universe(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a multi-asset manifest by domain and status."""
    if frame.empty:
        return pd.DataFrame(columns=["domain", "instrument_count", "ok_count", "planned_count", "partial_count", "last_date"])
    view = frame.copy()
    view["coverage_label"] = view.get("coverage_label", "PLANNED").fillna("PLANNED").astype(str).str.upper()
    grouped = view.groupby("domain", dropna=False)
    rows: list[dict[str, Any]] = []
    for domain, group in grouped:
        rows.append(
            {
                "domain": domain,
                "instrument_count": int(len(group)),
                "ok_count": int(group["coverage_label"].eq("OK").sum()),
                "planned_count": int(group["coverage_label"].eq("PLANNED").sum()),
                "partial_count": int(group["coverage_label"].isin(["PARTIAL", "NO_DATA", "FAILED"]).sum()),
                "last_date": group["last_date"].dropna().astype(str).max() if "last_date" in group.columns and group["last_date"].notna().any() else "",
            }
        )
    return pd.DataFrame(rows).sort_values("domain").reset_index(drop=True)
