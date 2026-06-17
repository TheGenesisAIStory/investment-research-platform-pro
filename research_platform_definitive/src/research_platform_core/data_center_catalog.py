"""Canonical target catalog for Data Center coverage."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .loaders.equity_universe import EQUITY_UNIVERSES
from .loaders.fama_french import FAMA_FRENCH_DATASETS
from .loaders.fx_commodities import COMMODITY_TICKERS, FX_TICKERS
from .loaders.risk_factors import FRED_RISK_SERIES, VOLATILITY_TICKERS
from .loaders.ecb_client import ECB_SERIES_PRESETS
from .loaders.bditalia_client import BDITALIA_SERIES_PRESETS
from .data_platform import get_ohlcv_parquet_root_info


AQR_TARGETS = {
    "QMJ": "Quality Minus Junk",
    "BAB": "Betting Against Beta",
    "HML_Devil": "The Devil in HML's Details",
    "Momentum": "AQR Momentum Indices",
    "Value_Momentum_Everywhere": "Value and Momentum Everywhere",
}


def _path_exists(path: Path) -> bool:
    return path.exists() and path.is_file()


def build_target_catalog(financial_db_root: Path | str, output_root: Path | str | None = None) -> pd.DataFrame:
    root = Path(financial_db_root).expanduser()
    ohlcv_info = get_ohlcv_parquet_root_info(root, output_root)
    rows: list[dict[str, Any]] = []
    for key, meta in EQUITY_UNIVERSES.items():
        path = root / "Equities" / meta["region"] / key / "constituents_current.csv"
        rows.append(
            {
                "domain": "equity_universe",
                "dataset": key,
                "label": meta["name"],
                "provider": meta["source"],
                "frequency": "daily_prices_quarterly_fundamentals",
                "start_date_target": meta["start_date"],
                "target_path": str(path),
                "exists": _path_exists(path),
                "priority": meta["priority"],
                "expected_items": meta["constituents"],
            }
        )
    for key, meta in FAMA_FRENCH_DATASETS.items():
        path = root / "Factors" / "FamaFrench" / f"{key}.csv"
        rows.append(
            {
                "domain": "factor_data",
                "dataset": key,
                "label": key.replace("_", " "),
                "provider": "fama_french",
                "frequency": meta["frequency"],
                "start_date_target": "source_history",
                "target_path": str(path),
                "exists": _path_exists(path),
                "priority": 1 if "daily" in key.lower() or "FF5" in key else 2,
                "expected_items": "",
            }
        )
    for key, label in AQR_TARGETS.items():
        path = root / "alpha_factor_library" / "aqr"
        rows.append(
            {
                "domain": "factor_data",
                "dataset": key,
                "label": label,
                "provider": "aqr",
                "frequency": "daily_or_monthly",
                "start_date_target": "source_history",
                "target_path": str(path),
                "exists": path.exists(),
                "priority": 2,
                "expected_items": "",
            }
        )
    for key, ticker in FX_TICKERS.items():
        path = root / "FX" / f"{key}.parquet"
        rows.append(
            {
                "domain": "fx",
                "dataset": key,
                "label": key,
                "provider": "yfinance",
                "frequency": "daily",
                "start_date_target": "2000-01-01",
                "target_path": str(path),
                "exists": _path_exists(path),
                "priority": 1,
                "expected_items": ticker,
            }
        )
    for key, ticker in COMMODITY_TICKERS.items():
        path = root / "Commodities" / f"{key}.parquet"
        rows.append(
            {
                "domain": "commodities",
                "dataset": key,
                "label": key,
                "provider": "yfinance",
                "frequency": "daily",
                "start_date_target": "2000-01-01",
                "target_path": str(path),
                "exists": _path_exists(path),
                "priority": 2,
                "expected_items": ticker,
            }
        )
    for key, label in FRED_RISK_SERIES.items():
        path = root / "RiskFactors" / "fred" / f"{key}.csv"
        rows.append(
            {
                "domain": "risk_factors",
                "dataset": key,
                "label": label,
                "provider": "fred",
                "frequency": "daily_or_business",
                "start_date_target": "2000-01-01",
                "target_path": str(path),
                "exists": _path_exists(path),
                "priority": 1,
                "expected_items": "",
            }
        )
    for key, ticker in VOLATILITY_TICKERS.items():
        path = root / "RiskFactors" / "volatility" / f"{key}.parquet"
        rows.append(
            {
                "domain": "risk_factors",
                "dataset": key,
                "label": key,
                "provider": "yfinance",
                "frequency": "daily",
                "start_date_target": "2000-01-01",
                "target_path": str(path),
                "exists": _path_exists(path),
                "priority": 1,
                "expected_items": ticker,
            }
        )
    for key, meta in ECB_SERIES_PRESETS.items():
        path = root / "OfficialMacro" / "ECB" / f"{key}.csv"
        rows.append(
            {
                "domain": "official_macro",
                "dataset": key,
                "label": meta["label"],
                "provider": "ecb",
                "frequency": meta["frequency"],
                "start_date_target": "source_history",
                "target_path": str(path),
                "exists": _path_exists(path),
                "priority": 1 if meta["category"] in {"inflation", "policy_rates"} else 2,
                "expected_items": f"{meta['flow_ref']}/{meta['key']}",
            }
        )
    for key, meta in BDITALIA_SERIES_PRESETS.items():
        if meta["category"] == "publication":
            continue
        path = root / "OfficialMacro" / "BancaItalia" / f"{key}.csv"
        rows.append(
            {
                "domain": "official_macro",
                "dataset": key,
                "label": meta["description"],
                "provider": "bancaditalia",
                "frequency": meta["frequency"],
                "start_date_target": "source_history",
                "target_path": str(path),
                "exists": _path_exists(path),
                "priority": 1 if meta["category"] in {"credit", "public_debt"} else 2,
                "expected_items": f"{meta['object_type']}/{meta['object_id']}",
            }
        )
    for key, meta in {
        "asset_master": {"label": "OHLCV asset master", "path": root / "MarketData" / "OHLCV" / "asset_master_candidates.csv", "priority": 1},
        "ohlcv_sqlite": {"label": "OHLCV SQLite/Postgres mirror", "path": root / "MarketData" / "ohlcv.sqlite", "priority": 1},
        "daily_manifest": {"label": "Daily OHLCV ingest manifest", "path": root / "MarketData" / "OHLCV" / "manifests" / "ohlcv_daily_manifest.csv", "priority": 1},
        "daily_parquet": {"label": "Daily OHLCV parquet root", "path": ohlcv_info.path / "daily", "priority": 1},
        "intraday_5m_manifest": {"label": "Optional 5m OHLCV manifest", "path": root / "MarketData" / "OHLCV" / "manifests" / "ohlcv_intraday_5m_manifest.csv", "priority": 3},
    }.items():
        path = meta["path"]
        rows.append(
            {
                "domain": "ohlcv_prices",
                "dataset": key,
                "label": meta["label"],
                "provider": "yfinance_stooq_alpha_vantage",
                "frequency": "daily_or_5m",
                "start_date_target": "2000-01-01",
                "target_path": str(path),
                "exists": path.exists(),
                "priority": meta["priority"],
                "expected_items": "",
            }
        )
    return pd.DataFrame(rows)


def summarize_target_catalog(catalog: pd.DataFrame) -> pd.DataFrame:
    if catalog.empty:
        return pd.DataFrame()
    out = (
        catalog.groupby("domain")
        .agg(targets=("dataset", "count"), available=("exists", "sum"), priority_min=("priority", "min"))
        .reset_index()
    )
    out["coverage_pct"] = (out["available"] / out["targets"] * 100).round(1)
    return out.sort_values(["priority_min", "domain"]).reset_index(drop=True)
