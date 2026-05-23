"""FX and commodities loaders using Drive-first storage and yfinance fallback."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ..batch_downloader import BatchDownloader
from ..data_platform import read_dataset, should_refresh, utc_now


FX_TICKERS = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "AUDUSD": "AUDUSD=X",
    "USDCAD": "USDCAD=X",
    "USDCHF": "USDCHF=X",
    "NZDUSD": "NZDUSD=X",
    "EURGBP": "EURGBP=X",
    "EURJPY": "EURJPY=X",
    "DXY": "DX-Y.NYB",
}

COMMODITY_TICKERS = {
    "WTI": "CL=F",
    "Brent": "BZ=F",
    "NaturalGas": "NG=F",
    "Gold": "GC=F",
    "Silver": "SI=F",
    "Platinum": "PL=F",
    "Palladium": "PA=F",
    "Copper": "HG=F",
    "Corn": "ZC=F",
    "Wheat": "ZW=F",
    "Soybeans": "ZS=F",
    "GSCI": "GSG",
    "DBC": "DBC",
}


def normalize_yfinance_frame(raw: pd.DataFrame, symbol: str, asset_name: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    df = raw.reset_index().copy()
    rename = {"Date": "date", "Datetime": "date", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Adj Close": "adjclose", "Volume": "volume"}
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    if "adjclose" not in df.columns and "close" in df.columns:
        df["adjclose"] = df["close"]
    df["symbol"] = symbol
    df["asset_name"] = asset_name
    for col in ["date", "open", "high", "low", "close", "adjclose", "volume", "symbol", "asset_name"]:
        if col not in df.columns:
            df[col] = pd.NA
    return df[["date", "open", "high", "low", "close", "adjclose", "volume", "symbol", "asset_name"]]


class ForexCommoditiesManager:
    """Download and refresh FX majors and commodities."""

    def __init__(self, financial_db_root: Path | str, start_date: str = "2000-01-01"):
        self.financial_db_root = Path(financial_db_root).expanduser()
        self.start_date = start_date
        self.fx_dir = self.financial_db_root / "FX"
        self.commodity_dir = self.financial_db_root / "Commodities"
        self.fx_dir.mkdir(parents=True, exist_ok=True)
        self.commodity_dir.mkdir(parents=True, exist_ok=True)

    def catalog(self) -> pd.DataFrame:
        rows = []
        for asset_class, mapping, base in [("fx", FX_TICKERS, self.fx_dir), ("commodities", COMMODITY_TICKERS, self.commodity_dir)]:
            for name, ticker in mapping.items():
                path = base / f"{name}.parquet"
                rows.append(
                    {
                        "asset_class": asset_class,
                        "name": name,
                        "ticker": ticker,
                        "target_path": str(path),
                        "exists": path.exists(),
                        "rows": len(read_dataset(path)) if path.exists() else 0,
                        "updated_at": pd.Timestamp(path.stat().st_mtime, unit="s").isoformat() if path.exists() else "",
                    }
                )
        return pd.DataFrame(rows)

    def _download_one(self, name: str, ticker: str, target: Path, refresh: bool = False, stale_hours: int = 24 * 7) -> dict[str, Any]:
        if target.exists() and not refresh and not should_refresh(target, max_age_hours=stale_hours, min_rows=10):
            return {"name": name, "ticker": ticker, "target_path": str(target), "status": "fresh_skip", "rows": len(read_dataset(target)), "updated_at": utc_now()}
        import yfinance as yf

        raw = yf.download(ticker, start=self.start_date, progress=False, auto_adjust=False, threads=False)
        df = normalize_yfinance_frame(raw, ticker, name)
        if df.empty:
            return {"name": name, "ticker": ticker, "target_path": str(target), "status": "empty_response", "rows": 0, "updated_at": utc_now()}
        target.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(target, index=False)
        meta = {"name": name, "ticker": ticker, "target_path": str(target), "provider": "yfinance", "rows": len(df), "status": "downloaded", "updated_at": utc_now()}
        target.with_suffix(".metadata.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        return meta

    def sync_fx(self, refresh: bool = False, max_symbols: int | None = None) -> pd.DataFrame:
        items = list(FX_TICKERS.items())[:max_symbols]
        rows = [self._download_one(name, ticker, self.fx_dir / f"{name}.parquet", refresh=refresh) for name, ticker in items]
        manifest = pd.DataFrame(rows)
        manifest.to_csv(self.fx_dir / "FX_manifest.csv", index=False)
        return manifest

    def sync_commodities(self, refresh: bool = False, max_symbols: int | None = None, max_workers: int = 3) -> pd.DataFrame:
        items = list(COMMODITY_TICKERS.items())[:max_symbols]
        downloader = BatchDownloader(max_workers=max_workers, requests_per_minute=60)

        def fetch(name: str) -> dict[str, Any]:
            ticker = COMMODITY_TICKERS[name]
            return self._download_one(name, ticker, self.commodity_dir / f"{name}.parquet", refresh=refresh)

        result = downloader.batch_download([name for name, _ in items], fetch)
        rows = list(result["results"].values()) + [{"name": k, "status": "failed", "error": v} for k, v in result["errors"].items()]
        manifest = pd.DataFrame(rows)
        manifest.to_csv(self.commodity_dir / "Commodities_manifest.csv", index=False)
        return manifest
