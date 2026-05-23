"""Risk factor loaders for credit, rates and volatility proxies."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from ..data_platform import utc_now
from .fx_commodities import normalize_yfinance_frame


FRED_RISK_SERIES = {
    "BAA": "Moodys Seasoned Baa Corporate Bond Yield",
    "AAA": "Moodys Seasoned Aaa Corporate Bond Yield",
    "BAMLH0A0HYM2": "ICE BofA US High Yield Option-Adjusted Spread",
    "DGS10": "10-Year Treasury Constant Maturity",
    "DGS2": "2-Year Treasury Constant Maturity",
    "DGS3MO": "3-Month Treasury Constant Maturity",
    "DFII10": "10-Year TIPS Real Yield",
    "T10YIE": "10-Year Breakeven Inflation",
}

VOLATILITY_TICKERS = {
    "VIX": "^VIX",
    "VXN": "^VXN",
    "RVX": "^RVX",
    "MOVE_proxy": "MOVE",
}


class RiskFactorsLoader:
    """Fetch FRED and yfinance risk proxies into RiskFactors."""

    def __init__(self, financial_db_root: Path | str, start_date: str = "2000-01-01"):
        self.financial_db_root = Path(financial_db_root).expanduser()
        self.start_date = start_date
        self.base_dir = self.financial_db_root / "RiskFactors"
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def catalog(self) -> pd.DataFrame:
        rows = []
        for code, desc in FRED_RISK_SERIES.items():
            path = self.base_dir / "fred" / f"{code}.csv"
            rows.append({"source": "fred", "series": code, "description": desc, "target_path": str(path), "exists": path.exists()})
        for name, ticker in VOLATILITY_TICKERS.items():
            path = self.base_dir / "volatility" / f"{name}.parquet"
            rows.append({"source": "yfinance", "series": name, "ticker": ticker, "target_path": str(path), "exists": path.exists()})
        return pd.DataFrame(rows)

    def sync_fred(self, refresh: bool = False, api_key: str | None = None) -> pd.DataFrame:
        api_key = api_key or os.environ.get("FRED_API_KEY", "")
        rows: list[dict[str, Any]] = []
        for code, desc in FRED_RISK_SERIES.items():
            target = self.base_dir / "fred" / f"{code}.csv"
            if target.exists() and not refresh:
                rows.append({"series": code, "status": "cache_hit", "target_path": str(target), "rows": len(pd.read_csv(target))})
                continue
            try:
                if api_key:
                    url = "https://api.stlouisfed.org/fred/series/observations"
                    params = {"series_id": code, "api_key": api_key, "file_type": "json", "observation_start": self.start_date}
                    import requests

                    payload = requests.get(url, params=params, timeout=30).json()
                    df = pd.DataFrame(payload.get("observations", []))
                    if not df.empty:
                        df = df[["date", "value"]]
                        df["value"] = pd.to_numeric(df["value"], errors="coerce")
                else:
                    csv_url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={code}"
                    df = pd.read_csv(csv_url).rename(columns={code: "value"})
                target.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(target, index=False)
                rows.append({"series": code, "description": desc, "status": "downloaded", "target_path": str(target), "rows": len(df)})
            except Exception as exc:
                rows.append({"series": code, "description": desc, "status": "failed", "error": str(exc), "target_path": str(target)})
        manifest = pd.DataFrame(rows)
        manifest.to_csv(self.base_dir / "fred_risk_manifest.csv", index=False)
        return manifest

    def sync_volatility(self, refresh: bool = False) -> pd.DataFrame:
        import yfinance as yf

        rows = []
        for name, ticker in VOLATILITY_TICKERS.items():
            target = self.base_dir / "volatility" / f"{name}.parquet"
            if target.exists() and not refresh:
                rows.append({"series": name, "ticker": ticker, "status": "cache_hit", "target_path": str(target)})
                continue
            try:
                raw = yf.download(ticker, start=self.start_date, progress=False, auto_adjust=False, threads=False)
                df = normalize_yfinance_frame(raw, ticker, name)
                target.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(target, index=False)
                rows.append({"series": name, "ticker": ticker, "status": "downloaded", "rows": len(df), "target_path": str(target), "updated_at": utc_now()})
            except Exception as exc:
                rows.append({"series": name, "ticker": ticker, "status": "failed", "error": str(exc), "target_path": str(target)})
        manifest = pd.DataFrame(rows)
        manifest.to_csv(self.base_dir / "volatility_manifest.csv", index=False)
        return manifest

    def build_derived_risk_factors(self) -> pd.DataFrame:
        fred_dir = self.base_dir / "fred"
        frames = []
        for code in FRED_RISK_SERIES:
            path = fred_dir / f"{code}.csv"
            if path.exists():
                df = pd.read_csv(path).rename(columns={"value": code})
                frames.append(df[["date", code]])
        if not frames:
            return pd.DataFrame()
        out = frames[0]
        for frame in frames[1:]:
            out = out.merge(frame, on="date", how="outer")
        for col in [c for c in out.columns if c != "date"]:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        if {"BAA", "AAA"}.issubset(out.columns):
            out["credit_spread_baa_aaa"] = out["BAA"] - out["AAA"]
        if {"DGS10", "DGS2"}.issubset(out.columns):
            out["term_spread_10y_2y"] = out["DGS10"] - out["DGS2"]
        if {"DGS10", "DGS3MO"}.issubset(out.columns):
            out["term_spread_10y_3m"] = out["DGS10"] - out["DGS3MO"]
        target = self.base_dir / "risk_factor_panel.csv"
        out.sort_values("date").to_csv(target, index=False)
        target.with_suffix(".metadata.json").write_text(json.dumps({"updated_at": utc_now(), "rows": len(out), "target_path": str(target)}, indent=2), encoding="utf-8")
        return out
