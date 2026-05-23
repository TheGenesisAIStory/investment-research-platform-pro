"""Market universe discovery for broad OHLCV ingestion.

Primary US coverage comes from NASDAQ Trader Symbol Directory files:

* https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt
* https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt
* definitions: https://nasdaqtrader.com/Trader.aspx?id=SymbolDirDefs

For Europe/Asia the free data landscape is fragmented. The module therefore
combines existing curated index constituents, optional user-provided exchange
CSV files, and provider-specific ticker normalization. Add full exchange files
under ``Database Finanziario/catalog/exchange_symbols`` to expand coverage
without changing code.
"""

from __future__ import annotations

import logging
import time
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd

from ..data_platform import resolve_data_platform_roots, utc_now
from .equity_universe import EQUITY_UNIVERSES, EquityUniverseManager


LOGGER = logging.getLogger(__name__)

NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

US_EXCHANGE_CODES = {
    "Q": "NASDAQ",
    "N": "NYSE",
    "A": "NYSE American",
    "P": "NYSE Arca",
    "Z": "Cboe BZX",
    "V": "IEX",
}

INDEX_MARKET_MAP = {
    "sp500": {"market": "us_large_cap", "exchange": "US", "country": "US"},
    "nasdaq100": {"market": "us_large_cap", "exchange": "NASDAQ", "country": "US"},
    "russell2000": {"market": "us_small_cap", "exchange": "US", "country": "US"},
    "eurostoxx50": {"market": "europe_major", "exchange": "Europe", "country": "EU"},
    "ftse100": {"market": "uk_major", "exchange": "LSE", "country": "GB"},
    "dax40": {"market": "germany_major", "exchange": "XETRA", "country": "DE"},
    "cac40": {"market": "france_major", "exchange": "Euronext Paris", "country": "FR"},
    "ibex35": {"market": "spain_major", "exchange": "BME", "country": "ES"},
    "ftsemib": {"market": "italy_major", "exchange": "Borsa Italiana", "country": "IT"},
    "nikkei225": {"market": "japan_major", "exchange": "Tokyo", "country": "JP"},
    "hangseng": {"market": "hongkong_major", "exchange": "HKEX", "country": "HK"},
}

DEFAULT_GLOBAL_ETFS = [
    {"ticker": "SPY", "provider_symbol": "SPY", "name": "SPDR S&P 500 ETF Trust", "exchange": "NYSE Arca", "country": "US", "type": "etf"},
    {"ticker": "QQQ", "provider_symbol": "QQQ", "name": "Invesco QQQ Trust", "exchange": "NASDAQ", "country": "US", "type": "etf"},
    {"ticker": "IWM", "provider_symbol": "IWM", "name": "iShares Russell 2000 ETF", "exchange": "NYSE Arca", "country": "US", "type": "etf"},
    {"ticker": "EFA", "provider_symbol": "EFA", "name": "iShares MSCI EAFE ETF", "exchange": "NYSE Arca", "country": "US", "type": "etf"},
    {"ticker": "EEM", "provider_symbol": "EEM", "name": "iShares MSCI Emerging Markets ETF", "exchange": "NYSE Arca", "country": "US", "type": "etf"},
    {"ticker": "EWJ", "provider_symbol": "EWJ", "name": "iShares MSCI Japan ETF", "exchange": "NYSE Arca", "country": "US", "type": "etf"},
    {"ticker": "VGK", "provider_symbol": "VGK", "name": "Vanguard FTSE Europe ETF", "exchange": "NYSE Arca", "country": "US", "type": "etf"},
]


def _read_nasdaqtrader_text(url: str) -> pd.DataFrame:
    import requests

    response = requests.get(url, timeout=30, headers={"User-Agent": "ResearchPlatform/1.0 MarketUniverse"})
    response.raise_for_status()
    lines = [line for line in response.text.splitlines() if line and not line.startswith("File Creation Time")]
    return pd.read_csv(StringIO("\n".join(lines)), sep="|")


def _normalize_symbol(value: Any) -> str:
    return str(value or "").strip().replace(".", "-").upper()


def _normalize_asset_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["ticker", "provider_symbol", "name", "exchange", "country", "type", "market", "primary_source", "active_flag"])
    out = df.copy()
    for col in ["ticker", "provider_symbol", "name", "exchange", "country", "type", "market", "primary_source"]:
        if col not in out.columns:
            out[col] = ""
    out["ticker"] = out["ticker"].map(_normalize_symbol)
    out["provider_symbol"] = out["provider_symbol"].where(out["provider_symbol"].astype(str).str.len().gt(0), out["ticker"])
    out["provider_symbol"] = out["provider_symbol"].astype(str).str.strip()
    out["type"] = out["type"].replace("", "stock").str.lower()
    out["active_flag"] = out.get("active_flag", True)
    out["created_at"] = utc_now()
    out["updated_at"] = utc_now()
    return out.dropna(subset=["ticker"]).drop_duplicates(["ticker", "exchange", "primary_source"]).reset_index(drop=True)


def _load_universe_cache_ttl(default_hours: float = 24) -> float:
    config_path = Path(__file__).resolve().parents[3] / "config" / "ohlcv_data_sources.yaml"
    if not config_path.exists():
        return default_hours
    try:
        import yaml

        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        return float(payload.get("ohlcv", {}).get("cache", {}).get("universe_ttl_hours", default_hours))
    except Exception:
        return default_hours


class MarketUniverseBuilder:
    """Build asset_master candidates for US/global price ingestion."""

    def __init__(self, financial_db_root: Path | str | None = None, output_root: Path | str | None = None):
        roots = resolve_data_platform_roots(financial_db_root=financial_db_root, repo_output_root=output_root)
        self.financial_db_root = roots.financial_db
        self.output_root = roots.repo_output
        self.catalog_dir = self.financial_db_root / "catalog"
        self.manual_dir = self.catalog_dir / "exchange_symbols"
        self.universe_ttl_hours = _load_universe_cache_ttl()
        self.catalog_dir.mkdir(parents=True, exist_ok=True)
        self.manual_dir.mkdir(parents=True, exist_ok=True)

    def _is_fresh(self, path: Path, ttl_hours: float) -> bool:
        return path.exists() and (time.time() - path.stat().st_mtime) <= ttl_hours * 3600

    def fetch_us_listings(self, include_etfs: bool = True) -> pd.DataFrame:
        """Fetch active US listings from NASDAQ Trader Symbol Directory."""
        cache_path = self.catalog_dir / "nasdaq_trader_us_listings.csv"
        if self._is_fresh(cache_path, self.universe_ttl_hours):
            out = _normalize_asset_frame(pd.read_csv(cache_path))
            if not include_etfs:
                out = out[out["type"].ne("etf")].copy()
            return out.reset_index(drop=True)

        nasdaq = _read_nasdaqtrader_text(NASDAQ_LISTED_URL)
        nasdaq = nasdaq[nasdaq.get("Test Issue", "N").eq("N")].copy()
        nasdaq_out = pd.DataFrame(
            {
                "ticker": nasdaq["Symbol"],
                "provider_symbol": nasdaq["Symbol"],
                "name": nasdaq["Security Name"],
                "exchange": "NASDAQ",
                "country": "US",
                "type": nasdaq.get("ETF", "N").map(lambda v: "etf" if str(v).upper() == "Y" else "stock"),
                "market": "us_all",
                "primary_source": "nasdaq_trader_symbol_directory",
                "active_flag": True,
            }
        )
        other = _read_nasdaqtrader_text(OTHER_LISTED_URL)
        other = other[other.get("Test Issue", "N").eq("N")].copy()
        other_out = pd.DataFrame(
            {
                "ticker": other["ACT Symbol"],
                "provider_symbol": other["ACT Symbol"],
                "name": other["Security Name"],
                "exchange": other["Exchange"].map(lambda code: US_EXCHANGE_CODES.get(str(code), str(code))),
                "country": "US",
                "type": other.get("ETF", "N").map(lambda v: "etf" if str(v).upper() == "Y" else "stock"),
                "market": "us_all",
                "primary_source": "nasdaq_trader_symbol_directory",
                "active_flag": True,
            }
        )
        out = _normalize_asset_frame(pd.concat([nasdaq_out, other_out], ignore_index=True, sort=False))
        out.to_csv(cache_path, index=False)
        if not include_etfs:
            out = out[out["type"].ne("etf")].copy()
        return out.reset_index(drop=True)

    def load_manual_exchange_files(self) -> pd.DataFrame:
        """Load user-provided exchange symbol CSV files, if present."""
        frames: list[pd.DataFrame] = []
        for path in sorted(self.manual_dir.glob("*.csv")):
            try:
                df = pd.read_csv(path)
                lower = {str(c).lower(): c for c in df.columns}
                rename = {
                    lower.get("ticker", lower.get("symbol", "")): "ticker",
                    lower.get("provider_symbol", lower.get("yfinance_symbol", lower.get("symbol", ""))): "provider_symbol",
                    lower.get("name", lower.get("security_name", "")): "name",
                    lower.get("exchange", ""): "exchange",
                    lower.get("country", ""): "country",
                    lower.get("type", ""): "type",
                }
                rename = {src: dst for src, dst in rename.items() if src}
                clean = df.rename(columns=rename)
                clean["market"] = path.stem
                clean["primary_source"] = f"manual_exchange_file:{path.name}"
                frames.append(clean)
            except Exception as exc:
                LOGGER.warning("Manual exchange file skipped path=%s error=%s", path, exc)
        return _normalize_asset_frame(pd.concat(frames, ignore_index=True, sort=False)) if frames else pd.DataFrame()

    def fetch_index_universes(self, universes: list[str] | None = None) -> pd.DataFrame:
        """Reuse existing index loaders for major Europe/Asia/US names."""
        manager = EquityUniverseManager(self.financial_db_root)
        rows: list[pd.DataFrame] = []
        for universe in universes or list(INDEX_MARKET_MAP):
            if universe not in EQUITY_UNIVERSES:
                continue
            try:
                constituents = manager.get_current_constituents(universe)
                if constituents.empty:
                    continue
                meta = INDEX_MARKET_MAP.get(universe, {"market": universe, "exchange": "", "country": ""})
                frame = pd.DataFrame(
                    {
                        "ticker": constituents["symbol"],
                        "provider_symbol": constituents["symbol"],
                        "name": constituents.get("name", ""),
                        "exchange": meta["exchange"],
                        "country": meta["country"],
                        "type": "stock",
                        "market": meta["market"],
                        "primary_source": f"index_constituents:{universe}",
                        "active_flag": True,
                    }
                )
                rows.append(frame)
            except Exception as exc:
                LOGGER.warning("Index universe skipped universe=%s error=%s", universe, exc)
        return _normalize_asset_frame(pd.concat(rows, ignore_index=True, sort=False)) if rows else pd.DataFrame()

    def global_etfs(self) -> pd.DataFrame:
        frame = pd.DataFrame(DEFAULT_GLOBAL_ETFS)
        frame["market"] = "global_etfs"
        frame["primary_source"] = "curated_global_etfs"
        frame["active_flag"] = True
        return _normalize_asset_frame(frame)

    def build_universe(
        self,
        markets: list[str] | None = None,
        include_us: bool = True,
        include_indices: bool = True,
        include_manual: bool = True,
        include_etfs: bool = True,
    ) -> pd.DataFrame:
        """Build an asset universe for requested market groups."""
        requested = {m.strip().lower() for m in (markets or ["us_all", "europe_major", "japan_major", "global_etfs"]) if m.strip()}
        frames: list[pd.DataFrame] = []
        if include_us and any(m in requested for m in {"us_all", "us_stocks", "us_etfs", "nyse", "nasdaq", "amex"}):
            us = self.fetch_us_listings(include_etfs=True)
            if "us_stocks" in requested:
                us = us[us["type"].eq("stock")]
            if "us_etfs" in requested:
                us = us[us["type"].eq("etf")]
            if "nyse" in requested:
                us = us[us["exchange"].eq("NYSE")]
            if "nasdaq" in requested:
                us = us[us["exchange"].eq("NASDAQ")]
            if "amex" in requested:
                us = us[us["exchange"].eq("NYSE American")]
            if not include_etfs:
                us = us[us["type"].ne("etf")]
            frames.append(us)
        if include_indices:
            index_markets = [k for k, meta in INDEX_MARKET_MAP.items() if meta["market"] in requested or k in requested]
            if index_markets:
                frames.append(self.fetch_index_universes(index_markets))
        if include_etfs and "global_etfs" in requested:
            frames.append(self.global_etfs())
        if include_manual:
            manual = self.load_manual_exchange_files()
            if not manual.empty:
                frames.append(manual[manual["market"].str.lower().isin(requested) | manual["exchange"].str.lower().isin(requested)])
        out = _normalize_asset_frame(pd.concat(frames, ignore_index=True, sort=False)) if frames else pd.DataFrame()
        target = self.catalog_dir / "ohlcv_asset_universe.csv"
        out.to_csv(target, index=False)
        return out
