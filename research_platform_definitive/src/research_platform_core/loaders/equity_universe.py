"""Global equity universe and fundamentals loaders."""

from __future__ import annotations

import json
from io import BytesIO
import time
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

from ..batch_downloader import BatchDownloader
from ..data_platform import read_dataset, should_refresh, utc_now
from .fx_commodities import normalize_yfinance_frame


EQUITY_UNIVERSES = {
    "sp500": {"name": "S&P 500", "region": "US", "constituents": 503, "start_date": "2000-01-01", "source": "wikipedia", "priority": 1},
    "nasdaq100": {"name": "Nasdaq 100", "region": "US", "constituents": 101, "start_date": "2000-01-01", "source": "wikipedia", "priority": 2},
    "russell2000": {"name": "Russell 2000", "region": "US", "constituents": 2000, "start_date": "2000-01-01", "source": "iShares ETF proxy", "priority": 3},
    "eurostoxx50": {"name": "Euro Stoxx 50", "region": "Europe", "constituents": 50, "start_date": "2000-01-01", "source": "local/yfinance", "priority": 1},
    "ftse100": {"name": "FTSE 100", "region": "Europe", "constituents": 100, "start_date": "2000-01-01", "source": "wikipedia", "priority": 2},
    "dax40": {"name": "DAX 40", "region": "Europe", "constituents": 40, "start_date": "2000-01-01", "source": "wikipedia", "priority": 2},
    "cac40": {"name": "CAC 40", "region": "Europe", "constituents": 40, "start_date": "2000-01-01", "source": "wikipedia", "priority": 3},
    "ibex35": {"name": "IBEX 35", "region": "Europe", "constituents": 35, "start_date": "2000-01-01", "source": "wikipedia", "priority": 3},
    "ftsemib": {"name": "FTSE MIB", "region": "Europe", "constituents": 40, "start_date": "2000-01-01", "source": "wikipedia", "priority": 1},
    "nikkei225": {"name": "Nikkei 225", "region": "Asia", "constituents": 225, "start_date": "2000-01-01", "source": "nikkei/wikipedia", "priority": 3},
    "csi300": {"name": "CSI 300", "region": "Asia", "constituents": 300, "start_date": "2005-01-01", "source": "ETF/proxy + best effort", "priority": 4},
    "hangseng": {"name": "Hang Seng", "region": "Asia", "constituents": 80, "start_date": "2000-01-01", "source": "wikipedia", "priority": 4},
    "kospi": {"name": "KOSPI", "region": "Asia", "constituents": 200, "start_date": "2000-01-01", "source": "ETF/proxy + best effort", "priority": 4},
    "msci_em_top100": {"name": "MSCI Emerging Markets Top 100", "region": "Emerging", "constituents": 100, "start_date": "2000-01-01", "source": "ETF/proxy + best effort", "priority": 4},
}


WIKIPEDIA_TABLES = {
    "sp500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "nasdaq100": "https://en.wikipedia.org/wiki/Nasdaq-100",
    "ftse100": "https://en.wikipedia.org/wiki/FTSE_100_Index",
    "dax40": "https://en.wikipedia.org/wiki/DAX",
    "cac40": "https://en.wikipedia.org/wiki/CAC_40",
    "ibex35": "https://en.wikipedia.org/wiki/IBEX_35",
    "ftsemib": "https://en.wikipedia.org/wiki/FTSE_MIB",
    "nikkei225": "https://en.wikipedia.org/wiki/Nikkei_225",
}

HTML_USER_AGENT = "TheGenesisAI-InvestmentResearchPlatform/1.0 (research data backfill; contact: local)"
MIN_USEFUL_CONSTITUENTS_CSV_BYTES = 96


def _clean_symbol(symbol: object) -> str:
    return str(symbol or "").strip().replace(".", "-").upper()


def _read_html_tables(url: str) -> list[pd.DataFrame]:
    """Read provider HTML tables with a user-agent to avoid generic bot 403s."""
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": HTML_USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        html = response.read()
    return pd.read_html(BytesIO(html))


def _is_cloud_storage_path(path: Path) -> bool:
    return any(part == "CloudStorage" for part in path.expanduser().parts)


def _is_suspicious_constituents_cache(path: Path) -> bool:
    """Detect placeholder/header-only Drive files before pandas can block on them."""
    try:
        size = path.stat().st_size
    except OSError:
        return True
    return _is_cloud_storage_path(path) and size < MIN_USEFUL_CONSTITUENTS_CSV_BYTES


def _remove_suspicious_constituents_cache(path: Path) -> bool:
    if not _is_suspicious_constituents_cache(path):
        return False
    try:
        path.unlink(missing_ok=True)
        return True
    except OSError:
        return False


def _empty_constituents_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["symbol", "name", "sector", "source", "as_of"])


class EquityUniverseManager:
    """Manage constituent manifests, prices and yfinance fundamentals."""

    def __init__(self, financial_db_root: Path | str, start_date: str = "2000-01-01"):
        self.financial_db_root = Path(financial_db_root).expanduser()
        self.start_date = start_date
        self.base_dir = self.financial_db_root / "Equities"
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def universe_catalog(self) -> pd.DataFrame:
        rows = []
        for key, meta in EQUITY_UNIVERSES.items():
            path = self.base_dir / meta["region"] / key / "constituents_current.csv"
            rows.append({"universe": key, "target_path": str(path), "exists": path.exists(), **meta})
        return pd.DataFrame(rows)

    def _constituents_path(self, universe: str) -> Path:
        meta = EQUITY_UNIVERSES[universe]
        return self.base_dir / meta["region"] / universe / "constituents_current.csv"

    def _read_constituents_cache(self, path: Path) -> pd.DataFrame:
        if not path.exists() or _is_suspicious_constituents_cache(path):
            return _empty_constituents_frame()
        try:
            frame = pd.read_csv(path)
        except Exception:
            return _empty_constituents_frame()
        if "symbol" not in frame.columns:
            return _empty_constituents_frame()
        frame["symbol"] = frame["symbol"].map(_clean_symbol)
        return frame.dropna(subset=["symbol"]).drop_duplicates("symbol").reset_index(drop=True)

    def get_current_constituents(self, universe: str) -> pd.DataFrame:
        universe = universe.lower()
        if universe not in WIKIPEDIA_TABLES:
            return _empty_constituents_frame()
        tables = _read_html_tables(WIKIPEDIA_TABLES[universe])
        candidates = [t for t in tables if any(str(c).lower() in {"symbol", "ticker", "ticker symbol", "code"} for c in t.columns)]
        if not candidates:
            candidates = tables[:1]
        raw = candidates[0].copy()
        symbol_col = next((c for c in raw.columns if str(c).lower() in {"symbol", "ticker", "ticker symbol", "code"}), raw.columns[0])
        name_col = next((c for c in raw.columns if str(c).lower() in {"security", "company", "company name", "name"}), None)
        sector_col = next((c for c in raw.columns if "sector" in str(c).lower() or "industry" in str(c).lower()), None)
        out = pd.DataFrame(
            {
                "symbol": raw[symbol_col].map(_clean_symbol),
                "name": raw[name_col].astype(str) if name_col is not None else "",
                "sector": raw[sector_col].astype(str) if sector_col is not None else "",
                "source": WIKIPEDIA_TABLES[universe],
                "as_of": pd.Timestamp.now(tz="UTC").date().isoformat(),
            }
        )
        return out.dropna(subset=["symbol"]).drop_duplicates("symbol").reset_index(drop=True)

    def write_constituents(self, universe: str, refresh: bool = False) -> dict[str, Any]:
        target = self._constituents_path(universe)
        if target.exists() and not refresh and not _is_suspicious_constituents_cache(target):
            cached = self._read_constituents_cache(target)
            return {"universe": universe, "status": "cache_hit", "target_path": str(target), "rows": len(cached)}
        try:
            df = self.get_current_constituents(universe)
        except Exception as exc:
            cached = self._read_constituents_cache(target) if target.exists() else _empty_constituents_frame()
            if not cached.empty:
                return {
                    "universe": universe,
                    "status": "cache_hit_after_provider_error",
                    "target_path": str(target),
                    "rows": len(cached),
                    "error": str(exc),
                    "updated_at": utc_now(),
                }
            df = _empty_constituents_frame()
            payload = {
                "universe": universe,
                "status": "provider_failed_empty_written",
                "target_path": str(target),
                "rows": 0,
                "error": str(exc),
                "updated_at": utc_now(),
            }
            target.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(target, index=False)
            target.with_suffix(".metadata.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
            return payload
        target.parent.mkdir(parents=True, exist_ok=True)
        removed_suspicious_cache = _remove_suspicious_constituents_cache(target)
        df.to_csv(target, index=False)
        payload = {
            "universe": universe,
            "status": "written",
            "target_path": str(target),
            "rows": len(df),
            "removed_suspicious_cache": removed_suspicious_cache,
            "updated_at": utc_now(),
        }
        target.with_suffix(".metadata.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return payload

    def _price_path(self, universe: str, symbol: str) -> Path:
        meta = EQUITY_UNIVERSES[universe]
        return self.base_dir / meta["region"] / universe / "prices" / f"{symbol.replace('.', '_').replace('-', '_')}.parquet"

    def _fundamental_path(self, universe: str, symbol: str, statement: str) -> Path:
        meta = EQUITY_UNIVERSES[universe]
        return self.base_dir / meta["region"] / universe / "fundamentals" / f"{symbol.replace('.', '_').replace('-', '_')}_{statement}.parquet"

    def sync_prices(self, universe: str, symbols: list[str] | None = None, max_symbols: int | None = None, refresh: bool = False, stale_hours: int = 24 * 7) -> pd.DataFrame:
        if symbols is None:
            constituents_path = self._constituents_path(universe)
            if not constituents_path.exists() or _is_suspicious_constituents_cache(constituents_path):
                self.write_constituents(universe)
            symbols = self._read_constituents_cache(constituents_path)["symbol"].dropna().astype(str).tolist()
        symbols = symbols[:max_symbols] if max_symbols else symbols
        import yfinance as yf

        rows = []
        for idx, symbol in enumerate(symbols, start=1):
            target = self._price_path(universe, symbol)
            if target.exists() and not refresh and not should_refresh(target, max_age_hours=stale_hours, min_rows=10):
                rows.append({"universe": universe, "symbol": symbol, "status": "fresh_skip", "rows": len(read_dataset(target)), "target_path": str(target)})
                continue
            try:
                raw = yf.download(symbol, start=self.start_date, progress=False, auto_adjust=False, threads=False)
                df = normalize_yfinance_frame(raw, symbol, symbol)
                target.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(target, index=False)
                rows.append({"universe": universe, "symbol": symbol, "status": "downloaded", "rows": len(df), "target_path": str(target)})
            except Exception as exc:
                rows.append({"universe": universe, "symbol": symbol, "status": "failed", "error": str(exc), "target_path": str(target)})
            if idx % 100 == 0:
                time.sleep(10)
        manifest = pd.DataFrame(rows)
        manifest_path = self.base_dir / EQUITY_UNIVERSES[universe]["region"] / universe / "prices_manifest.csv"
        manifest.to_csv(manifest_path, index=False)
        return manifest

    def sync_fundamentals(self, universe: str, symbols: list[str] | None = None, max_symbols: int | None = None, refresh: bool = False) -> pd.DataFrame:
        if symbols is None:
            constituents_path = self._constituents_path(universe)
            if not constituents_path.exists() or _is_suspicious_constituents_cache(constituents_path):
                self.write_constituents(universe)
            symbols = self._read_constituents_cache(constituents_path)["symbol"].dropna().astype(str).tolist()
        symbols = symbols[:max_symbols] if max_symbols else symbols
        import yfinance as yf

        rows = []
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                statements = {
                    "income_statement_quarterly": ticker.quarterly_financials,
                    "balance_sheet_quarterly": ticker.quarterly_balance_sheet,
                    "cash_flow_quarterly": ticker.quarterly_cashflow,
                }
                for statement, df in statements.items():
                    target = self._fundamental_path(universe, symbol, statement)
                    if target.exists() and not refresh:
                        rows.append({"universe": universe, "symbol": symbol, "statement": statement, "status": "cache_hit", "target_path": str(target)})
                        continue
                    out = df.T.reset_index().rename(columns={"index": "period"})
                    target.parent.mkdir(parents=True, exist_ok=True)
                    out.to_parquet(target, index=False)
                    rows.append({"universe": universe, "symbol": symbol, "statement": statement, "status": "downloaded", "rows": len(out), "target_path": str(target)})
            except Exception as exc:
                rows.append({"universe": universe, "symbol": symbol, "statement": "all", "status": "failed", "error": str(exc)})
        manifest = pd.DataFrame(rows)
        manifest_path = self.base_dir / EQUITY_UNIVERSES[universe]["region"] / universe / "fundamentals_manifest.csv"
        manifest.to_csv(manifest_path, index=False)
        return manifest
