"""Local market-data SQLite database builder.

Safe by default: no secrets are stored, generated SQLite/Parquet data live under
`data/local_market_data/` which is ignored by git. The full mode can download
7 years of daily prices for broad universes; smoke mode creates a small local DB
for verification and falls back to deterministic demo prices if a request fails.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests

DB_PATH = Path("data/local_market_data/market_data.sqlite")
YAHOO_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"


@dataclass(frozen=True)
class Instrument:
    symbol: str
    name: str
    asset_class: str
    universe: str
    exchange: str = ""
    currency: str = ""
    source: str = "yahoo"


MAJOR_ETFS = [
    Instrument("SPY", "SPDR S&P 500 ETF", "etf", "major_etf", "NYSEARCA", "USD"),
    Instrument("QQQ", "Invesco QQQ Trust", "etf", "major_etf", "NASDAQ", "USD"),
    Instrument("IWM", "iShares Russell 2000 ETF", "etf", "major_etf", "NYSEARCA", "USD"),
    Instrument("EFA", "iShares MSCI EAFE ETF", "etf", "major_etf", "NYSEARCA", "USD"),
    Instrument("EEM", "iShares MSCI Emerging Markets ETF", "etf", "major_etf", "NYSEARCA", "USD"),
    Instrument("GLD", "SPDR Gold Shares", "etf", "major_etf", "NYSEARCA", "USD"),
    Instrument("TLT", "iShares 20+ Year Treasury Bond ETF", "etf", "major_etf", "NASDAQ", "USD"),
]
MAJOR_FX = [Instrument(s, s.replace("=X", ""), "fx", "major_fx", "Yahoo", "") for s in ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X", "AUDUSD=X", "USDCAD=X", "NZDUSD=X"]]
MAJOR_CRYPTO = [Instrument(s, s.replace("-USD", ""), "crypto", "major_crypto", "Yahoo", "USD") for s in ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD", "ADA-USD", "DOGE-USD"]]
EUROPE_SEED = [
    Instrument("ASML.AS", "ASML Holding", "equity", "euro_stoxx50", "Euronext Amsterdam", "EUR"),
    Instrument("SAP.DE", "SAP", "equity", "euro_stoxx50", "XETRA", "EUR"),
    Instrument("MC.PA", "LVMH", "equity", "euro_stoxx50", "Euronext Paris", "EUR"),
    Instrument("OR.PA", "L'Oreal", "equity", "euro_stoxx50", "Euronext Paris", "EUR"),
    Instrument("SIE.DE", "Siemens", "equity", "euro_stoxx50", "XETRA", "EUR"),
]
FTSE_MIB_SEED = [
    Instrument("ENEL.MI", "Enel", "equity", "ftse_mib", "Borsa Italiana", "EUR"),
    Instrument("ENI.MI", "Eni", "equity", "ftse_mib", "Borsa Italiana", "EUR"),
    Instrument("ISP.MI", "Intesa Sanpaolo", "equity", "ftse_mib", "Borsa Italiana", "EUR"),
    Instrument("UCG.MI", "UniCredit", "equity", "ftse_mib", "Borsa Italiana", "EUR"),
    Instrument("RACE.MI", "Ferrari", "equity", "ftse_mib", "Borsa Italiana", "EUR"),
]
SP500_SEED = [
    Instrument("AAPL", "Apple", "equity", "sp500", "NASDAQ", "USD"),
    Instrument("MSFT", "Microsoft", "equity", "sp500", "NASDAQ", "USD"),
    Instrument("NVDA", "NVIDIA", "equity", "sp500", "NASDAQ", "USD"),
    Instrument("AMZN", "Amazon", "equity", "sp500", "NASDAQ", "USD"),
    Instrument("GOOGL", "Alphabet", "equity", "sp500", "NASDAQ", "USD"),
]


def connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS instruments (
            symbol TEXT NOT NULL,
            name TEXT NOT NULL,
            asset_class TEXT NOT NULL,
            universe TEXT NOT NULL,
            exchange TEXT,
            currency TEXT,
            source TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(symbol, universe)
        );
        CREATE TABLE IF NOT EXISTS prices_daily (
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            adj_close REAL,
            volume REAL,
            source TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(symbol, date, source)
        );
        CREATE INDEX IF NOT EXISTS idx_prices_symbol_date ON prices_daily(symbol, date);
        """
    )
    conn.commit()


def load_instruments(mode: str = "smoke") -> list[Instrument]:
    base = SP500_SEED + EUROPE_SEED + FTSE_MIB_SEED + MAJOR_ETFS + MAJOR_FX + MAJOR_CRYPTO
    if mode == "smoke":
        return base[:3] + EUROPE_SEED[:2] + MAJOR_FX[:1] + MAJOR_CRYPTO[:1] + MAJOR_ETFS[:1]
    instruments = list(dict.fromkeys(base))
    instruments.extend(fetch_fmp_global_top())
    return sorted({(i.symbol, i.universe): i for i in instruments}.values(), key=lambda x: (x.universe, x.symbol))


def fetch_fmp_global_top(limit: int = 500) -> list[Instrument]:
    key = os.getenv("FMP_API_KEY")
    if not key:
        return []
    url = "https://financialmodelingprep.com/api/v3/stock-screener"
    params = {"marketCapMoreThan": 1_000_000_000, "limit": limit, "apikey": key}
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        rows = response.json()[:limit]
    except Exception:
        return []
    return [Instrument(r.get("symbol", ""), r.get("companyName", ""), "equity", "global_top_500", r.get("exchangeShortName", ""), r.get("currency", ""), "fmp/yahoo") for r in rows if r.get("symbol")]


def upsert_instruments(conn: sqlite3.Connection, instruments: Iterable[Instrument]) -> None:
    conn.executemany(
        """INSERT OR REPLACE INTO instruments(symbol,name,asset_class,universe,exchange,currency,source)
           VALUES(?,?,?,?,?,?,?)""",
        [(i.symbol, i.name, i.asset_class, i.universe, i.exchange, i.currency, i.source) for i in instruments],
    )
    conn.commit()


def fetch_yahoo_prices(symbol: str, years: int = 7) -> pd.DataFrame:
    end = int(time.time())
    start = int((dt.datetime.now(dt.UTC) - dt.timedelta(days=365 * years + 10)).timestamp())
    params = {"period1": start, "period2": end, "interval": "1d", "events": "history", "includeAdjustedClose": "true"}
    response = requests.get(YAHOO_CHART.format(symbol=quote(symbol, safe="")), params=params, timeout=30, headers={"User-Agent": "ml-trading-thesis-bot/1.0"})
    response.raise_for_status()
    result = response.json()["chart"]["result"][0]
    timestamps = result.get("timestamp") or []
    quote_data = result["indicators"]["quote"][0]
    adj = result["indicators"].get("adjclose", [{}])[0].get("adjclose", quote_data.get("close"))
    frame = pd.DataFrame(quote_data)
    frame["adj_close"] = adj
    frame["date"] = pd.to_datetime(timestamps, unit="s", utc=True).date.astype(str)
    frame["symbol"] = symbol
    frame["source"] = "yahoo"
    return frame[["symbol", "date", "open", "high", "low", "close", "adj_close", "volume", "source"]].dropna(subset=["close"])


def demo_prices(symbol: str, years: int = 7, seed: int | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed or abs(hash(symbol)) % (2**32))
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=252 * years)
    returns = rng.normal(0.0002, 0.012, len(dates))
    close = 100 * np.exp(np.cumsum(returns))
    return pd.DataFrame({"symbol": symbol, "date": dates.date.astype(str), "open": close * (1 + rng.normal(0, .001, len(dates))), "high": close * 1.005, "low": close * .995, "close": close, "adj_close": close, "volume": rng.integers(100_000, 2_000_000, len(dates)), "source": "demo"})


def upsert_prices(conn: sqlite3.Connection, prices: pd.DataFrame) -> None:
    conn.executemany(
        """INSERT OR REPLACE INTO prices_daily(symbol,date,open,high,low,close,adj_close,volume,source)
           VALUES(?,?,?,?,?,?,?,?,?)""",
        prices[["symbol", "date", "open", "high", "low", "close", "adj_close", "volume", "source"]].itertuples(index=False, name=None),
    )
    conn.commit()


def build_database(db_path: Path = DB_PATH, mode: str = "smoke", years: int = 7, max_symbols: int | None = None, allow_demo_fallback: bool = True) -> dict[str, int]:
    instruments = load_instruments(mode)
    if max_symbols:
        instruments = instruments[:max_symbols]
    conn = connect(db_path)
    create_schema(conn)
    upsert_instruments(conn, instruments)
    downloaded = 0
    fallback = 0
    for inst in instruments:
        try:
            prices = fetch_yahoo_prices(inst.symbol, years=years)
            downloaded += 1
        except Exception:
            if not allow_demo_fallback:
                continue
            prices = demo_prices(inst.symbol, years=years)
            fallback += 1
        upsert_prices(conn, prices)
    counts = {
        "instruments": conn.execute("SELECT COUNT(*) FROM instruments").fetchone()[0],
        "prices": conn.execute("SELECT COUNT(*) FROM prices_daily").fetchone()[0],
        "downloaded_symbols": downloaded,
        "fallback_symbols": fallback,
    }
    conn.close()
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--years", type=int, default=7)
    parser.add_argument("--max-symbols", type=int, default=None)
    parser.add_argument("--no-demo-fallback", action="store_true")
    args = parser.parse_args()
    counts = build_database(args.db, args.mode, args.years, args.max_symbols, not args.no_demo_fallback)
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
