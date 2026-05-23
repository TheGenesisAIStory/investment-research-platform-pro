"""SQLite/Postgres-compatible OHLCV storage helpers."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from .data_platform import resolve_data_platform_roots, utc_now


ASSET_COLUMNS = ["ticker", "provider_symbol", "exchange", "country", "type", "name", "primary_source", "active_flag"]


def default_database_url(financial_db_root: Path | str | None = None) -> str:
    roots = resolve_data_platform_roots(financial_db_root=financial_db_root)
    return f"sqlite:///{roots.financial_db / 'MarketData' / 'ohlcv.sqlite'}"


class OhlcvDatabase:
    """Small DB adapter for asset_master and OHLCV tables.

    SQLite works out of the box. PostgreSQL/TimescaleDB can be used by setting
    ``MARKETDATA_DATABASE_URL`` to a SQLAlchemy URL if SQLAlchemy and the driver
    are installed in the environment.
    """

    def __init__(self, database_url: str | None = None, financial_db_root: Path | str | None = None):
        self.database_url = database_url or os.environ.get("MARKETDATA_DATABASE_URL") or default_database_url(financial_db_root)
        self.is_sqlite = self.database_url.startswith("sqlite:///") or "://" not in self.database_url
        self._engine = None
        if self.is_sqlite:
            path = self.database_url.replace("sqlite:///", "") if self.database_url.startswith("sqlite:///") else self.database_url
            self.path = Path(path).expanduser()
            self.path.parent.mkdir(parents=True, exist_ok=True)
        else:
            self.path = None

    def connect(self):
        if self.is_sqlite:
            conn = sqlite3.connect(str(self.path))
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            return conn
        if self._engine is None:
            from sqlalchemy import create_engine

            self._engine = create_engine(self.database_url, future=True)
        return self._engine.connect()

    def init_schema(self) -> None:
        if self.is_sqlite:
            ddl = [
                """
                CREATE TABLE IF NOT EXISTS asset_master (
                    asset_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    provider_symbol TEXT NOT NULL,
                    exchange TEXT,
                    country TEXT,
                    type TEXT,
                    name TEXT,
                    primary_source TEXT NOT NULL,
                    active_flag INTEGER DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(ticker, exchange, primary_source)
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS price_ohlcv_daily (
                    asset_id INTEGER NOT NULL,
                    date TEXT NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    adjclose REAL,
                    volume REAL,
                    source TEXT NOT NULL,
                    ingestion_ts TEXT NOT NULL,
                    PRIMARY KEY(asset_id, date)
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS price_ohlcv_intraday_5m (
                    asset_id INTEGER NOT NULL,
                    ts TEXT NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    source TEXT NOT NULL,
                    ingestion_ts TEXT NOT NULL,
                    PRIMARY KEY(asset_id, ts)
                )
                """,
                "CREATE INDEX IF NOT EXISTS idx_asset_symbol ON asset_master(provider_symbol)",
                "CREATE INDEX IF NOT EXISTS idx_daily_asset_date ON price_ohlcv_daily(asset_id, date)",
            ]
            with self.connect() as conn:
                for statement in ddl:
                    conn.execute(statement)
                conn.commit()
            return
        from sqlalchemy import text

        ddl = [
            """
            CREATE TABLE IF NOT EXISTS asset_master (
                asset_id BIGSERIAL PRIMARY KEY,
                ticker TEXT NOT NULL,
                provider_symbol TEXT NOT NULL,
                exchange TEXT,
                country TEXT,
                type TEXT,
                name TEXT,
                primary_source TEXT NOT NULL,
                active_flag BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMPTZ NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL,
                UNIQUE(ticker, exchange, primary_source)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS price_ohlcv_daily (
                asset_id BIGINT NOT NULL REFERENCES asset_master(asset_id),
                date DATE NOT NULL,
                open DOUBLE PRECISION,
                high DOUBLE PRECISION,
                low DOUBLE PRECISION,
                close DOUBLE PRECISION,
                adjclose DOUBLE PRECISION,
                volume DOUBLE PRECISION,
                source TEXT NOT NULL,
                ingestion_ts TIMESTAMPTZ NOT NULL,
                PRIMARY KEY(asset_id, date)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS price_ohlcv_intraday_5m (
                asset_id BIGINT NOT NULL REFERENCES asset_master(asset_id),
                ts TIMESTAMPTZ NOT NULL,
                open DOUBLE PRECISION,
                high DOUBLE PRECISION,
                low DOUBLE PRECISION,
                close DOUBLE PRECISION,
                volume DOUBLE PRECISION,
                source TEXT NOT NULL,
                ingestion_ts TIMESTAMPTZ NOT NULL,
                PRIMARY KEY(asset_id, ts)
            )
            """,
        ]
        with self.connect() as conn:
            for statement in ddl:
                conn.execute(text(statement))
            conn.commit()

    def upsert_assets(self, assets: pd.DataFrame) -> dict[tuple[str, str, str], int]:
        """Upsert asset_master rows and return natural-key to asset_id map."""
        if assets.empty:
            return {}
        self.init_schema()
        rows = assets.copy()
        for col in ASSET_COLUMNS:
            if col not in rows.columns:
                rows[col] = "" if col != "active_flag" else True
        rows["created_at"] = rows.get("created_at", utc_now())
        rows["updated_at"] = utc_now()
        rows = rows[ASSET_COLUMNS + ["created_at", "updated_at"]].drop_duplicates(["ticker", "exchange", "primary_source"])
        if self.is_sqlite:
            sql = """
            INSERT INTO asset_master (ticker, provider_symbol, exchange, country, type, name, primary_source, active_flag, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticker, exchange, primary_source) DO UPDATE SET
                provider_symbol=excluded.provider_symbol,
                country=excluded.country,
                type=excluded.type,
                name=excluded.name,
                active_flag=excluded.active_flag,
                updated_at=excluded.updated_at
            """
            values = [
                (
                    r.ticker,
                    r.provider_symbol,
                    r.exchange,
                    r.country,
                    r.type,
                    r.name,
                    r.primary_source,
                    int(bool(r.active_flag)),
                    r.created_at,
                    r.updated_at,
                )
                for r in rows.itertuples(index=False)
            ]
            with self.connect() as conn:
                conn.executemany(sql, values)
                conn.commit()
        else:
            from sqlalchemy import text

            sql = text(
                """
                INSERT INTO asset_master (ticker, provider_symbol, exchange, country, type, name, primary_source, active_flag, created_at, updated_at)
                VALUES (:ticker, :provider_symbol, :exchange, :country, :type, :name, :primary_source, :active_flag, :created_at, :updated_at)
                ON CONFLICT(ticker, exchange, primary_source) DO UPDATE SET
                    provider_symbol=excluded.provider_symbol,
                    country=excluded.country,
                    type=excluded.type,
                    name=excluded.name,
                    active_flag=excluded.active_flag,
                    updated_at=excluded.updated_at
                """
            )
            with self.connect() as conn:
                conn.execute(sql, rows.to_dict("records"))
                conn.commit()
        return self.asset_id_map(rows)

    def asset_id_map(self, assets: pd.DataFrame | None = None) -> dict[tuple[str, str, str], int]:
        query = "SELECT asset_id, ticker, exchange, primary_source FROM asset_master"
        if self.is_sqlite:
            with self.connect() as conn:
                df = pd.read_sql_query(query, conn)
        else:
            with self.connect() as conn:
                df = pd.read_sql_query(query, conn)
        if assets is not None and not assets.empty:
            keys = set(zip(assets["ticker"], assets["exchange"], assets["primary_source"]))
            df = df[df.apply(lambda row: (row["ticker"], row["exchange"], row["primary_source"]) in keys, axis=1)]
        return {(row.ticker, row.exchange, row.primary_source): int(row.asset_id) for row in df.itertuples(index=False)}

    def latest_daily_dates(self, asset_ids: list[int] | None = None) -> dict[int, str]:
        query = "SELECT asset_id, MAX(date) AS max_date FROM price_ohlcv_daily"
        params: list[Any] = []
        if asset_ids:
            placeholders = ",".join("?" for _ in asset_ids)
            query += f" WHERE asset_id IN ({placeholders})"
            params = list(asset_ids)
        query += " GROUP BY asset_id"
        if self.is_sqlite:
            with self.connect() as conn:
                df = pd.read_sql_query(query, conn, params=params)
        else:
            with self.connect() as conn:
                df = pd.read_sql_query(query.replace("?", "%s"), conn, params=params)
        return {int(row.asset_id): str(row.max_date) for row in df.itertuples(index=False) if pd.notna(row.max_date)}

    def latest_daily_dates_by_provider_symbol(self, provider_symbols: list[str] | None = None) -> dict[str, str]:
        """Return the max daily date across all asset sources for each provider symbol.

        This lets API incremental jobs respect static seed histories loaded under
        a different ``primary_source`` such as Kaggle.
        """
        query = """
        SELECT a.provider_symbol, MAX(p.date) AS max_date
        FROM price_ohlcv_daily p
        JOIN asset_master a ON a.asset_id = p.asset_id
        """
        params: list[Any] = []
        if provider_symbols:
            placeholders = ",".join("?" for _ in provider_symbols)
            query += f" WHERE a.provider_symbol IN ({placeholders})"
            params = list(provider_symbols)
        query += " GROUP BY a.provider_symbol"
        if self.is_sqlite:
            with self.connect() as conn:
                df = pd.read_sql_query(query, conn, params=params)
        else:
            with self.connect() as conn:
                df = pd.read_sql_query(query.replace("?", "%s"), conn, params=params)
        return {str(row.provider_symbol): str(row.max_date) for row in df.itertuples(index=False) if pd.notna(row.max_date)}

    def upsert_daily_prices(self, prices: pd.DataFrame) -> int:
        if prices.empty:
            return 0
        self.init_schema()
        rows = prices.copy()
        for col in ["asset_id", "date", "open", "high", "low", "close", "adjclose", "volume", "source", "ingestion_ts"]:
            if col not in rows.columns:
                rows[col] = None
        rows = rows[["asset_id", "date", "open", "high", "low", "close", "adjclose", "volume", "source", "ingestion_ts"]].dropna(subset=["asset_id", "date", "close"])
        rows["asset_id"] = rows["asset_id"].astype(int)
        if self.is_sqlite:
            sql = """
            INSERT INTO price_ohlcv_daily (asset_id, date, open, high, low, close, adjclose, volume, source, ingestion_ts)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(asset_id, date) DO UPDATE SET
                open=excluded.open, high=excluded.high, low=excluded.low, close=excluded.close,
                adjclose=excluded.adjclose, volume=excluded.volume, source=excluded.source,
                ingestion_ts=excluded.ingestion_ts
            """
            values = [tuple(r) for r in rows.itertuples(index=False, name=None)]
            with self.connect() as conn:
                conn.executemany(sql, values)
                conn.commit()
        else:
            from sqlalchemy import text

            sql = text(
                """
                INSERT INTO price_ohlcv_daily (asset_id, date, open, high, low, close, adjclose, volume, source, ingestion_ts)
                VALUES (:asset_id, :date, :open, :high, :low, :close, :adjclose, :volume, :source, :ingestion_ts)
                ON CONFLICT(asset_id, date) DO UPDATE SET
                    open=excluded.open, high=excluded.high, low=excluded.low, close=excluded.close,
                    adjclose=excluded.adjclose, volume=excluded.volume, source=excluded.source,
                    ingestion_ts=excluded.ingestion_ts
                """
            )
            with self.connect() as conn:
                conn.execute(sql, rows.to_dict("records"))
                conn.commit()
        return int(len(rows))

    def upsert_intraday_5m(self, prices: pd.DataFrame) -> int:
        if prices.empty:
            return 0
        self.init_schema()
        rows = prices.copy()
        for col in ["asset_id", "ts", "open", "high", "low", "close", "volume", "source", "ingestion_ts"]:
            if col not in rows.columns:
                rows[col] = None
        rows = rows[["asset_id", "ts", "open", "high", "low", "close", "volume", "source", "ingestion_ts"]].dropna(subset=["asset_id", "ts", "close"])
        rows["asset_id"] = rows["asset_id"].astype(int)
        if self.is_sqlite:
            sql = """
            INSERT INTO price_ohlcv_intraday_5m (asset_id, ts, open, high, low, close, volume, source, ingestion_ts)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(asset_id, ts) DO UPDATE SET
                open=excluded.open, high=excluded.high, low=excluded.low, close=excluded.close,
                volume=excluded.volume, source=excluded.source, ingestion_ts=excluded.ingestion_ts
            """
            values = [tuple(r) for r in rows.itertuples(index=False, name=None)]
            with self.connect() as conn:
                conn.executemany(sql, values)
                conn.commit()
        else:
            from sqlalchemy import text

            sql = text(
                """
                INSERT INTO price_ohlcv_intraday_5m (asset_id, ts, open, high, low, close, volume, source, ingestion_ts)
                VALUES (:asset_id, :ts, :open, :high, :low, :close, :volume, :source, :ingestion_ts)
                ON CONFLICT(asset_id, ts) DO UPDATE SET
                    open=excluded.open, high=excluded.high, low=excluded.low, close=excluded.close,
                    volume=excluded.volume, source=excluded.source, ingestion_ts=excluded.ingestion_ts
                """
            )
            with self.connect() as conn:
                conn.execute(sql, rows.to_dict("records"))
                conn.commit()
        return int(len(rows))
