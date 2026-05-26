"""Canonical research database schema and sample-data builders.

The module is intentionally small and SQLite-first: it gives the Research
Platform, Data Center, LLM Lab and QuantDinger bridge one shared operating
contract without requiring a live data vendor during local development.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCHEMA_VERSION = "2026-05-23-final"
DEFAULT_SYMBOLS = ("AAPL", "MSFT", "NVDA", "SPY", "ENEL.MI", "ISP.MI", "ASML.AS", "SAP.DE")


def utc_now_iso() -> str:
    """Return a timezone-aware UTC timestamp as an ISO string."""

    return pd.Timestamp.now(tz="UTC").isoformat()


def research_platform_root() -> Path:
    """Resolve the ``research_platform_definitive`` root from this module."""

    return Path(__file__).resolve().parents[2]


def default_research_database_path(root: Path | str | None = None) -> Path:
    """Return the default local SQLite path for the final research database."""

    env_path = os.environ.get("RESEARCH_PLATFORM_DB_PATH")
    if env_path:
        return Path(env_path).expanduser()
    base = Path(root).expanduser() if root else research_platform_root()
    return base / "output" / "data_cache" / "research_platform.sqlite"


@dataclass(slots=True)
class ResearchDatabase:
    """SQLite adapter for the final research-platform operating schema."""

    path: Path | str | None = None

    def __post_init__(self) -> None:
        self.path = Path(self.path).expanduser() if self.path else default_research_database_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        return conn

    def init_schema(self) -> None:
        """Create all canonical tables and indexes if they do not exist."""

        ddl = [
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS instruments (
                instrument_id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL,
                asset_class TEXT NOT NULL,
                exchange TEXT,
                country TEXT,
                currency TEXT,
                sector TEXT,
                industry TEXT,
                provider_symbol TEXT,
                active INTEGER NOT NULL DEFAULT 1,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS ohlcv_daily (
                instrument_id INTEGER NOT NULL,
                date TEXT NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                adj_close REAL NOT NULL,
                volume REAL NOT NULL,
                source TEXT NOT NULL,
                ingestion_ts TEXT NOT NULL,
                PRIMARY KEY (instrument_id, date, source),
                FOREIGN KEY (instrument_id) REFERENCES instruments(instrument_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS features_labels (
                instrument_id INTEGER NOT NULL,
                date TEXT NOT NULL,
                feature_set TEXT NOT NULL,
                return_1d REAL,
                return_21d REAL,
                momentum_63d REAL,
                volatility_21d REAL,
                zscore_63d REAL,
                volume_zscore_21d REAL,
                label_forward_21d REAL,
                label_direction_21d INTEGER,
                source TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (instrument_id, date, feature_set),
                FOREIGN KEY (instrument_id) REFERENCES instruments(instrument_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS ml_signals (
                signal_id INTEGER PRIMARY KEY AUTOINCREMENT,
                instrument_id INTEGER NOT NULL,
                date TEXT NOT NULL,
                strategy_name TEXT NOT NULL,
                model_name TEXT NOT NULL,
                signal REAL NOT NULL,
                target_weight REAL NOT NULL,
                score REAL NOT NULL,
                confidence REAL NOT NULL,
                horizon TEXT NOT NULL,
                reasoning TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE (instrument_id, date, strategy_name, model_name, horizon),
                FOREIGN KEY (instrument_id) REFERENCES instruments(instrument_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS backtest_results (
                backtest_id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_name TEXT NOT NULL,
                model_name TEXT NOT NULL,
                universe TEXT NOT NULL,
                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL,
                initial_capital REAL NOT NULL,
                final_equity REAL NOT NULL,
                total_return REAL NOT NULL,
                annualized_return REAL NOT NULL,
                annualized_volatility REAL NOT NULL,
                sharpe REAL NOT NULL,
                max_drawdown REAL NOT NULL,
                avg_turnover REAL NOT NULL,
                win_rate REAL NOT NULL,
                metrics_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS backtest_equity (
                backtest_id INTEGER NOT NULL,
                date TEXT NOT NULL,
                equity REAL NOT NULL,
                return REAL NOT NULL,
                drawdown REAL NOT NULL,
                PRIMARY KEY (backtest_id, date),
                FOREIGN KEY (backtest_id) REFERENCES backtest_results(backtest_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS experiment_logs (
                experiment_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_name TEXT NOT NULL,
                model_name TEXT NOT NULL,
                strategy_name TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                dataset_version TEXT NOT NULL,
                params_json TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                notes TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS strategy_registry (
                strategy_name TEXT PRIMARY KEY,
                family TEXT NOT NULL,
                description_it TEXT NOT NULL,
                formula TEXT NOT NULL,
                assumptions_it TEXT NOT NULL,
                limitations_it TEXT NOT NULL,
                default_params_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS ingestion_runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                job_name TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                rows_written INTEGER NOT NULL,
                artifacts_json TEXT NOT NULL,
                message TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS data_catalog_entries (
                dataset_id TEXT PRIMARY KEY,
                domain TEXT NOT NULL,
                table_name TEXT NOT NULL,
                title_it TEXT NOT NULL,
                description_it TEXT NOT NULL,
                frequency TEXT NOT NULL,
                primary_key TEXT NOT NULL,
                storage TEXT NOT NULL,
                owner TEXT NOT NULL,
                freshness_rule TEXT NOT NULL,
                sample_query TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_ohlcv_date ON ohlcv_daily(date)",
            "CREATE INDEX IF NOT EXISTS idx_features_date ON features_labels(date)",
            "CREATE INDEX IF NOT EXISTS idx_signals_date ON ml_signals(date)",
            "CREATE INDEX IF NOT EXISTS idx_backtests_strategy ON backtest_results(strategy_name, created_at)",
        ]
        with self.connect() as conn:
            for statement in ddl:
                conn.execute(statement)
            conn.execute(
                "INSERT OR REPLACE INTO metadata(key, value, updated_at) VALUES (?, ?, ?)",
                ("schema_version", SCHEMA_VERSION, utc_now_iso()),
            )
            conn.commit()

    def upsert_instruments(self, instruments: pd.DataFrame) -> dict[str, int]:
        """Upsert instrument metadata and return a ``symbol -> id`` map."""

        if instruments.empty:
            return {}
        self.init_schema()
        rows = instruments.copy()
        now = utc_now_iso()
        defaults = {
            "name": "",
            "asset_class": "equity",
            "exchange": "",
            "country": "",
            "currency": "",
            "sector": "",
            "industry": "",
            "provider_symbol": "",
            "active": 1,
            "metadata_json": "{}",
            "created_at": now,
            "updated_at": now,
        }
        for column, default in defaults.items():
            if column not in rows.columns:
                rows[column] = default
        rows["symbol"] = rows["symbol"].astype(str).str.upper()
        rows["provider_symbol"] = rows["provider_symbol"].where(rows["provider_symbol"].astype(str).str.len() > 0, rows["symbol"])
        records = rows[
            [
                "symbol",
                "name",
                "asset_class",
                "exchange",
                "country",
                "currency",
                "sector",
                "industry",
                "provider_symbol",
                "active",
                "metadata_json",
                "created_at",
                "updated_at",
            ]
        ].to_dict("records")
        sql = """
        INSERT INTO instruments (
            symbol, name, asset_class, exchange, country, currency, sector, industry,
            provider_symbol, active, metadata_json, created_at, updated_at
        )
        VALUES (
            :symbol, :name, :asset_class, :exchange, :country, :currency, :sector, :industry,
            :provider_symbol, :active, :metadata_json, :created_at, :updated_at
        )
        ON CONFLICT(symbol) DO UPDATE SET
            name=excluded.name,
            asset_class=excluded.asset_class,
            exchange=excluded.exchange,
            country=excluded.country,
            currency=excluded.currency,
            sector=excluded.sector,
            industry=excluded.industry,
            provider_symbol=excluded.provider_symbol,
            active=excluded.active,
            metadata_json=excluded.metadata_json,
            updated_at=excluded.updated_at
        """
        with self.connect() as conn:
            conn.executemany(sql, records)
            conn.commit()
        return self.instrument_id_map()

    def instrument_id_map(self) -> dict[str, int]:
        self.init_schema()
        with self.connect() as conn:
            rows = conn.execute("SELECT symbol, instrument_id FROM instruments").fetchall()
        return {str(row["symbol"]): int(row["instrument_id"]) for row in rows}

    def write_ohlcv(self, ohlcv: pd.DataFrame, source: str = "synthetic_sample") -> int:
        if ohlcv.empty:
            return 0
        symbol_map = self.instrument_id_map()
        rows = ohlcv.copy()
        rows["symbol"] = rows["symbol"].astype(str).str.upper()
        rows["instrument_id"] = rows["symbol"].map(symbol_map)
        rows = rows.dropna(subset=["instrument_id"]).copy()
        rows["instrument_id"] = rows["instrument_id"].astype(int)
        rows["date"] = pd.to_datetime(rows["date"]).dt.date.astype(str)
        rows["source"] = rows.get("source", source)
        rows["ingestion_ts"] = rows.get("ingestion_ts", utc_now_iso())
        columns = ["instrument_id", "date", "open", "high", "low", "close", "adj_close", "volume", "source", "ingestion_ts"]
        sql = """
        INSERT INTO ohlcv_daily (instrument_id, date, open, high, low, close, adj_close, volume, source, ingestion_ts)
        VALUES (:instrument_id, :date, :open, :high, :low, :close, :adj_close, :volume, :source, :ingestion_ts)
        ON CONFLICT(instrument_id, date, source) DO UPDATE SET
            open=excluded.open,
            high=excluded.high,
            low=excluded.low,
            close=excluded.close,
            adj_close=excluded.adj_close,
            volume=excluded.volume,
            ingestion_ts=excluded.ingestion_ts
        """
        return self._executemany(sql, rows[columns])

    def write_features_labels(self, features: pd.DataFrame, feature_set: str = "final_v1") -> int:
        if features.empty:
            return 0
        symbol_map = self.instrument_id_map()
        rows = features.copy()
        rows["symbol"] = rows["symbol"].astype(str).str.upper()
        rows["instrument_id"] = rows["symbol"].map(symbol_map)
        rows = rows.dropna(subset=["instrument_id"]).copy()
        rows["instrument_id"] = rows["instrument_id"].astype(int)
        rows["date"] = pd.to_datetime(rows["date"]).dt.date.astype(str)
        rows["feature_set"] = rows.get("feature_set", feature_set)
        rows["source"] = rows.get("source", "synthetic_sample")
        rows["created_at"] = rows.get("created_at", utc_now_iso())
        columns = [
            "instrument_id",
            "date",
            "feature_set",
            "return_1d",
            "return_21d",
            "momentum_63d",
            "volatility_21d",
            "zscore_63d",
            "volume_zscore_21d",
            "label_forward_21d",
            "label_direction_21d",
            "source",
            "created_at",
        ]
        sql = """
        INSERT INTO features_labels (
            instrument_id, date, feature_set, return_1d, return_21d, momentum_63d,
            volatility_21d, zscore_63d, volume_zscore_21d, label_forward_21d,
            label_direction_21d, source, created_at
        )
        VALUES (
            :instrument_id, :date, :feature_set, :return_1d, :return_21d, :momentum_63d,
            :volatility_21d, :zscore_63d, :volume_zscore_21d, :label_forward_21d,
            :label_direction_21d, :source, :created_at
        )
        ON CONFLICT(instrument_id, date, feature_set) DO UPDATE SET
            return_1d=excluded.return_1d,
            return_21d=excluded.return_21d,
            momentum_63d=excluded.momentum_63d,
            volatility_21d=excluded.volatility_21d,
            zscore_63d=excluded.zscore_63d,
            volume_zscore_21d=excluded.volume_zscore_21d,
            label_forward_21d=excluded.label_forward_21d,
            label_direction_21d=excluded.label_direction_21d,
            source=excluded.source,
            created_at=excluded.created_at
        """
        return self._executemany(sql, rows[columns])

    def write_signals(self, signals: pd.DataFrame) -> int:
        if signals.empty:
            return 0
        symbol_map = self.instrument_id_map()
        rows = signals.copy()
        rows["symbol"] = rows["symbol"].astype(str).str.upper()
        rows["instrument_id"] = rows["symbol"].map(symbol_map)
        rows = rows.dropna(subset=["instrument_id"]).copy()
        rows["instrument_id"] = rows["instrument_id"].astype(int)
        rows["date"] = pd.to_datetime(rows["date"]).dt.date.astype(str)
        rows["created_at"] = rows.get("created_at", utc_now_iso())
        columns = [
            "instrument_id",
            "date",
            "strategy_name",
            "model_name",
            "signal",
            "target_weight",
            "score",
            "confidence",
            "horizon",
            "reasoning",
            "status",
            "created_at",
        ]
        sql = """
        INSERT INTO ml_signals (
            instrument_id, date, strategy_name, model_name, signal, target_weight,
            score, confidence, horizon, reasoning, status, created_at
        )
        VALUES (
            :instrument_id, :date, :strategy_name, :model_name, :signal, :target_weight,
            :score, :confidence, :horizon, :reasoning, :status, :created_at
        )
        ON CONFLICT(instrument_id, date, strategy_name, model_name, horizon) DO UPDATE SET
            signal=excluded.signal,
            target_weight=excluded.target_weight,
            score=excluded.score,
            confidence=excluded.confidence,
            reasoning=excluded.reasoning,
            status=excluded.status,
            created_at=excluded.created_at
        """
        return self._executemany(sql, rows[columns])

    def write_backtest(self, summary: dict[str, Any], equity_curve: pd.DataFrame) -> int:
        self.init_schema()
        created_at = summary.get("created_at") or utc_now_iso()
        metrics_json = json.dumps(summary.get("metrics", {}), sort_keys=True)
        columns = [
            "strategy_name",
            "model_name",
            "universe",
            "start_date",
            "end_date",
            "initial_capital",
            "final_equity",
            "total_return",
            "annualized_return",
            "annualized_volatility",
            "sharpe",
            "max_drawdown",
            "avg_turnover",
            "win_rate",
            "metrics_json",
            "created_at",
        ]
        payload = {column: summary.get(column) for column in columns}
        payload["metrics_json"] = metrics_json
        payload["created_at"] = created_at
        sql = """
        INSERT INTO backtest_results (
            strategy_name, model_name, universe, start_date, end_date, initial_capital,
            final_equity, total_return, annualized_return, annualized_volatility,
            sharpe, max_drawdown, avg_turnover, win_rate, metrics_json, created_at
        )
        VALUES (
            :strategy_name, :model_name, :universe, :start_date, :end_date, :initial_capital,
            :final_equity, :total_return, :annualized_return, :annualized_volatility,
            :sharpe, :max_drawdown, :avg_turnover, :win_rate, :metrics_json, :created_at
        )
        """
        with self.connect() as conn:
            cursor = conn.execute(sql, payload)
            backtest_id = int(cursor.lastrowid)
            rows = equity_curve.copy()
            rows["backtest_id"] = backtest_id
            rows["date"] = pd.to_datetime(rows["date"]).dt.date.astype(str)
            conn.executemany(
                """
                INSERT OR REPLACE INTO backtest_equity (backtest_id, date, equity, return, drawdown)
                VALUES (:backtest_id, :date, :equity, :return, :drawdown)
                """,
                rows[["backtest_id", "date", "equity", "return", "drawdown"]].to_dict("records"),
            )
            conn.commit()
        return backtest_id

    def write_strategy_registry(self, strategies: pd.DataFrame) -> int:
        if strategies.empty:
            return 0
        rows = strategies.copy()
        rows["updated_at"] = rows.get("updated_at", utc_now_iso())
        columns = [
            "strategy_name",
            "family",
            "description_it",
            "formula",
            "assumptions_it",
            "limitations_it",
            "default_params_json",
            "updated_at",
        ]
        sql = """
        INSERT INTO strategy_registry (
            strategy_name, family, description_it, formula, assumptions_it,
            limitations_it, default_params_json, updated_at
        )
        VALUES (
            :strategy_name, :family, :description_it, :formula, :assumptions_it,
            :limitations_it, :default_params_json, :updated_at
        )
        ON CONFLICT(strategy_name) DO UPDATE SET
            family=excluded.family,
            description_it=excluded.description_it,
            formula=excluded.formula,
            assumptions_it=excluded.assumptions_it,
            limitations_it=excluded.limitations_it,
            default_params_json=excluded.default_params_json,
            updated_at=excluded.updated_at
        """
        return self._executemany(sql, rows[columns])

    def write_experiment_logs(self, experiments: pd.DataFrame) -> int:
        if experiments.empty:
            return 0
        columns = [
            "run_name",
            "model_name",
            "strategy_name",
            "status",
            "started_at",
            "finished_at",
            "dataset_version",
            "params_json",
            "metrics_json",
            "notes",
        ]
        sql = """
        INSERT INTO experiment_logs (
            run_name, model_name, strategy_name, status, started_at, finished_at,
            dataset_version, params_json, metrics_json, notes
        )
        VALUES (
            :run_name, :model_name, :strategy_name, :status, :started_at, :finished_at,
            :dataset_version, :params_json, :metrics_json, :notes
        )
        """
        return self._executemany(sql, experiments[columns])

    def write_ingestion_run(self, run: dict[str, Any]) -> int:
        self.init_schema()
        payload = {
            "source": run.get("source", "synthetic_sample"),
            "job_name": run.get("job_name", "populate_research_database"),
            "status": run.get("status", "ok"),
            "started_at": run.get("started_at", utc_now_iso()),
            "finished_at": run.get("finished_at", utc_now_iso()),
            "rows_written": int(run.get("rows_written", 0)),
            "artifacts_json": json.dumps(run.get("artifacts", {}), sort_keys=True),
            "message": run.get("message", ""),
        }
        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO ingestion_runs (source, job_name, status, started_at, finished_at, rows_written, artifacts_json, message)
                VALUES (:source, :job_name, :status, :started_at, :finished_at, :rows_written, :artifacts_json, :message)
                """,
                payload,
            )
            conn.commit()
            return int(cursor.lastrowid)

    def write_data_catalog(self, catalog: pd.DataFrame) -> int:
        if catalog.empty:
            return 0
        rows = catalog.copy()
        rows["updated_at"] = rows.get("updated_at", utc_now_iso())
        columns = [
            "dataset_id",
            "domain",
            "table_name",
            "title_it",
            "description_it",
            "frequency",
            "primary_key",
            "storage",
            "owner",
            "freshness_rule",
            "sample_query",
            "updated_at",
        ]
        sql = """
        INSERT INTO data_catalog_entries (
            dataset_id, domain, table_name, title_it, description_it, frequency,
            primary_key, storage, owner, freshness_rule, sample_query, updated_at
        )
        VALUES (
            :dataset_id, :domain, :table_name, :title_it, :description_it, :frequency,
            :primary_key, :storage, :owner, :freshness_rule, :sample_query, :updated_at
        )
        ON CONFLICT(dataset_id) DO UPDATE SET
            domain=excluded.domain,
            table_name=excluded.table_name,
            title_it=excluded.title_it,
            description_it=excluded.description_it,
            frequency=excluded.frequency,
            primary_key=excluded.primary_key,
            storage=excluded.storage,
            owner=excluded.owner,
            freshness_rule=excluded.freshness_rule,
            sample_query=excluded.sample_query,
            updated_at=excluded.updated_at
        """
        return self._executemany(sql, rows[columns])

    def table_counts(self) -> dict[str, int]:
        self.init_schema()
        tables = [
            "instruments",
            "ohlcv_daily",
            "features_labels",
            "ml_signals",
            "backtest_results",
            "backtest_equity",
            "experiment_logs",
            "strategy_registry",
            "ingestion_runs",
            "data_catalog_entries",
        ]
        with self.connect() as conn:
            return {table: int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]) for table in tables}

    def read_table(self, table_name: str, limit: int | None = None) -> pd.DataFrame:
        self.init_schema()
        allowed = {
            "instruments",
            "ohlcv_daily",
            "features_labels",
            "ml_signals",
            "backtest_results",
            "backtest_equity",
            "experiment_logs",
            "strategy_registry",
            "ingestion_runs",
            "data_catalog_entries",
        }
        if table_name not in allowed:
            raise ValueError(f"Unknown canonical table: {table_name}")
        query = f"SELECT * FROM {table_name}"
        if limit is not None:
            query += f" LIMIT {int(limit)}"
        with self.connect() as conn:
            return pd.read_sql_query(query, conn)

    def list_strategies(self) -> pd.DataFrame:
        self.init_schema()
        with self.connect() as conn:
            return pd.read_sql_query("SELECT * FROM strategy_registry ORDER BY strategy_name", conn)

    def latest_signals(self, limit: int = 50) -> pd.DataFrame:
        self.init_schema()
        query = """
        SELECT
            s.date,
            i.symbol,
            i.name,
            i.sector,
            s.strategy_name,
            s.model_name,
            s.signal,
            s.target_weight,
            s.score,
            s.confidence,
            s.horizon,
            s.reasoning,
            s.status
        FROM ml_signals s
        JOIN instruments i ON i.instrument_id = s.instrument_id
        ORDER BY s.date DESC, ABS(s.target_weight) DESC, i.symbol
        LIMIT ?
        """
        with self.connect() as conn:
            return pd.read_sql_query(query, conn, params=(int(limit),))

    def latest_backtests(self, limit: int = 20) -> pd.DataFrame:
        self.init_schema()
        query = """
        SELECT *
        FROM backtest_results
        ORDER BY created_at DESC, backtest_id DESC
        LIMIT ?
        """
        with self.connect() as conn:
            return pd.read_sql_query(query, conn, params=(int(limit),))

    def _executemany(self, sql: str, frame: pd.DataFrame) -> int:
        self.init_schema()
        clean = frame.replace([np.inf, -np.inf], np.nan).where(pd.notnull(frame), None)
        records = clean.to_dict("records")
        with self.connect() as conn:
            conn.executemany(sql, records)
            conn.commit()
        return len(records)


def build_strategy_registry() -> pd.DataFrame:
    """Return the canonical starter strategy registry with formulas."""

    rows = [
        {
            "strategy_name": "momentum_baseline",
            "family": "trend_following",
            "description_it": "Compra gli strumenti con rendimento a 63 sedute positivo e resta flat sugli altri.",
            "formula": "score_i,t = P_i,t / P_i,t-63 - 1; w_i,t = max(score_i,t, 0) / sum(max(score_j,t, 0))",
            "assumptions_it": "Il trend recente contiene informazione utile e non e' solo rumore. Il ribilanciamento usa solo dati gia' disponibili.",
            "limitations_it": "Soffre nei cambi di regime, nei mercati laterali e se i costi di rotazione aumentano.",
            "default_params_json": json.dumps({"lookback_days": 63, "rebalance": "daily", "long_only": True}, sort_keys=True),
        },
        {
            "strategy_name": "mispricing_zscore",
            "family": "relative_value",
            "description_it": "Cerca prezzi lontani dalla media mobile: positivo se il prezzo recupera dopo eccessi temporanei.",
            "formula": "z_i,t = (P_i,t - SMA_63(P_i,t)) / sigma_63(P_i,t); signal = -z_i,t per mean reversion, con filtri di rischio.",
            "assumptions_it": "Alcuni scostamenti estremi sono eccessi temporanei e non cambi strutturali di valore.",
            "limitations_it": "Un titolo puo' sembrare economico mentre sta prezzando un deterioramento reale.",
            "default_params_json": json.dumps({"window": 63, "entry_z": 1.0, "risk_filter": "volatility_21d"}, sort_keys=True),
        },
        {
            "strategy_name": "risk_balanced_blend",
            "family": "multi_factor",
            "description_it": "Combina momentum e z-score, poi riduce il peso degli strumenti piu' volatili.",
            "formula": "raw_i,t = 0.60*zscore_i,t + 0.40*momentum_i,t/vol_21d; w_i,t proportional max(raw_i,t, 0)/vol_21d",
            "assumptions_it": "Segnali diversi si compensano: momentum evita value trap, controllo volatilita' evita concentrazione di rischio.",
            "limitations_it": "La normalizzazione cross-section puo' amplificare errori se l'universo e' piccolo o molto correlato.",
            "default_params_json": json.dumps({"momentum_window": 63, "vol_window": 21, "max_weight": 0.25}, sort_keys=True),
        },
    ]
    return pd.DataFrame(rows)


def build_data_catalog_entries() -> pd.DataFrame:
    """Return Data Center catalog entries for the canonical SQLite schema."""

    rows = [
        {
            "dataset_id": "instrument_master",
            "domain": "reference_data",
            "table_name": "instruments",
            "title_it": "Anagrafica strumenti",
            "description_it": "Ticker, mercato, valuta, settore e simbolo provider usati da app e notebook.",
            "frequency": "on_change",
            "primary_key": "symbol",
            "storage": "sqlite:research_platform.sqlite",
            "owner": "Data Center",
            "freshness_rule": "Aggiorna quando cambia universo o provider.",
            "sample_query": "SELECT symbol, name, exchange, sector FROM instruments ORDER BY symbol;",
        },
        {
            "dataset_id": "ohlcv_daily",
            "domain": "market_data",
            "table_name": "ohlcv_daily",
            "title_it": "Prezzi OHLCV giornalieri",
            "description_it": "Open, high, low, close, adjusted close e volumi con tracciamento fonte.",
            "frequency": "daily",
            "primary_key": "instrument_id,date,source",
            "storage": "sqlite:research_platform.sqlite + parquet lake opzionale",
            "owner": "Data Center",
            "freshness_rule": "Refresh giornaliero per strumenti attivi; API solo se cache stale.",
            "sample_query": "SELECT * FROM ohlcv_daily ORDER BY date DESC LIMIT 20;",
        },
        {
            "dataset_id": "features_labels_final_v1",
            "domain": "ml_features",
            "table_name": "features_labels",
            "title_it": "Feature e label ML",
            "description_it": "Rendimenti, momentum, volatilita', z-score e label forward per esperimenti ML.",
            "frequency": "daily_after_prices",
            "primary_key": "instrument_id,date,feature_set",
            "storage": "sqlite:research_platform.sqlite",
            "owner": "Research Platform",
            "freshness_rule": "Rigenera dopo ogni refresh OHLCV o cambio formula.",
            "sample_query": "SELECT * FROM features_labels WHERE feature_set='final_v1' LIMIT 20;",
        },
        {
            "dataset_id": "ml_signals_latest",
            "domain": "signals",
            "table_name": "ml_signals",
            "title_it": "Segnali ML e target weight",
            "description_it": "Segnali pronti per dashboard, mobile e bridge QuantDinger.",
            "frequency": "on_backtest_or_refresh",
            "primary_key": "instrument_id,date,strategy_name,model_name,horizon",
            "storage": "sqlite:research_platform.sqlite",
            "owner": "LLM Lab / Research Platform",
            "freshness_rule": "Aggiorna a fine run; invalida se dati o modello cambiano.",
            "sample_query": "SELECT * FROM ml_signals ORDER BY date DESC LIMIT 20;",
        },
        {
            "dataset_id": "backtest_results",
            "domain": "backtests",
            "table_name": "backtest_results",
            "title_it": "Storico backtest",
            "description_it": "Metriche aggregate, universo, finestra e configurazione modello.",
            "frequency": "on_run",
            "primary_key": "backtest_id",
            "storage": "sqlite:research_platform.sqlite",
            "owner": "Research Platform",
            "freshness_rule": "Append-only: ogni esperimento produce un nuovo record.",
            "sample_query": "SELECT strategy_name, total_return, sharpe, max_drawdown FROM backtest_results;",
        },
    ]
    return pd.DataFrame(rows)


def build_synthetic_research_dataset(
    symbols: tuple[str, ...] | list[str] = DEFAULT_SYMBOLS,
    start: str = "2021-01-04",
    end: str = "2026-05-22",
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Build a realistic, deterministic sample dataset for local operation."""

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, end=end)
    market_shock = rng.normal(0.00025, 0.009, size=len(dates))
    instruments = _instrument_master(symbols)
    ohlcv_frames: list[pd.DataFrame] = []

    for idx, row in enumerate(instruments.itertuples(index=False)):
        beta = 0.75 + 0.12 * (idx % 5)
        drift = 0.06 + 0.015 * ((idx % 4) - 1)
        vol = 0.18 + 0.025 * (idx % 6)
        idiosyncratic = rng.normal(drift / 252.0, vol / np.sqrt(252.0), size=len(dates))
        shocks = beta * market_shock + idiosyncratic
        base_price = 35.0 + idx * 18.0 + rng.uniform(0, 12)
        close = base_price * np.exp(np.cumsum(shocks))
        close = np.maximum(close, 1.0)
        prev_close = np.r_[close[0], close[:-1]]
        open_ = prev_close * (1.0 + rng.normal(0.0, 0.004, size=len(dates)))
        span = np.abs(rng.normal(0.006, 0.004, size=len(dates)))
        high = np.maximum(open_, close) * (1.0 + span)
        low = np.minimum(open_, close) * (1.0 - span)
        base_volume = 1_200_000 + idx * 320_000
        volume = np.maximum(base_volume * rng.lognormal(0.0, 0.25, size=len(dates)), 10_000)
        ohlcv_frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "symbol": row.symbol,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "adj_close": close,
                    "volume": volume.round(0),
                    "source": "synthetic_sample",
                    "ingestion_ts": utc_now_iso(),
                }
            )
        )

    ohlcv = pd.concat(ohlcv_frames, ignore_index=True)
    features = _build_features(ohlcv)
    signals = _build_signals(features)
    backtest_summary, equity_curve = _build_backtest(ohlcv, features, symbols)
    experiments = _build_experiment_logs(backtest_summary)
    return {
        "instruments": instruments,
        "ohlcv": ohlcv,
        "features": features,
        "signals": signals,
        "backtest_summary": pd.DataFrame([backtest_summary]),
        "backtest_equity": equity_curve,
        "experiments": experiments,
        "strategies": build_strategy_registry(),
        "catalog": build_data_catalog_entries(),
    }


def populate_research_database(
    db_path: Path | str | None = None,
    sample_dir: Path | str | None = None,
    start: str = "2021-01-04",
    end: str = "2026-05-22",
    seed: int = 42,
    sample_rows_per_symbol: int = 252,
) -> dict[str, Any]:
    """Create or refresh the canonical SQLite DB and lightweight sample CSVs."""

    started_at = utc_now_iso()
    db = ResearchDatabase(db_path)
    dataset = build_synthetic_research_dataset(start=start, end=end, seed=seed)
    db.init_schema()
    db.upsert_instruments(dataset["instruments"])
    rows_written = {
        "instruments": len(dataset["instruments"]),
        "ohlcv_daily": db.write_ohlcv(dataset["ohlcv"]),
        "features_labels": db.write_features_labels(dataset["features"]),
        "ml_signals": db.write_signals(dataset["signals"]),
        "strategy_registry": db.write_strategy_registry(dataset["strategies"]),
        "experiment_logs": db.write_experiment_logs(dataset["experiments"]),
        "data_catalog_entries": db.write_data_catalog(dataset["catalog"]),
    }
    backtest_id = db.write_backtest(dataset["backtest_summary"].iloc[0].to_dict(), dataset["backtest_equity"])
    rows_written["backtest_results"] = 1
    rows_written["backtest_equity"] = len(dataset["backtest_equity"])

    artifacts: dict[str, str] = {"database": str(db.path)}
    if sample_dir is None:
        sample_dir = research_platform_root() / "data" / "sample"
    sample_paths = write_sample_csvs(dataset, sample_dir, sample_rows_per_symbol=sample_rows_per_symbol)
    artifacts.update({key: str(path) for key, path in sample_paths.items()})
    db.write_ingestion_run(
        {
            "source": "synthetic_sample",
            "job_name": "populate_research_database",
            "status": "ok",
            "started_at": started_at,
            "finished_at": utc_now_iso(),
            "rows_written": int(sum(rows_written.values())),
            "artifacts": artifacts,
            "message": "Synthetic but market-shaped dataset ready for local demos, tests and mobile snapshots.",
        }
    )
    counts = db.table_counts()
    summary = {"db_path": str(db.path), "artifacts": artifacts, "rows_written": rows_written, "table_counts": counts, "backtest_id": backtest_id}
    summary_path = Path(sample_dir).expanduser() / "research_database_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    summary["artifacts"]["summary"] = str(summary_path)
    return summary


def write_sample_csvs(
    dataset: dict[str, pd.DataFrame],
    sample_dir: Path | str,
    sample_rows_per_symbol: int = 252,
) -> dict[str, Path]:
    """Write lightweight CSV extracts intended to be tracked in Git."""

    out = Path(sample_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    ohlcv_sample = dataset["ohlcv"].groupby("symbol", group_keys=False).tail(sample_rows_per_symbol)
    features_sample = dataset["features"].groupby("symbol", group_keys=False).tail(sample_rows_per_symbol)
    paths = {
        "instruments_sample": out / "instruments_sample.csv",
        "ohlcv_daily_sample": out / "ohlcv_daily_sample.csv",
        "features_labels_sample": out / "features_labels_sample.csv",
        "ml_signals_sample": out / "ml_signals_sample.csv",
        "backtest_summary_sample": out / "backtest_summary_sample.csv",
        "backtest_equity_sample": out / "backtest_equity_sample.csv",
        "strategy_registry_sample": out / "strategy_registry_sample.csv",
        "data_catalog_sample": out / "data_catalog_sample.csv",
    }
    dataset["instruments"].to_csv(paths["instruments_sample"], index=False)
    ohlcv_sample.to_csv(paths["ohlcv_daily_sample"], index=False)
    features_sample.to_csv(paths["features_labels_sample"], index=False)
    dataset["signals"].to_csv(paths["ml_signals_sample"], index=False)
    dataset["backtest_summary"].to_csv(paths["backtest_summary_sample"], index=False)
    dataset["backtest_equity"].tail(sample_rows_per_symbol).to_csv(paths["backtest_equity_sample"], index=False)
    dataset["strategies"].to_csv(paths["strategy_registry_sample"], index=False)
    dataset["catalog"].to_csv(paths["data_catalog_sample"], index=False)
    return paths


def _instrument_master(symbols: tuple[str, ...] | list[str]) -> pd.DataFrame:
    metadata = {
        "AAPL": ("Apple Inc.", "NASDAQ", "US", "USD", "Information Technology", "Consumer Electronics"),
        "MSFT": ("Microsoft Corp.", "NASDAQ", "US", "USD", "Information Technology", "Software"),
        "NVDA": ("NVIDIA Corp.", "NASDAQ", "US", "USD", "Information Technology", "Semiconductors"),
        "SPY": ("SPDR S&P 500 ETF Trust", "NYSEARCA", "US", "USD", "ETF", "Broad Market"),
        "ENEL.MI": ("Enel SpA", "MIL", "IT", "EUR", "Utilities", "Electric Utilities"),
        "ISP.MI": ("Intesa Sanpaolo SpA", "MIL", "IT", "EUR", "Financials", "Banks"),
        "ASML.AS": ("ASML Holding NV", "AMS", "NL", "EUR", "Information Technology", "Semiconductor Equipment"),
        "SAP.DE": ("SAP SE", "XETRA", "DE", "EUR", "Information Technology", "Software"),
    }
    rows = []
    now = utc_now_iso()
    for symbol in symbols:
        name, exchange, country, currency, sector, industry = metadata.get(
            symbol,
            (symbol, "", "", "USD", "Unknown", "Unknown"),
        )
        rows.append(
            {
                "symbol": symbol.upper(),
                "name": name,
                "asset_class": "etf" if symbol.upper() == "SPY" else "equity",
                "exchange": exchange,
                "country": country,
                "currency": currency,
                "sector": sector,
                "industry": industry,
                "provider_symbol": symbol.upper(),
                "active": 1,
                "metadata_json": json.dumps({"sample": True, "schema_version": SCHEMA_VERSION}, sort_keys=True),
                "created_at": now,
                "updated_at": now,
            }
        )
    return pd.DataFrame(rows)


def _build_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for symbol, frame in ohlcv.sort_values(["symbol", "date"]).groupby("symbol"):
        data = frame[["date", "symbol", "close", "volume"]].copy()
        data["return_1d"] = data["close"].pct_change(fill_method=None)
        data["return_21d"] = data["close"].pct_change(21, fill_method=None)
        data["momentum_63d"] = data["close"].pct_change(63, fill_method=None)
        data["volatility_21d"] = data["return_1d"].rolling(21).std(ddof=0) * np.sqrt(252)
        rolling_mean = data["close"].rolling(63).mean()
        rolling_std = data["close"].rolling(63).std(ddof=0)
        data["zscore_63d"] = (data["close"] - rolling_mean) / rolling_std.replace(0.0, np.nan)
        volume_mean = data["volume"].rolling(21).mean()
        volume_std = data["volume"].rolling(21).std(ddof=0)
        data["volume_zscore_21d"] = (data["volume"] - volume_mean) / volume_std.replace(0.0, np.nan)
        data["label_forward_21d"] = data["close"].shift(-21) / data["close"] - 1.0
        data["label_direction_21d"] = (data["label_forward_21d"] > 0).astype(int)
        data["feature_set"] = "final_v1"
        data["source"] = "synthetic_sample"
        data["created_at"] = utc_now_iso()
        frames.append(data.drop(columns=["close", "volume"]))
    features = pd.concat(frames, ignore_index=True)
    return features.dropna(subset=["momentum_63d", "volatility_21d", "zscore_63d"]).reset_index(drop=True)


def _build_signals(features: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    data = features.copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values(["date", "symbol"])
    month_ends = data.groupby([data["date"].dt.to_period("M"), "symbol"]).tail(1)
    for date, frame in month_ends.groupby("date"):
        frame = frame.copy()
        risk = frame["volatility_21d"].replace(0.0, np.nan)
        raw_score = 0.60 * frame["zscore_63d"].fillna(0.0) + 0.40 * (frame["momentum_63d"].fillna(0.0) / risk).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        positive = raw_score.clip(lower=0.0)
        if float(positive.sum()) == 0.0:
            weights = pd.Series(0.0, index=frame.index)
        else:
            weights = positive / positive.sum()
            weights = weights.clip(upper=0.25)
            if weights.sum() > 0:
                weights = weights / weights.sum()
        out = pd.DataFrame(
            {
                "date": date.date().isoformat(),
                "symbol": frame["symbol"].values,
                "strategy_name": "risk_balanced_blend",
                "model_name": "synthetic_factor_baseline",
                "signal": np.sign(raw_score).astype(float).values,
                "target_weight": weights.values,
                "score": raw_score.values,
                "confidence": np.minimum(np.abs(raw_score.values) / 2.5, 1.0),
                "horizon": "21D",
                "reasoning": "Blend di z-score prezzo, momentum a 63 sedute e penalita' per volatilita' recente.",
                "status": "paper",
                "created_at": utc_now_iso(),
            }
        )
        rows.append(out)
    return pd.concat(rows, ignore_index=True)


def _build_backtest(ohlcv: pd.DataFrame, features: pd.DataFrame, symbols: tuple[str, ...] | list[str]) -> tuple[dict[str, Any], pd.DataFrame]:
    prices = ohlcv.pivot_table(index=pd.to_datetime(ohlcv["date"]), columns="symbol", values="close").sort_index().ffill()
    returns = prices.pct_change(fill_method=None).fillna(0.0)
    feature_data = features.copy()
    feature_data["date"] = pd.to_datetime(feature_data["date"])
    score = (
        0.60 * feature_data.pivot_table(index="date", columns="symbol", values="zscore_63d")
        + 0.40
        * (
            feature_data.pivot_table(index="date", columns="symbol", values="momentum_63d")
            / feature_data.pivot_table(index="date", columns="symbol", values="volatility_21d").replace(0.0, np.nan)
        )
    ).replace([np.inf, -np.inf], np.nan)
    score = score.reindex(prices.index).ffill().fillna(0.0)
    raw = score.clip(lower=0.0)
    weights = raw.div(raw.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0).clip(upper=0.25)
    weights = weights.div(weights.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    turnover = weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1))
    net_returns = weights.shift(1).fillna(0.0).mul(returns, axis=0).sum(axis=1) - turnover * (5.0 / 10_000.0)
    equity = 100_000.0 * (1.0 + net_returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    clean = net_returns.dropna()
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0) if len(equity) > 1 else 0.0
    annualized_vol = float(clean.std(ddof=0) * np.sqrt(252)) if len(clean) else 0.0
    sharpe = float(clean.mean() / clean.std(ddof=0) * np.sqrt(252)) if clean.std(ddof=0) else 0.0
    metrics = {
        "total_return": total_return,
        "annualized_return": float((1.0 + total_return) ** (252.0 / max(len(clean), 1)) - 1.0),
        "annualized_volatility": annualized_vol,
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
        "avg_turnover": float(turnover.mean()),
        "win_rate": float((clean > 0).mean()),
        "observations": float(len(clean)),
    }
    summary = {
        "strategy_name": "risk_balanced_blend",
        "model_name": "synthetic_factor_baseline",
        "universe": ",".join(symbols),
        "start_date": prices.index.min().date().isoformat(),
        "end_date": prices.index.max().date().isoformat(),
        "initial_capital": 100_000.0,
        "final_equity": float(equity.iloc[-1]),
        **metrics,
        "metrics": metrics,
        "created_at": utc_now_iso(),
    }
    equity_curve = pd.DataFrame(
        {
            "date": equity.index.date.astype(str),
            "equity": equity.values,
            "return": net_returns.values,
            "drawdown": drawdown.values,
        }
    )
    return summary, equity_curve


def _build_experiment_logs(backtest_summary: dict[str, Any]) -> pd.DataFrame:
    now = utc_now_iso()
    rows = [
        {
            "run_name": "final_v1_synthetic_baseline",
            "model_name": "synthetic_factor_baseline",
            "strategy_name": "risk_balanced_blend",
            "status": "ok",
            "started_at": now,
            "finished_at": now,
            "dataset_version": SCHEMA_VERSION,
            "params_json": json.dumps({"seed": 42, "cost_bps": 5, "rebalance": "daily"}, sort_keys=True),
            "metrics_json": json.dumps(backtest_summary["metrics"], sort_keys=True),
            "notes": "Baseline rigenerabile per demo locale, test e mobile snapshot.",
        },
        {
            "run_name": "momentum_ablation_63d",
            "model_name": "momentum_only_baseline",
            "strategy_name": "momentum_baseline",
            "status": "ok",
            "started_at": now,
            "finished_at": now,
            "dataset_version": SCHEMA_VERSION,
            "params_json": json.dumps({"lookback_days": 63, "long_only": True}, sort_keys=True),
            "metrics_json": json.dumps({"expected_use": "ablation", "production_ready": False}, sort_keys=True),
            "notes": "Ablation per capire quanta performance viene dal momentum puro.",
        },
        {
            "run_name": "mispricing_zscore_design_note",
            "model_name": "zscore_research_note",
            "strategy_name": "mispricing_zscore",
            "status": "designed",
            "started_at": now,
            "finished_at": now,
            "dataset_version": SCHEMA_VERSION,
            "params_json": json.dumps({"window": 63, "entry_z": 1.0}, sort_keys=True),
            "metrics_json": json.dumps({"requires_next_step": "walk_forward_validation"}, sort_keys=True),
            "notes": "Strategia documentata; richiede validazione walk-forward prima di essere promossa.",
        },
    ]
    return pd.DataFrame(rows)
