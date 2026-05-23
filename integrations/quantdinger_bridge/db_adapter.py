"""Database adapter for QuantDinger persistence and market data access."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
import re
import time
from typing import Any, Iterable, Mapping

import pandas as pd

from .config import QuantDingerBridgeConfig, load_config

try:
    from sqlalchemy import Engine, bindparam, create_engine, inspect, text
except Exception:  # pragma: no cover - import guard for minimal environments
    Engine = Any  # type: ignore
    bindparam = None  # type: ignore
    create_engine = None  # type: ignore
    inspect = None  # type: ignore
    text = None  # type: ignore


logger = logging.getLogger(__name__)
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(slots=True)
class BacktestPersistencePayload:
    """Normalized payload for writing ML backtests into QuantDinger tables."""

    strategy_name: str
    market: str
    symbol: str
    timeframe: str
    start_date: str
    end_date: str
    initial_capital: float
    metrics: Mapping[str, Any]
    equity_curve: pd.DataFrame
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)
    config: Mapping[str, Any] = field(default_factory=dict)
    user_id: int = 1
    strategy_id: int | None = None
    status: str = "success"
    error_message: str = ""


def create_engine_from_config(config: QuantDingerBridgeConfig | None = None) -> Engine:
    """Create a SQLAlchemy engine for the QuantDinger database."""

    if create_engine is None:
        raise RuntimeError("SQLAlchemy is required for db_adapter. Install sqlalchemy and psycopg2-binary.")
    cfg = config or load_config()
    return create_engine(cfg.database.url, echo=cfg.database.echo, pool_pre_ping=True, future=True)


def ensure_ml_tables(engine: Engine) -> None:
    """Create bridge-owned ML tables without altering upstream QuantDinger code."""

    dialect = engine.dialect.name
    id_type = "SERIAL PRIMARY KEY" if dialect == "postgresql" else "INTEGER PRIMARY KEY AUTOINCREMENT"
    timestamp_default = "NOW()" if dialect == "postgresql" else "CURRENT_TIMESTAMP"
    real_type = "DOUBLE PRECISION" if dialect == "postgresql" else "REAL"

    statements = [
        f"""
        CREATE TABLE IF NOT EXISTS qd_ml_signals (
            id {id_type},
            user_id INTEGER NOT NULL DEFAULT 1,
            model_name VARCHAR(120) NOT NULL,
            market VARCHAR(50) NOT NULL DEFAULT '',
            symbol VARCHAR(80) NOT NULL,
            signal_date VARCHAR(32) NOT NULL DEFAULT '',
            signal {real_type} DEFAULT 0,
            confidence {real_type} DEFAULT 0,
            target_price {real_type},
            stop_loss {real_type},
            horizon VARCHAR(32) DEFAULT '',
            payload_json TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT {timestamp_default}
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS qd_ml_model_metrics (
            id {id_type},
            user_id INTEGER NOT NULL DEFAULT 1,
            model_name VARCHAR(120) NOT NULL,
            run_id INTEGER,
            metric_date VARCHAR(32) DEFAULT '',
            metrics_json TEXT NOT NULL DEFAULT '{{}}',
            feature_importance_json TEXT NOT NULL DEFAULT '{{}}',
            created_at TIMESTAMP DEFAULT {timestamp_default}
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS qd_ml_backtests (
            id {id_type},
            user_id INTEGER NOT NULL DEFAULT 1,
            qd_run_id INTEGER,
            model_name VARCHAR(120) NOT NULL,
            market VARCHAR(50) NOT NULL DEFAULT '',
            symbol VARCHAR(255) NOT NULL DEFAULT '',
            timeframe VARCHAR(20) NOT NULL DEFAULT '',
            start_date VARCHAR(32) NOT NULL DEFAULT '',
            end_date VARCHAR(32) NOT NULL DEFAULT '',
            metrics_json TEXT NOT NULL DEFAULT '{{}}',
            config_json TEXT NOT NULL DEFAULT '{{}}',
            created_at TIMESTAMP DEFAULT {timestamp_default}
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_qd_ml_signals_symbol_date ON qd_ml_signals(symbol, signal_date)",
        "CREATE INDEX IF NOT EXISTS idx_qd_ml_signals_model ON qd_ml_signals(model_name)",
        "CREATE INDEX IF NOT EXISTS idx_qd_ml_metrics_model ON qd_ml_model_metrics(model_name)",
        "CREATE INDEX IF NOT EXISTS idx_qd_ml_backtests_qd_run_id ON qd_ml_backtests(qd_run_id)",
    ]
    with engine.begin() as conn:
        for statement in statements:
            conn.execute(text(statement))


def fetch_ohlcv(
    engine: Engine,
    symbol: str,
    start: str | None = None,
    end: str | None = None,
    timeframe: str = "1D",
    market: str = "USStock",
    table_name: str | None = None,
) -> pd.DataFrame:
    """Fetch OHLCV rows from a QuantDinger-compatible market data table.

    QuantDinger's live K-line service often fetches from providers/cache rather
    than a fixed SQL table. This function supports direct DB reads when you
    materialize bars into one of the conventional table names.
    """

    table = table_name or _first_existing_table(engine, ["qd_market_ohlcv", "qd_klines", "market_ohlcv", "ohlcv"])
    if table is None:
        raise RuntimeError("No OHLCV table found. Use api_adapter.get_market_data or configure table_name.")
    table = _safe_identifier(table)

    where = ["symbol = :symbol"]
    params: dict[str, Any] = {"symbol": symbol, "market": market, "timeframe": timeframe}
    columns = _table_columns(engine, table)
    if "market" in columns:
        where.append("market = :market")
    if "timeframe" in columns:
        where.append("timeframe = :timeframe")
    date_col = _date_column(columns)
    if start and date_col:
        where.append(f"{date_col} >= :start")
        params["start"] = start
    if end and date_col:
        where.append(f"{date_col} <= :end")
        params["end"] = end

    sql = f"SELECT * FROM {_quote_identifier(engine, table)} WHERE {' AND '.join(where)}"
    if date_col:
        sql += f" ORDER BY {date_col}"
    return pd.read_sql_query(text(sql), engine, params=params)


def fetch_instruments(engine: Engine, market: str | None = None, active_only: bool = True) -> pd.DataFrame:
    """Fetch instrument metadata from QuantDinger's ``qd_market_symbols`` table."""

    if not _table_exists(engine, "qd_market_symbols"):
        return pd.DataFrame(columns=["market", "symbol", "name", "exchange", "currency"])
    where: list[str] = []
    params: dict[str, Any] = {}
    if market:
        where.append("market = :market")
        params["market"] = market
    if active_only:
        where.append("(is_active = 1 OR is_active = TRUE)")
    clause = f"WHERE {' AND '.join(where)}" if where else ""
    return pd.read_sql_query(text(f"SELECT * FROM qd_market_symbols {clause} ORDER BY market, sort_order DESC, symbol"), engine, params=params)


def fetch_positions(engine: Engine, user_id: int | None = None) -> pd.DataFrame:
    """Fetch strategy and manual positions from QuantDinger tables."""

    frames: list[pd.DataFrame] = []
    if _table_exists(engine, "qd_strategy_positions"):
        frames.append(_read_user_table(engine, "qd_strategy_positions", user_id))
    if _table_exists(engine, "qd_manual_positions"):
        frames.append(_read_user_table(engine, "qd_manual_positions", user_id))
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def fetch_orders(engine: Engine, user_id: int | None = None) -> pd.DataFrame:
    """Fetch pending and quick-trade orders from QuantDinger tables."""

    frames: list[pd.DataFrame] = []
    if _table_exists(engine, "pending_orders"):
        frames.append(_read_user_table(engine, "pending_orders", user_id))
    if _table_exists(engine, "qd_quick_trades"):
        frames.append(_read_user_table(engine, "qd_quick_trades", user_id))
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def write_backtest_result(engine: Engine, payload: BacktestPersistencePayload) -> int:
    """Persist an ML backtest into QuantDinger's backtest and ML metric tables."""

    started = time.perf_counter()
    ensure_ml_tables(engine)
    result_json = json.dumps({"metrics": dict(payload.metrics)}, default=_json_default, ensure_ascii=False)
    config_json = json.dumps(dict(payload.config), default=_json_default, ensure_ascii=False)

    with engine.begin() as conn:
        run_id = _insert_returning_id(
            conn,
            "qd_backtest_runs",
            {
                "user_id": payload.user_id,
                "strategy_id": payload.strategy_id,
                "strategy_name": payload.strategy_name,
                "run_type": "ml",
                "market": payload.market,
                "symbol": payload.symbol,
                "timeframe": payload.timeframe,
                "start_date": payload.start_date,
                "end_date": payload.end_date,
                "initial_capital": payload.initial_capital,
                "strategy_config": config_json,
                "config_snapshot": config_json,
                "engine_version": "ml4t-quantdinger-bridge-v1",
                "status": payload.status,
                "error_message": payload.error_message,
                "result_json": result_json,
            },
        )

        for index, row in payload.trades.reset_index(drop=True).iterrows():
            conn.execute(
                text(
                    """
                    INSERT INTO qd_backtest_trades
                    (run_id, user_id, strategy_id, trade_index, trade_time, trade_type, side,
                     price, amount, profit, balance, reason, payload_json)
                    VALUES
                    (:run_id, :user_id, :strategy_id, :trade_index, :trade_time, :trade_type, :side,
                     :price, :amount, :profit, :balance, :reason, :payload_json)
                    """
                ),
                {
                    "run_id": run_id,
                    "user_id": payload.user_id,
                    "strategy_id": payload.strategy_id,
                    "trade_index": int(index),
                    "trade_time": str(row.get("date") or row.get("trade_time") or ""),
                    "trade_type": str(row.get("trade_type") or "rebalance"),
                    "side": str(row.get("side") or row.get("ticker") or ""),
                    "price": _float(row.get("price")),
                    "amount": _float(row.get("amount") or row.get("delta_weight")),
                    "profit": _float(row.get("profit")),
                    "balance": _float(row.get("balance") or row.get("equity")),
                    "reason": str(row.get("reason") or "ml_bridge"),
                    "payload_json": json.dumps(row.to_dict(), default=_json_default, ensure_ascii=False),
                },
            )

        equity = payload.equity_curve.copy()
        if "equity" not in equity.columns and "portfolio_equity" in equity.columns:
            equity["equity"] = equity["portfolio_equity"]
        for index, row in equity.reset_index(drop=True).iterrows():
            conn.execute(
                text(
                    """
                    INSERT INTO qd_backtest_equity_points
                    (run_id, point_index, point_time, point_value)
                    VALUES (:run_id, :point_index, :point_time, :point_value)
                    """
                ),
                {
                    "run_id": run_id,
                    "point_index": int(index),
                    "point_time": str(row.get("date") or row.get("point_time") or ""),
                    "point_value": _float(row.get("equity") or row.get("point_value")),
                },
            )

        conn.execute(
            text(
                """
                INSERT INTO qd_ml_model_metrics
                (user_id, model_name, run_id, metric_date, metrics_json, feature_importance_json)
                VALUES (:user_id, :model_name, :run_id, :metric_date, :metrics_json, :feature_importance_json)
                """
            ),
            {
                "user_id": payload.user_id,
                "model_name": payload.strategy_name,
                "run_id": run_id,
                "metric_date": payload.end_date,
                "metrics_json": result_json,
                "feature_importance_json": json.dumps(payload.config.get("feature_importance", {}), default=_json_default),
            },
        )
        conn.execute(
            text(
                """
                INSERT INTO qd_ml_backtests
                (user_id, qd_run_id, model_name, market, symbol, timeframe, start_date, end_date, metrics_json, config_json)
                VALUES
                (:user_id, :qd_run_id, :model_name, :market, :symbol, :timeframe, :start_date, :end_date, :metrics_json, :config_json)
                """
            ),
            {
                "user_id": payload.user_id,
                "qd_run_id": run_id,
                "model_name": payload.strategy_name,
                "market": payload.market,
                "symbol": payload.symbol,
                "timeframe": payload.timeframe,
                "start_date": payload.start_date,
                "end_date": payload.end_date,
                "metrics_json": result_json,
                "config_json": config_json,
            },
        )
    logger.info(
        "quantdinger_backtest_persisted",
        extra={"run_id": run_id, "strategy_name": payload.strategy_name, "elapsed_ms": round((time.perf_counter() - started) * 1000, 2)},
    )
    return int(run_id)


def write_ml_signals(
    engine: Engine,
    signals: pd.DataFrame,
    model_name: str,
    market: str = "USStock",
    user_id: int = 1,
) -> int:
    """Persist ML signals for display in QuantDinger web/mobile clients."""

    started = time.perf_counter()
    ensure_ml_tables(engine)
    required = {"symbol", "signal"}
    if signals.empty or not required.issubset(signals.columns):
        return 0
    rows = 0
    with engine.begin() as conn:
        for _, row in signals.iterrows():
            conn.execute(
                text(
                    """
                    INSERT INTO qd_ml_signals
                    (user_id, model_name, market, symbol, signal_date, signal, confidence,
                     target_price, stop_loss, horizon, payload_json)
                    VALUES
                    (:user_id, :model_name, :market, :symbol, :signal_date, :signal, :confidence,
                     :target_price, :stop_loss, :horizon, :payload_json)
                    """
                ),
                {
                    "user_id": user_id,
                    "model_name": model_name,
                    "market": market,
                    "symbol": str(row.get("symbol") or row.get("ticker") or ""),
                    "signal_date": str(row.get("date") or row.get("signal_date") or datetime.utcnow().date().isoformat()),
                    "signal": _float(row.get("signal")),
                    "confidence": _float(row.get("confidence")),
                    "target_price": _float(row.get("target_price")),
                    "stop_loss": _float(row.get("stop_loss")),
                    "horizon": str(row.get("horizon") or ""),
                    "payload_json": json.dumps(row.to_dict(), default=_json_default, ensure_ascii=False),
                },
            )
            rows += 1
    logger.info(
        "quantdinger_ml_signals_persisted",
        extra={"model_name": model_name, "market": market, "rows": rows, "elapsed_ms": round((time.perf_counter() - started) * 1000, 2)},
    )
    return rows


def fetch_ml_signals(
    engine: Engine,
    symbols: Iterable[str] | None = None,
    model_name: str | None = None,
    limit: int = 200,
) -> pd.DataFrame:
    """Fetch persisted ML signals from the bridge-owned signal table."""

    ensure_ml_tables(engine)
    where: list[str] = []
    params: dict[str, Any] = {"limit": int(limit)}
    symbol_list = [s for s in symbols or [] if s]
    if symbol_list:
        where.append("symbol IN :symbols")
        params["symbols"] = tuple(symbol_list)
    if model_name:
        where.append("model_name = :model_name")
        params["model_name"] = model_name
    clause = f"WHERE {' AND '.join(where)}" if where else ""
    query = text(f"SELECT * FROM qd_ml_signals {clause} ORDER BY created_at DESC LIMIT :limit")
    if symbol_list and bindparam is not None:
        query = query.bindparams(bindparam("symbols", expanding=True))
    return pd.read_sql_query(query, engine, params=params)


def _first_existing_table(engine: Engine, names: list[str]) -> str | None:
    for name in names:
        if _table_exists(engine, name):
            return name
    return None


def _table_exists(engine: Engine, table_name: str) -> bool:
    return bool(inspect(engine).has_table(_safe_identifier(table_name)))


def _table_columns(engine: Engine, table_name: str) -> set[str]:
    return {column["name"] for column in inspect(engine).get_columns(_safe_identifier(table_name))}


def _date_column(columns: set[str]) -> str | None:
    for column in ("date", "datetime", "timestamp", "time", "open_time", "created_at"):
        if column in columns:
            return column
    return None


def _read_user_table(engine: Engine, table_name: str, user_id: int | None) -> pd.DataFrame:
    table_name = _safe_identifier(table_name)
    columns = _table_columns(engine, table_name)
    params: dict[str, Any] = {}
    where = ""
    if user_id is not None and "user_id" in columns:
        where = "WHERE user_id = :user_id"
        params["user_id"] = int(user_id)
    return pd.read_sql_query(text(f"SELECT * FROM {_quote_identifier(engine, table_name)} {where}"), engine, params=params)


def _insert_returning_id(conn: Any, table_name: str, values: Mapping[str, Any]) -> int:
    table_name = _safe_identifier(table_name)
    columns = list(values)
    names = ", ".join(_quote_identifier(conn.engine, column) for column in columns)
    placeholders = ", ".join(f":{column}" for column in columns)
    quoted_table = _quote_identifier(conn.engine, table_name)
    if conn.engine.dialect.name == "postgresql":
        result = conn.execute(text(f"INSERT INTO {quoted_table} ({names}) VALUES ({placeholders}) RETURNING id"), dict(values))
        return int(result.scalar_one())
    result = conn.execute(text(f"INSERT INTO {quoted_table} ({names}) VALUES ({placeholders})"), dict(values))
    return int(result.lastrowid)


def smoke_test_db(config: QuantDingerBridgeConfig | None = None) -> dict[str, Any]:
    """Check DB connectivity and whether core QuantDinger tables are visible."""

    started = time.perf_counter()
    try:
        engine = create_engine_from_config(config)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        tables = {
            "qd_market_symbols": _table_exists(engine, "qd_market_symbols"),
            "qd_backtest_runs": _table_exists(engine, "qd_backtest_runs"),
            "qd_ml_signals": _table_exists(engine, "qd_ml_signals"),
        }
        return {"status": "ok", "tables": tables, "latency_ms": round((time.perf_counter() - started) * 1000, 2)}
    except Exception as exc:
        return {"status": "error", "error": str(exc), "latency_ms": round((time.perf_counter() - started) * 1000, 2)}


def _safe_identifier(value: str) -> str:
    if not _IDENTIFIER_RE.match(value):
        raise ValueError(f"Unsafe SQL identifier: {value!r}")
    return value


def _quote_identifier(engine: Engine, value: str) -> str:
    value = _safe_identifier(value)
    preparer = engine.dialect.identifier_preparer
    return preparer.quote(value)


def _float(value: Any) -> float:
    try:
        return float(value) if value is not None and value != "" else 0.0
    except (TypeError, ValueError):
        return 0.0


def _json_default(value: Any) -> str | float | int | None:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)
