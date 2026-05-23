from __future__ import annotations

import pandas as pd
import pytest

sqlalchemy = pytest.importorskip("sqlalchemy")
from sqlalchemy import create_engine, text

from integrations.quantdinger_bridge.db_adapter import (
    BacktestPersistencePayload,
    ensure_ml_tables,
    fetch_instruments,
    fetch_ml_signals,
    write_backtest_result,
    write_ml_signals,
)


def _engine():
    return create_engine("sqlite:///:memory:", future=True)


def _create_core_tables(engine) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE qd_market_symbols (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    market TEXT,
                    symbol TEXT,
                    name TEXT,
                    exchange TEXT,
                    currency TEXT,
                    is_active INTEGER,
                    is_hot INTEGER,
                    sort_order INTEGER
                )
                """
            )
        )
        conn.execute(text("INSERT INTO qd_market_symbols (market, symbol, name, is_active, sort_order) VALUES ('USStock', 'AAPL', 'Apple', 1, 10)"))
        conn.execute(
            text(
                """
                CREATE TABLE qd_backtest_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    strategy_id INTEGER,
                    strategy_name TEXT,
                    run_type TEXT,
                    market TEXT,
                    symbol TEXT,
                    timeframe TEXT,
                    start_date TEXT,
                    end_date TEXT,
                    initial_capital REAL,
                    strategy_config TEXT,
                    config_snapshot TEXT,
                    engine_version TEXT,
                    status TEXT,
                    error_message TEXT,
                    result_json TEXT
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE qd_backtest_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER,
                    user_id INTEGER,
                    strategy_id INTEGER,
                    trade_index INTEGER,
                    trade_time TEXT,
                    trade_type TEXT,
                    side TEXT,
                    price REAL,
                    amount REAL,
                    profit REAL,
                    balance REAL,
                    reason TEXT,
                    payload_json TEXT
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE qd_backtest_equity_points (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER,
                    point_index INTEGER,
                    point_time TEXT,
                    point_value REAL
                )
                """
            )
        )


def test_ensure_ml_tables_and_signals_roundtrip() -> None:
    engine = _engine()
    ensure_ml_tables(engine)

    count = write_ml_signals(
        engine,
        pd.DataFrame([{"symbol": "AAPL", "signal": 1, "confidence": 0.8, "date": "2024-01-01"}]),
        model_name="demo",
    )
    rows = fetch_ml_signals(engine, symbols=["AAPL"], model_name="demo")

    assert count == 1
    assert rows.iloc[0]["symbol"] == "AAPL"


def test_fetch_instruments_and_write_backtest() -> None:
    engine = _engine()
    _create_core_tables(engine)

    instruments = fetch_instruments(engine, market="USStock")
    run_id = write_backtest_result(
        engine,
        BacktestPersistencePayload(
            strategy_name="demo",
            market="USStock",
            symbol="AAPL",
            timeframe="1D",
            start_date="2020-01-01",
            end_date="2020-01-03",
            initial_capital=100_000,
            metrics={"sharpe": 1.2},
            equity_curve=pd.DataFrame({"date": ["2020-01-01", "2020-01-02"], "equity": [100_000, 101_000]}),
            trades=pd.DataFrame([{"date": "2020-01-02", "ticker": "AAPL", "delta_weight": 1.0, "price": 100}]),
        ),
    )

    assert instruments.iloc[0]["symbol"] == "AAPL"
    assert run_id == 1

