"""Convenience entrypoints for the QuantDinger integration bridge."""

from __future__ import annotations

import socket
import time
from typing import Any, Mapping
from urllib.parse import urlparse

import pandas as pd

from .api_adapter import QuantDingerAPIClient, get_market_data, get_orders, get_positions, smoke_test_api
from .config import QuantDingerBridgeConfig, load_config
from .db_adapter import create_engine_from_config, fetch_ml_signals, smoke_test_db


def run_ml_service() -> None:
    """Run the FastAPI ML sidecar service."""

    from .ml_service import run_ml_service as _run_ml_service

    _run_ml_service()


def get_ml_signals_for_universe(
    universe: list[str] | str,
    date: str | None = None,
    model_name: str = "momentum_baseline",
    market: str = "USStock",
) -> pd.DataFrame:
    """Return ML signals by calling the local service logic through HTTP APIs."""

    client = QuantDingerAPIClient(load_config().api)
    symbols = _parse_universe(universe)
    frames = {
        symbol: client.get_market_data(symbol=symbol, start=None, end=date, timeframe="1D", market=market)
        for symbol in symbols
    }
    from .ml_service import _momentum_signals

    return _momentum_signals(frames, model_name=model_name, as_of=date)


def run_ml_backtest_through_quantdinger(config: Mapping[str, Any]) -> dict[str, Any]:
    """Run a QuantDinger-fed ML backtest using the FastAPI service function."""

    from .ml_service import BacktestRequest, execute_backtest_request

    response = execute_backtest_request(BacktestRequest(**dict(config)))
    if hasattr(response, "model_dump"):
        return response.model_dump()
    return response.dict()


def get_persisted_ml_signals(symbols: list[str] | None = None, model_name: str | None = None, limit: int = 200) -> pd.DataFrame:
    """Fetch ML signals persisted into QuantDinger's database."""

    engine = create_engine_from_config(load_config())
    return fetch_ml_signals(engine, symbols=symbols, model_name=model_name, limit=limit)


def smoke_test_redis(config: QuantDingerBridgeConfig | None = None) -> dict[str, Any]:
    """Check Redis TCP connectivity without requiring the redis Python package."""

    cfg = config or load_config()
    parsed = urlparse(cfg.redis.url)
    host = parsed.hostname or cfg.redis.host
    port = parsed.port or cfg.redis.port
    started = time.perf_counter()
    try:
        with socket.create_connection((host, port), timeout=3):
            pass
        return {"status": "ok", "host": host, "port": port, "latency_ms": round((time.perf_counter() - started) * 1000, 2)}
    except OSError as exc:
        return {"status": "error", "host": host, "port": port, "error": str(exc), "latency_ms": round((time.perf_counter() - started) * 1000, 2)}


def smoke_test_bridge(config: QuantDingerBridgeConfig | None = None) -> dict[str, Any]:
    """Run DB, REST API and Redis smoke checks for the full bridge."""

    cfg = config or load_config()
    return {
        "db": smoke_test_db(cfg),
        "api": smoke_test_api(cfg),
        "redis": smoke_test_redis(cfg),
    }


def _parse_universe(universe: list[str] | str) -> list[str]:
    if isinstance(universe, str):
        return [part.strip().upper() for part in universe.split(",") if part.strip()]
    return [str(part).strip().upper() for part in universe if str(part).strip()]


__all__ = [
    "QuantDingerAPIClient",
    "QuantDingerBridgeConfig",
    "get_market_data",
    "get_ml_signals_for_universe",
    "get_orders",
    "get_persisted_ml_signals",
    "get_positions",
    "load_config",
    "run_ml_backtest_through_quantdinger",
    "run_ml_service",
    "smoke_test_api",
    "smoke_test_bridge",
    "smoke_test_db",
    "smoke_test_redis",
]
