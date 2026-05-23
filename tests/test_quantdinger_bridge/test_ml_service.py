from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from integrations.quantdinger_bridge import ml_service


def test_models_endpoint() -> None:
    client = TestClient(ml_service.app)
    response = client.get("/api/v1/ml/models")

    assert response.status_code == 200
    assert response.json()["models"][0]["name"] == "momentum_baseline"


def test_strategy_registry_endpoint() -> None:
    client = TestClient(ml_service.app)
    response = client.get("/api/v1/ml/strategies")

    assert response.status_code == 200
    names = {row["strategy_name"] for row in response.json()["strategies"]}
    assert "risk_balanced_blend" in names


def test_snapshot_endpoints_are_dashboard_safe() -> None:
    client = TestClient(ml_service.app)

    signals = client.get("/api/v1/ml/signals/snapshot", params={"limit": 3})
    backtests = client.get("/api/v1/ml/backtests/summary", params={"limit": 3})

    assert signals.status_code == 200
    assert "signals" in signals.json()
    assert backtests.status_code == 200
    assert "backtests" in backtests.json()


def test_signals_endpoint_with_mocked_bars(monkeypatch) -> None:
    def fake_load(symbols, market, timeframe, start_date, end_date):
        dates = pd.date_range("2024-01-01", periods=80, freq="D")
        return {symbol: pd.DataFrame({"date": dates, "close": range(100, 180)}) for symbol in symbols}

    monkeypatch.setattr(ml_service, "_load_universe_bars", fake_load)
    client = TestClient(ml_service.app)
    response = client.get("/api/v1/ml/signals", params={"universe": "AAPL", "persist": False})

    assert response.status_code == 200
    assert response.json()["signals"][0]["symbol"] == "AAPL"


def test_backtest_endpoint_with_mocked_bars(monkeypatch) -> None:
    def fake_load(symbols, market, timeframe, start_date, end_date):
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        return {symbol: pd.DataFrame({"date": dates, "close": range(100, 200)}) for symbol in symbols}

    monkeypatch.setattr(ml_service, "_load_universe_bars", fake_load)
    client = TestClient(ml_service.app)
    response = client.post(
        "/api/v1/ml/backtest",
        json={
            "universe": ["AAPL"],
            "start_date": "2024-01-01",
            "end_date": "2024-04-01",
            "persist": False,
        },
    )

    assert response.status_code == 200
    assert "metrics" in response.json()
