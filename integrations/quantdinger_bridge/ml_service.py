"""FastAPI ML sidecar service for QuantDinger."""

from __future__ import annotations

from datetime import datetime
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Any, Mapping

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from pydantic import BaseModel, Field
import requests

from .api_adapter import QuantDingerAPIClient, QuantDingerAPIError
from .config import QuantDingerBridgeConfig, load_config
from .db_adapter import (
    BacktestPersistencePayload,
    create_engine_from_config,
    write_backtest_result,
    write_ml_signals,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="ML4T QuantDinger Bridge",
    version="0.2.0",
    description="FastAPI sidecar exposing machine-learning-for-trading models inside QuantDinger.",
)

_OBSERVABILITY: dict[str, float] = {"requests": 0.0, "errors": 0.0, "backtests": 0.0}
_BAR_CACHE: dict[tuple[str, str, str, str | None, str | None], tuple[float, pd.DataFrame]] = {}
_CACHE_LOCK = Lock()


class BacktestRequest(BaseModel):
    """Request body for ML backtests launched from QuantDinger."""

    universe: list[str] = Field(default_factory=list)
    market: str = "USStock"
    timeframe: str = "1D"
    start_date: str
    end_date: str
    initial_capital: float = 100_000.0
    model_name: str = "momentum_baseline"
    strategy_name: str | None = None
    transaction_cost_bps: float = 0.0
    persist: bool = True
    params: dict[str, Any] = Field(default_factory=dict)
    user_id: int = 1


class ModelInfo(BaseModel):
    """Model metadata shown to web and mobile clients."""

    name: str
    description: str
    supports: list[str]


class ModelsResponse(BaseModel):
    """Response for model discovery."""

    models: list[ModelInfo]


class FeatureRow(BaseModel):
    """Feature coverage summary for one symbol."""

    symbol: str
    rows: int
    start: str | None = None
    end: str | None = None
    columns: list[str] = Field(default_factory=list)


class FeatureSummaryResponse(BaseModel):
    """Response for feature coverage inspection."""

    features: list[FeatureRow]


class BacktestResponse(BaseModel):
    """Serializable response for an ML backtest run."""

    run_id: int | None = None
    metrics: dict[str, float]
    equity_curve: list[dict[str, Any]]
    trades: list[dict[str, Any]]
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class SignalResponse(BaseModel):
    """Serializable response for current ML signals."""

    model_name: str
    signals: list[dict[str, Any]]
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """Structured error response shape for documented failure modes."""

    detail: str
    code: str = "error"


@app.middleware("http")
async def _log_requests(request: Request, call_next: Any) -> Any:
    started = time.perf_counter()
    _OBSERVABILITY["requests"] += 1
    try:
        response = await call_next(request)
        return response
    except Exception:
        _OBSERVABILITY["errors"] += 1
        logger.exception("ml_service_request_failed", extra={"path": request.url.path})
        raise
    finally:
        logger.info(
            "ml_service_request",
            extra={"path": request.url.path, "elapsed_ms": round((time.perf_counter() - started) * 1000, 2)},
        )


def _config() -> QuantDingerBridgeConfig:
    config = load_config()
    logging.basicConfig(level=getattr(logging, config.service.log_level, logging.INFO), format="%(asctime)s %(levelname)s %(name)s %(message)s")
    return config


def _authorize(authorization: str | None = Header(default=None)) -> None:
    config = _config()
    expected = config.service.service_auth_token
    if expected:
        if not authorization or authorization.replace("Bearer ", "", 1) != expected:
            raise HTTPException(status_code=401, detail="Invalid ML service token")
        return
    if config.api.verify_auth:
        if not authorization:
            raise HTTPException(status_code=401, detail="Missing QuantDinger authorization header")
        try:
            response = requests.get(config.api.auth_verify_url, headers={"Authorization": authorization}, timeout=config.api.timeout_seconds)
            response.raise_for_status()
            data = response.json()
            if data.get("code") not in {1, 200} and not data.get("success"):
                raise HTTPException(status_code=401, detail="QuantDinger auth verification failed")
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=401, detail=f"QuantDinger auth verification failed: {exc}") from exc


@app.get("/health")
def health() -> dict[str, Any]:
    """Return service health and configuration visibility."""

    config = _config()
    return {
        "status": "ok",
        "service": "ml4t-quantdinger-bridge",
        "quantdinger_api_base_url": config.api.base_url,
        "default_market": config.service.default_market,
        "api_prefix": config.service.api_prefix,
    }


@app.get("/api/v1/ml/models", response_model=ModelsResponse, responses={401: {"model": ErrorResponse}})
@app.get("/ml/models", response_model=ModelsResponse, include_in_schema=False)
def get_models(_: None = Depends(_authorize)) -> ModelsResponse:
    """List available ML models and built-in fallback models."""

    _activate_repo_imports(_config())
    models = [
        {
            "name": "momentum_baseline",
            "description": "Bridge fallback: trailing return signal from QuantDinger OHLCV bars.",
            "supports": ["signals", "backtest"],
        }
    ]
    try:
        import ml_stock_lab  # noqa: F401

        models.append(
            {
                "name": "ml_stock_lab",
                "description": "Host project package detected; wire concrete predictors through params.model_name.",
                "supports": ["features", "signals", "metrics"],
            }
        )
    except Exception:
        pass
    return ModelsResponse(models=[ModelInfo(**model) for model in models])


@app.get("/api/v1/ml/signals", response_model=SignalResponse, responses={401: {"model": ErrorResponse}, 502: {"model": ErrorResponse}})
@app.get("/ml/signals", response_model=SignalResponse, include_in_schema=False)
def get_ml_signals(
    universe: str = Query(..., description="Comma-separated symbols."),
    date: str | None = None,
    market: str | None = None,
    timeframe: str | None = None,
    model_name: str = "momentum_baseline",
    persist: bool = True,
    _: None = Depends(_authorize),
) -> SignalResponse:
    """Generate latest ML signals for a universe using QuantDinger data."""

    config = _config()
    symbols = _parse_universe(universe)
    if not symbols:
        raise HTTPException(status_code=422, detail="universe must contain at least one symbol")
    try:
        bars = _load_universe_bars(symbols, market or config.service.default_market, timeframe or config.service.default_timeframe, None, date)
    except QuantDingerAPIError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    signal_frame = _momentum_signals(bars, model_name=model_name, as_of=date)
    if persist and not signal_frame.empty:
        try:
            engine = create_engine_from_config(config)
            write_ml_signals(engine, signal_frame.rename(columns={"ticker": "symbol"}), model_name=model_name, market=market or config.service.default_market)
        except Exception as exc:
            return SignalResponse(
                model_name=model_name,
                signals=_records(signal_frame),
                diagnostics={"persist_status": "failed", "persist_error": str(exc)},
            )
    return SignalResponse(model_name=model_name, signals=_records(signal_frame), diagnostics={"symbols": symbols})


@app.get("/api/v1/ml/features", response_model=FeatureSummaryResponse, responses={401: {"model": ErrorResponse}, 502: {"model": ErrorResponse}})
@app.get("/ml/features", response_model=FeatureSummaryResponse, include_in_schema=False)
def get_feature_summary(
    universe: str,
    market: str | None = None,
    timeframe: str | None = None,
    _: None = Depends(_authorize),
) -> FeatureSummaryResponse:
    """Inspect simple feature coverage for QuantDinger-fed ML datasets."""

    config = _config()
    symbols = _parse_universe(universe)
    if not symbols:
        raise HTTPException(status_code=422, detail="universe must contain at least one symbol")
    try:
        bars = _load_universe_bars(symbols, market or config.service.default_market, timeframe or config.service.default_timeframe, None, None)
    except QuantDingerAPIError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    rows = []
    for symbol, frame in bars.items():
        rows.append(
            {
                "symbol": symbol,
                "rows": int(len(frame)),
                "start": str(frame["date"].min()) if "date" in frame and not frame.empty else None,
                "end": str(frame["date"].max()) if "date" in frame and not frame.empty else None,
                "columns": list(frame.columns),
            }
        )
    return FeatureSummaryResponse(features=[FeatureRow(**row) for row in rows])


@app.post("/api/v1/ml/backtest", response_model=BacktestResponse, responses={401: {"model": ErrorResponse}, 422: {"model": ErrorResponse}, 502: {"model": ErrorResponse}})
@app.post("/ml/backtest", response_model=BacktestResponse, include_in_schema=False)
def run_ml_backtest(request: BacktestRequest, _: None = Depends(_authorize)) -> BacktestResponse:
    """Run a simple ML backtest and optionally persist it into QuantDinger."""

    return execute_backtest_request(request)


def execute_backtest_request(request: BacktestRequest) -> BacktestResponse:
    """Execute an ML backtest request without FastAPI dependency injection."""

    config = _config()
    if not request.universe:
        raise HTTPException(status_code=422, detail="universe must contain at least one symbol")
    market = request.market or config.service.default_market
    timeframe = request.timeframe or config.service.default_timeframe
    try:
        bars = _load_universe_bars(request.universe, market, timeframe, request.start_date, request.end_date)
    except QuantDingerAPIError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    result = _run_momentum_backtest(
        bars=bars,
        start_date=request.start_date,
        end_date=request.end_date,
        initial_capital=request.initial_capital,
        transaction_cost_bps=request.transaction_cost_bps,
        model_name=request.model_name,
    )
    run_id: int | None = None
    if request.persist:
        try:
            engine = create_engine_from_config(config)
            symbol = ",".join(request.universe)
            run_id = write_backtest_result(
                engine,
                BacktestPersistencePayload(
                    strategy_name=request.strategy_name or request.model_name,
                    market=market,
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=request.start_date,
                    end_date=request.end_date,
                    initial_capital=request.initial_capital,
                    metrics=result["metrics"],
                    equity_curve=result["equity_curve"],
                    trades=result["trades"],
                    config={"request": _model_dump(request), "feature_importance": result.get("feature_importance", {})},
                    user_id=request.user_id,
                ),
            )
        except Exception as exc:
            logger.warning("ml_backtest_persist_failed", extra={"error": str(exc), "model_name": request.model_name})
            result["diagnostics"]["persist_status"] = "failed"
            result["diagnostics"]["persist_error"] = str(exc)

    _OBSERVABILITY["backtests"] += 1
    return BacktestResponse(
        run_id=run_id,
        metrics=result["metrics"],
        equity_curve=_records(result["equity_curve"]),
        trades=_records(result["trades"]),
        diagnostics=result["diagnostics"],
    )


@app.get("/api/v1/ml/metrics")
def get_service_metrics(_: None = Depends(_authorize)) -> dict[str, float]:
    """Return minimal in-process counters for service observability."""

    return dict(_OBSERVABILITY)


def run_ml_service() -> None:
    """Run the FastAPI service with Uvicorn."""

    import uvicorn

    config = _config()
    uvicorn.run("integrations.quantdinger_bridge.ml_service:app", host=config.service.host, port=config.service.port, reload=False)


def _activate_repo_imports(config: QuantDingerBridgeConfig) -> None:
    roots = [
        config.service.repo_root,
        config.service.repo_root / "research_platform_definitive" / "src",
        config.service.repo_root / "research_platform_definitive" / "portfolio_analysis" / "src",
    ]
    for root in reversed(roots):
        value = str(root)
        if root.exists() and value not in sys.path:
            sys.path.insert(0, value)


def _load_universe_bars(
    symbols: list[str],
    market: str,
    timeframe: str,
    start_date: str | None,
    end_date: str | None,
) -> dict[str, pd.DataFrame]:
    client = QuantDingerAPIClient(_config().api)
    frames: dict[str, pd.DataFrame] = {}
    if not symbols:
        return frames
    workers = min(max(len(symbols), 1), 8)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ml-bars") as executor:
        futures = {
            executor.submit(_load_symbol_bars, client, symbol, market, timeframe, start_date, end_date): symbol
            for symbol in symbols
        }
        for future in as_completed(futures):
            symbol = futures[future]
            frames[symbol] = future.result()
    return frames


def _load_symbol_bars(
    client: QuantDingerAPIClient,
    symbol: str,
    market: str,
    timeframe: str,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    config = _config()
    key = (symbol, market, timeframe, start_date, end_date)
    now = time.monotonic()
    if config.service.cache_ttl_seconds > 0:
        with _CACHE_LOCK:
            cached = _BAR_CACHE.get(key)
            if cached and now - cached[0] <= config.service.cache_ttl_seconds:
                return cached[1].copy()
    frame = client.get_market_data(symbol=symbol, start=start_date, end=end_date, timeframe=timeframe, market=market, limit=2000)
    if config.service.cache_ttl_seconds > 0:
        with _CACHE_LOCK:
            _BAR_CACHE[key] = (now, frame.copy())
    return frame


def _momentum_signals(bars: Mapping[str, pd.DataFrame], model_name: str, as_of: str | None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    signal_date = as_of or datetime.utcnow().date().isoformat()
    for symbol, frame in bars.items():
        data = frame.dropna(subset=["close"]).sort_values("date") if "close" in frame.columns else pd.DataFrame()
        if data.empty:
            rows.append({"date": signal_date, "ticker": symbol, "symbol": symbol, "signal": 0.0, "confidence": 0.0, "model_name": model_name})
            continue
        lookback = min(63, len(data) - 1)
        if lookback <= 0:
            signal = 0.0
        else:
            signal = float(data["close"].iloc[-1] / data["close"].iloc[-lookback - 1] - 1.0)
        close = float(data["close"].iloc[-1])
        confidence = float(min(abs(signal) * 5.0, 1.0))
        direction = 1.0 if signal > 0 else -1.0 if signal < 0 else 0.0
        rows.append(
            {
                "date": str(data["date"].iloc[-1]) if "date" in data else signal_date,
                "ticker": symbol,
                "symbol": symbol,
                "signal": direction,
                "raw_score": signal,
                "confidence": confidence,
                "target_price": close * (1.0 + max(signal, 0.01)),
                "stop_loss": close * 0.92,
                "horizon": "1M",
                "model_name": model_name,
            }
        )
    return pd.DataFrame(rows)


def _run_momentum_backtest(
    bars: Mapping[str, pd.DataFrame],
    start_date: str,
    end_date: str,
    initial_capital: float,
    transaction_cost_bps: float,
    model_name: str,
) -> dict[str, Any]:
    prices = []
    for symbol, frame in bars.items():
        if frame.empty or "close" not in frame.columns:
            continue
        series = frame[["date", "close"]].copy()
        series["date"] = pd.to_datetime(series["date"], errors="coerce")
        series = series.dropna(subset=["date"]).set_index("date").sort_index()
        prices.append(series["close"].rename(symbol))
    price_panel = pd.concat(prices, axis=1).ffill().dropna(how="all") if prices else pd.DataFrame()
    if price_panel.empty:
        return {"metrics": {}, "equity_curve": pd.DataFrame(), "trades": pd.DataFrame(), "diagnostics": {"engine_status": "no_data"}}

    returns = price_panel.pct_change(fill_method=None).fillna(0.0)
    momentum = price_panel.pct_change(63, fill_method=None).shift(1)
    raw_weights = momentum.clip(lower=0.0)
    gross = raw_weights.sum(axis=1).replace(0.0, np.nan)
    weights = raw_weights.div(gross, axis=0).fillna(0.0)
    portfolio_returns = weights.shift(1).fillna(0.0).mul(returns, axis=0).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1))
    costs = turnover * (transaction_cost_bps / 10_000.0)
    net_returns = portfolio_returns - costs
    equity = initial_capital * (1.0 + net_returns).cumprod()
    metrics = _metrics(net_returns, equity, turnover)
    equity_curve = pd.DataFrame({"date": net_returns.index, "return": net_returns.values, "equity": equity.values})
    trades = weights.diff().fillna(weights).stack().reset_index()
    trades.columns = ["date", "ticker", "delta_weight"]
    trades = trades[trades["delta_weight"].abs() > 1e-12].reset_index(drop=True)
    trades["trade_type"] = "rebalance"
    trades["reason"] = model_name
    return {
        "metrics": metrics,
        "equity_curve": equity_curve,
        "trades": trades,
        "feature_importance": {"momentum_63d": 1.0},
        "diagnostics": {"engine_status": "ok", "symbols": list(bars), "start_date": start_date, "end_date": end_date},
    }


def _metrics(returns: pd.Series, equity: pd.Series, turnover: pd.Series) -> dict[str, float]:
    clean = pd.to_numeric(returns, errors="coerce").dropna()
    if clean.empty:
        return {}
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0) if len(equity) > 1 else 0.0
    vol = float(clean.std(ddof=0) * np.sqrt(252))
    sharpe = float((clean.mean() / clean.std(ddof=0)) * np.sqrt(252)) if clean.std(ddof=0) else 0.0
    drawdown = float((equity / equity.cummax() - 1.0).min())
    return {
        "total_return": total_return,
        "annualized_return": float((1 + total_return) ** (252 / max(len(clean), 1)) - 1),
        "annualized_volatility": vol,
        "sharpe": sharpe,
        "max_drawdown": drawdown,
        "avg_turnover": float(turnover.mean()),
        "observations": float(len(clean)),
    }


def _parse_universe(value: str | list[str]) -> list[str]:
    if isinstance(value, str):
        return [item.strip().upper() for item in value.split(",") if item.strip()]
    return [str(item).strip().upper() for item in value if str(item).strip()]


def _model_dump(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    clean = frame.replace([np.inf, -np.inf], np.nan).where(pd.notnull(frame), None)
    records = clean.to_dict(orient="records")
    return [{key: _jsonable(value) for key, value in row.items()} for row in records]


def _jsonable(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


if __name__ == "__main__":
    run_ml_service()
