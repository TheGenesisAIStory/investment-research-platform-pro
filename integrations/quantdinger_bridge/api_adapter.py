"""REST API adapter for QuantDinger services."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import time
from typing import Any, Mapping

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import ApiSettings, QuantDingerBridgeConfig, load_config

logger = logging.getLogger(__name__)


class QuantDingerAPIError(RuntimeError):
    """Base exception for QuantDinger API adapter failures."""


class QuantDingerHTTPError(QuantDingerAPIError):
    """Raised when the QuantDinger API returns an HTTP error."""

    def __init__(self, message: str, status_code: int | None = None, payload: Any = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload


class QuantDingerBusinessError(QuantDingerAPIError):
    """Raised when QuantDinger returns a business-level error envelope."""

    def __init__(self, message: str, payload: Any = None) -> None:
        super().__init__(message)
        self.payload = payload


@dataclass(slots=True)
class Position:
    """Normalized QuantDinger position object."""

    symbol: str
    side: str
    quantity: float
    entry_price: float | None = None
    payload: Mapping[str, Any] | None = None


@dataclass(slots=True)
class Order:
    """Normalized QuantDinger order object."""

    symbol: str
    side: str
    order_type: str
    status: str
    quantity: float | None = None
    price: float | None = None
    payload: Mapping[str, Any] | None = None


class QuantDingerAPIClient:
    """Thin client for QuantDinger's Flask API and Agent Gateway endpoints."""

    def __init__(self, settings: ApiSettings | None = None) -> None:
        self.settings = settings or load_config().api
        self.session = requests.Session()
        retry = Retry(
            total=max(int(self.settings.retry_total), 0),
            connect=max(int(self.settings.retry_total), 0),
            read=max(int(self.settings.retry_total), 0),
            status=max(int(self.settings.retry_total), 0),
            backoff_factor=max(float(self.settings.retry_backoff_seconds), 0.0),
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET", "HEAD", "OPTIONS"}),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def get_market_data(
        self,
        symbol: str,
        start: str | None = None,
        end: str | None = None,
        timeframe: str = "1D",
        market: str = "USStock",
        limit: int = 1000,
        use_agent_api: bool = False,
    ) -> pd.DataFrame:
        """Fetch OHLCV bars via QuantDinger REST and return a DataFrame."""

        if use_agent_api:
            data = self._request(
                "GET",
                "/agent/v1/klines",
                params={"market": market, "symbol": symbol, "timeframe": timeframe, "limit": limit},
                agent=True,
            )
            rows = data.get("klines", []) if isinstance(data, Mapping) else []
        else:
            rows = self._request(
                "GET",
                "/indicator/kline",
                params={"market": market, "symbol": symbol, "timeframe": timeframe, "limit": limit},
            )
        frame = _normalize_ohlcv(rows, symbol=symbol, market=market, timeframe=timeframe)
        if start and "date" in frame.columns:
            frame = frame[frame["date"] >= pd.Timestamp(start)]
        if end and "date" in frame.columns:
            frame = frame[frame["date"] <= pd.Timestamp(end)]
        return frame.reset_index(drop=True)

    def get_latest_price(self, symbol: str, market: str = "USStock") -> Mapping[str, Any]:
        """Fetch latest price data for a symbol."""

        return self._request("GET", "/indicator/price", params={"market": market, "symbol": symbol})

    def get_positions(self, strategy_id: int | None = None, use_agent_api: bool = False) -> list[Position]:
        """Fetch normalized positions from QuantDinger strategy or Agent Gateway endpoints."""

        if use_agent_api:
            data = self._request("GET", "/agent/v1/portfolio/positions", agent=True)
        else:
            if strategy_id is None:
                data = self._request("GET", "/portfolio/positions")
            else:
                data = self._request("GET", "/strategies/positions", params={"id": strategy_id})
        rows = data.get("positions", data) if isinstance(data, Mapping) else data
        return [_to_position(row) for row in _ensure_list(rows)]

    def get_orders(self, limit: int = 100, use_agent_api: bool = False) -> list[Order]:
        """Fetch pending or paper orders from QuantDinger endpoints."""

        if use_agent_api:
            data = self._request("GET", "/agent/v1/portfolio/paper-orders", agent=True)
        else:
            data = self._request("GET", "/dashboard/pendingOrders", params={"limit": limit})
        rows = data.get("items", data.get("orders", data)) if isinstance(data, Mapping) else data
        return [_to_order(row) for row in _ensure_list(rows)]

    def search_instruments(self, keyword: str, market: str = "USStock", limit: int = 20) -> pd.DataFrame:
        """Search QuantDinger instrument metadata."""

        data = self._request("GET", "/market/symbols/search", params={"keyword": keyword, "market": market, "limit": limit})
        return pd.DataFrame(_ensure_list(data))

    def _request(
        self,
        method: str,
        path: str,
        params: Mapping[str, Any] | None = None,
        json_body: Mapping[str, Any] | None = None,
        agent: bool = False,
    ) -> Any:
        url = f"{self.settings.base_url.rstrip('/')}/{path.lstrip('/')}"
        headers: dict[str, str] = {}
        token = self.settings.agent_token if agent else self.settings.token
        if token:
            headers["Authorization"] = f"Bearer {token}"
        started = time.perf_counter()
        try:
            response = self.session.request(
                method=method,
                url=url,
                params=dict(params or {}),
                json=dict(json_body or {}) if json_body is not None else None,
                headers=headers,
                timeout=self.settings.timeout_seconds,
            )
        except requests.RequestException as exc:
            logger.warning("quantdinger_api_request_failed", extra={"url": url, "method": method, "error": str(exc)})
            raise QuantDingerAPIError(f"QuantDinger API request failed: {exc}") from exc

        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        logger.info(
            "quantdinger_api_request",
            extra={"url": url, "method": method, "status_code": response.status_code, "elapsed_ms": elapsed_ms},
        )
        payload: Any = None
        try:
            payload = response.json()
        except ValueError:
            payload = response.text
        if response.status_code >= 400:
            raise QuantDingerHTTPError(
                f"QuantDinger API returned HTTP {response.status_code} for {path}",
                status_code=response.status_code,
                payload=payload,
            )
        return _unwrap_response(payload)

    def smoke_test(self) -> dict[str, Any]:
        """Run a minimal REST connectivity check against QuantDinger."""

        started = time.perf_counter()
        try:
            data = self._request("GET", "/health")
            return {"status": "ok", "latency_ms": round((time.perf_counter() - started) * 1000, 2), "data": data}
        except Exception as exc:
            return {"status": "error", "error": str(exc), "latency_ms": round((time.perf_counter() - started) * 1000, 2)}


def get_market_data(
    symbol: str,
    start: str | None = None,
    end: str | None = None,
    timeframe: str = "1D",
    market: str = "USStock",
    config: QuantDingerBridgeConfig | None = None,
) -> pd.DataFrame:
    """Convenience wrapper around ``QuantDingerAPIClient.get_market_data``."""

    return QuantDingerAPIClient((config or load_config()).api).get_market_data(symbol, start, end, timeframe, market)


def get_positions(config: QuantDingerBridgeConfig | None = None) -> list[Position]:
    """Return positions via the configured QuantDinger REST API."""

    return QuantDingerAPIClient((config or load_config()).api).get_positions()


def get_orders(config: QuantDingerBridgeConfig | None = None) -> list[Order]:
    """Return orders via the configured QuantDinger REST API."""

    return QuantDingerAPIClient((config or load_config()).api).get_orders()


def smoke_test_api(config: QuantDingerBridgeConfig | None = None) -> dict[str, Any]:
    """Check QuantDinger REST API connectivity without requiring Docker exec."""

    return QuantDingerAPIClient((config or load_config()).api).smoke_test()


def _unwrap_response(payload: Any) -> Any:
    if not isinstance(payload, Mapping):
        return payload
    if payload.get("code") in {0, -1, 400, 401, 403, 500} and not payload.get("success"):
        raise QuantDingerBusinessError(str(payload.get("msg") or payload.get("message") or "QuantDinger API error"), payload=payload)
    if "data" in payload:
        return payload["data"]
    return payload


def _normalize_ohlcv(rows: Any, symbol: str, market: str, timeframe: str) -> pd.DataFrame:
    frame = pd.DataFrame(_ensure_list(rows))
    if frame.empty:
        return pd.DataFrame(columns=["date", "symbol", "market", "timeframe", "open", "high", "low", "close", "volume"])
    rename = {
        "t": "date",
        "time": "date",
        "timestamp": "date",
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume",
    }
    frame = frame.rename(columns={key: value for key, value in rename.items() if key in frame.columns})
    if "date" not in frame.columns:
        frame["date"] = pd.RangeIndex(len(frame))
    frame["date"] = _coerce_dates(frame["date"])
    frame["symbol"] = frame.get("symbol", symbol)
    frame["market"] = frame.get("market", market)
    frame["timeframe"] = frame.get("timeframe", timeframe)
    for column in ("open", "high", "low", "close", "volume"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.sort_values("date").reset_index(drop=True)


def _coerce_dates(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().any() and numeric.dropna().median() > 10_000_000_000:
        return pd.to_datetime(numeric, unit="ms", errors="coerce")
    if numeric.notna().any() and numeric.dropna().median() > 1_000_000_000:
        return pd.to_datetime(numeric, unit="s", errors="coerce")
    return pd.to_datetime(values, errors="coerce")


def _ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, Mapping):
        for key in ("items", "rows", "list", "records", "positions", "orders", "klines"):
            nested = value.get(key)
            if isinstance(nested, list):
                return nested
        return [value]
    return list(value) if isinstance(value, tuple) else []


def _to_position(row: Any) -> Position:
    data = dict(row or {})
    return Position(
        symbol=str(data.get("symbol", "")),
        side=str(data.get("side", data.get("direction", ""))),
        quantity=_float(data.get("quantity", data.get("amount"))),
        entry_price=_optional_float(data.get("entry_price")),
        payload=data,
    )


def _to_order(row: Any) -> Order:
    data = dict(row or {})
    return Order(
        symbol=str(data.get("symbol", "")),
        side=str(data.get("side", "")),
        order_type=str(data.get("order_type", data.get("type", ""))),
        status=str(data.get("status", "")),
        quantity=_optional_float(data.get("quantity", data.get("amount"))),
        price=_optional_float(data.get("price")),
        payload=data,
    )


def _float(value: Any) -> float:
    return _optional_float(value) or 0.0


def _optional_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None and value != "" else None
    except (TypeError, ValueError):
        return None
