"""Provider fallback, rate limiting and audit logging for data acquisition."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .cache_manager import DataCache
from .data_platform import utc_now


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    priority: int
    requests_per_minute: int = 60
    requests_per_day: int | None = None
    cost_tier: str = "free"
    enabled: bool = True
    max_retries: int = 2
    backoff_seconds: float = 1.5
    quality_score: float = 0.6
    coverage_score: float = 0.6
    cost_score: float = 0.8
    policy_stability_score: float = 0.6
    historical_start: str = "2000-01-01"
    intervals: tuple[str, ...] = ("1d",)
    regions: tuple[str, ...] = ("global",)
    asset_types: tuple[str, ...] = ("stock", "etf", "index")
    supports_bulk_daily: bool = False
    supports_intraday_5m: bool = False
    static_history: bool = False
    max_batch_size: int = 1
    requires_env: tuple[str, ...] = field(default_factory=tuple)
    notes: str = ""


DEFAULT_PROVIDERS = {
    "yfinance": ProviderSpec(
        "yfinance",
        priority=1,
        requests_per_minute=120,
        cost_tier="free",
        quality_score=0.68,
        coverage_score=0.78,
        cost_score=1.0,
        policy_stability_score=0.45,
        intervals=("1d",),
        regions=("global",),
        supports_bulk_daily=True,
        max_batch_size=100,
        notes="Unofficial Yahoo Finance wrapper; excellent bulk ergonomics, policy/rate-limit risk is higher.",
    ),
    "stooq": ProviderSpec(
        "stooq",
        priority=2,
        requests_per_minute=60,
        cost_tier="free",
        quality_score=0.76,
        coverage_score=0.72,
        cost_score=0.95,
        policy_stability_score=0.65,
        intervals=("1d",),
        regions=("US", "Europe", "JP", "global"),
        requires_env=("STOOQ_API_KEY",),
        notes="CSV historical data fallback. Direct downloads can require a Stooq API key.",
    ),
    "alpha_vantage": ProviderSpec(
        "alpha_vantage",
        priority=3,
        requests_per_minute=5,
        requests_per_day=25,
        cost_tier="free_limited",
        quality_score=0.82,
        coverage_score=0.70,
        cost_score=0.25,
        policy_stability_score=0.80,
        intervals=("1d", "5m"),
        regions=("US", "global"),
        supports_intraday_5m=True,
        requires_env=("ALPHAVANTAGE_API_KEY", "ALPHA_VANTAGE_API_KEY"),
        notes="Reliable documented API, but free tier is very small and should only be used for targeted fallback/intraday jobs.",
    ),
    "kaggle_seed": ProviderSpec(
        "kaggle_seed",
        priority=0,
        requests_per_minute=999999,
        cost_tier="static",
        quality_score=0.72,
        coverage_score=0.85,
        cost_score=1.0,
        policy_stability_score=0.95,
        intervals=("1d", "5m"),
        regions=("global",),
        asset_types=("stock", "etf", "index", "crypto"),
        static_history=True,
        notes="Local static seed datasets imported from Kaggle. Used to avoid historical API refetches; not called as a network provider.",
    ),
    "fmp": ProviderSpec("fmp", priority=4, requests_per_minute=300, cost_tier="limited", requires_env=("FMP_API_KEY",)),
    "polygon": ProviderSpec("polygon", priority=5, requests_per_minute=120, cost_tier="paid", requires_env=("POLYGON_API_KEY",)),
    "eodhd": ProviderSpec("eodhd", priority=6, requests_per_minute=60, cost_tier="paid", requires_env=("EODHD_API_KEY",), regions=("global", "Europe", "JP", "US")),
    "intrinio": ProviderSpec("intrinio", priority=7, requests_per_minute=60, cost_tier="paid", requires_env=("INTRINIO_API_KEY",), regions=("US", "global")),
}


@dataclass(frozen=True)
class ProviderRequest:
    """Normalized request context used by the provider policy engine."""

    provider_symbol: str
    interval: str = "1d"
    start: str | None = None
    end: str | None = None
    asset_type: str = "stock"
    country: str = ""
    exchange: str = ""
    bulk_size: int = 1
    allow_paid: bool = False
    allow_static: bool = False


@dataclass(frozen=True)
class ProviderDecision:
    provider: str
    score: float
    reason: str
    budget_remaining: int | None = None


class ProviderUsageTracker:
    """Persistent per-provider usage tracker for minute/day budgets."""

    def __init__(self, path: Path | str):
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.state = self._load()

    @staticmethod
    def _today() -> str:
        return pd.Timestamp.now(tz="UTC").date().isoformat()

    def _load(self) -> dict[str, Any]:
        if self.path.exists():
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
                if payload.get("date") == self._today():
                    return payload
            except Exception:
                pass
        return {"date": self._today(), "providers": {}}

    def _save(self) -> None:
        self.path.write_text(json.dumps(self.state, indent=2, default=str), encoding="utf-8")

    def _provider_state(self, name: str) -> dict[str, Any]:
        if self.state.get("date") != self._today():
            self.state = {"date": self._today(), "providers": {}}
        providers = self.state.setdefault("providers", {})
        return providers.setdefault(name, {"day_count": 0, "minute_times": []})

    @staticmethod
    def _env_available(spec: ProviderSpec) -> bool:
        return not spec.requires_env or any(os.environ.get(key) for key in spec.requires_env)

    def can_call(self, spec: ProviderSpec, units: int = 1) -> tuple[bool, str]:
        if not spec.enabled:
            return False, "provider_disabled"
        if not self._env_available(spec):
            return False, "missing_required_env"
        state = self._provider_state(spec.name)
        now = time.time()
        minute_times = [t for t in state.get("minute_times", []) if now - float(t) < 60]
        state["minute_times"] = minute_times
        if spec.requests_per_day is not None and int(state.get("day_count", 0)) + units > spec.requests_per_day:
            return False, "daily_budget_exhausted"
        if len(minute_times) + units > spec.requests_per_minute:
            return False, "minute_budget_exhausted"
        return True, "available"

    def record_call(self, provider_name: str, units: int = 1) -> None:
        state = self._provider_state(provider_name)
        now = time.time()
        state["minute_times"] = [t for t in state.get("minute_times", []) if now - float(t) < 60]
        state["minute_times"].extend([now] * units)
        state["day_count"] = int(state.get("day_count", 0)) + units
        self._save()

    def remaining_day_budget(self, spec: ProviderSpec) -> int | None:
        if spec.requests_per_day is None:
            return None
        state = self._provider_state(spec.name)
        return max(int(spec.requests_per_day) - int(state.get("day_count", 0)), 0)


class DataProviderRegistry:
    """Registry of provider profiles and their high-level capabilities."""

    def __init__(self, providers: dict[str, ProviderSpec] | None = None):
        self.providers = dict(providers or DEFAULT_PROVIDERS)

    def get(self, name: str) -> ProviderSpec:
        return self.providers[name]

    def names(self) -> list[str]:
        return list(self.providers)

    def update(self, name: str, **updates: Any) -> None:
        base = self.providers.get(name, ProviderSpec(name=name, priority=999))
        normalized = {}
        if "env_key" in updates and "requires_env" not in updates:
            updates["requires_env"] = (updates["env_key"],)
        if "env_keys" in updates and "requires_env" not in updates:
            updates["requires_env"] = tuple(updates["env_keys"])
        for key, value in updates.items():
            if not hasattr(base, key):
                continue
            if key in {"intervals", "regions", "asset_types", "requires_env"} and isinstance(value, list):
                value = tuple(value)
            normalized[key] = value
        self.providers[name] = replace(base, **normalized)

    @classmethod
    def from_config(cls, config: dict[str, Any] | None = None, base: dict[str, ProviderSpec] | None = None) -> "DataProviderRegistry":
        registry = cls(base)
        providers = ((config or {}).get("ohlcv", {}).get("providers", {}) if config else {}) or {}
        for name, values in providers.items():
            if isinstance(values, dict):
                registry.update(name, **values)
        return registry

    def capable(self, request: ProviderRequest, candidates: list[str] | None = None) -> list[ProviderSpec]:
        out = []
        names = candidates or self.names()
        for name in names:
            spec = self.providers.get(name)
            if not spec or not spec.enabled:
                continue
            if request.interval not in spec.intervals:
                continue
            if request.interval == "5m" and not spec.supports_intraday_5m:
                continue
            if request.bulk_size > 1 and not spec.supports_bulk_daily:
                continue
            if spec.static_history and not request.allow_static:
                continue
            if request.bulk_size > 1 and request.bulk_size > spec.max_batch_size:
                continue
            asset_type = (request.asset_type or "stock").lower()
            if asset_type and asset_type not in spec.asset_types:
                continue
            country = request.country.upper()
            exchange = request.exchange.upper()
            region_tokens = {country, exchange, "GLOBAL"}
            if country in {"GB", "UK", "DE", "FR", "IT", "ES", "NL", "BE", "CH", "SE", "NO", "DK", "FI"} or any(
                token in exchange for token in ["LSE", "XETRA", "EURONEXT", "BORSA", "MILAN", "PARIS", "FRANKFURT", "LONDON"]
            ):
                region_tokens.add("EUROPE")
            if country in {"JP", "JAPAN", "HK", "CN", "KR", "SG"} or any(token in exchange for token in ["TOKYO", "OSAKA", "HKEX", "SHANGHAI", "SHENZHEN"]):
                region_tokens.add("ASIA")
                if country in {"JP", "JAPAN"} or any(token in exchange for token in ["TOKYO", "OSAKA"]):
                    region_tokens.add("JP")
            if country in {"US", "USA"} or any(token in exchange for token in ["NASDAQ", "NYSE", "AMEX"]):
                region_tokens.add("US")
            spec_regions = {region.upper() for region in spec.regions}
            if "GLOBAL" not in spec_regions and not (region_tokens & spec_regions):
                continue
            if not request.allow_paid and spec.cost_tier == "paid":
                continue
            out.append(spec)
        return out


class DataProviderPolicyEngine:
    """Rank providers by quality, coverage, cost, stability and live budget."""

    def __init__(
        self,
        registry: DataProviderRegistry | None = None,
        usage_tracker: ProviderUsageTracker | None = None,
        weights: dict[str, float] | None = None,
    ):
        self.registry = registry or DataProviderRegistry()
        self.usage_tracker = usage_tracker
        self.weights = {
            "quality": 0.35,
            "coverage": 0.25,
            "cost": 0.20,
            "stability": 0.15,
            "priority": 0.05,
            **(weights or {}),
        }

    def _score(self, spec: ProviderSpec, request: ProviderRequest) -> tuple[float, str]:
        if self.usage_tracker:
            available, reason = self.usage_tracker.can_call(spec)
            if not available:
                return -1.0, reason
        historical_penalty = 0.0
        if request.start and pd.to_datetime(request.start, errors="coerce") < pd.to_datetime(spec.historical_start, errors="coerce"):
            historical_penalty = 0.12
        score = (
            spec.quality_score * self.weights["quality"]
            + spec.coverage_score * self.weights["coverage"]
            + spec.cost_score * self.weights["cost"]
            + spec.policy_stability_score * self.weights["stability"]
            + (1 / max(spec.priority, 1)) * self.weights["priority"]
            - historical_penalty
        )
        if request.bulk_size > 1 and spec.supports_bulk_daily:
            score += 0.08
        return round(score, 4), "ranked"

    def rank_providers(self, request: ProviderRequest, candidates: list[str] | None = None) -> list[ProviderDecision]:
        decisions: list[ProviderDecision] = []
        for spec in self.registry.capable(request, candidates=candidates):
            score, reason = self._score(spec, request)
            if score < 0:
                decisions.append(ProviderDecision(spec.name, score, reason, self.usage_tracker.remaining_day_budget(spec) if self.usage_tracker else None))
                continue
            decisions.append(ProviderDecision(spec.name, score, reason, self.usage_tracker.remaining_day_budget(spec) if self.usage_tracker else None))
        return sorted([d for d in decisions if d.score >= 0], key=lambda item: item.score, reverse=True)

    def provider_order(self, request: ProviderRequest, candidates: list[str] | None = None) -> list[str]:
        return [decision.provider for decision in self.rank_providers(request, candidates=candidates)]

    def reserve(self, provider_name: str, units: int = 1) -> bool:
        if not self.usage_tracker:
            return True
        spec = self.registry.get(provider_name)
        ok, _ = self.usage_tracker.can_call(spec, units=units)
        if ok:
            self.usage_tracker.record_call(provider_name, units=units)
        return ok


class RateLimiter:
    def __init__(self, requests_per_minute: int):
        self.requests_per_minute = max(int(requests_per_minute), 1)
        self.request_times: list[float] = []

    def wait(self) -> None:
        now = time.time()
        self.request_times = [t for t in self.request_times if now - t < 60]
        if len(self.request_times) >= self.requests_per_minute:
            sleep_time = 60 - (now - self.request_times[0])
            time.sleep(max(0.0, sleep_time))
        self.request_times.append(time.time())


class APIOrchestrator:
    """Try providers by priority, with local rate limiting and cache support."""

    def __init__(
        self,
        log_path: Path | str | None = None,
        providers: dict[str, ProviderSpec] | None = None,
        cache: DataCache | None = None,
        usage_tracker: ProviderUsageTracker | None = None,
        api_keys: dict[str, str] | None = None,
    ):
        self.log_path = Path(log_path or "output/logs/api_orchestrator.jsonl").expanduser()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.providers = providers or DEFAULT_PROVIDERS
        self.cache = cache
        self.api_keys = dict(api_keys or {})
        for key, value in self.api_keys.items():
            if value:
                os.environ.setdefault(str(key), str(value))
        self.usage_tracker = usage_tracker or ProviderUsageTracker(self.log_path.with_suffix(".usage.json"))
        self.limiters = {name: RateLimiter(spec.requests_per_minute) for name, spec in self.providers.items()}

    def _append_log(self, row: dict[str, Any]) -> None:
        payload = {"timestamp": utc_now(), **row}
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, default=str) + "\n")

    def fetch_with_fallback(
        self,
        key: str,
        fetchers: dict[str, Callable[[], Any]],
        ttl_hours: float | None = None,
        required_non_empty: bool = True,
        provider_order: list[str] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        if self.cache is not None and ttl_hours is not None:
            cached = self.cache.get(key, ttl_hours=ttl_hours)
            if cached is not None:
                meta = {"status": "cache_hit", "provider": "cache", "key": key, "updated_at": utc_now()}
                self._append_log(meta)
                return cached, meta

        if provider_order:
            provider_names = [name for name in provider_order if name in fetchers and self.providers.get(name, ProviderSpec(name, 999)).enabled]
        else:
            provider_names = sorted(
                [name for name in fetchers if self.providers.get(name, ProviderSpec(name, 999)).enabled],
                key=lambda name: self.providers.get(name, ProviderSpec(name, 999)).priority,
            )
        errors: list[dict[str, str]] = []
        for provider in provider_names:
            spec = self.providers.get(provider, ProviderSpec(provider, 999))
            available, budget_reason = self.usage_tracker.can_call(spec)
            if not available:
                self._append_log({"status": "budget_skipped", "key": key, "provider": provider, "reason": budget_reason})
                errors.append({"provider": provider, "error": budget_reason})
                continue
            last_exc: Exception | None = None
            for attempt in range(1, spec.max_retries + 2):
                try:
                    self.limiters.setdefault(provider, RateLimiter(spec.requests_per_minute)).wait()
                    self.usage_tracker.record_call(provider)
                    value = fetchers[provider]()
                    if required_non_empty and isinstance(value, pd.DataFrame) and value.empty:
                        raise ValueError("empty DataFrame")
                    if self.cache is not None and ttl_hours is not None:
                        self.cache.set(key, value, metadata={"provider": provider})
                    meta = {"status": "success", "provider": provider, "key": key, "attempt": attempt, "updated_at": utc_now()}
                    self._append_log(meta)
                    return value, meta
                except Exception as exc:
                    last_exc = exc
                    self._append_log({"status": "retry_failed", "key": key, "provider": provider, "attempt": attempt, "error": str(exc)})
                    if attempt <= spec.max_retries:
                        time.sleep(float(spec.backoff_seconds) * (2 ** (attempt - 1)))
            error = {"provider": provider, "error": str(last_exc)}
            errors.append(error)
            self._append_log({"status": "failed", "key": key, **error})
            continue
        raise RuntimeError(f"All providers failed for {key}: {errors}")

    def _api_key(self, *names: str) -> str:
        for name in names:
            if name in self.api_keys and self.api_keys[name]:
                return str(self.api_keys[name])
            if os.environ.get(name):
                return str(os.environ[name])
        return ""

    def _request_json(self, provider: str, url: str, *, params: dict[str, Any] | None = None, timeout: int = 30) -> Any | None:
        spec = self.providers.get(provider, ProviderSpec(provider, 999))
        available, reason = self.usage_tracker.can_call(spec)
        if not available:
            self._append_log({"status": "budget_skipped", "provider": provider, "key": url, "reason": reason})
            return None
        try:
            import requests

            self.limiters.setdefault(provider, RateLimiter(spec.requests_per_minute)).wait()
            self.usage_tracker.record_call(provider)
            response = requests.get(url, params=params, timeout=timeout, headers={"User-Agent": "Gen.is.IA ResearchPlatform/2.0"})
            if response.status_code == 429:
                time.sleep(float(spec.backoff_seconds))
                response = requests.get(url, params=params, timeout=timeout, headers={"User-Agent": "Gen.is.IA ResearchPlatform/2.0"})
            response.raise_for_status()
            self._append_log({"status": "success", "provider": provider, "key": url})
            return response.json()
        except Exception as exc:
            self._append_log({"status": "failed", "provider": provider, "key": url, "error": f"{type(exc).__name__}: {exc}"})
            return None

    @staticmethod
    def _first_payload(payload: Any) -> dict[str, Any]:
        if isinstance(payload, list):
            return payload[0] if payload and isinstance(payload[0], dict) else {}
        return payload if isinstance(payload, dict) else {}

    def get_fundamentals_fmp(self, ticker: str, statement: str = "income", period: str = "annual") -> dict[str, Any] | None:
        """Fetch structured fundamentals from Financial Modeling Prep."""
        key = self._api_key("FMP_API_KEY")
        if not key:
            return None
        endpoint_map = {
            "income": f"income-statement/{ticker}",
            "balance": f"balance-sheet-statement/{ticker}",
            "cashflow": f"cash-flow-statement/{ticker}",
            "key_metrics_ttm": f"key-metrics-ttm/{ticker}",
            "ratios_ttm": f"ratios-ttm/{ticker}",
            "earnings_surprises": f"earnings-surprises/{ticker}",
            "analyst_estimates": f"analyst-estimates/{ticker}",
            "institutional_holder": f"institutional-holder/{ticker}",
            "insider_trading": "insider-trading",
            "short_interest": "short_interest",
        }
        endpoint = endpoint_map.get(statement, endpoint_map["income"])
        version = "v4" if statement in {"insider_trading", "short_interest"} else "v3"
        url = f"https://financialmodelingprep.com/api/{version}/{endpoint}"
        params = {"apikey": key}
        if statement not in {"key_metrics_ttm", "ratios_ttm", "insider_trading", "short_interest"}:
            params["period"] = period
        if statement in {"insider_trading", "short_interest"}:
            params["symbol"] = ticker
            params["limit"] = 100
        payload = self._request_json("fmp", url, params=params)
        if payload is None:
            return None
        return {"provider": "fmp", "statement": statement, "payload": payload, **self._first_payload(payload)}

    def get_fundamentals_polygon(self, ticker: str, statement: str = "income") -> dict[str, Any] | None:
        """Fetch GAAP/IFRS fundamentals from Polygon when a key is configured."""
        key = self._api_key("POLYGON_API_KEY")
        if not key:
            return None
        url = "https://api.polygon.io/vX/reference/financials"
        payload = self._request_json("polygon", url, params={"ticker": ticker, "timeframe": "annual", "include_sources": "true", "apiKey": key})
        if payload is None:
            return None
        results = payload.get("results", []) if isinstance(payload, dict) else []
        first = results[0] if results and isinstance(results[0], dict) else {}
        return {"provider": "polygon", "statement": statement, "payload": payload, **first}

    def get_fundamentals_eodhd(self, ticker: str) -> dict[str, Any] | None:
        """Fetch global fundamentals from EODHD for US/EU/JP tickers."""
        key = self._api_key("EODHD_API_KEY")
        if not key:
            return None
        url = f"https://eodhistoricaldata.com/api/fundamentals/{ticker}"
        payload = self._request_json("eodhd", url, params={"api_token": key, "fmt": "json"})
        if payload is None:
            return None
        return {"provider": "eodhd", "payload": payload, **self._first_payload(payload)}

    def get_fundamentals_intrinio(self, ticker: str, tag: str) -> float | None:
        """Fetch one Intrinio data point, returning None when unavailable."""
        key = self._api_key("INTRINIO_API_KEY")
        if not key:
            return None
        url = f"https://api-v2.intrinio.com/companies/{ticker}/data_point/{tag}/number"
        payload = self._request_json("intrinio", url, params={"api_key": key})
        try:
            if isinstance(payload, dict):
                return float(payload.get("value", payload.get("number")))
            return float(payload)
        except (TypeError, ValueError):
            return None

    def get_price_data_polygon(self, ticker: str, start: str, end: str, timespan: str = "day") -> pd.DataFrame:
        """Fetch historical aggregate bars from Polygon."""
        key = self._api_key("POLYGON_API_KEY")
        if not key:
            return pd.DataFrame()
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/{timespan}/{start}/{end}"
        payload = self._request_json("polygon", url, params={"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": key})
        results = payload.get("results", []) if isinstance(payload, dict) else []
        if not results:
            return pd.DataFrame()
        frame = pd.DataFrame(results)
        if "t" in frame.columns:
            frame["date"] = pd.to_datetime(frame["t"], unit="ms", errors="coerce")
        return frame.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})

    @staticmethod
    def _completeness(payload: dict[str, Any]) -> float:
        if not payload:
            return 0.0
        scalar_items = [value for key, value in payload.items() if key not in {"payload", "provider"}]
        if not scalar_items:
            return 0.0
        filled = sum(value not in (None, "", [], {}) for value in scalar_items)
        return round(100.0 * filled / max(len(scalar_items), 1), 2)

    def get_fundamentals_waterfall(self, ticker: str) -> dict[str, Any]:
        """Try FMP, Polygon, EODHD and yfinance, returning a normalized envelope."""
        for provider, getter in [
            ("fmp", lambda: self.get_fundamentals_fmp(ticker, "key_metrics_ttm")),
            ("polygon", lambda: self.get_fundamentals_polygon(ticker)),
            ("eodhd", lambda: self.get_fundamentals_eodhd(ticker)),
        ]:
            payload = getter()
            if payload:
                return {
                    "ticker": ticker,
                    "source_provider": provider,
                    "data_completeness_pct": self._completeness(payload),
                    "data": payload,
                }
        if os.environ.get("ENABLE_YFINANCE_FUNDAMENTALS_FALLBACK", "").lower() in {"1", "true", "yes"}:
            try:
                import yfinance as yf

                info = yf.Ticker(ticker).info or {}
                if info:
                    return {
                        "ticker": ticker,
                        "source_provider": "yfinance",
                        "data_completeness_pct": self._completeness(info),
                        "data": info,
                    }
            except Exception as exc:
                self._append_log({"status": "failed", "provider": "yfinance", "key": ticker, "error": f"{type(exc).__name__}: {exc}"})
        return {"ticker": ticker, "source_provider": "none", "data_completeness_pct": 0.0, "data": {}}


ApiOrchestrator = APIOrchestrator


def get_fundamentals_waterfall(ticker: str, orchestrator: APIOrchestrator | None = None) -> dict[str, Any]:
    """Module-level convenience wrapper for provider waterfall fundamentals."""
    client = orchestrator or APIOrchestrator()
    return client.get_fundamentals_waterfall(ticker)
