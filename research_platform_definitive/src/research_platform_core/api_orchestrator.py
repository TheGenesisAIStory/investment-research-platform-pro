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
    "fmp": ProviderSpec("fmp", priority=4, requests_per_minute=30, cost_tier="limited", requires_env=("FMP_API_KEY",)),
    "polygon": ProviderSpec("polygon", priority=5, requests_per_minute=120, cost_tier="paid", requires_env=("POLYGON_API_KEY",)),
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
        return pd.Timestamp.utcnow().date().isoformat()

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
        log_path: Path | str,
        providers: dict[str, ProviderSpec] | None = None,
        cache: DataCache | None = None,
        usage_tracker: ProviderUsageTracker | None = None,
    ):
        self.log_path = Path(log_path).expanduser()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.providers = providers or DEFAULT_PROVIDERS
        self.cache = cache
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
