"""OHLCV provider client built on the shared APIOrchestrator."""

from __future__ import annotations

import os
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd

from ..api_orchestrator import (
    APIOrchestrator,
    DataProviderPolicyEngine,
    DataProviderRegistry,
    ProviderRequest,
    ProviderUsageTracker,
)
from ..cache_manager import DataCache
from ..data_platform import resolve_data_platform_roots, utc_now


DAILY_COLUMNS = ["date", "open", "high", "low", "close", "adjclose", "volume", "provider_symbol", "source"]
INTRADAY_COLUMNS = ["ts", "open", "high", "low", "close", "volume", "provider_symbol", "source"]


def normalize_ohlcv_frame(raw: pd.DataFrame, provider_symbol: str, source: str, interval: str = "1d") -> pd.DataFrame:
    """Normalize provider output to a stable OHLCV schema."""
    if raw is None or raw.empty:
        return pd.DataFrame(columns=INTRADAY_COLUMNS if interval != "1d" else DAILY_COLUMNS)
    df = raw.copy()
    if not isinstance(df.index, pd.RangeIndex):
        df = df.reset_index()
    rename = {
        "Date": "date",
        "Datetime": "ts",
        "timestamp": "ts",
        "time": "ts",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adjclose",
        "Adjusted Close": "adjclose",
        "Volume": "volume",
        "1. open": "open",
        "2. high": "high",
        "3. low": "low",
        "4. close": "close",
        "5. adjusted close": "adjclose",
        "5. volume": "volume",
        "6. volume": "volume",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    if interval == "1d":
        if "date" not in df.columns and "ts" in df.columns:
            df["date"] = df["ts"]
        if "adjclose" not in df.columns and "close" in df.columns:
            df["adjclose"] = df["close"]
        for col in ["open", "high", "low", "close", "adjclose", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date.astype("string")
        df["provider_symbol"] = provider_symbol
        df["source"] = source
        df["ingestion_ts"] = utc_now()
        return df[[c for c in DAILY_COLUMNS + ["ingestion_ts"] if c in df.columns]].dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    if "ts" not in df.columns and "date" in df.columns:
        df["ts"] = df["date"]
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)
    df["provider_symbol"] = provider_symbol
    df["source"] = source
    df["ingestion_ts"] = utc_now()
    return df[[c for c in INTRADAY_COLUMNS + ["ingestion_ts"] if c in df.columns]].dropna(subset=["ts", "close"]).sort_values("ts").reset_index(drop=True)


def parse_yfinance_bulk(raw: pd.DataFrame, symbols: list[str], source: str = "yfinance") -> dict[str, pd.DataFrame]:
    """Split a yfinance bulk download result into normalized per-symbol frames."""
    if raw is None or raw.empty:
        return {}
    result: dict[str, pd.DataFrame] = {}
    if isinstance(raw.columns, pd.MultiIndex):
        level0 = set(map(str, raw.columns.get_level_values(0)))
        level1 = set(map(str, raw.columns.get_level_values(1)))
        ticker_first = bool(set(symbols) & level0)
        for symbol in symbols:
            try:
                frame = raw[symbol] if ticker_first else raw.xs(symbol, axis=1, level=1)
                result[symbol] = normalize_ohlcv_frame(frame, symbol, source, interval="1d")
            except Exception:
                continue
    else:
        symbol = symbols[0] if symbols else ""
        result[symbol] = normalize_ohlcv_frame(raw, symbol, source, interval="1d")
    return {symbol: frame for symbol, frame in result.items() if not frame.empty}


def _load_ohlcv_config() -> dict[str, Any]:
    config_path = Path(__file__).resolve().parents[3] / "config" / "ohlcv_data_sources.yaml"
    if not config_path.exists():
        return {}
    try:
        import yaml

        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _asset_value(asset: Any, key: str, default: str = "") -> str:
    if asset is None:
        return default
    if isinstance(asset, dict):
        return str(asset.get(key, default) or default)
    try:
        return str(asset.get(key, default) or default)
    except Exception:
        return default


def _daily_quality_summary(frame: pd.DataFrame, start: str | None, end: str | None) -> dict[str, Any]:
    if frame is None or frame.empty or "date" not in frame.columns:
        return {"rows": 0, "coverage_ratio": 0.0, "first_date": None, "last_date": None}
    dates = pd.to_datetime(frame["date"], errors="coerce").dropna().dt.date
    if dates.empty:
        return {"rows": 0, "coverage_ratio": 0.0, "first_date": None, "last_date": None}
    start_dt = pd.to_datetime(start, errors="coerce").date() if start else min(dates)
    end_dt = pd.to_datetime(end, errors="coerce").date() if end else min(pd.Timestamp.utcnow().date(), max(dates))
    expected = pd.bdate_range(start_dt, end_dt)
    coverage = min(float(dates.nunique()) / max(len(expected), 1), 1.0)
    return {
        "rows": int(len(frame)),
        "coverage_ratio": round(coverage, 4),
        "first_date": min(dates).isoformat(),
        "last_date": max(dates).isoformat(),
    }


def merge_ohlcv_frames(frames: list[pd.DataFrame], provider_order: list[str]) -> pd.DataFrame:
    """Merge overlapping daily frames with earlier providers winning conflicts."""
    clean = [frame.copy() for frame in frames if frame is not None and not frame.empty]
    if not clean:
        return pd.DataFrame(columns=DAILY_COLUMNS)
    rank = {provider: idx for idx, provider in enumerate(provider_order)}
    merged = pd.concat(clean, ignore_index=True, sort=False)
    merged["_provider_rank"] = merged["source"].map(lambda value: rank.get(str(value), len(rank) + 1))
    merged = merged.sort_values(["date", "_provider_rank"]).drop_duplicates(["date"], keep="first")
    return merged.drop(columns=["_provider_rank"]).sort_values("date").reset_index(drop=True)


class BaseOhlcvProvider:
    """Minimal provider adapter surface used by the policy engine."""

    name = "base"
    supports_bulk_daily = False

    def get_ohlcv_daily(self, symbol: str, start: str, end: str | None = None) -> pd.DataFrame:
        raise NotImplementedError

    def get_ohlcv_daily_bulk(self, symbols: list[str], start: str, end: str | None = None) -> dict[str, pd.DataFrame]:
        return {}

    def get_ohlcv_intraday_5m(self, symbol: str, month: str | None = None) -> pd.DataFrame:
        raise NotImplementedError

    def get_symbol_universe(self, market: str) -> pd.DataFrame:
        return pd.DataFrame(columns=["ticker", "provider_symbol", "exchange", "country", "type", "source"])


class YFinanceOhlcvProvider(BaseOhlcvProvider):
    name = "yfinance"
    supports_bulk_daily = True

    def get_ohlcv_daily_bulk(self, symbols: list[str], start: str, end: str | None = None) -> dict[str, pd.DataFrame]:
        import yfinance as yf

        symbols = [str(s).strip() for s in symbols if str(s).strip()]
        if not symbols:
            return {}
        raw = yf.download(
            tickers=symbols,
            start=start,
            end=end,
            interval="1d",
            group_by="ticker",
            auto_adjust=False,
            actions=False,
            progress=False,
            threads=True,
        )
        return parse_yfinance_bulk(raw, symbols, source=self.name)

    def get_ohlcv_daily(self, symbol: str, start: str, end: str | None = None) -> pd.DataFrame:
        import yfinance as yf

        raw = yf.download(symbol, start=start, end=end, interval="1d", auto_adjust=False, actions=False, progress=False, threads=False)
        return normalize_ohlcv_frame(raw, symbol, self.name, interval="1d")


class StooqOhlcvProvider(BaseOhlcvProvider):
    name = "stooq"

    def get_ohlcv_daily(self, symbol: str, start: str, end: str | None = None) -> pd.DataFrame:
        import requests

        api_key = os.environ.get("STOOQ_API_KEY", "")
        if not api_key:
            raise RuntimeError("STOOQ_API_KEY missing; Stooq CSV downloads require a key.")
        stooq_symbol = symbol.lower()
        if "." not in stooq_symbol:
            stooq_symbol = f"{stooq_symbol}.us"
        params = {"s": stooq_symbol, "i": "d", "apikey": api_key}
        if start:
            params["d1"] = str(start).replace("-", "")
        if end:
            params["d2"] = str(end).replace("-", "")
        response = requests.get("https://stooq.com/q/d/l/", params=params, timeout=30, headers={"User-Agent": "ResearchPlatform/1.0 OhlcvClient"})
        response.raise_for_status()
        if "Get your apikey" in response.text:
            raise RuntimeError("Stooq rejected request; refresh STOOQ_API_KEY from stooq.com.")
        df = pd.read_csv(StringIO(response.text))
        return normalize_ohlcv_frame(df, symbol, self.name, interval="1d")


class AlphaVantageOhlcvProvider(BaseOhlcvProvider):
    name = "alpha_vantage"

    @staticmethod
    def _api_key() -> str:
        api_key = os.environ.get("ALPHAVANTAGE_API_KEY") or os.environ.get("ALPHA_VANTAGE_API_KEY", "")
        if not api_key:
            raise RuntimeError("ALPHAVANTAGE_API_KEY missing")
        return api_key

    def get_ohlcv_daily(self, symbol: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        import requests

        params = {"function": "TIME_SERIES_DAILY_ADJUSTED", "symbol": symbol, "outputsize": "full", "apikey": self._api_key()}
        payload = requests.get("https://www.alphavantage.co/query", params=params, timeout=60).json()
        series = payload.get("Time Series (Daily)", {})
        if not series:
            raise RuntimeError(str(payload)[:300])
        df = pd.DataFrame.from_dict(series, orient="index").reset_index().rename(columns={"index": "Date"})
        out = normalize_ohlcv_frame(df, symbol, self.name, interval="1d")
        if start:
            out = out[out["date"].ge(str(start))]
        if end:
            out = out[out["date"].le(str(end))]
        return out.reset_index(drop=True)

    def get_ohlcv_intraday_5m(self, symbol: str, month: str | None = None) -> pd.DataFrame:
        import requests

        params = {"function": "TIME_SERIES_INTRADAY", "symbol": symbol, "interval": "5min", "outputsize": "full", "apikey": self._api_key()}
        if month:
            params["month"] = month
        payload = requests.get("https://www.alphavantage.co/query", params=params, timeout=60).json()
        series = payload.get("Time Series (5min)", {})
        if not series:
            raise RuntimeError(str(payload)[:300])
        df = pd.DataFrame.from_dict(series, orient="index").reset_index().rename(columns={"index": "ts"})
        return normalize_ohlcv_frame(df, symbol, self.name, interval="5m")


class OhlcvClient:
    """Fetch daily and intraday OHLCV through the provider policy engine."""

    def __init__(
        self,
        financial_db_root: Path | str | None = None,
        output_root: Path | str | None = None,
        orchestrator: APIOrchestrator | None = None,
        cache_ttl_hours: float = 12,
    ):
        roots = resolve_data_platform_roots(financial_db_root=financial_db_root, repo_output_root=output_root)
        self.financial_db_root = roots.financial_db
        self.output_root = roots.repo_output
        self.config = _load_ohlcv_config()
        ohlcv_config = self.config.get("ohlcv", {})
        cache_config = ohlcv_config.get("cache", {})
        policy_config = ohlcv_config.get("policy", {})
        self.cache_enabled = bool(cache_config.get("enabled", True))
        self.cache_ttl_hours = float(cache_config.get("daily_request_ttl_hours", cache_ttl_hours))
        self.intraday_cache_ttl_hours = float(cache_config.get("intraday_request_ttl_hours", 1))
        self.cache = DataCache(roots.local_cache / "ohlcv") if self.cache_enabled else None
        self.registry = DataProviderRegistry.from_config(self.config)
        self.usage_tracker = ProviderUsageTracker(roots.repo_output / "logs" / "ohlcv_provider_usage.json")
        self.policy_engine = DataProviderPolicyEngine(
            registry=self.registry,
            usage_tracker=self.usage_tracker,
            weights=policy_config.get("weights") if isinstance(policy_config.get("weights"), dict) else None,
        )
        self.coverage_threshold = float(policy_config.get("incomplete_coverage_threshold", 0.65))
        self.allow_provider_blend = bool(policy_config.get("allow_provider_blend", True))
        self.orchestrator = orchestrator or APIOrchestrator(
            roots.repo_output / "logs" / "ohlcv_api_orchestrator.jsonl",
            providers=self.registry.providers,
            cache=self.cache,
            usage_tracker=self.usage_tracker,
        )
        self.provider_adapters: dict[str, BaseOhlcvProvider] = {
            "yfinance": YFinanceOhlcvProvider(),
            "stooq": StooqOhlcvProvider(),
            "alpha_vantage": AlphaVantageOhlcvProvider(),
        }

    def _request(
        self,
        symbol: str,
        interval: str,
        start: str | None = None,
        end: str | None = None,
        asset: Any = None,
        bulk_size: int = 1,
        allow_paid: bool = False,
    ) -> ProviderRequest:
        return ProviderRequest(
            provider_symbol=symbol,
            interval=interval,
            start=start,
            end=end,
            asset_type=_asset_value(asset, "type", "stock").lower(),
            country=_asset_value(asset, "country", ""),
            exchange=_asset_value(asset, "exchange", ""),
            bulk_size=bulk_size,
            allow_paid=allow_paid,
        )

    def _provider_order(
        self,
        request: ProviderRequest,
        preferred_providers: list[str] | None = None,
    ) -> list[str]:
        order = self.policy_engine.provider_order(request, candidates=preferred_providers)
        return [provider for provider in order if provider in self.provider_adapters]

    def _daily_fetchers(self, symbol: str, start: str, end: str | None, order: list[str]) -> dict[str, Any]:
        return {
            provider: (lambda provider=provider: self.provider_adapters[provider].get_ohlcv_daily(symbol, start, end))
            for provider in order
        }

    def fetch_yfinance_daily_bulk(self, symbols: list[str], start: str, end: str | None = None) -> dict[str, pd.DataFrame]:
        """Backward-compatible direct yfinance bulk helper."""
        return self.provider_adapters["yfinance"].get_ohlcv_daily_bulk(symbols, start, end)

    def fetch_yfinance_daily_one(self, symbol: str, start: str, end: str | None = None) -> pd.DataFrame:
        """Backward-compatible direct yfinance helper."""
        return self.provider_adapters["yfinance"].get_ohlcv_daily(symbol, start, end)

    def fetch_stooq_daily_one(self, symbol: str, start: str, end: str | None = None) -> pd.DataFrame:
        """Backward-compatible direct Stooq helper."""
        return self.provider_adapters["stooq"].get_ohlcv_daily(symbol, start, end)

    def fetch_alpha_vantage_daily_one(self, symbol: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        """Backward-compatible direct Alpha Vantage helper."""
        return self.provider_adapters["alpha_vantage"].get_ohlcv_daily(symbol, start, end)

    def fetch_daily_bulk(
        self,
        symbols: list[str],
        start: str,
        end: str | None = None,
        assets: list[dict[str, Any]] | None = None,
        preferred_providers: list[str] | None = None,
    ) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
        """Fetch a batch with the best bulk-capable provider selected at runtime."""
        symbols = [str(s).strip() for s in symbols if str(s).strip()]
        if not symbols:
            return {}, {"status": "empty_request", "provider": None, "updated_at": utc_now()}
        request = self._request(
            ",".join(symbols[:5]),
            interval="1d",
            start=start,
            end=end,
            asset=(assets[0] if assets else None),
            bulk_size=len(symbols),
        )
        order = self._provider_order(request, preferred_providers=preferred_providers)
        fetchers = {
            provider: (lambda provider=provider: self.provider_adapters[provider].get_ohlcv_daily_bulk(symbols, start, end))
            for provider in order
            if self.provider_adapters[provider].supports_bulk_daily
        }
        if not fetchers:
            return {}, {"status": "no_bulk_provider", "provider": None, "provider_order": order, "updated_at": utc_now()}

        def require_rows(provider: str):
            result = fetchers[provider]()
            if not result:
                raise ValueError("empty bulk result")
            return result

        cache_key = f"daily_bulk::{','.join(symbols)}::{start}::{end or ''}"
        result, meta = self.orchestrator.fetch_with_fallback(
            cache_key,
            {provider: (lambda provider=provider: require_rows(provider)) for provider in fetchers},
            ttl_hours=self.cache_ttl_hours if self.cache_enabled else None,
            required_non_empty=False,
            provider_order=list(fetchers),
        )
        meta["provider_order"] = order
        return result, meta

    def fetch_daily_one(
        self,
        symbol: str,
        start: str,
        end: str | None = None,
        asset: Any = None,
        preferred_providers: list[str] | None = None,
        allow_blend: bool | None = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Fetch one daily history through provider ranking and automatic fallback."""
        request = self._request(symbol, interval="1d", start=start, end=end, asset=asset)
        order = self._provider_order(request, preferred_providers=preferred_providers)
        if not order:
            raise RuntimeError(f"No daily provider available for {symbol}")
        cache_key = f"daily::{symbol}::{start}::{end or ''}"
        frame, meta = self.orchestrator.fetch_with_fallback(
            cache_key,
            self._daily_fetchers(symbol, start, end, order),
            ttl_hours=self.cache_ttl_hours if self.cache_enabled else None,
            provider_order=order,
        )
        meta["provider_order"] = order
        meta["quality"] = _daily_quality_summary(frame, start, end)
        blend_enabled = self.allow_provider_blend if allow_blend is None else allow_blend
        if not blend_enabled or meta["quality"]["coverage_ratio"] >= self.coverage_threshold or len(order) < 2:
            return frame, meta

        frames = [frame]
        used = [str(meta.get("provider"))]
        for provider in order:
            if provider == meta.get("provider"):
                continue
            try:
                extra, extra_meta = self.orchestrator.fetch_with_fallback(
                    f"{cache_key}::{provider}",
                    {provider: self._daily_fetchers(symbol, start, end, [provider])[provider]},
                    ttl_hours=self.cache_ttl_hours if self.cache_enabled else None,
                    provider_order=[provider],
                )
                frames.append(extra)
                used.append(provider)
                merged = merge_ohlcv_frames(frames, order)
                merged_quality = _daily_quality_summary(merged, start, end)
                if merged_quality["coverage_ratio"] >= self.coverage_threshold:
                    meta.update({"status": "success_blended", "provider": "+".join([p for p in used if p]), "quality": merged_quality})
                    return merged, meta
            except Exception as exc:
                meta.setdefault("blend_errors", []).append({"provider": provider, "error": str(exc)})
        merged = merge_ohlcv_frames(frames, order)
        meta.update({"status": "success_blended_partial", "provider": "+".join([p for p in used if p]), "quality": _daily_quality_summary(merged, start, end)})
        return merged, meta

    def fetch_alpha_vantage_intraday_5m(self, symbol: str, month: str | None = None) -> pd.DataFrame:
        """Backward-compatible direct Alpha Vantage 5 minute helper."""
        return self.provider_adapters["alpha_vantage"].get_ohlcv_intraday_5m(symbol, month=month)

    def fetch_intraday_5m_one(
        self,
        symbol: str,
        month: str | None = None,
        asset: Any = None,
        preferred_providers: list[str] | None = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        request = self._request(symbol, interval="5m", start=month, end=month, asset=asset)
        order = self._provider_order(request, preferred_providers=preferred_providers)
        if not order:
            raise RuntimeError(f"No 5m provider available for {symbol}")
        cache_key = f"intraday5m::{symbol}::{month or 'latest'}"
        frame, meta = self.orchestrator.fetch_with_fallback(
            cache_key,
            {
                provider: (lambda provider=provider: self.provider_adapters[provider].get_ohlcv_intraday_5m(symbol, month=month))
                for provider in order
            },
            ttl_hours=self.intraday_cache_ttl_hours if self.cache_enabled else None,
            provider_order=order,
        )
        meta["provider_order"] = order
        return frame, meta
