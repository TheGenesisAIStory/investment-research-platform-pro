"""API provider registry and safe credential helpers.

Secrets must come from environment variables or local `.env` files ignored by
git. This module stores provider metadata, small connectivity checks, and
masked credential status only.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from json import JSONDecodeError
from pathlib import Path
from typing import Any

import pandas as pd
import requests


@dataclass(frozen=True)
class APIProvider:
    name: str
    category: str
    url: str
    env_var: str
    openbb_credential: str | None
    data_type: str
    ml_use: str
    quality: str
    free_tier: str
    notes: str


API_PROVIDERS: tuple[APIProvider, ...] = (
    APIProvider("Alpha Vantage", "Market Data / Fundamentals", "https://www.alphavantage.co/", "ALPHA_VANTAGE_API_KEY", "alpha_vantage_api_key", "OHLCV, fundamentals, FX, crypto, indicators", "Feature engineering, prototyping, cross-asset signals", "Media-alta", "Yes, rate-limited", "Useful for prototypes; respect free-tier throttles."),
    APIProvider("Tiingo", "Market Data / News", "https://www.tiingo.com/", "TIINGO_TOKEN", "tiingo_token", "Equity prices, IEX data, news, crypto", "Backtests, news features, cleaned daily prices", "Alta", "Yes, limited", "Stable daily price source for research."),
    APIProvider("Financial Modeling Prep", "Fundamentals / Market Data", "https://financialmodelingprep.com/", "FMP_API_KEY", "fmp_api_key", "Statements, ratios, profiles, prices", "Fundamental factors, cross-sectional ML, screening", "Media-alta", "Yes, limited", "Validate fields and licensing before production use."),
    APIProvider("Polygon", "Market Data", "https://polygon.io/", "POLYGON_API_KEY", "polygon_api_key", "Stocks, options, indices, FX, crypto", "Intraday features, backtests, event studies", "Alta", "Yes, limited", "Paid plans needed for many historical/intraday workflows."),
    APIProvider("Intrinio", "Fundamentals / Market Data", "https://intrinio.com/", "INTRINIO_API_KEY", "intrinio_api_key", "Fundamentals, estimates, prices, ownership", "Fundamental ML and entity-level datasets", "Alta", "Developer/trial dependent", "Endpoint access varies by subscription."),
    APIProvider("FRED", "Macro", "https://fred.stlouisfed.org/", "FRED_API_KEY", "fred_api_key", "Macro, rates, inflation, labor", "Regime detection, macro alpha, risk features", "Alta", "Yes", "Core source for macro regimes."),
    APIProvider("BLS", "Macro / Labor", "https://www.bls.gov/developers/", "BLS_API_KEY", None, "Labor market, CPI, PPI, productivity", "Inflation/labor regimes and nowcasting", "Alta", "Yes", "Official US labor and price data."),
    APIProvider("CFTC", "Futures / Positioning", "https://publicreporting.cftc.gov/", "CFTC_APP_TOKEN", None, "Commitments of Traders, futures positioning", "Positioning and crowded-trade risk", "Alta", "Yes", "Socrata token improves public endpoint limits."),
    APIProvider("Congress.gov", "Government / Policy", "https://api.congress.gov/", "CONGRESS_GOV_API_KEY", None, "Bills, amendments, members, policy events", "Policy-event and legislative attention features", "Media-alta", "Yes", "Useful for regulatory/policy signals."),
    APIProvider("EconDB", "Macro", "https://www.econdb.com/", "ECONDB_API_KEY", None, "Global macro time series", "Global macro panels and regimes", "Media-alta", "Terms dependent", "Complements FRED for broader country coverage."),
    APIProvider("EIA", "Energy / Macro", "https://www.eia.gov/opendata/", "EIA_API_KEY", None, "Energy prices, inventories, production", "Energy factors and inflation inputs", "Alta", "Yes", "Official US energy data."),
    APIProvider("Nasdaq Data Link", "Market / Alternative / Fundamentals", "https://data.nasdaq.com/", "NASDAQ_API_KEY", "nasdaq_api_key", "Vendor datasets, prices, fundamentals, alternative data", "Research datasets and factor inputs", "Alta", "Some free datasets", "Former Quandl. Dataset-level licensing matters."),
)


def providers_to_dataframe() -> pd.DataFrame:
    return pd.DataFrame([asdict(provider) for provider in API_PROVIDERS])


def load_dotenv(path: Path | str = ".env") -> dict[str, str]:
    env_path = Path(path)
    loaded: dict[str, str] = {}
    if not env_path.exists():
        return loaded
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded[key] = value
    return loaded


def _masked(value: str) -> str:
    if not value:
        return ""
    if len(value) < 8:
        return "set"
    return f"{value[:4]}...{value[-4:]}"


def credential_status(mask: bool = True) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for provider in API_PROVIDERS:
        value = os.getenv(provider.env_var, "")
        rows.append(
            {
                "name": provider.name,
                "env_var": provider.env_var,
                "configured": bool(value),
                "credential_preview": _masked(value) if mask else ("set" if value else ""),
                "openbb_credential": provider.openbb_credential,
            }
        )
    return pd.DataFrame(rows)


def configure_openbb_from_env() -> dict[str, str]:
    configured: dict[str, str] = {}
    if os.getenv("OPENBB_USE_ENV_CREDENTIALS", "1") != "1":
        return configured
    try:
        from openbb import obb  # type: ignore
    except Exception:
        return configured
    for provider in API_PROVIDERS:
        if not provider.openbb_credential:
            continue
        value = os.getenv(provider.env_var)
        if not value:
            continue
        try:
            setattr(obb.user.credentials, provider.openbb_credential, value)
            configured[provider.openbb_credential] = provider.env_var
        except Exception:
            continue
    return configured


def simple_health_checks(timeout: int = 10) -> pd.DataFrame:
    checks = {provider.name: provider.url for provider in API_PROVIDERS}
    rows: list[dict[str, Any]] = []
    session = requests.Session()
    session.headers.update({"User-Agent": "ml4t-provider-registry/1.0"})
    for name, url in checks.items():
        try:
            response = session.get(url, timeout=timeout)
            rows.append({"name": name, "url": url, "ok": response.ok, "reachable": response.status_code < 500, "status_code": response.status_code, "error": ""})
        except Exception as exc:
            rows.append({"name": name, "url": url, "ok": False, "reachable": False, "status_code": None, "error": str(exc)})
    return pd.DataFrame(rows)


def active_api_smoke_checks(timeout: int = 20) -> pd.DataFrame:
    """Run minimal credentialed checks for configured providers."""

    def key(env_var: str) -> str:
        return os.getenv(env_var, "").strip()

    rows: list[dict[str, Any]] = []
    session = requests.Session()
    session.headers.update({"User-Agent": "ml4t-private-api-smoke-check/1.0"})

    def add(name: str, env_var: str, ok: bool, status_code: int | None, endpoint: str, detail: str) -> None:
        rows.append(
            {
                "name": name,
                "env_var": env_var,
                "configured": bool(key(env_var)),
                "credential_preview": _masked(key(env_var)),
                "ok": ok,
                "status_code": status_code,
                "endpoint": endpoint,
                "detail": detail[:240],
            }
        )

    specs = [
        ("Alpha Vantage", "ALPHA_VANTAGE_API_KEY", "https://www.alphavantage.co/query", {"function": "GLOBAL_QUOTE", "symbol": "IBM", "apikey": "{key}"}),
        ("Tiingo", "TIINGO_TOKEN", "https://api.tiingo.com/tiingo/daily/aapl/prices", {"token": "{key}", "startDate": "2024-01-02", "endDate": "2024-01-05"}),
        ("Financial Modeling Prep", "FMP_API_KEY", "https://financialmodelingprep.com/api/v3/profile/AAPL", {"apikey": "{key}"}),
        ("Polygon", "POLYGON_API_KEY", "https://api.polygon.io/v2/aggs/ticker/AAPL/prev", {"adjusted": "true", "apiKey": "{key}"}),
        ("Intrinio", "INTRINIO_API_KEY", "https://api-v2.intrinio.com/companies/AAPL", {"api_key": "{key}"}),
        ("FRED", "FRED_API_KEY", "https://api.stlouisfed.org/fred/series/observations", {"series_id": "DGS10", "api_key": "{key}", "file_type": "json", "limit": 1, "sort_order": "desc"}),
        ("Congress.gov", "CONGRESS_GOV_API_KEY", "https://api.congress.gov/v3/bill", {"api_key": "{key}", "limit": 1, "format": "json"}),
        ("EIA", "EIA_API_KEY", "https://api.eia.gov/v2/electricity/retail-sales/data/", {"api_key": "{key}", "frequency": "monthly", "data[0]": "price", "length": 1}),
        ("Nasdaq Data Link", "NASDAQ_API_KEY", "https://data.nasdaq.com/api/v3/datasets/FRED/DGS10.json", {"api_key": "{key}", "rows": 1}),
    ]
    for name, env_var, url, params in specs:
        value = key(env_var)
        if not value:
            add(name, env_var, False, None, url, "skipped: missing credential")
            continue
        try:
            response = session.get(url, params={k: (value if v == "{key}" else v) for k, v in params.items()}, timeout=timeout)
            text = response.text[:500].lower()
            invalid = any(marker in text for marker in ["invalid api", "invalid key", "error message"])
            add(name, env_var, response.ok and not invalid, response.status_code, url, "response received")
        except Exception as exc:
            add(name, env_var, False, None, url, str(exc))

    bls_key = key("BLS_API_KEY")
    if bls_key:
        try:
            response = session.post("https://api.bls.gov/publicAPI/v2/timeseries/data/", json={"seriesid": ["CUUR0000SA0"], "latest": "true", "registrationkey": bls_key}, timeout=timeout)
            try:
                status = response.json().get("status", "")
            except (JSONDecodeError, ValueError):
                status = ""
            add("BLS", "BLS_API_KEY", response.ok and status == "REQUEST_SUCCEEDED", response.status_code, "https://api.bls.gov/publicAPI/v2/timeseries/data/", f"status={status or 'unknown'}")
        except Exception as exc:
            add("BLS", "BLS_API_KEY", False, None, "https://api.bls.gov/publicAPI/v2/timeseries/data/", str(exc))
    else:
        add("BLS", "BLS_API_KEY", False, None, "https://api.bls.gov/publicAPI/v2/timeseries/data/", "skipped: missing credential")

    cftc_key = key("CFTC_APP_TOKEN")
    try:
        headers = {"X-App-Token": cftc_key} if cftc_key else {}
        response = session.get("https://publicreporting.cftc.gov/resource/6dca-aqww.json", params={"$limit": 1}, headers=headers, timeout=timeout)
        add("CFTC", "CFTC_APP_TOKEN", response.ok, response.status_code, "https://publicreporting.cftc.gov/resource/6dca-aqww.json", "public endpoint; token optional")
    except Exception as exc:
        add("CFTC", "CFTC_APP_TOKEN", False, None, "https://publicreporting.cftc.gov/resource/6dca-aqww.json", str(exc))

    try:
        response = session.get("https://www.econdb.com/api/series/", params={"ticker": "RGDPUS"}, timeout=timeout)
        detail = "credential configured; public endpoint reached" if key("ECONDB_API_KEY") else "skipped: missing credential; endpoint reached"
        add("EconDB", "ECONDB_API_KEY", response.status_code < 500 and bool(key("ECONDB_API_KEY")), response.status_code, "https://www.econdb.com/api/series/", detail)
    except Exception as exc:
        add("EconDB", "ECONDB_API_KEY", False, None, "https://www.econdb.com/api/series/", str(exc))

    return pd.DataFrame(rows)
