"""Configuration helpers for the QuantDinger bridge."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Mapping
from urllib.parse import quote_plus


@dataclass(frozen=True, slots=True)
class DatabaseSettings:
    """Connection details for QuantDinger metadata and market data storage."""

    url: str
    host: str = "postgres"
    port: int = 5432
    dbname: str = "quantdinger"
    user: str = "quantdinger"
    password: str = "quantdinger123"
    echo: bool = False


@dataclass(frozen=True, slots=True)
class RedisSettings:
    """Redis connection settings shared with the QuantDinger stack."""

    url: str
    host: str = "redis"
    port: int = 6379
    db: int = 0


@dataclass(frozen=True, slots=True)
class ApiSettings:
    """QuantDinger REST API settings."""

    base_url: str = "http://backend:5000/api"
    token: str | None = None
    agent_token: str | None = None
    verify_auth: bool = False
    auth_verify_url: str = "http://backend:5000/api/auth/info"
    timeout_seconds: int = 30
    retry_total: int = 3
    retry_backoff_seconds: float = 0.3


@dataclass(frozen=True, slots=True)
class ServiceSettings:
    """Runtime settings for the ML sidecar service."""

    host: str = "0.0.0.0"
    port: int = 8000
    service_auth_token: str | None = None
    repo_root: Path = Path("/workspace/ml4t")
    default_market: str = "USStock"
    default_timeframe: str = "1D"
    api_prefix: str = "/api/v1/ml"
    cache_ttl_seconds: int = 30
    log_level: str = "INFO"


@dataclass(frozen=True, slots=True)
class QuantDingerBridgeConfig:
    """Central bridge configuration loaded from environment variables."""

    database: DatabaseSettings
    redis: RedisSettings
    api: ApiSettings
    service: ServiceSettings


def load_config(env: Mapping[str, str] | None = None) -> QuantDingerBridgeConfig:
    """Load bridge settings from environment variables.

    The variables intentionally mirror QuantDinger's Compose defaults. Prefix
    with ``QUANTDINGER_`` for bridge-specific overrides, or rely on
    QuantDinger's existing ``DATABASE_URL``/``REDIS_*`` values inside Docker.
    """

    source = env or os.environ
    db_host = source.get("QUANTDINGER_DB_HOST") or source.get("POSTGRES_HOST") or "postgres"
    db_port = _to_int(source.get("QUANTDINGER_DB_PORT") or source.get("POSTGRES_PORT"), 5432)
    dbname = source.get("QUANTDINGER_DB_NAME") or source.get("POSTGRES_DB") or "quantdinger"
    db_user = source.get("QUANTDINGER_DB_USER") or source.get("POSTGRES_USER") or "quantdinger"
    db_password = source.get("QUANTDINGER_DB_PASSWORD") or source.get("POSTGRES_PASSWORD") or "quantdinger123"
    db_url = (
        source.get("QUANTDINGER_DATABASE_URL")
        or source.get("DATABASE_URL")
        or build_database_url(db_host, db_port, dbname, db_user, db_password)
    )

    redis_host = source.get("QUANTDINGER_REDIS_HOST") or source.get("REDIS_HOST") or "redis"
    redis_port = _to_int(source.get("QUANTDINGER_REDIS_PORT") or source.get("REDIS_PORT"), 6379)
    redis_db = _to_int(source.get("QUANTDINGER_REDIS_DB") or source.get("REDIS_DB"), 0)
    redis_url = source.get("QUANTDINGER_REDIS_URL") or source.get("REDIS_URL") or f"redis://{redis_host}:{redis_port}/{redis_db}"

    api_base = (source.get("QUANTDINGER_API_BASE_URL") or "http://backend:5000/api").rstrip("/")
    repo_root = Path(source.get("ML4T_REPO_ROOT") or source.get("QUANTDINGER_ML_REPO_ROOT") or "/workspace/ml4t")

    return QuantDingerBridgeConfig(
        database=DatabaseSettings(
            url=db_url,
            host=db_host,
            port=db_port,
            dbname=dbname,
            user=db_user,
            password=db_password,
            echo=_to_bool(source.get("QUANTDINGER_DB_ECHO"), False),
        ),
        redis=RedisSettings(url=redis_url, host=redis_host, port=redis_port, db=redis_db),
        api=ApiSettings(
            base_url=api_base,
            token=source.get("QUANTDINGER_API_TOKEN") or source.get("QUANTDINGER_USER_JWT"),
            agent_token=source.get("QUANTDINGER_AGENT_TOKEN"),
            verify_auth=_to_bool(source.get("QUANTDINGER_VERIFY_AUTH"), False),
            auth_verify_url=source.get("QUANTDINGER_AUTH_VERIFY_URL") or f"{api_base}/auth/info",
            timeout_seconds=_to_int(source.get("QUANTDINGER_API_TIMEOUT"), 30),
            retry_total=_to_int(source.get("QUANTDINGER_API_RETRY_TOTAL"), 3),
            retry_backoff_seconds=_to_float(source.get("QUANTDINGER_API_RETRY_BACKOFF_SECONDS"), 0.3),
        ),
        service=ServiceSettings(
            host=source.get("ML_SERVICE_HOST", "0.0.0.0"),
            port=_to_int(source.get("ML_SERVICE_PORT"), 8000),
            service_auth_token=source.get("ML_SERVICE_AUTH_TOKEN"),
            repo_root=repo_root,
            default_market=source.get("QUANTDINGER_DEFAULT_MARKET", "USStock"),
            default_timeframe=source.get("QUANTDINGER_DEFAULT_TIMEFRAME", "1D"),
            api_prefix=_normalize_prefix(source.get("ML_SERVICE_API_PREFIX", "/api/v1/ml")),
            cache_ttl_seconds=_to_int(source.get("ML_SERVICE_CACHE_TTL_SECONDS"), 30),
            log_level=source.get("ML_SERVICE_LOG_LEVEL", "INFO").upper(),
        ),
    )


def build_database_url(host: str, port: int, dbname: str, user: str, password: str) -> str:
    """Build a SQLAlchemy-compatible PostgreSQL connection URL."""

    return f"postgresql+psycopg2://{quote_plus(user)}:{quote_plus(password)}@{host}:{port}/{quote_plus(dbname)}"


def _to_bool(value: str | None, default: bool) -> bool:
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _to_int(value: str | None, default: int) -> int:
    try:
        return int(value) if value not in (None, "") else default
    except (TypeError, ValueError):
        return default


def _to_float(value: str | None, default: float) -> float:
    try:
        return float(value) if value not in (None, "") else default
    except (TypeError, ValueError):
        return default


def _normalize_prefix(value: str) -> str:
    prefix = (value or "/api/v1/ml").strip()
    if not prefix.startswith("/"):
        prefix = f"/{prefix}"
    return prefix.rstrip("/") or "/api/v1/ml"
