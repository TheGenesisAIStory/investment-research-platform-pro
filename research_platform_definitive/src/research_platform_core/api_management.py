"""Secret-safe API governance helpers for the research platform."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from .data_platform import (
    discover_financial_database_root,
    load_catalog_tables,
    provider_fallback_plan,
    utc_now,
)


DEFAULT_API_ROOT_CANDIDATES = [
    Path("/content/drive/MyDrive/Database Finanziario/API"),
    Path.home() / "Library/CloudStorage/GoogleDrive-sfn.gns@gmail.com/Il mio Drive/Database Finanziario" / "API",
]

API_ROOT_ENV_KEYS = ["API_CREDENTIALS_ROOT", "FINANCIAL_API_ROOT", "RESEARCH_PLATFORM_API_ROOT"]
SECRET_ENV_HINTS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "APP_TOKEN")


@dataclass(frozen=True)
class ApiControlRoots:
    financial_db: Path
    api_root: Path
    api_available: bool
    api_source: str


def discover_api_credentials_root(
    financial_db_root: Path | str | None = None,
    extra_candidates: Iterable[Path | str] | None = None,
) -> tuple[Path, str, bool]:
    """Resolve the private API folder without reading any secret values."""
    for key in API_ROOT_ENV_KEYS:
        value = os.environ.get(key)
        if value:
            path = Path(value).expanduser()
            return path, f"env:{key}", path.exists()

    candidates: list[Path] = []
    if financial_db_root:
        candidates.append(Path(financial_db_root).expanduser() / "API")
    candidates.extend(Path(p).expanduser() for p in (extra_candidates or []))
    candidates.extend(DEFAULT_API_ROOT_CANDIDATES)

    seen: set[str] = set()
    unique_candidates = []
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(candidate)

    for path in unique_candidates:
        if path.exists():
            return path, "discovered", True
    fallback = unique_candidates[0] if unique_candidates else Path.cwd() / "API"
    return fallback, "fallback_missing", False


def resolve_api_control_roots(
    financial_db_root: Path | str | None = None,
    api_root: Path | str | None = None,
) -> ApiControlRoots:
    if financial_db_root is None:
        financial_db, _, _ = discover_financial_database_root()
    else:
        financial_db = Path(financial_db_root).expanduser()
    if api_root is None:
        resolved_api, source, available = discover_api_credentials_root(financial_db)
    else:
        resolved_api = Path(api_root).expanduser()
        source = "explicit"
        available = resolved_api.exists()
    return ApiControlRoots(financial_db, resolved_api, available, source)


def mask_secret(value: object, show: int = 3) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if len(text) <= show + 2:
        return "***"
    return f"{text[:show]}...{text[-2:]} ({len(text)} chars)"


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "configured", "ok"}


def _normalise_env_var(value: object) -> str:
    return re.sub(r"[^A-Z0-9_]", "", str(value or "").strip().upper())


def load_api_folder_inventory(api_root: Path | str) -> pd.DataFrame:
    """Inventory API-folder files without opening credential payloads."""
    root = Path(api_root).expanduser()
    if not root.exists():
        return pd.DataFrame([{"path": str(root), "exists": False, "updated_at": utc_now()}])

    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        stat = path.stat()
        modified = pd.Timestamp(stat.st_mtime, unit="s", tz="UTC")
        rows.append(
            {
                "relative_path": str(path.relative_to(root)),
                "file": path.name,
                "suffix": path.suffix.lower(),
                "size_kb": round(stat.st_size / 1024, 2),
                "modified_utc": modified.isoformat(),
                "age_hours": round((pd.Timestamp.now(tz="UTC") - modified).total_seconds() / 3600, 2),
                "exists": True,
                "secret_policy": "private_drive_file_do_not_commit",
            }
        )
    return pd.DataFrame(rows)


def credential_master_sections(api_root: Path | str) -> pd.DataFrame:
    """Read only markdown headings from the private credential master."""
    path = Path(api_root).expanduser() / "api_credentials_master.md"
    if not path.exists():
        return pd.DataFrame([{"section": "api_credentials_master.md missing", "level": 0, "path": str(path)}])

    rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped.startswith("#"):
            continue
        level = len(stripped) - len(stripped.lstrip("#"))
        section = stripped[level:].strip()
        if section:
            rows.append({"section": section, "level": level, "path": str(path)})
    return pd.DataFrame(rows)


def accepted_api_env_vars(provider_registry: pd.DataFrame) -> list[str]:
    if provider_registry.empty or "env_var" not in provider_registry.columns:
        return []
    env_vars = provider_registry["env_var"].dropna().map(_normalise_env_var)
    return sorted({value for value in env_vars if value})


def parse_env_text(text: str, allowed_env_vars: Iterable[str] | None = None) -> dict[str, str]:
    """Parse .env-style text and return allowed secret values for session use."""
    allowed = {_normalise_env_var(v) for v in (allowed_env_vars or []) if _normalise_env_var(v)}
    out: dict[str, str] = {}
    for raw in str(text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env_var = _normalise_env_var(key)
        secret = value.strip().strip('"').strip("'")
        if not env_var or not secret:
            continue
        if allowed and env_var not in allowed:
            continue
        if not allowed and not any(hint in env_var for hint in SECRET_ENV_HINTS):
            continue
        out[env_var] = secret
    return out


def apply_env_text_to_session(text: str, allowed_env_vars: Iterable[str] | None = None) -> pd.DataFrame:
    parsed = parse_env_text(text, allowed_env_vars=allowed_env_vars)
    rows = []
    for env_var, secret in parsed.items():
        os.environ[env_var] = secret
        rows.append({"env_var": env_var, "configured": True, "credential_preview": mask_secret(secret), "source": "session_env"})
    return pd.DataFrame(rows)


def build_credential_status(provider_registry: pd.DataFrame, catalog_credentials: pd.DataFrame | None = None) -> pd.DataFrame:
    catalog_credentials = catalog_credentials if catalog_credentials is not None else pd.DataFrame()
    rows = []
    for _, provider in provider_registry.iterrows() if not provider_registry.empty else []:
        env_var = _normalise_env_var(provider.get("env_var", ""))
        if not env_var:
            continue
        env_value = os.environ.get(env_var, "")
        catalog_row = pd.DataFrame()
        if not catalog_credentials.empty and "env_var" in catalog_credentials.columns:
            catalog_row = catalog_credentials[catalog_credentials["env_var"].astype(str).str.upper().eq(env_var)]
        catalog_configured = False
        catalog_preview = ""
        if not catalog_row.empty:
            first = catalog_row.iloc[0]
            catalog_configured = _as_bool(first.get("configured", False))
            catalog_preview = str(first.get("credential_preview", "") or "")
        configured_env = bool(env_value)
        rows.append(
            {
                "name": provider.get("name", ""),
                "category": provider.get("category", ""),
                "env_var": env_var,
                "configured": bool(configured_env or catalog_configured),
                "configured_env": configured_env,
                "configured_catalog": catalog_configured,
                "credential_preview": mask_secret(env_value) if configured_env else catalog_preview,
                "openbb_credential": provider.get("openbb_credential", ""),
                "data_type": provider.get("data_type", ""),
                "free_tier": provider.get("free_tier", ""),
                "source": "session_env" if configured_env else "catalog_masked" if catalog_configured else "missing",
            }
        )
    return pd.DataFrame(rows)


def api_control_status(
    financial_db_root: Path | str | None = None,
    api_root: Path | str | None = None,
) -> dict[str, pd.DataFrame]:
    roots = resolve_api_control_roots(financial_db_root=financial_db_root, api_root=api_root)
    catalog = load_catalog_tables(roots.financial_db)
    providers = catalog.get("api_providers", pd.DataFrame())
    catalog_credentials = catalog.get("credential_status_masked", pd.DataFrame())
    credentials = build_credential_status(providers, catalog_credentials)
    health = catalog.get("api_provider_health_checks", pd.DataFrame())
    fallback = provider_fallback_plan(roots.financial_db)
    api_files = load_api_folder_inventory(roots.api_root)
    sections = credential_master_sections(roots.api_root)

    summary = pd.DataFrame(
        [
            {
                "generated_at": utc_now(),
                "financial_db_root": str(roots.financial_db),
                "api_root": str(roots.api_root),
                "api_root_available": roots.api_available,
                "api_root_source": roots.api_source,
                "providers": len(providers),
                "configured_credentials": int(credentials["configured"].sum()) if "configured" in credentials else 0,
                "reachable_providers": int(health["reachable"].sum()) if "reachable" in health else 0,
                "api_folder_files": int(len(api_files[api_files.get("exists", False).eq(True)])) if not api_files.empty and "exists" in api_files else 0,
            }
        ]
    )
    return {
        "summary": summary,
        "api_providers": providers,
        "credential_status": credentials,
        "api_provider_health_checks": health,
        "provider_fallback_plan": fallback,
        "api_folder_inventory": api_files,
        "credential_master_sections": sections,
    }


def build_env_template(provider_registry: pd.DataFrame) -> str:
    lines = []
    for env_var in accepted_api_env_vars(provider_registry):
        lines.append(f"{env_var}=")
    return "\n".join(lines) + ("\n" if lines else "")


def write_api_control_status(
    financial_db_root: Path | str,
    output_root: Path | str,
    api_root: Path | str | None = None,
) -> dict[str, Path]:
    status = api_control_status(financial_db_root=financial_db_root, api_root=api_root)
    output_root = Path(output_root).expanduser()
    table_dir = output_root / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    for name, df in status.items():
        path = table_dir / f"DataAPI_{name}.csv"
        df.to_csv(path, index=False)
        paths[name] = path

    summary = status["summary"].iloc[0].to_dict() if not status["summary"].empty else {}
    contract = {
        "generated_at": utc_now(),
        "financial_db_root": str(financial_db_root),
        "api_root": summary.get("api_root", str(api_root or "")),
        "secret_policy": "No secret values are written to this contract; only masked status is exported.",
        "tables": {name: str(path) for name, path in paths.items()},
        "summary": summary,
        "providers": status["credential_status"].to_dict(orient="records"),
    }
    api_dir = output_root / "api_contracts"
    api_dir.mkdir(parents=True, exist_ok=True)
    contract_path = api_dir / "data_api_control_contract.json"
    contract_path.write_text(json.dumps(contract, indent=2, default=str), encoding="utf-8")
    paths["api_contract"] = contract_path
    return paths
