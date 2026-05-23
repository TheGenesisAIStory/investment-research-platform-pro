"""Shared dataframe and alias helpers for notebook-first platform modules."""

from __future__ import annotations

import re
from typing import Any, Mapping

import pandas as pd


def as_df(value: Any) -> pd.DataFrame:
    """Return a defensive DataFrame copy for common notebook objects."""
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, pd.Series):
        return value.to_frame().T
    if isinstance(value, list):
        try:
            return pd.DataFrame(value)
        except Exception:
            return pd.DataFrame()
    if isinstance(value, dict):
        try:
            return pd.DataFrame([value])
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def first_available(namespace: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    """Return the first non-None object in a notebook namespace."""
    for name in names:
        value = namespace.get(name)
        if value is not None:
            return value
    return default


def normalize_name(value: Any) -> str:
    """Normalize a column/field name for resilient matching."""
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def normalize_ticker(value: Any) -> str:
    """Normalize tickers without assuming an exchange mapping."""
    if pd.isna(value):
        return ""
    return str(value).strip().upper().replace("/", "-")


def resolve_alias(df: pd.DataFrame, field: str, aliases: Mapping[str, list[str]] | None = None) -> str | None:
    """Resolve a field against exact, case-insensitive and normalized aliases."""
    aliases = aliases or {}
    candidates = [field, *aliases.get(field, [])]
    columns = list(df.columns)
    exact = {str(c): c for c in columns}
    lower = {str(c).lower(): c for c in columns}
    normalized = {normalize_name(c): c for c in columns}
    for candidate in candidates:
        if candidate in exact:
            return exact[candidate]
        if str(candidate).lower() in lower:
            return lower[str(candidate).lower()]
        key = normalize_name(candidate)
        if key in normalized:
            return normalized[key]
    return None

