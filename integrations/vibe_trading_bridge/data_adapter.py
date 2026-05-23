"""Data access adapter for Vibe-Trading integration.

The functions in this module expose a small, stable data interface that Vibe
generated analyses can call without knowing the internal notebook or artifact
layout of this repository. They prefer the local ``ml_stock_lab`` package when
available and allow callers to inject project-specific loader callables through
``DataBridgeConfig``.
"""

from __future__ import annotations

from dataclasses import dataclass
from inspect import signature
import os
from pathlib import Path
import sys
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


FrameLoader = Callable[..., pd.DataFrame]


@dataclass(slots=True)
class DataBridgeConfig:
    """Configuration for adapter calls into the host ML4T project.

    Parameters are deliberately path- and callable-oriented so the bridge can be
    reused from notebooks, CLIs, Vibe-Trading runs, and tests without relying on
    absolute local paths.
    """

    repo_root: str | Path | None = None
    output_root: str | Path | None = None
    financial_db_root: str | Path | None = None
    price_loader: FrameLoader | None = None
    feature_loader: FrameLoader | None = None
    target_loader: FrameLoader | None = None
    env_var: str = "ML4T_REPO_ROOT"


def discover_repo_root(start: str | Path | None = None, env_var: str = "ML4T_REPO_ROOT") -> Path:
    """Return the project root from an environment variable or parent search."""

    env_value = os.environ.get(env_var)
    if env_value:
        return Path(env_value).expanduser().resolve()

    current = Path(start).expanduser().resolve() if start else Path.cwd().resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists() or (candidate / "ml_stock_lab").exists():
            return candidate
    return current


def get_price_history(
    symbols: Sequence[str] | str | None,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    fields: Sequence[str] | None = None,
    config: DataBridgeConfig | None = None,
) -> pd.DataFrame:
    """Return normalized historical price data for the requested symbols.

    The returned frame is long-form and always tries to include ``date`` and
    ``ticker`` columns plus the requested price fields when present or
    derivable from common aliases such as ``price``, ``close`` or
    ``market_value``.
    """

    cfg = _as_config(config)
    wanted_symbols = _coerce_symbols(symbols)
    wanted_fields = tuple(fields or ("open", "high", "low", "close", "adj_close", "volume", "price"))

    if cfg.price_loader is not None:
        raw = _call_loader(
            cfg.price_loader,
            symbols=wanted_symbols,
            start_date=start_date,
            end_date=end_date,
            fields=wanted_fields,
            config=cfg,
        )
        panel = _normalize_panel(raw, cfg)
    else:
        panel = _load_base_panel(cfg, identifiers=wanted_symbols)

    panel = _filter_panel(panel, wanted_symbols, start_date, end_date)
    panel = _ensure_price_aliases(panel, wanted_fields)

    columns = ["date", "ticker"]
    columns.extend(field for field in wanted_fields if field in panel.columns)
    if len(columns) == 2:
        price_col = _best_price_column(panel)
        if price_col:
            columns.append(price_col)
    return panel.loc[:, _existing(columns, panel)].sort_values(_sort_columns(panel)).reset_index(drop=True)


def get_feature_matrix(
    universe: Sequence[str] | str | None,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    feature_set_name: str = "all",
    config: DataBridgeConfig | None = None,
) -> pd.DataFrame:
    """Return an ML-ready feature matrix for a universe and date range.

    ``feature_set_name`` is intentionally lightweight. The default ``"all"``
    applies the repository's basic and technical feature helpers when present.
    Use ``"basic"`` to avoid technical return features, or inject a custom
    ``feature_loader`` in ``DataBridgeConfig`` for production-grade feature
    stores.
    """

    cfg = _as_config(config)
    wanted_symbols = _coerce_symbols(universe)

    if cfg.feature_loader is not None:
        raw = _call_loader(
            cfg.feature_loader,
            universe=wanted_symbols,
            start_date=start_date,
            end_date=end_date,
            feature_set_name=feature_set_name,
            config=cfg,
        )
        panel = _normalize_panel(raw, cfg)
        return _filter_panel(panel, wanted_symbols, start_date, end_date).reset_index(drop=True)

    panel = _filter_panel(_load_base_panel(cfg, identifiers=wanted_symbols), wanted_symbols, start_date, end_date)
    if panel.empty:
        return pd.DataFrame(columns=["date", "ticker"])

    _activate_project_imports(cfg)
    try:
        from ml_stock_lab.datasets import add_basic_features, select_numeric_features
    except Exception:
        add_basic_features = None
        select_numeric_features = None

    try:
        from ml_stock_lab.features.technical import add_return_features, add_volatility_features
    except Exception:
        add_return_features = None
        add_volatility_features = None

    out = panel.copy()
    if add_basic_features is not None:
        out = add_basic_features(out)

    feature_set = feature_set_name.lower().strip()
    if feature_set in {"all", "technical", "tech"} and add_return_features is not None:
        price_col = _best_price_column(out) or "price"
        out = add_return_features(out, price_col=price_col)
        if add_volatility_features is not None:
            out = add_volatility_features(out, return_col="tech_ret_1m")

    if select_numeric_features is not None:
        feature_cols = select_numeric_features(out, target="forward_return", min_non_null=1)
    else:
        excluded = {"date"}
        feature_cols = [col for col in out.columns if col not in excluded and pd.api.types.is_numeric_dtype(out[col])]

    if feature_set in {"technical", "tech"}:
        feature_cols = [col for col in feature_cols if col.startswith("tech_")]
    elif feature_set == "basic":
        feature_cols = [col for col in feature_cols if not col.startswith("tech_")]

    columns = ["date", "ticker", *feature_cols]
    return out.loc[:, _existing(columns, out)].sort_values(_sort_columns(out)).reset_index(drop=True)


def get_target_labels(
    universe: Sequence[str] | str | None,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    label_spec: Mapping[str, Any] | str | None = None,
    config: DataBridgeConfig | None = None,
) -> pd.DataFrame:
    """Return target labels for supervised learning or strategy diagnostics.

    ``label_spec`` may be a string column name or a mapping with
    ``target_col``, ``price_col`` and ``horizon`` keys. If the requested target
    column is absent, the adapter creates forward returns using the repository's
    ``make_forward_returns`` helper when available.
    """

    cfg = _as_config(config)
    wanted_symbols = _coerce_symbols(universe)
    spec = _coerce_label_spec(label_spec)

    if cfg.target_loader is not None:
        raw = _call_loader(
            cfg.target_loader,
            universe=wanted_symbols,
            start_date=start_date,
            end_date=end_date,
            label_spec=spec,
            config=cfg,
        )
        panel = _normalize_panel(raw, cfg)
        return _filter_panel(panel, wanted_symbols, start_date, end_date).reset_index(drop=True)

    panel = _filter_panel(_load_base_panel(cfg, identifiers=wanted_symbols), wanted_symbols, start_date, end_date)
    target_col = str(spec.get("target_col") or spec.get("name") or "forward_return")

    if target_col not in panel.columns:
        _activate_project_imports(cfg)
        try:
            from ml_stock_lab.datasets import make_forward_returns
        except Exception:
            make_forward_returns = None
        price_col = str(spec.get("price_col") or _best_price_column(panel) or "price")
        horizon = int(spec.get("horizon", 1))
        if make_forward_returns is not None and price_col in panel.columns:
            panel = make_forward_returns(panel, price_col=price_col, horizon=horizon)
            target_col = "forward_return"

    columns = ["date", "ticker"]
    if target_col in panel.columns:
        panel = panel.copy()
        panel["label"] = pd.to_numeric(panel[target_col], errors="coerce")
        columns.extend([target_col, "label"] if target_col != "label" else ["label"])
    return panel.loc[:, _existing(columns, panel)].sort_values(_sort_columns(panel)).reset_index(drop=True)


def _as_config(config: DataBridgeConfig | None) -> DataBridgeConfig:
    return config if config is not None else DataBridgeConfig()


def _activate_project_imports(config: DataBridgeConfig) -> None:
    root = discover_repo_root(config.repo_root, env_var=config.env_var)
    candidates = [
        root,
        root / "research_platform_definitive" / "src",
        root / "research_platform_definitive" / "portfolio_analysis" / "src",
    ]
    for path in reversed(candidates):
        value = str(path)
        if path.exists() and value not in sys.path:
            sys.path.insert(0, value)


def _load_base_panel(config: DataBridgeConfig, identifiers: Iterable[str] | None = None) -> pd.DataFrame:
    _activate_project_imports(config)
    try:
        from ml_stock_lab.datasets import load_financial_db_panel
    except Exception as exc:
        raise RuntimeError("Could not import ml_stock_lab.datasets.load_financial_db_panel") from exc

    panel = _call_loader(
        load_financial_db_panel,
        output_root=config.output_root,
        financial_db_root=config.financial_db_root,
        identifiers=list(identifiers or []),
    )
    return _normalize_panel(panel, config)


def _normalize_panel(frame: pd.DataFrame, config: DataBridgeConfig) -> pd.DataFrame:
    if frame is None:
        return pd.DataFrame(columns=["date", "ticker"])
    _activate_project_imports(config)
    try:
        from ml_stock_lab.datasets import normalize_panel
    except Exception:
        normalize_panel = None
    out = normalize_panel(frame) if normalize_panel is not None else frame.copy()
    if "ticker" not in out.columns and "symbol" in out.columns:
        out["ticker"] = out["symbol"]
    if "date" not in out.columns:
        for candidate in ("timestamp", "datetime", "as_of_date", "report_date"):
            if candidate in out.columns:
                out["date"] = out[candidate]
                break
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if "ticker" in out.columns:
        out["ticker"] = out["ticker"].astype(str).str.upper()
    return out


def _filter_panel(
    panel: pd.DataFrame,
    symbols: Sequence[str] | None,
    start_date: str | pd.Timestamp | None,
    end_date: str | pd.Timestamp | None,
) -> pd.DataFrame:
    if panel is None or panel.empty:
        return pd.DataFrame(columns=["date", "ticker"])
    out = panel.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        if start_date is not None:
            out = out[out["date"] >= pd.Timestamp(start_date)]
        if end_date is not None:
            out = out[out["date"] <= pd.Timestamp(end_date)]
    if symbols and "ticker" in out.columns:
        wanted = {symbol.upper() for symbol in symbols}
        out = out[out["ticker"].astype(str).str.upper().isin(wanted)]
    return out


def _coerce_symbols(symbols: Sequence[str] | str | None) -> list[str] | None:
    if symbols is None:
        return None
    if isinstance(symbols, str):
        return [part.strip().upper() for part in symbols.split(",") if part.strip()]
    return [str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()]


def _coerce_label_spec(label_spec: Mapping[str, Any] | str | None) -> dict[str, Any]:
    if label_spec is None:
        return {"target_col": "forward_return", "horizon": 1}
    if isinstance(label_spec, str):
        return {"target_col": label_spec, "horizon": 1}
    return dict(label_spec)


def _call_loader(loader: FrameLoader, **kwargs: Any) -> pd.DataFrame:
    try:
        params = signature(loader).parameters
    except (TypeError, ValueError):
        return loader(**kwargs)
    accepts_kwargs = any(param.kind.name == "VAR_KEYWORD" for param in params.values())
    filtered = kwargs if accepts_kwargs else {key: value for key, value in kwargs.items() if key in params}
    result = loader(**filtered)
    if not isinstance(result, pd.DataFrame):
        raise TypeError(f"Loader {loader!r} returned {type(result).__name__}, expected pandas.DataFrame")
    return result


def _ensure_price_aliases(panel: pd.DataFrame, fields: Sequence[str]) -> pd.DataFrame:
    out = panel.copy()
    alias_map = {
        "open": ("open", "Open"),
        "high": ("high", "High"),
        "low": ("low", "Low"),
        "close": ("close", "Close", "adj_close", "price", "mkt_price", "market_value"),
        "adj_close": ("adj_close", "adjusted_close", "close", "price", "mkt_price"),
        "price": ("price", "close", "adj_close", "mkt_price", "market_value"),
        "volume": ("volume", "Volume", "vol"),
    }
    lower_lookup = {str(col).lower(): col for col in out.columns}
    for field in fields:
        if field in out.columns:
            continue
        for alias in alias_map.get(field, (field,)):
            actual = lower_lookup.get(alias.lower())
            if actual is not None:
                out[field] = pd.to_numeric(out[actual], errors="coerce")
                break
    return out


def _best_price_column(panel: pd.DataFrame) -> str | None:
    for column in ("adj_close", "close", "price", "mkt_price", "market_value"):
        if column in panel.columns and pd.to_numeric(panel[column], errors="coerce").notna().any():
            return column
    numeric_cols = [col for col in panel.columns if pd.api.types.is_numeric_dtype(panel[col])]
    return numeric_cols[0] if numeric_cols else None


def _existing(columns: Sequence[str], frame: pd.DataFrame) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for column in columns:
        if column in frame.columns and column not in seen:
            out.append(column)
            seen.add(column)
    return out


def _sort_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in ("date", "ticker") if column in frame.columns]

