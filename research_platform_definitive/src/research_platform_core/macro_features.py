"""Macro feature engineering for ECB and Banca d'Italia official data.

The functions in this module convert official macro/statistical releases into
ML-ready panels while keeping a lightweight lineage record in
``DataFrame.attrs["feature_metadata"]``. The defaults target:

* inflation nowcasting: ECB HICP plus Banca d'Italia credit/deposit indicators;
* credit risk: ECB policy/monetary aggregates plus Banca d'Italia credit and
  public finance indicators.

Use these panels as inputs to scikit-learn, PyTorch/Lightning or regulatory
model-development notebooks after freezing the source metadata with the saved
sidecar JSON files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data_platform import utc_now
from .loaders.bditalia_client import BancaDItaliaClient
from .loaders.ecb_client import EcbClient


DEFAULT_TIMEZONE = "Europe/Rome"


def _clean_feature_name(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = "".join(ch if ch.isalnum() else "_" for ch in text)
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_") or "feature"


def normalize_datetime_column(df: pd.DataFrame, date_col: str | None = None, timezone: str = DEFAULT_TIMEZONE) -> pd.Series:
    """Return a timezone-normalized timestamp series for official macro frames."""
    if df.empty:
        return pd.Series(dtype="datetime64[ns]")
    if date_col is None:
        date_col = next((c for c in ["date", "TIME_PERIOD", "time_period", "DATA", "data"] if c in df.columns), None)
    if date_col is None:
        return pd.Series(pd.NaT, index=df.index)
    parsed = pd.to_datetime(df[date_col], errors="coerce")
    try:
        if getattr(parsed.dt, "tz", None) is not None:
            parsed = parsed.dt.tz_convert(timezone).dt.tz_localize(None)
    except Exception:
        pass
    return parsed


def to_month_end(series: pd.Series) -> pd.Series:
    """Convert timestamps to month-end labels for monthly ML panels."""
    parsed = pd.to_datetime(series, errors="coerce")
    return parsed.dt.to_period("M").dt.to_timestamp("M")


def _value_column(df: pd.DataFrame) -> str | None:
    for candidate in ["value", "OBS_VALUE", "obs_value", "VALORE", "valore"]:
        if candidate in df.columns:
            return candidate
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return numeric_cols[0] if numeric_cols else None


def series_to_wide(
    df: pd.DataFrame,
    prefix: str,
    date_col: str | None = None,
    value_col: str | None = None,
    name_col: str | None = None,
    timezone: str = DEFAULT_TIMEZONE,
) -> pd.DataFrame:
    """Convert long official observations into a date-indexed wide feature frame."""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out["date"] = normalize_datetime_column(out, date_col=date_col, timezone=timezone)
    out = out.dropna(subset=["date"])
    if out.empty:
        return pd.DataFrame()
    out["date"] = to_month_end(out["date"])
    value_col = value_col or _value_column(out)
    if value_col is None:
        return pd.DataFrame()
    out[value_col] = pd.to_numeric(out[value_col], errors="coerce")
    if name_col is None:
        name_col = next((c for c in ["dataset", "series_code", "series_key", "item_code", "KEY"] if c in out.columns), None)
    if name_col is None:
        feature_name = prefix
        wide = out.groupby("date", as_index=False)[value_col].mean().rename(columns={value_col: feature_name})
        return wide
    out["feature_name"] = prefix + "_" + out[name_col].map(_clean_feature_name)
    wide = out.pivot_table(index="date", columns="feature_name", values=value_col, aggfunc="mean").reset_index()
    wide.columns.name = None
    return wide.sort_values("date").reset_index(drop=True)


def align_macro_frames(
    frames: list[pd.DataFrame],
    frequency: str = "monthly",
    fill_method: str | None = "ffill",
    max_fill_periods: int | None = 3,
) -> pd.DataFrame:
    """Outer-join macro frames and align to a monthly or quarterly calendar."""
    usable = [frame.copy() for frame in frames if frame is not None and not frame.empty]
    if not usable:
        return pd.DataFrame()
    panel = usable[0]
    for frame in usable[1:]:
        panel = panel.merge(frame, on="date", how="outer")
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    panel = panel.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last")
    panel = panel.set_index("date")
    rule = "QE" if frequency.lower().startswith("q") else "ME"
    panel = panel.resample(rule).last()
    if fill_method:
        panel = getattr(panel, fill_method)(limit=max_fill_periods)
    return panel.reset_index()


def add_standard_transformations(
    panel: pd.DataFrame,
    yoy: bool = True,
    log_diff: bool = True,
    zscore: bool = False,
    exclude: set[str] | None = None,
) -> pd.DataFrame:
    """Add standard macro transformations with deterministic column names."""
    if panel.empty:
        return panel.copy()
    exclude = exclude or {"date"}
    out = panel.copy()
    numeric_cols = [c for c in out.columns if c not in exclude and pd.api.types.is_numeric_dtype(out[c])]
    lineage: dict[str, dict[str, Any]] = {}
    for col in numeric_cols:
        if yoy:
            new_col = f"{col}_yoy"
            out[new_col] = out[col].pct_change(12)
            lineage[new_col] = {"source_column": col, "transform": "pct_change", "periods": 12}
        if log_diff:
            new_col = f"{col}_logdiff"
            positive = out[col].where(out[col] > 0)
            out[new_col] = np.log(positive).diff()
            lineage[new_col] = {"source_column": col, "transform": "log_diff", "periods": 1}
        if zscore:
            new_col = f"{col}_z"
            std = out[col].std(ddof=0)
            out[new_col] = (out[col] - out[col].mean()) / std if std and not np.isnan(std) else np.nan
            lineage[new_col] = {"source_column": col, "transform": "zscore", "fit_scope": "full_panel"}
    metadata = dict(out.attrs.get("feature_metadata", {}))
    metadata.setdefault("transform_lineage", {}).update(lineage)
    out.attrs["feature_metadata"] = metadata
    return out


def _attach_metadata(panel: pd.DataFrame, use_case: str, sources: list[str], timezone: str, fill_method: str | None) -> pd.DataFrame:
    panel.attrs["feature_metadata"] = {
        "use_case": use_case,
        "sources": sources,
        "timezone_policy": timezone,
        "calendar_policy": "month_end_alignment",
        "missing_policy": fill_method or "none",
        "created_at": utc_now(),
        "regulatory_note": "Keep source_url/source metadata and sidecar JSON files with model artifacts for auditability.",
    }
    return panel


def build_inflation_nowcasting_dataset(
    ecb_hicp: pd.DataFrame | None = None,
    bdi_credit: pd.DataFrame | None = None,
    bdi_deposits: pd.DataFrame | None = None,
    ecb_client: EcbClient | None = None,
    bditalia_client: BancaDItaliaClient | None = None,
    start: str | None = None,
    end: str | None = None,
    fill_method: str | None = "ffill",
    add_transforms: bool = True,
    timezone: str = DEFAULT_TIMEZONE,
) -> pd.DataFrame:
    """Build a monthly inflation nowcasting panel from ECB HICP and BdI indicators."""
    ecb_client = ecb_client or (None if ecb_hicp is not None else EcbClient())
    bditalia_client = bditalia_client or (None if bdi_credit is not None and bdi_deposits is not None else BancaDItaliaClient())
    if ecb_hicp is None and ecb_client is not None:
        ecb_hicp = ecb_client.get_hicp_euro_area(start=start, end=end)
    if bdi_credit is None and bditalia_client is not None:
        bdi_credit = bditalia_client.get_credit_series(start=start, end=end)
    if bdi_deposits is None and bditalia_client is not None:
        bdi_deposits = bditalia_client.get_deposits_series(start=start, end=end)
    frames = [
        series_to_wide(ecb_hicp if ecb_hicp is not None else pd.DataFrame(), "ecb_hicp", timezone=timezone),
        series_to_wide(bdi_credit if bdi_credit is not None else pd.DataFrame(), "bdi_credit", timezone=timezone),
        series_to_wide(bdi_deposits if bdi_deposits is not None else pd.DataFrame(), "bdi_deposits", timezone=timezone),
    ]
    panel = align_macro_frames(frames, frequency="monthly", fill_method=fill_method)
    panel = _attach_metadata(panel, "inflation_nowcasting", ["ecb_hicp", "bditalia_credit", "bditalia_deposits"], timezone, fill_method)
    if add_transforms:
        panel = add_standard_transformations(panel, yoy=True, log_diff=True)
    return panel


def build_credit_risk_macro_dataset(
    ecb_policy_rates: pd.DataFrame | None = None,
    ecb_mfi: pd.DataFrame | None = None,
    bdi_credit: pd.DataFrame | None = None,
    bdi_public_debt: pd.DataFrame | None = None,
    ecb_client: EcbClient | None = None,
    bditalia_client: BancaDItaliaClient | None = None,
    start: str | None = None,
    end: str | None = None,
    fill_method: str | None = "ffill",
    add_transforms: bool = True,
    timezone: str = DEFAULT_TIMEZONE,
) -> pd.DataFrame:
    """Build a monthly macro feature panel for PD/LGD and credit risk models."""
    ecb_client = ecb_client or (None if ecb_policy_rates is not None and ecb_mfi is not None else EcbClient())
    bditalia_client = bditalia_client or (None if bdi_credit is not None and bdi_public_debt is not None else BancaDItaliaClient())
    if ecb_policy_rates is None and ecb_client is not None:
        ecb_policy_rates = ecb_client.get_policy_rates(start=start, end=end)
    if ecb_mfi is None and ecb_client is not None:
        ecb_mfi = ecb_client.get_mfi_balance_sheet(start=start, end=end)
    if bdi_credit is None and bditalia_client is not None:
        bdi_credit = bditalia_client.get_credit_series(start=start, end=end)
    if bdi_public_debt is None and bditalia_client is not None:
        bdi_public_debt = bditalia_client.get_public_debt_series(start=start, end=end)
    frames = [
        series_to_wide(ecb_policy_rates if ecb_policy_rates is not None else pd.DataFrame(), "ecb_rate", timezone=timezone),
        series_to_wide(ecb_mfi if ecb_mfi is not None else pd.DataFrame(), "ecb_mfi", timezone=timezone),
        series_to_wide(bdi_credit if bdi_credit is not None else pd.DataFrame(), "bdi_credit", timezone=timezone),
        series_to_wide(bdi_public_debt if bdi_public_debt is not None else pd.DataFrame(), "bdi_public_debt", timezone=timezone),
    ]
    panel = align_macro_frames(frames, frequency="monthly", fill_method=fill_method)
    panel = _attach_metadata(panel, "credit_risk_macro", ["ecb_policy_rates", "ecb_mfi_bsi", "bditalia_credit", "bditalia_public_debt"], timezone, fill_method)
    if add_transforms:
        panel = add_standard_transformations(panel, yoy=True, log_diff=True)
    return panel


def save_macro_feature_panel(panel: pd.DataFrame, path: str | Path, fmt: str = "csv") -> Path:
    """Save a feature panel and its lineage metadata sidecar."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if fmt.lower() == "parquet" or target.suffix.lower() == ".parquet":
        target = target.with_suffix(".parquet")
        panel.to_parquet(target, index=False)
    else:
        target = target.with_suffix(".csv")
        panel.to_csv(target, index=False)
    metadata = panel.attrs.get("feature_metadata", {})
    target.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return target
