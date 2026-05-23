"""Fundamental feature helpers for date/ticker stock panels."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml_stock_lab.datasets import normalize_panel


def lag_fundamentals(
    df: pd.DataFrame,
    lag_months: int = 3,
    date_col: str = "date",
    ticker_col: str = "ticker",
    prefix: str = "fund_",
    suffix: str = "_lag",
) -> pd.DataFrame:
    """Add lagged copies of fundamental columns.

    The function assumes one row per ticker/date observation. In monthly
    panels, `lag_months=3` corresponds to shifting fundamentals by three rows
    within each ticker. Existing lag columns are preserved.
    """
    out = normalize_panel(df)
    if out.empty:
        return out

    sort_cols = [col for col in [ticker_col, date_col] if col in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).copy()

    fundamental_cols = [
        col
        for col in out.columns
        if col.startswith(prefix)
        and not col.endswith(suffix)
        and pd.api.types.is_numeric_dtype(out[col])
    ]
    if not fundamental_cols:
        return out

    periods = max(int(lag_months), 0)
    for col in fundamental_cols:
        lag_col = f"{col}{suffix}"
        if lag_col in out.columns:
            continue
        if ticker_col in out.columns:
            out[lag_col] = out.groupby(ticker_col)[col].shift(periods)
        else:
            out[lag_col] = out[col].shift(periods)
    return out


def add_log_mcap_target(
    df: pd.DataFrame,
    mkt_cap_col: str = "mkt_market_cap",
    out_col: str = "target_log_mcap",
) -> pd.DataFrame:
    """Add a log market-cap target column from a positive market-cap field."""
    out = normalize_panel(df)
    if mkt_cap_col not in out.columns and "market_value" in out.columns:
        mkt_cap_col = "market_value"
    values = pd.to_numeric(out.get(mkt_cap_col, pd.Series(index=out.index, dtype=float)), errors="coerce")
    out[out_col] = np.log(values.where(values > 0))
    return out


def add_forward_returns(
    df: pd.DataFrame,
    price_col: str = "mkt_price",
    horizon_months: int = 1,
    out_col: str = "target_ret_1m_fwd",
    date_col: str = "date",
    ticker_col: str = "ticker",
) -> pd.DataFrame:
    """Add forward returns by ticker from a price column."""
    out = normalize_panel(df)
    if price_col not in out.columns and "price" in out.columns:
        price_col = "price"
    if price_col not in out.columns:
        out[out_col] = np.nan
        return out

    sort_cols = [col for col in [ticker_col, date_col] if col in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).copy()
    prices = pd.to_numeric(out[price_col], errors="coerce")
    if ticker_col in out.columns:
        out[out_col] = out.groupby(ticker_col)[price_col].transform(
            lambda s: pd.to_numeric(s, errors="coerce").shift(-horizon_months) / pd.to_numeric(s, errors="coerce") - 1
        )
    else:
        out[out_col] = prices.shift(-horizon_months) / prices - 1
    return out


def filter_valid_rows(df: pd.DataFrame, required_cols: list[str]) -> pd.DataFrame:
    """Keep rows with finite values in all required columns that exist."""
    out = normalize_panel(df).replace([np.inf, -np.inf], np.nan)
    cols = [col for col in required_cols if col in out.columns]
    if not cols:
        return out
    return out.dropna(subset=cols).reset_index(drop=True)
