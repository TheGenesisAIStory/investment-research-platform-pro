"""Technical feature helpers for price panels."""

from __future__ import annotations

import pandas as pd

from ml_stock_lab.datasets import normalize_panel


def add_return_features(
    df: pd.DataFrame,
    price_col: str = "mkt_price",
    windows: tuple[int, ...] = (1, 3, 6, 12),
    date_col: str = "date",
    ticker_col: str = "ticker",
) -> pd.DataFrame:
    """Add trailing percentage-return features for the requested windows."""
    out = normalize_panel(df)
    if price_col not in out.columns and "price" in out.columns:
        price_col = "price"
    if price_col not in out.columns:
        return out

    sort_cols = [col for col in [ticker_col, date_col] if col in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).copy()
    prices = pd.to_numeric(out[price_col], errors="coerce")
    for window in windows:
        col = f"tech_ret_{window}m"
        if ticker_col in out.columns:
            out[col] = out.groupby(ticker_col)[price_col].transform(lambda s: pd.to_numeric(s, errors="coerce").pct_change(window))
        else:
            out[col] = prices.pct_change(window)
    return out


def add_volatility_features(
    df: pd.DataFrame,
    return_col: str = "tech_ret_1m",
    windows: tuple[int, ...] = (6, 12),
    date_col: str = "date",
    ticker_col: str = "ticker",
) -> pd.DataFrame:
    """Add trailing volatility features from an existing return column."""
    out = normalize_panel(df)
    if return_col not in out.columns:
        return out
    sort_cols = [col for col in [ticker_col, date_col] if col in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).copy()
    for window in windows:
        col = f"tech_vol_{window}m"
        if ticker_col in out.columns:
            out[col] = out.groupby(ticker_col)[return_col].transform(lambda s: pd.to_numeric(s, errors="coerce").rolling(window).std())
        else:
            out[col] = pd.to_numeric(out[return_col], errors="coerce").rolling(window).std()
    return out
