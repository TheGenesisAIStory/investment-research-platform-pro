"""Commodity factor construction from Macro DB commodity proxies."""

from __future__ import annotations

import numpy as np
import pandas as pd


COMMODITY_ALIASES: dict[str, tuple[str, ...]] = {
    "wti": ("CL=F", "WTI", "USO"),
    "brent": ("BZ=F", "BRENT"),
    "gold": ("GC=F", "GLD", "GOLD"),
    "copper": ("HG=F", "CPER", "COPPER"),
}


def _wide_from_macro_history(macro_history_df: pd.DataFrame) -> pd.DataFrame:
    if macro_history_df is None or macro_history_df.empty or "date" not in macro_history_df.columns:
        return pd.DataFrame()
    frame = macro_history_df.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    value_col = "close" if "close" in frame.columns else "last_close" if "last_close" in frame.columns else ""
    if "symbol" in frame.columns and value_col:
        frame["symbol"] = frame["symbol"].astype(str).str.upper().str.strip()
        frame[value_col] = pd.to_numeric(frame[value_col], errors="coerce")
        return frame.pivot_table(index="date", columns="symbol", values=value_col, aggfunc="last").sort_index()
    numeric = frame.select_dtypes("number").copy()
    numeric.index = frame["date"]
    return numeric.sort_index()


def _pick(close: pd.DataFrame, aliases: tuple[str, ...]) -> pd.Series | None:
    columns = {str(col).upper(): col for col in close.columns}
    for alias in aliases:
        if alias.upper() in columns:
            return pd.to_numeric(close[columns[alias.upper()]], errors="coerce")
    return None


def build_commodity_factors(macro_history_df: pd.DataFrame) -> pd.DataFrame:
    """Build lagged commodity momentum, trend, carry proxy and mean reversion.

    References: Gorton and Rouwenhorst (2006), Asness et al. (2013), Bartram et
    al. (2021).  Carry is a risk-adjusted return proxy when true futures-basis
    data are unavailable.
    """
    close = _wide_from_macro_history(macro_history_df).sort_index().ffill()
    if close.empty:
        return pd.DataFrame()
    out = pd.DataFrame(index=close.index)
    for commodity, aliases in COMMODITY_ALIASES.items():
        series = _pick(close, aliases)
        if series is None:
            continue
        lagged = series.shift(1)
        ret63 = lagged.pct_change(63)
        vol63 = lagged.pct_change().rolling(63, min_periods=20).std() * np.sqrt(252)
        ma50 = lagged.rolling(50, min_periods=20).mean()
        ma200 = lagged.rolling(200, min_periods=60).mean()
        mean252 = lagged.rolling(252, min_periods=60).mean()
        std252 = lagged.rolling(252, min_periods=60).std()
        out[f"commodity_mom_{commodity}"] = lagged.shift(21) / lagged.shift(252) - 1
        out[f"commodity_trend_{commodity}"] = (ma50 - ma200) / ma200.replace(0, np.nan)
        out[f"commodity_carry_{commodity}"] = ret63 / vol63.replace(0, np.nan)
        out[f"commodity_mean_reversion_{commodity}"] = (lagged - mean252) / std252.replace(0, np.nan)
    out["date"] = out.index
    return out.reset_index(drop=True)


__all__ = ["COMMODITY_ALIASES", "build_commodity_factors"]
