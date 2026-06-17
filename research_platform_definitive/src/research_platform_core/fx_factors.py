"""FX factor construction from the Macro DB.

The implementation follows the parsimonious FX factor vocabulary used in
institutional factor research: carry proxy, 12-1 momentum, trend and realized
volatility.  All features are built from lagged spot levels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


FX_SYMBOLS: tuple[str, ...] = (
    "EURUSD=X",
    "GBPUSD=X",
    "USDJPY=X",
    "USDCHF=X",
    "AUDUSD=X",
    "USDCAD=X",
    "NZDUSD=X",
    "EURGBP=X",
    "EURJPY=X",
    "EURCHF=X",
    "DX-Y.NYB",
)


def sanitize_symbol(symbol: object) -> str:
    text = str(symbol or "").upper().replace("=X", "").replace("^", "")
    if text in {"DX-Y.NYB", "DXY", "USDOLLAR"}:
        return "dxy"
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")


def _wide_from_macro_history(macro_history_df: pd.DataFrame) -> pd.DataFrame:
    if macro_history_df is None or macro_history_df.empty or "date" not in macro_history_df.columns:
        return pd.DataFrame()
    frame = macro_history_df.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    if {"symbol", "close"}.issubset(frame.columns):
        frame["symbol"] = frame["symbol"].astype(str).str.upper().str.strip()
        frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
        return frame.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index()
    if {"symbol", "last_close"}.issubset(frame.columns):
        frame["symbol"] = frame["symbol"].astype(str).str.upper().str.strip()
        frame["last_close"] = pd.to_numeric(frame["last_close"], errors="coerce")
        return frame.pivot_table(index="date", columns="symbol", values="last_close", aggfunc="last").sort_index()
    numeric = frame.select_dtypes("number").copy()
    numeric.index = frame["date"]
    return numeric.sort_index()


def _select_fx_columns(close: pd.DataFrame) -> list[str]:
    if close.empty:
        return []
    upper = {str(col).upper(): col for col in close.columns}
    selected = [upper[symbol] for symbol in FX_SYMBOLS if symbol.upper() in upper]
    selected.extend(col for col in close.columns if str(col).upper().endswith("=X") and col not in selected)
    return list(dict.fromkeys(selected))


def build_fx_factors(macro_history_df: pd.DataFrame) -> pd.DataFrame:
    """Build lagged FX carry proxy, momentum, trend and volatility factors.

    References: Lustig and Verdelhan (2007), Menkhoff et al. (2012), Asness,
    Moskowitz and Pedersen (2013), Bartram et al. (2021).
    """
    close = _wide_from_macro_history(macro_history_df).sort_index().ffill()
    if close.empty:
        return pd.DataFrame()
    fx_cols = _select_fx_columns(close)
    if not fx_cols:
        return pd.DataFrame({"date": close.index})
    lagged = close[fx_cols].shift(1)
    dxy_col = next((col for col in close.columns if sanitize_symbol(col) == "dxy"), None)
    dxy_ret63 = close[dxy_col].shift(1).pct_change(63) if dxy_col is not None else pd.Series(0.0, index=close.index)
    out = pd.DataFrame(index=close.index)
    for col in fx_cols:
        sid = sanitize_symbol(col)
        series = lagged[col]
        ret63 = series.pct_change(63)
        out[f"fx_carry_{sid}"] = ret63 - dxy_ret63
        out[f"fx_mom_{sid}"] = series.shift(21) / series.shift(252) - 1
        ma50 = series.rolling(50, min_periods=20).mean()
        ma200 = series.rolling(200, min_periods=60).mean()
        out[f"fx_trend_{sid}"] = (ma50 - ma200) / ma200.replace(0, np.nan)
        out[f"fx_vol_{sid}"] = series.pct_change().rolling(63, min_periods=20).std() * np.sqrt(252)
    out["date"] = out.index
    return out.reset_index(drop=True)


__all__ = ["FX_SYMBOLS", "build_fx_factors", "sanitize_symbol"]
