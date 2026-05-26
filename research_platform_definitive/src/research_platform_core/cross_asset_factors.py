"""Cross-asset momentum factors from Macro DB proxies."""

from __future__ import annotations

import numpy as np
import pandas as pd


CORE_CROSS_ASSET_SYMBOLS: tuple[str, ...] = ("SPY", "QQQ", "DX-Y.NYB", "TLT", "HYG", "LQD", "BZ=F", "GC=F", "HG=F", "BTC-USD")


def sanitize_symbol(symbol: object) -> str:
    text = str(symbol or "").upper().replace("=X", "").replace("^", "")
    replacements = {"DX-Y.NYB": "DXY", "BZ=F": "BRENT", "GC=F": "GOLD", "HG=F": "COPPER", "BTC-USD": "BTC"}
    text = replacements.get(text, text)
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")


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


def build_cross_asset_momentum(macro_history_df: pd.DataFrame) -> pd.DataFrame:
    """Build lagged momentum-everywhere features across Macro DB proxies.

    Reference: Asness, Moskowitz and Pedersen (2013); Bartram et al. (2021).
    """
    close = _wide_from_macro_history(macro_history_df).sort_index().ffill()
    if close.empty:
        return pd.DataFrame()
    selected = [col for col in close.columns if str(col).upper() in {s.upper() for s in CORE_CROSS_ASSET_SYMBOLS}]
    if not selected:
        selected = list(close.columns[:20])
    out = pd.DataFrame(index=close.index)
    for col in selected:
        sid = sanitize_symbol(col)
        series = pd.to_numeric(close[col], errors="coerce").shift(1)
        mom = series.shift(21) / series.shift(252) - 1
        vol = series.pct_change().rolling(63, min_periods=20).std() * np.sqrt(252)
        ma50 = series.rolling(50, min_periods=20).mean()
        ma200 = series.rolling(200, min_periods=60).mean()
        out[f"xasset_mom_{sid}"] = mom
        out[f"xasset_trend_{sid}"] = np.sign(ma50 - ma200)
        out[f"xasset_vol_adj_mom_{sid}"] = mom / vol.replace(0, np.nan)
    out["date"] = out.index
    return out.reset_index(drop=True)


def cross_asset_momentum(macro_history_df: pd.DataFrame) -> pd.DataFrame:
    """Alias for the implemented cross-asset momentum builder."""
    return build_cross_asset_momentum(macro_history_df)


__all__ = ["CORE_CROSS_ASSET_SYMBOLS", "build_cross_asset_momentum", "cross_asset_momentum", "sanitize_symbol"]
