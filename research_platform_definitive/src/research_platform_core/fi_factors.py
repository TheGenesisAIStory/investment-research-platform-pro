"""Fixed-income factor construction from Macro DB ETF/yield proxies."""

from __future__ import annotations

import numpy as np
import pandas as pd


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


def build_fi_factors(macro_history_df: pd.DataFrame) -> pd.DataFrame:
    """Build lagged term carry, credit carry, bond momentum and real-yield proxy.

    References: Fama and Bliss (1987), Elton et al. (2001), Asness et al.
    (2013), Bartram et al. (2021).
    """
    close = _wide_from_macro_history(macro_history_df).sort_index().ffill()
    if close.empty:
        return pd.DataFrame()
    tlt = _pick(close, ("TLT",))
    shy = _pick(close, ("SHY",))
    hyg = _pick(close, ("HYG",))
    lqd = _pick(close, ("LQD",))
    tip = _pick(close, ("TIP",))
    out = pd.DataFrame(index=close.index)
    if tlt is not None and shy is not None:
        out["fi_term_carry_tlt_shy"] = tlt.shift(1).pct_change(21) - shy.shift(1).pct_change(21)
    if hyg is not None and lqd is not None:
        out["fi_credit_carry_hyg_lqd"] = hyg.shift(1).pct_change(21) - lqd.shift(1).pct_change(21)
    if tlt is not None:
        lagged = tlt.shift(1)
        out["fi_momentum_tlt"] = lagged.shift(21) / lagged.shift(252) - 1
    if tip is not None and tlt is not None:
        out["fi_real_yield_tip_tlt"] = tip.shift(1).pct_change(21) - tlt.shift(1).pct_change(21)
    else:
        out["fi_real_yield_tip_tlt"] = np.nan
    out["date"] = out.index
    return out.reset_index(drop=True)


__all__ = ["build_fi_factors"]
