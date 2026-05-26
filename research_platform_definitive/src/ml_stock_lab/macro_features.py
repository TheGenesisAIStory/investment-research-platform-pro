"""Lagged Macro DB context features for ML Stock Lab panels."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


MACRO_CONTEXT_FEATURES: tuple[str, ...] = (
    "macro_spy_ret21d",
    "macro_spy_ret63d",
    "macro_dxy_ret21d",
    "macro_brent_ret21d",
    "macro_tlt_ret21d",
    "macro_credit_spread",
    "macro_yield_slope",
    "macro_vix_level",
    "macro_regime_encoded",
)

REGIME_ENCODING = {"risk_on": 1.0, "recovery": 0.5, "risk_off": -0.5, "crisis": -1.0}


def _warn_missing(symbol: str) -> None:
    warnings.warn(f"macro_context proxy missing: {symbol}", RuntimeWarning, stacklevel=2)


def _pick(close: pd.DataFrame, aliases: list[str]) -> pd.Series | None:
    columns = {str(col).upper(): col for col in close.columns}
    for alias in aliases:
        col = columns.get(alias.upper())
        if col is not None:
            return pd.to_numeric(close[col], errors="coerce")
    return None


def _rolling_percentile(series: pd.Series, window: int = 252) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.rolling(window, min_periods=max(20, min(60, window))).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1],
        raw=False,
    )


def _wide_from_macro_history(macro_history_df: pd.DataFrame) -> pd.DataFrame:
    if macro_history_df.empty:
        return pd.DataFrame()
    frame = macro_history_df.copy()
    if "date" not in frame.columns:
        return pd.DataFrame()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    if {"symbol", "close"}.issubset(frame.columns):
        frame["symbol"] = frame["symbol"].astype(str).str.upper().str.strip()
        frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
        return frame.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index()
    if {"symbol", "last_close"}.issubset(frame.columns):
        frame["symbol"] = frame["symbol"].astype(str).str.upper().str.strip()
        frame["last_close"] = pd.to_numeric(frame["last_close"], errors="coerce")
        return frame.pivot_table(index="date", columns="symbol", values="last_close", aggfunc="last").sort_index()
    return pd.DataFrame()


def _feature_frame_from_wide(close: pd.DataFrame) -> pd.DataFrame:
    close = close.sort_index().ffill()
    features = pd.DataFrame(index=close.index)
    spy = _pick(close, ["SPY", "^GSPC", "ACWI", "IWDA.AS"])
    dxy = _pick(close, ["DXY", "DX-Y.NYB"])
    brent = _pick(close, ["BRENT", "BZ=F", "CL=F"])
    tlt = _pick(close, ["TLT"])
    hyg = _pick(close, ["HYG"])
    vix = _pick(close, ["VIX", "^VIX"])
    tnx = _pick(close, ["TNX", "^TNX"])
    irx = _pick(close, ["IRX", "^IRX"])

    if spy is None:
        _warn_missing("SPY")
    else:
        features["macro_spy_ret21d"] = spy.pct_change(21)
        features["macro_spy_ret63d"] = spy.pct_change(63)
    if dxy is None:
        _warn_missing("DXY")
    else:
        features["macro_dxy_ret21d"] = dxy.pct_change(21)
    if brent is None:
        _warn_missing("Brent")
    else:
        features["macro_brent_ret21d"] = brent.pct_change(21)
    if tlt is None:
        _warn_missing("TLT")
    else:
        features["macro_tlt_ret21d"] = tlt.pct_change(21)
    if hyg is None or tlt is None:
        _warn_missing("HYG/TLT credit spread")
    else:
        features["macro_credit_spread"] = hyg.pct_change(21) - tlt.pct_change(21)
    if tnx is not None and irx is not None:
        features["macro_yield_slope"] = tnx - irx
    elif tlt is not None:
        features["macro_yield_slope"] = tlt.pct_change(63)
    else:
        _warn_missing("yield slope")
    if vix is None:
        _warn_missing("VIX")
    else:
        features["macro_vix_level"] = _rolling_percentile(vix, 252)

    score_parts: list[pd.Series] = []
    if "macro_spy_ret21d" in features:
        score_parts.append((features["macro_spy_ret21d"] > 0).astype(float))
    if "macro_credit_spread" in features:
        score_parts.append((features["macro_credit_spread"] > -0.02).astype(float))
    if "macro_vix_level" in features:
        score_parts.append((features["macro_vix_level"] < 0.70).astype(float))
    if score_parts:
        risk_score = pd.concat(score_parts, axis=1).mean(axis=1, skipna=True)
        features["macro_regime_encoded"] = np.select(
            [risk_score >= 0.67, risk_score >= 0.45, risk_score >= 0.25, risk_score < 0.25],
            [1.0, 0.5, -0.5, -1.0],
            default=np.nan,
        )
    features = features.shift(1)
    features["date"] = features.index
    return features.reset_index(drop=True)


def _feature_frame_from_existing(macro_history_df: pd.DataFrame) -> pd.DataFrame:
    frame = macro_history_df.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    rename_map = {
        "macro_credit_hyg_tlt_ret63d": "macro_credit_spread",
        "macro_curve_tnx_irx": "macro_yield_slope",
        "regime_encoded": "macro_regime_encoded",
    }
    frame = frame.rename(columns={k: v for k, v in rename_map.items() if k in frame.columns and v not in frame.columns})
    if "regime" in frame.columns and "macro_regime_encoded" not in frame.columns:
        frame["macro_regime_encoded"] = frame["regime"].astype(str).str.lower().map(REGIME_ENCODING)
    if "macro_vix_raw" in frame.columns and "macro_vix_level" not in frame.columns:
        frame["macro_vix_level"] = _rolling_percentile(pd.to_numeric(frame["macro_vix_raw"], errors="coerce"), 252)
    cols = ["date", *[col for col in MACRO_CONTEXT_FEATURES if col in frame.columns]]
    out = frame[cols].sort_values("date")
    for col in MACRO_CONTEXT_FEATURES:
        if col not in out.columns:
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce")
    value_cols = [col for col in out.columns if col != "date"]
    out[value_cols] = out[value_cols].shift(1)
    return out


def build_macro_context_features(factor_panel_df: pd.DataFrame, macro_history_df: pd.DataFrame) -> pd.DataFrame:
    """As-of join lagged macro_context features onto an equity factor panel.

    The function only uses information available strictly before the equity
    observation date: every macro feature is delayed by one macro row before the
    as-of merge.
    """
    if factor_panel_df.empty or "date" not in factor_panel_df.columns or macro_history_df.empty:
        return factor_panel_df.copy()
    if {"symbol", "close"}.issubset(macro_history_df.columns) or {"symbol", "last_close"}.issubset(macro_history_df.columns):
        macro_features = _feature_frame_from_wide(_wide_from_macro_history(macro_history_df))
    else:
        macro_features = _feature_frame_from_existing(macro_history_df)
    if macro_features.empty or "date" not in macro_features.columns:
        return factor_panel_df.copy()
    feature_cols = [col for col in MACRO_CONTEXT_FEATURES if col in macro_features.columns]
    if not feature_cols:
        return factor_panel_df.copy()
    left = factor_panel_df.copy()
    left["__row_order"] = np.arange(len(left))
    left["date"] = pd.to_datetime(left["date"], errors="coerce")
    right = macro_features[["date", *feature_cols]].dropna(subset=["date"]).sort_values("date")
    merged = pd.merge_asof(left.sort_values("date"), right, on="date", direction="backward")
    return merged.sort_values("__row_order").drop(columns=["__row_order"]).reset_index(drop=True)


__all__ = ["MACRO_CONTEXT_FEATURES", "REGIME_ENCODING", "build_macro_context_features"]
