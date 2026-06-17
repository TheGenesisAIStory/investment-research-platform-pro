"""Macro context features for equity ML models.

The canonical equity factor layer remains single-name/cross-sectional.  This
module adds optional, experimental macro features derived from the Macro DB and
lagged by one observation before joining to equity panels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .data_platform import resolve_data_platform_roots, utc_now
from .macro_market import load_macro_market_artifacts


MACRO_CONTEXT_SYMBOLS: tuple[str, ...] = ("SPY", "DXY", "BRENT", "WTI", "TLT", "SHY", "HYG", "LQD", "VIX", "TNX", "IRX", "GLD", "BTC")
MACRO_CONTEXT_REL = Path("macro_market") / "tables" / "MacroContextPanel.csv"


def _read_csv(path: Path, **kwargs: object) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 1:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, **kwargs)
    except Exception:
        return pd.DataFrame()


def _read_macro_history(path: Path, symbol: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 1:
        return pd.DataFrame()
    try:
        frame = pd.read_parquet(path, columns=["date", "close"])
    except Exception:
        try:
            frame = pd.read_parquet(path)
        except Exception:
            return pd.DataFrame()
    if frame.empty or "date" not in frame.columns:
        return pd.DataFrame()
    close_col = "close" if "close" in frame.columns else "adjclose" if "adjclose" in frame.columns else ""
    if not close_col:
        return pd.DataFrame()
    out = frame[["date", close_col]].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out[symbol] = pd.to_numeric(out[close_col], errors="coerce")
    return out.dropna(subset=["date"]).sort_values("date")[["date", symbol]]


def load_macro_close_panel(
    financial_db_root: str | Path | None = None,
    output_root: str | Path | None = None,
    *,
    symbols: Iterable[str] = MACRO_CONTEXT_SYMBOLS,
) -> pd.DataFrame:
    """Load Macro DB close levels as a date-indexed wide frame."""
    roots = resolve_data_platform_roots(financial_db_root=financial_db_root, repo_output_root=output_root)
    artifacts = load_macro_market_artifacts(roots.financial_db, roots.repo_output)
    manifest = artifacts.get("manifest", pd.DataFrame())
    if manifest.empty:
        manifest = _read_csv(roots.repo_output / "macro_market" / "tables" / "MacroAssetManifest.csv")
    wanted = {str(symbol).upper() for symbol in symbols}
    frames: list[pd.DataFrame] = []
    if not manifest.empty and {"symbol", "target_path"}.issubset(manifest.columns):
        for _, row in manifest.iterrows():
            symbol = str(row.get("symbol", "")).upper()
            if symbol not in wanted:
                continue
            path = Path(str(row.get("target_path", "")))
            frame = _read_macro_history(path, symbol)
            if not frame.empty:
                frames.append(frame)
    if not frames:
        sample = artifacts.get("history_sample", pd.DataFrame())
        if not sample.empty and {"date", "symbol", "close"}.issubset(sample.columns):
            sample = sample[sample["symbol"].astype(str).str.upper().isin(wanted)].copy()
            sample["date"] = pd.to_datetime(sample["date"], errors="coerce")
            return sample.pivot_table(index="date", columns=sample["symbol"].astype(str).str.upper(), values="close", aggfunc="last").sort_index()
        return pd.DataFrame()
    panel = frames[0]
    for frame in frames[1:]:
        panel = panel.merge(frame, on="date", how="outer")
    return panel.sort_values("date").set_index("date").sort_index()


def build_macro_context_panel(
    financial_db_root: str | Path | None = None,
    output_root: str | Path | None = None,
    *,
    write: bool = True,
) -> pd.DataFrame:
    """Build lagged macro_context features from Macro DB price/yield proxies."""
    close = load_macro_close_panel(financial_db_root, output_root)
    if close.empty:
        return pd.DataFrame()
    close = close.sort_index().ffill()
    out = pd.DataFrame(index=close.index)

    for symbol in ["SPY", "DXY", "BRENT", "WTI", "TLT", "GLD", "BTC"]:
        if symbol in close.columns:
            out[f"macro_{symbol.lower()}_ret21d"] = close[symbol].pct_change(21)
            out[f"macro_{symbol.lower()}_ret63d"] = close[symbol].pct_change(63)
    if "VIX" in close.columns:
        out["macro_vix_raw"] = close["VIX"]
        out["macro_vix_level"] = close["VIX"]
        out["macro_vix_change21d"] = close["VIX"].diff(21)
        out["macro_vix_percentile_252d"] = close["VIX"].rolling(252, min_periods=60).apply(
            lambda values: pd.Series(values).rank(pct=True).iloc[-1],
            raw=False,
        )
    if {"HYG", "TLT"}.issubset(close.columns):
        out["macro_credit_spread"] = close["HYG"].pct_change(21) - close["TLT"].pct_change(21)
        out["macro_credit_hyg_tlt_ret63d"] = close["HYG"].pct_change(63) - close["TLT"].pct_change(63)
    if {"LQD", "TLT"}.issubset(close.columns):
        out["macro_credit_lqd_tlt_ret63d"] = close["LQD"].pct_change(63) - close["TLT"].pct_change(63)
    if {"TNX", "IRX"}.issubset(close.columns):
        out["macro_yield_slope"] = pd.to_numeric(close["TNX"], errors="coerce") - pd.to_numeric(close["IRX"], errors="coerce")
        out["macro_curve_tnx_irx"] = pd.to_numeric(close["TNX"], errors="coerce") - pd.to_numeric(close["IRX"], errors="coerce")
    elif {"TLT", "SHY"}.issubset(close.columns):
        out["macro_yield_slope"] = close["TLT"].pct_change(63) - close["SHY"].pct_change(63)
        out["macro_curve_tlt_shy_ret63d"] = close["TLT"].pct_change(63) - close["SHY"].pct_change(63)

    conditions: list[pd.Series] = []
    if "macro_spy_ret63d" in out.columns:
        conditions.append(out["macro_spy_ret63d"] > 0)
    if "macro_credit_hyg_tlt_ret63d" in out.columns:
        conditions.append(out["macro_credit_hyg_tlt_ret63d"] > 0)
    if "macro_dxy_ret63d" in out.columns:
        conditions.append(out["macro_dxy_ret63d"] < 0.03)
    if "macro_vix_level" in out.columns:
        vix_median = out["macro_vix_level"].rolling(252, min_periods=60).median()
        conditions.append(out["macro_vix_level"] <= vix_median)
    if conditions:
        condition_frame = pd.concat([condition.astype(float) for condition in conditions], axis=1)
        out["macro_risk_on_score"] = condition_frame.mean(axis=1, skipna=True) * 100.0
        out["macro_context_score"] = out["macro_risk_on_score"]
        out["macro_regime_encoded"] = np.select(
            [
                out["macro_risk_on_score"] >= 67.0,
                out["macro_risk_on_score"].between(45.0, 67.0, inclusive="left"),
                out["macro_risk_on_score"].between(25.0, 45.0, inclusive="left"),
                out["macro_risk_on_score"] < 25.0,
            ],
            [1.0, 0.5, -0.5, -1.0],
            default=np.nan,
        )

    if "macro_spy_ret63d" in out.columns:
        out["regime_spy_trend_sign"] = np.sign(pd.to_numeric(out["macro_spy_ret63d"], errors="coerce"))
    if "macro_vix_level" in out.columns:
        vix = pd.to_numeric(out["macro_vix_level"], errors="coerce")
        out["regime_vix_regime"] = np.select([vix > 25.0, vix < 15.0], [1, -1], default=0)
    if "macro_curve_tnx_irx" in out.columns:
        out["regime_yield_curve_slope"] = pd.to_numeric(out["macro_curve_tnx_irx"], errors="coerce")
    elif "macro_curve_tlt_shy_ret63d" in out.columns:
        out["regime_yield_curve_slope"] = pd.to_numeric(out["macro_curve_tlt_shy_ret63d"], errors="coerce")
    if "macro_dxy_ret21d" in out.columns:
        out["regime_dxy_trend"] = pd.to_numeric(out["macro_dxy_ret21d"], errors="coerce")
    if "macro_gld_ret21d" in out.columns:
        out["regime_gold_trend"] = pd.to_numeric(out["macro_gld_ret21d"], errors="coerce")

    out = out.shift(1)
    out["date"] = out.index
    out["macro_feature_asof_date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["macro_context_generated_at"] = utc_now()
    out = out.reset_index(drop=True)
    if write:
        roots = resolve_data_platform_roots(financial_db_root=financial_db_root, repo_output_root=output_root)
        path = roots.repo_output / MACRO_CONTEXT_REL
        path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(path, index=False)
    return out


def load_macro_context_panel(
    financial_db_root: str | Path | None = None,
    output_root: str | Path | None = None,
    *,
    refresh_if_missing: bool = True,
) -> pd.DataFrame:
    roots = resolve_data_platform_roots(financial_db_root=financial_db_root, repo_output_root=output_root)
    path = roots.repo_output / MACRO_CONTEXT_REL
    frame = _read_csv(path)
    if frame.empty and refresh_if_missing:
        frame = build_macro_context_panel(roots.financial_db, roots.repo_output, write=True)
    if not frame.empty and "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    return frame


def add_macro_context_features(
    panel: pd.DataFrame,
    financial_db_root: str | Path | None = None,
    output_root: str | Path | None = None,
) -> pd.DataFrame:
    """As-of join lagged macro_context features onto an equity factor panel."""
    if panel.empty or "date" not in panel.columns:
        return panel.copy()
    macro = load_macro_context_panel(financial_db_root, output_root)
    if macro.empty or "date" not in macro.columns:
        return panel.copy()
    feature_cols = [col for col in macro.columns if col.startswith("macro_") or col.startswith("regime_")]
    if not feature_cols:
        return panel.copy()
    left = panel.copy()
    left["__row_order"] = np.arange(len(left))
    left["date"] = pd.to_datetime(left["date"], errors="coerce")
    right = macro[["date", *feature_cols]].dropna(subset=["date"]).sort_values("date")
    left_sorted = left.sort_values("date")
    merged = pd.merge_asof(left_sorted, right, on="date", direction="backward")
    return merged.sort_values("__row_order").drop(columns=["__row_order"]).reset_index(drop=True)
