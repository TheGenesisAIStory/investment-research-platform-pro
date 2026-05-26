"""Explainable market-regime detection from Macro DB context features.

The classifier is deliberately transparent.  It combines lagged equity
momentum, credit relative performance, volatility, yield-curve slope and
commodity momentum into four labels used as context in Home, Macro View,
Portfolio and ML model cards:

``risk_on`` | ``risk_off`` | ``crisis`` | ``recovery``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data_platform import resolve_data_platform_roots, utc_now
from .macro_context import build_macro_context_panel, load_macro_context_panel


RegimeLabel = str

REGIME_HISTORY_REL = Path("macro_market") / "tables" / "MarketRegimeHistory.csv"
REGIME_LATEST_REL = Path("macro_market") / "tables" / "MarketRegimeLatest.csv"

REGIME_ENCODING: dict[str, float] = {
    "risk_on": 1.0,
    "recovery": 0.5,
    "risk_off": -0.5,
    "crisis": -1.0,
    "unknown": 0.0,
}


def _read_csv(path: str | Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    target = Path(path)
    if not target.exists() or target.stat().st_size <= 1:
        return pd.DataFrame()
    try:
        return pd.read_csv(target)
    except Exception:
        return pd.DataFrame()


def _numeric(value: object, default: float = np.nan) -> float:
    try:
        parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        return float(parsed) if pd.notna(parsed) else default
    except Exception:
        return default


def _first_existing(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_map = {str(col).lower(): col for col in frame.columns}
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
        match = lower_map.get(candidate.lower())
        if match is not None:
            return match
    return None


def _symbol_close_panel(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a date-indexed close panel from long MacroHistory-style data."""
    if frame.empty or "date" not in frame.columns:
        return pd.DataFrame()
    if {"symbol", "close"}.issubset(frame.columns):
        out = frame.copy()
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out["symbol"] = out["symbol"].astype(str).str.upper().str.strip()
        out["close"] = pd.to_numeric(out["close"], errors="coerce")
        return out.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index()
    if "last_close" in frame.columns and "symbol" in frame.columns:
        out = frame.copy()
        out["date"] = pd.to_datetime(out.get("date", pd.Timestamp.utcnow()), errors="coerce")
        if out["date"].isna().all():
            out["date"] = pd.Timestamp.utcnow().normalize()
        out["symbol"] = out["symbol"].astype(str).str.upper().str.strip()
        out["last_close"] = pd.to_numeric(out["last_close"], errors="coerce")
        return out.pivot_table(index="date", columns="symbol", values="last_close", aggfunc="last").sort_index()
    return pd.DataFrame()


def _pick_symbol(close: pd.DataFrame, aliases: list[str]) -> pd.Series | None:
    if close.empty:
        return None
    col_map = {str(col).upper(): col for col in close.columns}
    for alias in aliases:
        col = col_map.get(alias.upper())
        if col is not None:
            return pd.to_numeric(close[col], errors="coerce")
    return None


def _rolling_percentile(series: pd.Series, window: int = 252) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.rolling(window, min_periods=max(20, min(60, window))).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1],
        raw=False,
    )


def _features_from_close_panel(close: pd.DataFrame) -> pd.DataFrame:
    close = close.sort_index().ffill()
    out = pd.DataFrame(index=close.index)
    spy = _pick_symbol(close, ["SPY", "^GSPC", "ACWI", "IWDA.AS", "URTH"])
    dxy = _pick_symbol(close, ["DXY", "DX-Y.NYB"])
    brent = _pick_symbol(close, ["BRENT", "BZ=F"])
    gold = _pick_symbol(close, ["GLD", "GC=F"])
    hyg = _pick_symbol(close, ["HYG"])
    tlt = _pick_symbol(close, ["TLT"])
    vix = _pick_symbol(close, ["VIX", "^VIX"])
    tnx = _pick_symbol(close, ["TNX", "^TNX"])
    irx = _pick_symbol(close, ["IRX", "^IRX"])

    if spy is not None:
        out["equity_momentum_21d"] = spy.pct_change(21)
        out["equity_momentum_63d"] = spy.pct_change(63)
    if dxy is not None:
        out["dxy_momentum_21d"] = dxy.pct_change(21)
    if brent is not None:
        out["commodity_momentum"] = brent.pct_change(21)
    elif gold is not None:
        out["commodity_momentum"] = gold.pct_change(21)
    if hyg is not None and tlt is not None:
        out["credit_spread"] = hyg.pct_change(21) - tlt.pct_change(21)
    if vix is not None:
        out["vix_proxy"] = vix
        out["vix_percentile_252d"] = _rolling_percentile(vix, 252)
    if tnx is not None and irx is not None:
        out["yield_slope"] = pd.to_numeric(tnx, errors="coerce") - pd.to_numeric(irx, errors="coerce")
    out["date"] = out.index
    return out.reset_index(drop=True)


def _coerce_macro_signal_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize MacroContextPanel, MacroHistorySample or LatestSnapshot data."""
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "equity_momentum_21d",
                "equity_momentum_63d",
                "credit_spread",
                "vix_proxy",
                "yield_slope",
                "commodity_momentum",
            ]
        )
    if "date" not in frame.columns:
        frame = frame.copy()
        frame["date"] = pd.Timestamp.utcnow().normalize()

    close = _symbol_close_panel(frame)
    if not close.empty and len(close) > 1:
        signals = _features_from_close_panel(close)
    else:
        signals = frame.copy()
        signals["date"] = pd.to_datetime(signals["date"], errors="coerce")

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(signals["date"], errors="coerce")

    aliases = {
        "equity_momentum_21d": ["equity_momentum_21d", "macro_spy_ret21d", "return_1m"],
        "equity_momentum_63d": ["equity_momentum_63d", "macro_spy_ret63d", "return_3m"],
        "credit_spread": ["credit_spread", "macro_credit_spread", "macro_credit_hyg_tlt_ret63d", "macro_credit_lqd_tlt_ret63d"],
        "vix_proxy": ["vix_proxy", "macro_vix_raw", "macro_vix_level", "last_close"],
        "yield_slope": ["yield_slope", "macro_yield_slope", "macro_curve_tnx_irx", "regime_yield_curve_slope"],
        "commodity_momentum": ["commodity_momentum", "macro_brent_ret21d", "macro_wti_ret21d", "macro_gld_ret21d"],
    }
    for target, candidates in aliases.items():
        col = _first_existing(signals, candidates)
        out[target] = pd.to_numeric(signals[col], errors="coerce") if col is not None else np.nan

    if "symbol" in frame.columns and "return_1m" in frame.columns:
        latest = frame.copy()
        latest["symbol"] = latest["symbol"].astype(str).str.upper()
        latest["return_1m"] = pd.to_numeric(latest["return_1m"], errors="coerce")
        latest["return_3m"] = pd.to_numeric(latest.get("return_3m"), errors="coerce")
        if out["equity_momentum_21d"].isna().all():
            spy = latest[latest["symbol"].isin(["SPY", "^GSPC", "ACWI", "IWDA.AS"])]
            if not spy.empty:
                out["equity_momentum_21d"] = spy["return_1m"].iloc[-1]
                out["equity_momentum_63d"] = spy["return_3m"].iloc[-1] if "return_3m" in spy else np.nan
        if out["credit_spread"].isna().all():
            hyg = latest[latest["symbol"].eq("HYG")]["return_1m"]
            tlt = latest[latest["symbol"].eq("TLT")]["return_1m"]
            if not hyg.empty and not tlt.empty:
                out["credit_spread"] = float(hyg.iloc[-1] - tlt.iloc[-1])

    out = out.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last")
    return out.reset_index(drop=True)


def _classify_row(row: pd.Series, previous_regime: str | None = None) -> tuple[str, float, str, float]:
    equity21 = _numeric(row.get("equity_momentum_21d"))
    equity63 = _numeric(row.get("equity_momentum_63d"))
    credit = _numeric(row.get("credit_spread"))
    vix = _numeric(row.get("vix_proxy"))
    slope = _numeric(row.get("yield_slope"))
    commodity = _numeric(row.get("commodity_momentum"))

    available = [
        pd.notna(equity21),
        pd.notna(credit),
        pd.notna(vix),
        pd.notna(slope),
        pd.notna(commodity),
    ]
    availability = sum(available) / max(1, len(available))
    credit_stress = pd.notna(credit) and credit < -0.02
    vix_high = pd.notna(vix) and ((vix > 30.0) if vix > 1.5 else (vix > 0.90))
    vix_warning = pd.notna(vix) and ((vix > 20.0) if vix > 1.5 else (vix > 0.70))
    vix_low = pd.isna(vix) or ((vix < 20.0) if vix > 1.5 else (vix < 0.70))
    curve_warning = pd.notna(slope) and slope < 0

    equity_crisis = (pd.notna(equity21) and equity21 < -0.05) or (pd.notna(equity63) and equity63 < -0.10)
    if vix_high and equity_crisis and credit_stress:
        regime = "crisis"
        confidence = 0.85
    elif previous_regime in {"risk_off", "crisis"} and pd.notna(equity21) and equity21 > 0 and not vix_high:
        regime = "recovery"
        confidence = 0.70
    elif vix_warning or (pd.notna(equity21) and equity21 < -0.02) or credit_stress or curve_warning:
        regime = "risk_off"
        confidence = 0.68
    elif pd.notna(equity21) and equity21 > 0 and vix_low and not credit_stress:
        regime = "risk_on"
        confidence = 0.72
    else:
        regime = "risk_off" if sum(bool(x) for x in [vix_warning, credit_stress, curve_warning]) else "risk_on"
        confidence = 0.45

    drivers = []
    if pd.notna(equity21):
        drivers.append(f"SPY 21d {equity21:+.1%}")
    if pd.notna(equity63):
        drivers.append(f"SPY 63d {equity63:+.1%}")
    if pd.notna(credit):
        drivers.append(f"HYG-TLT 21d {credit:+.1%}")
    if pd.notna(vix):
        drivers.append(f"VIX {vix:.1f}" if vix > 1.5 else f"VIX pctile {vix:.0%}")
    if pd.notna(slope):
        drivers.append(f"10Y-2Y {slope:+.2f}")
    if pd.notna(commodity):
        drivers.append(f"Commodity 21d {commodity:+.1%}")

    confidence = float(np.clip(confidence * (0.5 + 0.5 * availability), 0.05, 0.99))
    return regime, confidence, "; ".join(drivers), REGIME_ENCODING.get(regime, 0.0)


def classify_market_regime(row: pd.Series, previous_regime: str | None = None) -> tuple[str, str]:
    """Compatibility helper returning ``(regime, drivers)`` for one row."""
    regime, _, drivers, _ = _classify_row(row, previous_regime=previous_regime)
    return regime, drivers


def build_regime_history(
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    macro_history_path: str | Path | None = None,
    output_root: str | Path | None = None,
    *,
    write: bool = True,
) -> pd.DataFrame:
    """Build ``MarketRegimeHistory.csv`` from a macro history/snapshot table."""
    frame = _read_csv(macro_history_path)
    if frame.empty and output_root is not None:
        root = Path(output_root).expanduser()
        for rel in [
            Path("macro_market") / "tables" / "MacroContextPanel.csv",
            Path("macro_market") / "tables" / "MacroHistorySample.csv",
            Path("macro_market") / "tables" / "MacroLatestSnapshot.csv",
        ]:
            frame = _read_csv(root / rel)
            if not frame.empty:
                break
    signals = _coerce_macro_signal_frame(frame)
    if signals.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "regime",
                "confidence",
                "regime_score",
                "equity_momentum_21d",
                "credit_spread",
                "vix_proxy",
                "yield_slope",
                "commodity_momentum",
                "drivers",
                "updated_at",
            ]
        )
    signals["date"] = pd.to_datetime(signals["date"], errors="coerce")
    if start is not None:
        signals = signals[signals["date"] >= pd.to_datetime(start, errors="coerce")]
    if end is not None:
        signals = signals[signals["date"] <= pd.to_datetime(end, errors="coerce")]

    rows: list[dict[str, Any]] = []
    previous: str | None = None
    for _, row in signals.dropna(subset=["date"]).sort_values("date").iterrows():
        regime, confidence, drivers, encoded = _classify_row(row, previous_regime=previous)
        previous = regime
        rows.append(
            {
                "date": pd.to_datetime(row["date"]).date().isoformat(),
                "regime": regime,
                "confidence": confidence,
                "regime_score": (encoded + 1.0) * 50.0,
                "regime_encoded": encoded,
                "equity_momentum_21d": row.get("equity_momentum_21d"),
                "equity_momentum_63d": row.get("equity_momentum_63d"),
                "credit_spread": row.get("credit_spread"),
                "vix_proxy": row.get("vix_proxy"),
                "yield_slope": row.get("yield_slope"),
                "commodity_momentum": row.get("commodity_momentum"),
                "drivers": drivers,
                "updated_at": utc_now(),
            }
        )
    history = pd.DataFrame(rows)
    if write and output_root is not None:
        root = Path(output_root).expanduser()
        table_root = root / REGIME_HISTORY_REL.parent
        table_root.mkdir(parents=True, exist_ok=True)
        history.to_csv(root / REGIME_HISTORY_REL, index=False)
        history.tail(1).to_csv(root / REGIME_LATEST_REL, index=False)
    return history


def build_market_regime_history(
    financial_db_root: str | Path | None = None,
    output_root: str | Path | None = None,
    *,
    write: bool = True,
) -> pd.DataFrame:
    """Compatibility wrapper that builds history from the MacroContext panel."""
    roots = resolve_data_platform_roots(financial_db_root=financial_db_root, repo_output_root=output_root)
    macro = load_macro_context_panel(roots.financial_db, roots.repo_output, refresh_if_missing=False)
    if macro.empty:
        macro = build_macro_context_panel(roots.financial_db, roots.repo_output, write=True)
    if macro.empty:
        return build_regime_history(output_root=roots.repo_output, write=write)
    tmp = roots.repo_output / REGIME_HISTORY_REL.parent / "_macro_context_for_regime.csv"
    if write:
        tmp.parent.mkdir(parents=True, exist_ok=True)
        macro.to_csv(tmp, index=False)
        history = build_regime_history(macro_history_path=tmp, output_root=roots.repo_output, write=write)
        try:
            tmp.unlink()
        except OSError:
            pass
        return history
    return build_regime_history(macro_history_path=None, output_root=roots.repo_output, write=False) if macro.empty else build_regime_history_from_frame(macro)


def build_regime_history_from_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Build regime history from an in-memory macro signal frame without writing."""
    signals = _coerce_macro_signal_frame(frame)
    rows: list[dict[str, Any]] = []
    previous: str | None = None
    for _, row in signals.dropna(subset=["date"]).sort_values("date").iterrows():
        regime, confidence, drivers, encoded = _classify_row(row, previous_regime=previous)
        previous = regime
        rows.append(
            {
                "date": pd.to_datetime(row["date"]).date().isoformat(),
                "regime": regime,
                "confidence": confidence,
                "regime_score": (encoded + 1.0) * 50.0,
                "regime_encoded": encoded,
                "equity_momentum_21d": row.get("equity_momentum_21d"),
                "equity_momentum_63d": row.get("equity_momentum_63d"),
                "credit_spread": row.get("credit_spread"),
                "vix_proxy": row.get("vix_proxy"),
                "yield_slope": row.get("yield_slope"),
                "commodity_momentum": row.get("commodity_momentum"),
                "drivers": drivers,
                "updated_at": utc_now(),
            }
        )
    return pd.DataFrame(rows)


def load_market_regime_history(
    financial_db_root: str | Path | None = None,
    output_root: str | Path | None = None,
    *,
    refresh_if_missing: bool = True,
) -> pd.DataFrame:
    roots = resolve_data_platform_roots(financial_db_root=financial_db_root, repo_output_root=output_root)
    path = roots.repo_output / REGIME_HISTORY_REL
    history = _read_csv(path)
    if not history.empty:
        return history
    if refresh_if_missing:
        return build_market_regime_history(roots.financial_db, roots.repo_output, write=True)
    return pd.DataFrame()


def detect_market_regime(
    date: str | pd.Timestamp | None = None,
    macro_snapshot_path: str | Path | None = None,
    financial_db_root: str | Path | None = None,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    """Return the current/as-of market regime with confidence and signals."""
    if macro_snapshot_path is not None:
        history = build_regime_history(
            start=None,
            end=date,
            macro_history_path=macro_snapshot_path,
            output_root=output_root,
            write=output_root is not None,
        )
    else:
        history = load_market_regime_history(financial_db_root, output_root, refresh_if_missing=True)
    if history.empty:
        return {"status": "MISSING", "regime": "unknown", "date": "", "confidence": 0.0, "signals": {}, "drivers": ""}
    view = history.copy()
    view["date"] = pd.to_datetime(view["date"], errors="coerce")
    view = view.dropna(subset=["date"]).sort_values("date")
    if date is not None:
        asof = pd.to_datetime(date, errors="coerce")
        if pd.notna(asof):
            view = view[view["date"] <= asof]
    if view.empty:
        return {"status": "MISSING", "regime": "unknown", "date": "", "confidence": 0.0, "signals": {}, "drivers": ""}
    row = view.iloc[-1].to_dict()
    signals = {
        "equity_momentum_21d": row.get("equity_momentum_21d"),
        "credit_spread": row.get("credit_spread"),
        "vix_proxy": row.get("vix_proxy"),
        "yield_slope": row.get("yield_slope"),
        "commodity_momentum": row.get("commodity_momentum"),
    }
    return {
        **row,
        "status": "OK",
        "date": pd.to_datetime(row.get("date")).date().isoformat(),
        "confidence": _numeric(row.get("confidence"), 0.0),
        "signals": signals,
        "drivers": str(row.get("drivers", "")),
    }


def get_current_regime(
    output_root: str | Path | None = None,
    financial_db_root: str | Path | None = None,
) -> dict[str, Any]:
    """Return the latest persisted regime, or ``unknown`` when unavailable."""
    history = load_market_regime_history(financial_db_root=financial_db_root, output_root=output_root, refresh_if_missing=False)
    if history.empty:
        return {"status": "MISSING", "regime": "unknown", "date": "", "confidence": 0.0, "signals": {}, "drivers": ""}
    return detect_market_regime(financial_db_root=financial_db_root, output_root=output_root)


__all__ = [
    "REGIME_ENCODING",
    "REGIME_HISTORY_REL",
    "REGIME_LATEST_REL",
    "RegimeLabel",
    "build_market_regime_history",
    "build_regime_history",
    "build_regime_history_from_frame",
    "classify_market_regime",
    "detect_market_regime",
    "get_current_regime",
    "load_market_regime_history",
]
