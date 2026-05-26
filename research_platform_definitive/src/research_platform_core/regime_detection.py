"""Simple, explainable market regime detection from Macro DB features."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .data_platform import resolve_data_platform_roots, utc_now
from .macro_context import build_macro_context_panel, load_macro_context_panel


REGIME_HISTORY_REL = Path("macro_market") / "tables" / "MarketRegimeHistory.csv"
REGIME_LATEST_REL = Path("macro_market") / "tables" / "MarketRegimeLatest.csv"


def classify_market_regime(row: pd.Series) -> tuple[str, str]:
    """Return a compact regime label and driver text for one macro row."""
    spy = pd.to_numeric(pd.Series([row.get("macro_spy_ret63d")]), errors="coerce").iloc[0]
    credit = pd.to_numeric(pd.Series([row.get("macro_credit_hyg_tlt_ret63d")]), errors="coerce").iloc[0]
    vix = pd.to_numeric(pd.Series([row.get("macro_vix_level")]), errors="coerce").iloc[0]
    dxy = pd.to_numeric(pd.Series([row.get("macro_dxy_ret63d")]), errors="coerce").iloc[0]
    risk_on = pd.to_numeric(pd.Series([row.get("macro_risk_on_score")]), errors="coerce").iloc[0]

    drivers: list[str] = []
    if pd.notna(spy):
        drivers.append(f"SPY 63d {spy:+.1%}")
    if pd.notna(credit):
        drivers.append(f"HYG-TLT 63d {credit:+.1%}")
    if pd.notna(vix):
        drivers.append(f"VIX {vix:.1f}")
    if pd.notna(dxy):
        drivers.append(f"DXY 63d {dxy:+.1%}")

    if (pd.notna(vix) and vix >= 35) or (pd.notna(spy) and spy <= -0.15 and pd.notna(credit) and credit < 0):
        return "crisis", "; ".join(drivers)
    if pd.notna(risk_on) and risk_on >= 70 and (pd.isna(spy) or spy >= 0):
        return "risk_on", "; ".join(drivers)
    if (pd.notna(spy) and spy < -0.05) or (pd.notna(credit) and credit < -0.03) or (pd.notna(vix) and vix >= 25):
        return "risk_off", "; ".join(drivers)
    if pd.notna(spy) and spy > 0 and pd.notna(vix) and vix < 25 and pd.notna(risk_on) and risk_on >= 45:
        return "recovery", "; ".join(drivers)
    return "neutral", "; ".join(drivers)


def build_market_regime_history(
    financial_db_root: str | Path | None = None,
    output_root: str | Path | None = None,
    *,
    write: bool = True,
) -> pd.DataFrame:
    """Build and optionally persist an explainable daily regime history."""
    roots = resolve_data_platform_roots(financial_db_root=financial_db_root, repo_output_root=output_root)
    macro = load_macro_context_panel(roots.financial_db, roots.repo_output, refresh_if_missing=False)
    if macro.empty:
        macro = build_macro_context_panel(roots.financial_db, roots.repo_output, write=True)
    if macro.empty or "date" not in macro.columns:
        return pd.DataFrame(columns=["date", "regime", "regime_score", "drivers", "updated_at"])
    rows: list[dict[str, Any]] = []
    view = macro.copy()
    view["date"] = pd.to_datetime(view["date"], errors="coerce")
    view = view.dropna(subset=["date"]).sort_values("date")
    for _, row in view.iterrows():
        regime, drivers = classify_market_regime(row)
        rows.append(
            {
                "date": row["date"].date().isoformat(),
                "regime": regime,
                "regime_score": row.get("macro_risk_on_score"),
                "macro_spy_ret63d": row.get("macro_spy_ret63d"),
                "macro_credit_hyg_tlt_ret63d": row.get("macro_credit_hyg_tlt_ret63d"),
                "macro_vix_level": row.get("macro_vix_level"),
                "macro_dxy_ret63d": row.get("macro_dxy_ret63d"),
                "drivers": drivers,
                "updated_at": utc_now(),
            }
        )
    history = pd.DataFrame(rows)
    if write:
        table_root = roots.repo_output / REGIME_HISTORY_REL.parent
        table_root.mkdir(parents=True, exist_ok=True)
        history.to_csv(roots.repo_output / REGIME_HISTORY_REL, index=False)
        latest = history.tail(1).copy()
        latest.to_csv(roots.repo_output / REGIME_LATEST_REL, index=False)
    return history


def load_market_regime_history(
    financial_db_root: str | Path | None = None,
    output_root: str | Path | None = None,
    *,
    refresh_if_missing: bool = True,
) -> pd.DataFrame:
    roots = resolve_data_platform_roots(financial_db_root=financial_db_root, repo_output_root=output_root)
    path = roots.repo_output / REGIME_HISTORY_REL
    if path.exists() and path.stat().st_size > 1:
        try:
            return pd.read_csv(path)
        except Exception:
            pass
    if refresh_if_missing:
        return build_market_regime_history(roots.financial_db, roots.repo_output, write=True)
    return pd.DataFrame()


def detect_market_regime(
    date: str | pd.Timestamp | None = None,
    financial_db_root: str | Path | None = None,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    """Return the latest/as-of market regime as a dictionary."""
    history = load_market_regime_history(financial_db_root, output_root)
    if history.empty:
        return {"status": "MISSING", "regime": "unknown", "date": "", "drivers": ""}
    view = history.copy()
    view["date"] = pd.to_datetime(view["date"], errors="coerce")
    view = view.dropna(subset=["date"]).sort_values("date")
    if date is not None:
        asof = pd.to_datetime(date, errors="coerce")
        if pd.notna(asof):
            view = view[view["date"] <= asof]
    if view.empty:
        return {"status": "MISSING", "regime": "unknown", "date": "", "drivers": ""}
    row = view.iloc[-1].to_dict()
    row["status"] = "OK"
    row["date"] = pd.to_datetime(row.get("date")).date().isoformat()
    return row

