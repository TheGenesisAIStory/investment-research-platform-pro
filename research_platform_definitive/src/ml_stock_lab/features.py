"""Feature engineering for ML equity valuation and prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd


FUNDAMENTAL_ALIASES = {
    "pe": ["peratio", "pe_ratio", "p_e"],
    "pb": ["pbratio", "pb_ratio", "p_b"],
    "ev_ebitda": ["evebitda", "ev_ebitda"],
    "revenue_growth": ["revenuegrowth", "revenue_growth"],
    "debt_to_equity": ["debttoequity", "debt_equity"],
    "quality_score": ["qualityscore", "quality_score"],
    "valuation_score": ["valuationscore", "valuation_score"],
    "momentum_score": ["momentumscore", "momentum_score"],
    "risk_score": ["riskscore", "risk_score"],
}


def add_basic_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Add robust fundamental/technical features where raw columns exist."""
    if panel.empty:
        return panel.copy()
    out = panel.copy()
    lower = {str(c).lower(): c for c in out.columns}
    for canonical, aliases in FUNDAMENTAL_ALIASES.items():
        if canonical not in out.columns:
            for alias in aliases:
                if alias.lower() in lower:
                    out[canonical] = pd.to_numeric(out[lower[alias.lower()]], errors="coerce")
                    break
    if "market_value" in out.columns:
        mv = pd.to_numeric(out["market_value"], errors="coerce")
        out["log_market_value"] = np.log(mv.where(mv > 0))
    if {"quality_score", "valuation_score"}.issubset(out.columns):
        out["quality_value_blend"] = pd.to_numeric(out["quality_score"], errors="coerce") + pd.to_numeric(out["valuation_score"], errors="coerce")
    if {"momentum_score", "risk_score"}.issubset(out.columns):
        out["risk_adjusted_momentum"] = pd.to_numeric(out["momentum_score"], errors="coerce") - pd.to_numeric(out["risk_score"], errors="coerce")
    for ret_col in ["ret21d", "ret63d", "ret126d"]:
        if ret_col in out.columns:
            out[f"{ret_col}_rank"] = out.groupby("date")[ret_col].rank(pct=True) if "date" in out.columns else out[ret_col].rank(pct=True)
    return out


def select_numeric_features(panel: pd.DataFrame, target: str = "market_value", min_non_null: int = 5) -> list[str]:
    """Select stable numeric features for notebook-safe modelling."""
    cols: list[str] = []
    for col in panel.columns:
        if col in {target, "forward_return"}:
            continue
        if pd.api.types.is_numeric_dtype(panel[col]) and panel[col].notna().sum() >= min_non_null:
            cols.append(col)
    return cols


def make_forward_returns(panel: pd.DataFrame, price_col: str = "price", horizon: int = 1) -> pd.DataFrame:
    """Create forward returns by ticker if price history is available."""
    out = panel.copy()
    if price_col not in out.columns or "ticker" not in out.columns:
        return out
    out = out.sort_values(["ticker", "date"] if "date" in out.columns else ["ticker"])
    prices = pd.to_numeric(out[price_col], errors="coerce")
    out["forward_return"] = out.groupby("ticker")[price_col].transform(lambda s: pd.to_numeric(s, errors="coerce").shift(-horizon) / pd.to_numeric(s, errors="coerce") - 1)
    return out


def add_model_based_mispricing_features(
    panel: pd.DataFrame,
    fair_value_col: str,
    market_value_col: str,
    prefix: str = "model",
) -> pd.DataFrame:
    """Add reusable model-based mispricing factors.

    The function is valuation-model agnostic: DCF, residual income, EVA,
    Smart Money overlays or ML fair-value models can all provide `fair_value_col`.
    """
    out = panel.copy()
    if out.empty or fair_value_col not in out.columns or market_value_col not in out.columns:
        return out
    fair = pd.to_numeric(out[fair_value_col], errors="coerce")
    market = pd.to_numeric(out[market_value_col], errors="coerce")
    mispricing = fair / market.replace(0, np.nan) - 1.0
    out[f"{prefix}_mispricing"] = mispricing
    if "date" in out.columns:
        out[f"{prefix}_mispricing_zscore"] = out.groupby("date")[f"{prefix}_mispricing"].transform(
            lambda x: (x - x.mean()) / x.std(ddof=0) if x.std(ddof=0) not in [0, np.nan] else np.nan
        )
        out[f"{prefix}_mispricing_rank"] = out.groupby("date")[f"{prefix}_mispricing"].rank(pct=True)
    else:
        std = mispricing.std(ddof=0)
        out[f"{prefix}_mispricing_zscore"] = (mispricing - mispricing.mean()) / std if std and np.isfinite(std) else np.nan
        out[f"{prefix}_mispricing_rank"] = mispricing.rank(pct=True)
    out[f"{prefix}_factor_family"] = "model_based_mispricing"
    return out
