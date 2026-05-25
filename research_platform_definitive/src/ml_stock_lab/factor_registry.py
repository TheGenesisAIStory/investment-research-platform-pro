"""Canonical factor vocabulary for ML Stock Lab and Screener.

The module keeps the research semantics in one place: value, quality,
momentum, risk/low-vol, size and growth are computed as point-in-time
cross-sectional percentiles where the raw columns are available.  Scores are
0-100 where higher is better for the desk interpretation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FactorBlock:
    id: str
    label: str
    columns: tuple[str, ...]
    description: str


FACTOR_BLOCKS: dict[str, FactorBlock] = {
    "value": FactorBlock(
        "value",
        "Value",
        ("valuation_score", "pe", "pb", "ev_ebit", "ev_ebitda", "dividend_yield"),
        "Lower valuation multiples and higher shareholder yield.",
    ),
    "quality": FactorBlock(
        "quality",
        "Quality",
        ("quality_score", "roe", "roic", "gross_margin", "operating_margin", "debt_to_equity"),
        "Profitability, capital efficiency, margins and balance-sheet discipline.",
    ),
    "momentum": FactorBlock(
        "momentum",
        "Momentum",
        ("momentum_score", "momentum_12_1", "ret252d", "ret126d", "ret63d", "ret21d"),
        "Intermediate-term price momentum with a preference for 12-1 month momentum when available.",
    ),
    "risk": FactorBlock(
        "risk",
        "Low Vol / Risk",
        ("risk_score", "vol252d", "vol126d", "vol63d", "volatility", "beta", "max_drawdown"),
        "Lower realized volatility, beta and drawdown risk.",
    ),
    "size": FactorBlock(
        "size",
        "Size / Liquidity",
        ("size_score", "log_market_value", "market_value", "marketcap", "market_cap"),
        "Market-cap/liquidity proxy used to stabilize investability screens.",
    ),
    "growth": FactorBlock(
        "growth",
        "Growth",
        ("growth_score", "revenue_growth", "revenue_cagr", "sales_cagr", "eps_growth"),
        "Revenue or earnings growth proxies where fundamentals are available.",
    ),
    "model_based": FactorBlock(
        "model_based",
        "Model Based Mispricing",
        (
            "model_mispricing",
            "model_mispricing_zscore",
            "model_mispricing_rank",
            "dcf_mispricing",
            "residual_income_mispricing",
            "eva_mispricing",
        ),
        "Valuation-model mispricing factors exported by valuation artifacts.",
    ),
}


RAW_FACTOR_COLUMNS: tuple[str, ...] = tuple(dict.fromkeys(col for block in FACTOR_BLOCKS.values() for col in block.columns))
FACTOR_SCORE_COLUMNS: tuple[str, ...] = (
    "value_score",
    "valuation_score",
    "quality_score",
    "momentum_score",
    "momentum_12_1_score",
    "risk_score",
    "size_score",
    "growth_score",
    "factor_composite_score",
)

LEAKAGE_EXACT_COLUMNS = {
    "actual",
    "prediction",
    "prediction_rank",
    "selected_topk",
    "fair_value_hat",
    "mispricing_rel",
    "zscore",
    "signal",
    "score",
    "ml_score",
    "score_composite",
    "screening_percentile",
    "screening_quintile",
}

LEAKAGE_PATTERNS = (
    re.compile(r"(^|_)future(_|$)", re.I),
    re.compile(r"(^|_)forward(_|$)", re.I),
    re.compile(r"(^|_)target(_|$)", re.I),
    re.compile(r"(^|_)label(_|$)", re.I),
    re.compile(r"(^|_)selected(_|$)", re.I),
    re.compile(r"(^|_)prediction(_|$)", re.I),
)


def available_factor_blocks(panel: pd.DataFrame) -> list[str]:
    """Return factor block ids with at least one available column."""
    if panel.empty:
        return []
    columns = set(panel.columns)
    return [block_id for block_id, block in FACTOR_BLOCKS.items() if columns.intersection(block.columns)]


def is_leakage_feature(column: str, target: str | None = None) -> bool:
    """True when a column should not enter predictive model features."""
    name = str(column)
    lower = name.lower()
    if target and lower == str(target).lower():
        return True
    if lower in {item.lower() for item in LEAKAGE_EXACT_COLUMNS}:
        return True
    if lower == "forward_return" or lower.startswith("forward_return_"):
        return True
    return any(pattern.search(lower) for pattern in LEAKAGE_PATTERNS)


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce") if column in frame.columns else pd.Series(np.nan, index=frame.index)


def _pct_rank(series: pd.Series, higher_is_better: bool = True, group: pd.Series | None = None) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if group is not None:
        return values.groupby(group).rank(pct=True, ascending=higher_is_better) * 100
    return values.rank(pct=True, ascending=higher_is_better) * 100


def _mean_available(columns: Iterable[pd.Series], index: pd.Index) -> pd.Series:
    frame = pd.concat([col for col in columns if col is not None], axis=1) if columns else pd.DataFrame(index=index)
    if frame.empty:
        return pd.Series(np.nan, index=index)
    return frame.mean(axis=1, skipna=True)


def _group_key(panel: pd.DataFrame) -> pd.Series | None:
    if "date" not in panel.columns:
        return None
    dates = pd.to_datetime(panel["date"], errors="coerce")
    return dates.dt.strftime("%Y-%m-%d").where(dates.notna())


def add_factor_scores(panel: pd.DataFrame) -> pd.DataFrame:
    """Add canonical factor scores and 12-1 momentum where inputs exist."""
    if panel.empty:
        return panel.copy()
    out = panel.copy()
    group = _group_key(out)

    if "ret252d" in out.columns and "ret21d" in out.columns and "momentum_12_1" not in out.columns:
        ret252 = _numeric(out, "ret252d")
        ret21 = _numeric(out, "ret21d")
        out["momentum_12_1"] = (1 + ret252) / (1 + ret21.replace(-1, np.nan)) - 1

    value_parts = []
    for col in ("pe", "pb", "ev_ebit", "ev_ebitda"):
        if col in out.columns:
            value_parts.append(_pct_rank(_numeric(out, col), higher_is_better=False, group=group))
    for col in ("dividend_yield",):
        if col in out.columns:
            value_parts.append(_pct_rank(_numeric(out, col), higher_is_better=True, group=group))
    computed_value = _mean_available(value_parts, out.index)
    if "valuation_score" not in out.columns or _numeric(out, "valuation_score").notna().sum() == 0:
        out["valuation_score"] = computed_value
    out["value_score"] = _mean_available([_numeric(out, "valuation_score"), computed_value], out.index)

    quality_parts = []
    for col in ("roe", "roic", "gross_margin", "operating_margin"):
        if col in out.columns:
            quality_parts.append(_pct_rank(_numeric(out, col), higher_is_better=True, group=group))
    if "debt_to_equity" in out.columns:
        quality_parts.append(_pct_rank(_numeric(out, "debt_to_equity"), higher_is_better=False, group=group))
    computed_quality = _mean_available(quality_parts, out.index)
    if "quality_score" not in out.columns or _numeric(out, "quality_score").notna().sum() == 0:
        out["quality_score"] = computed_quality

    momentum_source = "momentum_12_1" if "momentum_12_1" in out.columns else "ret252d" if "ret252d" in out.columns else ""
    if momentum_source:
        out["momentum_12_1_score"] = _pct_rank(_numeric(out, momentum_source), higher_is_better=True, group=group)
        if "momentum_score" not in out.columns or _numeric(out, "momentum_score").notna().sum() == 0:
            out["momentum_score"] = out["momentum_12_1_score"]

    risk_parts = []
    for col in ("vol252d", "vol126d", "vol63d", "volatility", "beta", "max_drawdown"):
        if col in out.columns:
            risk_parts.append(_pct_rank(_numeric(out, col).abs(), higher_is_better=False, group=group))
    computed_risk = _mean_available(risk_parts, out.index)
    if "risk_score" not in out.columns or _numeric(out, "risk_score").notna().sum() == 0:
        out["risk_score"] = computed_risk

    size_source = "log_market_value" if "log_market_value" in out.columns else "market_value" if "market_value" in out.columns else ""
    if size_source:
        out["size_score"] = _pct_rank(_numeric(out, size_source), higher_is_better=True, group=group)

    growth_parts = []
    for col in ("revenue_growth", "revenue_cagr", "sales_cagr", "eps_growth"):
        if col in out.columns:
            growth_parts.append(_pct_rank(_numeric(out, col), higher_is_better=True, group=group))
    out["growth_score"] = _mean_available(growth_parts, out.index)

    score_cols = [col for col in ("value_score", "quality_score", "momentum_score", "risk_score", "size_score", "growth_score") if col in out.columns]
    out["factor_composite_score"] = out[score_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True) if score_cols else np.nan
    return out


def feature_columns_for_blocks(
    panel: pd.DataFrame,
    blocks: Iterable[str] | None = None,
    target: str | None = None,
    min_non_null: int = 5,
    include_extra_numeric: bool = False,
) -> list[str]:
    """Select model features from declared factor blocks with leakage controls."""
    if panel.empty:
        return []
    selected_blocks = list(blocks or FACTOR_BLOCKS)
    candidates: list[str] = []
    for block_id in selected_blocks:
        block = FACTOR_BLOCKS.get(str(block_id))
        if block:
            candidates.extend(block.columns)
    candidates.extend(FACTOR_SCORE_COLUMNS)
    if include_extra_numeric:
        candidates.extend(panel.select_dtypes("number").columns.tolist())
    out: list[str] = []
    for col in dict.fromkeys(candidates):
        if col not in panel.columns or is_leakage_feature(col, target=target):
            continue
        values = pd.to_numeric(panel[col], errors="coerce")
        if values.notna().sum() >= int(min_non_null):
            out.append(col)
    return out
