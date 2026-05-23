"""Reusable utilities for the company valuation notebook.

The notebook remains self-contained for Colab, but these helper-style functions
mirror the core logic so the workflow can later be migrated into a Python package
without rewriting the research code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def normalize_ticker(value: object) -> str | None:
    """Normalize ticker symbols for Yahoo/FMP-style market data joins."""
    if pd.isna(value):
        return None
    return str(value).strip().upper().replace("/", "-")


def safe_rank(series: pd.Series, ascending: bool = True) -> pd.Series:
    """Percentile rank with robust NaN handling."""
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() == 0:
        return pd.Series(np.nan, index=series.index)
    return values.rank(pct=True, ascending=ascending)


def latest_per_ticker(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the latest row for each ticker from a date/ticker panel."""
    if frame.empty:
        return frame.copy()
    required = {"ticker", "date"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"latest_per_ticker missing required columns: {sorted(missing)}")
    return (
        frame.sort_values(["ticker", "date"])
        .groupby("ticker", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )


def build_output_dirs(root: str | Path) -> dict[str, Path]:
    """Create and return canonical output directories for valuation artifacts."""
    root = Path(root)
    dirs = {
        "root": root,
        "tables": root / "tables",
        "figures": root / "figures",
        "dashboard": root / "dashboard",
        "logs": root / "logs",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def performance_stats_from_returns(
    returns: pd.Series | list[float],
    periods_per_year: int = 12,
) -> dict[str, float]:
    """Compute compact portfolio-performance statistics."""
    returns = pd.Series(returns, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()
    if returns.empty:
        return {
            "annual_return": np.nan,
            "annual_volatility": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "hit_rate": np.nan,
        }
    annual_return = (1 + returns.mean()) ** periods_per_year - 1
    annual_volatility = returns.std() * np.sqrt(periods_per_year)
    equity = (1 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1
    sharpe = annual_return / annual_volatility if annual_volatility and not np.isclose(annual_volatility, 0) else np.nan
    return {
        "annual_return": float(annual_return),
        "annual_volatility": float(annual_volatility),
        "sharpe": float(sharpe) if pd.notna(sharpe) else np.nan,
        "max_drawdown": float(drawdown.min()) if not drawdown.empty else np.nan,
        "hit_rate": float((returns > 0).mean()),
    }


def estimate_intrinsic_value(
    row: Mapping[str, object] | pd.Series,
    peer_pe: float = np.nan,
    peer_pb: float = np.nan,
    discount_rate: float = 0.09,
    terminal_growth: float = 0.025,
    forecast_years: int = 10,
) -> pd.Series:
    """Estimate fair value using FCF, excess-returns, and relative valuation fallbacks."""
    price = row.get("adj_close", np.nan)
    shares = row.get("shares_outstanding", np.nan)
    free_cash_flow = row.get("free_cash_flow", np.nan)
    revenue_growth = row.get("revenue_growth", np.nan)
    roe = row.get("roe", np.nan)
    book_value_per_share = row.get("book_value_per_share", np.nan)
    eps = row.get("eps", np.nan)

    growth = float(np.clip(np.nan_to_num(revenue_growth, nan=0.04), -0.05, 0.15))
    fair_values: list[float] = []
    model_used: str | None = None

    if (
        pd.notna(free_cash_flow)
        and pd.notna(shares)
        and shares > 0
        and free_cash_flow > 0
        and discount_rate > terminal_growth
    ):
        current_fcf = float(free_cash_flow)
        projected_fcf = []
        for g in np.linspace(growth, terminal_growth, forecast_years):
            current_fcf *= 1 + g
            projected_fcf.append(current_fcf)
        pv_fcf = sum(cf / ((1 + discount_rate) ** (idx + 1)) for idx, cf in enumerate(projected_fcf))
        terminal_value = projected_fcf[-1] * (1 + terminal_growth) / (discount_rate - terminal_growth)
        pv_terminal = terminal_value / ((1 + discount_rate) ** forecast_years)
        fair_values.append((pv_fcf + pv_terminal) / shares)
        model_used = "two_stage_fcf"

    if pd.notna(book_value_per_share) and book_value_per_share > 0 and pd.notna(roe) and discount_rate > terminal_growth:
        excess_return_value = book_value_per_share + ((roe - discount_rate) * book_value_per_share) / (discount_rate - terminal_growth)
        if np.isfinite(excess_return_value) and excess_return_value > 0:
            fair_values.append(float(excess_return_value))
            model_used = model_used or "excess_returns"

    relative_values = []
    if pd.notna(eps) and eps > 0 and pd.notna(peer_pe) and peer_pe > 0:
        relative_values.append(eps * peer_pe)
    if pd.notna(book_value_per_share) and book_value_per_share > 0 and pd.notna(peer_pb) and peer_pb > 0:
        relative_values.append(book_value_per_share * peer_pb)
    if relative_values:
        fair_values.append(float(np.nanmedian(relative_values)))
        model_used = model_used or "relative_value"

    fair_value = float(np.nanmedian(fair_values)) if fair_values else np.nan
    upside = fair_value / price - 1 if pd.notna(fair_value) and pd.notna(price) and price > 0 else np.nan
    return pd.Series(
        {
            "fair_value_estimate": fair_value,
            "upside_to_fair_value": upside,
            "fair_value_model": model_used or "insufficient_data",
        }
    )
