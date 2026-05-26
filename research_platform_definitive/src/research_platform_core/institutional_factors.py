"""Institutional equity factor helpers.

These functions implement the compact equity factor set that complements the
existing Gen.is.IA factor panel: short-term reversal, profitability, investment,
accruals, payout and distress.  They are deliberately DataFrame-first and
fail-soft: missing source columns produce NaN columns instead of exceptions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _num(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce") if column in frame.columns else pd.Series(np.nan, index=frame.index)


def _first(frame: pd.DataFrame, candidates: tuple[str, ...]) -> pd.Series:
    for column in candidates:
        if column in frame.columns:
            return _num(frame, column)
    return pd.Series(np.nan, index=frame.index)


def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denom = pd.to_numeric(denominator, errors="coerce").replace(0, np.nan)
    return pd.to_numeric(numerator, errors="coerce") / denom


def _date_sort(frame: pd.DataFrame) -> pd.DataFrame:
    if {"ticker", "date"}.issubset(frame.columns):
        out = frame.copy()
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        return out.sort_values(["ticker", "date"])
    if {"ticker", "report_date"}.issubset(frame.columns):
        out = frame.copy()
        out["report_date"] = pd.to_datetime(out["report_date"], errors="coerce")
        return out.sort_values(["ticker", "report_date"])
    return frame.copy()


def short_term_reversal(panel: pd.DataFrame, price_col: str = "price", window: int = 21) -> pd.Series:
    """Contrarian one-month reversal signal, `-return_1m`, using lagged prices.

    Reference: Jegadeesh (1990), Bartram et al. (2021).
    """
    if panel.empty or price_col not in panel.columns:
        return pd.Series(np.nan, index=panel.index)
    frame = _date_sort(panel)
    price = pd.to_numeric(frame[price_col], errors="coerce")
    if "ticker" in frame.columns:
        grouped_return = price.groupby(frame["ticker"]).pct_change(window)
        values = -grouped_return.groupby(frame["ticker"]).shift(1)
    else:
        values = -price.pct_change(window).shift(1)
    return values.reindex(frame.index).reindex(panel.index)


def gross_profitability(fundamentals: pd.DataFrame) -> pd.Series:
    """Gross profitability: `(revenue - COGS) / total_assets`.

    Reference: Novy-Marx (2013).
    """
    revenue = _first(fundamentals, ("revenue", "total_revenue", "sales"))
    cogs = _first(fundamentals, ("cost_of_goods_sold", "cogs", "cost_of_revenue"))
    gross_profit = _first(fundamentals, ("gross_profit",))
    gross_profit = gross_profit.where(gross_profit.notna(), revenue - cogs)
    assets = _first(fundamentals, ("total_assets", "assets"))
    return _safe_div(gross_profit, assets)


def investment_factor(fundamentals: pd.DataFrame) -> pd.Series:
    """CMA-style asset growth: `delta(total_assets) / lag(total_assets)`.

    Reference: Fama and French (2015), Cooper et al. (2008).
    """
    if fundamentals.empty:
        return pd.Series(dtype=float)
    frame = _date_sort(fundamentals)
    assets = _first(frame, ("total_assets", "assets"))
    if "ticker" in frame.columns:
        values = assets.groupby(frame["ticker"]).pct_change()
    else:
        values = assets.pct_change()
    return values.reindex(frame.index).reindex(fundamentals.index)


def accruals(fundamentals: pd.DataFrame) -> pd.Series:
    """Accruals ratio: `(net_income - cash_from_operations) / total_assets`.

    Reference: Sloan (1996).
    """
    net_income = _first(fundamentals, ("net_income", "netIncome"))
    cfo = _first(fundamentals, ("cash_from_operations", "operating_cash_flow", "cash_flow_from_operations", "cfo"))
    assets = _first(fundamentals, ("total_assets", "assets"))
    return _safe_div(net_income - cfo, assets)


def cash_profitability(fundamentals: pd.DataFrame) -> pd.Series:
    """Cash-based profitability: `cash_from_operations / total_assets`."""
    cfo = _first(fundamentals, ("cash_from_operations", "operating_cash_flow", "cash_flow_from_operations", "cfo"))
    assets = _first(fundamentals, ("total_assets", "assets"))
    return _safe_div(cfo, assets)


def earnings_quality(fundamentals: pd.DataFrame) -> pd.Series:
    """Cash earnings quality: `cash_from_operations / abs(net_income)`."""
    cfo = _first(fundamentals, ("cash_from_operations", "operating_cash_flow", "cash_flow_from_operations", "cfo"))
    net_income = _first(fundamentals, ("net_income", "netIncome")).abs()
    return _safe_div(cfo, net_income)


def net_payout_yield(fundamentals: pd.DataFrame) -> pd.Series:
    """Net payout yield from dividends and buybacks scaled by market cap."""
    dividends = _first(fundamentals, ("dividends_paid", "dividends", "cash_dividends_paid")).abs()
    buybacks = _first(fundamentals, ("buybacks", "share_repurchases", "repurchase_of_stock")).abs()
    issuance = _first(fundamentals, ("stock_issuance", "issuance_of_stock", "common_stock_issued")).abs()
    market_cap = _first(fundamentals, ("market_cap", "market_value", "marketcap"))
    return _safe_div(dividends + buybacks - issuance.fillna(0.0), market_cap)


def distress_risk(fundamentals: pd.DataFrame) -> pd.Series:
    """Altman-style distress score proxy; higher values indicate lower distress."""
    total_assets = _first(fundamentals, ("total_assets", "assets"))
    current_assets = _first(fundamentals, ("current_assets",))
    current_liabilities = _first(fundamentals, ("current_liabilities",))
    working_capital = _first(fundamentals, ("working_capital",))
    working_capital = working_capital.where(working_capital.notna(), current_assets - current_liabilities)
    retained_earnings = _first(fundamentals, ("retained_earnings",))
    ebit = _first(fundamentals, ("ebit", "operating_income"))
    market_cap = _first(fundamentals, ("market_cap", "market_value", "marketcap"))
    liabilities = _first(fundamentals, ("total_liabilities", "liabilities"))
    revenue = _first(fundamentals, ("revenue", "total_revenue", "sales"))
    return (
        1.2 * _safe_div(working_capital, total_assets)
        + 1.4 * _safe_div(retained_earnings, total_assets)
        + 3.3 * _safe_div(ebit, total_assets)
        + 0.6 * _safe_div(market_cap, liabilities)
        + 1.0 * _safe_div(revenue, total_assets)
    )


def idiosyncratic_volatility(
    returns: pd.DataFrame | pd.Series,
    market_return: pd.Series,
    window: int = 252,
) -> pd.DataFrame | pd.Series:
    """Rolling CAPM residual volatility, shifted one observation.

    Reference: Ang et al. (2006).  Inputs are returns, not prices.
    """
    y = returns.apply(pd.to_numeric, errors="coerce") if isinstance(returns, pd.DataFrame) else pd.to_numeric(returns, errors="coerce")
    m = pd.to_numeric(market_return, errors="coerce").reindex(y.index)
    var_m = m.rolling(window, min_periods=max(20, min(window, 60))).var()
    if isinstance(y, pd.DataFrame):
        out = pd.DataFrame(index=y.index, columns=y.columns, dtype=float)
        for col in y.columns:
            cov = y[col].rolling(window, min_periods=max(20, min(window, 60))).cov(m)
            beta = cov / var_m.replace(0, np.nan)
            residual = y[col] - beta * m
            out[col] = residual.rolling(window, min_periods=max(20, min(window, 60))).std() * np.sqrt(252)
        return out.shift(1)
    cov = y.rolling(window, min_periods=max(20, min(window, 60))).cov(m)
    beta = cov / var_m.replace(0, np.nan)
    residual = y - beta * m
    return (residual.rolling(window, min_periods=max(20, min(window, 60))).std() * np.sqrt(252)).shift(1)


def compute_institutional_equity_factors(panel: pd.DataFrame) -> pd.DataFrame:
    """Append institutional equity factor columns without mutating input."""
    out = panel.copy()
    price_col = "price" if "price" in out.columns else "close" if "close" in out.columns else ""
    out["short_term_reversal"] = short_term_reversal(out, price_col=price_col) if price_col else np.nan
    out["gross_profitability"] = gross_profitability(out)
    out["investment_factor"] = investment_factor(out)
    out["accruals"] = accruals(out)
    out["cash_profitability"] = cash_profitability(out)
    out["earnings_quality"] = earnings_quality(out)
    out["net_payout_yield"] = net_payout_yield(out)
    out["distress_risk"] = distress_risk(out)
    return out


__all__ = [
    "accruals",
    "cash_profitability",
    "compute_institutional_equity_factors",
    "distress_risk",
    "earnings_quality",
    "gross_profitability",
    "idiosyncratic_volatility",
    "investment_factor",
    "net_payout_yield",
    "short_term_reversal",
]
