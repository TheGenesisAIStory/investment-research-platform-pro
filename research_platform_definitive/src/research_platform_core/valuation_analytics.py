"""Valuation analytics helpers for DCF, WACC, multiples, EVA/DDM and comps."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _num(payload: dict[str, Any] | pd.Series, key: str, default: float = np.nan) -> float:
    try:
        value = payload.get(key, default) if hasattr(payload, "get") else default
        return float(value) if value not in (None, "") and pd.notna(value) else default
    except Exception:
        return default


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den and pd.notna(num) and pd.notna(den) else np.nan


def compute_wacc(
    data: dict[str, Any] | pd.Series,
    *,
    risk_free_rate: float = 0.04,
    equity_risk_premium: float = 0.055,
    default_cost_of_debt: float = 0.05,
    default_tax_rate: float = 0.21,
) -> dict[str, float]:
    market_cap = _num(data, "market_cap", _num(data, "marketCap", 0.0))
    total_debt = _num(data, "total_debt", _num(data, "totalDebt", 0.0))
    beta = _num(data, "beta", 1.0)
    interest_expense = abs(_num(data, "interest_expense", _num(data, "interestExpense", np.nan)))
    pretax_income = _num(data, "pretax_income", _num(data, "pretaxIncome", np.nan))
    tax_expense = abs(_num(data, "tax_expense", _num(data, "taxExpense", np.nan)))
    tax_rate = _safe_div(tax_expense, pretax_income) if pretax_income and pretax_income > 0 else default_tax_rate
    tax_rate = float(np.clip(tax_rate if pd.notna(tax_rate) else default_tax_rate, 0.0, 0.5))
    cost_of_equity = risk_free_rate + beta * equity_risk_premium
    cost_of_debt = _safe_div(interest_expense, total_debt)
    if pd.isna(cost_of_debt) or cost_of_debt <= 0:
        cost_of_debt = default_cost_of_debt
    enterprise_capital = market_cap + total_debt
    equity_weight = _safe_div(market_cap, enterprise_capital) if enterprise_capital else 1.0
    debt_weight = _safe_div(total_debt, enterprise_capital) if enterprise_capital else 0.0
    wacc = equity_weight * cost_of_equity + debt_weight * cost_of_debt * (1.0 - tax_rate)
    return {
        "market_cap": market_cap,
        "total_debt": total_debt,
        "beta": beta,
        "tax_rate": tax_rate,
        "cost_of_equity": cost_of_equity,
        "cost_of_debt": cost_of_debt,
        "equity_weight": equity_weight,
        "debt_weight": debt_weight,
        "wacc": float(wacc),
    }


def compute_dcf_valuation(
    data: dict[str, Any] | pd.Series,
    *,
    current_price: float | None = None,
    wacc: float | None = None,
    growth_stage1: float = 0.06,
    terminal_growth: float = 0.025,
    n_years: int = 5,
    risk_free_rate: float = 0.04,
    equity_risk_premium: float = 0.055,
) -> pd.DataFrame:
    wacc_payload = compute_wacc(data, risk_free_rate=risk_free_rate, equity_risk_premium=equity_risk_premium)
    discount_rate = float(wacc if wacc is not None else wacc_payload["wacc"])
    if discount_rate <= terminal_growth:
        discount_rate = terminal_growth + 0.01
    fcf0 = _num(data, "free_cash_flow", _num(data, "freeCashflow", _num(data, "operating_cash_flow", 0.0)))
    cash = _num(data, "cash", _num(data, "total_cash", _num(data, "totalCash", 0.0)))
    debt = _num(data, "total_debt", _num(data, "totalDebt", 0.0))
    shares = _num(data, "shares_outstanding", _num(data, "sharesOutstanding", 1.0))
    price = float(current_price if current_price is not None else _num(data, "current_price", _num(data, "price", np.nan)))
    rows: list[dict[str, float | str]] = []
    fcf_values: list[float] = []
    for year in range(1, int(n_years) + 1):
        fcf = fcf0 * ((1.0 + growth_stage1) ** year)
        pv = fcf / ((1.0 + discount_rate) ** year)
        fcf_values.append(fcf)
        rows.append({"component": f"PV FCF Year {year}", "year": year, "fcf_projection": fcf, "present_value": pv})
    terminal_fcf = fcf_values[-1] * (1.0 + terminal_growth) if fcf_values else 0.0
    terminal_value = terminal_fcf / (discount_rate - terminal_growth) if discount_rate > terminal_growth else np.nan
    pv_terminal = terminal_value / ((1.0 + discount_rate) ** int(n_years)) if pd.notna(terminal_value) else np.nan
    enterprise_value = float(np.nansum([row["present_value"] for row in rows]) + pv_terminal)
    equity_value = enterprise_value - debt + cash
    intrinsic_price = _safe_div(equity_value, shares)
    upside = _safe_div(intrinsic_price - price, price)
    rows.append({"component": "PV Terminal Value", "year": int(n_years), "fcf_projection": terminal_fcf, "present_value": pv_terminal})
    rows.append({"component": "Enterprise Value DCF", "year": 0, "fcf_projection": np.nan, "present_value": enterprise_value})
    rows.append({"component": "Equity Value DCF", "year": 0, "fcf_projection": np.nan, "present_value": equity_value})
    rows.append({"component": "Intrinsic Price DCF", "year": 0, "fcf_projection": np.nan, "present_value": intrinsic_price})
    out = pd.DataFrame(rows)
    out["wacc"] = discount_rate
    out["terminal_growth"] = terminal_growth
    out["growth_stage1"] = growth_stage1
    out["terminal_value"] = terminal_value
    out["enterprise_value_dcf"] = enterprise_value
    out["equity_value_dcf"] = equity_value
    out["intrinsic_price_dcf"] = intrinsic_price
    out["upside_dcf"] = upside
    return out


def compute_market_multiples(data: dict[str, Any] | pd.Series) -> dict[str, float]:
    price = _num(data, "price", _num(data, "current_price", np.nan))
    market_cap = _num(data, "market_cap", _num(data, "marketCap", np.nan))
    enterprise_value = _num(data, "enterprise_value", _num(data, "enterpriseValue", market_cap + _num(data, "total_debt", 0.0) - _num(data, "cash", 0.0)))
    revenue = _num(data, "revenue_ttm", _num(data, "totalRevenue", np.nan))
    ebitda = _num(data, "ebitda_ttm", _num(data, "ebitda", np.nan))
    ebit = _num(data, "ebit_ttm", _num(data, "ebit", np.nan))
    fcf = _num(data, "free_cash_flow", _num(data, "freeCashflow", np.nan))
    ocf = _num(data, "operating_cash_flow", _num(data, "operatingCashflow", np.nan))
    eps = _num(data, "eps_ttm", _num(data, "trailingEps", np.nan))
    book_value = _num(data, "book_value_per_share", np.nan)
    shares = _num(data, "shares_outstanding", _num(data, "sharesOutstanding", np.nan))
    total_equity = _num(data, "total_equity", _num(data, "totalStockholderEquity", np.nan))
    if pd.isna(book_value):
        book_value = _safe_div(total_equity, shares)
    dividend = _num(data, "annual_dividend", _num(data, "dividendRate", np.nan))
    net_income = _num(data, "net_income", _num(data, "netIncomeToCommon", np.nan))
    earnings_growth = _num(data, "earnings_growth_rate_5y", _num(data, "earningsGrowth", np.nan))
    return {
        "pe_ratio": _safe_div(price, eps),
        "pb_ratio": _safe_div(price, book_value),
        "ps_ratio": _safe_div(market_cap, revenue),
        "pcf_ratio": _safe_div(market_cap, ocf),
        "ev_ebitda": _safe_div(enterprise_value, ebitda),
        "ev_ebit": _safe_div(enterprise_value, ebit),
        "ev_sales": _safe_div(enterprise_value, revenue),
        "ev_fcf": _safe_div(enterprise_value, fcf),
        "dividend_yield": _safe_div(dividend, price),
        "payout_ratio": _safe_div(dividend * shares, net_income),
        "peg_ratio": _safe_div(_safe_div(price, eps), earnings_growth * 100 if earnings_growth and abs(earnings_growth) < 1 else earnings_growth),
    }


def compute_eva_residual_income(data: dict[str, Any] | pd.Series, *, wacc: float | None = None, terminal_growth: float = 0.025) -> dict[str, float]:
    wacc_value = float(wacc if wacc is not None else compute_wacc(data)["wacc"])
    ebit = _num(data, "ebit_ttm", _num(data, "ebit", np.nan))
    tax_rate = compute_wacc(data)["tax_rate"]
    debt = _num(data, "total_debt", _num(data, "totalDebt", 0.0))
    cash = _num(data, "cash", _num(data, "totalCash", 0.0))
    equity = _num(data, "total_equity", _num(data, "totalStockholderEquity", np.nan))
    invested_capital = equity + debt - cash
    nopat = ebit * (1.0 - tax_rate)
    roic = _safe_div(nopat, invested_capital)
    roe = _num(data, "roe", _safe_div(_num(data, "net_income", np.nan), equity))
    eva = nopat - wacc_value * invested_capital
    intrinsic_pb = 1.0 + _safe_div(roe - wacc_value, wacc_value - terminal_growth) if wacc_value > terminal_growth else np.nan
    return {
        "nopat": nopat,
        "invested_capital": invested_capital,
        "roic": roic,
        "wacc": wacc_value,
        "wacc_spread": roic - wacc_value if pd.notna(roic) else np.nan,
        "eva": eva,
        "intrinsic_pb": intrinsic_pb,
    }


def compute_ddm_valuation(data: dict[str, Any] | pd.Series, *, cost_of_equity: float | None = None, growth: float = 0.025, stage1_growth: float = 0.06, n_years: int = 5) -> dict[str, float]:
    wacc_payload = compute_wacc(data)
    re = float(cost_of_equity if cost_of_equity is not None else wacc_payload["cost_of_equity"])
    dividend = _num(data, "annual_dividend", _num(data, "dividendRate", 0.0))
    if re <= growth:
        re = growth + 0.01
    d1 = dividend * (1.0 + growth)
    gordon = d1 / (re - growth) if re > growth else np.nan
    pv_dividends = 0.0
    current_div = dividend
    for year in range(1, int(n_years) + 1):
        current_div *= 1.0 + stage1_growth
        pv_dividends += current_div / ((1.0 + re) ** year)
    terminal = current_div * (1.0 + growth) / (re - growth)
    two_stage = pv_dividends + terminal / ((1.0 + re) ** int(n_years))
    return {"intrinsic_price_ddm": gordon, "intrinsic_price_ddm_2stage": two_stage, "cost_of_equity": re, "dividend_growth": growth}


def compute_asset_based_valuation(data: dict[str, Any] | pd.Series) -> dict[str, float]:
    assets = _num(data, "total_assets", _num(data, "totalAssets", np.nan))
    liabilities = _num(data, "total_liabilities", _num(data, "totalLiab", np.nan))
    equity = _num(data, "total_equity", _num(data, "totalStockholderEquity", assets - liabilities))
    intangibles = _num(data, "intangibles", _num(data, "intangibleAssets", 0.0))
    goodwill = _num(data, "goodwill", 0.0)
    shares = _num(data, "shares_outstanding", _num(data, "sharesOutstanding", np.nan))
    current_assets = _num(data, "current_assets", _num(data, "totalCurrentAssets", np.nan))
    ppe = _num(data, "ppe", _num(data, "propertyPlantEquipment", np.nan))
    nav = assets - liabilities - intangibles - goodwill
    liquidation = current_assets * 0.8 + ppe * 0.5 - liabilities
    return {
        "book_value_per_share": _safe_div(equity, shares),
        "tangible_book_per_share": _safe_div(equity - intangibles - goodwill, shares),
        "net_asset_value": nav,
        "liquidation_value_approx": liquidation,
    }


def compute_comps_valuation(company_row: dict[str, Any] | pd.Series, peer_frame: pd.DataFrame, sector_col: str = "sector") -> dict[str, float]:
    multiples = compute_market_multiples(company_row)
    sector = str(company_row.get(sector_col, "") if hasattr(company_row, "get") else "")
    peers = peer_frame.copy() if peer_frame is not None else pd.DataFrame()
    if not peers.empty and sector_col in peers.columns and sector:
        peers = peers[peers[sector_col].astype(str).eq(sector)]
    if peers.empty:
        return {**multiples, "sector_pe_median": np.nan, "sector_ev_ebitda_median": np.nan}
    peer_mults = pd.DataFrame([compute_market_multiples(row) for _, row in peers.iterrows()])
    pe_med = pd.to_numeric(peer_mults.get("pe_ratio"), errors="coerce").median()
    ev_ebitda_med = pd.to_numeric(peer_mults.get("ev_ebitda"), errors="coerce").median()
    eps = _num(company_row, "eps_ttm", _num(company_row, "trailingEps", np.nan))
    ebitda = _num(company_row, "ebitda_ttm", _num(company_row, "ebitda", np.nan))
    debt = _num(company_row, "total_debt", _num(company_row, "totalDebt", 0.0))
    cash = _num(company_row, "cash", _num(company_row, "totalCash", 0.0))
    shares = _num(company_row, "shares_outstanding", _num(company_row, "sharesOutstanding", np.nan))
    price = _num(company_row, "price", _num(company_row, "current_price", np.nan))
    implied_pe = eps * pe_med if pd.notna(pe_med) else np.nan
    implied_ev_ebitda = _safe_div(ebitda * ev_ebitda_med - debt + cash, shares) if pd.notna(ev_ebitda_med) else np.nan
    implied_values = [value for value in [implied_pe, implied_ev_ebitda] if pd.notna(value)]
    implied_blend = float(np.mean(implied_values)) if implied_values else np.nan
    return {
        **multiples,
        "sector_pe_median": pe_med,
        "sector_ev_ebitda_median": ev_ebitda_med,
        "implied_price_pe": implied_pe,
        "implied_price_ev_ebitda": implied_ev_ebitda,
        "premium_discount_to_sector": _safe_div(price - implied_blend, implied_blend),
    }


def fair_value_summary(models: dict[str, float], current_price: float | None = None) -> pd.DataFrame:
    rows = [{"model": key, "fair_value": value} for key, value in models.items() if pd.notna(value)]
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    price = float(current_price) if current_price is not None and pd.notna(current_price) else np.nan
    frame["current_price"] = price
    frame["upside"] = (frame["fair_value"] - price) / price if price else np.nan
    frame["range_low"] = frame["fair_value"].min()
    frame["range_high"] = frame["fair_value"].max()
    return frame
