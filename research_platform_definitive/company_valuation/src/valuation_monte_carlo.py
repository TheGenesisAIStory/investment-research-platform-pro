"""DCF Monte Carlo research extension.

This module is intentionally small and notebook-friendly. It adds a reusable
simulation pattern to the existing valuation platform without creating a
separate DCF project or changing the core valuation methodology.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


MARKET_PARAMETER_PRESETS: dict[str, dict[str, float | str]] = {
    "US": {
        "currency": "USD",
        "risk_free_rate": 0.043,
        "market_risk_premium": 0.050,
        "tax_rate": 0.21,
        "terminal_growth_mid": 0.025,
        "wacc_floor": 0.055,
        "wacc_cap": 0.150,
    },
    "UK": {
        "currency": "GBP",
        "risk_free_rate": 0.040,
        "market_risk_premium": 0.055,
        "tax_rate": 0.25,
        "terminal_growth_mid": 0.020,
        "wacc_floor": 0.060,
        "wacc_cap": 0.155,
    },
    "DE": {
        "currency": "EUR",
        "risk_free_rate": 0.028,
        "market_risk_premium": 0.055,
        "tax_rate": 0.30,
        "terminal_growth_mid": 0.018,
        "wacc_floor": 0.055,
        "wacc_cap": 0.150,
    },
    "IT": {
        "currency": "EUR",
        "risk_free_rate": 0.038,
        "market_risk_premium": 0.060,
        "tax_rate": 0.28,
        "terminal_growth_mid": 0.015,
        "wacc_floor": 0.060,
        "wacc_cap": 0.165,
    },
    "JP": {
        "currency": "JPY",
        "risk_free_rate": 0.012,
        "market_risk_premium": 0.055,
        "tax_rate": 0.30,
        "terminal_growth_mid": 0.010,
        "wacc_floor": 0.040,
        "wacc_cap": 0.130,
    },
    "HK": {
        "currency": "HKD",
        "risk_free_rate": 0.038,
        "market_risk_premium": 0.060,
        "tax_rate": 0.165,
        "terminal_growth_mid": 0.020,
        "wacc_floor": 0.060,
        "wacc_cap": 0.170,
    },
    "India": {
        "currency": "INR",
        "risk_free_rate": 0.070,
        "market_risk_premium": 0.065,
        "tax_rate": 0.25,
        "terminal_growth_mid": 0.040,
        "wacc_floor": 0.080,
        "wacc_cap": 0.200,
    },
    "China": {
        "currency": "CNY",
        "risk_free_rate": 0.025,
        "market_risk_premium": 0.065,
        "tax_rate": 0.25,
        "terminal_growth_mid": 0.030,
        "wacc_floor": 0.065,
        "wacc_cap": 0.180,
    },
}


@dataclass
class DCFMonteCarloConfig:
    region: str = "US"
    simulations: int = 5000
    horizon_years: int = 5
    beta: float = 1.0
    cost_of_debt: float = 0.045
    debt_to_capital: float = 0.25
    base_growth: float = 0.05
    growth_sigma: float = 0.025
    terminal_growth_sigma: float = 0.006
    wacc_sigma: float = 0.012
    seed: int = 42


@dataclass
class DCFInput:
    ticker: str
    current_price: float
    free_cash_flow: float
    shares_outstanding: float
    net_debt: float = 0.0
    revenue: float | None = None
    currency: str | None = None


@dataclass
class ResidualIncomeInput:
    ticker: str
    current_price: float
    book_value_per_share: float
    roe: float
    cost_of_equity: float
    currency: str | None = None


@dataclass
class EVAScenarioInput:
    ticker: str
    current_price: float
    nopat: float
    invested_capital: float
    wacc: float
    shares_outstanding: float
    net_debt: float = 0.0
    currency: str | None = None


def market_parameters_frame() -> pd.DataFrame:
    """Return region-level assumptions as a table for UI/notebook display."""
    rows = [{"region": region, **values} for region, values in MARKET_PARAMETER_PRESETS.items()]
    return pd.DataFrame(rows)


def infer_region(ticker: str | None, fallback: str = "US") -> str:
    ticker = str(ticker or "").upper()
    if ticker.endswith(".MI"):
        return "IT"
    if ticker.endswith(".DE"):
        return "DE"
    if ticker.endswith(".L"):
        return "UK"
    if ticker.endswith(".T"):
        return "JP"
    if ticker.endswith(".HK"):
        return "HK"
    if ticker.endswith(".SS") or ticker.endswith(".SZ"):
        return "China"
    return fallback


def _first_numeric(row: pd.Series, names: list[str], default: float = np.nan) -> float:
    lower = {str(c).lower(): c for c in row.index}
    for name in names:
        col = lower.get(name.lower())
        if col is None:
            continue
        value = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]
        if pd.notna(value):
            return float(value)
    return default


def build_dcf_input_from_frame(frame: pd.DataFrame, ticker: str | None = None) -> DCFInput | None:
    """Extract conservative DCF inputs from an existing valuation/cross-section table."""
    if frame is None or frame.empty:
        return None
    data = frame.copy()
    if ticker and "ticker" in data.columns:
        filtered = data[data["ticker"].astype(str).str.upper().eq(str(ticker).upper())]
        if not filtered.empty:
            data = filtered
    row = data.iloc[0]
    inferred_ticker = str(row.get("ticker", ticker or "UNKNOWN")).upper()
    price = _first_numeric(row, ["current_price", "adj_close", "close", "last_price", "price", "market_value"])
    fcf = _first_numeric(row, ["free_cash_flow", "fcf", "free_cashflow", "operating_cash_flow"])
    shares = _first_numeric(row, ["shares_outstanding", "weighted_average_shares", "shares"], default=1.0)
    net_debt = _first_numeric(row, ["net_debt", "total_debt_net_cash", "debt_minus_cash"], default=0.0)
    revenue = _first_numeric(row, ["revenue", "sales", "total_revenue"], default=np.nan)
    market_cap = _first_numeric(row, ["marketcapest", "market_cap", "market_value"], default=np.nan)

    if not np.isfinite(price) and np.isfinite(market_cap) and np.isfinite(shares) and shares > 0:
        price = market_cap / shares
    if (not np.isfinite(fcf) or fcf <= 0) and np.isfinite(revenue) and revenue > 0:
        # Conservative proxy only when the notebook has revenue but no FCF.
        fcf = revenue * 0.06
    if not np.isfinite(price) or price <= 0 or not np.isfinite(fcf) or fcf <= 0:
        return None
    shares = shares if np.isfinite(shares) and shares > 0 else 1.0
    return DCFInput(
        ticker=inferred_ticker,
        current_price=float(price),
        free_cash_flow=float(fcf),
        shares_outstanding=float(shares),
        net_debt=float(net_debt) if np.isfinite(net_debt) else 0.0,
        revenue=float(revenue) if np.isfinite(revenue) else None,
        currency=str(row.get("currency", "")) or None,
    )


def build_residual_income_input_from_frame(frame: pd.DataFrame, ticker: str | None = None, region: str | None = None) -> ResidualIncomeInput | None:
    """Extract residual-income inputs from existing valuation/fundamental rows."""
    if frame is None or frame.empty:
        return None
    data = frame.copy()
    if ticker and "ticker" in data.columns:
        filtered = data[data["ticker"].astype(str).str.upper().eq(str(ticker).upper())]
        if not filtered.empty:
            data = filtered
    row = data.iloc[0]
    inferred_ticker = str(row.get("ticker", ticker or "UNKNOWN")).upper()
    price = _first_numeric(row, ["current_price", "adj_close", "close", "last_price", "price", "market_value"])
    shares = _first_numeric(row, ["shares_outstanding", "weighted_average_shares", "shares"], default=np.nan)
    market_cap = _first_numeric(row, ["marketcapest", "market_cap", "market_value"], default=np.nan)
    if not np.isfinite(price) and np.isfinite(market_cap) and np.isfinite(shares) and shares > 0:
        price = market_cap / shares

    bvps = _first_numeric(row, ["book_value_per_share", "bvps", "bookvaluepershare"], default=np.nan)
    equity = _first_numeric(row, ["total_equity", "shareholders_equity", "book_value", "equity"], default=np.nan)
    if not np.isfinite(bvps) and np.isfinite(equity) and np.isfinite(shares) and shares > 0:
        bvps = equity / shares

    roe = _first_numeric(row, ["roe", "return_on_equity", "returnonequity"], default=np.nan)
    net_income = _first_numeric(row, ["net_income", "netincome"], default=np.nan)
    if not np.isfinite(roe) and np.isfinite(net_income) and np.isfinite(equity) and equity:
        roe = net_income / equity

    resolved_region = region or infer_region(inferred_ticker)
    market = MARKET_PARAMETER_PRESETS.get(resolved_region, MARKET_PARAMETER_PRESETS["US"])
    beta = _first_numeric(row, ["beta"], default=1.0)
    cost_of_equity = _first_numeric(row, ["cost_of_equity", "costofequity"], default=np.nan)
    if not np.isfinite(cost_of_equity):
        cost_of_equity = float(market["risk_free_rate"]) + float(beta if np.isfinite(beta) else 1.0) * float(market["market_risk_premium"])

    if not all(np.isfinite(x) and x > 0 for x in [price, bvps, cost_of_equity]) or not np.isfinite(roe):
        return None
    return ResidualIncomeInput(
        ticker=inferred_ticker,
        current_price=float(price),
        book_value_per_share=float(bvps),
        roe=float(roe),
        cost_of_equity=float(cost_of_equity),
        currency=str(row.get("currency", "")) or None,
    )


def build_eva_input_from_frame(frame: pd.DataFrame, ticker: str | None = None, region: str | None = None) -> EVAScenarioInput | None:
    """Extract EVA inputs from existing valuation/fundamental rows."""
    if frame is None or frame.empty:
        return None
    data = frame.copy()
    if ticker and "ticker" in data.columns:
        filtered = data[data["ticker"].astype(str).str.upper().eq(str(ticker).upper())]
        if not filtered.empty:
            data = filtered
    row = data.iloc[0]
    inferred_ticker = str(row.get("ticker", ticker or "UNKNOWN")).upper()
    price = _first_numeric(row, ["current_price", "adj_close", "close", "last_price", "price", "market_value"])
    shares = _first_numeric(row, ["shares_outstanding", "weighted_average_shares", "shares"], default=np.nan)
    market_cap = _first_numeric(row, ["marketcapest", "market_cap", "market_value"], default=np.nan)
    if not np.isfinite(price) and np.isfinite(market_cap) and np.isfinite(shares) and shares > 0:
        price = market_cap / shares

    tax_rate = float(MARKET_PARAMETER_PRESETS.get(region or infer_region(inferred_ticker), MARKET_PARAMETER_PRESETS["US"])["tax_rate"])
    nopat = _first_numeric(row, ["nopat"], default=np.nan)
    ebit = _first_numeric(row, ["ebit", "operating_income", "operatingincome"], default=np.nan)
    if not np.isfinite(nopat) and np.isfinite(ebit):
        nopat = ebit * (1.0 - tax_rate)

    invested_capital = _first_numeric(row, ["invested_capital", "investedcapital", "capital_employed", "capitalemployed"], default=np.nan)
    equity = _first_numeric(row, ["total_equity", "shareholders_equity", "equity"], default=np.nan)
    debt = _first_numeric(row, ["total_debt", "debt"], default=0.0)
    cash = _first_numeric(row, ["cash", "cash_and_equivalents"], default=0.0)
    if not np.isfinite(invested_capital) and np.isfinite(equity):
        invested_capital = equity + (debt if np.isfinite(debt) else 0.0) - (cash if np.isfinite(cash) else 0.0)

    wacc = _first_numeric(row, ["wacc", "discount_rate", "discountrate"], default=np.nan)
    if not np.isfinite(wacc):
        cfg = DCFMonteCarloConfig(region=region or infer_region(inferred_ticker))
        market = MARKET_PARAMETER_PRESETS.get(cfg.region, MARKET_PARAMETER_PRESETS["US"])
        cost_of_equity = float(market["risk_free_rate"]) + cfg.beta * float(market["market_risk_premium"])
        wacc = (1.0 - cfg.debt_to_capital) * cost_of_equity + cfg.debt_to_capital * cfg.cost_of_debt * (1.0 - float(market["tax_rate"]))

    net_debt = _first_numeric(row, ["net_debt", "total_debt_net_cash", "debt_minus_cash"], default=0.0)
    if not all(np.isfinite(x) and x > 0 for x in [price, nopat, invested_capital, wacc, shares]):
        return None
    return EVAScenarioInput(
        ticker=inferred_ticker,
        current_price=float(price),
        nopat=float(nopat),
        invested_capital=float(invested_capital),
        wacc=float(wacc),
        shares_outstanding=float(shares),
        net_debt=float(net_debt) if np.isfinite(net_debt) else 0.0,
        currency=str(row.get("currency", "")) or None,
    )


def calculate_dcf_value(
    free_cash_flow: float,
    growth: float,
    wacc: float,
    terminal_growth: float,
    horizon_years: int,
    shares_outstanding: float,
    net_debt: float = 0.0,
) -> float:
    """Classic FCFF DCF with Gordon terminal value."""
    if wacc <= terminal_growth:
        return np.nan
    years = np.arange(1, int(horizon_years) + 1)
    fcfs = free_cash_flow * np.power(1.0 + growth, years)
    discounts = np.power(1.0 + wacc, years)
    pv_fcfs = np.sum(fcfs / discounts)
    terminal_fcf = fcfs[-1] * (1.0 + terminal_growth)
    terminal_value = terminal_fcf / (wacc - terminal_growth)
    pv_terminal = terminal_value / discounts[-1]
    equity_value = pv_fcfs + pv_terminal - net_debt
    return float(equity_value / shares_outstanding)


def calculate_residual_income_value(
    book_value_per_share: float,
    roe: float,
    cost_of_equity: float,
    terminal_growth: float,
    horizon_years: int,
    book_growth: float | None = None,
) -> float:
    """Residual income value per share using book value plus discounted excess returns."""
    if cost_of_equity <= terminal_growth:
        return np.nan
    book_growth = float(book_growth if book_growth is not None else np.clip(roe * 0.40, -0.05, 0.12))
    book = float(book_value_per_share)
    pv_ri = 0.0
    for year in range(1, int(horizon_years) + 1):
        residual_income = (roe - cost_of_equity) * book
        pv_ri += residual_income / ((1.0 + cost_of_equity) ** year)
        book *= 1.0 + book_growth
    terminal_ri = (roe - cost_of_equity) * book * (1.0 + terminal_growth)
    terminal_value = terminal_ri / (cost_of_equity - terminal_growth)
    pv_terminal = terminal_value / ((1.0 + cost_of_equity) ** int(horizon_years))
    return float(book_value_per_share + pv_ri + pv_terminal)


def run_dcf_monte_carlo(dcf_input: DCFInput, config: DCFMonteCarloConfig | None = None) -> dict[str, pd.DataFrame]:
    """Run a parameterized DCF Monte Carlo simulation."""
    config = config or DCFMonteCarloConfig(region=infer_region(dcf_input.ticker))
    market = MARKET_PARAMETER_PRESETS.get(config.region, MARKET_PARAMETER_PRESETS["US"])
    rng = np.random.default_rng(config.seed)
    simulations = max(100, int(config.simulations))

    cost_of_equity = float(market["risk_free_rate"]) + float(config.beta) * float(market["market_risk_premium"])
    base_wacc = (1.0 - config.debt_to_capital) * cost_of_equity + config.debt_to_capital * config.cost_of_debt * (1.0 - float(market["tax_rate"]))
    base_terminal = float(market["terminal_growth_mid"])

    growth = np.clip(rng.normal(config.base_growth, config.growth_sigma, simulations), -0.10, 0.25)
    terminal_growth = np.clip(rng.normal(base_terminal, config.terminal_growth_sigma, simulations), -0.02, 0.06)
    wacc = np.clip(rng.normal(base_wacc, config.wacc_sigma, simulations), float(market["wacc_floor"]), float(market["wacc_cap"]))
    terminal_growth = np.minimum(terminal_growth, wacc - 0.01)

    fair_values = np.array(
        [
            calculate_dcf_value(
                dcf_input.free_cash_flow,
                float(g),
                float(w),
                float(tg),
                config.horizon_years,
                dcf_input.shares_outstanding,
                dcf_input.net_debt,
            )
            for g, w, tg in zip(growth, wacc, terminal_growth)
        ]
    )
    simulations_df = pd.DataFrame(
        {
            "ticker": dcf_input.ticker,
            "simulation": np.arange(1, simulations + 1),
            "growth": growth,
            "wacc": wacc,
            "terminal_growth": terminal_growth,
            "fair_value": fair_values,
            "current_price": dcf_input.current_price,
            "upside": fair_values / dcf_input.current_price - 1.0,
            "region": config.region,
            "currency": dcf_input.currency or market["currency"],
        }
    ).replace([np.inf, -np.inf], np.nan).dropna(subset=["fair_value"])

    summary = {
        "ticker": dcf_input.ticker,
        "region": config.region,
        "currency": dcf_input.currency or market["currency"],
        "current_price": dcf_input.current_price,
        "simulations": len(simulations_df),
        "fair_value_p05": simulations_df["fair_value"].quantile(0.05),
        "fair_value_p25": simulations_df["fair_value"].quantile(0.25),
        "fair_value_median": simulations_df["fair_value"].median(),
        "fair_value_mean": simulations_df["fair_value"].mean(),
        "fair_value_p75": simulations_df["fair_value"].quantile(0.75),
        "fair_value_p95": simulations_df["fair_value"].quantile(0.95),
        "upside_median": simulations_df["upside"].median(),
        "probability_undervalued": float((simulations_df["fair_value"] > dcf_input.current_price).mean()),
        "base_wacc": base_wacc,
        "cost_of_equity": cost_of_equity,
        "base_terminal_growth": base_terminal,
    }
    summary_df = pd.DataFrame([summary])

    scenario_rows = []
    for label, g_shift, w_shift in [("bear", -config.growth_sigma, config.wacc_sigma), ("base", 0.0, 0.0), ("bull", config.growth_sigma, -config.wacc_sigma)]:
        fv = calculate_dcf_value(
            dcf_input.free_cash_flow,
            config.base_growth + g_shift,
            max(float(market["wacc_floor"]), base_wacc + w_shift),
            min(base_terminal, max(float(market["wacc_floor"]), base_wacc + w_shift) - 0.01),
            config.horizon_years,
            dcf_input.shares_outstanding,
            dcf_input.net_debt,
        )
        scenario_rows.append(
            {
                "ticker": dcf_input.ticker,
                "scenario": label,
                "growth": config.base_growth + g_shift,
                "wacc": max(float(market["wacc_floor"]), base_wacc + w_shift),
                "terminal_growth": min(base_terminal, max(float(market["wacc_floor"]), base_wacc + w_shift) - 0.01),
                "fair_value": fv,
                "upside": fv / dcf_input.current_price - 1.0 if dcf_input.current_price else np.nan,
            }
        )
    scenarios_df = pd.DataFrame(scenario_rows)

    factor_df = pd.DataFrame(
        [
            {
                "ticker": dcf_input.ticker,
                "dcf_fair_value_median": summary["fair_value_median"],
                "dcf_mispricing": summary["upside_median"],
                "dcf_scenario_spread": summary["fair_value_p95"] / max(summary["fair_value_p05"], 1e-9) - 1.0,
                "dcf_probability_undervalued": summary["probability_undervalued"],
                "model_factor_family": "model_based_mispricing",
                "model_name": "dcf_monte_carlo",
            }
        ]
    )

    assumptions_df = pd.DataFrame([{**asdict(config), **market, "ticker": dcf_input.ticker, "current_price": dcf_input.current_price, "free_cash_flow": dcf_input.free_cash_flow, "shares_outstanding": dcf_input.shares_outstanding, "net_debt": dcf_input.net_debt}])
    return {
        "simulations": simulations_df,
        "summary": summary_df,
        "scenarios": scenarios_df,
        "assumptions": assumptions_df,
        "factors": factor_df,
        "market_parameters": market_parameters_frame(),
    }


def run_residual_income_monte_carlo(ri_input: ResidualIncomeInput, config: DCFMonteCarloConfig | None = None) -> dict[str, pd.DataFrame]:
    """Run residual-income Monte Carlo with the same artifact shape as DCF."""
    config = config or DCFMonteCarloConfig(region=infer_region(ri_input.ticker))
    market = MARKET_PARAMETER_PRESETS.get(config.region, MARKET_PARAMETER_PRESETS["US"])
    rng = np.random.default_rng(config.seed + 17)
    simulations = max(100, int(config.simulations))
    terminal_mid = float(market["terminal_growth_mid"])

    roe = np.clip(rng.normal(ri_input.roe, max(0.015, abs(ri_input.roe) * 0.20), simulations), -0.25, 0.45)
    cost_of_equity = np.clip(rng.normal(ri_input.cost_of_equity, config.wacc_sigma, simulations), float(market["wacc_floor"]), float(market["wacc_cap"]))
    terminal_growth = np.clip(rng.normal(terminal_mid, config.terminal_growth_sigma, simulations), -0.02, 0.06)
    terminal_growth = np.minimum(terminal_growth, cost_of_equity - 0.01)
    book_growth = np.clip(roe * 0.40, -0.05, 0.12)

    fair_values = np.array([
        calculate_residual_income_value(
            ri_input.book_value_per_share,
            float(r),
            float(coe),
            float(tg),
            config.horizon_years,
            float(bg),
        )
        for r, coe, tg, bg in zip(roe, cost_of_equity, terminal_growth, book_growth)
    ])
    simulations_df = pd.DataFrame({
        "ticker": ri_input.ticker,
        "simulation": np.arange(1, simulations + 1),
        "roe": roe,
        "cost_of_equity": cost_of_equity,
        "terminal_growth": terminal_growth,
        "book_growth": book_growth,
        "fair_value": fair_values,
        "current_price": ri_input.current_price,
        "upside": fair_values / ri_input.current_price - 1.0,
        "region": config.region,
        "currency": ri_input.currency or market["currency"],
    }).replace([np.inf, -np.inf], np.nan).dropna(subset=["fair_value"])

    summary = {
        "ticker": ri_input.ticker,
        "region": config.region,
        "currency": ri_input.currency or market["currency"],
        "current_price": ri_input.current_price,
        "simulations": len(simulations_df),
        "fair_value_p05": simulations_df["fair_value"].quantile(0.05),
        "fair_value_p25": simulations_df["fair_value"].quantile(0.25),
        "fair_value_median": simulations_df["fair_value"].median(),
        "fair_value_mean": simulations_df["fair_value"].mean(),
        "fair_value_p75": simulations_df["fair_value"].quantile(0.75),
        "fair_value_p95": simulations_df["fair_value"].quantile(0.95),
        "upside_median": simulations_df["upside"].median(),
        "probability_undervalued": float((simulations_df["fair_value"] > ri_input.current_price).mean()),
        "base_roe": ri_input.roe,
        "base_cost_of_equity": ri_input.cost_of_equity,
        "base_terminal_growth": terminal_mid,
    }
    summary_df = pd.DataFrame([summary])

    scenario_rows = []
    for label, roe_shift, coe_shift in [("bear", -0.03, config.wacc_sigma), ("base", 0.0, 0.0), ("bull", 0.03, -config.wacc_sigma)]:
        coe = max(float(market["wacc_floor"]), ri_input.cost_of_equity + coe_shift)
        tg = min(terminal_mid, coe - 0.01)
        r = ri_input.roe + roe_shift
        fv = calculate_residual_income_value(ri_input.book_value_per_share, r, coe, tg, config.horizon_years)
        scenario_rows.append({
            "ticker": ri_input.ticker,
            "scenario": label,
            "roe": r,
            "cost_of_equity": coe,
            "terminal_growth": tg,
            "fair_value": fv,
            "upside": fv / ri_input.current_price - 1.0 if ri_input.current_price else np.nan,
        })
    scenarios_df = pd.DataFrame(scenario_rows)

    factor_df = pd.DataFrame([{
        "ticker": ri_input.ticker,
        "residual_income_fair_value_median": summary["fair_value_median"],
        "residual_income_mispricing": summary["upside_median"],
        "residual_income_scenario_spread": summary["fair_value_p95"] / max(summary["fair_value_p05"], 1e-9) - 1.0,
        "residual_income_probability_undervalued": summary["probability_undervalued"],
        "model_factor_family": "model_based_mispricing",
        "model_name": "residual_income_monte_carlo",
    }])
    assumptions_df = pd.DataFrame([{**asdict(config), **market, **asdict(ri_input)}])
    return {
        "simulations": simulations_df,
        "summary": summary_df,
        "scenarios": scenarios_df,
        "assumptions": assumptions_df,
        "factors": factor_df,
        "market_parameters": market_parameters_frame(),
    }


def run_eva_scenario_bands(eva_input: EVAScenarioInput, config: DCFMonteCarloConfig | None = None) -> dict[str, pd.DataFrame]:
    """Create EVA bear/base/bull scenario bands for enterprise-value uncertainty."""
    config = config or DCFMonteCarloConfig(region=infer_region(eva_input.ticker))
    market = MARKET_PARAMETER_PRESETS.get(config.region, MARKET_PARAMETER_PRESETS["US"])
    terminal_mid = float(market["terminal_growth_mid"])
    rows = []
    for scenario, growth_shift, wacc_shift in [("bear", -config.growth_sigma, config.wacc_sigma), ("base", 0.0, 0.0), ("bull", config.growth_sigma, -config.wacc_sigma)]:
        growth = config.base_growth + growth_shift
        wacc = max(float(market["wacc_floor"]), eva_input.wacc + wacc_shift)
        terminal_growth = min(terminal_mid, wacc - 0.01)
        invested_capital = eva_input.invested_capital
        pv_eva = 0.0
        eva_last = np.nan
        for year in range(1, int(config.horizon_years) + 1):
            nopat = eva_input.nopat * ((1.0 + growth) ** year)
            capital = invested_capital * ((1.0 + max(growth * 0.5, -0.02)) ** year)
            eva = nopat - wacc * capital
            eva_last = eva
            pv_eva += eva / ((1.0 + wacc) ** year)
        terminal_eva = eva_last * (1.0 + terminal_growth) / (wacc - terminal_growth) if wacc > terminal_growth else np.nan
        enterprise_value = invested_capital + pv_eva + terminal_eva / ((1.0 + wacc) ** int(config.horizon_years))
        equity_value = enterprise_value - eva_input.net_debt
        fair_value = equity_value / eva_input.shares_outstanding
        rows.append({
            "ticker": eva_input.ticker,
            "scenario": scenario,
            "growth": growth,
            "wacc": wacc,
            "terminal_growth": terminal_growth,
            "eva_last": eva_last,
            "enterprise_value": enterprise_value,
            "fair_value": fair_value,
            "current_price": eva_input.current_price,
            "upside": fair_value / eva_input.current_price - 1.0 if eva_input.current_price else np.nan,
            "region": config.region,
            "currency": eva_input.currency or market["currency"],
        })
    bands_df = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
    summary_df = pd.DataFrame([{
        "ticker": eva_input.ticker,
        "region": config.region,
        "currency": eva_input.currency or market["currency"],
        "current_price": eva_input.current_price,
        "fair_value_bear": bands_df.loc[bands_df["scenario"].eq("bear"), "fair_value"].mean(),
        "fair_value_base": bands_df.loc[bands_df["scenario"].eq("base"), "fair_value"].mean(),
        "fair_value_bull": bands_df.loc[bands_df["scenario"].eq("bull"), "fair_value"].mean(),
        "upside_base": bands_df.loc[bands_df["scenario"].eq("base"), "upside"].mean(),
    }])
    bear = pd.to_numeric(summary_df["fair_value_bear"], errors="coerce").iloc[0]
    bull = pd.to_numeric(summary_df["fair_value_bull"], errors="coerce").iloc[0]
    base = pd.to_numeric(summary_df["fair_value_base"], errors="coerce").iloc[0]
    factor_df = pd.DataFrame([{
        "ticker": eva_input.ticker,
        "eva_fair_value_base": base,
        "eva_value_gap": base / eva_input.current_price - 1.0 if eva_input.current_price else np.nan,
        "eva_scenario_spread": bull / max(bear, 1e-9) - 1.0 if np.isfinite(bear) else np.nan,
        "model_factor_family": "model_based_mispricing",
        "model_name": "eva_scenario_bands",
    }])
    assumptions_df = pd.DataFrame([{**asdict(config), **market, **asdict(eva_input)}])
    return {
        "bands": bands_df,
        "summary": summary_df,
        "assumptions": assumptions_df,
        "factors": factor_df,
        "market_parameters": market_parameters_frame(),
    }


def write_dcf_monte_carlo_outputs(outputs: dict[str, pd.DataFrame], tables_dir: Path) -> dict[str, str]:
    tables_dir = Path(tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    names = {
        "simulations": "DCFMonteCarloSimulations.csv",
        "summary": "DCFMonteCarloSummary.csv",
        "scenarios": "DCFMonteCarloScenarios.csv",
        "assumptions": "DCFMonteCarloAssumptions.csv",
        "factors": "DCFModelBasedFactors.csv",
        "market_parameters": "DCFMarketParameters.csv",
    }
    paths: dict[str, str] = {}
    for key, filename in names.items():
        df = outputs.get(key, pd.DataFrame())
        if isinstance(df, pd.DataFrame):
            path = tables_dir / filename
            df.to_csv(path, index=False)
            paths[key] = str(path)
    return paths


def write_residual_income_monte_carlo_outputs(outputs: dict[str, pd.DataFrame], tables_dir: Path) -> dict[str, str]:
    tables_dir = Path(tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    names = {
        "simulations": "ResidualIncomeMonteCarloSimulations.csv",
        "summary": "ResidualIncomeMonteCarloSummary.csv",
        "scenarios": "ResidualIncomeMonteCarloScenarios.csv",
        "assumptions": "ResidualIncomeMonteCarloAssumptions.csv",
        "factors": "ResidualIncomeModelBasedFactors.csv",
        "market_parameters": "DCFMarketParameters.csv",
    }
    paths: dict[str, str] = {}
    for key, filename in names.items():
        df = outputs.get(key, pd.DataFrame())
        if isinstance(df, pd.DataFrame):
            path = tables_dir / filename
            df.to_csv(path, index=False)
            paths[key] = str(path)
    return paths


def write_eva_scenario_outputs(outputs: dict[str, pd.DataFrame], tables_dir: Path) -> dict[str, str]:
    tables_dir = Path(tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    names = {
        "bands": "EVAScenarioBands.csv",
        "summary": "EVAScenarioSummary.csv",
        "assumptions": "EVAScenarioAssumptions.csv",
        "factors": "EVAModelBasedFactors.csv",
        "market_parameters": "DCFMarketParameters.csv",
    }
    paths: dict[str, str] = {}
    for key, filename in names.items():
        df = outputs.get(key, pd.DataFrame())
        if isinstance(df, pd.DataFrame):
            path = tables_dir / filename
            df.to_csv(path, index=False)
            paths[key] = str(path)
    return paths
