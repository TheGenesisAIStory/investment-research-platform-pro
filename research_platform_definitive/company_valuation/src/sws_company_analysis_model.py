"""Executable Simply Wall St-style company analysis model.

This module translates the public Simply Wall St Company Analysis Model
methodology into deterministic Python utilities for the company valuation
research project. It is not affiliated with Simply Wall St and does not use
proprietary data; when an input is unavailable, the related check is marked
unavailable rather than silently passed or failed.

Implemented areas:
- Value: DCF/DDM/excess-returns/relative-valuation checks.
- Future: earnings/revenue/ROE growth checks.
- Past: EPS, ROE, ROCE, ROA checks.
- Health: non-financial and financial-institution balance-sheet checks.
- Income: dividend yield, dividend growth/volatility and payout coverage.
- Management: compensation, tenure and insider-alignment diagnostics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SWSModelConfig:
    """Thresholds used by the SWS-style check engine."""

    moderate_undervaluation: float = 0.20
    deep_undervaluation: float = 0.40
    peg_fair_min: float = 0.0
    peg_fair_max: float = 1.0
    high_growth_threshold: float = 0.20
    high_roe_threshold: float = 0.20
    debt_to_equity_threshold: float = 0.40
    ocf_to_debt_threshold: float = 0.20
    interest_cover_threshold: float = 5.0
    financial_assets_to_equity_max: float = 20.0
    bad_loan_coverage_min: float = 1.0
    deposits_to_liabilities_min: float = 0.50
    loans_to_assets_max: float = 1.10
    loans_to_deposits_max: float = 1.25
    net_charge_off_ratio_max: float = 0.03
    payout_ratio_max: float = 0.90
    reit_payout_ratio_max: float = 1.00
    dividend_drop_threshold: float = -0.10
    dividend_history_years: int = 10
    discount_rate: float = 0.09
    terminal_growth: float = 0.025
    forecast_years: int = 10
    market_pe: float = 18.0
    market_earnings_growth: float = 0.05
    market_revenue_growth: float = 0.04
    low_risk_savings_rate: float = 0.02
    inflation_rate: float = 0.02
    industry_pe: float | None = None
    industry_pb: float | None = None
    industry_eps_growth: float | None = None
    industry_roa: float | None = None
    similar_company_ceo_compensation_median: float | None = None
    min_management_tenure_years: float = 3.0
    min_board_tenure_years: float = 3.0


@dataclass(frozen=True)
class CheckResult:
    axis: str
    check_id: int
    check_name: str
    passed: bool | None
    available: bool
    value: float | str | None
    threshold: float | str | None
    description: str

    @property
    def score(self) -> float:
        return float(self.passed is True)


def _get(row: Mapping[str, Any] | pd.Series, *names: str, default: Any = np.nan) -> Any:
    for name in names:
        if name in row and pd.notna(row[name]):
            return row[name]
    return default


def _finite(value: Any) -> bool:
    try:
        return bool(pd.notna(value) and np.isfinite(float(value)))
    except Exception:
        return False


def _ratio(numerator: Any, denominator: Any) -> float:
    if not _finite(numerator) or not _finite(denominator) or float(denominator) == 0:
        return np.nan
    return float(numerator) / float(denominator)


def _check(axis: str, check_id: int, name: str, condition: bool | None, value: Any, threshold: Any, description: str) -> CheckResult:
    available = condition is not None
    return CheckResult(axis, check_id, name, bool(condition) if available else None, available, value if _finite(value) or isinstance(value, str) else None, threshold, description)


def weighted_linear_growth(values: Iterable[float], weights: Iterable[float] | None = None) -> float:
    """SWS-style trend growth: fitted slope divided by mean absolute level."""
    y = pd.Series(list(values), dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()
    if len(y) < 2:
        return np.nan
    x = np.arange(len(y), dtype="float64")
    w = np.ones(len(y)) if weights is None else np.asarray(list(weights), dtype="float64")[-len(y):]
    if len(w) != len(y) or np.any(~np.isfinite(w)):
        w = np.ones(len(y))
    slope = np.polyfit(x, y.to_numpy(), deg=1, w=w)[0]
    denom = np.mean(np.abs(y.to_numpy()))
    return float(slope / denom) if denom else np.nan


def two_stage_fcf_value_per_share(row: Mapping[str, Any] | pd.Series, config: SWSModelConfig = SWSModelConfig()) -> float:
    fcf = _get(row, "free_cash_flow", "levered_free_cash_flow")
    shares = _get(row, "shares_outstanding", "weightedAverageShsOut")
    growth = _get(row, "revenue_growth", "free_cash_flow_growth", default=0.04)
    if not (_finite(fcf) and _finite(shares) and float(fcf) > 0 and float(shares) > 0 and config.discount_rate > config.terminal_growth):
        return np.nan
    growth = float(np.clip(float(growth), -0.05, 0.20)) if _finite(growth) else 0.04
    current_fcf = float(fcf)
    projected = []
    for rate in np.linspace(growth, config.terminal_growth, config.forecast_years):
        current_fcf *= 1 + rate
        projected.append(current_fcf)
    pv_stage_1 = sum(cf / ((1 + config.discount_rate) ** (idx + 1)) for idx, cf in enumerate(projected))
    terminal_value = projected[-1] * (1 + config.terminal_growth) / (config.discount_rate - config.terminal_growth)
    pv_terminal = terminal_value / ((1 + config.discount_rate) ** config.forecast_years)
    return float((pv_stage_1 + pv_terminal) / float(shares))


def dividend_discount_value_per_share(row: Mapping[str, Any] | pd.Series, config: SWSModelConfig = SWSModelConfig()) -> float:
    dividend = _get(row, "dividend_per_share", "dps", "expected_dividend_per_share")
    growth = _get(row, "dividend_growth", "expected_dividend_growth", default=config.terminal_growth)
    if not (_finite(dividend) and float(dividend) > 0 and config.discount_rate > float(growth)):
        return np.nan
    return float(float(dividend) * (1 + float(growth)) / (config.discount_rate - float(growth)))


def excess_returns_value_per_share(row: Mapping[str, Any] | pd.Series, config: SWSModelConfig = SWSModelConfig()) -> float:
    book_value_per_share = _get(row, "book_value_per_share", "bvps")
    roe = _get(row, "future_roe", "roe")
    if not (_finite(book_value_per_share) and float(book_value_per_share) > 0 and _finite(roe) and config.discount_rate > config.terminal_growth):
        return np.nan
    excess_return = (float(roe) - config.discount_rate) * float(book_value_per_share)
    terminal_excess = excess_return * (1 + config.terminal_growth) / (config.discount_rate - config.terminal_growth)
    value = float(book_value_per_share) + terminal_excess / (1 + config.discount_rate)
    return float(value) if value > 0 else np.nan


def relative_value_per_share(row: Mapping[str, Any] | pd.Series, config: SWSModelConfig = SWSModelConfig()) -> float:
    eps = _get(row, "eps", "earnings_per_share")
    bvps = _get(row, "book_value_per_share", "bvps")
    values = []
    if _finite(eps) and float(eps) > 0:
        values.append(float(eps) * float(config.industry_pe or config.market_pe))
    if _finite(bvps) and float(bvps) > 0 and config.industry_pb:
        values.append(float(bvps) * float(config.industry_pb))
    return float(np.nanmedian(values)) if values else np.nan


def estimate_sws_fair_value(row: Mapping[str, Any] | pd.Series, config: SWSModelConfig = SWSModelConfig()) -> dict[str, Any]:
    """Select and combine SWS-style valuation models based on data availability."""
    model_values = {
        "two_stage_fcf": two_stage_fcf_value_per_share(row, config),
        "dividend_discount": dividend_discount_value_per_share(row, config),
        "excess_returns": excess_returns_value_per_share(row, config),
        "relative_value": relative_value_per_share(row, config),
    }
    valid = {name: value for name, value in model_values.items() if _finite(value) and value > 0}
    fair_value = float(np.nanmedian(list(valid.values()))) if valid else np.nan
    price = _get(row, "adj_close", "close", "price")
    discount = fair_value / float(price) - 1 if _finite(fair_value) and _finite(price) and float(price) > 0 else np.nan
    return {"fair_value": fair_value, "discount_to_fair_value": discount, "valuation_models_used": ",".join(valid.keys()) or "insufficient_data", **{f"value_{k}": v for k, v in model_values.items()}}


def value_checks(row: Mapping[str, Any] | pd.Series, config: SWSModelConfig) -> list[CheckResult]:
    valuation = estimate_sws_fair_value(row, config)
    price = _get(row, "adj_close", "close", "price")
    pe = _get(row, "pe_ratio")
    pb = _get(row, "pb_ratio")
    growth = _get(row, "earnings_growth", "net_income_growth", "revenue_growth")
    peg = _ratio(pe, float(growth) * 100 if _finite(growth) else np.nan)
    fv_discount = valuation["discount_to_fair_value"]
    industry_pe = config.industry_pe or config.market_pe
    industry_pb = config.industry_pb
    return [
        _check("value", 1, "moderately_undervalued_dcf", fv_discount >= config.moderate_undervaluation if _finite(fv_discount) else None, fv_discount, config.moderate_undervaluation, "Share price is at least 20% below fair value."),
        _check("value", 2, "deeply_undervalued_dcf", fv_discount >= config.deep_undervaluation if _finite(fv_discount) else None, fv_discount, config.deep_undervaluation, "Share price is at least 40% below fair value."),
        _check("value", 3, "pe_below_market", (pe > 0 and pe < config.market_pe) if _finite(pe) else None, pe, config.market_pe, "PE ratio is positive and below market average."),
        _check("value", 4, "pe_below_industry", (pe > 0 and pe < industry_pe) if _finite(pe) else None, pe, industry_pe, "PE ratio is positive and below industry average."),
        _check("value", 5, "peg_fair", (peg > config.peg_fair_min and peg <= config.peg_fair_max) if _finite(peg) else None, peg, "0-1", "PEG ratio is within fair range."),
        _check("value", 6, "pb_below_industry", (pb > 0 and pb < industry_pb) if _finite(pb) and _finite(industry_pb) else None, pb, industry_pb, "PB ratio is positive and below industry average."),
    ]


def future_checks(row: Mapping[str, Any] | pd.Series, config: SWSModelConfig) -> list[CheckResult]:
    earnings_growth = _get(row, "future_earnings_growth", "earnings_growth", "net_income_growth")
    revenue_growth = _get(row, "future_revenue_growth", "revenue_growth")
    future_roe = _get(row, "future_roe", "roe")
    hurdle = config.low_risk_savings_rate + config.inflation_rate
    market_earnings = config.market_earnings_growth
    market_revenue = config.market_revenue_growth
    return [
        _check("future", 1, "earnings_growth_above_savings_inflation", earnings_growth > hurdle if _finite(earnings_growth) else None, earnings_growth, hurdle, "Expected earnings growth exceeds savings plus inflation hurdle."),
        _check("future", 2, "earnings_growth_above_market", earnings_growth > market_earnings if _finite(earnings_growth) else None, earnings_growth, market_earnings, "Expected earnings growth exceeds market average."),
        _check("future", 3, "revenue_growth_above_market", revenue_growth > market_revenue if _finite(revenue_growth) else None, revenue_growth, market_revenue, "Expected revenue growth exceeds market average."),
        _check("future", 4, "earnings_high_growth", earnings_growth > config.high_growth_threshold if _finite(earnings_growth) else None, earnings_growth, config.high_growth_threshold, "Expected earnings growth exceeds 20%."),
        _check("future", 5, "revenue_high_growth", revenue_growth > config.high_growth_threshold if _finite(revenue_growth) else None, revenue_growth, config.high_growth_threshold, "Expected revenue growth exceeds 20%."),
        _check("future", 6, "future_roe_above_20", future_roe > config.high_roe_threshold if _finite(future_roe) else None, future_roe, config.high_roe_threshold, "Forecast/current ROE exceeds 20%."),
    ]


def past_checks(row: Mapping[str, Any] | pd.Series, config: SWSModelConfig) -> list[CheckResult]:
    eps_growth = _get(row, "eps_growth", "earnings_growth", "net_income_growth")
    eps_growth_5y = _get(row, "eps_growth_5y", "earnings_growth_5y")
    eps_growth_1y = _get(row, "eps_growth_1y", "earnings_growth_1y", default=eps_growth)
    roe = _get(row, "roe")
    roce = _get(row, "roce")
    roce_3y = _get(row, "roce_3y_ago")
    roa = _get(row, "roa")
    industry_eps = config.industry_eps_growth if config.industry_eps_growth is not None else 0.0
    industry_roa = config.industry_roa if config.industry_roa is not None else numeric_context(row, "roa", default=0.0)
    return [
        _check("past", 1, "eps_growth_above_industry", eps_growth_1y > industry_eps if _finite(eps_growth_1y) else None, eps_growth_1y, industry_eps, "EPS growth exceeds industry average."),
        _check("past", 2, "eps_increased_5y", eps_growth_5y > 0 if _finite(eps_growth_5y) else None, eps_growth_5y, 0, "EPS increased over five years."),
        _check("past", 3, "current_eps_growth_above_5y_average", eps_growth_1y > eps_growth_5y if _finite(eps_growth_1y) and _finite(eps_growth_5y) else None, eps_growth_1y, eps_growth_5y, "Current EPS growth exceeds five-year average."),
        _check("past", 4, "roe_above_20", roe > config.high_roe_threshold if _finite(roe) else None, roe, config.high_roe_threshold, "ROE exceeds 20%."),
        _check("past", 5, "roce_improved_3y", roce > roce_3y if _finite(roce) and _finite(roce_3y) else None, roce, roce_3y, "ROCE improved from three years ago."),
        _check("past", 6, "roa_above_industry", roa > industry_roa if _finite(roa) and _finite(industry_roa) else None, roa, industry_roa, "ROA exceeds industry average."),
    ]


def numeric_context(row: Mapping[str, Any] | pd.Series, name: str, default: float = np.nan) -> float:
    return float(row[name]) if name in row and _finite(row[name]) else default


def health_checks(row: Mapping[str, Any] | pd.Series, config: SWSModelConfig, is_financial: bool = False) -> list[CheckResult]:
    if is_financial:
        assets = _get(row, "total_assets")
        equity = _get(row, "total_equity", "totalStockholdersEquity")
        liabilities = _get(row, "total_liabilities")
        deposits = _get(row, "total_deposits", "deposits")
        loans = _get(row, "total_loans", "net_loans", "loans")
        bad_loans = _get(row, "non_performing_loans", "bad_loans")
        allowance = _get(row, "allowance_for_loan_losses", "allowance_for_nonperforming_loans")
        charge_offs = _get(row, "net_charge_offs")
        leverage = _ratio(assets, equity)
        bad_loan_coverage = _ratio(allowance, bad_loans)
        deposits_to_liabilities = _ratio(deposits, liabilities)
        loans_to_assets = _ratio(loans, assets)
        loans_to_deposits = _ratio(loans, deposits)
        charge_off_ratio = _ratio(charge_offs, loans)
        return [
            _check("health", 1, "financial_assets_to_equity_below_20x", leverage < config.financial_assets_to_equity_max if _finite(leverage) else None, leverage, config.financial_assets_to_equity_max, "Financial leverage is below 20x assets/equity."),
            _check("health", 2, "bad_loan_coverage_above_100", bad_loan_coverage > config.bad_loan_coverage_min if _finite(bad_loan_coverage) else None, bad_loan_coverage, config.bad_loan_coverage_min, "Bad loan provisions exceed bad loans."),
            _check("health", 3, "deposits_to_liabilities_above_50", deposits_to_liabilities > config.deposits_to_liabilities_min if _finite(deposits_to_liabilities) else None, deposits_to_liabilities, config.deposits_to_liabilities_min, "Deposits exceed 50% of liabilities."),
            _check("health", 4, "loans_to_assets_below_110", loans_to_assets < config.loans_to_assets_max if _finite(loans_to_assets) else None, loans_to_assets, config.loans_to_assets_max, "Net loans are below 110% of assets."),
            _check("health", 5, "loans_to_deposits_below_125", loans_to_deposits < config.loans_to_deposits_max if _finite(loans_to_deposits) else None, loans_to_deposits, config.loans_to_deposits_max, "Loans are below 125% of deposits."),
            _check("health", 6, "charge_off_ratio_below_3", charge_off_ratio < config.net_charge_off_ratio_max if _finite(charge_off_ratio) else None, charge_off_ratio, config.net_charge_off_ratio_max, "Net charge-offs are below 3% of loans."),
        ]

    short_assets = _get(row, "short_term_assets", "current_assets", "cash_and_equivalents")
    short_liabilities = _get(row, "short_term_liabilities", "current_liabilities")
    long_liabilities = _get(row, "long_term_liabilities")
    debt_to_equity = _get(row, "debt_to_equity")
    debt = _get(row, "total_debt")
    ocf = _get(row, "operating_cash_flow")
    ebit = _get(row, "ebit")
    interest = abs(float(_get(row, "interest_expense", default=np.nan))) if _finite(_get(row, "interest_expense", default=np.nan)) else np.nan
    ocf_to_debt = _ratio(ocf, debt)
    interest_cover = _ratio(ebit, interest)
    return [
        _check("health", 1, "short_assets_gt_short_liabilities", short_assets > short_liabilities if _finite(short_assets) and _finite(short_liabilities) else None, short_assets, short_liabilities, "Short-term assets exceed short-term liabilities."),
        _check("health", 2, "short_assets_gt_long_liabilities", short_assets > long_liabilities if _finite(short_assets) and _finite(long_liabilities) else None, short_assets, long_liabilities, "Short-term assets exceed long-term liabilities."),
        _check("health", 3, "debt_to_equity_not_increased", _get(row, "debt_to_equity_change_5y") <= 0 if _finite(_get(row, "debt_to_equity_change_5y")) else None, _get(row, "debt_to_equity_change_5y"), 0, "Debt/equity has not increased over five years."),
        _check("health", 4, "debt_to_equity_below_40", debt_to_equity < config.debt_to_equity_threshold if _finite(debt_to_equity) else None, debt_to_equity, config.debt_to_equity_threshold, "Debt/equity is below 40%."),
        _check("health", 5, "debt_covered_by_ocf", ocf_to_debt > config.ocf_to_debt_threshold if _finite(ocf_to_debt) else None, ocf_to_debt, config.ocf_to_debt_threshold, "Operating cash flow exceeds 20% of debt."),
        _check("health", 6, "interest_cover_above_5x", interest_cover > config.interest_cover_threshold if _finite(interest_cover) else None, interest_cover, config.interest_cover_threshold, "EBIT covers interest expense more than 5x."),
    ]


def income_checks(row: Mapping[str, Any] | pd.Series, config: SWSModelConfig, is_reit: bool = False) -> list[CheckResult]:
    dividend_yield = _get(row, "dividend_yield")
    market_yield_p25 = _get(row, "market_dividend_yield_p25", default=0.02)
    market_yield_p75 = _get(row, "market_dividend_yield_p75", default=0.05)
    dps_growth_10y = _get(row, "dividend_growth_10y", "dividend_growth")
    max_dps_drop = _get(row, "max_dividend_drop_10y")
    payout = _get(row, "payout_ratio")
    future_payout = _get(row, "future_payout_ratio", default=payout)
    payout_max = config.reit_payout_ratio_max if is_reit else config.payout_ratio_max
    return [
        _check("income", 1, "dividend_yield_above_market_p25", dividend_yield > market_yield_p25 if _finite(dividend_yield) else None, dividend_yield, market_yield_p25, "Dividend yield is above market 25th percentile."),
        _check("income", 2, "dividend_yield_above_market_p75", dividend_yield > market_yield_p75 if _finite(dividend_yield) else None, dividend_yield, market_yield_p75, "Dividend yield is above market 75th percentile."),
        _check("income", 3, "dividend_not_volatile_10y", max_dps_drop > config.dividend_drop_threshold if _finite(max_dps_drop) else None, max_dps_drop, config.dividend_drop_threshold, "No dividend-per-share drop worse than 10% in ten years."),
        _check("income", 4, "dividend_increased_10y", dps_growth_10y > 0 if _finite(dps_growth_10y) else None, dps_growth_10y, 0, "Dividend increased over ten years."),
        _check("income", 5, "dividend_covered_by_earnings", (payout > 0 and payout < payout_max) if _finite(payout) else None, payout, payout_max, "Current payout ratio is positive and covered."),
        _check("income", 6, "future_dividend_covered_by_earnings", (future_payout > 0 and future_payout < payout_max) if _finite(future_payout) else None, future_payout, payout_max, "Future payout ratio is expected to be covered."),
    ]


def management_checks(row: Mapping[str, Any] | pd.Series, config: SWSModelConfig) -> list[CheckResult]:
    ceo_comp = _get(row, "ceo_total_compensation")
    ceo_comp_growth = _get(row, "ceo_compensation_growth")
    eps_growth = _get(row, "eps_growth", "earnings_growth")
    management_tenure = _get(row, "management_avg_tenure_years")
    board_tenure = _get(row, "board_avg_tenure_years")
    insider_buys = _get(row, "insider_shares_bought", default=0)
    insider_sells = _get(row, "insider_shares_sold", default=0)
    comp_benchmark = config.similar_company_ceo_compensation_median
    return [
        _check("management", 1, "ceo_comp_below_peer_median", ceo_comp <= comp_benchmark if _finite(ceo_comp) and _finite(comp_benchmark) else None, ceo_comp, comp_benchmark, "CEO compensation is not above similar-company median."),
        _check("management", 2, "ceo_comp_growth_aligned_with_eps", ceo_comp_growth <= eps_growth if _finite(ceo_comp_growth) and _finite(eps_growth) else None, ceo_comp_growth, eps_growth, "CEO compensation growth is not above EPS growth."),
        _check("management", 3, "management_tenure_stable", management_tenure >= config.min_management_tenure_years if _finite(management_tenure) else None, management_tenure, config.min_management_tenure_years, "Average management tenure indicates stability."),
        _check("management", 4, "board_tenure_stable", board_tenure >= config.min_board_tenure_years if _finite(board_tenure) else None, board_tenure, config.min_board_tenure_years, "Average board tenure indicates stability."),
        _check("management", 5, "insider_net_buying", insider_buys > insider_sells if _finite(insider_buys) and _finite(insider_sells) else None, insider_buys - insider_sells if _finite(insider_buys) and _finite(insider_sells) else np.nan, "net buys > 0", "Insiders are net buyers over the last year."),
        _check("management", 6, "insider_not_net_selling", insider_sells <= insider_buys if _finite(insider_buys) and _finite(insider_sells) else None, insider_sells - insider_buys if _finite(insider_buys) and _finite(insider_sells) else np.nan, "net sells <= 0", "Insiders are not net sellers."),
    ]


def score_axis(checks: list[CheckResult]) -> dict[str, float | int]:
    available = sum(check.available for check in checks)
    score = sum(check.score for check in checks)
    return {"score": float(score), "available_checks": int(available), "score_pct_available": float(score / available) if available else np.nan, "score_pct_total": float(score / 6.0)}


def analyze_company(row: Mapping[str, Any] | pd.Series, config: SWSModelConfig = SWSModelConfig(), is_financial: bool | None = None, is_reit: bool = False) -> tuple[dict[str, Any], list[CheckResult]]:
    sector = str(_get(row, "sector", "industry", default="")).lower()
    financial = bool(is_financial) if is_financial is not None else any(token in sector for token in ["bank", "financial", "insurance"])
    checks = []
    checks.extend(value_checks(row, config))
    checks.extend(future_checks(row, config))
    checks.extend(past_checks(row, config))
    checks.extend(health_checks(row, config, is_financial=financial))
    checks.extend(income_checks(row, config, is_reit=is_reit))
    management = management_checks(row, config)
    checks.extend(management)
    valuation = estimate_sws_fair_value(row, config)

    axes = ["value", "future", "past", "health", "income"]
    summary: dict[str, Any] = {"ticker": _get(row, "ticker", default=None), **valuation}
    for axis in axes + ["management"]:
        axis_summary = score_axis([check for check in checks if check.axis == axis])
        summary[f"sws_{axis}_score"] = axis_summary["score"]
        summary[f"sws_{axis}_available_checks"] = axis_summary["available_checks"]
        summary[f"sws_{axis}_score_pct_available"] = axis_summary["score_pct_available"]
        summary[f"sws_{axis}_score_pct_total"] = axis_summary["score_pct_total"]
    summary["sws_snowflake_score"] = float(np.nansum([summary[f"sws_{axis}_score"] for axis in axes]))
    summary["sws_snowflake_score_pct"] = summary["sws_snowflake_score"] / 30.0
    summary["sws_total_score_including_management"] = summary["sws_snowflake_score"] + summary["sws_management_score"]
    summary["sws_total_score_including_management_pct"] = summary["sws_total_score_including_management"] / 36.0
    return summary, checks


def score_companies(df: pd.DataFrame, config: SWSModelConfig = SWSModelConfig(), financial_tickers: set[str] | None = None, reit_tickers: set[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score a dataframe and return company-level scores plus check-level audit rows."""
    score_rows: list[dict[str, Any]] = []
    check_rows: list[dict[str, Any]] = []
    financial_tickers = financial_tickers or set()
    reit_tickers = reit_tickers or set()
    for _, row in df.iterrows():
        ticker = _get(row, "ticker", default=None)
        summary, checks = analyze_company(row, config=config, is_financial=ticker in financial_tickers if ticker else None, is_reit=ticker in reit_tickers if ticker else False)
        score_rows.append(summary)
        for check in checks:
            check_dict = asdict(check)
            check_dict["ticker"] = ticker
            check_dict["score"] = check.score
            check_rows.append(check_dict)
    scores = pd.DataFrame(score_rows)
    checks = pd.DataFrame(check_rows)
    if not scores.empty:
        scores["sws_rank"] = scores["sws_total_score_including_management_pct"].rank(ascending=False, method="first")
        scores = scores.sort_values("sws_rank").reset_index(drop=True)
    return scores, checks
