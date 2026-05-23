"""Core helpers for a compact financial valuation pipeline notebook.

The functions in this module are intentionally lightweight: they provide a
working blueprint for DCF, market multiples, peer comparables, sensitivity
analysis, and notebook-ready validation tables without calling external APIs or
writing files as a side effect.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


REQUIRED_FINANCIAL_COLUMNS = {
    "ticker",
    "fiscal_year",
    "revenue",
    "ebit",
    "depreciation_amortization",
    "capex",
    "change_nwc",
    "tax_rate",
    "shares_outstanding",
    "net_debt",
    "net_income",
}

REQUIRED_MARKET_COLUMNS = {
    "ticker",
    "as_of_date",
    "market_price",
    "shares_outstanding",
    "net_debt",
}

REQUIRED_COMPARABLE_COLUMNS = {
    "company",
    "ticker",
    "sector",
    "ev_revenue",
    "ev_ebitda",
    "pe_ratio",
}


def make_sample_valuation_data(ticker: str = "ACME") -> dict[str, pd.DataFrame]:
    """Return deterministic in-memory data for a clean first notebook run.

    Values are expressed in USD millions, except per-share market prices and
    valuation multiples. This sample replaces missing local CSV placeholders
    without writing anything to disk.
    """
    ticker = ticker.upper()
    financials = pd.DataFrame(
        {
            "ticker": [ticker] * 5,
            "fiscal_year": [2021, 2022, 2023, 2024, 2025],
            "revenue": [8200, 8900, 9650, 10420, 11200],
            "ebit": [1210, 1360, 1545, 1700, 1904],
            "depreciation_amortization": [310, 325, 350, 385, 420],
            "capex": [450, 475, 515, 570, 620],
            "change_nwc": [95, 105, 115, 120, 130],
            "tax_rate": [0.24, 0.24, 0.24, 0.24, 0.24],
            "shares_outstanding": [520, 516, 511, 505, 500],
            "net_debt": [2100, 2020, 1880, 1680, 1520],
            "cash": [760, 820, 880, 930, 980],
            "net_income": [820, 930, 1080, 1195, 1340],
        }
    )
    market_data = pd.DataFrame(
        {
            "ticker": [ticker],
            "as_of_date": ["2026-05-23"],
            "market_price": [62.50],
            "shares_outstanding": [500],
            "net_debt": [1520],
            "currency": ["USD"],
        }
    )
    comparables = pd.DataFrame(
        {
            "company": [
                "Northstar Analytics",
                "Vector Systems",
                "Orion Data Group",
                "Helios Software",
                "Atlas Automation",
            ],
            "ticker": ["NSA", "VECT", "ODG", "HELI", "ATLA"],
            "sector": ["Software"] * 5,
            "ev_revenue": [3.0, 3.6, 4.1, 3.4, 4.4],
            "ev_ebitda": [11.2, 12.8, 14.5, 13.1, 15.0],
            "pe_ratio": [21.0, 24.5, 28.0, 25.2, 30.0],
        }
    )
    return {
        "financials": financials,
        "market_data": market_data,
        "comparables": comparables,
    }


def _read_csv_or_sample(
    path: str | Path,
    sample: pd.DataFrame,
    dataset_name: str,
    use_sample_data_if_missing: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read a CSV when available, otherwise return the supplied sample data."""
    path = Path(path).expanduser()
    if path.exists():
        frame = pd.read_csv(path)
        source_type = "local_csv"
    elif use_sample_data_if_missing:
        frame = sample.copy()
        source_type = "sample_in_memory"
    else:
        raise FileNotFoundError(f"{dataset_name} CSV not found: {path}")

    metadata = {
        "dataset": dataset_name,
        "path": str(path),
        "source_type": source_type,
        "rows": len(frame),
        "columns": len(frame.columns),
    }
    return frame, metadata


def load_valuation_inputs(
    financials_path: str | Path,
    market_data_path: str | Path,
    comparables_path: str | Path,
    ticker: str,
    use_sample_data_if_missing: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load valuation inputs from local CSV placeholders or sample data."""
    sample = make_sample_valuation_data(ticker)
    frames = {}
    metadata = []

    for dataset_name, path in [
        ("financials", financials_path),
        ("market_data", market_data_path),
        ("comparables", comparables_path),
    ]:
        frame, source = _read_csv_or_sample(
            path=path,
            sample=sample[dataset_name],
            dataset_name=dataset_name,
            use_sample_data_if_missing=use_sample_data_if_missing,
        )
        frames[dataset_name] = frame
        metadata.append(source)

    return (
        frames["financials"],
        frames["market_data"],
        frames["comparables"],
        pd.DataFrame(metadata),
    )


def validate_input_data(
    financials: pd.DataFrame,
    market_data: pd.DataFrame,
    comparables: pd.DataFrame,
) -> pd.DataFrame:
    """Return notebook-friendly quality checks for the three input tables."""
    checks: list[dict[str, str]] = []

    def add_check(dataset: str, check: str, ok: bool, detail: str) -> None:
        checks.append(
            {
                "dataset": dataset,
                "check": check,
                "status": "pass" if ok else "review",
                "detail": detail,
            }
        )

    required_sets = {
        "financials": (financials, REQUIRED_FINANCIAL_COLUMNS),
        "market_data": (market_data, REQUIRED_MARKET_COLUMNS),
        "comparables": (comparables, REQUIRED_COMPARABLE_COLUMNS),
    }
    for dataset, (frame, required_columns) in required_sets.items():
        missing_columns = sorted(required_columns - set(frame.columns))
        add_check(
            dataset,
            "required_columns",
            not missing_columns,
            "ok" if not missing_columns else f"missing: {missing_columns}",
        )

        missing_rate = frame.isna().mean().max() if not frame.empty else 1.0
        add_check(
            dataset,
            "missing_values",
            missing_rate <= 0.10,
            f"max column missing rate: {missing_rate:.1%}",
        )

    numeric_ranges = [
        ("financials", financials, "revenue", 0, np.inf),
        ("financials", financials, "ebit", -np.inf, np.inf),
        ("financials", financials, "tax_rate", 0, 1),
        ("financials", financials, "shares_outstanding", 0, np.inf),
        ("market_data", market_data, "market_price", 0, np.inf),
        ("comparables", comparables, "ev_revenue", 0, np.inf),
        ("comparables", comparables, "ev_ebitda", 0, np.inf),
        ("comparables", comparables, "pe_ratio", 0, np.inf),
    ]

    for dataset, frame, column, lower, upper in numeric_ranges:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        ok = values.between(lower, upper, inclusive="both").all()
        add_check(
            dataset,
            f"range_{column}",
            bool(ok),
            f"min={values.min():.3g}, max={values.max():.3g}",
        )

    return pd.DataFrame(checks)


def prepare_financials(
    financials: pd.DataFrame,
    ticker: str,
    historical_years: int,
) -> pd.DataFrame:
    """Clean and enrich historical financials for one company."""
    frame = financials.copy()
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame = frame[frame["ticker"] == ticker.upper()].copy()
    if frame.empty:
        raise ValueError(f"No financial rows found for ticker {ticker!r}.")

    numeric_columns = sorted(REQUIRED_FINANCIAL_COLUMNS - {"ticker"})
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.sort_values("fiscal_year").tail(historical_years).reset_index(drop=True)
    frame["nopat"] = frame["ebit"] * (1 - frame["tax_rate"])
    frame["free_cash_flow"] = (
        frame["nopat"]
        + frame["depreciation_amortization"]
        - frame["capex"]
        - frame["change_nwc"]
    )
    frame["revenue_growth"] = frame["revenue"].pct_change()
    frame["ebit_margin"] = frame["ebit"] / frame["revenue"]
    frame["fcf_margin"] = frame["free_cash_flow"] / frame["revenue"]
    frame["ebitda"] = frame["ebit"] + frame["depreciation_amortization"]
    return frame


def _latest_market_row(market_data: pd.DataFrame, ticker: str) -> pd.Series:
    frame = market_data.copy()
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame = frame[frame["ticker"] == ticker.upper()].copy()
    if frame.empty:
        raise ValueError(f"No market data found for ticker {ticker!r}.")
    frame["as_of_date"] = pd.to_datetime(frame["as_of_date"], errors="coerce")
    return frame.sort_values("as_of_date").iloc[-1]


def _safe_ratio(numerator: float, denominator: float, fallback: float) -> float:
    if denominator == 0 or not np.isfinite(denominator):
        return fallback
    value = numerator / denominator
    return float(value) if np.isfinite(value) else fallback


def _historical_reinvestment_ratios(financials: pd.DataFrame) -> dict[str, float]:
    latest = financials.iloc[-1]
    revenue = float(latest["revenue"])
    return {
        "depreciation_revenue_pct": _safe_ratio(
            float(latest["depreciation_amortization"]),
            revenue,
            0.035,
        ),
        "capex_revenue_pct": _safe_ratio(float(latest["capex"]), revenue, 0.050),
        "nwc_revenue_pct": _safe_ratio(float(latest["change_nwc"]), revenue, 0.010),
    }


def run_dcf_valuation(
    financials: pd.DataFrame,
    market_data: pd.DataFrame,
    params: Mapping[str, Any],
    scenario_name: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Run a compact unlevered free-cash-flow DCF valuation."""
    latest = financials.iloc[-1]
    ticker = str(latest["ticker"])
    market = _latest_market_row(market_data, ticker)
    ratios = _historical_reinvestment_ratios(financials)

    horizon = int(params.get("forecast_horizon", 5))
    revenue_growth = float(params["revenue_growth"])
    ebit_margin = float(params["ebit_margin"])
    tax_rate = float(params["tax_rate"])
    wacc = float(params["wacc"])
    terminal_growth = float(params["terminal_growth"])

    if wacc <= terminal_growth:
        raise ValueError("WACC must be greater than terminal growth.")

    revenue = float(latest["revenue"])
    forecast_rows = []
    for step in range(1, horizon + 1):
        revenue *= 1 + revenue_growth
        ebit = revenue * ebit_margin
        nopat = ebit * (1 - tax_rate)
        depreciation = revenue * float(
            params.get(
                "depreciation_revenue_pct",
                ratios["depreciation_revenue_pct"],
            )
        )
        capex = revenue * float(params.get("capex_revenue_pct", ratios["capex_revenue_pct"]))
        change_nwc = revenue * float(params.get("nwc_revenue_pct", ratios["nwc_revenue_pct"]))
        free_cash_flow = nopat + depreciation - capex - change_nwc
        discount_factor = 1 / ((1 + wacc) ** step)
        forecast_rows.append(
            {
                "scenario": scenario_name,
                "forecast_year": int(latest["fiscal_year"]) + step,
                "revenue": revenue,
                "ebit": ebit,
                "nopat": nopat,
                "free_cash_flow": free_cash_flow,
                "discount_factor": discount_factor,
                "present_value_fcf": free_cash_flow * discount_factor,
            }
        )

    forecast = pd.DataFrame(forecast_rows)
    final_fcf = float(forecast.iloc[-1]["free_cash_flow"])
    terminal_value = final_fcf * (1 + terminal_growth) / (wacc - terminal_growth)
    present_value_terminal = terminal_value / ((1 + wacc) ** horizon)
    enterprise_value = float(forecast["present_value_fcf"].sum() + present_value_terminal)
    net_debt = float(market["net_debt"])
    shares = float(market["shares_outstanding"])
    equity_value = enterprise_value - net_debt
    target_price = equity_value / shares
    market_price = float(market["market_price"])

    result = {
        "scenario": scenario_name,
        "method": "dcf",
        "enterprise_value": enterprise_value,
        "equity_value": equity_value,
        "target_price": target_price,
        "market_price": market_price,
        "upside_downside_pct": target_price / market_price - 1,
        "net_debt": net_debt,
        "shares_outstanding": shares,
        "terminal_value": terminal_value,
        "present_value_terminal": present_value_terminal,
        "wacc": wacc,
        "terminal_growth": terminal_growth,
        "revenue_growth": revenue_growth,
        "ebit_margin": ebit_margin,
    }
    return result, forecast


def run_multiples_valuation(
    financials: pd.DataFrame,
    market_data: pd.DataFrame,
    comparables: pd.DataFrame,
    params: Mapping[str, Any],
    scenario_name: str,
) -> dict[str, Any]:
    """Estimate value from median peer EV/Revenue, EV/EBITDA, and P/E."""
    latest = financials.iloc[-1]
    ticker = str(latest["ticker"])
    market = _latest_market_row(market_data, ticker)
    peer_multiples = comparables[["ev_revenue", "ev_ebitda", "pe_ratio"]].apply(
        pd.to_numeric,
        errors="coerce",
    )

    revenue = float(latest["revenue"]) * (1 + float(params["revenue_growth"]))
    ebitda = (
        float(latest["ebit"]) + float(latest["depreciation_amortization"])
    ) * (1 + float(params["revenue_growth"]))
    net_income = float(latest["net_income"]) * (1 + float(params["revenue_growth"]))
    net_debt = float(market["net_debt"])
    shares = float(market["shares_outstanding"])
    market_price = float(market["market_price"])

    ev_revenue_value = revenue * float(peer_multiples["ev_revenue"].median())
    ev_ebitda_value = ebitda * float(peer_multiples["ev_ebitda"].median())
    pe_equity_value = net_income * float(peer_multiples["pe_ratio"].median())
    peer_weight = params.get(
        "multiples_weights",
        {"ev_revenue": 0.25, "ev_ebitda": 0.50, "pe_ratio": 0.25},
    )
    enterprise_value = (
        ev_revenue_value * float(peer_weight.get("ev_revenue", 0.25))
        + ev_ebitda_value * float(peer_weight.get("ev_ebitda", 0.50))
        + (pe_equity_value + net_debt) * float(peer_weight.get("pe_ratio", 0.25))
    )
    equity_value = enterprise_value - net_debt
    target_price = equity_value / shares

    return {
        "scenario": scenario_name,
        "method": "multiples",
        "enterprise_value": enterprise_value,
        "equity_value": equity_value,
        "target_price": target_price,
        "market_price": market_price,
        "upside_downside_pct": target_price / market_price - 1,
        "net_debt": net_debt,
        "shares_outstanding": shares,
        "peer_ev_revenue": float(peer_multiples["ev_revenue"].median()),
        "peer_ev_ebitda": float(peer_multiples["ev_ebitda"].median()),
        "peer_pe": float(peer_multiples["pe_ratio"].median()),
    }


def run_comparables_valuation(
    financials: pd.DataFrame,
    market_data: pd.DataFrame,
    comparables: pd.DataFrame,
    params: Mapping[str, Any],
    scenario_name: str,
) -> dict[str, Any]:
    """Estimate value from a peer percentile range around EV/EBITDA."""
    latest = financials.iloc[-1]
    ticker = str(latest["ticker"])
    market = _latest_market_row(market_data, ticker)
    peer_ev_ebitda = pd.to_numeric(comparables["ev_ebitda"], errors="coerce").dropna()

    ebitda = (
        float(latest["ebit"]) + float(latest["depreciation_amortization"])
    ) * (1 + float(params["revenue_growth"]))
    net_debt = float(market["net_debt"])
    shares = float(market["shares_outstanding"])
    market_price = float(market["market_price"])
    low_multiple = float(peer_ev_ebitda.quantile(0.25))
    mid_multiple = float(peer_ev_ebitda.median())
    high_multiple = float(peer_ev_ebitda.quantile(0.75))

    enterprise_value = ebitda * mid_multiple
    equity_value = enterprise_value - net_debt
    target_price = equity_value / shares

    return {
        "scenario": scenario_name,
        "method": "comparables",
        "enterprise_value": enterprise_value,
        "equity_value": equity_value,
        "target_price": target_price,
        "market_price": market_price,
        "upside_downside_pct": target_price / market_price - 1,
        "net_debt": net_debt,
        "shares_outstanding": shares,
        "comparable_low_price": (ebitda * low_multiple - net_debt) / shares,
        "comparable_mid_price": target_price,
        "comparable_high_price": (ebitda * high_multiple - net_debt) / shares,
        "peer_low_ev_ebitda": low_multiple,
        "peer_mid_ev_ebitda": mid_multiple,
        "peer_high_ev_ebitda": high_multiple,
    }


def _normalize_weights(
    weights: Mapping[str, float],
    available_methods: list[str],
) -> dict[str, float]:
    selected = {
        method: float(weights.get(method, 0.0))
        for method in available_methods
    }
    total = sum(selected.values())
    if total <= 0:
        equal_weight = 1 / len(available_methods)
        return {method: equal_weight for method in available_methods}
    return {method: value / total for method, value in selected.items()}


def run_valuation_scenarios(
    financials: pd.DataFrame,
    market_data: pd.DataFrame,
    comparables: pd.DataFrame,
    scenarios: Mapping[str, Mapping[str, Any]],
    enabled_methods: Mapping[str, bool],
    valuation_weights: Mapping[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run all enabled valuation methods for each scenario."""
    scenario_results = []
    forecast_frames = []
    method_details = []

    for scenario_name, scenario_params in scenarios.items():
        method_outputs: dict[str, dict[str, Any]] = {}

        if enabled_methods.get("dcf", False):
            dcf_result, dcf_forecast = run_dcf_valuation(
                financials,
                market_data,
                scenario_params,
                scenario_name,
            )
            method_outputs["dcf"] = dcf_result
            forecast_frames.append(dcf_forecast)
            method_details.append(dcf_result)

        if enabled_methods.get("multiples", False):
            multiples_result = run_multiples_valuation(
                financials,
                market_data,
                comparables,
                scenario_params,
                scenario_name,
            )
            method_outputs["multiples"] = multiples_result
            method_details.append(multiples_result)

        if enabled_methods.get("comparables", False):
            comparables_result = run_comparables_valuation(
                financials,
                market_data,
                comparables,
                scenario_params,
                scenario_name,
            )
            method_outputs["comparables"] = comparables_result
            method_details.append(comparables_result)

        if not method_outputs:
            raise ValueError("At least one valuation method must be enabled.")

        method_names = list(method_outputs)
        weights = _normalize_weights(valuation_weights, method_names)
        target_price = sum(
            method_outputs[method]["target_price"] * weights[method]
            for method in method_names
        )
        reference = next(iter(method_outputs.values()))
        shares = float(reference["shares_outstanding"])
        net_debt = float(reference["net_debt"])
        market_price = float(reference["market_price"])
        equity_value = target_price * shares
        enterprise_value = equity_value + net_debt

        row = {
            "scenario": scenario_name,
            "enabled_methods": ", ".join(method_names),
            "enterprise_value": enterprise_value,
            "equity_value": equity_value,
            "target_price": target_price,
            "market_price": market_price,
            "upside_downside_pct": target_price / market_price - 1,
            "net_debt": net_debt,
            "shares_outstanding": shares,
            "wacc": scenario_params.get("wacc", np.nan),
            "terminal_growth": scenario_params.get("terminal_growth", np.nan),
            "revenue_growth": scenario_params.get("revenue_growth", np.nan),
            "ebit_margin": scenario_params.get("ebit_margin", np.nan),
        }
        for method, result in method_outputs.items():
            row[f"{method}_target_price"] = result["target_price"]
            row[f"{method}_equity_value"] = result["equity_value"]
        scenario_results.append(row)

    results = pd.DataFrame(scenario_results)
    forecasts = pd.concat(forecast_frames, ignore_index=True) if forecast_frames else pd.DataFrame()
    details = pd.DataFrame(method_details)
    scenario_order = {name: idx for idx, name in enumerate(scenarios)}
    results["scenario_order"] = results["scenario"].map(scenario_order)
    results = results.sort_values("scenario_order").drop(columns="scenario_order")
    return results.reset_index(drop=True), forecasts, details.reset_index(drop=True)


def build_sensitivity_table(
    financials: pd.DataFrame,
    market_data: pd.DataFrame,
    base_params: Mapping[str, Any],
    wacc_values: list[float] | np.ndarray,
    terminal_growth_values: list[float] | np.ndarray,
) -> pd.DataFrame:
    """Return a WACC x terminal-growth table of DCF target prices."""
    rows = []
    for wacc in wacc_values:
        row: dict[float, float] = {}
        for growth in terminal_growth_values:
            params = dict(base_params)
            params["wacc"] = float(wacc)
            params["terminal_growth"] = float(growth)
            try:
                result, _ = run_dcf_valuation(
                    financials,
                    market_data,
                    params,
                    scenario_name="sensitivity",
                )
                row[float(growth)] = result["target_price"]
            except ValueError:
                row[float(growth)] = np.nan
        rows.append(pd.Series(row, name=float(wacc)))

    sensitivity = pd.DataFrame(rows)
    sensitivity.index.name = "wacc"
    sensitivity.columns.name = "terminal_growth"
    return sensitivity


def build_summary_report(
    valuation_results: pd.DataFrame,
    ticker: str,
    currency: str = "USD",
) -> str:
    """Build a concise Markdown summary from scenario results."""
    if valuation_results.empty:
        return "### Summary report\n\n- No valuation results available."

    results = valuation_results.copy()
    base = results[results["scenario"].str.lower() == "base"]
    base_row = base.iloc[0] if not base.empty else results.iloc[len(results) // 2]
    min_price = results["target_price"].min()
    max_price = results["target_price"].max()
    market_price = float(base_row["market_price"])
    scenario_lines = []
    for _, row in results.iterrows():
        scenario_lines.append(
            "- "
            f"{row['scenario'].title()}: fair value {row['target_price']:.2f} "
            f"{currency}/share, upside/downside {row['upside_downside_pct']:.1%}."
        )

    return "\n".join(
        [
            "### Summary report",
            "",
            f"- Ticker analizzato: **{ticker.upper()}**.",
            f"- Prezzo di mercato di riferimento: **{market_price:.2f} {currency}/share**.",
            *scenario_lines,
            (
                "- Range fair value: "
                f"**{min_price:.2f} - {max_price:.2f} {currency}/share**."
            ),
            (
                "- Driver principali: WACC, crescita terminale, crescita ricavi "
                "e marginalita EBIT spiegano la maggior parte della dispersione "
                "tra scenari."
            ),
        ]
    )
