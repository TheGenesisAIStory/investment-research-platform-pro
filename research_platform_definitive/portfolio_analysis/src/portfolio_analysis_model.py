"""Executable Simply Wall St-style Portfolio Analysis Model.

Independent Python implementation based on the public SWS Portfolio Analysis
Model documentation. It combines holdings, transactions, company-level scores,
returns, benchmark comparison, contributors, currency placeholders, and dashboard
summaries in a transparent and notebook-friendly format.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PortfolioConfig:
    base_currency: str = "EUR"
    benchmark_ticker: str = "SPY"
    annualization_days: int = 252
    outlier_pe_cap: float = 200.0
    outlier_pb_cap: float = 20.0
    outlier_beta_cap: float = 5.0
    min_annualization_years: float = 1.0


def normalize_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    """Normalize transaction ledger columns."""
    required = {"ticker", "date", "transaction_type", "shares", "price"}
    missing = required - set(transactions.columns)
    if missing:
        raise ValueError(f"transactions missing required columns: {sorted(missing)}")
    tx = transactions.copy()
    tx["ticker"] = tx["ticker"].astype(str).str.upper().str.strip()
    tx["date"] = pd.to_datetime(tx["date"], errors="coerce")
    tx["transaction_type"] = tx["transaction_type"].astype(str).str.lower().str.strip()
    tx["shares"] = pd.to_numeric(tx["shares"], errors="coerce")
    tx["price"] = pd.to_numeric(tx["price"], errors="coerce")
    tx["fx_rate"] = pd.to_numeric(tx.get("fx_rate", 1.0), errors="coerce").fillna(1.0)
    tx["fees"] = pd.to_numeric(tx.get("fees", 0.0), errors="coerce").fillna(0.0)
    tx = tx.dropna(subset=["ticker", "date", "transaction_type", "shares", "price"])
    tx["cash_flow_base"] = np.where(
        tx["transaction_type"].isin(["buy", "reinvest_dividend"]),
        -(tx["shares"] * tx["price"] * tx["fx_rate"] + tx["fees"]),
        np.where(tx["transaction_type"].eq("sell"), tx["shares"] * tx["price"] * tx["fx_rate"] - tx["fees"], 0.0),
    )
    return tx.sort_values(["date", "ticker"]).reset_index(drop=True)


def build_holdings_from_transactions(transactions: pd.DataFrame, latest_prices: pd.DataFrame) -> pd.DataFrame:
    """Convert transaction ledger into current holdings and cost basis."""
    tx = normalize_transactions(transactions)
    shares_signed = np.where(tx["transaction_type"].isin(["buy", "reinvest_dividend"]), tx["shares"], np.where(tx["transaction_type"].eq("sell"), -tx["shares"], 0.0))
    tx = tx.assign(shares_signed=shares_signed)
    shares = tx.groupby("ticker")["shares_signed"].sum().rename("shares")
    buys = tx[tx["transaction_type"].isin(["buy", "reinvest_dividend"])].copy()
    cost = (buys["shares"] * buys["price"] * buys["fx_rate"] + buys["fees"]).groupby(buys["ticker"]).sum().rename("total_bought_base")
    sold = tx[tx["transaction_type"].eq("sell")].copy()
    proceeds = (sold["shares"] * sold["price"] * sold["fx_rate"] - sold["fees"]).groupby(sold["ticker"]).sum().rename("total_sold_base")
    holdings = pd.concat([shares, cost, proceeds], axis=1).fillna(0).reset_index()
    prices = latest_prices.copy()
    prices["ticker"] = prices["ticker"].astype(str).str.upper().str.strip()
    holdings = holdings.merge(prices[["ticker", "price", "fx_rate"]], on="ticker", how="left")
    holdings["fx_rate"] = holdings["fx_rate"].fillna(1.0)
    holdings["current_value_base"] = holdings["shares"] * holdings["price"] * holdings["fx_rate"]
    holdings = holdings[holdings["shares"].abs() > 1e-10].copy()
    total_value = holdings["current_value_base"].sum()
    holdings["weight"] = holdings["current_value_base"] / total_value if total_value else np.nan
    return holdings.reset_index(drop=True)


def make_equal_weight_watchlist(tickers: Iterable[str], latest_prices: pd.DataFrame, portfolio_value: float = 100_000.0) -> pd.DataFrame:
    """Create SWS watchlist-style pseudo holdings with equal final value."""
    prices = latest_prices.copy()
    prices["ticker"] = prices["ticker"].astype(str).str.upper().str.strip()
    prices = prices[prices["ticker"].isin([t.upper() for t in tickers])].copy()
    if prices.empty:
        raise ValueError("no tickers matched latest_prices")
    target_value = portfolio_value / len(prices)
    prices["fx_rate"] = pd.to_numeric(prices.get("fx_rate", 1.0), errors="coerce").fillna(1.0)
    holdings = prices[["ticker", "price", "fx_rate"]].copy()
    holdings["shares"] = target_value / (holdings["price"] * holdings["fx_rate"])
    holdings["current_value_base"] = target_value
    holdings["weight"] = 1.0 / len(holdings)
    holdings["total_bought_base"] = target_value
    holdings["total_sold_base"] = 0.0
    return holdings


def weighted_average_metric(holdings: pd.DataFrame, company_scores: pd.DataFrame, metric: str, positive_only: bool = False, cap: float | None = None) -> float:
    """Weighted average of company metrics using current holding value weights."""
    if metric not in company_scores.columns:
        return np.nan
    merged = holdings[["ticker", "weight"]].merge(company_scores[["ticker", metric]], on="ticker", how="left")
    values = pd.to_numeric(merged[metric], errors="coerce")
    if positive_only:
        values = values.where(values > 0)
    if cap is not None:
        values = values.clip(upper=cap)
    valid = values.notna() & merged["weight"].notna()
    if not valid.any():
        return np.nan
    weights = merged.loc[valid, "weight"] / merged.loc[valid, "weight"].sum()
    return float((values.loc[valid] * weights).sum())


def portfolio_snowflake(holdings: pd.DataFrame, company_scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Weighted average SWS snowflake and best/worst contributors by axis."""
    axes = ["sws_value_score", "sws_future_score", "sws_past_score", "sws_health_score", "sws_income_score"]
    merged = holdings[["ticker", "weight", "current_value_base"]].merge(company_scores, on="ticker", how="left")
    summary_rows = []
    contributor_rows = []
    for axis in axes:
        if axis not in merged.columns:
            summary_rows.append({"axis": axis, "portfolio_score": np.nan, "N": 0})
            continue
        valid = merged[axis].notna() & merged["weight"].notna()
        if not valid.any():
            summary_rows.append({"axis": axis, "portfolio_score": np.nan, "N": 0})
            continue
        weights = merged.loc[valid, "weight"] / merged.loc[valid, "weight"].sum()
        contribution = merged.loc[valid, axis] * weights
        axis_frame = merged.loc[valid, ["ticker", axis, "weight", "current_value_base"]].copy()
        axis_frame["contribution"] = contribution
        summary_rows.append({"axis": axis, "portfolio_score": float(contribution.sum()), "N": int(valid.sum())})
        best = axis_frame.sort_values("contribution", ascending=False).head(1)
        worst = axis_frame.sort_values("contribution", ascending=True).head(1)
        for label, frame in [("best", best), ("worst", worst)]:
            if not frame.empty:
                row = frame.iloc[0]
                contributor_rows.append({"axis": axis, "type": label, "ticker": row["ticker"], "score": row[axis], "weight": row["weight"], "contribution": row["contribution"]})
    return pd.DataFrame(summary_rows), pd.DataFrame(contributor_rows)


def portfolio_intrinsic_value(holdings: pd.DataFrame, company_scores: pd.DataFrame) -> dict[str, float]:
    """Portfolio intrinsic value: sum individual fair values times shares."""
    fair_col = "fair_value" if "fair_value" in company_scores.columns else "fair_value_estimate"
    if fair_col not in company_scores.columns:
        return {"portfolio_intrinsic_value": np.nan, "coverage_weight": 0.0, "upside_to_intrinsic": np.nan}
    merged = holdings.merge(company_scores[["ticker", fair_col]], on="ticker", how="left")
    valid = merged[fair_col].notna() & merged["shares"].notna()
    intrinsic = float((merged.loc[valid, fair_col] * merged.loc[valid, "shares"] * merged.loc[valid, "fx_rate"].fillna(1.0)).sum()) if valid.any() else np.nan
    coverage = float(merged.loc[valid, "current_value_base"].sum() / merged["current_value_base"].sum()) if merged["current_value_base"].sum() else 0.0
    current = float(merged["current_value_base"].sum())
    upside = intrinsic / current - 1 if np.isfinite(intrinsic) and current else np.nan
    return {"portfolio_intrinsic_value": intrinsic, "coverage_weight": coverage, "upside_to_intrinsic": upside}


def dollar_weighted_return(transactions: pd.DataFrame, holdings: pd.DataFrame, as_of_date: str | pd.Timestamp | None = None, config: PortfolioConfig = PortfolioConfig()) -> dict[str, float]:
    """SWS-style dollar-weighted return approximation using average years invested."""
    tx = normalize_transactions(transactions)
    as_of = pd.Timestamp(as_of_date) if as_of_date is not None else pd.Timestamp.today().normalize()
    buys = tx[tx["transaction_type"].isin(["buy", "reinvest_dividend"])].copy()
    sells = tx[tx["transaction_type"].eq("sell")].copy()
    total_bought = float((buys["shares"] * buys["price"] * buys["fx_rate"] + buys["fees"]).sum())
    total_sold = float((sells["shares"] * sells["price"] * sells["fx_rate"] - sells["fees"]).sum())
    current_value = float(holdings["current_value_base"].sum())
    total_gain = total_sold + current_value - total_bought
    total_return = total_gain / total_bought if total_bought else np.nan
    if not buys.empty and total_bought:
        capital = buys["shares"] * buys["price"] * buys["fx_rate"] + buys["fees"]
        years = (as_of - buys["date"]).dt.days / 365.25
        average_years_invested = float((capital * years).sum() / capital.sum())
    else:
        average_years_invested = np.nan
    annualized = (1 + total_return) ** (1 / average_years_invested) - 1 if np.isfinite(total_return) and average_years_invested >= config.min_annualization_years and total_return > -1 else np.nan
    return {"total_bought_base": total_bought, "total_sold_base": total_sold, "current_value_base": current_value, "total_gain_base": total_gain, "total_return": total_return, "average_years_invested": average_years_invested, "annualized_return": annualized}


def time_weighted_returns(price_panel: pd.DataFrame, holdings: pd.DataFrame, benchmark: pd.Series | None = None) -> tuple[pd.DataFrame, dict[str, float]]:
    """Compute static-weight time-weighted portfolio returns and benchmark comparison."""
    prices = price_panel.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    price_col = "adj_close" if "adj_close" in prices.columns else "close"
    wide = prices.pivot(index="date", columns="ticker", values=price_col).sort_index()
    returns = wide.pct_change().dropna(how="all")
    weights = holdings.set_index("ticker")["weight"].reindex(returns.columns).fillna(0.0)
    portfolio_return = returns.mul(weights, axis=1).sum(axis=1).rename("portfolio_return")
    out = portfolio_return.to_frame()
    if benchmark is not None:
        bench = benchmark.reindex(out.index).ffill().pct_change().rename("benchmark_return")
        out = out.join(bench)
        out["active_return"] = out["portfolio_return"] - out["benchmark_return"]
    out["portfolio_equity"] = (1 + out["portfolio_return"].fillna(0)).cumprod()
    if "benchmark_return" in out:
        out["benchmark_equity"] = (1 + out["benchmark_return"].fillna(0)).cumprod()
    stats = _return_stats(out["portfolio_return"])
    if "benchmark_return" in out:
        stats.update({f"benchmark_{k}": v for k, v in _return_stats(out["benchmark_return"]).items()})
        stats["tracking_error"] = float(out["active_return"].std() * np.sqrt(252))
        stats["information_ratio"] = float((out["active_return"].mean() * 252) / stats["tracking_error"]) if stats["tracking_error"] else np.nan
    return out.reset_index(), stats


def _return_stats(returns: pd.Series) -> dict[str, float]:
    r = pd.Series(returns).replace([np.inf, -np.inf], np.nan).dropna()
    if r.empty:
        return {"annual_return": np.nan, "annual_volatility": np.nan, "sharpe": np.nan, "max_drawdown": np.nan, "hit_rate": np.nan}
    annual_return = (1 + r.mean()) ** 252 - 1
    annual_vol = r.std() * np.sqrt(252)
    equity = (1 + r).cumprod()
    drawdown = equity / equity.cummax() - 1
    return {"annual_return": float(annual_return), "annual_volatility": float(annual_vol), "sharpe": float(annual_return / annual_vol) if annual_vol else np.nan, "max_drawdown": float(drawdown.min()), "hit_rate": float((r > 0).mean())}


def portfolio_metric_summary(holdings: pd.DataFrame, company_scores: pd.DataFrame, config: PortfolioConfig = PortfolioConfig()) -> pd.DataFrame:
    """Weighted average portfolio metrics with SWS-style outlier handling."""
    metrics = [
        ("pe_ratio", True, config.outlier_pe_cap),
        ("peg_proxy", True, None),
        ("pb_ratio", True, config.outlier_pb_cap),
        ("beta", False, config.outlier_beta_cap),
        ("future_earnings_growth", False, None),
        ("future_revenue_growth", False, None),
        ("roe", False, None),
        ("debt_to_equity", False, None),
        ("dividend_yield", False, None),
    ]
    rows = []
    for metric, positive_only, cap in metrics:
        rows.append({"metric": metric, "portfolio_weighted_value": weighted_average_metric(holdings, company_scores, metric, positive_only=positive_only, cap=cap), "N": int(metric in company_scores.columns)})
    return pd.DataFrame(rows)


def analyze_portfolio(transactions: pd.DataFrame, latest_prices: pd.DataFrame, company_scores: pd.DataFrame, price_panel: pd.DataFrame | None = None, benchmark: pd.Series | None = None, config: PortfolioConfig = PortfolioConfig()) -> dict[str, pd.DataFrame | dict[str, float]]:
    """Run the full SWS-style portfolio analysis model."""
    holdings = build_holdings_from_transactions(transactions, latest_prices)
    snowflake, contributors = portfolio_snowflake(holdings, company_scores)
    intrinsic = portfolio_intrinsic_value(holdings, company_scores)
    dollar_weighted = dollar_weighted_return(transactions, holdings, as_of_date=latest_prices.get("date", pd.Series([pd.Timestamp.today()])).max(), config=config)
    metric_summary = portfolio_metric_summary(holdings, company_scores, config=config)
    if price_panel is not None:
        twr, twr_stats = time_weighted_returns(price_panel, holdings, benchmark=benchmark)
    else:
        twr = pd.DataFrame()
        twr_stats = {}
    overview = {**intrinsic, **dollar_weighted, **twr_stats, "n_holdings": int(len(holdings)), "portfolio_value_base": float(holdings["current_value_base"].sum())}
    return {"holdings": holdings, "snowflake": snowflake, "contributors": contributors, "metric_summary": metric_summary, "time_weighted_returns": twr, "overview": overview}


def export_portfolio_analysis(results: dict[str, pd.DataFrame | dict[str, float]], output_dir: str | Path) -> dict[str, str]:
    """Export portfolio analysis results to CSV files."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for name, obj in results.items():
        path = output / f"portfolio_{name}.csv"
        if isinstance(obj, pd.DataFrame):
            obj.to_csv(path, index=False)
        else:
            pd.DataFrame([obj]).to_csv(path, index=False)
        paths[name] = str(path)
    return paths
