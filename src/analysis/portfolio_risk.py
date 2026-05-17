"""Portfolio risk engine for Analysis Studio."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .common import StudioResult, load_positions, load_price_panel, px


class PortfolioRiskEngine:
    """Estimate portfolio risk, concentration, VaR, and risk contribution."""

    analysis_name = "portfolio_risk"

    def __init__(self, portfolio_file: str | None = None) -> None:
        self.portfolio_file = portfolio_file
        self.last_result: StudioResult | None = None
        self.last_charts: list[object] = []
        self.last_summary = ""

    def run(self, portfolio_file: str | None = None) -> pd.DataFrame:
        positions = load_positions(portfolio_file or self.portfolio_file)
        panel = load_price_panel(positions["ticker"].tolist(), periods=504)
        returns = panel.pivot(index="date", columns="ticker", values="adj_close").pct_change().dropna(how="all")
        weights = positions.set_index("ticker")["weight"].reindex(returns.columns).fillna(0.0)
        cov = returns.cov() * 252
        portfolio_vol = float(np.sqrt(weights.T @ cov @ weights)) if not cov.empty else np.nan
        marginal = cov @ weights if not cov.empty else pd.Series(0, index=weights.index)
        risk_contrib = weights * marginal / portfolio_vol if portfolio_vol and np.isfinite(portfolio_vol) else weights * 0
        port_ret = returns.mul(weights, axis=1).sum(axis=1)
        var_95 = float(port_ret.quantile(0.05)) if not port_ret.empty else np.nan
        cvar_95 = float(port_ret[port_ret <= var_95].mean()) if not port_ret.empty else np.nan
        result = positions.copy()
        result["annual_volatility"] = returns.std().reindex(result["ticker"]).to_numpy() * np.sqrt(252)
        result["risk_contribution"] = risk_contrib.reindex(result["ticker"]).fillna(0).to_numpy()
        result["portfolio_volatility"] = portfolio_vol
        result["var_95_1d"] = var_95
        result["cvar_95_1d"] = cvar_95
        result["N"] = len(result)
        charts = []
        if px is not None and not result.empty:
            charts.append(px.pie(result, names="ticker", values="weight", title="Portfolio Weights"))
            charts.append(px.bar(result, x="ticker", y="risk_contribution", title="Annualized Risk Contribution"))
        summary = f"Portfolio volatility is {portfolio_vol:.2%}; 1-day 95% VaR is {var_95:.2%}." if np.isfinite(portfolio_vol) else "Portfolio risk metrics unavailable."
        self.last_charts = charts
        self.last_summary = summary
        self.last_result = StudioResult(self.analysis_name, result, summary, charts, {"portfolio_file": portfolio_file or self.portfolio_file})
        return result

