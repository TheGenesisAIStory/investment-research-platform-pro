"""Performance and risk metrics for ML stock experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd


def annualized_volatility(returns: pd.Series, periods_per_year: int = 12) -> float:
    """Return annualized volatility from period returns."""
    r = pd.to_numeric(returns, errors="coerce").dropna()
    if r.empty:
        return float("nan")
    return float(r.std(ddof=0) * np.sqrt(periods_per_year))


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 12) -> float:
    """Return annualized Sharpe ratio from period returns."""
    r = pd.to_numeric(returns, errors="coerce").dropna()
    if r.empty:
        return float("nan")
    excess = r - (risk_free_rate / periods_per_year)
    vol = excess.std(ddof=0)
    if vol == 0 or np.isnan(vol):
        return float("nan")
    return float((excess.mean() / vol) * np.sqrt(periods_per_year))


def max_drawdown(returns: pd.Series) -> float:
    """Return maximum drawdown from a return series."""
    r = pd.to_numeric(returns, errors="coerce").dropna()
    if r.empty:
        return float("nan")
    wealth = (1 + r).cumprod()
    drawdown = wealth / wealth.cummax() - 1
    return float(drawdown.min())


def turnover(weights: pd.DataFrame, date_col: str = "date", ticker_col: str = "ticker", weight_col: str = "weight") -> float:
    """Return average one-way turnover from a date/ticker weight table."""
    if weights is None or weights.empty or not {date_col, ticker_col, weight_col}.issubset(weights.columns):
        return float("nan")
    pivot = weights.pivot_table(index=date_col, columns=ticker_col, values=weight_col, aggfunc="sum").fillna(0.0)
    return float(pivot.diff().abs().sum(axis=1).mean())


def quintile_risk_report(
    quintile_returns: pd.DataFrame,
    periods_per_year: int = 12,
    avg_n_names: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build mean/vol/Sharpe diagnostics for Q1..Q5 and LS portfolios.

    Accepts either a wide table of return columns or the long-form output of
    `make_quantile_portfolios` with `date`, `quantile`, `return`.
    """
    if quintile_returns is None or quintile_returns.empty:
        return pd.DataFrame(columns=["mean_return", "annualized_vol", "sharpe", "avg_n_names"])

    if {"date", "quantile", "return"}.issubset(quintile_returns.columns):
        returns = quintile_returns.pivot_table(index="date", columns="quantile", values="return", aggfunc="mean").sort_index()
        names = quintile_returns.pivot_table(index="date", columns="quantile", values="name_count", aggfunc="mean") if "name_count" in quintile_returns.columns else None
    else:
        returns = quintile_returns.copy()
        names = avg_n_names

    rows = []
    for col in returns.columns:
        label = f"Q{int(col)}" if str(col).replace(".0", "").isdigit() else str(col)
        series = pd.to_numeric(returns[col], errors="coerce")
        row = {
            "portfolio": label,
            "mean_return": series.mean(),
            "annualized_vol": annualized_volatility(series, periods_per_year=periods_per_year),
            "sharpe": sharpe_ratio(series, periods_per_year=periods_per_year),
            "max_drawdown": max_drawdown(series),
        }
        if names is not None and col in names.columns:
            row["avg_n_names"] = pd.to_numeric(names[col], errors="coerce").mean()
        else:
            row["avg_n_names"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("portfolio")


class PerformanceMetrics:
    """Convenience summary for a portfolio return series or table."""

    def __init__(self, returns: pd.Series | pd.DataFrame, periods_per_year: int = 12) -> None:
        self.returns = returns
        self.periods_per_year = periods_per_year

    def _as_series(self) -> pd.Series:
        if isinstance(self.returns, pd.Series):
            return pd.to_numeric(self.returns, errors="coerce").dropna()
        if isinstance(self.returns, pd.DataFrame):
            if "return" in self.returns.columns:
                return pd.to_numeric(self.returns["return"], errors="coerce").dropna()
            if self.returns.shape[1] == 1:
                return pd.to_numeric(self.returns.iloc[:, 0], errors="coerce").dropna()
        return pd.Series(dtype=float)

    def summary(self) -> pd.DataFrame:
        """Return a one-row performance summary."""
        r = self._as_series()
        if r.empty:
            return pd.DataFrame([{
                "periods": 0,
                "mean_return": np.nan,
                "annualized_return": np.nan,
                "annualized_vol": np.nan,
                "sharpe": np.nan,
                "max_drawdown": np.nan,
            }])
        return pd.DataFrame([{
            "periods": int(r.shape[0]),
            "mean_return": float(r.mean()),
            "annualized_return": float((1 + r.mean()) ** self.periods_per_year - 1),
            "annualized_vol": annualized_volatility(r, self.periods_per_year),
            "sharpe": sharpe_ratio(r, periods_per_year=self.periods_per_year),
            "max_drawdown": max_drawdown(r),
        }])


class FactorAlphaEvaluator:
    """Run a simple factor alpha regression for a portfolio return series."""

    def __init__(
        self,
        portfolio_returns: pd.Series,
        factor_data: pd.DataFrame | None = None,
        periods_per_year: int = 12,
    ) -> None:
        self.portfolio_returns = pd.to_numeric(portfolio_returns, errors="coerce")
        self.factor_data = factor_data
        self.periods_per_year = periods_per_year

    def run(self) -> pd.DataFrame:
        """Return alpha, beta coefficients and R2 where factor data is available."""
        y = self.portfolio_returns.rename("portfolio_return").dropna()
        if self.factor_data is None or self.factor_data.empty:
            alpha = y.mean() * self.periods_per_year if not y.empty else np.nan
            return pd.DataFrame([{"alpha_ann": alpha, "r2": np.nan, "n_obs": int(y.shape[0])}])

        factors = self.factor_data.copy()
        data = pd.concat([y, factors], axis=1, join="inner").dropna()
        if data.empty:
            return pd.DataFrame([{"alpha_ann": np.nan, "r2": np.nan, "n_obs": 0}])
        yv = data.iloc[:, 0].to_numpy(dtype=float)
        X = data.iloc[:, 1:].to_numpy(dtype=float)
        X = np.c_[np.ones(len(X)), X]
        beta = np.linalg.pinv(X.T @ X) @ X.T @ yv
        fitted = X @ beta
        resid = yv - fitted
        sst = ((yv - yv.mean()) ** 2).sum()
        r2 = 1 - (resid @ resid) / sst if sst else np.nan
        row = {"alpha_ann": float(beta[0] * self.periods_per_year), "r2": float(r2), "n_obs": int(len(data))}
        for name, value in zip(data.columns[1:], beta[1:]):
            row[f"beta_{name}"] = float(value)
        return pd.DataFrame([row])


def plot_cumulative_returns(returns: pd.Series | pd.DataFrame, title: str = "Cumulative Returns"):
    """Plot cumulative returns and return the matplotlib axis."""
    import matplotlib.pyplot as plt

    if isinstance(returns, pd.Series):
        cumulative = (1 + pd.to_numeric(returns, errors="coerce").fillna(0)).cumprod() - 1
    else:
        numeric = returns.apply(pd.to_numeric, errors="coerce").fillna(0)
        cumulative = (1 + numeric).cumprod() - 1
    ax = cumulative.plot(figsize=(10, 5), title=title)
    ax.set_ylabel("Cumulative return")
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.4)
    plt.tight_layout()
    return ax
