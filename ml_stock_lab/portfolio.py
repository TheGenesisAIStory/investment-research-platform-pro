"""Portfolio builders for quantile and long-short experiments."""

from __future__ import annotations

import pandas as pd


class QuantilePortfolioBuilder:
    """Build equal- or value-weighted quantile return series."""

    def __init__(
        self,
        return_col: str = "target_ret_1m_fwd",
        quantile_col: str = "quantile",
        weighting: str = "equal",
        value_col: str = "mkt_market_cap",
        date_col: str = "date",
    ) -> None:
        self.return_col = return_col
        self.quantile_col = quantile_col
        self.weighting = weighting
        self.value_col = value_col
        self.date_col = date_col

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a wide date-indexed table with Q1..Qn return columns."""
        required = {self.return_col, self.quantile_col}
        if self.date_col in df.columns:
            required.add(self.date_col)
        if not required.issubset(df.columns):
            return pd.DataFrame()

        out = df.dropna(subset=[self.return_col, self.quantile_col]).copy()
        if out.empty:
            return pd.DataFrame()
        group_cols = [self.quantile_col]
        if self.date_col in out.columns:
            group_cols = [self.date_col, self.quantile_col]
        out[self.return_col] = pd.to_numeric(out[self.return_col], errors="coerce")
        if self.weighting == "value" and self.value_col in out.columns:
            out["_weight"] = pd.to_numeric(out[self.value_col], errors="coerce").clip(lower=0).fillna(0.0)
            out["_weighted_return"] = out[self.return_col] * out["_weight"]
            grouped = out.groupby(group_cols, dropna=True)
            sums = grouped[["_weighted_return", "_weight"]].sum()
            means = grouped[self.return_col].mean()
            weighted = sums["_weighted_return"] / sums["_weight"].replace(0, pd.NA)
            returns = weighted.fillna(means).rename("return").reset_index()
        else:
            returns = out.groupby(group_cols, dropna=True)[self.return_col].mean().rename("return").reset_index()
        if self.date_col in returns.columns:
            wide = returns.pivot(index=self.date_col, columns=self.quantile_col, values="return").sort_index()
        else:
            wide = returns.set_index(self.quantile_col)[["return"]].T
        wide.columns = [f"Q{int(col)}" if str(col).replace(".0", "").isdigit() else str(col) for col in wide.columns]
        return wide

    def long_short(self, quantile_returns: pd.DataFrame, long_q: int = 5, short_q: int = 1) -> pd.Series:
        """Return long-high minus short-low quantile returns."""
        if quantile_returns is None or quantile_returns.empty:
            return pd.Series(dtype=float, name="long_short")
        long_col = f"Q{long_q}"
        short_col = f"Q{short_q}"
        if long_col not in quantile_returns.columns or short_col not in quantile_returns.columns:
            return pd.Series(dtype=float, name="long_short")
        return (quantile_returns[long_col] - quantile_returns[short_col]).rename("long_short")


class LongShortPortfolioBuilder:
    """Convenience builder for a long-short portfolio from an existing signal."""

    def __init__(
        self,
        signal_col: str,
        return_col: str = "target_ret_1m_fwd",
        n_quantiles: int = 5,
        weighting: str = "equal",
    ) -> None:
        self.signal_col = signal_col
        self.return_col = return_col
        self.n_quantiles = n_quantiles
        self.weighting = weighting

    def build(self, df: pd.DataFrame) -> pd.Series:
        """Assign quantiles and return the top-minus-bottom return series."""
        from ml_stock_lab.screening import QuantileSorter

        sorted_df = QuantileSorter(self.signal_col, self.n_quantiles).assign_quantiles(df)
        builder = QuantilePortfolioBuilder(self.return_col, weighting=self.weighting)
        quantile_returns = builder.build(sorted_df)
        return builder.long_short(quantile_returns, long_q=self.n_quantiles, short_q=1)
