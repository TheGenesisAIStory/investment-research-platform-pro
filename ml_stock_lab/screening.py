"""Screening and ranking classes for stock-selection experiments."""

from __future__ import annotations

import pandas as pd

from ml_stock_lab.signals import assign_quantiles, rank_scores


class ScreeningFunction:
    """Apply hard banking screens such as CET1, NPL and liquidity thresholds."""

    def __init__(
        self,
        cet1_col: str = "fund_cet1_lag",
        npl_col: str = "fund_npl_ratio_lag",
        liq_col: str = "fund_ldr_lag",
        cet1_min: float = 11.0,
        npl_max: float = 0.08,
        liq_min: float = 0.0,
        out_col: str = "screen_pass",
    ) -> None:
        self.cet1_col = cet1_col
        self.npl_col = npl_col
        self.liq_col = liq_col
        self.cet1_min = cet1_min
        self.npl_max = npl_max
        self.liq_min = liq_min
        self.out_col = out_col

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Append a Boolean screen-pass column."""
        out = df.copy()
        passed = pd.Series(True, index=out.index)
        if self.cet1_col in out.columns:
            passed &= pd.to_numeric(out[self.cet1_col], errors="coerce") >= self.cet1_min
        if self.npl_col in out.columns:
            passed &= pd.to_numeric(out[self.npl_col], errors="coerce") <= self.npl_max
        if self.liq_col in out.columns:
            passed &= pd.to_numeric(out[self.liq_col], errors="coerce") >= self.liq_min
        out[self.out_col] = passed.fillna(False)
        return out


class TopNSelector:
    """Select top-N names by score, optionally within each date."""

    def __init__(self, score_col: str, n: int = 10, date_col: str = "date", ascending: bool = False) -> None:
        self.score_col = score_col
        self.n = n
        self.date_col = date_col
        self.ascending = ascending

    def select(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return the selected top-N rows."""
        if self.date_col in df.columns:
            ranked = rank_scores(df, self.score_col, ascending=self.ascending, group_col=self.date_col)
            return ranked[ranked["rank"] <= self.n].copy()
        return rank_scores(df, self.score_col, ascending=self.ascending).head(self.n).copy()


class QuantileSorter:
    """Assign 1..N quantile buckets from a signal column."""

    def __init__(self, signal_col: str, n_quantiles: int = 5, date_col: str = "date") -> None:
        self.signal_col = signal_col
        self.n_quantiles = n_quantiles
        self.date_col = date_col

    def assign_quantiles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Append a `quantile` column."""
        return assign_quantiles(df, self.signal_col, q=self.n_quantiles, date_col=self.date_col)
