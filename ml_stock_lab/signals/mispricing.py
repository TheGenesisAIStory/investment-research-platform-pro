"""Mispricing signal utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_absolute_mispricing(fair_value: pd.Series, market_value: pd.Series) -> pd.Series:
    """Return absolute mispricing: `fair_value - market_value`."""
    return (pd.to_numeric(fair_value, errors="coerce") - pd.to_numeric(market_value, errors="coerce")).rename("mispricing_abs")


def compute_relative_mispricing(fair_value: pd.Series, market_value: pd.Series) -> pd.Series:
    """Return relative mispricing: `(fair_value - market_value) / market_value`."""
    mv = pd.to_numeric(market_value, errors="coerce").replace(0, np.nan)
    return ((pd.to_numeric(fair_value, errors="coerce") - mv) / mv).rename("mispricing_rel")


def cross_sectional_zscore(values: pd.Series, groups: pd.Series | None = None) -> pd.Series:
    """Return z-scores globally or within each date/group cross-section."""
    x = pd.to_numeric(values, errors="coerce")
    if groups is None:
        std = x.std(ddof=0)
        return ((x - x.mean()) / std if std else x * np.nan).rename("zscore")

    def zscore(group: pd.Series) -> pd.Series:
        std = group.std(ddof=0)
        return (group - group.mean()) / std if std else group * np.nan

    return x.groupby(groups).transform(zscore).rename("zscore")


class MispricingSignal:
    """Compute and standardize fair-value mispricing signals."""

    def __init__(
        self,
        fair_value_col: str = "fair_value",
        market_value_col: str = "mkt_market_cap",
        method: str = "relative",
    ) -> None:
        self.fair_value_col = fair_value_col
        self.market_value_col = market_value_col
        self.method = method

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Append an absolute or relative mispricing column to `df`."""
        out = df.copy()
        if self.method == "absolute":
            out["mispricing_abs"] = compute_absolute_mispricing(out[self.fair_value_col], out[self.market_value_col])
        else:
            out["mispricing_rel"] = compute_relative_mispricing(out[self.fair_value_col], out[self.market_value_col])
        return out

    def standardize(
        self,
        df: pd.DataFrame,
        signal_col: str = "mispricing_rel",
        out_col: str = "mispricing_z",
        groupby: str | None = "date",
    ) -> pd.DataFrame:
        """Append a z-scored mispricing column, optionally by date."""
        out = df.copy()
        groups = out[groupby] if groupby and groupby in out.columns else None
        out[out_col] = cross_sectional_zscore(out[signal_col], groups=groups)
        return out


class EnsembleMispricingSignal:
    """Combine multiple z-scored mispricing signals into one score."""

    def __init__(
        self,
        input_signals: list[tuple[str, pd.Series]] | list[str],
        combine_method: str = "zmean",
        out_col: str = "mispricing_ens",
    ) -> None:
        self.input_signals = input_signals
        self.combine_method = combine_method
        self.out_col = out_col

    def _signal_frame(self, base_df: pd.DataFrame | None = None) -> pd.DataFrame:
        if not self.input_signals:
            return pd.DataFrame()
        if isinstance(self.input_signals[0], str):
            if base_df is None:
                raise ValueError("base_df is required when input_signals are column names.")
            return base_df[list(self.input_signals)].copy()
        return pd.concat({name: pd.to_numeric(series, errors="coerce") for name, series in self.input_signals}, axis=1)

    def combine(self, base_df: pd.DataFrame | None = None) -> pd.DataFrame:
        """Return `base_df` with the ensemble column or a new signal frame."""
        signals = self._signal_frame(base_df)
        if signals.empty:
            out = pd.DataFrame() if base_df is None else base_df.copy()
            out[self.out_col] = np.nan
            return out

        if self.combine_method == "median":
            combined = signals.median(axis=1, skipna=True)
        else:
            combined = signals.mean(axis=1, skipna=True)

        out = signals.copy() if base_df is None else base_df.copy()
        for col in signals.columns:
            if col not in out.columns:
                out[col] = signals[col]
        out[self.out_col] = combined
        return out
