"""Signal construction: mispricing, z-scores and ensembles."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_relative_mispricing(fair_value: pd.Series, market_value: pd.Series) -> pd.Series:
    """MP_rel = (V_hat - V_mkt) / V_mkt."""
    fv = pd.to_numeric(fair_value, errors="coerce")
    mv = pd.to_numeric(market_value, errors="coerce")
    return ((fv - mv) / mv.replace(0, np.nan)).rename("mispricing_rel")


def compute_absolute_mispricing(fair_value: pd.Series, market_value: pd.Series) -> pd.Series:
    return (pd.to_numeric(fair_value, errors="coerce") - pd.to_numeric(market_value, errors="coerce")).rename("mispricing_abs")


def cross_sectional_zscore(signal: pd.Series, dates: pd.Series | None = None) -> pd.Series:
    """z_i,t = (signal_i,t - mean_t) / std_t."""
    values = pd.to_numeric(signal, errors="coerce")
    if dates is None:
        std = values.std(ddof=0)
        return ((values - values.mean()) / (std if std else np.nan)).rename("zscore")
    df = pd.DataFrame({"signal": values, "date": dates})
    z = df.groupby("date")["signal"].transform(lambda s: (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) else np.nan))
    return z.rename("zscore")


class EnsembleMispricingSignal:
    """Combine multiple mispricing signals by z-score averaging."""

    def __init__(self, combine: str = "zmean") -> None:
        self.combine = combine

    def fit(self, signals: pd.DataFrame, dates: pd.Series | None = None) -> "EnsembleMispricingSignal":
        return self

    def predict(self, signals: pd.DataFrame, dates: pd.Series | None = None) -> pd.Series:
        z = pd.DataFrame({col: cross_sectional_zscore(signals[col], dates) for col in signals.columns})
        return z.mean(axis=1, skipna=True).rename("ensemble_mispricing")

    def fit_predict(self, signals: pd.DataFrame, dates: pd.Series | None = None) -> pd.Series:
        return self.fit(signals, dates).predict(signals, dates)
