"""EPS and earnings-revision factor helpers.

The module is intentionally independent from the stable core pipeline.  It
accepts point-in-time earnings/estimate panels and returns lagged experimental
features.  When consensus EPS is unavailable, a rolling historical EPS mean is
used as a naive forecast proxy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


EPS_FACTOR_COLUMNS: tuple[str, ...] = (
    "eps_surprise",
    "eps_revision",
    "eps_revision_3m",
    "eps_forecast_accuracy",
    "eps_growth_momentum",
    "earnings_yield",
)


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce") if column in frame.columns else pd.Series(np.nan, index=frame.index)


def _first(frame: pd.DataFrame, candidates: tuple[str, ...]) -> pd.Series:
    for column in candidates:
        if column in frame.columns:
            return _numeric(frame, column)
    return pd.Series(np.nan, index=frame.index)


def _date_column(frame: pd.DataFrame) -> str:
    for column in ("as_of_date", "filed_date", "report_date", "period_of_report", "date"):
        if column in frame.columns:
            return column
    return ""


def _sort_panel(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    out = frame.copy()
    date_col = _date_column(out)
    if date_col:
        out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    sort_cols = [col for col in ("ticker", date_col) if col and col in out.columns]
    return out.sort_values(sort_cols) if sort_cols else out


def _group_shift(series: pd.Series, frame: pd.DataFrame, lag_periods: int) -> pd.Series:
    if "ticker" in frame.columns:
        return series.groupby(frame["ticker"]).shift(lag_periods)
    return series.shift(lag_periods)


def _consensus_or_proxy(frame: pd.DataFrame, proxy_window: int = 4) -> pd.Series:
    actual = _first(frame, ("actual_eps", "eps_actual", "eps", "eps_ttm"))
    consensus = _first(frame, ("consensus_eps", "eps_consensus", "mean_estimate", "estimated_eps"))
    if consensus.notna().any():
        return consensus
    if "ticker" in frame.columns:
        return actual.groupby(frame["ticker"]).transform(lambda s: s.shift(1).rolling(proxy_window, min_periods=2).mean())
    return actual.shift(1).rolling(proxy_window, min_periods=2).mean()


def eps_surprise(frame: pd.DataFrame, *, lag_periods: int = 1) -> pd.Series:
    """Lagged EPS surprise: `(actual_EPS - consensus_EPS) / abs(consensus_EPS)`.

    References: Ball and Brown (1968); Bernard and Thomas (1989).  Consensus
    falls back to a naive rolling historical EPS forecast when unavailable.
    """
    sorted_frame = _sort_panel(frame)
    actual = _first(sorted_frame, ("actual_eps", "eps_actual", "eps", "eps_ttm"))
    consensus = _consensus_or_proxy(sorted_frame)
    surprise = (actual - consensus) / consensus.abs().replace(0, np.nan)
    return _group_shift(surprise, sorted_frame, lag_periods).reindex(sorted_frame.index).reindex(frame.index)


def eps_revision(frame: pd.DataFrame, *, periods: int = 1, lag_periods: int = 1) -> pd.Series:
    """Lagged analyst EPS revision over `periods` observations."""
    sorted_frame = _sort_panel(frame)
    consensus = _consensus_or_proxy(sorted_frame)
    if "ticker" in sorted_frame.columns:
        revision = consensus.groupby(sorted_frame["ticker"]).pct_change(periods)
    else:
        revision = consensus.pct_change(periods)
    return _group_shift(revision, sorted_frame, lag_periods).reindex(sorted_frame.index).reindex(frame.index)


def eps_forecast_accuracy(frame: pd.DataFrame, *, window: int = 4, lag_periods: int = 1) -> pd.Series:
    """Lagged rolling MAE of consensus EPS forecast errors."""
    sorted_frame = _sort_panel(frame)
    actual = _first(sorted_frame, ("actual_eps", "eps_actual", "eps", "eps_ttm"))
    consensus = _consensus_or_proxy(sorted_frame)
    error = (actual - consensus).abs()
    if "ticker" in sorted_frame.columns:
        accuracy = error.groupby(sorted_frame["ticker"]).transform(lambda s: s.rolling(window, min_periods=2).mean())
    else:
        accuracy = error.rolling(window, min_periods=2).mean()
    return _group_shift(accuracy, sorted_frame, lag_periods).reindex(sorted_frame.index).reindex(frame.index)


def eps_growth_momentum(frame: pd.DataFrame, *, periods: int = 4, lag_periods: int = 1) -> pd.Series:
    """Lagged YoY quarterly EPS growth: `EPS_t / EPS_t-4 - 1`."""
    sorted_frame = _sort_panel(frame)
    eps = _first(sorted_frame, ("actual_eps", "eps_actual", "eps", "eps_ttm"))
    if "ticker" in sorted_frame.columns:
        growth = eps.groupby(sorted_frame["ticker"]).pct_change(periods)
    else:
        growth = eps.pct_change(periods)
    return _group_shift(growth, sorted_frame, lag_periods).reindex(sorted_frame.index).reindex(frame.index)


def earnings_yield(frame: pd.DataFrame, *, lag_periods: int = 1) -> pd.Series:
    """Lagged earnings yield: `EPS / price`."""
    sorted_frame = _sort_panel(frame)
    eps = _first(sorted_frame, ("actual_eps", "eps_actual", "eps", "eps_ttm"))
    price = _first(sorted_frame, ("price", "close", "adj_close", "last_price"))
    ey = eps / price.replace(0, np.nan)
    return _group_shift(ey, sorted_frame, lag_periods).reindex(sorted_frame.index).reindex(frame.index)


def build_eps_factors(frame: pd.DataFrame, *, lag_periods: int = 1) -> pd.DataFrame:
    """Append experimental EPS factor columns without mutating input.

    `lag_periods=1` is the minimum fiscal-period lag.  For quarterly earnings
    panels this enforces a one-quarter point-in-time delay before features can
    enter a prediction row.
    """
    if frame is None or frame.empty:
        return pd.DataFrame(columns=[*EPS_FACTOR_COLUMNS])
    out = frame.copy()
    out["eps_surprise"] = eps_surprise(out, lag_periods=lag_periods)
    out["eps_revision"] = eps_revision(out, periods=1, lag_periods=lag_periods)
    out["eps_revision_3m"] = eps_revision(out, periods=3, lag_periods=lag_periods)
    out["eps_forecast_accuracy"] = eps_forecast_accuracy(out, lag_periods=lag_periods)
    out["eps_growth_momentum"] = eps_growth_momentum(out, lag_periods=lag_periods)
    out["earnings_yield"] = earnings_yield(out, lag_periods=lag_periods)
    return out


__all__ = [
    "EPS_FACTOR_COLUMNS",
    "build_eps_factors",
    "earnings_yield",
    "eps_forecast_accuracy",
    "eps_growth_momentum",
    "eps_revision",
    "eps_surprise",
]
