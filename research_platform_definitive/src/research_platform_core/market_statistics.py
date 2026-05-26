"""Basic market statistics for ticker, screener and portfolio views."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .time_series_forecasting import load_equity_series, load_macro_series


TRADING_DAYS = 252


def _clean_symbols(symbols: Iterable[str], max_symbols: int = 30) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for symbol in symbols:
        clean = str(symbol or "").strip().upper()
        if not clean or clean in seen:
            continue
        out.append(clean)
        seen.add(clean)
        if len(out) >= int(max_symbols):
            break
    return out


def load_price_return_matrix(
    symbols: Iterable[str],
    *,
    benchmark: str | None = None,
    financial_db_root: str | Path | None = None,
    output_root: str | Path | None = None,
    window: int = 252,
    max_symbols: int = 30,
) -> pd.DataFrame:
    """Load aligned daily returns for a bounded list of symbols."""
    selected = _clean_symbols(symbols, max_symbols=max_symbols)
    if benchmark:
        bench = str(benchmark).strip().upper()
        if bench and bench not in selected:
            selected.append(bench)
    frames: list[pd.DataFrame] = []
    for symbol in selected:
        history = load_equity_series(symbol, financial_db_root, output_root, tail_rows=int(window) + 5)
        if history.empty:
            history = load_macro_series(symbol, financial_db_root, output_root, tail_rows=int(window) + 5)
        if history.empty or "date" not in history.columns or "close" not in history.columns:
            continue
        frame = history[["date", "close"]].copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
        frame = frame.dropna(subset=["date", "close"]).sort_values("date")
        frame[symbol] = frame["close"].pct_change()
        frames.append(frame[["date", symbol]].dropna())
    if not frames:
        return pd.DataFrame()
    out = frames[0]
    for frame in frames[1:]:
        out = out.merge(frame, on="date", how="outer")
    return out.sort_values("date").tail(int(window)).reset_index(drop=True)


def compute_ticker_market_statistics(
    ticker: str,
    *,
    benchmark: str = "SPY",
    financial_db_root: str | Path | None = None,
    output_root: str | Path | None = None,
    windows: Iterable[int] = (21, 63, 252),
) -> pd.DataFrame:
    """Compute annualized volatility, variance, beta and benchmark correlation."""
    clean = str(ticker or "").strip().upper()
    if not clean:
        return pd.DataFrame()
    max_window = max(int(w) for w in windows)
    returns = load_price_return_matrix([clean], benchmark=benchmark, financial_db_root=financial_db_root, output_root=output_root, window=max_window + 5, max_symbols=2)
    if returns.empty or clean not in returns.columns:
        return pd.DataFrame()
    rows: list[dict[str, float | int | str]] = []
    bench = str(benchmark or "").strip().upper()
    for window in windows:
        view = returns.tail(int(window)).copy()
        asset = pd.to_numeric(view[clean], errors="coerce")
        volatility = float(asset.std() * np.sqrt(TRADING_DAYS)) if asset.notna().sum() >= 3 else np.nan
        variance = float(volatility**2) if pd.notna(volatility) else np.nan
        beta = np.nan
        correlation = np.nan
        if bench and bench in view.columns:
            aligned = view[[clean, bench]].dropna()
            benchmark_var = aligned[bench].var() if len(aligned) else np.nan
            if len(aligned) >= 10 and pd.notna(benchmark_var) and float(benchmark_var) != 0.0:
                beta = float(aligned[clean].cov(aligned[bench]) / benchmark_var)
                correlation = float(aligned[clean].corr(aligned[bench]))
        rows.append(
            {
                "ticker": clean,
                "benchmark": bench,
                "window_days": int(window),
                "observations": int(asset.notna().sum()),
                "annualized_volatility": volatility,
                "annualized_variance": variance,
                "beta_to_benchmark": beta,
                "correlation_to_benchmark": correlation,
            }
        )
    return pd.DataFrame(rows)


def compute_correlation_matrix(
    symbols: Iterable[str],
    *,
    financial_db_root: str | Path | None = None,
    output_root: str | Path | None = None,
    window: int = 252,
    max_symbols: int = 25,
) -> pd.DataFrame:
    """Return a correlation matrix for a bounded symbol list."""
    returns = load_price_return_matrix(symbols, financial_db_root=financial_db_root, output_root=output_root, window=window, max_symbols=max_symbols)
    if returns.empty:
        return pd.DataFrame()
    numeric = returns.drop(columns=["date"], errors="ignore").apply(pd.to_numeric, errors="coerce")
    numeric = numeric.dropna(axis=1, thresh=max(10, int(window * 0.25)))
    if numeric.shape[1] < 2:
        return pd.DataFrame()
    return numeric.corr().round(4)


def summarize_correlation_matrix(corr: pd.DataFrame, benchmark: str | None = None) -> dict[str, float | int | str]:
    """Produce compact UI metrics from a correlation matrix."""
    if corr.empty:
        return {"status": "MISSING", "asset_count": 0, "avg_pairwise_corr": np.nan, "avg_corr_to_benchmark": np.nan}
    values = corr.to_numpy(dtype=float)
    mask = ~np.eye(values.shape[0], dtype=bool)
    pairwise = values[mask]
    benchmark_clean = str(benchmark or "").strip().upper()
    avg_benchmark = np.nan
    if benchmark_clean and benchmark_clean in corr.columns:
        series = corr[benchmark_clean].drop(index=benchmark_clean, errors="ignore")
        avg_benchmark = float(series.mean()) if not series.empty else np.nan
    return {
        "status": "OK",
        "asset_count": int(corr.shape[0]),
        "avg_pairwise_corr": float(np.nanmean(pairwise)) if pairwise.size else np.nan,
        "avg_corr_to_benchmark": avg_benchmark,
    }
