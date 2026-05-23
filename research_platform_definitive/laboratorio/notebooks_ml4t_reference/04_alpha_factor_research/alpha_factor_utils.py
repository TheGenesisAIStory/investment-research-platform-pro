"""Reusable alpha-factor utilities for modern notebook workflows.

The helpers favor deterministic local execution: they use cached Parquet files
when available and synthetic sample data otherwise, so notebooks remain
executable without licensed datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


TRADING_DAYS = 252


@dataclass(frozen=True)
class BacktestSummary:
    annual_return: float
    annual_volatility: float
    sharpe: float
    max_drawdown: float
    hit_rate: float


def make_synthetic_prices(
    tickers: Iterable[str] | None = None,
    periods: int = 756,
    start: str = "2021-01-01",
    seed: int = 42,
) -> pd.DataFrame:
    tickers = list(tickers or ["ASML.AS", "SAP.DE", "MC.PA", "SIE.DE", "SAN.MC", "ENEL.MI"])
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=periods)
    frames = []
    for i, ticker in enumerate(tickers):
        drift = 0.0002 + i * 0.000015
        vol = 0.012 + i * 0.001
        returns = rng.normal(drift, vol, len(dates))
        close = 100 * np.exp(np.cumsum(returns))
        open_ = close * (1 + rng.normal(0, 0.002, len(dates)))
        high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.01, len(dates)))
        low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.01, len(dates)))
        volume = rng.integers(250_000, 3_000_000, len(dates))
        frames.append(pd.DataFrame({"date": dates, "ticker": ticker, "open": open_, "high": high, "low": low, "close": close, "volume": volume}))
    return pd.concat(frames, ignore_index=True)


def load_market_panel(repo_root: Path | None = None) -> pd.DataFrame:
    repo_root = repo_root or Path.cwd()
    candidates = [
        repo_root / "data" / "modern_datasets" / "market" / "market_ohlcv.parquet",
        repo_root / "data" / "europe_stoxx_companies" / "prices.parquet",
        repo_root / "data" / "open_market_data" / "market_ohlcv.parquet",
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_parquet(path)
            if {"date", "ticker", "close"}.issubset(df.columns):
                return df
    return make_synthetic_prices()


def compute_forward_returns(prices: pd.DataFrame, horizons: Iterable[int] = (1, 5, 21)) -> pd.DataFrame:
    df = prices.sort_values(["ticker", "date"]).copy()
    for horizon in horizons:
        df[f"return_{horizon}d"] = df.groupby("ticker")["close"].shift(-horizon).div(df["close"]).sub(1)
    df["return_1d_lag"] = df.groupby("ticker")["close"].pct_change()
    return df


def compute_technical_factors(prices: pd.DataFrame) -> pd.DataFrame:
    df = compute_forward_returns(prices).copy()
    group = df.groupby("ticker", group_keys=False)
    df["momentum_21d"] = group["close"].pct_change(21)
    df["momentum_63d"] = group["close"].pct_change(63)
    df["volatility_21d"] = group["return_1d_lag"].rolling(21).std().reset_index(level=0, drop=True)
    df["dollar_volume"] = df["close"].mul(df.get("volume", 0))
    df["dollar_volume_21d"] = group["dollar_volume"].rolling(21).mean().reset_index(level=0, drop=True)
    ma20 = group["close"].rolling(20).mean().reset_index(level=0, drop=True)
    sd20 = group["close"].rolling(20).std().reset_index(level=0, drop=True)
    df["bb_percent"] = (df["close"] - (ma20 - 2 * sd20)).div(4 * sd20).replace([np.inf, -np.inf], np.nan)
    delta = group["close"].diff()
    gain = delta.clip(lower=0).groupby(df["ticker"]).rolling(14).mean().reset_index(level=0, drop=True)
    loss = (-delta.clip(upper=0)).groupby(df["ticker"]).rolling(14).mean().reset_index(level=0, drop=True)
    rs = gain.div(loss.replace(0, np.nan))
    df["rsi_14"] = 100 - (100 / (1 + rs))
    factor_cols = ["momentum_21d", "momentum_63d", "volatility_21d", "dollar_volume_21d", "bb_percent", "rsi_14"]
    for col in factor_cols:
        df[f"{col}_rank"] = df.groupby("date")[col].rank(pct=True)
    return df


def save_alpha_panel(panel: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(output_path, index=False)
    return output_path


def rank_ic(panel: pd.DataFrame, factor_cols: Iterable[str], target: str = "return_5d") -> pd.DataFrame:
    rows = []
    for date, group in panel.dropna(subset=[target]).groupby("date"):
        for factor in factor_cols:
            subset = group[[factor, target]].dropna()
            if len(subset) >= 3:
                rows.append({"date": date, "factor": factor, "ic": subset[factor].corr(subset[target], method="spearman"), "n": len(subset)})
    return pd.DataFrame(rows)


def walk_forward_splits(dates: Iterable[pd.Timestamp], train_days: int = 252, test_days: int = 63) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    unique_dates = pd.Index(sorted(pd.to_datetime(pd.Series(list(dates)).dropna().unique())))
    splits = []
    start = 0
    while start + train_days + test_days <= len(unique_dates):
        train_start = unique_dates[start]
        train_end = unique_dates[start + train_days - 1]
        test_start = unique_dates[start + train_days]
        test_end = unique_dates[start + train_days + test_days - 1]
        splits.append((train_start, train_end, test_start, test_end))
        start += test_days
    return splits


def long_short_returns(predictions: pd.DataFrame, score_col: str = "prediction", return_col: str = "return_5d", quantile: float = 0.2) -> pd.DataFrame:
    rows = []
    for date, group in predictions.dropna(subset=[score_col, return_col]).groupby("date"):
        if len(group) < 4:
            continue
        low = group[score_col].quantile(quantile)
        high = group[score_col].quantile(1 - quantile)
        long_ret = group.loc[group[score_col] >= high, return_col].mean()
        short_ret = group.loc[group[score_col] <= low, return_col].mean()
        rows.append({"date": date, "long_return": long_ret, "short_return": short_ret, "long_short_return": long_ret - short_ret})
    return pd.DataFrame(rows)


def performance_summary(returns: pd.Series) -> BacktestSummary:
    returns = returns.dropna()
    if returns.empty:
        return BacktestSummary(0.0, 0.0, 0.0, 0.0, 0.0)
    annual_return = (1 + returns.mean()) ** TRADING_DAYS - 1
    annual_volatility = returns.std() * np.sqrt(TRADING_DAYS)
    sharpe = annual_return / annual_volatility if annual_volatility else 0.0
    cumulative = (1 + returns).cumprod()
    drawdown = cumulative.div(cumulative.cummax()).sub(1)
    return BacktestSummary(float(annual_return), float(annual_volatility), float(sharpe), float(drawdown.min()), float((returns > 0).mean()))
