"""Modern strategy-evaluation helpers with deterministic local data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


TRADING_DAYS = 252


@dataclass(frozen=True)
class PerformanceStats:
    annual_return: float
    annual_volatility: float
    sharpe: float
    max_drawdown: float
    hit_rate: float


def make_price_panel(symbols: list[str] | None = None, periods: int = 756, seed: int = 21) -> pd.DataFrame:
    symbols = symbols or ["ASML.AS", "SAP.DE", "MC.PA", "SIE.DE", "ENEL.MI", "SAN.MC"]
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=periods)
    frames = []
    for i, symbol in enumerate(symbols):
        returns = rng.normal(0.00025 + i * 0.00002, 0.012 + i * 0.001, len(dates))
        close = 100 * np.exp(np.cumsum(returns))
        frames.append(pd.DataFrame({"date": dates, "ticker": symbol, "close": close, "return": pd.Series(close).pct_change().fillna(0).to_numpy()}))
    return pd.concat(frames, ignore_index=True)


def returns_matrix(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pivot(index="date", columns="ticker", values="return").fillna(0)


def moving_average_signal(prices: pd.DataFrame, fast: int = 20, slow: int = 60) -> pd.DataFrame:
    df = prices.sort_values(["ticker", "date"]).copy()
    group = df.groupby("ticker", group_keys=False)
    df["fast_ma"] = group["close"].rolling(fast).mean().reset_index(level=0, drop=True)
    df["slow_ma"] = group["close"].rolling(slow).mean().reset_index(level=0, drop=True)
    df["signal"] = np.where(df["fast_ma"] > df["slow_ma"], 1.0, 0.0)
    df["strategy_return"] = group["signal"].shift(1).fillna(0).mul(df["return"])
    return df


def equal_weight_returns(returns: pd.DataFrame) -> pd.Series:
    return returns.mean(axis=1).rename("equal_weight")


def performance_stats(returns: pd.Series) -> PerformanceStats:
    returns = returns.dropna()
    if returns.empty:
        return PerformanceStats(0.0, 0.0, 0.0, 0.0, 0.0)
    annual_return = (1 + returns.mean()) ** TRADING_DAYS - 1
    annual_volatility = returns.std() * np.sqrt(TRADING_DAYS)
    sharpe = annual_return / annual_volatility if annual_volatility else 0.0
    cumulative = (1 + returns).cumprod()
    drawdown = cumulative / cumulative.cummax() - 1
    return PerformanceStats(float(annual_return), float(annual_volatility), float(sharpe), float(drawdown.min()), float((returns > 0).mean()))


def performance_table(series: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for name, returns in series.items():
        stats = performance_stats(returns)
        rows.append({"strategy": name, **stats.__dict__})
    return pd.DataFrame(rows).sort_values("sharpe", ascending=False)


def mean_variance_weights(returns: pd.DataFrame, risk_aversion: float = 4.0) -> pd.Series:
    mu = returns.mean() * TRADING_DAYS
    cov = returns.cov() * TRADING_DAYS
    inv_cov = np.linalg.pinv(cov.to_numpy())
    raw = inv_cov @ mu.to_numpy() / risk_aversion
    weights = np.maximum(raw, 0)
    if weights.sum() == 0:
        weights = np.ones_like(weights)
    return pd.Series(weights / weights.sum(), index=returns.columns, name="weight")


def portfolio_returns(returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    aligned = weights.reindex(returns.columns).fillna(0)
    return returns.mul(aligned, axis=1).sum(axis=1).rename("portfolio")


def kelly_fraction(win_probability: float, win_loss_ratio: float) -> float:
    if win_loss_ratio <= 0:
        return 0.0
    fraction = win_probability - (1 - win_probability) / win_loss_ratio
    return float(np.clip(fraction, 0, 1))


def export_strategy_outputs(output_dir: Path, **frames: pd.DataFrame | pd.Series) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for name, obj in frames.items():
        path = output_dir / f"{name}.parquet"
        if isinstance(obj, pd.Series):
            obj.to_frame().to_parquet(path)
        else:
            obj.to_parquet(path, index=False)
        outputs[name] = str(path)
    return outputs
