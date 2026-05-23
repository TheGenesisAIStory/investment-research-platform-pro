"""Modern dataset creation helpers for the ML4T repository."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def repo_root_from(path: Path | None = None) -> Path:
    path = (path or Path.cwd()).resolve()
    for candidate in [path, *path.parents]:
        if (candidate / ".git").exists() or (candidate / "README.md").exists():
            return candidate
    return path


def make_synthetic_ohlcv(symbols: list[str] | None = None, periods: int = 504, seed: int = 7) -> pd.DataFrame:
    symbols = symbols or ["ASML.AS", "SAP.DE", "MC.PA", "SIE.DE", "ENEL.MI", "SAN.MC"]
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=periods)
    frames = []
    for i, symbol in enumerate(symbols):
        returns = rng.normal(0.0002 + i * 0.00001, 0.012 + i * 0.001, len(dates))
        close = 100 * np.exp(np.cumsum(returns))
        open_ = close * (1 + rng.normal(0, 0.002, len(dates)))
        high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.01, len(dates)))
        low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.01, len(dates)))
        volume = rng.integers(300_000, 4_000_000, len(dates))
        frames.append(pd.DataFrame({"date": dates, "ticker": symbol, "open": open_, "high": high, "low": low, "close": close, "volume": volume, "source": "synthetic"}))
    return pd.concat(frames, ignore_index=True)


def make_macro_sample(periods: int = 240) -> pd.DataFrame:
    dates = pd.date_range("2005-01-31", periods=periods, freq="ME")
    cycle = np.sin(np.linspace(0, 10, periods))
    return pd.DataFrame(
        {
            "date": dates,
            "policy_rate": 2.0 + cycle + np.linspace(0, 1.2, periods) / 10,
            "inflation_proxy": 2.2 + 0.7 * np.sin(np.linspace(1, 12, periods)),
            "growth_proxy": 1.5 + 0.5 * np.cos(np.linspace(0, 8, periods)),
            "source": "synthetic_macro",
        }
    )


def make_alternative_sample() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"source": "Polymarket", "feature": "prediction_market_probability", "ml_use": "event-risk features", "quality": "medium", "free": True},
            {"source": "Stocktwits", "feature": "social_message_volume", "ml_use": "attention/sentiment features", "quality": "medium", "free": True},
            {"source": "GDELT", "feature": "global_news_attention", "ml_use": "news/event signals", "quality": "medium-high", "free": True},
            {"source": "Wikipedia Pageviews", "feature": "entity_attention", "ml_use": "attention factor", "quality": "medium", "free": True},
            {"source": "OpenInsider", "feature": "insider_transactions", "ml_use": "insider-sentiment factors", "quality": "medium-high", "free": True},
        ]
    )


def save_dataset(df: pd.DataFrame, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    elif path.suffix == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {path.suffix}")
    return {"path": str(path), "rows": len(df), "columns": len(df.columns)}


def build_modern_datasets(repo_root: Path) -> pd.DataFrame:
    output_root = repo_root / "data" / "modern_datasets"
    outputs = []
    outputs.append(save_dataset(make_synthetic_ohlcv(), output_root / "market" / "market_ohlcv.parquet"))
    outputs.append(save_dataset(make_macro_sample(), output_root / "macro" / "fred_macro_sample.parquet"))
    outputs.append(save_dataset(make_alternative_sample(), output_root / "alternative" / "alternative_sample.parquet"))
    manifest = pd.DataFrame(outputs)
    manifest_path = output_root / "metadata" / "modern_dataset_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    return manifest
