"""Common data and metric helpers for Analysis Studio engines."""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import DB_BASE, SECTOR_UNIVERSES, UNIVERSES, ensure_base_directories

LOGGER = logging.getLogger("analysis_studio")

try:  # Plotly is optional for CLI imports; notebooks install it explicitly.
    import plotly.express as px
    import plotly.graph_objects as go
except Exception:  # pragma: no cover - optional dependency fallback
    px = None
    go = None


@dataclass
class StudioResult:
    """Container used by CLI, dashboard, reports, and scheduler."""

    analysis_name: str
    data: pd.DataFrame
    summary: str
    charts: list[object] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)


def stable_seed(value: str) -> int:
    """Create a reproducible seed from a string."""
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16)


def zscore(series: pd.Series) -> pd.Series:
    """Robust z-score that tolerates constant or empty series."""
    values = pd.to_numeric(series, errors="coerce")
    std = values.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return pd.Series(np.zeros(len(values)), index=series.index)
    return (values - values.mean()) / std


def normalize_tickers(tickers: Iterable[str] | str | None, universe: str = "core", sector: str | None = None) -> list[str]:
    """Resolve tickers from explicit input, universe name, and optional sector."""
    if tickers is None:
        raw = SECTOR_UNIVERSES.get((sector or "").lower()) or UNIVERSES.get(universe.lower(), UNIVERSES["core"])
    elif isinstance(tickers, str):
        if "," in tickers:
            raw = [item.strip() for item in tickers.split(",")]
        else:
            raw = UNIVERSES.get(tickers.lower(), [tickers])
    else:
        raw = list(tickers)
    clean = [str(ticker).upper().strip() for ticker in raw if str(ticker).strip()]
    return list(dict.fromkeys(clean))


def local_price_candidates(ticker: str) -> list[Path]:
    """Return likely local CSV/parquet files in DB_BASE for one ticker."""
    safe = ticker.lower().replace(".", "_").replace("-", "_")
    return [
        DB_BASE / "market" / f"{safe}.parquet",
        DB_BASE / "market" / f"{safe}.csv",
        DB_BASE / "europe_italy_market_data" / f"{safe}_1d.parquet",
        DB_BASE / "europe_stoxx_companies" / f"{safe}_1d.parquet",
    ]


def _load_local_price_file(ticker: str) -> pd.DataFrame | None:
    for candidate in local_price_candidates(ticker):
        if not candidate.exists():
            continue
        try:
            if candidate.suffix.lower() == ".parquet":
                frame = pd.read_parquet(candidate)
            else:
                frame = pd.read_csv(candidate)
            frame = frame.copy()
            frame.columns = [str(col).lower() for col in frame.columns]
            if "date" not in frame.columns:
                frame = frame.reset_index().rename(columns={"index": "date"})
            price_col = "adj_close" if "adj_close" in frame.columns else "close"
            if price_col not in frame.columns:
                continue
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
            frame["ticker"] = ticker
            frame["close"] = pd.to_numeric(frame.get("close", frame[price_col]), errors="coerce")
            frame["adj_close"] = pd.to_numeric(frame[price_col], errors="coerce")
            frame["open"] = pd.to_numeric(frame.get("open", frame["adj_close"]), errors="coerce")
            frame["high"] = pd.to_numeric(frame.get("high", frame[["open", "adj_close"]].max(axis=1)), errors="coerce")
            frame["low"] = pd.to_numeric(frame.get("low", frame[["open", "adj_close"]].min(axis=1)), errors="coerce")
            frame["volume"] = pd.to_numeric(frame.get("volume", 1_000_000), errors="coerce").fillna(1_000_000)
            frame["source"] = "local_db"
            return frame[["date", "ticker", "open", "high", "low", "close", "adj_close", "volume", "source"]].dropna(subset=["date", "adj_close"])
        except Exception as exc:
            LOGGER.warning("Failed to read local price file %s: %s", candidate, exc)
    return None


def synthetic_price_panel(tickers: list[str], periods: int = 504, end: str | None = None) -> pd.DataFrame:
    """Generate realistic synthetic daily OHLCV data when real data are unavailable."""
    end_date = pd.Timestamp(end or datetime.today().date())
    dates = pd.bdate_range(end=end_date, periods=periods)
    frames: list[pd.DataFrame] = []
    for ticker in tickers:
        rng = np.random.default_rng(stable_seed(ticker))
        drift = rng.normal(0.00035, 0.00015)
        vol = abs(rng.normal(0.018, 0.006))
        shocks = rng.normal(drift, vol, len(dates))
        cycle = 0.0008 * np.sin(np.linspace(0, 8 * np.pi, len(dates)) + rng.uniform(0, 2 * np.pi))
        close = 100 * np.exp(np.cumsum(shocks + cycle))
        open_ = close * (1 + rng.normal(0, 0.0025, len(dates)))
        high = np.maximum(open_, close) * (1 + rng.uniform(0.0005, 0.015, len(dates)))
        low = np.minimum(open_, close) * (1 - rng.uniform(0.0005, 0.015, len(dates)))
        volume = rng.lognormal(mean=14.3, sigma=0.45, size=len(dates)).astype(int)
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "ticker": ticker,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "adj_close": close,
                    "volume": volume,
                    "source": "synthetic",
                    "price_synthetic": True,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def load_price_panel(tickers: Iterable[str] | str | None, periods: int = 504, end: str | None = None) -> pd.DataFrame:
    """Load local/remote prices with synthetic fallback and no repo-local writes."""
    ensure_base_directories()
    ticker_list = normalize_tickers(tickers)
    frames: list[pd.DataFrame] = []
    missing: list[str] = []
    for ticker in ticker_list:
        local = _load_local_price_file(ticker)
        if local is not None and not local.empty:
            frames.append(local.tail(periods))
        else:
            missing.append(ticker)

    if missing and os.environ.get("ML_TRADING_OFFLINE", "0") != "1":
        try:
            import yfinance as yf

            downloaded = yf.download(missing, period=f"{max(periods // 21, 2)}mo", auto_adjust=False, progress=False, group_by="ticker")
            for ticker in missing[:]:
                try:
                    raw = downloaded[ticker] if isinstance(downloaded.columns, pd.MultiIndex) else downloaded
                    if raw.empty:
                        continue
                    frame = raw.reset_index()
                    frame.columns = [str(col).lower().replace(" ", "_") for col in frame.columns]
                    frame["ticker"] = ticker
                    frame["adj_close"] = frame.get("adj_close", frame.get("close"))
                    frame["source"] = "yfinance"
                    frames.append(frame[["date", "ticker", "open", "high", "low", "close", "adj_close", "volume", "source"]].dropna())
                    missing.remove(ticker)
                except Exception as exc:
                    LOGGER.warning("Failed to normalize yfinance data for %s: %s", ticker, exc)
        except Exception as exc:
            LOGGER.warning("Real data fetch failed; using synthetic fallback: %s", exc)

    if missing:
        LOGGER.warning("Synthetic price fallback for tickers: %s", ", ".join(missing))
        frames.append(synthetic_price_panel(missing, periods=periods, end=end))

    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    panel["adj_close"] = pd.to_numeric(panel["adj_close"], errors="coerce")
    panel["volume"] = pd.to_numeric(panel["volume"], errors="coerce").fillna(0)
    return panel.dropna(subset=["date", "ticker", "adj_close"]).sort_values(["ticker", "date"]).reset_index(drop=True)


def add_return_features(price_panel: pd.DataFrame) -> pd.DataFrame:
    """Add lag-safe return and risk features by ticker."""
    frame = price_panel.sort_values(["ticker", "date"]).copy()
    grouped = frame.groupby("ticker", group_keys=False)
    frame["return_1d"] = grouped["adj_close"].pct_change()
    frame["return_21d"] = grouped["adj_close"].pct_change(21)
    frame["return_63d"] = grouped["adj_close"].pct_change(63)
    frame["volatility_63d"] = grouped["return_1d"].transform(lambda s: s.shift(1).rolling(63, min_periods=20).std() * np.sqrt(252))
    frame["dollar_volume"] = frame["adj_close"] * frame["volume"]
    frame["avg_dollar_volume_21d"] = grouped["dollar_volume"].transform(lambda s: s.shift(1).rolling(21, min_periods=5).mean())
    return frame


def latest_feature_snapshot(price_panel: pd.DataFrame) -> pd.DataFrame:
    """Return the latest feature row per ticker."""
    features = add_return_features(price_panel)
    latest = features.sort_values(["ticker", "date"]).groupby("ticker", as_index=False).tail(1)
    return latest.reset_index(drop=True)


def default_positions() -> pd.DataFrame:
    """Default demo portfolio used when no positions file is provided."""
    return pd.DataFrame(
        {
            "ticker": ["AAPL", "MSFT", "NVDA", "JPM", "SPY"],
            "weight": [0.22, 0.22, 0.18, 0.13, 0.25],
        }
    )


def load_positions(portfolio_file: str | Path | None = None) -> pd.DataFrame:
    """Load portfolio positions, falling back to a small default portfolio."""
    if portfolio_file is None:
        return default_positions()
    path = Path(portfolio_file).expanduser()
    if not path.exists():
        LOGGER.warning("Portfolio file not found; using default positions: %s", path)
        return default_positions()
    frame = pd.read_csv(path)
    frame.columns = [str(col).lower().strip() for col in frame.columns]
    if "ticker" not in frame.columns:
        raise ValueError("portfolio file must include a ticker column")
    frame["ticker"] = frame["ticker"].astype(str).str.upper().str.strip()
    if "weight" not in frame.columns:
        if "shares" in frame.columns:
            frame["weight"] = pd.to_numeric(frame["shares"], errors="coerce")
            frame["weight"] = frame["weight"] / frame["weight"].sum()
        else:
            frame["weight"] = 1 / len(frame)
    frame["weight"] = pd.to_numeric(frame["weight"], errors="coerce").fillna(0)
    total = frame["weight"].sum()
    frame["weight"] = frame["weight"] / total if total else 1 / len(frame)
    return frame[["ticker", "weight"]].drop_duplicates("ticker")


def dataframe_preview_markdown(frame: pd.DataFrame, rows: int = 12) -> str:
    """Return a Markdown preview without requiring tabulate."""
    if frame.empty:
        return "_No rows._"
    preview = frame.head(rows).copy()
    try:
        return preview.to_markdown(index=False)
    except Exception:
        return preview.to_string(index=False)
