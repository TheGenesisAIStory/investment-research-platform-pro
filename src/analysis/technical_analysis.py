"""Technical analysis engine for Analysis Studio."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .common import StudioResult, load_price_panel, px, go


class TechnicalAnalysisEngine:
    """Compute moving-average, RSI, and MACD diagnostics."""

    analysis_name = "technical_analysis"

    def __init__(self, ticker: str = "AAPL") -> None:
        self.ticker = ticker.upper()
        self.last_result: StudioResult | None = None
        self.last_charts: list[object] = []
        self.last_summary = ""

    def run(self, ticker: str | None = None) -> pd.DataFrame:
        ticker = (ticker or self.ticker).upper()
        frame = load_price_panel([ticker], periods=260)
        frame = frame.sort_values("date").copy()
        close = frame["adj_close"]
        frame["sma_20"] = close.shift(1).rolling(20, min_periods=10).mean()
        frame["sma_50"] = close.shift(1).rolling(50, min_periods=20).mean()
        delta = close.diff()
        gain = delta.clip(lower=0).shift(1).rolling(14, min_periods=7).mean()
        loss = (-delta.clip(upper=0)).shift(1).rolling(14, min_periods=7).mean()
        rs = gain / loss.replace(0, np.nan)
        frame["rsi_14"] = 100 - 100 / (1 + rs)
        ema_12 = close.shift(1).ewm(span=12, adjust=False).mean()
        ema_26 = close.shift(1).ewm(span=26, adjust=False).mean()
        frame["macd"] = ema_12 - ema_26
        frame["macd_signal"] = frame["macd"].ewm(span=9, adjust=False).mean()
        frame["trend_signal"] = np.where(frame["sma_20"] > frame["sma_50"], "bullish", "bearish")
        result = frame.tail(120).reset_index(drop=True)
        charts = []
        if go is not None:
            fig = go.Figure()
            fig.add_scatter(x=result["date"], y=result["adj_close"], name="Adj Close")
            fig.add_scatter(x=result["date"], y=result["sma_20"], name="SMA 20")
            fig.add_scatter(x=result["date"], y=result["sma_50"], name="SMA 50")
            fig.update_layout(title=f"{ticker} Price Trend")
            charts.append(fig)
        if px is not None:
            charts.append(px.line(result, x="date", y=["rsi_14", "macd", "macd_signal"], title=f"{ticker} Momentum Indicators"))
        latest = result.iloc[-1]
        summary = f"{ticker} latest trend is {latest['trend_signal']} with RSI {latest['rsi_14']:.1f}."
        self.last_charts = charts
        self.last_summary = summary
        self.last_result = StudioResult(self.analysis_name, result, summary, charts, {"ticker": ticker})
        return result

