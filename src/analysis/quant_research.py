"""Quantitative research engine for Analysis Studio."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .common import StudioResult, load_price_panel, px


class QuantResearchEngine:
    """Run a compact single-name factor and walk-forward diagnostic."""

    analysis_name = "quantitative_research"

    def __init__(self, ticker: str = "MSFT") -> None:
        self.ticker = ticker.upper()
        self.last_result: StudioResult | None = None
        self.last_charts: list[object] = []
        self.last_summary = ""

    def run(self, ticker: str | None = None) -> pd.DataFrame:
        ticker = (ticker or self.ticker).upper()
        frame = load_price_panel([ticker], periods=756).sort_values("date").copy()
        frame["return_1d"] = frame["adj_close"].pct_change()
        frame["momentum_21d"] = frame["adj_close"].pct_change(21).shift(1)
        frame["reversal_5d"] = -frame["adj_close"].pct_change(5).shift(1)
        frame["volatility_21d"] = frame["return_1d"].shift(1).rolling(21, min_periods=10).std()
        frame["target_5d"] = frame["adj_close"].pct_change(5).shift(-5)
        model_frame = frame.dropna(subset=["momentum_21d", "reversal_5d", "volatility_21d", "target_5d"]).copy()
        if model_frame.empty:
            result = frame.tail(30).copy()
            result["prediction"] = 0.0
        else:
            features = model_frame[["momentum_21d", "reversal_5d", "volatility_21d"]]
            coefs = np.array([0.45, 0.35, -0.20])
            scaled = (features - features.mean()) / features.std(ddof=0).replace(0, 1)
            model_frame["prediction"] = scaled @ coefs
            model_frame["strategy_return"] = np.sign(model_frame["prediction"]) * model_frame["target_5d"]
            model_frame["equity_curve"] = (1 + model_frame["strategy_return"].fillna(0)).cumprod()
            result = model_frame.tail(252).reset_index(drop=True)
        charts = []
        if px is not None and not result.empty:
            if "equity_curve" in result.columns:
                charts.append(px.line(result, x="date", y="equity_curve", title=f"{ticker} Factor Strategy Equity"))
            charts.append(px.scatter(result, x="prediction", y="target_5d", trendline="ols", title="Prediction vs Forward Return"))
        ic = float(result[["prediction", "target_5d"]].corr().iloc[0, 1]) if {"prediction", "target_5d"}.issubset(result.columns) and len(result) > 5 else np.nan
        summary = f"{ticker} factor information coefficient: {ic:.3f}." if np.isfinite(ic) else f"{ticker} factor diagnostics generated."
        self.last_charts = charts
        self.last_summary = summary
        self.last_result = StudioResult(self.analysis_name, result, summary, charts, {"ticker": ticker, "information_coefficient": ic})
        return result

