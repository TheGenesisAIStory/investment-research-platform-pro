"""Earnings analysis engine for Analysis Studio."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .common import StudioResult, load_price_panel, px, stable_seed


class EarningsAnalysisEngine:
    """Build a compact earnings surprise and post-event drift view."""

    analysis_name = "earnings_analysis"

    def __init__(self, ticker: str = "NVDA") -> None:
        self.ticker = ticker.upper()
        self.last_result: StudioResult | None = None
        self.last_charts: list[object] = []
        self.last_summary = ""

    def run(self, ticker: str | None = None) -> pd.DataFrame:
        ticker = (ticker or self.ticker).upper()
        _ = load_price_panel([ticker], periods=504)
        rng = np.random.default_rng(stable_seed(f"earnings-{ticker}"))
        quarters = pd.period_range(end=pd.Timestamp.today().to_period("Q"), periods=12, freq="Q").to_timestamp("Q")
        eps_est = rng.normal(1.6, 0.35, len(quarters)).clip(0.1)
        surprise = rng.normal(0.04, 0.11, len(quarters))
        eps_actual = eps_est * (1 + surprise)
        drift = 0.35 * surprise + rng.normal(0.0, 0.045, len(quarters))
        result = pd.DataFrame(
            {
                "ticker": ticker,
                "earnings_date": quarters,
                "eps_estimate": eps_est,
                "eps_actual": eps_actual,
                "surprise_pct": surprise,
                "post_earnings_drift_20d": drift,
                "source": "synthetic_event_calendar",
                "earnings_synthetic": True,
            }
        )
        result["signal"] = pd.cut(result["surprise_pct"], [-np.inf, -0.05, 0.05, np.inf], labels=["miss", "inline", "beat"])
        charts = []
        if px is not None:
            charts.append(px.bar(result, x="earnings_date", y="surprise_pct", color="signal", title=f"{ticker} Earnings Surprise"))
            charts.append(px.scatter(result, x="surprise_pct", y="post_earnings_drift_20d", trendline="ols", title="Surprise vs 20D Drift"))
        hit_rate = float((np.sign(result["surprise_pct"]) == np.sign(result["post_earnings_drift_20d"])).mean())
        summary = f"{ticker} synthetic event study hit rate: {hit_rate:.1%} over {len(result)} quarters."
        self.last_charts = charts
        self.last_summary = summary
        self.last_result = StudioResult(self.analysis_name, result, summary, charts, {"ticker": ticker})
        return result

