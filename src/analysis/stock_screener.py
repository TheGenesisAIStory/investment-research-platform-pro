"""Stock screening engine for Analysis Studio."""

from __future__ import annotations

import pandas as pd

from .common import StudioResult, latest_feature_snapshot, load_price_panel, normalize_tickers, px, zscore


class StockScreenerEngine:
    """Rank equities using momentum, risk, and liquidity features."""

    analysis_name = "stock_screening"

    def __init__(self, universe: str | list[str] = "core", sector: str | None = None) -> None:
        self.universe = universe
        self.sector = sector
        self.last_result: StudioResult | None = None
        self.last_charts: list[object] = []
        self.last_summary = ""

    def run(self, top_n: int = 10, universe: str | list[str] | None = None, sector: str | None = None) -> pd.DataFrame:
        tickers = normalize_tickers(universe or self.universe, sector=sector or self.sector)
        panel = load_price_panel(tickers, periods=504)
        latest = latest_feature_snapshot(panel)
        latest["momentum_score"] = zscore(latest["return_63d"].fillna(0)) + 0.5 * zscore(latest["return_21d"].fillna(0))
        latest["risk_score"] = -zscore(latest["volatility_63d"].fillna(latest["volatility_63d"].median()))
        latest["liquidity_score"] = zscore(latest["avg_dollar_volume_21d"].fillna(0))
        latest["quality_score"] = latest["momentum_score"] + latest["risk_score"] + 0.25 * latest["liquidity_score"]
        result = latest.sort_values("quality_score", ascending=False).head(top_n).copy()
        result["rank"] = range(1, len(result) + 1)
        cols = [
            "rank",
            "ticker",
            "date",
            "adj_close",
            "return_21d",
            "return_63d",
            "volatility_63d",
            "avg_dollar_volume_21d",
            "quality_score",
            "source",
        ]
        result = result[[col for col in cols if col in result.columns]].reset_index(drop=True)
        charts = []
        if px is not None and not result.empty:
            charts.append(px.bar(result, x="ticker", y="quality_score", title="Top Screening Scores"))
            charts.append(px.scatter(result, x="volatility_63d", y="return_63d", size="avg_dollar_volume_21d", color="ticker", title="Momentum vs Risk"))
        summary = f"Screened {len(tickers)} tickers and selected top {len(result)} names by quality score."
        self.last_charts = charts
        self.last_summary = summary
        self.last_result = StudioResult(self.analysis_name, result, summary, charts, {"tickers": tickers})
        return result

