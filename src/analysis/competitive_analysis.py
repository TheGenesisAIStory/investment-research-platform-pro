"""Competitive analysis engine for Analysis Studio."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .common import StudioResult, latest_feature_snapshot, load_price_panel, normalize_tickers, px, stable_seed, zscore


class CompetitiveAnalysisEngine:
    """Score companies within a sector on growth, margin, risk, and valuation."""

    analysis_name = "competitive_analysis"

    def __init__(self, sector: str = "semiconductors") -> None:
        self.sector = sector
        self.last_result: StudioResult | None = None
        self.last_charts: list[object] = []
        self.last_summary = ""

    def run(self, sector: str | None = None) -> pd.DataFrame:
        sector = (sector or self.sector).lower()
        tickers = normalize_tickers(None, sector=sector)
        snapshot = latest_feature_snapshot(load_price_panel(tickers, periods=504))
        rows = []
        for _, row in snapshot.iterrows():
            ticker = row["ticker"]
            rng = np.random.default_rng(stable_seed(f"competitive-{sector}-{ticker}"))
            rows.append(
                {
                    "ticker": ticker,
                    "sector": sector,
                    "revenue_growth": rng.normal(0.10, 0.08),
                    "operating_margin": rng.normal(0.24, 0.07),
                    "r_and_d_intensity": rng.normal(0.13, 0.04),
                    "valuation_multiple": abs(rng.normal(28, 9)),
                    "price_momentum_63d": row.get("return_63d", np.nan),
                    "source": "market_plus_synthetic_fundamentals",
                    "fundamentals_synthetic": True,
                }
            )
        result = pd.DataFrame(rows)
        result["moat_score"] = (
            zscore(result["revenue_growth"]) + zscore(result["operating_margin"]) + 0.5 * zscore(result["r_and_d_intensity"]) - 0.35 * zscore(result["valuation_multiple"])
        )
        result = result.sort_values("moat_score", ascending=False).reset_index(drop=True)
        result["rank"] = range(1, len(result) + 1)
        charts = []
        if px is not None:
            charts.append(px.bar(result, x="ticker", y="moat_score", title=f"{sector.title()} Competitive Moat Score"))
            charts.append(px.scatter(result, x="valuation_multiple", y="revenue_growth", size="operating_margin", color="ticker", title="Growth vs Valuation"))
        leader = result.iloc[0]["ticker"] if not result.empty else "N/A"
        summary = f"{leader} ranks first in the {sector} competitive screen."
        self.last_charts = charts
        self.last_summary = summary
        self.last_result = StudioResult(self.analysis_name, result, summary, charts, {"sector": sector})
        return result

