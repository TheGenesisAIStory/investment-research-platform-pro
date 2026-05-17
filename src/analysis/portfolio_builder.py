"""Portfolio builder engine for Analysis Studio."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .common import StudioResult, latest_feature_snapshot, load_price_panel, normalize_tickers, px, zscore


RISK_PROFILES = {
    "conservative": {"risk_penalty": 1.25, "max_weight": 0.18},
    "moderate": {"risk_penalty": 0.85, "max_weight": 0.25},
    "aggressive": {"risk_penalty": 0.45, "max_weight": 0.35},
}


class PortfolioBuilderEngine:
    """Create a simple risk-aware allocation from a candidate universe."""

    analysis_name = "portfolio_builder"

    def __init__(self, universe: str | list[str] = "core", risk_profile: str = "moderate") -> None:
        self.universe = universe
        self.risk_profile = risk_profile
        self.last_result: StudioResult | None = None
        self.last_charts: list[object] = []
        self.last_summary = ""

    def run(self, risk_profile: str | None = None, universe: str | list[str] | None = None) -> pd.DataFrame:
        profile = (risk_profile or self.risk_profile).lower()
        params = RISK_PROFILES.get(profile, RISK_PROFILES["moderate"])
        tickers = normalize_tickers(universe or self.universe)
        snapshot = latest_feature_snapshot(load_price_panel(tickers, periods=504))
        snapshot["expected_return_score"] = zscore(snapshot["return_63d"].fillna(0)) + 0.5 * zscore(snapshot["return_21d"].fillna(0))
        snapshot["risk_penalty"] = zscore(snapshot["volatility_63d"].fillna(snapshot["volatility_63d"].median()))
        snapshot["allocation_score"] = snapshot["expected_return_score"] - params["risk_penalty"] * snapshot["risk_penalty"]
        raw = np.exp(snapshot["allocation_score"].clip(-3, 3))
        weights = raw / raw.sum()
        weights = np.minimum(weights, params["max_weight"])
        weights = weights / weights.sum()
        result = snapshot[["ticker", "return_63d", "volatility_63d", "allocation_score"]].copy()
        result["target_weight"] = weights
        result["risk_profile"] = profile
        result = result.sort_values("target_weight", ascending=False).reset_index(drop=True)
        charts = []
        if px is not None:
            charts.append(px.bar(result, x="ticker", y="target_weight", title=f"{profile.title()} Target Weights"))
            charts.append(px.scatter(result, x="volatility_63d", y="return_63d", size="target_weight", color="ticker", title="Allocation Map"))
        summary = f"Built a {profile} portfolio across {len(result)} tickers with max weight {params['max_weight']:.0%}."
        self.last_charts = charts
        self.last_summary = summary
        self.last_result = StudioResult(self.analysis_name, result, summary, charts, {"risk_profile": profile, "tickers": tickers})
        return result

