"""Macro analysis engine for Analysis Studio."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .common import StudioResult, load_positions, px, stable_seed


class MacroAnalysisEngine:
    """Estimate portfolio sensitivity to a small macro factor set."""

    analysis_name = "macro_analysis"

    def __init__(self, portfolio_file: str | None = None) -> None:
        self.portfolio_file = portfolio_file
        self.last_result: StudioResult | None = None
        self.last_charts: list[object] = []
        self.last_summary = ""

    def run(self, portfolio_file: str | None = None) -> pd.DataFrame:
        positions = load_positions(portfolio_file or self.portfolio_file)
        factors = ["rates", "inflation", "usd", "oil", "credit_spread", "global_growth"]
        rows = []
        portfolio_tag = ",".join(positions["ticker"].tolist())
        for factor in factors:
            rng = np.random.default_rng(stable_seed(f"macro-{factor}-{portfolio_tag}"))
            exposure = rng.normal(0, 0.35)
            shock = rng.normal(0, 1.0)
            rows.append(
                {
                    "factor": factor,
                    "portfolio_exposure": exposure,
                    "current_zscore": shock,
                    "bear_shock": -2.0,
                    "base_shock": 0.0,
                    "bull_shock": 2.0,
                    "bear_pnl_estimate": exposure * -2.0,
                    "base_pnl_estimate": 0.0,
                    "bull_pnl_estimate": exposure * 2.0,
                    "source": "synthetic_macro_proxy",
                    "macro_synthetic": True,
                }
            )
        result = pd.DataFrame(rows)
        result["abs_exposure"] = result["portfolio_exposure"].abs()
        result = result.sort_values("abs_exposure", ascending=False).reset_index(drop=True)
        charts = []
        if px is not None:
            charts.append(px.bar(result, x="factor", y="portfolio_exposure", title="Macro Factor Exposure"))
            scenario = result.melt(id_vars="factor", value_vars=["bear_pnl_estimate", "base_pnl_estimate", "bull_pnl_estimate"], var_name="scenario", value_name="pnl_estimate")
            charts.append(px.bar(scenario, x="factor", y="pnl_estimate", color="scenario", barmode="group", title="Scenario PnL Estimate"))
        top = result.iloc[0]["factor"] if not result.empty else "N/A"
        summary = f"Largest modeled macro sensitivity is {top}."
        self.last_charts = charts
        self.last_summary = summary
        self.last_result = StudioResult(self.analysis_name, result, summary, charts, {"portfolio_rows": len(positions)})
        return result

