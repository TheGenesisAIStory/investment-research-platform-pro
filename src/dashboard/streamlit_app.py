"""Shared Streamlit page renderer for Analysis Studio."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.analysis import (
    CompetitiveAnalysisEngine,
    EarningsAnalysisEngine,
    MacroAnalysisEngine,
    PortfolioBuilderEngine,
    PortfolioRiskEngine,
    QuantResearchEngine,
    StockScreenerEngine,
    TechnicalAnalysisEngine,
)
from src.reporting import export_analysis_report


@dataclass(frozen=True)
class PageSpec:
    title: str
    description: str
    runner: object


def _render_downloads(st, frame: pd.DataFrame, manifest: dict[str, str]) -> None:
    st.download_button("Download CSV", frame.to_csv(index=False), file_name=manifest["csv_path"].split("/")[-1], mime="text/csv")
    st.markdown(f"Report: `{manifest['html_path']}`")


def run_page(slug: str) -> None:
    import streamlit as st

    st.set_page_config(page_title="Analysis Studio", layout="wide")
    st.title("Analysis Studio")
    st.caption("Modular research workflows integrated with the local ML Trading workspace.")

    with st.sidebar:
        st.header("Controls")
        offline = st.toggle("Offline fallback mode", value=True)
        if offline:
            import os

            os.environ["ML_TRADING_OFFLINE"] = "1"

    if slug == "stock_screening":
        universe = st.sidebar.selectbox("Universe", ["core", "sp500", "technology", "semiconductors", "banks"])
        sector = st.sidebar.selectbox("Sector override", ["", "technology", "semiconductors", "financials"])
        top_n = st.sidebar.slider("Top N", 3, 20, 10)
        engine = StockScreenerEngine(universe=universe, sector=sector or None)
        run_kwargs = {"top_n": top_n}
        title = "Stock Screening"
    elif slug == "portfolio_risk":
        portfolio_file = st.sidebar.text_input("Portfolio CSV path", "")
        engine = PortfolioRiskEngine(portfolio_file=portfolio_file or None)
        run_kwargs = {}
        title = "Portfolio Risk"
    elif slug == "earnings_analysis":
        ticker = st.sidebar.text_input("Ticker", "NVDA").upper()
        engine = EarningsAnalysisEngine(ticker=ticker)
        run_kwargs = {}
        title = "Earnings Analysis"
    elif slug == "portfolio_builder":
        risk_profile = st.sidebar.selectbox("Risk profile", ["conservative", "moderate", "aggressive"], index=1)
        universe = st.sidebar.selectbox("Universe", ["core", "sp500", "technology", "semiconductors"])
        engine = PortfolioBuilderEngine(universe=universe, risk_profile=risk_profile)
        run_kwargs = {}
        title = "Portfolio Builder"
    elif slug == "technical_analysis":
        ticker = st.sidebar.text_input("Ticker", "AAPL").upper()
        engine = TechnicalAnalysisEngine(ticker=ticker)
        run_kwargs = {}
        title = "Technical Analysis"
    elif slug == "competitive_analysis":
        sector = st.sidebar.selectbox("Sector", ["semiconductors", "technology", "financials"])
        engine = CompetitiveAnalysisEngine(sector=sector)
        run_kwargs = {}
        title = "Competitive Analysis"
    elif slug == "quantitative_research":
        ticker = st.sidebar.text_input("Ticker", "MSFT").upper()
        engine = QuantResearchEngine(ticker=ticker)
        run_kwargs = {}
        title = "Quantitative Research"
    elif slug == "macro_analysis":
        portfolio_file = st.sidebar.text_input("Portfolio CSV path", "")
        engine = MacroAnalysisEngine(portfolio_file=portfolio_file or None)
        run_kwargs = {}
        title = "Macro Analysis"
    else:
        st.error(f"Unknown page: {slug}")
        return

    st.subheader(title)
    if st.button("Run analysis", type="primary"):
        frame = engine.run(**run_kwargs)
        manifest = export_analysis_report(engine.last_result)
        st.success(engine.last_summary)
        for chart in engine.last_charts:
            st.plotly_chart(chart, use_container_width=True)
        st.dataframe(frame, use_container_width=True)
        _render_downloads(st, frame, manifest)
    else:
        st.info("Choose filters in the sidebar and run the analysis.")

