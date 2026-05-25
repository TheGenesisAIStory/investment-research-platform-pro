from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf


APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from support import (
    FOOTER_TEXT,
    configure_page,
    latest_modified_label,
    load_company_artifacts,
    load_ohlcv_coverage_manifest,
    load_portfolio_artifacts,
    ohlcv_coverage_counts,
    render_context_bar,
    render_page_intro,
    render_section_kicker,
    run_history_frame,
    sidebar_roots,
)
from data_bootstrap import render_bootstrap_banner


configure_page("Home")


@st.cache_data(ttl=300, show_spinner=False)
def load_market_snapshot() -> dict[str, object]:
    symbols = {"^GSPC": "S&P 500", "^VIX": "VIX"}
    frames: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        try:
            frame = yf.Ticker(symbol).history(period="35d", interval="1d", auto_adjust=False)
            frames[symbol] = frame.dropna(subset=["Close"]) if not frame.empty else pd.DataFrame()
        except Exception:
            frames[symbol] = pd.DataFrame()

    spx = frames.get("^GSPC", pd.DataFrame())
    vix = frames.get("^VIX", pd.DataFrame())
    sentiment = "Waiting for data"
    delta = "n/a"
    if len(spx) >= 2:
        spx_delta = float(spx["Close"].iloc[-1] - spx["Close"].iloc[-2])
        spx_delta_pct = spx_delta / float(spx["Close"].iloc[-2])
        vix_last = float(vix["Close"].iloc[-1]) if not vix.empty else 20.0
        if spx_delta_pct > 0 and vix_last < 20:
            sentiment = "Risk-on"
        elif spx_delta_pct < 0 or vix_last > 25:
            sentiment = "Risk-off"
        else:
            sentiment = "Neutral"
        delta = f"{spx_delta_pct:.2%} vs prev close"
    return {"sentiment": sentiment, "delta": delta, "spx": spx}


def action_button(label: str, page: str) -> None:
    if st.button(label, width="stretch"):
        st.switch_page(page)


roots = sidebar_roots()
company = load_company_artifacts(roots["company"])
portfolio = load_portfolio_artifacts(roots["portfolio"])
market = load_market_snapshot()
runs = run_history_frame()
ohlcv_manifest = load_ohlcv_coverage_manifest(roots)
ohlcv_counts = ohlcv_coverage_counts(ohlcv_manifest)

st.title("Investment Research Platform")
st.caption("Research dashboard for valuation, portfolio construction, data operations and ML-assisted analysis.")
render_context_bar()
render_page_intro(
    "Start here to understand platform health, market context, data coverage and the most useful daily research links.",
    "Use Quick Actions to move into Screening, Valuation, Portfolio or Data Platform.",
)
render_bootstrap_banner(roots, required=["equity_metadata", "company_screener", "ml_signals", "smart_money_scores"])

screeners = company["screener_presets"]
active_screeners = len(screeners) if not screeners.empty else (1 if not company["screener_results"].empty else 0)
allocation = portfolio["allocation"] if not portfolio["allocation"].empty else portfolio["selection_results"]
portfolio_count = allocation["portfolio"].nunique() if not allocation.empty and "portfolio" in allocation.columns else len(allocation)

kpi_cols = st.columns(4)
kpi_cols[0].metric("Market Sentiment", str(market["sentiment"]), str(market["delta"]))
kpi_cols[1].metric("Active Screeners", active_screeners)
kpi_cols[2].metric("Last Valuation Run", latest_modified_label(roots["company"]))
kpi_cols[3].metric("Portfolio Count", portfolio_count)

left, right = st.columns([0.56, 0.44])
with left:
    render_section_kicker("Market pulse")
    spx = market["spx"]
    if isinstance(spx, pd.DataFrame) and not spx.empty:
        chart = spx.tail(30).reset_index()
        st.plotly_chart(
            px.line(chart, x=chart.columns[0], y="Close", title="S&P 500 · last 30 trading days", template="plotly_white"),
            width="stretch",
        )
    else:
        st.info("Market data is temporarily unavailable. Cached artifacts remain usable.")
with right:
    render_section_kicker("Data coverage")
    dc1, dc2 = st.columns(2)
    dc1.metric("OHLCV OK", ohlcv_counts["OK"])
    dc2.metric("Limited History", ohlcv_counts["LIMITED_HISTORY"])
    dc3, dc4 = st.columns(2)
    dc3.metric("Delisted", ohlcv_counts["DELISTED"])
    dc4.metric("Network Timeouts", ohlcv_counts["NETWORK_TIMEOUT"])
    with st.expander("What do coverage states mean?", expanded=False):
        st.write("`LIMITED_HISTORY` means the ticker is active but listed after the requested 2000 start date. It is usable; pre-listing rows are not forced.")

render_section_kicker("Quick actions")
with st.container(border=True):
    q1, q2, q3 = st.columns(3)
    with q1:
        action_button("Open Valuation Research", "pages/2_🔬_Valuation_Research.py")
        action_button("Open Portfolio Research", "pages/3_📁_Portfolio_Research.py")
    with q2:
        action_button("Open Smart Money Macro", "pages/1_📡_Smart_Money_Macro.py")
        action_button("Open Screener Builder", "pages/4_🔍_Screener_Builder.py")
    with q3:
        action_button("Open Export Center", "pages/5_📤_Export_Center.py")
        action_button("Open Notebook Runner", "pages/6_🧪_Notebook_Runner.py")

render_section_kicker("Recent activity")
if runs.empty:
    placeholder = pd.DataFrame(
        [
            {"time": "pending", "area": "Valuation", "activity": "Run the company valuation notebook or use Notebook Runner."},
            {"time": "pending", "area": "Portfolio", "activity": "Refresh allocation artifacts after a valuation run."},
            {"time": "pending", "area": "Data Center", "activity": "Open Data Platform to check source freshness."},
        ]
    )
    st.dataframe(placeholder, width="stretch", hide_index=True)
else:
    display_cols = ["run_id", "job_id", "status", "created_at", "finished_at", "runner_type"]
    st.dataframe(runs[[c for c in display_cols if c in runs.columns]].head(8), width="stretch", hide_index=True)

st.caption(FOOTER_TEXT)
