from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_DIR.parents[0]
SRC_DIR = PROJECT_ROOT / "src"
for path in [APP_DIR, PROJECT_ROOT, SRC_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from support import configure_page, sidebar_roots

try:
    from src.research_platform_core import run_banks_data_pipeline
except Exception:
    try:
        from research_platform_core import run_banks_data_pipeline
    except Exception:
        run_banks_data_pipeline = None


def _read_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


configure_page("Banking Data Lab")
roots = sidebar_roots()

default_output = roots["workspace"] / "banks_pipeline"
default_cache = roots["workspace"] / "banks_pipeline" / "_cache"

st.title("Banking Data Lab")
st.caption("Open-source Italian and euro-area bank universe, regulatory/macro placeholders and listed-bank market panels.")

st.markdown(
    """
    <div class="rp-note">
    <b>Data policy:</b> Drive/repo cache first, public official sources next, API fallbacks only for listed-bank market data.
    ECB/Wikipedia/Banca d'Italia outputs are stored as auditable CSV/SQLite artifacts.
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Banking Pipeline")
    output_dir = Path(st.text_input("Output directory", str(default_output))).expanduser()
    cache_dir = Path(st.text_input("Cache directory", str(default_cache))).expanduser()
    market_start = st.text_input("Market start", "2015-01-01")
    include_market = st.checkbox("Refresh listed-bank market panel", value=False)
    include_ecb_macro = st.checkbox("Call ECB BSI macro now", value=False)

if st.button("Run Banking Data Pipeline", width="stretch", disabled=run_banks_data_pipeline is None):
    if run_banks_data_pipeline is None:
        st.error("Banking data engine is not importable in this app session.")
    else:
        with st.spinner("Refreshing banking universe and artifacts..."):
            artifacts = run_banks_data_pipeline(
                output_dir=output_dir,
                cache_dir=cache_dir,
                market_start=market_start,
                include_market=include_market,
                include_ecb_macro=include_ecb_macro,
            )
        st.success("Banking artifacts refreshed.")
        st.json({name: len(df) for name, df in artifacts.items()})

universe = _read_csv(output_dir / "banks_universe.csv")
macro = _read_csv(output_dir / "banks_macro_regulatory.csv")
fundamentals = _read_csv(output_dir / "banks_fundamentals_panel.csv")
market = _read_csv(output_dir / "banks_market_panel.csv")

cols = st.columns(4)
cols[0].metric("Universe Banks", len(universe))
cols[1].metric("Italian Banks", int(universe["country"].astype(str).eq("IT").sum()) if "country" in universe else 0)
cols[2].metric("Listed Banks", int(universe["listed_flag"].fillna(False).astype(bool).sum()) if "listed_flag" in universe else 0)
cols[3].metric("Market Rows", len(market))

tab_universe, tab_market, tab_macro, tab_contracts = st.tabs(["Universe", "Market Panel", "Regulatory / Macro", "Artifacts"])

with tab_universe:
    if universe.empty:
        st.info("No banking universe artifact found yet. Run the pipeline from the sidebar or from the Italian Banks notebook.")
    else:
        country_filter = st.multiselect("Country", sorted(universe["country"].dropna().astype(str).unique()), default=["IT"] if "IT" in set(universe["country"].astype(str)) else None)
        status_filter = st.multiselect("Status", sorted(universe["status"].dropna().astype(str).unique()) if "status" in universe else [])
        view = universe.copy()
        if country_filter:
            view = view[view["country"].astype(str).isin(country_filter)]
        if status_filter and "status" in view:
            view = view[view["status"].astype(str).isin(status_filter)]
        st.dataframe(view, width="stretch", hide_index=True)

with tab_market:
    if market.empty:
        st.info("No market panel yet. Enable market refresh in the sidebar or use the notebook yfinance fallback.")
    else:
        st.dataframe(market.head(500), width="stretch", hide_index=True)
        if {"date", "ticker", "mkt_price"}.issubset(market.columns):
            import plotly.express as px

            plot = market.copy()
            plot["date"] = pd.to_datetime(plot["date"], errors="coerce")
            st.plotly_chart(px.line(plot, x="date", y="mkt_price", color="ticker", title="Listed Bank Price Panel"), width="stretch")

with tab_macro:
    if macro.empty:
        st.info("No ECB/Banca d'Italia macro rows yet. ECB calls are optional; Banca d'Italia BDS export links can be configured in notebook/user config.")
    else:
        st.dataframe(macro.head(500), width="stretch", hide_index=True)
    if fundamentals.empty:
        st.caption("No external bank fundamentals loaded yet.")
    else:
        st.subheader("Fundamentals")
        st.dataframe(fundamentals.head(500), width="stretch", hide_index=True)

with tab_contracts:
    rows = []
    for name in ["banks_universe", "banks_macro_regulatory", "banks_fundamentals_panel", "banks_market_panel"]:
        path = output_dir / f"{name}.csv"
        rows.append(
            {
                "artifact": f"{name}.csv",
                "exists": path.exists(),
                "rows": len(_read_csv(path)),
                "path": str(path),
                "modified": pd.Timestamp(path.stat().st_mtime, unit="s").isoformat() if path.exists() else "",
            }
        )
    rows.append({"artifact": "banks_data.sqlite", "exists": (output_dir / "banks_data.sqlite").exists(), "rows": "", "path": str(output_dir / "banks_data.sqlite"), "modified": ""})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

