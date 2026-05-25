from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_DIR.parent
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in [APP_DIR, PROJECT_ROOT, SRC_ROOT]:
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import pandas as pd
import plotly.express as px
import streamlit as st

from support import configure_page, dataframe_with_download, render_context_bar, render_footer, render_page_header, render_page_intro, safe_page_link, sidebar_roots

from research_platform_core import detect_market_regime, load_multi_asset_universe_manifest, load_time_series_forecast_artifacts, summarize_multi_asset_universe
from research_platform_core.macro_market import compile_macro_asset_database, load_macro_market_artifacts, macro_asset_catalog
from research_platform_core.sentiment_analysis import collect_ticker_sentiment


configure_page("Macro View")


def pct(value: object) -> str:
    try:
        if pd.isna(value):
            return "n/a"
        return f"{float(value):+.1%}"
    except Exception:
        return "n/a"


def status_badge(status: str) -> str:
    value = str(status or "UNKNOWN").upper()
    tone = {
        "OK": ("#ECFDF3", "#067647"),
        "READY": ("#ECFDF3", "#067647"),
        "PROXY": ("#E6F4F7", "#006D77"),
        "NO_DATA": ("#FFF4DF", "#8A4B00"),
        "FAILED": ("#FDECEC", "#8A1F11"),
    }.get(value, ("#F6F8FB", "#344054"))
    return f"<span style='background:{tone[0]};color:{tone[1]};border-radius:999px;padding:4px 9px;font-weight:700;font-size:12px'>{value}</span>"


roots = sidebar_roots()
artifacts = load_macro_market_artifacts(roots["financial_db"], roots["workspace"])
catalog = artifacts["catalog"]
manifest = artifacts["manifest"]
latest = artifacts["latest"]
history = artifacts["history_sample"]
ts_artifacts = load_time_series_forecast_artifacts(roots["workspace"])
ts_latest = ts_artifacts.get("latest", pd.DataFrame())
forecast_symbols = set(ts_latest["symbol"].dropna().astype(str).str.upper().tolist()) if not ts_latest.empty and "symbol" in ts_latest.columns else set()
multi_asset_manifest = load_multi_asset_universe_manifest(roots["financial_db"], roots["workspace"])
multi_asset_summary = summarize_multi_asset_universe(multi_asset_manifest)
current_regime = detect_market_regime(financial_db_root=roots["financial_db"], output_root=roots["workspace"])

render_page_header(
    "Macro View",
    "Global, USA, Europe, Italy, crypto, FX, commodities, ETFs and fixed-income context separated from issuer-level Smart Money evidence.",
    "◎",
    module="RESEARCH",
    status="WIP",
)
render_context_bar()
render_page_intro(
    "Explore broad market regimes and cross-asset proxies separately from issuer-level Smart Money signals.",
    "Select a region/asset class, review performance, then compile or refresh the macro database when needed.",
)

with st.container(border=True):
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Catalog Assets", len(catalog))
    c2.metric("Downloaded Assets", int(manifest["status"].astype(str).eq("OK").sum()) if not manifest.empty and "status" in manifest.columns else 0)
    c3.metric("Regions", catalog["region"].nunique() if not catalog.empty and "region" in catalog.columns else 0)
    c4.metric("Asset Classes", catalog["asset_class"].nunique() if not catalog.empty and "asset_class" in catalog.columns else 0)
    c5.metric("Current Regime", str(current_regime.get("regime", "unknown")).replace("_", " ").title(), str(current_regime.get("date", "")))
    if current_regime.get("drivers"):
        st.caption(f"Regime drivers: {current_regime.get('drivers')}")

with st.container(border=True):
    st.markdown("**Compile / refresh macro database**")
    st.caption("Writes metadata and market history under Database Finanziario plus lightweight preview tables under `output/macro_market/tables`.")
    regions = sorted(catalog["region"].dropna().astype(str).unique().tolist()) if not catalog.empty and "region" in catalog.columns else ["global", "usa", "eu", "italy", "crypto"]
    classes = sorted(catalog["asset_class"].dropna().astype(str).unique().tolist()) if not catalog.empty and "asset_class" in catalog.columns else []
    selected_regions = st.multiselect("Regions", regions, default=[r for r in ["global", "usa", "eu", "italy", "crypto"] if r in regions])
    selected_classes = st.multiselect("Asset classes", classes, default=classes)
    c1, c2, c3, c4 = st.columns(4)
    period = c1.selectbox("History window", ["1y", "2y", "5y", "10y", "max"], index=2)
    interval = c2.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)
    max_assets = c3.number_input("Max assets", min_value=0, max_value=250, value=0, help="0 means all selected assets.")
    refresh = c4.toggle("Force refresh", value=False)
    if st.button("Compile macro database", width="stretch"):
        with st.spinner("Downloading macro proxies and writing manifests..."):
            result = compile_macro_asset_database(
                roots["financial_db"],
                roots["workspace"],
                regions=selected_regions,
                asset_classes=selected_classes,
                period=period,
                interval=interval,
                refresh=refresh,
                max_assets=int(max_assets) or None,
            )
        st.success(f"Macro database updated: {result['summary']['ok_count']} / {result['summary']['asset_count']} assets with data.")
        st.json(result["summary"])
        st.rerun()

filters = st.container(border=True)
with filters:
    st.markdown("**Macro filters**")
    f1, f2, f3 = st.columns(3)
    view_region = f1.selectbox("Region lens", ["All", *regions], index=0)
    view_class = f2.selectbox("Asset class", ["All", *classes], index=0)
    sort_metric = f3.selectbox("Sort metric", ["return_1m", "return_3m", "return_1y", "last_close"], index=1)
    if forecast_symbols:
        shortcut_symbols = sorted(forecast_symbols)
        c1, c2 = st.columns([0.35, 0.65])
        ts_symbol = c1.selectbox("Forecast shortcut", shortcut_symbols, help="Open Time Series Lab with this macro series pre-selected.")
        if c2.button("Open forecast in Time Series Lab", width="stretch"):
            st.session_state["ts_lab_source"] = "macro"
            st.session_state["ts_lab_symbol"] = ts_symbol
            st.switch_page("pages/14_⏱️_Time_Series_Lab.py")

if not latest.empty and not catalog.empty and "symbol" in latest.columns and "symbol" in catalog.columns:
    latest_symbols = set(latest["symbol"].dropna().astype(str).str.upper())
    planned_assets = catalog[~catalog["symbol"].astype(str).str.upper().isin(latest_symbols)].copy()
    if not planned_assets.empty:
        planned_assets["status"] = planned_assets.get("status", "PLANNED")
    view = pd.concat([latest, planned_assets], ignore_index=True, sort=False)
else:
    view = latest.copy()
    if view.empty:
        view = catalog.copy()
if not view.empty and "symbol" in view.columns:
    view["ts_forecast"] = view["symbol"].astype(str).str.upper().map(lambda symbol: "Forecast available" if symbol in forecast_symbols else "No forecast")
    if forecast_symbols and not ts_latest.empty and "horizon_days" in ts_latest.columns:
        horizon_map = (
            ts_latest.assign(symbol=ts_latest["symbol"].astype(str).str.upper())
            .groupby("symbol")["horizon_days"]
            .apply(lambda values: "/".join(str(int(v)) for v in sorted(pd.Series(values).dropna().astype(int).unique())))
            .to_dict()
        )
        view["ts_horizons"] = view["symbol"].astype(str).str.upper().map(horizon_map).fillna("")
if view_region != "All" and "region" in view.columns:
    view = view[view["region"].astype(str).eq(view_region)].copy()
if view_class != "All" and "asset_class" in view.columns:
    view = view[view["asset_class"].astype(str).eq(view_class)].copy()
if sort_metric in view.columns:
    view = view.sort_values(sort_metric, ascending=False)

tabs = st.tabs(["Overview", "Global", "USA", "EU", "Italy", "Crypto", "Sentiment", "Metadata & Database"])

with tabs[0]:
    if latest.empty:
        st.info("No macro price snapshot yet. Compile the macro database to populate charts and performance cards.")
    else:
        cards = st.columns(4)
        for idx, row in enumerate(view.head(4).itertuples(index=False)):
            with cards[idx % 4]:
                st.markdown(f"**{getattr(row, 'symbol', '')}**")
                st.caption(getattr(row, "name", ""))
                st.metric("Last", f"{getattr(row, 'last_close', 'n/a')}", pct(getattr(row, "return_1m", None)))
                if str(getattr(row, "symbol", "")).upper() in forecast_symbols:
                    st.caption(f"Forecast available ({getattr(row, 'ts_horizons', '')}d)")
                st.markdown(status_badge(getattr(row, "status", "PLANNED")), unsafe_allow_html=True)
        chart_cols = [col for col in ["return_1m", "return_3m", "return_1y"] if col in view.columns]
        if chart_cols:
            melted = view[["symbol", "asset_class", *chart_cols]].melt(id_vars=["symbol", "asset_class"], var_name="horizon", value_name="return")
            st.plotly_chart(px.bar(melted.dropna(), x="symbol", y="return", color="horizon", facet_row="asset_class", title="Cross-asset return map", template="plotly_white"), width="stretch")
    dataframe_with_download("Macro current view", view, "macro_current_view.csv")

for tab, region_name in zip(tabs[1:6], ["global", "usa", "eu", "italy", "crypto"]):
    with tab:
        region_latest = latest[latest["region"].astype(str).eq(region_name)].copy() if not latest.empty and "region" in latest.columns else pd.DataFrame()
        region_catalog = catalog[catalog["region"].astype(str).eq(region_name)].copy() if not catalog.empty and "region" in catalog.columns else pd.DataFrame()
        if not region_latest.empty and "symbol" in region_latest.columns:
            region_latest["ts_forecast"] = region_latest["symbol"].astype(str).str.upper().map(lambda symbol: "Forecast available" if symbol in forecast_symbols else "No forecast")
        st.markdown(f"### {region_name.upper()} macro board")
        if region_latest.empty:
            st.warning("This region has metadata, but no downloaded market history yet. Use Compile macro database.")
            dataframe_with_download(f"{region_name} metadata", region_catalog, f"macro_{region_name}_metadata.csv")
        else:
            k1, k2, k3 = st.columns(3)
            k1.metric("Assets", len(region_latest))
            k2.metric("Best 1M", pct(region_latest["return_1m"].max()) if "return_1m" in region_latest.columns else "n/a")
            k3.metric("Worst 1M", pct(region_latest["return_1m"].min()) if "return_1m" in region_latest.columns else "n/a")
            st.plotly_chart(
                px.scatter(
                    region_latest,
                    x="return_1m",
                    y="return_3m" if "return_3m" in region_latest.columns else "return_1m",
                    size="last_close" if "last_close" in region_latest.columns else None,
                    color="asset_class",
                    hover_name="name",
                    text="symbol",
                    title=f"{region_name.upper()} regime map",
                    template="plotly_white",
                ),
                width="stretch",
            )
            dataframe_with_download(f"{region_name} macro snapshot", region_latest, f"macro_{region_name}_snapshot.csv")

with tabs[6]:
    st.markdown("### Social sentiment")
    st.caption("Research context from optional public/social providers. It is noisy, rate-limited and never treated as a trade decision.")
    default_ticker = str(st.session_state.get("selected_ticker", "") or "AAPL").upper()
    sent_cols = st.columns([0.25, 0.35, 0.2, 0.2])
    ticker = sent_cols[0].text_input("Ticker / symbol", value=default_ticker).strip().upper()
    providers = sent_cols[1].multiselect("Providers", ["stocktwits", "reddit", "x"], default=["stocktwits", "reddit"])
    limit = sent_cols[2].number_input("Items / provider", 5, 100, 25, step=5)
    run = sent_cols[3].button("Run sentiment", width="stretch")
    if run:
        with st.spinner("Collecting public social mentions..."):
            sentiment = collect_ticker_sentiment(ticker, roots["workspace"], providers=providers, limit=int(limit))
        summary = sentiment["summary"]
        mentions = sentiment["mentions"]
        if not summary.empty:
            s1, s2, s3, s4 = st.columns(4)
            row = summary.iloc[0]
            s1.metric("Mentions", int(row.get("mention_count", 0)))
            s2.metric("Avg sentiment", f"{float(row.get('avg_sentiment', 0.0)):+.2f}")
            s3.metric("Positive", int(row.get("positive_count", 0)))
            s4.metric("Negative", int(row.get("negative_count", 0)))
        if mentions.empty:
            st.info("No mentions returned. Check provider availability or API credentials.")
        else:
            show_cols = [col for col in ["provider", "ticker", "created_at", "author", "sentiment_label", "sentiment_score", "status", "text", "url", "error"] if col in mentions.columns]
            st.dataframe(mentions[show_cols].head(200), width="stretch", hide_index=True)
            ok = mentions[mentions.get("status", pd.Series("", index=mentions.index)).astype(str).eq("OK")] if "status" in mentions.columns else mentions
            if not ok.empty and "sentiment_label" in ok.columns:
                st.plotly_chart(px.histogram(ok, x="sentiment_label", color="provider", title="Mention sentiment distribution", template="plotly_white"), width="stretch")
    with st.expander("Provider setup", expanded=False):
        st.markdown(
            """
            - StockTwits: public symbol stream endpoint; no key configured by default.
            - Reddit: public JSON search for selected subreddits; set `REDDIT_USER_AGENT` for polite API usage.
            - X: set `X_BEARER_TOKEN`; otherwise the panel reports `UNCONFIGURED`.
            """
        )

with tabs[7]:
    st.markdown("### Macro metadata & database")
    st.caption("This is the asset map used to populate FX, commodities, ETFs, fixed income, crypto and country/regional macro proxies.")
    if not multi_asset_summary.empty:
        st.markdown("**Multi-asset universe coverage**")
        st.dataframe(multi_asset_summary, width="stretch", hide_index=True)
    dataframe_with_download("Macro asset catalog", catalog, "MacroAssetCatalog.csv")
    dataframe_with_download("Macro asset manifest", manifest, "MacroAssetManifest.csv")
    dataframe_with_download("Multi-asset universe manifest", multi_asset_manifest, "MultiAssetUniverseManifest.csv")
    if not history.empty:
        st.plotly_chart(px.line(history, x="date", y="close", color="symbol", title="History sample", template="plotly_white"), width="stretch")
        dataframe_with_download("Macro history sample", history, "MacroHistorySample.csv")
    with st.expander("Database paths", expanded=False):
        st.code(str(roots["financial_db"] / "MarketData" / "Macro"))
        st.code(str(roots["workspace"] / "macro_market" / "tables"))
        safe_page_link("pages/8_🗄️_Data_Platform.py", "Open Data Platform")

render_footer()
