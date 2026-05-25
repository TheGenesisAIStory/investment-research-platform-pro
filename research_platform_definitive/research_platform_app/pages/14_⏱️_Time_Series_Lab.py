from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_DIR.parent
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in [PROJECT_ROOT, APP_DIR, SRC_ROOT]:
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import pandas as pd
import plotly.express as px
import streamlit as st

from support import (
    configure_page,
    dataframe_with_download,
    render_context_bar,
    render_footer,
    render_metric_metadata_expander,
    render_page_header,
    render_page_intro,
    safe_page_link,
    sidebar_roots,
)

from research_platform_core import (
    TimeSeriesForecastConfig,
    available_time_series_assets,
    load_equity_series,
    load_macro_series,
    load_time_series_forecast_artifacts,
    run_time_series_forecast,
)


configure_page("Time Series Lab")


@st.cache_data(ttl=300)
def _assets(financial_db_root: str, output_root: str) -> pd.DataFrame:
    return available_time_series_assets(financial_db_root, output_root)


@st.cache_data(ttl=120)
def _history(source: str, symbol: str, financial_db_root: str, output_root: str, max_rows: int | None) -> pd.DataFrame:
    if source == "macro":
        return load_macro_series(symbol, financial_db_root, output_root, tail_rows=max_rows)
    return load_equity_series(symbol, financial_db_root, output_root, tail_rows=max_rows)


def _pct(value: object) -> str:
    try:
        if pd.isna(value):
            return "n/a"
        return f"{float(value):+.2%}"
    except Exception:
        return "n/a"


roots = sidebar_roots()

render_page_header(
    "Time Series Lab",
    "Single-series forecasting for indices, macro proxies, FX, commodities, crypto and selected equities. It complements, rather than replaces, the cross-sectional ML Stock Lab.",
    "⏱",
    module="LABS",
    status="WIP",
)
render_context_bar()
render_page_intro(
    "Forecast one series at a time with lag/rolling features, baseline persistence and ML regressors.",
    "Choose a macro/equity series, run forecasts for 5/21/63 day horizons, then use the output as Macro or Portfolio context.",
)

assets = _assets(str(roots["financial_db"]), str(roots["workspace"]))
saved = load_time_series_forecast_artifacts(roots["workspace"])

with st.container(border=True):
    st.markdown("**Forecast setup**")
    st.caption("This Lab is for time-series scenario context. Stock selection remains in the cross-sectional ML Stock Lab.")
    source_col, symbol_col, model_col = st.columns([0.22, 0.38, 0.4])
    source = source_col.radio("Series source", ["macro", "equity"], horizontal=True, help="Macro uses the Macro DB; equity uses local OHLCV parquet files.")

    if source == "macro":
        macro_assets = assets[assets["source_type"].astype(str).eq("macro")].copy() if not assets.empty else pd.DataFrame()
        labels = macro_assets["display_label"].tolist() if not macro_assets.empty else ["SPY · SPDR S&P 500 ETF"]
        default_idx = 0
        selected_label = symbol_col.selectbox("Macro / market series", labels, index=default_idx, help="Global, USA, EU, Italy, FX, commodity, fixed-income and crypto proxies.")
        if not macro_assets.empty:
            symbol = str(macro_assets.loc[macro_assets["display_label"].eq(selected_label), "symbol"].iloc[0])
        else:
            symbol = "SPY"
    else:
        default_ticker = str(st.session_state.get("selected_ticker", "") or "SPY").upper()
        symbol = symbol_col.text_input("Equity ticker", value=default_ticker, help="Uses the same selected_ticker context as Screener, Valuation and Portfolio.").strip().upper()
        if symbol:
            st.session_state["selected_ticker"] = symbol

    horizons = model_col.multiselect("Forecast horizons", [5, 21, 63, 126], default=[5, 21, 63], help="Forward return horizons in trading days.")
    model_options = ["naive", "ols", "gbrt"]
    models = model_col.multiselect("Models", model_options, default=model_options, help="Naive = persistence baseline; OLS/GBRT use lag and rolling features.")

    with st.expander("Advanced options", expanded=False):
        a1, a2, a3 = st.columns(3)
        train_end_year = int(a1.number_input("Train end year", min_value=2000, max_value=2026, value=2018, step=1))
        test_start_year = int(a2.number_input("Test start year", min_value=2000, max_value=2026, value=2019, step=1))
        max_rows_value = int(a3.number_input("Max rows", min_value=0, max_value=100000, value=0, step=250, help="0 means use all rows available."))
        max_rows = max_rows_value or None

    run_cols = st.columns([0.28, 0.24, 0.48])
    run_button = run_cols[0].button("Run time-series forecast", width="stretch", disabled=not symbol or not horizons or not models)
    with run_cols[1]:
        safe_page_link("pages/13_🌍_Macro_View.py", "Open Macro View")
    with run_cols[2]:
        st.caption("Outputs are saved under `output/time_series_lab/tables` and can be used by Macro/Portfolio context panels.")

if run_button:
    config = TimeSeriesForecastConfig(
        source=source,
        symbol=symbol,
        horizons=tuple(int(h) for h in horizons),
        models=tuple(models),
        train_end_year=train_end_year,
        test_start_year=test_start_year,
        max_rows=max_rows,
        write_artifacts=True,
    )
    try:
        with st.spinner(f"Running time-series forecasts for {symbol}..."):
            result = run_time_series_forecast(config, roots["financial_db"], roots["workspace"])
        if result.metrics.empty:
            st.warning(f"No model results were produced for {symbol}. Check data coverage and horizon settings.")
        else:
            st.success(f"Forecast run completed for {symbol}: {len(result.metrics)} model/horizon diagnostics.")
        saved = load_time_series_forecast_artifacts(roots["workspace"])
    except Exception as exc:
        st.error("Time Series Lab failed cleanly. Check data availability or reduce horizons/models, then retry.")
        st.caption(f"Diagnostic: {type(exc).__name__}: {exc}")

history = _history(source, symbol, str(roots["financial_db"]), str(roots["workspace"]), max_rows if "max_rows" in locals() else None)
if not history.empty:
    history = history.copy()
    history["date"] = pd.to_datetime(history["date"], errors="coerce")

latest = saved.get("latest", pd.DataFrame())
metrics = saved.get("metrics", pd.DataFrame())
predictions = saved.get("predictions", pd.DataFrame())
feature_schema = saved.get("feature_schema", pd.DataFrame())
manifest = saved.get("manifest", {}) or {}

tabs = st.tabs(["Series & Forecast", "Validation", "Features", "Design Notes"])

with tabs[0]:
    if history.empty:
        st.warning("No price history found for the selected series. Compile Macro DB or verify OHLCV parquet coverage in Data Platform.")
        safe_page_link("pages/8_🗄️_Data_Platform.py", "Open Data Platform")
    else:
        c1, c2, c3, c4 = st.columns(4)
        last = history.dropna(subset=["close"]).iloc[-1]
        c1.metric("Series", symbol)
        c2.metric("Rows", f"{len(history):,}")
        c3.metric("Last close", f"{float(last['close']):,.2f}")
        c4.metric("Last date", pd.to_datetime(last["date"]).date().isoformat())
        st.plotly_chart(
            px.line(history, x="date", y="close", title=f"{symbol} historical close", template="plotly_white"),
            width="stretch",
        )

    if latest.empty:
        st.info("No saved forecast artifact yet. Run the setup above to populate latest forecasts.")
    else:
        latest_view = latest[latest["symbol"].astype(str).str.upper().eq(symbol.upper())].copy() if "symbol" in latest.columns else latest.copy()
        if latest_view.empty:
            latest_view = latest.copy()
        l1, l2, l3 = st.columns(3)
        best = latest_view.sort_values("forecast_return", ascending=False).head(1) if "forecast_return" in latest_view.columns else pd.DataFrame()
        l1.metric("Saved run symbol", str(manifest.get("symbol", latest_view.get("symbol", pd.Series(["n/a"])).iloc[0] if not latest_view.empty else "n/a")))
        l2.metric("Best forecast", _pct(best["forecast_return"].iloc[0]) if not best.empty else "n/a")
        l3.metric("Updated", str(manifest.get("updated_at", "n/a"))[:19])
        if "forecast_return" in latest_view.columns:
            st.plotly_chart(
                px.bar(
                    latest_view,
                    x="horizon_days",
                    y="forecast_return",
                    color="model",
                    barmode="group",
                    title="Latest forecasted forward returns",
                    template="plotly_white",
                ),
                width="stretch",
            )
        dataframe_with_download("Latest forecasts", latest_view, "TimeSeriesForecast_latest.csv")

with tabs[1]:
    if metrics.empty:
        st.info("No validation metrics yet. Run a forecast first.")
    else:
        metric_view = metrics[metrics["symbol"].astype(str).str.upper().eq(symbol.upper())].copy() if "symbol" in metrics.columns else metrics.copy()
        if metric_view.empty:
            metric_view = metrics.copy()
        st.caption("MAE/RMSE/MAPE evaluate return forecast error. Directional accuracy checks sign correctness.")
        dataframe_with_download("Time-series validation metrics", metric_view, "TimeSeriesForecast_metrics.csv")
        render_metric_metadata_expander(["mae", "rmse", "mape", "directional_accuracy", "forecast_return", "forecast_level"])
        if not predictions.empty:
            pred_view = predictions[predictions["symbol"].astype(str).str.upper().eq(symbol.upper())].copy() if "symbol" in predictions.columns else predictions.copy()
            if not pred_view.empty and {"model", "horizon_days"}.issubset(pred_view.columns):
                p1, p2 = st.columns(2)
                model_choice = p1.selectbox("Prediction model", sorted(pred_view["model"].dropna().astype(str).unique().tolist()))
                horizon_choice = p2.selectbox("Prediction horizon", sorted(pred_view["horizon_days"].dropna().astype(int).unique().tolist()))
                chart = pred_view[(pred_view["model"].astype(str).eq(model_choice)) & (pred_view["horizon_days"].astype(int).eq(int(horizon_choice)))].copy()
                chart["date"] = pd.to_datetime(chart["date"], errors="coerce")
                if not chart.empty:
                    melted = chart[["date", "realized_forward_return", "predicted_forward_return"]].melt("date", var_name="series", value_name="return")
                    st.plotly_chart(px.line(melted, x="date", y="return", color="series", title="Forecast vs realized forward return", template="plotly_white"), width="stretch")

with tabs[2]:
    if feature_schema.empty:
        st.info("Feature schema will appear after the first forecast run.")
    else:
        feature_view = feature_schema[feature_schema["symbol"].astype(str).str.upper().eq(symbol.upper())].copy() if "symbol" in feature_schema.columns else feature_schema.copy()
        if feature_view.empty:
            feature_view = feature_schema.copy()
        dataframe_with_download("Lag and rolling feature schema", feature_view, "TimeSeriesForecast_feature_schema.csv")
    st.markdown(
        """
        <div class="rp-note">
        Leakage policy: every feature is lagged or rolling from information available at the forecast date.
        The target is the future return over the selected horizon and is never used as an input feature.
        </div>
        """,
        unsafe_allow_html=True,
    )

with tabs[3]:
    st.markdown("### What this adds")
    st.markdown(
        """
        - **Already covered elsewhere:** ML Stock Lab predicts cross-sectional forward returns for 21/63/252 day horizons and validates ranking quality with IC, RankIC and long-short Sharpe.
        - **Added here:** Time Series Lab forecasts one asset/index/macro series at a time using baseline persistence and feature-based OLS/GBRT models.
        - **Still planned:** ARIMA/ETS baselines, forecast intervals, multivariate macro regressors and optional deep sequence models.
        """
    )
    with st.expander("Recommended usage", expanded=False):
        st.markdown(
            """
            Use this module as scenario context for Macro View and Portfolio diagnostics.
            It should not override the cross-sectional ranking layer or be interpreted as a direct trading signal.
            """
        )
    safe_page_link("pages/9_ML_Stock_Lab.py", "Open ML Stock Lab")

render_footer()
