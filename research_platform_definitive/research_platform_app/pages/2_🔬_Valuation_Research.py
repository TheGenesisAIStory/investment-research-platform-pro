from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
PROJECT_ROOT = APP_DIR.parent
for extra in [PROJECT_ROOT, PROJECT_ROOT / "company_valuation" / "src"]:
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

from data_bootstrap import render_bootstrap_banner
from screener_workbench import format_valuation_explanation
from support import configure_page, dataframe_with_download, load_company_artifacts, load_smart_money_artifacts, load_ml_stock_lab_artifacts, metric_value, numeric_cols, render_context_bar, render_footer, render_page_intro, render_selected_ticker_context, render_workflow_steps, safe_page_link, show_empty, sidebar_roots
from ui_ops import render_missing_data_cta

try:
    from valuation_monte_carlo import (
        DCFMonteCarloConfig,
        build_dcf_input_from_frame,
        build_eva_input_from_frame,
        build_residual_income_input_from_frame,
        infer_region,
        market_parameters_frame,
        run_dcf_monte_carlo,
        run_eva_scenario_bands,
        run_residual_income_monte_carlo,
        write_dcf_monte_carlo_outputs,
        write_eva_scenario_outputs,
        write_residual_income_monte_carlo_outputs,
    )
except Exception:
    DCFMonteCarloConfig = None
    build_dcf_input_from_frame = build_eva_input_from_frame = build_residual_income_input_from_frame = None
    infer_region = market_parameters_frame = run_dcf_monte_carlo = run_eva_scenario_bands = run_residual_income_monte_carlo = None
    write_dcf_monte_carlo_outputs = write_eva_scenario_outputs = write_residual_income_monte_carlo_outputs = None


configure_page("Valuation Research")


@st.cache_data(ttl=300, show_spinner=False)
def fetch_live_valuation_snapshot(ticker: str) -> dict[str, object]:
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception as exc:
        return {"ticker": ticker.upper(), "error": str(exc)}
    return {
        "ticker": ticker.upper(),
        "shortName": info.get("shortName") or info.get("longName"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "marketCap": info.get("marketCap"),
        "trailingPE": info.get("trailingPE"),
        "enterpriseToEbitda": info.get("enterpriseToEbitda"),
        "priceToBook": info.get("priceToBook"),
        "currency": info.get("currency"),
    }


def format_live_value(value: object) -> str:
    if value is None or value == "":
        return "n/a"
    if isinstance(value, (int, float)):
        if abs(float(value)) >= 1_000_000:
            return f"{float(value) / 1_000_000_000:.2f}B"
        return f"{float(value):.2f}"
    return str(value)

roots = sidebar_roots()
root = roots["company"]
data = load_company_artifacts(root)
smart_money = load_smart_money_artifacts(roots["workspace"])
ml_lab = load_ml_stock_lab_artifacts(roots["workspace"])

st.title("Equity Valuation Research")
st.caption("Notebook-generated fair value, ranking, screener context, diagnostics and model evidence.")
render_context_bar()
render_page_intro(
    "Open a ticker-level valuation workspace with live multiples, exported DCF/EVA/RI artifacts and explainable assumptions.",
    "Select a ticker or arrive from Screener to review Snapshot, Assumptions and Explain-this-valuation tabs.",
)
render_bootstrap_banner(roots, required=["equity_metadata", "company_screener"])

context_ticker = str(st.session_state.get("selected_ticker", "") or "").strip().upper()

with st.container(border=True):
    st.markdown("**Live valuation snapshot · yfinance cached 5 minutes**")
    live_ticker = st.text_input("Ticker", value=context_ticker or "AAPL", key="live_valuation_ticker").strip().upper() or "AAPL"
    st.session_state["selected_ticker"] = live_ticker
    live_snapshot = fetch_live_valuation_snapshot(live_ticker)
    if live_snapshot.get("error"):
        st.warning(f"Live valuation data is temporarily unavailable for {live_ticker}: {live_snapshot['error']}")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("P/E", format_live_value(live_snapshot.get("trailingPE")))
        c2.metric("EV/EBITDA", format_live_value(live_snapshot.get("enterpriseToEbitda")))
        c3.metric("Price / Book", format_live_value(live_snapshot.get("priceToBook")))
        c4.metric("Market Cap", format_live_value(live_snapshot.get("marketCap")))
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "ticker": live_snapshot.get("ticker"),
                        "name": live_snapshot.get("shortName"),
                        "sector": live_snapshot.get("sector"),
                        "industry": live_snapshot.get("industry"),
                        "currency": live_snapshot.get("currency"),
                        "trailing_pe": live_snapshot.get("trailingPE"),
                        "ev_to_ebitda": live_snapshot.get("enterpriseToEbitda"),
                        "price_to_book": live_snapshot.get("priceToBook"),
                    }
                ]
            ),
            width="stretch",
            hide_index=True,
        )

render_selected_ticker_context(roots, st.session_state.get("selected_ticker", live_ticker), expanded=False)

valuation_gap = data["valuation_gap"]
extended = data["extended_valuation"]
screener = data["screener_results"]
summary = data["screener_summary"]
smart_scores = smart_money["scores"]
smart_events = smart_money["events"]
dcf_summary = data["dcf_mc_summary"]
dcf_simulations = data["dcf_mc_simulations"]
dcf_scenarios = data["dcf_mc_scenarios"]
dcf_assumptions = data["dcf_mc_assumptions"]
dcf_factors = data["dcf_model_factors"]
dcf_market_parameters = data["dcf_market_parameters"]
ri_summary = data["ri_mc_summary"]
ri_simulations = data["ri_mc_simulations"]
ri_scenarios = data["ri_mc_scenarios"]
ri_assumptions = data["ri_mc_assumptions"]
ri_factors = data["ri_model_factors"]
eva_bands = data["eva_scenario_bands"]
eva_summary = data["eva_scenario_summary"]
eva_assumptions = data["eva_scenario_assumptions"]
eva_factors = data["eva_model_factors"]

tickers: list[str] = []
for frame in [valuation_gap, extended, screener]:
    if not frame.empty and "ticker" in frame.columns:
        tickers.extend(frame["ticker"].dropna().astype(str).unique().tolist())
if not smart_scores.empty and "ticker" in smart_scores.columns:
    tickers.extend(smart_scores["ticker"].dropna().astype(str).unique().tolist())
tickers = sorted(set(tickers))

control = st.container(border=True)
with control:
    c1, c2, c3 = st.columns(3)
    default_index = ["All", *tickers].index(context_ticker) if context_ticker in tickers else 0
    selected = c1.selectbox("Ticker", ["All", *tickers], index=default_index) if tickers else "All"
    if selected != "All":
        st.session_state["selected_ticker"] = selected
    universe_options = ["All"]
    for col in ["index_membership", "index", "country", "sector"]:
        if col in screener.columns:
            universe_options += sorted(screener[col].dropna().astype(str).unique().tolist())
            break
    universe = c2.selectbox("Universe / Segment", sorted(set(universe_options)))
    profile = c3.selectbox("Research Profile", ["Balanced", "Value", "Growth", "Quality", "Dividend"])


def filter_ticker(df: pd.DataFrame) -> pd.DataFrame:
    if selected == "All" or df.empty or "ticker" not in df.columns:
        return df
    return df[df["ticker"].astype(str).eq(selected)]


def filter_universe(df: pd.DataFrame) -> pd.DataFrame:
    if universe == "All" or df.empty:
        return df
    for col in ["index_membership", "index", "country", "sector"]:
        if col in df.columns:
            return df[df[col].astype(str).eq(universe)]
    return df


valuation_view = filter_ticker(filter_universe(valuation_gap if not valuation_gap.empty else extended))
screener_view = filter_ticker(filter_universe(screener))
smart_view = filter_ticker(smart_scores)
smart_event_view = filter_ticker(smart_events)
ml_signal_view = filter_ticker(ml_lab["signals"])

cols = st.columns(4)
cols[0].metric("Ticker", selected if selected != "All" else "Universe")
cols[1].metric("Fair Value", metric_value(valuation_view, ["blended_fair_value", "fair_value", "target_price", "value"], "n/a"))
cols[2].metric("Upside", metric_value(valuation_view, ["upside", "upside_to_fair_value", "valuation_gap"], "n/a"))
cols[3].metric("Screener Matches", len(screener_view) if not screener_view.empty else 0)

if valuation_view.empty:
    render_missing_data_cta(
        "Valuation",
        job_id="valuation_research_refresh",
        output_path=root / "tables" / "ScreenerResults.csv",
        cli_hint="python research_platform_app/scheduler.py --once --jobs valuation_research_refresh",
    )

status_box = st.container(border=True)
with status_box:
    st.markdown("**Data status**")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Coverage Rows", len(screener_view) if not screener_view.empty else len(valuation_view))
    s2.metric("Fundamental Age", metric_value(screener_view, ["dayssincefundamental", "days_since_fundamental"], "n/a"))
    s3.metric("Robustness", metric_value(screener_view, ["robustness_status", "data_quality_status"], "n/a"))
    s4.metric("Smart Money", metric_value(smart_view, ["composite_institutional_interest_score"], "n/a"))

st.markdown(
    """
    <div class="rp-note">
    Valuation ranking is thesis support, not a price target guarantee. Pair fair value with coverage,
    staleness, model assumptions, robustness and screener audit before relying on a shortlist.
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("Explain this valuation", expanded=False):
    explain_ticker = selected if selected != "All" else st.session_state.get("selected_ticker", live_ticker)
    st.markdown(f"**{explain_ticker} · valuation reasoning pre-read**")
    explanation_source = valuation_view.iloc[0] if not valuation_view.empty else pd.Series(live_snapshot)
    st.write(format_valuation_explanation(str(explain_ticker), explanation_source))
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("P/E", format_live_value(live_snapshot.get("trailingPE")) if not live_snapshot.get("error") else "n/a")
    e2.metric("EV/EBITDA", format_live_value(live_snapshot.get("enterpriseToEbitda")) if not live_snapshot.get("error") else "n/a")
    e3.metric("P/B", format_live_value(live_snapshot.get("priceToBook")) if not live_snapshot.get("error") else "n/a")
    e4.metric("Artifact Upside", metric_value(valuation_view, ["upside", "upside_to_fair_value", "valuation_gap"], "n/a"))
    assumptions_view = dcf_assumptions if not dcf_assumptions.empty else data["valuation_assumptions"]
    if assumptions_view.empty:
        st.info("DCF/WACC/growth assumption tables are not available for this ticker yet.")
    else:
        dataframe_with_download("Valuation assumptions", assumptions_view, "valuation_assumptions_for_selected_ticker.csv")

tab_overview, tab_models, tab_dcf_mc, tab_screener, tab_smart_money, tab_ml_lab, tab_diagnostics = st.tabs(["Overview", "Models", "DCF Monte Carlo", "Screener Context", "Smart Money", "ML Lab", "Diagnostics & Caveats"])

with tab_overview:
    if valuation_view.empty:
        show_empty("Valuation summary", root)
    else:
        dataframe_with_download("Valuation summary", valuation_view, "valuation_summary.csv")
    if not valuation_view.empty and len(numeric_cols(valuation_view)) >= 1:
        value_col = st.selectbox("Chart metric", numeric_cols(valuation_view), key="valuation_metric")
        label_col = "ticker" if "ticker" in valuation_view.columns else valuation_view.index
        st.plotly_chart(px.bar(valuation_view.head(40), x=label_col, y=value_col, title=f"{value_col} by ticker", template="plotly_white"), width="stretch")

with tab_models:
    dataframe_with_download("Extended valuation results", filter_ticker(extended), "extended_valuation_results.csv")
    dataframe_with_download("Valuation model registry", data["valuation_models"], "valuation_model_registry.csv")
    dataframe_with_download("Valuation assumptions", data["valuation_assumptions"], "valuation_assumptions.csv")
    with st.expander("Model interpretation", expanded=True):
        st.markdown("DCF, residual income, multiples and scenario outputs should be compared as a range. Wide dispersion is a model-risk signal, not just noise.")

with tab_dcf_mc:
    render_workflow_steps(
        [
            ("Company", "Use valuation artifacts or selected ticker"),
            ("Assumptions", "Region WACC, growth and terminal-growth bands"),
            ("Simulation", "Monte Carlo on growth / WACC / terminal growth"),
            ("Results", "Distribution, scenarios and model-based factors"),
        ],
        active_index=3 if not dcf_summary.empty else 1,
    )
    st.markdown(
        """
        <div class="rp-note">
        This is a reusable uncertainty layer around DCF. It keeps Database Finanziario and notebook artifacts as the source of truth;
        the app only runs a light simulation when requested.
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    region_options = market_parameters_frame()["region"].tolist() if market_parameters_frame else ["US", "IT", "DE", "UK", "JP", "HK", "India", "China"]
    default_region = infer_region(selected if selected != "All" else None) if infer_region else "US"
    region = c1.selectbox("Market region", region_options, index=region_options.index(default_region) if default_region in region_options else 0)
    simulations = c2.number_input("Simulations", min_value=500, max_value=50000, value=5000, step=500)
    horizon_years = c3.slider("DCF horizon", min_value=3, max_value=15, value=5)
    beta = c4.slider("Beta", min_value=0.3, max_value=2.5, value=1.0, step=0.05)

    run_now = st.button("Run light DCF Monte Carlo", width="stretch", disabled=run_dcf_monte_carlo is None)
    if run_now:
        source = valuation_view if not valuation_view.empty else screener_view
        dcf_input = build_dcf_input_from_frame(source, ticker=None if selected == "All" else selected) if build_dcf_input_from_frame else None
        if dcf_input is None:
            st.warning("Could not infer price and FCF/revenue fields from current artifacts. Run the valuation notebook first or select a richer ticker row.")
        else:
            cfg = DCFMonteCarloConfig(region=region, simulations=int(simulations), horizon_years=int(horizon_years), beta=float(beta))
            outputs = run_dcf_monte_carlo(dcf_input, cfg)
            paths = write_dcf_monte_carlo_outputs(outputs, root / "tables")
            st.success("DCF Monte Carlo artifacts written.")
            st.json(paths)
            dcf_summary = outputs["summary"]
            dcf_simulations = outputs["simulations"]
            dcf_scenarios = outputs["scenarios"]
            dcf_assumptions = outputs["assumptions"]
            dcf_factors = outputs["factors"]
            dcf_market_parameters = outputs["market_parameters"]

    if dcf_summary.empty:
        st.info("No DCF Monte Carlo artifacts yet. Use the button above or run the DCF Monte Carlo Lab cell in the valuation notebook.")
    else:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Median FV", metric_value(dcf_summary, ["fair_value_median"], "n/a"))
        k2.metric("Median Upside", metric_value(dcf_summary, ["upside_median"], "n/a"))
        k3.metric("P(Undervalued)", metric_value(dcf_summary, ["probability_undervalued"], "n/a"))
        k4.metric("Base WACC", metric_value(dcf_summary, ["base_wacc"], "n/a"))
        dataframe_with_download("DCF Monte Carlo summary", dcf_summary, "DCFMonteCarloSummary.csv")

    if not dcf_simulations.empty and "fair_value" in dcf_simulations.columns:
        st.plotly_chart(
            px.histogram(dcf_simulations, x="fair_value", nbins=60, title="DCF fair value distribution", template="plotly_white", color_discrete_sequence=["#01696f"]),
            width="stretch",
        )
        if {"wacc", "growth", "fair_value"}.issubset(dcf_simulations.columns):
            st.plotly_chart(
                px.scatter(dcf_simulations.sample(min(len(dcf_simulations), 1500), random_state=7), x="wacc", y="growth", color="fair_value", title="Simulation parameter cloud", template="plotly_white", color_continuous_scale="RdYlGn"),
                width="stretch",
            )
    if not dcf_scenarios.empty:
        dataframe_with_download("DCF scenarios", dcf_scenarios, "DCFMonteCarloScenarios.csv")
        if {"scenario", "fair_value"}.issubset(dcf_scenarios.columns):
            st.plotly_chart(px.bar(dcf_scenarios, x="scenario", y="fair_value", color="scenario", title="Bear / Base / Bull fair value", template="plotly_white"), width="stretch")
    with st.expander("Market parameters and model-based factors", expanded=False):
        dataframe_with_download("DCF market parameters", dcf_market_parameters, "DCFMarketParameters.csv")
        dataframe_with_download("DCF assumptions", dcf_assumptions, "DCFMonteCarloAssumptions.csv")
        dataframe_with_download("DCF model-based factors", dcf_factors, "DCFModelBasedFactors.csv")

    st.divider()
    ri_tab, eva_tab, growth_tab, factor_tab = st.tabs(["Residual Income MC", "EVA Bands", "Historical Growth", "Model Factors"])
    source_for_models = valuation_view if not valuation_view.empty else screener_view

    with ri_tab:
        st.markdown("Residual Income Monte Carlo uses book value, ROE and cost of equity when those fields exist in the valuation artifacts.")
        run_ri = st.button("Run Residual Income Monte Carlo", width="stretch", disabled=run_residual_income_monte_carlo is None)
        if run_ri:
            ri_input = build_residual_income_input_from_frame(source_for_models, ticker=None if selected == "All" else selected, region=region) if build_residual_income_input_from_frame else None
            if ri_input is None:
                st.warning("Residual Income needs price, book value per share or equity/shares, and ROE/net income. Current artifacts do not expose enough data.")
            else:
                cfg = DCFMonteCarloConfig(region=region, simulations=int(simulations), horizon_years=int(horizon_years), beta=float(beta))
                outputs = run_residual_income_monte_carlo(ri_input, cfg)
                paths = write_residual_income_monte_carlo_outputs(outputs, root / "tables")
                st.success("Residual Income Monte Carlo artifacts written.")
                st.json(paths)
                ri_summary = outputs["summary"]
                ri_simulations = outputs["simulations"]
                ri_scenarios = outputs["scenarios"]
                ri_assumptions = outputs["assumptions"]
                ri_factors = outputs["factors"]
        if ri_summary.empty:
            st.info("No Residual Income Monte Carlo artifacts yet.")
        else:
            r1, r2, r3 = st.columns(3)
            r1.metric("RI Median FV", metric_value(ri_summary, ["fair_value_median"], "n/a"))
            r2.metric("RI Median Upside", metric_value(ri_summary, ["upside_median"], "n/a"))
            r3.metric("RI P(Undervalued)", metric_value(ri_summary, ["probability_undervalued"], "n/a"))
            dataframe_with_download("Residual Income Monte Carlo summary", ri_summary, "ResidualIncomeMonteCarloSummary.csv")
        if not ri_simulations.empty and "fair_value" in ri_simulations.columns:
            st.plotly_chart(px.histogram(ri_simulations, x="fair_value", nbins=60, title="Residual Income fair value distribution", template="plotly_white"), width="stretch")
        if not ri_scenarios.empty and {"scenario", "fair_value"}.issubset(ri_scenarios.columns):
            st.plotly_chart(px.bar(ri_scenarios, x="scenario", y="fair_value", color="scenario", title="Residual Income scenario values", template="plotly_white"), width="stretch")

    with eva_tab:
        st.markdown("EVA bands show valuation sensitivity to NOPAT growth, invested capital and WACC. Missing NOPAT/capital is left explicit.")
        run_eva = st.button("Run EVA scenario bands", width="stretch", disabled=run_eva_scenario_bands is None)
        if run_eva:
            eva_input = build_eva_input_from_frame(source_for_models, ticker=None if selected == "All" else selected, region=region) if build_eva_input_from_frame else None
            if eva_input is None:
                st.warning("EVA needs price, NOPAT or EBIT, invested capital or equity/debt/cash, shares and WACC/region assumptions.")
            else:
                cfg = DCFMonteCarloConfig(region=region, simulations=int(simulations), horizon_years=int(horizon_years), beta=float(beta))
                outputs = run_eva_scenario_bands(eva_input, cfg)
                paths = write_eva_scenario_outputs(outputs, root / "tables")
                st.success("EVA scenario artifacts written.")
                st.json(paths)
                eva_bands = outputs["bands"]
                eva_summary = outputs["summary"]
                eva_assumptions = outputs["assumptions"]
                eva_factors = outputs["factors"]
        if eva_summary.empty:
            st.info("No EVA scenario artifacts yet.")
        else:
            e1, e2, e3 = st.columns(3)
            e1.metric("EVA Base FV", metric_value(eva_summary, ["fair_value_base"], "n/a"))
            e2.metric("EVA Bull FV", metric_value(eva_summary, ["fair_value_bull"], "n/a"))
            e3.metric("EVA Base Upside", metric_value(eva_summary, ["upside_base"], "n/a"))
            dataframe_with_download("EVA scenario summary", eva_summary, "EVAScenarioSummary.csv")
        if not eva_bands.empty and {"scenario", "fair_value"}.issubset(eva_bands.columns):
            st.plotly_chart(px.bar(eva_bands, x="scenario", y="fair_value", color="scenario", title="EVA fair value bands", template="plotly_white"), width="stretch")
            dataframe_with_download("EVA scenario bands", eva_bands, "EVAScenarioBands.csv")

    with growth_tab:
        hist = source_for_models.copy()
        revenue_col = next((c for c in ["revenue", "sales", "total_revenue"] if c in hist.columns), None)
        fcf_col = next((c for c in ["free_cash_flow", "fcf", "free_cashflow", "operating_cash_flow"] if c in hist.columns), None)
        date_col = next((c for c in ["date", "fiscal_year", "year", "report_date", "effectivefundamentaldate"] if c in hist.columns), None)
        if hist.empty or date_col is None or (revenue_col is None and fcf_col is None):
            st.info("Historical revenue/FCF growth chart appears when Database Finanziario artifacts expose multi-year revenue or FCF by ticker.")
        else:
            chart_cols = [c for c in [revenue_col, fcf_col] if c]
            view = hist[["ticker", date_col, *chart_cols]].dropna(how="all", subset=chart_cols).copy()
            view = view.sort_values(date_col).tail(40)
            long = view.melt(id_vars=["ticker", date_col], value_vars=chart_cols, var_name="metric", value_name="value")
            st.plotly_chart(px.line(long, x=date_col, y="value", color="metric", markers=True, title="Historical revenue / FCF path", template="plotly_white"), width="stretch")
            dataframe_with_download("Historical growth source rows", view, "historical_growth_source.csv")

    with factor_tab:
        combined = pd.concat([df for df in [dcf_factors, ri_factors, eva_factors] if not df.empty], ignore_index=True) if any(not df.empty for df in [dcf_factors, ri_factors, eva_factors]) else pd.DataFrame()
        dataframe_with_download("DCF model-based factors", dcf_factors, "DCFModelBasedFactors.csv")
        dataframe_with_download("Residual Income model-based factors", ri_factors, "ResidualIncomeModelBasedFactors.csv")
        dataframe_with_download("EVA model-based factors", eva_factors, "EVAModelBasedFactors.csv")
        if not combined.empty:
            (root / "tables").mkdir(parents=True, exist_ok=True)
            combined.to_csv(root / "tables" / "ModelBasedFactors.csv", index=False)
            dataframe_with_download("Combined model-based factors", combined, "ModelBasedFactors.csv")

with tab_screener:
    dataframe_with_download("Screener summary", summary, "screener_summary.csv")
    dataframe_with_download("Screener results", screener_view, "screener_results.csv")
    if not screener_view.empty and {"valuation_score", "quality_score"}.issubset(screener_view.columns):
        color_col = "screener_score" if "screener_score" in screener_view.columns else None
        st.plotly_chart(px.scatter(screener_view, x="valuation_score", y="quality_score", color=color_col, hover_name="ticker" if "ticker" in screener_view.columns else None, title="Valuation vs Quality", template="plotly_white"), width="stretch")

with tab_smart_money:
    st.markdown(
        """
        Smart Money is an official-source overlay. It should confirm or challenge valuation theses,
        not replace DCF, fundamentals, governance diagnostics or model assumptions.
        """
    )
    if smart_view.empty:
        st.info("Run the Smart Money Government Data Refresh job to compute ownership, insider, activism, macro-flow and public-spending overlays.")
        safe_page_link("pages/6_🧪_Notebook_Runner.py", "Open Notebook Runner")
    else:
        dataframe_with_download("Smart Money issuer overlay", smart_view, "valuation_smart_money_overlay.csv")
        if "composite_institutional_interest_score" in smart_view.columns:
            x_col = "ticker" if "ticker" in smart_view.columns else "issuer_name"
            st.plotly_chart(px.bar(smart_view.head(30), x=x_col, y="composite_institutional_interest_score", color="coverage_note" if "coverage_note" in smart_view.columns else None, title="Smart Money Composite Overlay", template="plotly_white"), width="stretch")
    with st.expander("Issuer event feed", expanded=False):
        dataframe_with_download("Smart Money event feed", smart_event_view, "valuation_smart_money_events.csv")

with tab_ml_lab:
    st.markdown("ML Stock Lab provides model-implied fair value and mispricing signals that can be compared with the valuation engine.")
    if ml_signal_view.empty:
        st.info("Run ML Stock Lab refresh to compute fair value ML, mispricing and quintile artifacts.")
        safe_page_link("pages/9_ML_Stock_Lab.py", "Open ML Stock Lab")
    else:
        dataframe_with_download("ML Stock Lab valuation overlay", ml_signal_view, "valuation_ml_stock_lab_overlay.csv")
        if {"market_value", "fair_value_hat"}.issubset(ml_signal_view.columns):
            st.plotly_chart(px.scatter(ml_signal_view, x="market_value", y="fair_value_hat", hover_name="ticker" if "ticker" in ml_signal_view.columns else None, title="ML Fair Value vs Market Value", template="plotly_white"), width="stretch")
        if "zscore" in ml_signal_view.columns:
            st.plotly_chart(px.histogram(ml_signal_view, x="zscore", nbins=30, title="ML Mispricing z-score", template="plotly_white"), width="stretch")

with tab_diagnostics:
    dataframe_with_download("Refresh provenance", data["refresh_provenance"], "refresh_provenance.csv")
    dataframe_with_download("Provider registry", data["provider_registry"], "provider_registry.csv")
    dataframe_with_download("ML leaderboard", data["ml_leaderboard"], "ml_leaderboard.csv")
    dataframe_with_download("Smart Money coverage", smart_money["coverage"], "smart_money_coverage.csv")
    dataframe_with_download("ML Stock Lab metrics", ml_lab["metrics"], "ml_stock_lab_metrics.csv")
    with st.expander("Caveats", expanded=True):
        st.markdown(
            """
            - Missing provider data is surfaced through diagnostics instead of filled silently.
            - A high valuation score can still be a value trap if quality, leverage or freshness are weak.
            - ML outputs are overlays and should not override valuation assumptions without review.
            """
        )

render_footer()
