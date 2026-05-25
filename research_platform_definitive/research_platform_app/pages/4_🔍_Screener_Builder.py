from __future__ import annotations

import sys
import time
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import pandas as pd
import plotly.express as px
import streamlit as st

from app_settings import load_platform_settings, save_platform_settings
from operations import generate_company_artifacts, generate_portfolio_artifacts
from data_bootstrap import render_bootstrap_banner
from screener_workbench import (
    PRESETS,
    apply_screening_filters,
    build_screening_frame,
    delete_screener,
    display_columns,
    explain_ml_signal,
    explain_smart_money,
    explain_valuation,
    list_saved_screeners,
    load_screener_config,
    numeric_range,
    save_screener_config,
    screening_zero_result_suggestions,
)
from support import (
    configure_page,
    dataframe_with_download,
    load_company_artifacts,
    load_ml_stock_lab_artifacts,
    load_portfolio_artifacts,
    load_smart_money_artifacts,
    render_context_bar,
    render_feature_metadata_expander,
    render_footer,
    render_metric_metadata_expander,
    render_page_header,
    render_page_intro,
    render_selected_ticker_context,
    safe_page_link,
    sidebar_roots,
)
from ui_ops import render_missing_data_cta

from research_platform_core.data_health import get_data_status_for_tickers, get_stage_health_for_universes
from research_platform_core.llm_advisors import explain_stock_picks, suggest_screener_config
from research_platform_core import compute_correlation_matrix, summarize_correlation_matrix


configure_page("Screener Builder")


def state_default(key: str, value):
    if key not in st.session_state:
        st.session_state[key] = value
    return st.session_state[key]


def unique_options(frame: pd.DataFrame, column: str) -> list[str]:
    if frame.empty or column not in frame.columns:
        return []
    values = frame[column].dropna().astype(str)
    values = values[values.str.len() > 0]
    return sorted(values.unique().tolist())


def metric_or_na(value) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{float(value):.2f}"
    return str(value)


def clamp(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


@st.cache_data(ttl=300)
def _cached_universe_health(financial_db_root: str, output_root: str, selected_universes: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for stage in ["equity_fundamentals", "equity_prices"]:
        frame = get_stage_health_for_universes(stage, list(selected_universes), financial_db_root, output_root)
        rows.append(frame)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


@st.cache_data(ttl=300)
def _cached_ticker_status(financial_db_root: str, output_root: str, tickers: tuple[str, ...]) -> pd.DataFrame:
    statuses = get_data_status_for_tickers(list(tickers), financial_db_root, output_root)
    return pd.DataFrame([status.to_dict() for status in statuses.values()])


def _status_message(status: str, label: str) -> None:
    value = str(status or "UNKNOWN").upper()
    text = f"{label}: {value}"
    if value in {"OK"}:
        st.success(text)
    elif value in {"PARTIAL", "RUNNING", "MISSING", "UNKNOWN"}:
        st.warning(text)
    else:
        st.error(text)


def _worst_status(values: list[str]) -> str:
    priority = {"FAILED": 4, "MISSING": 3, "PARTIAL": 2, "RUNNING": 2, "UNKNOWN": 1, "OK": 0}
    clean = [str(value or "UNKNOWN").upper() for value in values]
    return max(clean, key=lambda value: priority.get(value, 1)) if clean else "UNKNOWN"


state_default("selected_ticker", "")
state_default("active_screener_config", {})
state_default("active_screener_name", "Custom")
state_default("preferred_screening_columns", [])

roots = sidebar_roots()
platform_settings = load_platform_settings(roots["workspace"])
if not st.session_state.get("preferred_screening_columns") and platform_settings.get("screener", {}).get("default_columns"):
    st.session_state["preferred_screening_columns"] = platform_settings["screener"]["default_columns"]
company = load_company_artifacts(roots["company"])
portfolio = load_portfolio_artifacts(roots["portfolio"])
smart_money = load_smart_money_artifacts(roots["workspace"])
ml_lab = load_ml_stock_lab_artifacts(roots["workspace"])

render_page_header(
    "Screener",
    "Buy-side idea generation layer integrated with fundamentals, factor scores, ML Stock Lab, valuation, portfolio and Smart Money artifacts.",
    "⌕",
    module="RESEARCH",
    status="READY",
)
render_context_bar()
render_page_intro(
    "Build reusable equity idea screens from fundamentals, ML signals, Smart Money events and valuation artifacts.",
    "Start with a template, apply filters, then open selected names in Valuation, Portfolio, ML or Smart Money.",
)

unified = build_screening_frame(company, portfolio, smart_money, ml_lab, roots["financial_db"])
saved_screeners = list_saved_screeners(roots["workspace"], unified)
render_bootstrap_banner(roots, required=["equity_metadata", "company_screener", "ml_signals", "smart_money_scores"])

with st.expander("Operational refresh", expanded=False):
    st.markdown("Rebuild lightweight screener/selection artifacts from already exported notebook tables.")
    left, right = st.columns(2)
    if left.button("Regenerate Company Screener Layer", width="stretch"):
        started = time.perf_counter()
        try:
            result = generate_company_artifacts(roots["company"])
            elapsed = time.perf_counter() - started
            (st.success if result["ok"] else st.warning)(f"{result['message']} Completed in {elapsed:.1f}s.")
            if result.get("details"):
                st.json(result["details"])
            st.rerun()
        except Exception as exc:
            st.error(f"Company layer refresh failed cleanly: {type(exc).__name__}. Use Notebook Runner for the full job log.")
    if right.button("Regenerate Portfolio Selection Layer", width="stretch"):
        started = time.perf_counter()
        try:
            result = generate_portfolio_artifacts(roots["portfolio"], roots["workspace"])
            elapsed = time.perf_counter() - started
            (st.success if result["ok"] else st.warning)(f"{result['message']} Completed in {elapsed:.1f}s.")
            if result.get("details"):
                st.json(result["details"])
            st.rerun()
        except Exception as exc:
            st.error(f"Portfolio layer refresh failed cleanly: {type(exc).__name__}. Use Notebook Runner for the full job log.")

if unified.empty:
    render_missing_data_cta(
        "Screening universe",
        job_id="screener_refresh",
        output_path=roots["company"] / "tables" / "ScreenerResults.csv",
        cli_hint="python research_platform_app/scheduler.py --once --jobs screener_refresh",
    )
    render_footer()
    st.stop()

left_panel, right_panel = st.columns([0.32, 0.68], gap="large")

with left_panel:
    st.subheader("Builder")

    if not saved_screeners.empty:
        saved_labels = [
            f"{row['name']} · {row.get('current_count', 'n/a')} names"
            for _, row in saved_screeners.iterrows()
        ]
        selected_saved = st.selectbox("My / Team screeners", ["Select saved screener", *saved_labels])
        if selected_saved != "Select saved screener":
            idx = saved_labels.index(selected_saved)
            selected_path = saved_screeners.iloc[idx]["path"]
            b1, b2 = st.columns(2)
            if b1.button("Load", width="stretch"):
                payload = load_screener_config(Path(selected_path))
                st.session_state["active_screener_config"] = payload.get("config", {})
                st.session_state["active_screener_name"] = payload.get("name", "Loaded screener")
                st.rerun()
            if b2.button("Delete", width="stretch"):
                delete_screener(selected_path)
                st.success("Screener config deleted.")
                st.rerun()
        with st.expander("Saved screener metadata", expanded=False):
            st.dataframe(saved_screeners, width="stretch", hide_index=True)
    else:
        st.info("No saved screeners yet. Configure filters and save the first one.")

    preset_name = st.selectbox("Screener template", list(PRESETS), index=list(PRESETS).index(st.session_state.get("active_screener_name", "Custom")) if st.session_state.get("active_screener_name") in PRESETS else 0)
    if st.button("Apply template", width="stretch"):
        st.session_state["active_screener_config"] = PRESETS[preset_name].copy()
        st.session_state["active_screener_name"] = preset_name
        st.rerun()

    seed_config = {**PRESETS.get(preset_name, {}), **st.session_state.get("active_screener_config", {})}
    screener_settings = platform_settings.get("screener", {})
    model_settings = platform_settings.get("models", {})
    custom_universes = [item for item in screener_settings.get("custom_universes", []) if isinstance(item, dict)]
    model_registry = model_settings.get("registry", [])
    model_ids = [str(row.get("id")) for row in model_registry if row.get("id")]
    active_model_ids = [str(value) for value in model_settings.get("active_models", ["ols"]) if str(value) in model_ids] or (model_ids[:1] if model_ids else [])
    composite_weights = {str(k): float(v) for k, v in model_settings.get("composite_weights", {}).items()}

    with st.expander("Screener Settings", expanded=False):
        st.caption("Saved under output/config and reused by Screener, ML Lab and daily desk workflows.")
        default_template = st.selectbox(
            "Default screener template",
            list(PRESETS),
            index=list(PRESETS).index(screener_settings.get("default_template", "Custom"))
            if screener_settings.get("default_template", "Custom") in PRESETS
            else 0,
            help="Template suggested for new screening sessions.",
        )
        default_columns = st.multiselect(
            "Default result columns",
            [col for col in unified.columns if col not in {"source_artifact"}],
            default=[col for col in screener_settings.get("default_columns", []) if col in unified.columns],
            help="Preferred columns loaded automatically in the Results table.",
        )
        custom_name = st.text_input("Custom watchlist name", value="", placeholder="e.g. PM focus list")
        custom_tickers_raw = st.text_area(
            "Custom watchlist tickers",
            value="",
            height=80,
            placeholder="AAPL, MSFT, NVDA",
            help="Comma, space or newline separated tickers. Existing saved watchlists are preserved.",
        )
        if st.button("Save Screener Settings", width="stretch"):
            updated_custom = list(custom_universes)
            tickers = [ticker.strip().upper() for ticker in custom_tickers_raw.replace("\n", ",").replace(" ", ",").split(",") if ticker.strip()]
            if custom_name.strip() and tickers:
                updated_custom = [item for item in updated_custom if item.get("name") != custom_name.strip()]
                updated_custom.append({"name": custom_name.strip(), "tickers": sorted(set(tickers))})
            platform_settings["screener"] = {
                **screener_settings,
                "default_template": default_template,
                "default_columns": default_columns,
                "custom_universes": updated_custom,
            }
            save_path = save_platform_settings(roots["workspace"], platform_settings)
            st.session_state["preferred_screening_columns"] = default_columns
            st.success(f"Screener settings saved: {save_path.name}")

    with st.expander("LLM Screener Assistant", expanded=False):
        st.caption("Ollama can draft filter settings from a research idea. Review the JSON before applying anything.")
        nl_request = st.text_area(
            "Research idea",
            value="Quality at reasonable price with positive momentum and manageable leverage",
            height=80,
        )
        llm_model = st.text_input("Ollama model", value="llama3.1", key="screener_llm_model")
        if st.button("Draft screener config with Ollama", width="stretch"):
            context = {
                "available_universes": unique_options(unified, "universe")[:50],
                "available_sectors": unique_options(unified, "sector")[:50],
                "templates": list(PRESETS),
                "active_models": active_model_ids,
                "supported_numeric_filters": [
                    "max_pe",
                    "max_pb",
                    "min_dividend_yield",
                    "min_roe",
                    "max_debt_to_equity",
                    "min_ml_score",
                    "min_smart_money_score",
                    "ml_quintiles",
                    "require_smart_events",
                    "sort_by",
                ],
            }
            with st.spinner("Drafting filter configuration..."):
                advice = suggest_screener_config(nl_request, available_context=context, model=llm_model)
            if advice["status"] == "OK":
                st.code(advice["content"], language="json")
            else:
                st.warning(f"Ollama unavailable: {advice.get('error') or 'no response'}")

    st.markdown("#### Universe")
    universe_options = unique_options(unified, "universe")
    country_options = unique_options(unified, "country")
    sector_options = unique_options(unified, "sector")
    industry_options = unique_options(unified, "industry")
    universes = st.multiselect("Investment universe / index", universe_options, default=seed_config.get("universes") or [])
    custom_names = [str(item.get("name")) for item in custom_universes if item.get("name")]
    selected_custom_universe = st.selectbox(
        "Custom watchlist",
        ["No custom watchlist", *custom_names],
        help="Optional desk-defined ticker list saved in Screener Settings.",
    )
    custom_tickers = []
    if selected_custom_universe != "No custom watchlist":
        match = next((item for item in custom_universes if str(item.get("name")) == selected_custom_universe), {})
        custom_tickers = [str(ticker).upper() for ticker in match.get("tickers", [])]
    countries = st.multiselect("Region / country", country_options, default=seed_config.get("countries") or [])
    sectors = st.multiselect("Sector", sector_options, default=seed_config.get("sectors") or [])
    industries = st.multiselect("Industry", industry_options, default=seed_config.get("industries") or [])
    query = st.text_input("Ticker / issuer search", value=seed_config.get("query", ""))

    st.markdown("#### Fundamentals")
    pe_min, pe_max = numeric_range(unified, "pe", (0.0, 80.0))
    pb_min, pb_max = numeric_range(unified, "pb", (0.0, 20.0))
    dy_min, dy_max = numeric_range(unified, "dividend_yield", (0.0, 10.0))
    roe_min, roe_max = numeric_range(unified, "roe", (-30.0, 60.0))
    debt_min, debt_max = numeric_range(unified, "debt_to_equity", (0.0, 500.0))
    pe_hi = float(max(pe_max, pe_min + 1.0))
    pb_hi = float(max(pb_max, pb_min + 1.0))
    dy_hi = float(max(dy_max, dy_min + 1.0))
    roe_hi = float(max(roe_max, roe_min + 1.0))
    debt_hi = float(max(debt_max, debt_min + 1.0))
    max_pe = st.slider("Max P/E (x)", float(pe_min), pe_hi, clamp(seed_config.get("max_pe", pe_max), pe_min, pe_hi))
    max_pb = st.slider("Max Price / Book (x)", float(pb_min), pb_hi, clamp(seed_config.get("max_pb", pb_max), pb_min, pb_hi))
    min_dividend_yield = st.slider("Min Dividend Yield (%)", float(dy_min), dy_hi, clamp(seed_config.get("min_dividend_yield", dy_min), dy_min, dy_hi))
    min_roe = st.slider("Min ROE (%)", float(roe_min), roe_hi, clamp(seed_config.get("min_roe", roe_min), roe_min, roe_hi))
    max_debt_to_equity = st.slider("Max Net Debt / Equity (x)", float(debt_min), debt_hi, clamp(seed_config.get("max_debt_to_equity", debt_max), debt_min, debt_hi))

    st.markdown("#### ML / Smart Money")
    min_ml_score = st.slider("Min ML Conviction Score", 0.0, 100.0, float(seed_config.get("min_ml_score", 0.0)))
    min_smart_money_score = st.slider("Min Smart Money Composite", 0.0, 100.0, float(seed_config.get("min_smart_money_score", 0.0)))
    min_conviction = st.slider("Min Composite Conviction", 0.0, 100.0, float(seed_config.get("min_conviction", 0.0)))
    ml_quintiles = st.multiselect("ML quintiles", ["Q1", "Q2", "Q3", "Q4", "Q5"], default=seed_config.get("ml_quintiles", []))
    require_smart_events = st.toggle(
        "Require Recent Official-Source Event",
        value=bool(seed_config.get("require_smart_events", platform_settings.get("smart_money", {}).get("require_recent_event_default", False))),
    )
    with st.expander("Model routing", expanded=False):
        st.caption("Choose which validated model scores feed the desk composite. Score columns are used when available in ML artifacts.")
        selected_models = st.multiselect(
            "Models active in Screener",
            model_ids,
            default=active_model_ids,
            help="The Screener displays current artifact scores and records this model stack in saved configs.",
        )
        model_weights = {}
        for model_id in selected_models:
            model_weights[model_id] = st.slider(
                f"{model_id} composite weight",
                0.0,
                1.0,
                float(composite_weights.get(model_id, 1.0 / max(len(selected_models), 1))),
                step=0.05,
            )
        if st.button("Save Model Routing", width="stretch"):
            platform_settings["models"] = {
                **model_settings,
                "active_models": selected_models,
                "composite_weights": model_weights,
                "screener_enabled": True,
            }
            save_path = save_platform_settings(roots["workspace"], platform_settings)
            st.success(f"Model routing saved: {save_path.name}")
    selected_models = locals().get("selected_models", active_model_ids)
    model_weights = locals().get("model_weights", composite_weights)

    st.markdown("#### Sorting")
    sortable = [col for col in ["composite_conviction_score", "ml_score", "valuation_signal_score", "smart_money_score", "screener_score", "selection_score", "quality_proxy", "dividend_yield"] if col in unified.columns]
    sort_by = st.selectbox("Sort by", sortable, index=sortable.index(seed_config.get("sort_by", "composite_conviction_score")) if seed_config.get("sort_by") in sortable else 0)
    sort_ascending = st.toggle("Ascending sort", value=bool(seed_config.get("sort_ascending", False)))

    active_config = {
        "universes": universes,
        "countries": countries,
        "sectors": sectors,
        "industries": industries,
        "custom_universe": selected_custom_universe,
        "custom_tickers": custom_tickers,
        "query": query,
        "max_pe": max_pe,
        "max_pb": max_pb,
        "min_dividend_yield": min_dividend_yield,
        "min_roe": min_roe,
        "max_debt_to_equity": max_debt_to_equity,
        "min_ml_score": min_ml_score,
        "min_smart_money_score": min_smart_money_score,
        "min_conviction": min_conviction,
        "ml_quintiles": ml_quintiles,
        "require_smart_events": require_smart_events,
        "active_models": selected_models,
        "model_weights": model_weights,
        "sort_by": sort_by,
        "sort_ascending": sort_ascending,
    }
    st.session_state["active_screener_config"] = active_config

    st.markdown("#### Save")
    screener_name = st.text_input("Screener name", value=st.session_state.get("active_screener_name", "Custom"))
    screener_description = st.text_area("Description", value="", height=80)

filter_started = time.perf_counter()
with st.spinner("Applying institutional screener filters..."):
    filtered = apply_screening_filters(unified, active_config)
filter_elapsed = time.perf_counter() - filter_started
st.session_state["last_screening_result_count"] = len(filtered)

with left_panel:
    if st.button("Save screener config", width="stretch"):
        path = save_screener_config(roots["workspace"], screener_name, screener_description, active_config, len(filtered))
        st.session_state["active_screener_name"] = screener_name.strip() or "Untitled screener"
        st.success(f"Saved: {path.name}")
    if st.button("Reset filters", width="stretch"):
        st.session_state["active_screener_config"] = {}
        st.session_state["active_screener_name"] = "Custom"
        st.rerun()

with right_panel:
    st.subheader("Data Status")
    universe_scope = tuple(active_config.get("universes") or [])
    ticker_scope = tuple(filtered["ticker"].dropna().astype(str).str.upper().head(500).tolist()) if not filtered.empty and "ticker" in filtered.columns else tuple()
    with st.container(border=True):
        try:
            universe_health = _cached_universe_health(str(roots["financial_db"]), str(roots["workspace"]), universe_scope)
            ticker_status = _cached_ticker_status(str(roots["financial_db"]), str(roots["workspace"]), ticker_scope) if ticker_scope else pd.DataFrame()
            h1, h2, h3, h4 = st.columns(4)
            fundamentals_status = universe_health[universe_health["stage"].astype(str).eq("equity_fundamentals")]["status"].astype(str).str.upper().tolist()
            prices_status = universe_health[universe_health["stage"].astype(str).eq("equity_prices")]["status"].astype(str).str.upper().tolist()
            h1.metric("Fundamentals", _worst_status(fundamentals_status))
            h2.metric("Prices", _worst_status(prices_status))
            incomplete = 0
            if not ticker_status.empty and "overall_status" in ticker_status.columns:
                incomplete = int(ticker_status["overall_status"].astype(str).str.upper().ne("OK").sum())
            h3.metric("Ticker sample", len(ticker_status) if not ticker_status.empty else 0, f"{incomplete} watch")
            h4.metric("Scope", ", ".join(universe_scope[:2]) if universe_scope else "global")
            stage_watch = [status for status in [*fundamentals_status, *prices_status] if status not in {"OK"}]
            if ticker_status.empty and not stage_watch:
                st.success("Data status is OK for the current screener scope.")
            elif incomplete or stage_watch:
                st.warning(
                    f"Dati incompleti per {incomplete} dei {len(ticker_status)} ticker campionati. "
                    "Per refresh/restart apri Data Platform -> Data Health & Restart."
                )
            else:
                st.success("Dati: OK per lo scope corrente.")
            if not ticker_status.empty:
                with st.expander("Ticker data status sample", expanded=False):
                    show_cols = ["ticker", "overall_status", "fundamentals_status", "prices_status", "price_coverage_status", "provider_error_type", "last_price_date", "message"]
                    st.dataframe(ticker_status[[col for col in show_cols if col in ticker_status.columns]], width="stretch", hide_index=True)
            safe_page_link("pages/8_🗄️_Data_Platform.py", "Open Data Platform")
        except Exception as exc:
            st.warning(f"Data status temporarily unavailable: {type(exc).__name__}. Use Data Platform for detailed health.")

    st.subheader("Results")
    feedback = f"Filters applied to {len(unified):,} names - {len(filtered):,} match criteria in {filter_elapsed:.2f}s."
    if filtered.empty:
        st.warning(feedback)
    else:
        st.success(feedback)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Current Names", len(filtered), f"from {len(unified)}")
    k2.metric("ML Coverage", int(filtered["ml_score"].notna().sum()) if "ml_score" in filtered else 0)
    k3.metric("Smart Events", int(filtered["smart_recent_events"].sum()) if "smart_recent_events" in filtered else 0)
    k4.metric("Avg Conviction", metric_or_na(filtered["composite_conviction_score"].mean() if "composite_conviction_score" in filtered and not filtered.empty else None))
    model_stack = ", ".join(active_config.get("active_models") or []) or "default artifact score"
    st.caption(f"ML model stack in use: {model_stack}. Composite weights are saved with the screener config for reproducibility.")
    render_feature_metadata_expander(
        [
            "value_score",
            "quality_score",
            "momentum_score",
            "risk_score",
            "size_score",
            "growth_score",
            "ml_score",
            "score_composite",
            "valuation_signal_score",
            "smart_money_score",
            "composite_institutional_interest_score",
            "pe",
            "pb",
            "ev_ebitda",
            "roe",
            "debt_to_equity",
        ],
        "Column glossary for Screener scores",
    )

    default_cols = display_columns(filtered)
    column_options = [col for col in filtered.columns if col not in {"source_artifact"}]
    preferred_cols = [col for col in st.session_state.get("preferred_screening_columns", []) if col in column_options] or default_cols
    selected_cols = st.multiselect(
        "Results columns",
        column_options,
        default=preferred_cols,
    )
    st.session_state["preferred_screening_columns"] = selected_cols

    if filtered.empty:
        st.warning("No names match the current filter set. Suggested relaxations based on active filters:")
        for suggestion in screening_zero_result_suggestions(active_config, len(unified)):
            st.write(f"- {suggestion}")
        selected_ticker = ""
    else:
        view = filtered[selected_cols].copy() if selected_cols else filtered[default_cols].copy()
        event = st.dataframe(
            view,
            width="stretch",
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key="screener_results_grid",
        )
        selected_rows = getattr(getattr(event, "selection", None), "rows", []) if event is not None else []
        selected_ticker = ""
        if selected_rows:
            selected_ticker = str(filtered.iloc[selected_rows[0]]["ticker"])
            st.session_state["selected_ticker"] = selected_ticker
        elif st.session_state.get("selected_ticker") in set(filtered["ticker"].astype(str)):
            selected_ticker = st.session_state["selected_ticker"]
        else:
            selected_ticker = str(filtered.iloc[0]["ticker"])
            st.session_state["selected_ticker"] = selected_ticker

    action_cols = st.columns(4)
    if action_cols[0].button("Open in Valuation Research", width="stretch", disabled=not bool(selected_ticker)):
        st.session_state["selected_ticker"] = selected_ticker
        st.switch_page("pages/2_🔬_Valuation_Research.py")
    if action_cols[1].button("Open in Portfolio Research", width="stretch", disabled=not bool(selected_ticker)):
        st.session_state["selected_ticker"] = selected_ticker
        st.switch_page("pages/3_📁_Portfolio_Research.py")
    if action_cols[2].button("Explain ML signal", width="stretch", disabled=not bool(selected_ticker)):
        st.session_state["selected_ticker"] = selected_ticker
        st.session_state["active_explain_tab"] = "ML reasoning"
    if action_cols[3].button("Smart Money detail", width="stretch", disabled=not bool(selected_ticker)):
        st.session_state["selected_ticker"] = selected_ticker
        st.session_state["active_explain_tab"] = "Smart Money"

    st.caption("Navigation links use the shared selected ticker context.")
    link_cols = st.columns(4)
    with link_cols[0]:
        safe_page_link("pages/2_🔬_Valuation_Research.py", "Valuation Research")
    with link_cols[1]:
        safe_page_link("pages/3_📁_Portfolio_Research.py", "Portfolio Research")
    with link_cols[2]:
        safe_page_link("pages/9_ML_Stock_Lab.py", "ML Stock Lab")
    with link_cols[3]:
        safe_page_link("pages/1_📡_Smart_Money_Macro.py", "Smart Money")

    render_selected_ticker_context(roots, selected_ticker, expanded=False)

    tabs = st.tabs(["Results Diagnostics", "Risk & Correlation", "ML reasoning", "Smart Money", "Valuation summary", "Saved configs"])

    with tabs[0]:
        if not filtered.empty:
            chart_cols = [c for c in ["composite_conviction_score", "ml_score", "smart_money_score", "screener_score", "selection_score"] if c in filtered.columns]
            if chart_cols:
                metric = st.selectbox("Distribution metric", chart_cols)
                st.plotly_chart(px.histogram(filtered, x=metric, nbins=30, template="plotly_white", title=f"{metric} distribution"), width="stretch")
            if "sector" in filtered.columns and "composite_conviction_score" in filtered.columns:
                sector = filtered.groupby("sector", dropna=False, as_index=False)["composite_conviction_score"].mean().sort_values("composite_conviction_score", ascending=False)
                st.plotly_chart(px.bar(sector.head(20), x="sector", y="composite_conviction_score", template="plotly_white", title="Average conviction by sector"), width="stretch")
            dataframe_with_download("Filtered screener results", filtered[display_columns(filtered)], "screening_results.csv")
            with st.expander("Explain current top picks with Ollama", expanded=False):
                st.caption("The LLM explains the already-computed ranking; it does not change scores or select trades.")
                top_n = st.slider("Top names to explain", 3, 15, 8)
                if st.button("Explain top picks", width="stretch"):
                    with st.spinner("Writing stock-pick explanation..."):
                        advice = explain_stock_picks(filtered.head(top_n), model=st.session_state.get("screener_llm_model", "llama3.1"), top_n=top_n)
                    if advice["status"] == "OK":
                        st.markdown(advice["content"])
                    else:
                        st.warning(f"Ollama unavailable: {advice.get('error') or 'no response'}")

    with tabs[1]:
        if filtered.empty or "ticker" not in filtered.columns:
            st.info("Run a screener with at least two names to inspect quick correlations.")
        else:
            st.markdown("**Quick correlations**")
            st.caption("Computed from local OHLCV for a bounded subset of names. This is a fast risk lens, not a full optimizer.")
            default_limit = min(15, max(2, len(filtered)))
            c1, c2, c3 = st.columns(3)
            max_names = c1.slider("Names in matrix", 2, min(30, max(2, len(filtered))), default_limit)
            window = c2.slider("Return window", 63, 504, 252, step=21)
            benchmark = c3.text_input("Benchmark", value=str(st.session_state.get("benchmark_ticker", "SPY") or "SPY")).strip().upper()
            tickers_for_corr = filtered["ticker"].dropna().astype(str).str.upper().head(int(max_names)).tolist()
            with st.spinner("Computing quick correlation matrix from OHLCV..."):
                corr = compute_correlation_matrix(
                    tickers_for_corr,
                    financial_db_root=roots["financial_db"],
                    output_root=roots["workspace"],
                    window=int(window),
                    max_symbols=int(max_names),
                )
            summary = summarize_correlation_matrix(corr, benchmark=benchmark)
            s1, s2, s3 = st.columns(3)
            s1.metric("Assets", summary.get("asset_count", 0))
            s2.metric("Avg inter-name corr", f"{float(summary.get('avg_pairwise_corr')):.2f}" if pd.notna(summary.get("avg_pairwise_corr")) else "n/a")
            s3.metric(f"Avg corr vs {benchmark}", f"{float(summary.get('avg_corr_to_benchmark')):.2f}" if pd.notna(summary.get("avg_corr_to_benchmark")) else "n/a")
            if corr.empty:
                st.warning("No correlation matrix available for this subset. Check OHLCV coverage or reduce the universe.")
            else:
                st.plotly_chart(px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Screener subset correlation heatmap"), width="stretch")
                st.dataframe(corr, width="stretch")
            render_metric_metadata_expander(["avg_pairwise_corr", "correlation_to_benchmark", "annualized_volatility", "annualized_variance", "beta_to_benchmark"], "Risk statistic glossary")

    with tabs[2]:
        if selected_ticker:
            drivers, explanation = explain_ml_signal(filtered, selected_ticker)
            st.markdown(f"**{selected_ticker} · ML signal explanation**")
            st.write(explanation)
            if drivers.empty:
                st.info("No local feature attribution table is available for this ticker yet.")
            else:
                st.dataframe(drivers, width="stretch", hide_index=True)
                st.plotly_chart(px.bar(drivers, x="driver", y="value", color="family", template="plotly_white", title="SHAP-like driver proxy"), width="stretch")

    with tabs[3]:
        if selected_ticker:
            events, explanation = explain_smart_money(smart_money, selected_ticker, filtered)
            st.markdown(f"**{selected_ticker} · Smart Money / Gov Data**")
            st.write(explanation)
            if events.empty:
                st.info("No issuer-level event rows are present for this ticker in current Smart Money artifacts.")
            else:
                st.dataframe(events, width="stretch", hide_index=True)

    with tabs[4]:
        if selected_ticker:
            valuation, explanation = explain_valuation(filtered, selected_ticker)
            st.markdown(f"**{selected_ticker} · Valuation pre-read**")
            st.write(explanation)
            if valuation.empty:
                st.info("Detailed DCF/multiple contribution artifacts are not available for this ticker yet.")
            else:
                st.dataframe(valuation, width="stretch", hide_index=True)
                st.plotly_chart(px.bar(valuation, x="driver", y="value", template="plotly_white", title="Available valuation drivers"), width="stretch")

    with tabs[5]:
        if saved_screeners.empty:
            st.info("Saved configs will appear here after the first save.")
        else:
            st.dataframe(saved_screeners, width="stretch", hide_index=True)

with st.expander("Design note", expanded=False):
    st.markdown(
        """
        This page treats the screener as a reusable research object: filters, context, sorting and current count are saved
        as JSON under `output/screeners/`. The explanation panels intentionally distinguish calculated fields from
        proxy attribution when formal SHAP/local feature importance is not available.
        """
    )

render_footer()
