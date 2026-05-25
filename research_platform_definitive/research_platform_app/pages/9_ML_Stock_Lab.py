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

from app_settings import build_model_score_view, load_platform_settings, model_registry_frame, save_platform_settings
from data_bootstrap import render_bootstrap_banner
from screener_workbench import explain_ml_signal, normalize_ticker
from support import configure_page, dataframe_with_download, load_ml_stock_lab_artifacts, metric_value, render_context_bar, render_footer, render_page_intro, safe_page_link, sidebar_roots
from ui_ops import render_missing_data_cta

from research_platform_core.data_health import get_data_status_for_tickers, get_stage_health_for_universes
from research_platform_core.llm_advisors import advise_forecast_horizon, advise_model_configuration, audit_model_governance
from ml_stock_lab.factor_registry import FACTOR_BLOCKS
from ml_stock_lab import run_ml_stock_lab_experiment


configure_page("ML Stock Lab")


@st.cache_data(ttl=300)
def _cached_global_stage_health(financial_db_root: str, output_root: str) -> pd.DataFrame:
    rows = [
        get_stage_health_for_universes("equity_fundamentals", [], financial_db_root, output_root),
        get_stage_health_for_universes("equity_prices", [], financial_db_root, output_root),
    ]
    return pd.concat(rows, ignore_index=True, sort=False)


@st.cache_data(ttl=300)
def _cached_ticker_status(financial_db_root: str, output_root: str, tickers: tuple[str, ...]) -> pd.DataFrame:
    statuses = get_data_status_for_tickers(list(tickers), financial_db_root, output_root)
    return pd.DataFrame([status.to_dict() for status in statuses.values()])


def _worst_status(values: list[str]) -> str:
    priority = {"FAILED": 4, "MISSING": 3, "PARTIAL": 2, "RUNNING": 2, "UNKNOWN": 1, "OK": 0}
    clean = [str(value or "UNKNOWN").upper() for value in values]
    return max(clean, key=lambda value: priority.get(value, 1)) if clean else "UNKNOWN"

roots = sidebar_roots()
output_root = roots["workspace"] / "ml_stock_lab"
platform_settings = load_platform_settings(roots["workspace"])
model_settings = platform_settings.get("models", {})
model_registry_df = model_registry_frame(platform_settings)
model_ids = model_registry_df["id"].astype(str).tolist() if not model_registry_df.empty and "id" in model_registry_df.columns else ["ols"]
active_model_ids = [model_id for model_id in model_settings.get("active_models", ["ols"]) if model_id in model_ids] or model_ids[:1]
composite_weights = {str(k): float(v) for k, v in model_settings.get("composite_weights", {}).items()}
data = load_ml_stock_lab_artifacts(roots["workspace"])

st.title("ML Stock Lab")
st.caption("Fair value ML, mispricing, z-score screening, expected return prediction and quintile portfolio diagnostics.")
render_context_bar()
render_page_intro(
    "Review model signals, compare model stacks and refresh ML artifacts without opening the training notebooks.",
    "Select a model, inspect signals and save the composite model used by the Screener.",
)
render_bootstrap_banner(roots, required=["equity_metadata", "ml_signals"])

with st.sidebar:
    st.markdown("### ML Controls")
    universe = st.selectbox("Universe", ["Existing artifacts", "Database Finanziario subset"], help="Existing artifacts are fastest; DB subset is for controlled refreshes.")
    model = st.selectbox(
        "Primary model",
        model_ids,
        index=model_ids.index(active_model_ids[0]) if active_model_ids and active_model_ids[0] in model_ids else 0,
        help="Model used when launching an on-demand refresh.",
    )
    with st.expander("Advanced options", expanded=False):
        max_rows = st.number_input("Max rows", min_value=100, max_value=10000, value=2000, step=100)
        target_horizon_days = st.selectbox("Expected-return horizon", [21, 63, 252], index=0, help="Forward return horizon used when the page has to create targets from prices.")
        selected_feature_blocks = st.multiselect(
            "Feature blocks",
            list(FACTOR_BLOCKS.keys()),
            default=["value", "quality", "momentum", "risk", "size", "growth", "model_based"],
            help="Canonical factor blocks used by model refreshes and model cards.",
        )
        ollama_model = st.text_input("Ollama model", value="llama3.1", help="Used only when an LLM advisor button is clicked.")

status_signals = data["signals"]
status_tickers = tuple(status_signals["ticker"].dropna().map(normalize_ticker).unique().tolist()[:500]) if not status_signals.empty and "ticker" in status_signals.columns else tuple()
with st.container(border=True):
    st.markdown("**Data Status & Readiness**")
    st.caption("Readiness uses Data Health 2.0. Detailed failure diagnostics and restarts stay in Data Platform.")
    try:
        stage_health = _cached_global_stage_health(str(roots["financial_db"]), str(roots["workspace"]))
        ticker_status = _cached_ticker_status(str(roots["financial_db"]), str(roots["workspace"]), status_tickers) if status_tickers else pd.DataFrame()
        h1, h2, h3, h4 = st.columns(4)
        fundamentals_status = stage_health[stage_health["stage"].astype(str).eq("equity_fundamentals")]["status"].astype(str).str.upper().tolist()
        prices_status = stage_health[stage_health["stage"].astype(str).eq("equity_prices")]["status"].astype(str).str.upper().tolist()
        h1.metric("Fundamentals", _worst_status(fundamentals_status))
        h2.metric("Prices", _worst_status(prices_status))
        if ticker_status.empty:
            complete = partial = missing = 0
        else:
            statuses = ticker_status["overall_status"].astype(str).str.upper()
            complete = int(statuses.eq("OK").sum())
            partial = int(statuses.eq("PARTIAL").sum())
            missing = int(statuses.isin(["FAILED", "MISSING"]).sum())
        h3.metric("Ticker readiness", f"{complete} OK", f"{partial} partial")
        h4.metric("Missing / Failed", missing)
        stage_worst = _worst_status([*fundamentals_status, *prices_status])
        if stage_worst in {"FAILED", "MISSING"} or missing > 0:
            st.error("I dati per questo universo/ticker sono incompleti; valuta un refresh da Data Platform prima di interpretare i punteggi ML.")
        elif stage_worst in {"PARTIAL", "RUNNING"} or partial > 0:
            st.warning("Dati parziali o in aggiornamento: interpreta score e quintili con cautela e verifica Data Platform.")
        else:
            st.success("Data readiness OK per gli artifact ML correnti.")
        run_rows = stage_health[stage_health["run_id"].fillna("").astype(str).str.len().gt(0)] if "run_id" in stage_health.columns else pd.DataFrame()
        if not run_rows.empty:
            st.caption("Recent run context: " + " · ".join(f"{row.stage}: {row.run_id}" for row in run_rows.itertuples()))
        button_cols = st.columns([0.35, 0.65])
        if button_cols[0].button("Apri Data Health & Restart", width="stretch"):
            st.switch_page("pages/8_🗄️_Data_Platform.py")
        with button_cols[1]:
            safe_page_link("pages/8_🗄️_Data_Platform.py", "Open Data Platform")
        if not ticker_status.empty:
            with st.expander("Ticker readiness sample", expanded=False):
                show_cols = ["ticker", "overall_status", "fundamentals_status", "prices_status", "price_coverage_status", "provider_error_type", "last_price_date", "message"]
                st.dataframe(ticker_status[[col for col in show_cols if col in ticker_status.columns]], width="stretch", hide_index=True)
    except Exception as exc:
        st.warning(f"Data readiness temporarily unavailable: {type(exc).__name__}. Open Data Platform for full diagnostics.")

with st.container(border=True):
    st.markdown("**Operational controls**")
    c1, c2 = st.columns(2)
    if c1.button("Run ML Stock Lab refresh", width="stretch"):
        try:
            result = run_ml_stock_lab_experiment(
                output_root,
                roots["financial_db"],
                model=model,
                max_rows=int(max_rows),
                feature_blocks=selected_feature_blocks,
                target_horizon_days=int(target_horizon_days),
            )
            st.success(f"ML Stock Lab artifacts written to {output_root}")
            st.json({"status": result["status"], "paths": result.get("paths", {})})
            data = load_ml_stock_lab_artifacts(roots["workspace"])
        except Exception as exc:
            log_dir = output_root / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"ml_stock_lab_{pd.Timestamp.now(tz='UTC').strftime('%Y%m%d_%H%M%S')}.log"
            log_path.write_text(f"{type(exc).__name__}: {exc}", encoding="utf-8")
            st.error("ML Stock Lab refresh failed cleanly. Check input artifacts, then retry from Notebook Runner or this page.")
            st.caption(f"Diagnostic log: {log_path}")
    with c2:
        safe_page_link("pages/6_🧪_Notebook_Runner.py", "Open Notebook Runner")

signals = data["signals"]
model_score_view = build_model_score_view(signals, active_model_ids, composite_weights)
quintiles = data["quintile_returns"]
metrics = data["metrics"]
signal_options = [col for col in ["score_composite", "ml_score", "zscore", "mispricing_rel", "fair_value_hat", *[f"score_{model_id}" for model_id in active_model_ids]] if col in (model_score_view.columns if not model_score_view.empty else signals.columns)]
signal = st.selectbox("Signal", signal_options or ["score"], help="Metric visualized in signal charts when present.")

if signals.empty:
    render_missing_data_cta(
        "ML Stock Lab",
        job_id="ml_stock_lab_experiments_refresh",
        output_path=output_root / "tables" / "MLStockLab_signals.csv",
        cli_hint="python research_platform_app/scheduler.py --once --jobs ml_stock_lab_experiments_refresh",
    )

context_ticker = normalize_ticker(st.session_state.get("selected_ticker", ""))
signal_tickers = sorted(signals["ticker"].dropna().map(normalize_ticker).unique().tolist()) if not signals.empty and "ticker" in signals.columns else []
selected_signal_ticker = ""
if signal_tickers:
    default_index = signal_tickers.index(context_ticker) if context_ticker in signal_tickers else 0
    selected_signal_ticker = st.selectbox("Selected ticker for ML explanation", signal_tickers, index=default_index)
    st.session_state["selected_ticker"] = selected_signal_ticker

cols = st.columns(4)
cols[0].metric("Panel Rows", len(data["panel"]))
cols[1].metric("Signal Rows", len(signals))
cols[2].metric("R2_OS", metric_value(metrics, ["r2_os"], "n/a"))
cols[3].metric("Sharpe LS", metric_value(metrics, ["sharpe_long_short"], "n/a"))

st.markdown(
    """
    <div class="rp-note">
    ML Stock Lab estimates fair value from fundamentals/features, converts valuation gaps into mispricing signals,
    and evaluates ranking quality through quintile portfolios. It is a research layer, not a standalone production strategy.
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("Explain selected ML signal", expanded=bool(selected_signal_ticker)):
    if not selected_signal_ticker:
        st.info("Select a ticker with ML artifacts to see signal reasoning.")
    else:
        explanation_frame = signals.copy()
        if "score" in explanation_frame.columns and "ml_score" not in explanation_frame.columns:
            explanation_frame["ml_score"] = explanation_frame["score"]
        drivers, explanation = explain_ml_signal(explanation_frame, selected_signal_ticker)
        st.write(explanation)
        if drivers.empty:
            st.info("No SHAP-like driver proxy is available for this ticker yet.")
        else:
            st.dataframe(drivers, width="stretch", hide_index=True)
            st.plotly_chart(px.bar(drivers, x="driver", y="value", color="family", template="plotly_white", title="ML driver proxy"), width="stretch")

tab_overview, tab_signals, tab_quintiles, tab_models, tab_docs = st.tabs(["Overview", "Signals", "Quintile Backtest", "Models & Settings", "Methodology"])

with tab_overview:
    dataframe_with_download("ML Stock Lab metrics", metrics, "MLStockLab_metrics.csv")
    if not signals.empty and {"market_value", "fair_value_hat"}.issubset(signals.columns):
        st.plotly_chart(px.scatter(signals, x="market_value", y="fair_value_hat", hover_name="ticker" if "ticker" in signals.columns else None, title="Fair Value vs Market Value", template="plotly_white"), width="stretch")
    elif signals.empty:
        st.info("No ML Stock Lab artifacts found yet. Run the refresh button or the orchestration job.")

with tab_signals:
    dataframe_with_download("ML signals", model_score_view if not model_score_view.empty else signals, "MLStockLab_signals.csv")
    chart_frame = model_score_view if not model_score_view.empty else signals
    if not chart_frame.empty and signal in chart_frame.columns:
        st.plotly_chart(px.histogram(chart_frame, x=signal, nbins=30, title=f"{signal} distribution", template="plotly_white"), width="stretch")
        x = "ticker" if "ticker" in chart_frame.columns else chart_frame.index
        st.plotly_chart(px.bar(chart_frame.sort_values(signal, ascending=False).head(30), x=x, y=signal, title=f"Top {signal}", template="plotly_white"), width="stretch")
    left, right = st.columns(2)
    with left:
        dataframe_with_download("Top names", data["top"], "MLStockLab_top.csv")
    with right:
        dataframe_with_download("Bottom names", data["bottom"], "MLStockLab_bottom.csv")

with tab_quintiles:
    dataframe_with_download("Quintile returns", quintiles, "MLStockLab_quintile_returns.csv")
    dataframe_with_download("Quintile metrics", data["quintile_metrics"], "MLStockLab_quintile_metrics.csv")
    if not quintiles.empty and {"quantile", "return"}.issubset(quintiles.columns):
        st.plotly_chart(px.bar(quintiles, x="quantile", y="return", color="date" if "date" in quintiles.columns else None, title="Quintile / Long-Short Returns", template="plotly_white"), width="stretch")

with tab_models:
    st.subheader("Model Settings")
    st.caption("This is the app-level model routing used by Screener_Builder. Training artifacts remain versioned by ML Stock Lab jobs.")
    if not model_registry_df.empty:
        st.dataframe(model_registry_df, width="stretch", hide_index=True)
    selected_models = st.multiselect(
        "Models active in Screener and comparison views",
        model_ids,
        default=active_model_ids,
        help="Choose one or more validated models. Score columns are displayed when matching artifacts exist.",
    )
    weights = {}
    if selected_models:
        weight_cols = st.columns(min(len(selected_models), 4))
        for idx, model_id in enumerate(selected_models):
            with weight_cols[idx % len(weight_cols)]:
                weights[model_id] = st.slider(
                    f"{model_id} weight",
                    0.0,
                    1.0,
                    float(composite_weights.get(model_id, 1.0 / max(len(selected_models), 1))),
                    step=0.05,
                    help="Weight used in the composite score when model-specific score columns are available.",
                )
    if st.button("Save Model Settings", width="stretch"):
        platform_settings["models"] = {
            **model_settings,
            "active_models": selected_models,
            "composite_weights": weights,
            "screener_enabled": True,
        }
        save_path = save_platform_settings(roots["workspace"], platform_settings)
        st.success(f"Model settings saved: {save_path.name}")
    if not model_score_view.empty:
        score_cols = [col for col in model_score_view.columns if col.startswith("score_")]
        show_cols = list(dict.fromkeys([col for col in ["ticker", *score_cols, "score_composite", "zscore", "mispricing_rel", "fair_value_hat"] if col in model_score_view.columns]))
        if show_cols:
            st.dataframe(model_score_view[show_cols].head(500), width="stretch", hide_index=True)
    dataframe_with_download("Model comparison", data["model_comparison"], "MLStockLab_model_comparison.csv")
    dataframe_with_download("Prediction metrics", data["prediction_metrics"], "MLStockLab_prediction_metrics.csv")
    with st.expander("LLM Model Advisor", expanded=False):
        st.caption("Ollama reviews configurations and governance; quantitative rankings remain produced by the ML/factor models.")
        use_case = st.text_input("Use case", value="medium-term stock picking for a buy-side research workflow")
        advisor_cols = st.columns(3)
        if advisor_cols[0].button("Suggest model configuration", width="stretch"):
            with st.spinner("Asking Ollama for a governed model configuration..."):
                advice = advise_model_configuration(
                    universe=universe,
                    target_horizon_days=int(target_horizon_days),
                    feature_blocks=selected_feature_blocks,
                    available_models=model_ids,
                    active_models=selected_models if "selected_models" in locals() else active_model_ids,
                    metrics=data.get("training_metrics", metrics),
                    constraints={"use_case": use_case, "interpretability": "balanced", "latency": "interactive app"},
                    model=ollama_model,
                )
            if advice["status"] == "OK":
                st.markdown(advice["content"])
            else:
                st.warning(f"Ollama unavailable: {advice.get('error') or 'no response'}")
        if advisor_cols[1].button("Advise forecast horizon", width="stretch"):
            with st.spinner("Reviewing forecast horizon trade-offs..."):
                advice = advise_forecast_horizon(
                    use_case=use_case,
                    current_horizon_days=int(target_horizon_days),
                    turnover_hint="lower turnover preferred unless IC is materially stronger",
                    cost_bps=10.0,
                    feature_blocks=selected_feature_blocks,
                    metrics=data.get("training_metrics", metrics),
                    model=ollama_model,
                )
            if advice["status"] == "OK":
                st.markdown(advice["content"])
            else:
                st.warning(f"Ollama unavailable: {advice.get('error') or 'no response'}")
        if advisor_cols[2].button("Audit model governance", width="stretch"):
            with st.spinner("Running LLM governance audit..."):
                advice = audit_model_governance(
                    model_cards=data.get("training_model_cards", pd.DataFrame()),
                    metrics=data.get("training_metrics", metrics),
                    feature_importance=data.get("training_feature_importance", pd.DataFrame()),
                    model=ollama_model,
                )
            if advice["status"] == "OK":
                st.markdown(advice["content"])
            else:
                st.warning(f"Ollama unavailable: {advice.get('error') or 'no response'}")
    st.markdown("#### Training artifacts")
    dataframe_with_download("Training metrics", data.get("training_metrics", pd.DataFrame()), "MLTraining_metrics.csv")
    dataframe_with_download("Model cards", data.get("training_model_cards", pd.DataFrame()), "MLTraining_model_cards.csv")
    feature_importance = data.get("training_feature_importance", pd.DataFrame())
    if not feature_importance.empty:
        dataframe_with_download("Feature importance", feature_importance, "MLTraining_feature_importance.csv")
        if {"feature", "importance", "model"}.issubset(feature_importance.columns):
            st.plotly_chart(
                px.bar(
                    feature_importance.sort_values("importance", key=lambda s: pd.to_numeric(s, errors="coerce").abs(), ascending=False).head(25),
                    x="feature",
                    y="importance",
                    color="model",
                    template="plotly_white",
                    title="Top model feature importances / coefficients",
                ),
                width="stretch",
            )

with tab_docs:
    st.markdown(
        """
        Core formulas:

        - Fair value: `V_hat = f_t(X)`
        - Relative mispricing: `(V_hat - V_mkt) / V_mkt`
        - Cross-sectional z-score: `(MP - mean_t(MP)) / std_t(MP)`
        - Long-short quintile: `R_Q5 - R_Q1`
        - Out-of-sample R2: `1 - SSE_model / SSE_benchmark`

        The page reuses `output/ml_stock_lab/tables/MLStockLab_*` when available and only recalculates on demand.
        """
    )

render_footer()
