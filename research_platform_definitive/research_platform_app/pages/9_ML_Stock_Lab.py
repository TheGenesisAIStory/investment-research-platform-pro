from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_DIR.parent
for candidate in [APP_DIR, PROJECT_ROOT]:
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import plotly.express as px
import streamlit as st

from support import configure_page, dataframe_with_download, load_ml_stock_lab_artifacts, metric_value, sidebar_roots

try:
    from src.ml_stock_lab import run_ml_stock_lab_experiment
except Exception:
    from ml_stock_lab import run_ml_stock_lab_experiment


configure_page("ML Stock Lab")

roots = sidebar_roots()
output_root = roots["workspace"] / "ml_stock_lab"
data = load_ml_stock_lab_artifacts(roots["workspace"])

st.title("ML Stock Lab")
st.caption("Fair value ML, mispricing, z-score screening, expected return prediction and quintile portfolio diagnostics.")

with st.sidebar:
    universe = st.selectbox("Universe", ["Existing artifacts", "Database Finanziario subset"])
    model = st.selectbox("Model", ["ols", "lasso", "rf", "gbrt", "ensemble"])
    signal = st.selectbox("Signal", ["zscore", "mispricing_rel", "fair_value_hat"])
    max_rows = st.number_input("Max rows", min_value=100, max_value=10000, value=2000, step=100)

with st.container(border=True):
    st.markdown("**Operational controls**")
    c1, c2 = st.columns(2)
    if c1.button("Run ML Stock Lab refresh", width="stretch"):
        result = run_ml_stock_lab_experiment(output_root, roots["financial_db"], model=model, max_rows=int(max_rows))
        st.success(f"ML Stock Lab artifacts written to {output_root}")
        st.json({"status": result["status"], "paths": result.get("paths", {})})
        data = load_ml_stock_lab_artifacts(roots["workspace"])
    c2.page_link("pages/5_Run_Notebooks.py", label="Open Run Notebooks")

signals = data["signals"]
quintiles = data["quintile_returns"]
metrics = data["metrics"]

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

tab_overview, tab_signals, tab_quintiles, tab_models, tab_docs = st.tabs(["Overview", "Signals", "Quintile Backtest", "Models", "Methodology"])

with tab_overview:
    dataframe_with_download("ML Stock Lab metrics", metrics, "MLStockLab_metrics.csv")
    if not signals.empty and {"market_value", "fair_value_hat"}.issubset(signals.columns):
        st.plotly_chart(px.scatter(signals, x="market_value", y="fair_value_hat", hover_name="ticker" if "ticker" in signals.columns else None, title="Fair Value vs Market Value", template="plotly_white"), width="stretch")
    elif signals.empty:
        st.info("No ML Stock Lab artifacts found yet. Run the refresh button or the orchestration job.")

with tab_signals:
    dataframe_with_download("ML signals", signals, "MLStockLab_signals.csv")
    if not signals.empty and signal in signals.columns:
        st.plotly_chart(px.histogram(signals, x=signal, nbins=30, title=f"{signal} distribution", template="plotly_white"), width="stretch")
        x = "ticker" if "ticker" in signals.columns else signals.index
        st.plotly_chart(px.bar(signals.sort_values(signal, ascending=False).head(30), x=x, y=signal, title=f"Top {signal}", template="plotly_white"), width="stretch")
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
    dataframe_with_download("Model comparison", data["model_comparison"], "MLStockLab_model_comparison.csv")
    dataframe_with_download("Prediction metrics", data["prediction_metrics"], "MLStockLab_prediction_metrics.csv")

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
