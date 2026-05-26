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

from support import configure_page, dataframe_with_download, render_context_bar, render_footer, render_page_header, render_page_intro, sidebar_roots

from research_platform_core import metadata_frame, metrics_metadata_frame


configure_page("Academic Reference")
roots = sidebar_roots()


def _ff_factor_files(output_root: Path) -> list[Path]:
    return sorted((output_root / "ff_factors").glob("*.parquet"))


def _load_ff_file(path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    return frame


def _factor_stats(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "date" not in frame.columns:
        return pd.DataFrame()
    rows = []
    for col in [c for c in frame.columns if c != "date"]:
        series = pd.to_numeric(frame[col], errors="coerce").dropna()
        if series.empty:
            continue
        rows.append(
            {
                "factor": col,
                "mean_ann": float(series.mean() * 12),
                "vol_ann": float(series.std(ddof=0) * (12**0.5)),
                "sharpe_ann": float((series.mean() * 12) / (series.std(ddof=0) * (12**0.5))) if series.std(ddof=0) else None,
                "min": float(series.min()),
                "max": float(series.max()),
                "observations": int(series.count()),
            }
        )
    return pd.DataFrame(rows)


render_page_header(
    "Academic Reference",
    "Interactive browser for feature definitions, metrics, Fama-French/AQR factors and bibliography used by Gen.is.IA.",
    "□",
    module="LABS",
    status="READY",
)
render_context_bar()
render_page_intro(
    "Use this page as the research-methodology index behind Screener, ML Stock Lab, Valuation and Portfolio.",
    "Filter features by category, inspect formulas, then compare cached academic factor returns where available.",
)

tabs = st.tabs(["Factor Browser", "Fama-French & AQR Factor Returns", "Metrics", "Bibliography"])

with tabs[0]:
    features = metadata_frame()
    if features.empty:
        st.info("Feature metadata registry is empty.")
    else:
        c1, c2, c3 = st.columns(3)
        categories = ["All", *sorted(features["category"].dropna().astype(str).unique())]
        category = c1.selectbox("Category", categories)
        asset_classes = ["All", *sorted(features.get("asset_class", pd.Series(["equity"])).dropna().astype(str).unique())]
        asset_class = c2.selectbox("Asset class", asset_classes)
        papers = ["All", *sorted(features.get("source_paper", pd.Series(dtype=str)).dropna().astype(str).replace("", pd.NA).dropna().unique())]
        paper = c3.selectbox("Paper", papers)
        view = features.copy()
        if category != "All":
            view = view[view["category"].astype(str).eq(category)]
        if asset_class != "All" and "asset_class" in view.columns:
            view = view[view["asset_class"].astype(str).eq(asset_class)]
        if paper != "All" and "source_paper" in view.columns:
            view = view[view["source_paper"].astype(str).eq(paper)]
        columns = [
            col
            for col in [
                "id",
                "name",
                "category",
                "sub_category",
                "formula",
                "formula_latex",
                "source_paper",
                "leakage_risk",
                "point_in_time_safe",
                "lag_required",
                "data_requirement",
                "interpretation",
            ]
            if col in view.columns
        ]
        dataframe_with_download("Feature registry", view[columns], "feature_registry.csv")
        selected = st.selectbox("Feature detail", view["id"].astype(str).tolist()) if not view.empty else ""
        if selected:
            row = view[view["id"].astype(str).eq(selected)].iloc[0].to_dict()
            st.markdown(f"### {row.get('name', selected)}")
            if row.get("formula_latex") and str(row.get("formula_latex")).lower() != "nan":
                st.latex(str(row["formula_latex"]).strip("$"))
            st.write(row.get("description", ""))
            st.caption(f"Formula: {row.get('formula', '')}")
            st.caption(f"Interpretation: {row.get('interpretation', '')}")
            st.caption(f"Source: {row.get('source_paper', 'n/a')} · PIT safe: {row.get('point_in_time_safe', True)}")

with tabs[1]:
    files = _ff_factor_files(roots["workspace"])
    if not files:
        st.info("No cached FF factor parquet files yet. Run `scripts/sync_universe.py --mode=full` to populate `output/ff_factors`.")
    else:
        file_map = {path.stem: path for path in files}
        choice = st.selectbox("Cached factor file", list(file_map))
        frame = _load_ff_file(file_map[choice])
        if frame.empty:
            st.warning("Selected factor file could not be loaded.")
        else:
            dataframe_with_download("Factor rows", frame.tail(250), f"{choice}.csv")
            value_cols = [col for col in frame.columns if col != "date"]
            selected_factors = st.multiselect("Factors", value_cols, default=value_cols[: min(4, len(value_cols))])
            if selected_factors:
                plot = frame[["date", *selected_factors]].melt("date", var_name="factor", value_name="return").dropna()
                plot["cum_return"] = plot.groupby("factor")["return"].transform(lambda s: (1 + s).cumprod() - 1)
                st.plotly_chart(px.line(plot, x="date", y="cum_return", color="factor", template="plotly_white", title=f"Cumulative factor return · {choice}"), width="stretch")
                corr = frame[selected_factors].corr()
                st.plotly_chart(px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu", title="Factor correlation"), width="stretch")
            stats = _factor_stats(frame)
            dataframe_with_download("Factor statistics", stats, f"{choice}_stats.csv")

with tabs[2]:
    metrics = metrics_metadata_frame()
    if metrics.empty:
        st.info("Metric metadata registry is empty.")
    else:
        category = st.selectbox("Metric category", ["All", *sorted(metrics["category"].dropna().astype(str).unique())])
        view = metrics if category == "All" else metrics[metrics["category"].astype(str).eq(category)]
        dataframe_with_download("Metric registry", view, "metric_registry.csv")

with tabs[3]:
    reference_path = PROJECT_ROOT / "docs" / "FEATURE_ACADEMIC_REFERENCE.md"
    if reference_path.exists():
        st.markdown(reference_path.read_text(encoding="utf-8"))
    else:
        st.info("Academic reference document not found.")

render_footer()
