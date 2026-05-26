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
import streamlit as st

from support import configure_page, dataframe_with_download, render_context_bar, render_footer, render_page_header, render_page_intro
from research_platform_core import metadata_frame, metrics_metadata_frame


configure_page("Feature & Metrics Glossary")

render_page_header(
    "Feature & Metrics Glossary",
    "Explore the full Gen.is.IA feature and metric registry with academic formulas, factor-zoo categories and implementation notes.",
    "□",
    module="RESEARCH",
    status="READY",
)
render_context_bar()
render_page_intro(
    "This glossary is the analyst-facing contract for the platform metadata layer.",
    "Use filters to inspect point-in-time safety, academic sources and implementation modules before using a feature in research.",
)

features = metadata_frame()
metrics = metrics_metadata_frame()

f_count = len(features)
m_count = len(metrics)
paper_count = len(
    set(features.get("source_paper", pd.Series(dtype=str)).dropna().astype(str))
    | set(metrics.get("source_paper", metrics.get("paper", pd.Series(dtype=str))).dropna().astype(str))
)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Features", f"{f_count}")
c2.metric("Metrics", f"{m_count}")
c3.metric("Feature categories", f"{features['category'].nunique() if not features.empty and 'category' in features else 0}")
c4.metric("Papers / sources", f"{paper_count}")

tab_features, tab_metrics, tab_export = st.tabs(["Features", "Metrics", "Export"])


def _filter_options(frame: pd.DataFrame, column: str) -> list[str]:
    if frame.empty or column not in frame.columns:
        return ["All"]
    values = sorted(frame[column].dropna().astype(str).replace("", pd.NA).dropna().unique())
    return ["All", *values]


with tab_features:
    if features.empty:
        st.info("No feature metadata available.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        category = c1.selectbox("Category", _filter_options(features, "category"))
        zoo = c2.selectbox("Factor zoo", _filter_options(features, "factor_zoo_category"))
        asset_class = c3.selectbox("Asset class", _filter_options(features, "asset_class"))
        paper = c4.selectbox("Source", _filter_options(features, "source_paper"))
        view = features.copy()
        for col, value in {
            "category": category,
            "factor_zoo_category": zoo,
            "asset_class": asset_class,
            "source_paper": paper,
        }.items():
            if value != "All" and col in view.columns:
                view = view[view[col].astype(str).eq(value)]
        visible_cols = [
            col
            for col in [
                "id",
                "name",
                "category",
                "factor_zoo_category",
                "source_paper",
                "source_doi",
                "direction",
                "leakage_risk",
                "point_in_time_safe",
                "lag_required",
                "implementation_module",
            ]
            if col in view.columns
        ]
        dataframe_with_download("Feature metadata", view[visible_cols], "feature_metadata_export.csv")
        selected = st.selectbox("Feature detail", view["id"].astype(str).tolist()) if not view.empty else ""
        if selected:
            row = view[view["id"].astype(str).eq(selected)].iloc[0].to_dict()
            with st.expander(f"{row.get('name', selected)} detail", expanded=True):
                st.write(row.get("description", ""))
                if row.get("formula_latex"):
                    st.latex(str(row["formula_latex"]).strip("$"))
                st.markdown(f"**Formula:** `{row.get('formula', '')}`")
                st.markdown(f"**Rationale:** {row.get('economic_rationale', '')}")
                st.markdown(f"**Interpretation:** {row.get('interpretation', '')}")
                doi = str(row.get("source_doi", "") or "")
                if doi:
                    target = doi if doi.startswith("http") else f"https://doi.org/{doi}"
                    st.markdown(f"**Source:** [{row.get('source_paper', 'source')}]({target})")

with tab_metrics:
    if metrics.empty:
        st.info("No metric metadata available.")
    else:
        c1, c2, c3 = st.columns(3)
        family = c1.selectbox("Metric family", _filter_options(metrics, "metric_family"))
        category = c2.selectbox("Metric category", _filter_options(metrics, "category"))
        source = c3.selectbox("Metric source", _filter_options(metrics, "source_paper"))
        view = metrics.copy()
        for col, value in {"metric_family": family, "category": category, "source_paper": source}.items():
            if value != "All" and col in view.columns:
                view = view[view[col].astype(str).eq(value)]
        visible_cols = [
            col
            for col in [
                "id",
                "name",
                "category",
                "metric_family",
                "source_paper",
                "source_doi",
                "is_higher_better",
                "formula",
                "interpretation_range",
            ]
            if col in view.columns
        ]
        dataframe_with_download("Metric metadata", view[visible_cols], "metric_metadata_export.csv")
        selected = st.selectbox("Metric detail", view["id"].astype(str).tolist()) if not view.empty else ""
        if selected:
            row = view[view["id"].astype(str).eq(selected)].iloc[0].to_dict()
            with st.expander(f"{row.get('name', selected)} detail", expanded=True):
                st.write(row.get("definition", ""))
                if row.get("formula_latex"):
                    st.latex(str(row["formula_latex"]).strip("$"))
                st.markdown(f"**Formula:** `{row.get('formula', '')}`")
                st.markdown(f"**Interpretation:** {row.get('interpretation', '')}")
                st.markdown(f"**Range:** {row.get('interpretation_range', row.get('typical_range', ''))}")

with tab_export:
    st.download_button(
        "Export Feature Metadata CSV",
        features.to_csv(index=False).encode("utf-8"),
        "genisia_feature_metadata.csv",
        "text/csv",
        use_container_width=True,
    )
    st.download_button(
        "Export Metric Metadata CSV",
        metrics.to_csv(index=False).encode("utf-8"),
        "genisia_metric_metadata.csv",
        "text/csv",
        use_container_width=True,
    )

render_footer()
