from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_DIR.parent
for candidate in [APP_DIR, PROJECT_ROOT]:
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from support import configure_page, dataframe_with_download, sidebar_roots


configure_page("Laboratorio Research Library")

roots = sidebar_roots()
lab_root = PROJECT_ROOT / "laboratorio"
manifest_path = lab_root / "LABORATORIO_MANIFEST.csv"
repo_map_path = lab_root / "BEST_PRACTICE_REPOSITORIES.csv"
review_path = lab_root / "BEST_PRACTICE_REVIEW.md"

st.title("Laboratorio Research Library")
st.caption(
    "Curated notebook library: only references that can improve the canonical Research Platform are retained."
)

if not manifest_path.exists():
    st.error(f"Laboratorio manifest not found: {manifest_path}")
    st.stop()

manifest = pd.read_csv(manifest_path)
repo_map = pd.read_csv(repo_map_path) if repo_map_path.exists() else pd.DataFrame()

notebooks = manifest[manifest["kind"].eq("ipynb")].copy() if "kind" in manifest.columns else pd.DataFrame()
categories = sorted(notebooks["category"].dropna().unique()) if not notebooks.empty and "category" in notebooks.columns else []
targets = sorted(notebooks["promotion_target"].dropna().unique()) if not notebooks.empty and "promotion_target" in notebooks.columns else []

cols = st.columns(4)
cols[0].metric("Curated notebooks", len(notebooks))
cols[1].metric("Retained files", len(manifest))
cols[2].metric("Research categories", len(categories))
cols[3].metric("Promotion targets", len(targets))

st.markdown(
    """
    <div class="rp-note">
    This page is intentionally a research library, not another raw dump. A notebook stays here only when it has a clear
    bridge to ML Stock Lab, Portfolio Research, Company Valuation, Smart Money data, the Data Platform, or a future
    microstructure lab backed by local data.
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Library filters")
    selected_categories = st.multiselect("Category", categories, default=categories)
    selected_targets = st.multiselect("Promotion target", targets, default=targets)
    search = st.text_input("Search path / rationale", "")

view = notebooks.copy()
if selected_categories and "category" in view.columns:
    view = view[view["category"].isin(selected_categories)]
if selected_targets and "promotion_target" in view.columns:
    view = view[view["promotion_target"].isin(selected_targets)]
if search:
    haystack = (
        view.get("relative_path", pd.Series("", index=view.index)).astype(str)
        + " "
        + view.get("rationale", pd.Series("", index=view.index)).astype(str)
    ).str.lower()
    view = view[haystack.str.contains(search.lower(), regex=False)]

tab_inventory, tab_best_practices, tab_promotion, tab_review = st.tabs(
    ["Inventory", "Best-Practice Map", "Promotion Pipeline", "Review Notes"]
)

with tab_inventory:
    st.subheader("Curated notebook inventory")
    display_cols = [
        col
        for col in ["category", "promotion_target", "relative_path", "rationale", "size_kb"]
        if col in view.columns
    ]
    dataframe_with_download("Laboratorio curated notebooks", view[display_cols], "Laboratorio_Curated_Notebooks.csv")

    if not view.empty and "category" in view.columns:
        by_category = view.groupby("category", as_index=False).size().rename(columns={"size": "notebooks"})
        st.plotly_chart(
            px.bar(
                by_category.sort_values("notebooks", ascending=False),
                x="category",
                y="notebooks",
                title="Notebook count by research category",
                template="plotly_white",
            ),
            width="stretch",
        )

with tab_best_practices:
    st.subheader("External repository patterns reviewed")
    if repo_map.empty:
        st.info("No best-practice repository map is available.")
    else:
        dataframe_with_download("Best-practice repository map", repo_map, "Best_Practice_Repositories.csv")

    st.markdown(
        """
        Practical interpretation:

        - Use factor diagnostics and quintile tests before promoting any ML signal.
        - Keep portfolio analytics aligned with risk contribution, drawdown, turnover and robust reporting.
        - Treat deep learning and RL as optional research tracks unless they improve current artifacts.
        - Promote stable notebook ideas into `src/` modules before exposing them as app controls.
        """
    )

with tab_promotion:
    st.subheader("Promotion pipeline")
    if view.empty:
        st.info("No notebooks match the current filters.")
    else:
        pipeline = (
            view.groupby(["promotion_target", "category"], as_index=False)
            .size()
            .rename(columns={"size": "notebooks"})
            .sort_values(["promotion_target", "notebooks"], ascending=[True, False])
        )
        st.dataframe(pipeline, width="stretch", hide_index=True)
        st.plotly_chart(
            px.treemap(
                pipeline,
                path=["promotion_target", "category"],
                values="notebooks",
                title="Research library mapped to platform components",
                template="plotly_white",
            ),
            width="stretch",
        )

with tab_review:
    st.subheader("Selection rationale")
    if review_path.exists():
        st.markdown(review_path.read_text(encoding="utf-8"))
    else:
        st.info("Best-practice review notes are not available.")
