from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import pandas as pd
import streamlit as st

from orchestration import build_freshness_table
from support import add_artifact_usage, configure_page, find_artifacts, sidebar_roots


configure_page("Artifacts / Exports")

roots = sidebar_roots()
artifacts = add_artifact_usage(find_artifacts(roots))
freshness = build_freshness_table(roots)

st.title("Artifacts / Exports")
st.caption("Contract-level visibility into CSV, JSON and HTML outputs consumed by the app and future APIs.")

if artifacts.empty:
    st.info("No artifacts found. Run notebook export cells, then refresh.")
    st.stop()

view = artifacts.copy()
if not freshness.empty and "path" in freshness.columns:
    status_cols = freshness[["path", "status", "modified", "age_hours", "required"]].rename(columns={"modified": "contract_modified"})
    view = view.merge(status_cols, on="path", how="left")
    view["status"] = view["status"].fillna("UNCONTRACTED")
else:
    view["status"] = "UNCONTRACTED"

namespace_options = sorted(view["domain"].dropna().unique())
type_options = sorted(view["type"].dropna().unique())
status_options = sorted(view["status"].dropna().unique())

c1, c2, c3 = st.columns(3)
domains = c1.multiselect("Namespace / Job Domain", namespace_options, default=namespace_options)
file_types = c2.multiselect("Type", type_options, default=type_options)
statuses = c3.multiselect("Status", status_options, default=status_options)
query = st.text_input("Search name/path", "")

filtered = view[view["domain"].isin(domains) & view["type"].isin(file_types) & view["status"].isin(statuses)].copy()
if query:
    q = query.lower()
    filtered = filtered[filtered["name"].str.lower().str.contains(q, na=False) | filtered["path"].str.lower().str.contains(q, na=False)]

cols = st.columns(4)
cols[0].metric("Files", len(filtered))
cols[1].metric("CSV", int(filtered["type"].eq("csv").sum()))
cols[2].metric("JSON", int(filtered["type"].eq("json").sum()))
cols[3].metric("HTML", int(filtered["type"].eq("html").sum()))

is_data_platform = filtered["name"].str.contains("DataPlatform_|data_platform", case=False, na=False) | filtered["path"].str.contains("api_contracts", case=False, na=False)
is_smart_money = filtered["name"].str.contains("SmartMoney|smart_money", case=False, na=False) | filtered["path"].str.contains("smart_money", case=False, na=False)
is_ml_lab = filtered["name"].str.contains("MLStockLab|ml_stock_lab", case=False, na=False) | filtered["path"].str.contains("ml_stock_lab", case=False, na=False)
notebook_artifacts = filtered[~is_data_platform & ~is_smart_money & ~is_ml_lab].copy()
data_platform_artifacts = filtered[is_data_platform].copy()
smart_money_artifacts = filtered[is_smart_money].copy()
ml_lab_artifacts = filtered[is_ml_lab].copy()

tab_notebook, tab_data_platform, tab_smart_money, tab_ml_lab, tab_contract = st.tabs(["Notebook Artifacts", "Data Platform Artifacts", "Smart Money Artifacts", "ML Lab Artifacts", "Contract Freshness"])

display_cols = ["domain", "type", "name", "status", "modified", "age_hours", "used_by", "path", "size_kb"]

with tab_notebook:
    st.dataframe(notebook_artifacts[[c for c in display_cols if c in notebook_artifacts.columns]], width="stretch", hide_index=True)

with tab_data_platform:
    st.dataframe(data_platform_artifacts[[c for c in display_cols if c in data_platform_artifacts.columns]], width="stretch", hide_index=True)

with tab_smart_money:
    st.dataframe(smart_money_artifacts[[c for c in display_cols if c in smart_money_artifacts.columns]], width="stretch", hide_index=True)
    st.page_link("pages/8_Smart_Money_Gov_Data.py", label="Open Smart Money Government Data Engine")

with tab_ml_lab:
    st.dataframe(ml_lab_artifacts[[c for c in display_cols if c in ml_lab_artifacts.columns]], width="stretch", hide_index=True)
    st.page_link("pages/9_ML_Stock_Lab.py", label="Open ML Stock Lab")

with tab_contract:
    if freshness.empty:
        st.info("No artifact contracts configured.")
    else:
        st.dataframe(freshness, width="stretch", hide_index=True)

selected_path = st.selectbox("Select artifact for download", filtered["path"].tolist() if not filtered.empty else [])
if selected_path:
    path = Path(selected_path)
    if path.exists():
        mime = "text/csv" if path.suffix.lower() == ".csv" else "application/json" if path.suffix.lower() == ".json" else "text/html"
        st.download_button("Download selected artifact", path.read_bytes(), file_name=path.name, mime=mime, width="content")
        if path.suffix.lower() == ".html":
            st.link_button("Open HTML artifact path", f"file://{path}")

with st.expander("Artifact contract", expanded=True):
    st.markdown(
        """
        - **Notebook Artifacts** are produced by valuation and portfolio notebooks or their lightweight module refreshes.
        - **Data Platform Artifacts** describe Database Finanziario inventory, provider state and API-ready contracts.
        - **Smart Money Artifacts** are official-source overlays for valuation, portfolio and screener workflows.
        - **ML Lab Artifacts** are fair-value, mispricing, z-score and quintile diagnostics generated by `ml_stock_lab`.
        - `OK`, `STALE` and `MISSING_*` are contract statuses, not investment conclusions.
        """
    )
