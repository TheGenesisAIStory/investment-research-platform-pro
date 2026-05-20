from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
PROJECT_ROOT = APP_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(APP_DIR.parent) not in sys.path:
    sys.path.insert(0, str(APP_DIR.parent))

from support import configure_page, sidebar_roots

try:
    from src.research_platform_core.data_platform import (
        dataset_status,
        provider_fallback_plan,
        publish_artifacts_to_financial_db,
        refresh_europe_stoxx_prices_incremental,
        resolve_data_platform_roots,
        write_data_platform_status,
    )
except Exception:
    from research_platform_core.data_platform import (
        dataset_status,
        provider_fallback_plan,
        publish_artifacts_to_financial_db,
        refresh_europe_stoxx_prices_incremental,
        resolve_data_platform_roots,
        write_data_platform_status,
    )

configure_page("Data Platform")

import plotly.express as px
import streamlit as st


roots = sidebar_roots()
platform_roots = resolve_data_platform_roots(
    financial_db_root=roots["financial_db"],
    repo_output_root=roots["workspace"],
)

st.title("Data Platform / Database Finanziario")
st.caption("Drive-first data lake status, provider coverage, freshness and sync controls.")

cols = st.columns(4)
cols[0].metric("Financial DB", "available" if platform_roots.available else "missing")
cols[1].metric("Source", platform_roots.source)
cols[2].metric("Local Cache", platform_roots.local_cache.name)
cols[3].metric("Repo Output", platform_roots.repo_output.name)

st.code(str(platform_roots.financial_db))

max_files = st.sidebar.slider("Inventory file scan limit", min_value=500, max_value=12000, value=5000, step=500)
status = dataset_status(platform_roots.financial_db, max_files=max_files)
inventory = status["inventory"]
summary = status["summary"]

if inventory.empty or not platform_roots.available:
    st.warning("Database Finanziario is not available. Mount Google Drive or set FINANCIAL_DB_ROOT.")
    st.stop()

top = st.columns(4)
top[0].metric("Inventoried Files", len(inventory))
top[1].metric("Total MB", round(float(inventory.get("size_mb", []).sum()), 2))
top[2].metric("Roles", inventory["role"].nunique() if "role" in inventory else 0)
top[3].metric("Freshest Age h", round(float(inventory["age_hours"].min()), 2) if "age_hours" in inventory else "n/a")

tabs = st.tabs(["Inventory", "Freshness", "Providers", "Credentials", "Price Refresh", "Sync / Contracts"])

with tabs[0]:
    st.subheader("Inventory")
    st.dataframe(summary, width="stretch", hide_index=True)
    if not summary.empty and {"role", "files"}.issubset(summary.columns):
        st.plotly_chart(px.bar(summary, x="role", y="files", color="freshness_status", template="plotly_white", title="Dataset Inventory by Role"), width="stretch")
    with st.expander("Operating policy", expanded=True):
        st.markdown(
            """
            - **Drive-first:** Database Finanziario is the canonical data lake.
            - **Cache-second:** local/repo artifacts are reused before provider calls.
            - **API-last:** providers are called only for missing, stale or incomplete data.
            - **Weekly refresh + staleness refresh:** recurring checks avoid repeated full downloads.
            - **Credit discipline:** incremental writes, deduplication and provenance are preferred over bulk refresh.
            """
        )

with tabs[1]:
    st.subheader("Freshness")
    role_filter = st.multiselect("Role", sorted(inventory["role"].dropna().unique()), default=[])
    suffix_filter = st.multiselect("Suffix", sorted(inventory["suffix"].dropna().unique()), default=[])
    query = st.text_input("Search inventory", "")
    view = inventory.copy()
    if role_filter:
        view = view[view["role"].isin(role_filter)]
    if suffix_filter:
        view = view[view["suffix"].isin(suffix_filter)]
    if query:
        q = query.lower()
        view = view[view["relative_path"].str.lower().str.contains(q, na=False)]
    if "freshness_status" in view.columns:
        status_counts = view["freshness_status"].value_counts().to_dict()
        fcols = st.columns(4)
        for status_name, col in zip(["OK", "STALE", "UNKNOWN", "MISSING"], fcols):
            col.metric(status_name, status_counts.get(status_name, 0))
    st.dataframe(view.head(2000), width="stretch", hide_index=True)
    st.download_button("Download inventory CSV", inventory.to_csv(index=False), "data_platform_inventory.csv", "text/csv")

with tabs[2]:
    st.subheader("Providers")
    fallback = provider_fallback_plan(platform_roots.financial_db)
    st.dataframe(fallback, width="stretch", hide_index=True)
    for name in ["data_inventory", "data_domains"]:
        df = status.get(name)
        st.markdown(f"### {name}")
        if df is not None and not df.empty:
            st.dataframe(df, width="stretch", hide_index=True)
        else:
            st.info(f"{name} not found in `catalog/`.")

with tabs[3]:
    st.subheader("Credentials")
    left, right = st.columns(2)
    with left:
        st.markdown("### Provider Registry")
        providers = status.get("api_providers")
        if providers is not None and not providers.empty:
            st.dataframe(providers, width="stretch", hide_index=True)
        else:
            st.info("No provider registry found.")
    with right:
        st.markdown("### Credentials / Health")
        creds = status.get("credential_status_masked")
        health = status.get("api_provider_health_checks")
        if creds is not None and not creds.empty:
            st.dataframe(creds, width="stretch", hide_index=True)
        if health is not None and not health.empty:
            st.dataframe(health, width="stretch", hide_index=True)
    with st.expander("Credit optimization rulebook", expanded=True):
        st.markdown(
            """
            Credentials are displayed only as masked status. API calls should be triggered after freshness checks.
            Free providers are fallback/enrichment, not the primary store. Expensive/full-history calls should
            write back to Drive with metadata and be reused by notebooks and Streamlit.
            """
        )

with tabs[4]:
    st.subheader("Incremental Europe/STOXX Price Refresh")
    st.markdown(
        """
        This refresh is Drive-first and stale-aware. It inspects existing parquet files and calls yfinance only
        for stale/missing symbols, unless forced.
        """
    )
    r1, r2, r3 = st.columns(3)
    max_symbols = r1.number_input("Max symbols to inspect", min_value=1, max_value=250, value=25, step=1)
    stale_hours = r2.number_input("Stale threshold hours", min_value=24, max_value=24 * 60, value=24 * 7, step=24)
    force = r3.checkbox("Force refresh", value=False)
    if st.button("Run incremental price refresh", width="stretch"):
        manifest = refresh_europe_stoxx_prices_incremental(
            platform_roots.financial_db,
            max_symbols=int(max_symbols),
            stale_hours=int(stale_hours),
            force=bool(force),
        )
        st.dataframe(manifest, width="stretch", hide_index=True)

with tabs[5]:
    st.subheader("Status Exports")
    if st.button("Write DataPlatform status to workspace output", width="stretch"):
        paths = write_data_platform_status(platform_roots.financial_db, roots["workspace"], max_files=max_files)
        st.json({k: str(v) for k, v in paths.items()})
    st.subheader("API-ready contracts")
    fallback = provider_fallback_plan(platform_roots.financial_db)
    st.dataframe(fallback, width="stretch", hide_index=True)
    contract = roots["workspace"] / "api_contracts" / "data_platform_contract.json"
    if contract.exists():
        st.download_button("Download data_platform_contract.json", contract.read_bytes(), file_name=contract.name, mime="application/json")
    st.subheader("Publish notebook artifacts to Database Finanziario")
    domain = st.selectbox("Target domain", ["company_valuation_platform", "portfolio_analysis_model", "research_platform_app"])
    source_root = st.selectbox("Source output root", ["company", "portfolio", "workspace"])
    if st.button("Publish selected artifacts to Drive", width="stretch"):
        manifest = publish_artifacts_to_financial_db(roots[source_root], platform_roots.financial_db, domain)
        st.dataframe(manifest, width="stretch", hide_index=True)
