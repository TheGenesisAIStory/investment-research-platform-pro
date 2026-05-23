from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
for candidate in [APP_DIR, PROJECT_ROOT, PROJECT_ROOT / "src"]:
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

try:
    from support import configure_page, sidebar_roots
    from data_api_config import CONFIG_PATH, load_data_api_config
except Exception:
    from research_platform_app.support import configure_page, sidebar_roots
    from research_platform_app.data_api_config import CONFIG_PATH, load_data_api_config

try:
    from src.research_platform_core.api_management import (
        accepted_api_env_vars,
        api_control_status,
        build_env_template,
        resolve_api_control_roots,
        write_api_control_status,
    )
    from src.research_platform_core.batch_download import (
        EXPORT_FORMATS,
        create_batch_download,
        enrich_inventory_for_export,
        estimate_batch_size,
    )
    from src.research_platform_core.data_bridge import DataBridge
    from src.research_platform_core.data_center_catalog import build_target_catalog, summarize_target_catalog
    from src.research_platform_core.data_platform import (
        dataset_status,
        provider_fallback_plan,
        read_dataset_drive_first,
        refresh_europe_stoxx_prices_incremental,
        resolve_data_platform_roots,
        write_data_platform_status,
    )
except Exception:
    from research_platform_core.api_management import (
        accepted_api_env_vars,
        api_control_status,
        build_env_template,
        resolve_api_control_roots,
        write_api_control_status,
    )
    from research_platform_core.batch_download import (
        EXPORT_FORMATS,
        create_batch_download,
        enrich_inventory_for_export,
        estimate_batch_size,
    )
    from research_platform_core.data_bridge import DataBridge
    from research_platform_core.data_center_catalog import build_target_catalog, summarize_target_catalog
    from research_platform_core.data_platform import (
        dataset_status,
        provider_fallback_plan,
        read_dataset_drive_first,
        refresh_europe_stoxx_prices_incremental,
        resolve_data_platform_roots,
        write_data_platform_status,
    )


def _theme_css(theme: dict[str, str]) -> str:
    primary = theme.get("primary_color", "#1E3A8A")
    success = theme.get("success_color", "#10B981")
    warning = theme.get("warning_color", "#F59E0B")
    error = theme.get("error_color", "#EF4444")
    neutral = theme.get("neutral_color", "#6B7280")
    return f"""
    <style>
    .block-container {{ padding-top: 1.4rem; max-width: 1500px; }}
    div[data-testid="stMetric"] {{
        background:#ffffff;
        border:1px solid #e5e7eb;
        border-radius:8px;
        padding:12px;
        box-shadow:0 1px 2px rgba(15,23,42,.04);
    }}
    .rp-kpi {{
        border:1px solid #e5e7eb;
        border-radius:8px;
        background:#ffffff;
        padding:14px;
        min-height:118px;
        box-shadow:0 1px 2px rgba(15,23,42,.04);
    }}
    .rp-kpi-label {{ color:{neutral}; font-size:12px; font-weight:700; text-transform:uppercase; }}
    .rp-kpi-value {{ color:#111827; font-size:26px; font-weight:780; margin-top:4px; }}
    .rp-kpi-sub {{ color:{neutral}; font-size:12px; margin-top:4px; }}
    .rp-action {{
        border:1px solid #dbe4ef;
        border-radius:8px;
        padding:10px 12px;
        margin:8px 0;
        background:#ffffff;
    }}
    .rp-status-ok {{ color:{success}; font-weight:750; }}
    .rp-status-warn {{ color:{warning}; font-weight:750; }}
    .rp-status-bad {{ color:{error}; font-weight:750; }}
    .rp-section-note {{
        border-left:4px solid {primary};
        background:#f8fafc;
        padding:10px 12px;
        border-radius:8px;
        color:#374151;
    }}
    .rp-pill {{
        display:inline-block;
        padding:4px 8px;
        margin:2px;
        border-radius:999px;
        border:1px solid #e5e7eb;
        color:#374151;
        background:#ffffff;
        font-size:12px;
        font-weight:650;
    }}
    </style>
    """


def _metric_value(df: pd.DataFrame, column: str, default: object = 0) -> object:
    if df.empty or column not in df.columns:
        return default
    return df[column].iloc[0]


def _bool_sum(df: pd.DataFrame, column: str) -> int:
    if df.empty or column not in df.columns:
        return 0
    return int(df[column].fillna(False).astype(bool).sum())


def _health_score(platform_available: bool, api_available: bool, providers: pd.DataFrame, health: pd.DataFrame) -> float:
    provider_count = max(len(providers), 1)
    reachable_ratio = _bool_sum(health, "reachable") / provider_count if not health.empty else 0.0
    base = 0.35 if platform_available else 0.0
    api = 0.15 if api_available else 0.0
    return round(min(1.0, base + api + (0.5 * reachable_ratio)), 3)


def _health_label(score: float) -> tuple[str, str]:
    if score >= 0.9:
        return "Healthy", "ok"
    if score >= 0.75:
        return "Watch", "warn"
    return "Needs attention", "bad"


def _gauge(score: float, theme: dict[str, str]) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(score * 100, 1),
            number={"suffix": "%", "font": {"size": 24}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 0},
                "bar": {"color": theme.get("primary_color", "#1E3A8A")},
                "steps": [
                    {"range": [0, 75], "color": "#fee2e2"},
                    {"range": [75, 90], "color": "#fef3c7"},
                    {"range": [90, 100], "color": "#d1fae5"},
                ],
            },
        )
    )
    fig.update_layout(height=180, margin=dict(l=10, r=10, t=20, b=0))
    return fig


def _kpi_card(icon: str, label: str, value: object, sub: str = "") -> None:
    st.markdown(
        f"""
        <div class="rp-kpi">
          <div class="rp-kpi-label">{icon} {label}</div>
          <div class="rp-kpi-value">{value}</div>
          <div class="rp-kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_table(df: pd.DataFrame, key: str, height: int = 420) -> None:
    if df.empty:
        st.info("No rows available.")
        return
    try:
        from st_aggrid import AgGrid, GridOptionsBuilder

        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_default_column(filter=True, sortable=True, resizable=True)
        gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=25)
        AgGrid(df, gridOptions=gb.build(), height=height, fit_columns_on_grid_load=False, key=key)
    except Exception:
        safe_df = df.copy()
        for col in safe_df.select_dtypes(include=["object"]).columns:
            safe_df[col] = safe_df[col].astype(str)
        st.dataframe(safe_df, width="stretch", height=height, hide_index=True)


def _latest_batch_runs(output_root: Path, limit: int = 6) -> pd.DataFrame:
    root = output_root / "batch_downloads"
    if not root.exists():
        return pd.DataFrame()
    rows = []
    for meta in sorted(root.glob("*/metadata.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]:
        try:
            data = pd.read_json(meta, typ="series").to_dict()
        except Exception:
            data = {}
        rows.append(
            {
                "run": meta.parent.name,
                "generated_at": data.get("generated_at", ""),
                "format": data.get("export_format", ""),
                "files": (data.get("summary") or {}).get("files", ""),
                "size_mb": (data.get("summary") or {}).get("size_mb", ""),
                "path": str(meta.parent),
            }
        )
    return pd.DataFrame(rows)


def _activity_feed(inventory: pd.DataFrame, health: pd.DataFrame, output_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not inventory.empty and "modified_utc" in inventory.columns:
        latest = inventory.sort_values("modified_utc", ascending=False).head(5)
        for _, row in latest.iterrows():
            rows.append(
                {
                    "type": "dataset",
                    "item": row.get("file", row.get("relative_path", "")),
                    "status": row.get("role", ""),
                    "time": row.get("modified_utc", ""),
                }
            )
    if not health.empty:
        for _, row in health.head(5).iterrows():
            rows.append(
                {
                    "type": "api",
                    "item": row.get("name", ""),
                    "status": "reachable" if bool(row.get("reachable", False)) else "check",
                    "time": row.get("status_code", row.get("error", "")),
                }
            )
    batch = _latest_batch_runs(output_root, limit=3)
    for _, row in batch.iterrows():
        rows.append({"type": "batch", "item": row.get("run", ""), "status": row.get("format", ""), "time": row.get("generated_at", "")})
    return pd.DataFrame(rows)


def _filter_inventory(inventory: pd.DataFrame, prefix: str = "inventory") -> pd.DataFrame:
    view = enrich_inventory_for_export(inventory)
    if view.empty:
        return view
    with st.container(border=True):
        f1, f2, f3, f4 = st.columns([1, 1, 1, 2])
        roles = f1.multiselect("Role", sorted(view["role"].dropna().unique()), key=f"{prefix}_role")
        providers = f2.multiselect("Provider", sorted(view["provider"].dropna().unique()), key=f"{prefix}_provider")
        suffixes = f3.multiselect("Suffix", sorted(view["suffix"].dropna().unique()) if "suffix" in view else [], key=f"{prefix}_suffix")
        query = f4.text_input("Search datasets", key=f"{prefix}_query")
        freshness = st.multiselect("Freshness", sorted(view["freshness_bucket"].dropna().unique()), key=f"{prefix}_freshness")
    if roles:
        view = view[view["role"].isin(roles)]
    if providers:
        view = view[view["provider"].isin(providers)]
    if suffixes and "suffix" in view:
        view = view[view["suffix"].isin(suffixes)]
    if freshness:
        view = view[view["freshness_bucket"].isin(freshness)]
    if query and "relative_path" in view:
        q = query.lower()
        view = view[view["relative_path"].astype(str).str.lower().str.contains(q, na=False)]
    return view


def _market_context_widget(financial_db: Path, config: dict[str, Any]) -> None:
    market_cfg = config.get("market_context", {})
    presets = market_cfg.get("currencies", {})
    default_currency = market_cfg.get("default_currency", "USD")
    currency = st.selectbox("Currency", list(presets) or [default_currency], index=0, key="data_api_currency")
    preset = presets.get(currency, {})
    benchmark = st.text_input("Benchmark", preset.get("benchmark", "SPY"), key="data_api_benchmark")
    overlays = preset.get("overlays", ["UUP", "TLT", "GLD", "DBC"])
    st.caption(f"FX proxy: {preset.get('fx_pair', 'n/a')}")

    prices, meta = read_dataset_drive_first(financial_db, "prices", benchmark, allow_stale=True)
    if prices.empty or "close" not in prices.columns:
        st.info("Benchmark chart not found in Drive cache. API fallback stays disabled until you refresh explicitly.")
    else:
        plot = prices.copy()
        date_col = "date_time" if "date_time" in plot.columns else plot.columns[0]
        plot[date_col] = pd.to_datetime(plot[date_col], errors="coerce")
        plot = plot.dropna(subset=[date_col]).tail(90)
        if not plot.empty:
            first = float(plot["close"].iloc[0])
            last = float(plot["close"].iloc[-1])
            change = ((last / first) - 1.0) * 100 if first else 0.0
            st.metric(benchmark, f"{last:,.2f}", f"{change:.2f}%")
            st.plotly_chart(px.line(plot, x=date_col, y="close", template="plotly_white", height=160), width="stretch")
    overlay_rows = []
    for ticker in overlays:
        cached, cached_meta = read_dataset_drive_first(financial_db, "prices", ticker, allow_stale=True)
        overlay_rows.append({"ticker": ticker, "cache_status": cached_meta.get("status", "missing"), "rows": len(cached)})
    st.dataframe(pd.DataFrame(overlay_rows), width="stretch", hide_index=True)


def _render_dashboard(
    roots: dict[str, Path],
    platform_roots: Any,
    api_roots: Any,
    inventory: pd.DataFrame,
    summary: pd.DataFrame,
    providers: pd.DataFrame,
    credentials: pd.DataFrame,
    health: pd.DataFrame,
    config: dict[str, Any],
    theme: dict[str, str],
) -> None:
    score = _health_score(platform_roots.available, api_roots.api_available, providers, health)
    health_text, health_tone = _health_label(score)

    k1, k2, k3, k4 = st.columns([1, 1, 1, 1])
    with k1:
        _kpi_card("DB", "Total Datasets", len(inventory), f"{estimate_batch_size(enrich_inventory_for_export(inventory))['size_mb']} MB indexed")
    with k2:
        reachable = _bool_sum(health, "reachable")
        _kpi_card("API", "API Providers", len(providers), f"{reachable} reachable · {_bool_sum(credentials, 'configured')} credentials active")
    with k3:
        _kpi_card("CAT", "Data Roles", inventory["role"].nunique() if "role" in inventory else 0, "Drive-first inventory taxonomy")
    with k4:
        _kpi_card("OK", "Health Status", health_text, f"{round(score * 100, 1)}% overall")

    left, center, right = st.columns([0.9, 1.45, 1.0])
    with left:
        st.subheader("Quick Actions")
        if st.button("Sync All Databases", width="stretch"):
            data_paths = write_data_platform_status(platform_roots.financial_db, roots["workspace"], max_files=len(inventory) or 5000)
            api_paths = write_api_control_status(platform_roots.financial_db, roots["workspace"], api_roots.api_root)
            st.toast("Contracts refreshed.", icon="✅")
            st.json({"data": {k: str(v) for k, v in data_paths.items()}, "api": {k: str(v) for k, v in api_paths.items()}})
        if st.button("Batch Download All Data", width="stretch"):
            st.session_state["data_api_select_all"] = True
            st.toast("Open Data Inventory to launch the export.", icon="📥")
        if st.button("Search Datasets", width="stretch"):
            st.session_state["inventory_query"] = ""
            st.toast("Use the Data Inventory search box.", icon="🔍")
        if st.button("API Configuration", width="stretch"):
            st.toast("Open the API Registry or Credentials tabs.", icon="⚙️")
        st.markdown(
            f"""
            <div class="rp-section-note">
            Scheduler targets: market data {config.get('schedules', {}).get('market_data_sync')},
            fundamentals {config.get('schedules', {}).get('fundamentals_refresh')},
            API health every {config.get('schedules', {}).get('api_health_minutes')} min.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with center:
        st.subheader("Activity Feed & Recent Updates")
        feed = _activity_feed(inventory, health, roots["workspace"])
        _render_table(feed, "activity_feed", height=300)
        if not summary.empty:
            fig = px.bar(summary, x="role", y="files", color="freshness_status", template="plotly_white", height=240)
            fig.update_layout(margin=dict(l=8, r=8, t=16, b=8), legend_title_text="")
            st.plotly_chart(fig, width="stretch")

    with right:
        st.subheader("System Health")
        st.plotly_chart(_gauge(score, theme), width="stretch")
        st.markdown(f"<span class='rp-status-{health_tone}'>{health_text}</span>", unsafe_allow_html=True)
        st.subheader("Market Context")
        _market_context_widget(platform_roots.financial_db, config)


def _render_target_catalog_sections(financial_db: Path) -> None:
    catalog = build_target_catalog(financial_db)
    summary = summarize_target_catalog(catalog)
    st.markdown("### Strategic Data Center Coverage")
    if not summary.empty:
        domains = ["ohlcv_prices", "equity_universe", "official_macro", "factor_data", "fx", "risk_factors"]
        cols = st.columns(len(domains))
        for idx, domain in enumerate(domains):
            row = summary[summary["domain"].eq(domain)]
            value = "0/0"
            delta = "0%"
            if not row.empty:
                value = f"{int(row['available'].iloc[0])}/{int(row['targets'].iloc[0])}"
                delta = f"{row['coverage_pct'].iloc[0]}%"
            cols[idx].metric(domain.replace("_", " ").title(), value, delta)
    labels = {
        "ohlcv_prices": "OHLCV Prices",
        "equity_universe": "Equity Indices",
        "official_macro": "Official Macro",
        "factor_data": "Factor Data",
        "fx": "FX Majors",
        "commodities": "Commodities",
        "risk_factors": "Risk Factors",
    }
    for domain, label in labels.items():
        section = catalog[catalog["domain"].eq(domain)].sort_values(["priority", "dataset"])
        with st.expander(label, expanded=domain in {"equity_universe", "factor_data"}):
            _render_table(section, f"target_catalog_{domain}", height=260)
            missing = section[~section["exists"]]
            if not missing.empty:
                if domain == "official_macro":
                    script_hint = "scripts/sync_official_macro.py --execute --build-features"
                elif domain == "ohlcv_prices":
                    script_hint = "scripts/sync_ohlcv_prices.py --execute --mode incremental"
                else:
                    script_hint = "scripts/initial_setup.py --execute"
                st.caption(f"Missing targets: {len(missing)}. Use `{script_hint}` or the specific sync scripts when ready.")


def _render_inventory_tab(inventory: pd.DataFrame, roots: dict[str, Path], config: dict[str, Any]) -> None:
    _render_target_catalog_sections(roots["financial_db"])
    st.divider()
    st.subheader("Dataset Explorer")
    view = _filter_inventory(inventory, prefix="inventory")
    stats = estimate_batch_size(view)
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Filtered Files", stats["files"])
    s2.metric("Estimated MB", stats["size_mb"])
    s3.metric("Roles", stats["roles"])
    s4.metric("Providers", stats["providers"])
    _render_table(view.head(int(config.get("batch_download", {}).get("max_preview_rows", 2000))), "inventory_table", height=430)

    st.markdown("### Batch Download Manager")
    b1, b2, b3, b4 = st.columns([1, 1, 1, 1])
    export_format = b1.selectbox("Export format", EXPORT_FORMATS, index=EXPORT_FORMATS.index(config.get("batch_download", {}).get("default_format", "csv")))
    max_default = int(config.get("batch_download", {}).get("max_interactive_export_files", 250))
    export_all = b2.checkbox("Select all filtered", value=bool(st.session_state.get("data_api_select_all", False)))
    allow_large = b3.checkbox("Allow large export", value=False)
    max_files = b4.number_input("Max files", min_value=1, max_value=max(1, len(view)), value=min(max_default, max(1, len(view))), step=1)

    selected = view if export_all else view.head(int(max_files))
    if not allow_large and len(selected) > max_default:
        selected = selected.head(max_default)
        st.warning(f"Interactive export capped at {max_default} files. Enable large export to include more.")
    preview = estimate_batch_size(selected)
    st.caption(f"Selected: {preview['files']} files · {preview['size_mb']} MB · format={export_format}")

    progress = st.progress(0)
    status_box = st.empty()
    if st.button("Start Batch Export", width="stretch", disabled=selected.empty):
        def _progress(current: int, total: int, row: dict[str, Any]) -> None:
            ratio = min(1.0, current / max(total, 1))
            progress.progress(ratio)
            status_box.write(f"{current}/{total} · {row.get('file', row.get('source_path', 'dataset'))} · {row.get('status', '')}")

        result = create_batch_download(
            selected,
            roots["workspace"],
            export_format=export_format,
            max_files=None,
            run_label=None,
            progress_callback=_progress,
        )
        st.success(f"Batch export completed: {result['export_root']}")
        _render_table(result["manifest"], "batch_manifest", height=260)
        st.download_button("Download manifest CSV", result["manifest"].to_csv(index=False), "batch_manifest.csv", "text/csv")

    recent = _latest_batch_runs(roots["workspace"])
    if not recent.empty:
        st.markdown("### Recent Batch Runs")
        _render_table(recent, "recent_batch_runs", height=220)

    with st.expander("ml_stock_lab Bridge", expanded=False):
        bridge = DataBridge(PROJECT_ROOT, roots["workspace"])
        notebooks = bridge.discover_ml_stock_lab_notebooks()
        st.caption("Publish a manifest of the currently selected datasets so ml_stock_lab can consume the same Drive-first data contracts.")
        n1, n2 = st.columns(2)
        n1.metric("Selected datasets", len(selected))
        n2.metric("Discovered notebooks", len(notebooks))
        copy_files = st.checkbox("Copy files into bridge cache", value=False)
        if st.button("Push selected manifest to ml_stock_lab", width="stretch", disabled=selected.empty):
            paths = bridge.push_manifest(selected, copy_files=copy_files)
            st.success("ml_stock_lab bridge contract written.")
            st.json({name: str(path) for name, path in paths.items()})
        _render_table(notebooks, "ml_stock_lab_notebooks", height=180)


def _render_api_registry_tab(providers: pd.DataFrame, health: pd.DataFrame, fallback: pd.DataFrame) -> None:
    st.subheader("API Registry")
    p1, p2, p3 = st.columns(3)
    p1.metric("Registered Providers", len(providers))
    p2.metric("Reachable", _bool_sum(health, "reachable"))
    p3.metric("Fallback Sources", len(fallback))
    _render_table(providers, "api_providers", height=320)
    st.markdown("### API Health Checks")
    _render_table(health, "api_health", height=260)
    st.markdown("### Fallback Policy")
    _render_table(fallback, "fallback_policy", height=320)


def _render_credentials_tab(providers: pd.DataFrame, credentials: pd.DataFrame, api_files: pd.DataFrame, sections: pd.DataFrame) -> None:
    st.subheader("Credential Vault")
    st.caption("Values are kept in private Drive files or runtime environment. Exports contain masked status only.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Accepted Env Vars", len(accepted_api_env_vars(providers)))
    c2.metric("Configured", _bool_sum(credentials, "configured"))
    c3.metric("Private API Files", len(api_files[api_files["exists"].eq(True)]) if not api_files.empty and "exists" in api_files else 0)
    _render_table(credentials, "credential_status", height=280)

    env_vars = accepted_api_env_vars(providers)
    template = build_env_template(providers)
    if template:
        st.download_button("Download .env template", template, "research_platform_api.env.example", "text/plain")
    if env_vars:
        with st.expander("Session credential loader", expanded=False):
            selected = st.selectbox("Environment variable", env_vars)
            secret = st.text_input("Credential value", value="", type="password")
            strength = min(100, len(secret.strip()) * 4)
            st.progress(strength / 100)
            if st.button("Load for this session", width="stretch", disabled=not bool(secret.strip())):
                os.environ[selected] = secret.strip()
                st.success(f"{selected} configured in the current process.")
                st.rerun()
    with st.expander("Private API folder inventory", expanded=False):
        _render_table(api_files, "api_file_inventory", height=220)
    with st.expander("Credential master sections", expanded=False):
        _render_table(sections, "credential_sections", height=180)


def _render_analytics_tab(inventory: pd.DataFrame, summary: pd.DataFrame, providers: pd.DataFrame, health: pd.DataFrame, roots: dict[str, Path]) -> None:
    st.subheader("Data & API Analytics")
    enriched = enrich_inventory_for_export(inventory)
    target_catalog = build_target_catalog(roots["financial_db"])
    target_summary = summarize_target_catalog(target_catalog)
    if not target_summary.empty:
        st.markdown("### Strategic Coverage Heatmap")
        heat = target_summary[["domain", "coverage_pct", "targets", "available"]].copy()
        fig = px.imshow(
            [heat["coverage_pct"].tolist()],
            x=heat["domain"].tolist(),
            y=["coverage %"],
            color_continuous_scale=["#fee2e2", "#fef3c7", "#d1fae5"],
            zmin=0,
            zmax=100,
            text_auto=True,
            aspect="auto",
        )
        fig.update_layout(height=180, margin=dict(l=8, r=8, t=10, b=8))
        st.plotly_chart(fig, width="stretch")
    a1, a2 = st.columns(2)
    with a1:
        if not summary.empty:
            st.plotly_chart(px.pie(summary, values="files", names="role", template="plotly_white", hole=0.45), width="stretch")
    with a2:
        if not enriched.empty:
            provider_counts = enriched["provider"].value_counts().reset_index()
            provider_counts.columns = ["provider", "files"]
            st.plotly_chart(px.bar(provider_counts.head(20), x="provider", y="files", template="plotly_white"), width="stretch")
    b1, b2 = st.columns(2)
    with b1:
        if "age_hours" in enriched:
            st.plotly_chart(px.histogram(enriched, x="age_hours", nbins=40, template="plotly_white", title="Dataset Age Hours"), width="stretch")
    with b2:
        if not health.empty and "reachable" in health:
            health_counts = health["reachable"].fillna(False).astype(bool).value_counts().rename(index={True: "reachable", False: "watch"}).reset_index()
            health_counts.columns = ["status", "providers"]
            st.plotly_chart(px.bar(health_counts, x="status", y="providers", color="status", template="plotly_white"), width="stretch")
    st.markdown("### Batch Export History")
    _render_table(_latest_batch_runs(roots["workspace"], limit=20), "batch_history", height=260)
    st.markdown("### Usage Proxies")
    usage_proxy = enriched.sort_values(["age_hours", "size_mb"], ascending=[True, False]).head(25)
    _render_table(usage_proxy[["dataset_id", "provider", "role", "file", "size_mb", "age_hours", "relative_path"]], "usage_proxy", height=300)


def _render_settings_tab(
    roots: dict[str, Path],
    platform_roots: Any,
    api_roots: Any,
    config: dict[str, Any],
    max_files: int,
) -> None:
    st.subheader("Settings")
    st.markdown("### Paths")
    st.code(f"Financial DB: {platform_roots.financial_db}\nAPI root: {api_roots.api_root}\nWorkspace output: {roots['workspace']}", language=None)
    st.markdown("### Configuration")
    st.caption(str(CONFIG_PATH))
    st.json(config)

    st.markdown("### Contracts")
    c1, c2 = st.columns(2)
    if c1.button("Write Data Platform Contract", width="stretch"):
        paths = write_data_platform_status(platform_roots.financial_db, roots["workspace"], max_files=max_files)
        st.json({name: str(path) for name, path in paths.items()})
    if c2.button("Write API Control Contract", width="stretch"):
        paths = write_api_control_status(platform_roots.financial_db, roots["workspace"], api_roots.api_root)
        st.json({name: str(path) for name, path in paths.items()})

    st.markdown("### Incremental Refresh")
    r1, r2, r3 = st.columns(3)
    max_symbols = r1.number_input("Max symbols", min_value=1, max_value=250, value=25, step=1)
    stale_hours = r2.number_input("Stale hours", min_value=24, max_value=24 * 90, value=24 * 7, step=24)
    force = r3.checkbox("Force refresh", value=False)
    if st.button("Run stale-aware price refresh", width="stretch"):
        manifest = refresh_europe_stoxx_prices_incremental(
            platform_roots.financial_db,
            max_symbols=int(max_symbols),
            stale_hours=int(stale_hours),
            force=bool(force),
        )
        _render_table(manifest, "refresh_manifest", height=260)

    st.markdown("### Internal API Reference")
    st.code(
        "GET /api/datasets\n"
        "GET /api/datasets/{id}/download\n"
        "POST /api/sync\n"
        "GET /api/health\n"
        "GET /api/market-context\n",
        language="http",
    )
    st.caption("These routes are specified as the internal contract for the next API service layer; the current Streamlit app writes file contracts consumed by downstream services.")

    st.markdown("### Data Center Scripts")
    st.code(
        "python scripts/initial_setup.py --execute --max-items 5\n"
        "python scripts/sync_ohlcv_prices.py --execute --mode incremental --markets us_all,europe_major,japan_major,global_etfs --max-assets 500\n"
        "python scripts/sync_prices.py --execute --universes sp500,ftsemib --max-symbols 25\n"
        "python scripts/sync_fundamentals.py --execute --universes sp500,ftsemib --max-symbols 10\n"
        "python scripts/sync_official_macro.py --execute --build-features --start 2010-01\n"
        "python scripts/validate_data.py\n",
        language="bash",
    )


def render_data_api_control_center() -> None:
    config = load_data_api_config()
    theme = config.get("theme", {})
    configure_page(config.get("app", {}).get("title", "Data/API Control Center"))
    st.markdown(_theme_css(theme), unsafe_allow_html=True)

    roots = sidebar_roots()
    platform_roots = resolve_data_platform_roots(
        financial_db_root=roots["financial_db"],
        repo_output_root=roots["workspace"],
    )
    api_roots = resolve_api_control_roots(financial_db_root=platform_roots.financial_db)

    st.title("Data/API Control Center")
    st.caption("Drive-first data lake, API governance, batch export and platform health for machine-learning-for-trading.")

    default_limit = int(config.get("app", {}).get("default_inventory_limit", 5000))
    max_files = st.sidebar.slider("Inventory scan limit", min_value=500, max_value=20000, value=default_limit, step=500)

    data_status = dataset_status(platform_roots.financial_db, max_files=max_files)
    inventory = enrich_inventory_for_export(data_status.get("inventory", pd.DataFrame()))
    summary = data_status.get("summary", pd.DataFrame())
    api_status = api_control_status(platform_roots.financial_db, api_roots.api_root)
    providers = api_status.get("api_providers", pd.DataFrame())
    credentials = api_status.get("credential_status", pd.DataFrame())
    health = api_status.get("api_provider_health_checks", pd.DataFrame())
    api_files = api_status.get("api_folder_inventory", pd.DataFrame())
    sections = api_status.get("credential_master_sections", pd.DataFrame())
    fallback = provider_fallback_plan(platform_roots.financial_db)

    if not platform_roots.available:
        st.warning("Database Finanziario is not available. Mount Google Drive or set FINANCIAL_DB_ROOT.")

    tabs = st.tabs(["📊 Dashboard", "🗄️ Data Inventory", "🔌 API Registry", "🔑 Credentials", "📈 Analytics", "⚙️ Settings"])
    with tabs[0]:
        _render_dashboard(roots, platform_roots, api_roots, inventory, summary, providers, credentials, health, config, theme)
    with tabs[1]:
        _render_inventory_tab(inventory, roots, config)
    with tabs[2]:
        _render_api_registry_tab(providers, health, fallback)
    with tabs[3]:
        _render_credentials_tab(providers, credentials, api_files, sections)
    with tabs[4]:
        _render_analytics_tab(inventory, summary, providers, health, roots)
    with tabs[5]:
        _render_settings_tab(roots, platform_roots, api_roots, config, max_files)
