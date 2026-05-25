"""Read-only Streamlit adapters for notebook-generated research artifacts."""

from __future__ import annotations

import json
import os
import sys
from html import escape
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from research_platform_core import as_df, discover_financial_database_root, export_storage_env, resolve_storage_roots
except Exception:
    def as_df(value: Any) -> pd.DataFrame:
        if isinstance(value, pd.DataFrame):
            return value.copy()
        if isinstance(value, pd.Series):
            return value.to_frame().T
        if isinstance(value, list):
            return pd.DataFrame(value)
        if isinstance(value, dict):
            return pd.DataFrame([value])
        return pd.DataFrame()

    def discover_financial_database_root():
        candidates = [
            Path(os.getenv("FINANCIAL_DB_ROOT", "")) if os.getenv("FINANCIAL_DB_ROOT") else None,
            Path("/content/drive/MyDrive/Database Finanziario"),
            Path.home() / "Library/CloudStorage/GoogleDrive-sfn.gns@gmail.com/Il mio Drive/Database Finanziario",
        ]
        for candidate in [c for c in candidates if c is not None]:
            if candidate.exists():
                return candidate, "fallback_discovered", True
        return Path.cwd() / "Database Finanziario", "fallback_missing", False

    def resolve_storage_roots():
        financial_db, _, _ = discover_financial_database_root()
        project_root = Path(os.getenv("RESEARCH_PLATFORM_ROOT", PROJECT_ROOT)).expanduser()
        output_root = Path(os.getenv("RESEARCH_PLATFORM_OUTPUT_ROOT", project_root / "output")).expanduser()
        cache_root = Path(os.getenv("RESEARCH_PLATFORM_LOCAL_CACHE", output_root / "data_cache")).expanduser()
        class Roots:
            pass
        roots = Roots()
        roots.project_root = project_root
        roots.financial_db_root = financial_db
        roots.output_root = output_root
        roots.cache_root = cache_root
        roots.source = "fallback"
        roots.storage_mode = "drive" if "CloudStorage" in str(project_root) or "/content/drive/" in str(project_root) else "local"
        return roots

    def export_storage_env(roots):
        values = {
            "RESEARCH_PLATFORM_ROOT": str(roots.project_root),
            "FINANCIAL_DB_ROOT": str(roots.financial_db_root),
            "RESEARCH_PLATFORM_OUTPUT_ROOT": str(roots.output_root),
            "RESEARCH_PLATFORM_LOCAL_CACHE": str(roots.cache_root),
        }
        os.environ.update(values)
        return values


APP_TITLE = "Gen.is.IA Investment Research Workstation"
APP_VERSION = "v2"
DATA_SNAPSHOT_LABEL = "Data snapshot: 2000-2026 · factor v2 · ML v2"
FOOTER_TEXT = "Gen.is.IA Investment Research Workstation | TheGenesisAI"
DEFAULT_MARKET_CONTEXT = {
    "reporting_currency": "USD",
    "fx_pair": "DX-Y.NYB",
    "benchmark_ticker": "SPY",
    "macro_fx_tickers": "UUP, TLT, GLD, DBC",
    "selected_ticker": "",
    "active_screener_name": "",
}


def _safe_page_link(page: str, label: str) -> None:
    import streamlit as st

    try:
        st.page_link(page, label=label)
    except Exception:
        st.caption(label)


def safe_page_link(page: str, label: str) -> None:
    _safe_page_link(page, label)


def render_platform_sidebar() -> None:
    import streamlit as st

    st.sidebar.markdown("## Gen.is.IA")
    st.sidebar.caption("Investment Research Workstation")
    st.sidebar.caption(f"{APP_VERSION} · {DATA_SNAPSHOT_LABEL}")
    st.sidebar.markdown("---")

    st.sidebar.markdown("### RESEARCH")
    _safe_page_link("pages/0_🏠_Home.py", "Home / Overview")
    _safe_page_link("pages/4_🔍_Screener_Builder.py", "Screening & Research")
    _safe_page_link("pages/2_🔬_Valuation_Research.py", "Valuation")
    _safe_page_link("pages/3_📁_Portfolio_Research.py", "Portfolio")
    _safe_page_link("pages/13_🌍_Macro_View.py", "Macro View")
    _safe_page_link("pages/1_📡_Smart_Money_Macro.py", "Smart Money Intelligence")
    st.sidebar.markdown("---")

    st.sidebar.markdown("### LABS")
    _safe_page_link("pages/9_ML_Stock_Lab.py", "ML Stock Lab & Models")
    _safe_page_link("pages/14_⏱️_Time_Series_Lab.py", "Time Series Lab")
    _safe_page_link("pages/12_Banking_Data_Lab.py", "Banking Data Lab")
    _safe_page_link("pages/10_Laboratorio_Research_Library.py", "Research Library")
    st.sidebar.markdown("---")

    st.sidebar.markdown("### PLATFORM OPS")
    _safe_page_link("pages/8_🗄️_Data_Platform.py", "Data Platform")
    _safe_page_link("pages/6_🧪_Notebook_Runner.py", "Notebook Runner")
    _safe_page_link("pages/5_📤_Export_Center.py", "Export Center")
    _safe_page_link("pages/7_⚙️_Run_Hi_Freq_Engine.py", "Hi-Freq Engine")
    _safe_page_link("pages/11_Data_API_Control_Center.py", "Data/API Control Center")
    st.sidebar.markdown("---")


def render_footer() -> None:
    import streamlit as st

    st.divider()
    st.caption(FOOTER_TEXT)


def get_platform_context(state: dict[str, Any] | None = None) -> dict[str, str]:
    if state is None:
        import streamlit as st

        state = st.session_state
    context: dict[str, str] = {}
    for key, default in DEFAULT_MARKET_CONTEXT.items():
        value = state.get(key, default)
        value = "" if value is None else str(value).strip()
        context[key] = value or "not set"
    return context


def render_context_bar(context: dict[str, str] | None = None, show_actions: bool = True) -> None:
    import streamlit as st

    ctx = context or get_platform_context()

    def badge(label: str, key: str, tone: str = "neutral") -> str:
        value = ctx.get(key, "not set")
        return small_badge(label, value, "warn" if value == "not set" else tone)

    st.markdown(
        """
        <style>
        .rp-context-bar {
            display:flex;
            flex-wrap:wrap;
            gap:6px;
            align-items:center;
            padding:8px 0 12px 0;
        }
        .rp-context-title {
            font-size:12px;
            font-weight:750;
            color:#94a3b8;
            text-transform:uppercase;
            margin-right:3px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='rp-context-bar'>"
        "<span class='rp-context-title'>Context</span>"
        f"{badge('CCY', 'reporting_currency', 'ok')}"
        f"{badge('FX', 'fx_pair')}"
        f"{badge('Benchmark', 'benchmark_ticker')}"
        f"{badge('Overlays', 'macro_fx_tickers')}"
        f"{badge('Ticker', 'selected_ticker', 'ok')}"
        f"{badge('Screener', 'active_screener_name', 'ok')}"
        "</div>",
        unsafe_allow_html=True,
    )
    if show_actions:
        action_cols = st.columns([0.18, 0.18, 0.64])
        selected = ctx.get("selected_ticker", "not set")
        if action_cols[0].button(
            "Clear ticker",
            disabled=selected == "not set",
            help="Reset the shared selected ticker used by Valuation, Portfolio, ML and Smart Money pages.",
            width="stretch",
        ):
            st.session_state["selected_ticker"] = ""
            st.rerun()
        if action_cols[1].button(
            "Open Screener",
            help="Jump to the idea-generation workspace while keeping current context.",
            width="stretch",
        ):
            try:
                st.switch_page("pages/4_🔍_Screener_Builder.py")
            except Exception:
                st.info("Open the Screener Builder from the sidebar.")


def render_selected_ticker_context(
    roots: dict[str, Path],
    ticker: str | None = None,
    *,
    title: str = "Selected Ticker Context",
    expanded: bool = False,
) -> dict[str, Any] | None:
    """Render a compact shared ticker context panel.

    The lookup lives in research_platform_core; this function only translates it
    into a consistent Streamlit panel shared by Screener, ML, Valuation and
    Portfolio pages.
    """
    import streamlit as st

    selected = str(ticker or st.session_state.get("selected_ticker", "") or "").strip().upper()
    with st.container(border=True):
        st.markdown(f"**{title}**")
        if not selected:
            st.caption("No selected ticker yet. Select a row in Screener, ML Stock Lab, Valuation or Portfolio to share context across pages.")
            nav_cols = st.columns([0.25, 0.75])
            with nav_cols[0]:
                safe_page_link("pages/4_🔍_Screener_Builder.py", "Open Screener")
            return None

        try:
            from research_platform_core import get_ticker_context

            context = get_ticker_context(
                selected,
                financial_db_root=roots.get("financial_db"),
                output_root=roots.get("workspace"),
                company_root=roots.get("company"),
                portfolio_root=roots.get("portfolio"),
            )
        except Exception as exc:
            st.warning(f"Selected ticker context is temporarily unavailable for {selected}: {type(exc).__name__}.")
            return None

        info = context.get("basic_info", {}) or {}
        name = str(info.get("name") or "Name not available")
        sector = str(info.get("sector") or "Sector not set")
        universes = str(info.get("universes") or "Universe not set")
        coverage = str(info.get("coverage") or "UNKNOWN").upper()
        last_price_date = str(info.get("last_price_date") or "n/a")

        c1, c2, c3, c4 = st.columns([0.18, 0.34, 0.22, 0.26])
        c1.metric("Ticker", selected)
        c2.metric("Name", name[:42] + ("..." if len(name) > 42 else ""))
        c3.metric("Sector", sector[:26] + ("..." if len(sector) > 26 else ""))
        c4.metric("Coverage", coverage, f"last price {last_price_date}" if last_price_date != "n/a" else None)

        detail = f"Industry: {info.get('industry') or 'not set'} · Country: {info.get('country') or 'not set'} · Universes: {universes}"
        st.caption(detail)

        modules = context.get("modules")
        if isinstance(modules, pd.DataFrame) and not modules.empty:
            tone_for = {
                "OK": "ok",
                "RUNNING": "warn",
                "PARTIAL": "warn",
                "LIMITED_HISTORY": "warn",
                "PLANNED": "info",
                "NOT_TICKER_SPECIFIC": "info",
                "MISSING": "warn",
                "FAILED": "bad",
            }
            badges = []
            for row in modules.itertuples(index=False):
                module = escape(str(getattr(row, "module", "")))
                status = str(getattr(row, "status", "MISSING") or "MISSING").upper()
                badges.append(small_badge(module, escape(status), tone_for.get(status, "neutral")))
            st.markdown("".join(badges), unsafe_allow_html=True)
            with st.expander("Module availability details", expanded=expanded):
                show_cols = [col for col in ["module", "status", "rows", "detail", "path"] if col in modules.columns]
                st.dataframe(modules[show_cols], width="stretch", hide_index=True)

        try:
            from research_platform_core import compute_ticker_market_statistics

            benchmark = str(st.session_state.get("benchmark_ticker", "SPY") or "SPY").strip().upper()
            market_stats = compute_ticker_market_statistics(
                selected,
                benchmark=benchmark,
                financial_db_root=roots.get("financial_db"),
                output_root=roots.get("workspace"),
            )
        except Exception:
            market_stats = pd.DataFrame()
        if not market_stats.empty:
            latest_window = market_stats.sort_values("window_days").tail(1).iloc[0]

            def fmt_pct(value: Any) -> str:
                try:
                    return f"{float(value):.1%}"
                except Exception:
                    return "n/a"

            stat_cols = st.columns(4)
            stat_cols[0].metric("Volatility 252d", fmt_pct(latest_window.get("annualized_volatility")))
            stat_cols[1].metric("Variance 252d", fmt_pct(latest_window.get("annualized_variance")))
            stat_cols[2].metric(f"Beta vs {benchmark}", f"{float(latest_window.get('beta_to_benchmark')):.2f}" if pd.notna(latest_window.get("beta_to_benchmark")) else "n/a")
            stat_cols[3].metric(f"Corr vs {benchmark}", f"{float(latest_window.get('correlation_to_benchmark')):.2f}" if pd.notna(latest_window.get("correlation_to_benchmark")) else "n/a")
            with st.expander("Basic market statistics", expanded=False):
                st.caption("Computed from local OHLCV and the shared benchmark context; annualized volatility uses daily returns scaled by sqrt(252).")
                st.dataframe(market_stats, width="stretch", hide_index=True)

        action_cols = st.columns(4)
        if action_cols[0].button("Open Screener", key=f"{selected}_ctx_screener", width="stretch"):
            st.session_state["selected_ticker"] = selected
            st.switch_page("pages/4_🔍_Screener_Builder.py")
        if action_cols[1].button("Open ML Lab", key=f"{selected}_ctx_ml", width="stretch"):
            st.session_state["selected_ticker"] = selected
            st.switch_page("pages/9_ML_Stock_Lab.py")
        if action_cols[2].button("Open Valuation", key=f"{selected}_ctx_valuation", width="stretch"):
            st.session_state["selected_ticker"] = selected
            st.switch_page("pages/2_🔬_Valuation_Research.py")
        if action_cols[3].button("Open Portfolio", key=f"{selected}_ctx_portfolio", width="stretch"):
            st.session_state["selected_ticker"] = selected
            st.switch_page("pages/3_📁_Portfolio_Research.py")
        return context


def default_roots() -> dict[str, Path]:
    storage = resolve_storage_roots(project_root=PROJECT_ROOT)
    export_storage_env(storage)
    project_root = storage.project_root
    financial_db = storage.financial_db_root
    return {
        "company": Path(os.getenv("COMPANY_VALUATION_OUTPUT_ROOT", project_root / "company_valuation" / "output")),
        "portfolio": Path(os.getenv("PORTFOLIO_OUTPUT_ROOT", project_root / "portfolio_analysis" / "output")),
        "workspace": Path(os.getenv("RESEARCH_PLATFORM_OUTPUT_ROOT", project_root / "output")),
        "financial_db": Path(os.getenv("FINANCIAL_DB_ROOT", financial_db)),
    }


def configure_page(page_title: str) -> None:
    import streamlit as st

    try:
        st.set_page_config(page_title=f"{APP_TITLE} · {page_title}", layout="wide")
    except Exception:
        pass
    st.markdown(
        """
        <style>
        :root {
            --rp-ink:#111827;
            --rp-muted:#667085;
            --rp-border:#D9E2EC;
            --rp-panel:#FFFFFF;
            --rp-bg:#F6F8FB;
            --rp-primary:#006D77;
            --rp-accent:#2A9D8F;
            --rp-lab:#4C5FD7;
            --rp-ops:#6A4C93;
            --rp-macro:#0B6E99;
            --rp-smart:#7A5C00;
            --rp-warn:#B54708;
            --rp-bad:#B42318;
        }
        .stApp { background:var(--rp-bg); color:var(--rp-ink); }
        .block-container { padding-top: 1.35rem; padding-bottom: 2.25rem; max-width: 1480px; }
        h1 { letter-spacing:0; font-weight:760; margin-bottom:0.2rem; }
        h2, h3 { letter-spacing:0; font-weight:720; }
        div[data-testid="stMetric"] {
            background:var(--rp-panel);
            border:1px solid var(--rp-border);
            border-radius:8px;
            padding:13px 14px;
            box-shadow:0 1px 2px rgba(16,24,40,0.04);
        }
        div[data-testid="stMetric"] label { color:var(--rp-muted); font-weight:650; }
        div[data-testid="stMetricValue"] { color:var(--rp-ink); font-weight:760; }
        section[data-testid="stSidebar"] { border-right:1px solid var(--rp-border); }
        .stButton > button, .stDownloadButton > button {
            border-radius:7px;
            border:1px solid #B7CDD1;
            font-weight:650;
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            border-color:var(--rp-primary);
            color:var(--rp-primary);
        }
        .rp-card {
            background:var(--rp-panel);
            border:1px solid var(--rp-border);
            border-radius:8px;
            padding:15px 16px;
            box-shadow:0 1px 2px rgba(16,24,40,0.04);
            margin:0.35rem 0 0.85rem 0;
        }
        .rp-card-title {
            font-size:0.86rem;
            font-weight:760;
            color:var(--rp-ink);
            margin-bottom:0.2rem;
        }
        .rp-card-caption {
            font-size:0.78rem;
            color:var(--rp-muted);
            line-height:1.35;
        }
        .rp-section-kicker {
            font-size:0.72rem;
            color:var(--rp-primary);
            font-weight:760;
            text-transform:uppercase;
            margin-top:0.5rem;
        }
        .rp-note { background:#ECFDF3; border-left:5px solid var(--rp-primary); padding:12px 14px; border-radius:8px; color:#344054; }
        .rp-warn { background:#FFF7ED; border-left:5px solid #DA7101; padding:12px 14px; border-radius:8px; color:#344054; }
        .genisia-brand-strip {
            display:flex;
            justify-content:space-between;
            align-items:center;
            gap:10px;
            border:1px solid var(--rp-border);
            border-radius:8px;
            padding:10px 13px;
            background:#FFFFFF;
            box-shadow:0 1px 2px rgba(16,24,40,0.04);
            margin-bottom:0.9rem;
        }
        .genisia-logo {
            font-weight:820;
            color:var(--rp-primary);
            font-size:0.98rem;
        }
        .genisia-snapshot {
            color:var(--rp-muted);
            font-size:0.78rem;
            text-align:right;
        }
        .genisia-page-header {
            border:1px solid var(--rp-border);
            background:linear-gradient(180deg,#FFFFFF 0%,#F9FBFC 100%);
            border-radius:8px;
            padding:17px 18px;
            margin:0.25rem 0 0.8rem 0;
            box-shadow:0 1px 2px rgba(16,24,40,0.04);
        }
        .genisia-title-row {
            display:flex;
            justify-content:space-between;
            align-items:flex-start;
            gap:14px;
        }
        .genisia-title {
            font-size:2rem;
            font-weight:780;
            line-height:1.15;
            color:var(--rp-ink);
            margin:0;
        }
        .genisia-subtitle {
            max-width:880px;
            color:var(--rp-muted);
            line-height:1.45;
            margin-top:6px;
            font-size:0.94rem;
        }
        .genisia-module-pill {
            display:inline-block;
            border-radius:999px;
            padding:5px 9px;
            font-size:0.72rem;
            font-weight:760;
            text-transform:uppercase;
            background:#E6F4F7;
            color:var(--rp-primary);
            margin-left:5px;
            white-space:nowrap;
        }
        .genisia-status-ready { background:#ECFDF3; color:#067647; }
        .genisia-status-wip { background:#FFF4DF; color:#8A4B00; }
        .genisia-status-planned { background:#EEF2FF; color:#363F72; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_card(title: str, caption: str = "", body: str = "") -> None:
    import streamlit as st

    st.markdown(
        "<div class='rp-card'>"
        f"<div class='rp-card-title'>{title}</div>"
        f"<div class='rp-card-caption'>{caption}</div>"
        f"{body}"
        "</div>",
        unsafe_allow_html=True,
    )


def render_page_header(
    title: str,
    subtitle: str = "",
    icon: str = "",
    *,
    module: str = "RESEARCH",
    status: str = "READY",
) -> None:
    """Render the Gen.is.IA page header shared by all major pages."""
    import streamlit as st

    status_clean = str(status or "READY").upper()
    status_class = {
        "READY": "genisia-status-ready",
        "OK": "genisia-status-ready",
        "WIP": "genisia-status-wip",
        "PARTIAL": "genisia-status-wip",
        "PLANNED": "genisia-status-planned",
    }.get(status_clean, "genisia-status-wip")
    icon_html = f"{escape(icon)} " if icon else ""
    st.markdown(
        f"""
        <div class="genisia-brand-strip">
            <div>
                <span class="genisia-logo">Gen.is.IA</span>
                <span class="genisia-module-pill">{escape(module)}</span>
                <span class="genisia-module-pill {status_class}">{escape(status_clean)}</span>
            </div>
            <div class="genisia-snapshot">{escape(APP_VERSION)} · {escape(DATA_SNAPSHOT_LABEL)}</div>
        </div>
        <div class="genisia-page-header">
            <div class="genisia-title-row">
                <div>
                    <div class="genisia-title">{icon_html}{escape(title)}</div>
                    <div class="genisia-subtitle">{escape(subtitle)}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_page_intro(sentence: str, primary_action: str = "") -> None:
    import streamlit as st

    action = f"<br><b>Primary action:</b> {primary_action}" if primary_action else ""
    st.markdown(
        f"""
        <div class="rp-card">
            <div class="rp-card-title">Desk workflow</div>
            <div class="rp-card-caption">{sentence}{action}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_kicker(text: str) -> None:
    import streamlit as st

    st.markdown(f"<div class='rp-section-kicker'>{text}</div>", unsafe_allow_html=True)


def render_feature_metadata_expander(columns: Iterable[str], title: str = "Feature and factor glossary") -> None:
    import streamlit as st

    try:
        from research_platform_core import metadata_frame

        frame = metadata_frame(columns)
    except Exception:
        frame = pd.DataFrame()
    if frame.empty:
        return
    with st.expander(title, expanded=False):
        st.caption("Definitions used by Gen.is.IA UI labels and explainers. These are metadata only; they do not change model calculations.")
        show = [col for col in ["id", "name", "category", "description", "formula", "interpretation"] if col in frame.columns]
        st.dataframe(frame[show], width="stretch", hide_index=True)


def render_metric_metadata_expander(metrics: Iterable[str], title: str = "Metric glossary") -> None:
    import streamlit as st

    try:
        from research_platform_core import metrics_metadata_frame

        frame = metrics_metadata_frame(metrics)
    except Exception:
        frame = pd.DataFrame()
    if frame.empty:
        return
    with st.expander(title, expanded=False):
        st.caption("Metric definitions for model diagnostics, portfolio risk and validation panels.")
        show = [col for col in ["id", "name", "category", "definition", "formula", "interpretation"] if col in frame.columns]
        st.dataframe(frame[show], width="stretch", hide_index=True)


def render_workflow_steps(steps: list[tuple[str, str]], active_index: int = 0) -> None:
    """Render a compact fintech-style workflow bar."""
    import streamlit as st

    html = "<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:8px;margin:8px 0 14px 0'>"
    for idx, (label, detail) in enumerate(steps):
        active = idx == active_index
        border = "#01696f" if active else "#d9e2ec"
        bg = "#f0fbfc" if active else "#ffffff"
        html += (
            f"<div style='border:1px solid {border};background:{bg};border-radius:10px;padding:10px'>"
            f"<div style='font-size:12px;color:#667085'>Step {idx + 1}</div>"
            f"<div style='font-weight:750;color:#01696f'>{label}</div>"
            f"<div style='font-size:12px;color:#344054'>{detail}</div>"
            "</div>"
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def small_badge(label: str, value: str, tone: str = "neutral") -> str:
    colors = {
        "ok": ("#ECFDF3", "#067647"),
        "warn": ("#FFF4DF", "#8A4B00"),
        "bad": ("#FDECEC", "#8A1F11"),
        "neutral": ("#F6F8FB", "#344054"),
        "info": ("#E6F4F7", "#006D77"),
    }
    bg, color = colors.get(tone, colors["neutral"])
    return f"<span style='display:inline-block;margin:3px;padding:5px 8px;border-radius:999px;background:{bg};color:{color};font-weight:650'>{label}: {value}</span>"


def sidebar_roots() -> dict[str, Path]:
    import streamlit as st

    roots = default_roots()
    render_platform_sidebar()
    with st.sidebar:
        st.header("Artifact Roots")
        try:
            storage = resolve_storage_roots(project_root=PROJECT_ROOT)
            st.caption(f"Storage mode: {storage.storage_mode.upper()} · source: {storage.source}")
            st.code(str(storage.project_root), language=None)
        except Exception:
            st.caption("Notebooks generate the data. Streamlit reads the exported CSV/JSON/HTML files.")
        company = st.text_input("Company output root", str(roots["company"]))
        portfolio = st.text_input("Portfolio output root", str(roots["portfolio"]))
        workspace = st.text_input("Workspace output root", str(roots["workspace"]))
        financial_db = st.text_input("Financial DB root", str(roots["financial_db"]))
        st.divider()
        st.header("Market Context")
        st.caption("Shared app context for pages that need currency, FX and macro overlays.")
        currency_presets = {
            "USD": {"fx_pair": "DX-Y.NYB", "benchmark": "SPY", "macro": "UUP, TLT, GLD, DBC"},
            "EUR": {"fx_pair": "EURUSD=X", "benchmark": "SX5E.DE", "macro": "EURUSD=X, FEZ, EWI, GLD"},
            "GBP": {"fx_pair": "GBPUSD=X", "benchmark": "ISF.L", "macro": "GBPUSD=X, EWU, GLD"},
            "CHF": {"fx_pair": "CHF=X", "benchmark": "EWL", "macro": "CHF=X, EWL, GLD"},
            "JPY": {"fx_pair": "JPY=X", "benchmark": "EWJ", "macro": "JPY=X, EWJ, GLD"},
        }
        selected_currency = st.selectbox(
            "Reporting currency",
            list(currency_presets),
            index=list(currency_presets).index(st.session_state.get("reporting_currency", "USD"))
            if st.session_state.get("reporting_currency", "USD") in currency_presets
            else 0,
        )
        preset = currency_presets[selected_currency]
        st.session_state["reporting_currency"] = selected_currency
        st.session_state["fx_pair"] = st.text_input("FX proxy", st.session_state.get("fx_pair", preset["fx_pair"]))
        st.session_state["benchmark_ticker"] = st.text_input("Benchmark", st.session_state.get("benchmark_ticker", preset["benchmark"]))
        st.session_state["macro_fx_tickers"] = st.text_input("Macro / FX overlay tickers", st.session_state.get("macro_fx_tickers", preset["macro"]))
    return {
        "company": Path(company).expanduser(),
        "portfolio": Path(portfolio).expanduser(),
        "workspace": Path(workspace).expanduser(),
        "financial_db": Path(financial_db).expanduser(),
    }


def read_csv_any(root: Path, candidates: Iterable[str]) -> pd.DataFrame:
    for rel in candidates:
        path = root / rel
        if path.exists():
            try:
                return pd.read_csv(path)
            except Exception:
                continue
    return pd.DataFrame()


def read_json_any(root: Path, candidates: Iterable[str]) -> dict[str, Any]:
    for rel in candidates:
        path = root / rel
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
    return {}


def find_artifacts(roots: dict[str, Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    skip_parts = {"runs", "data_cache", "_cache", "__pycache__"}
    for domain, root in roots.items():
        if domain == "financial_db":
            continue
        if not root.exists():
            rows.append({"domain": domain, "type": "missing_root", "name": root.name, "path": str(root), "size_kb": 0.0})
            continue
        scanned = 0
        for path in root.rglob("*"):
            if scanned >= 5000:
                rows.append({"domain": domain, "type": "scan_limited", "name": "scan_limit_reached", "path": str(root), "size_kb": 0.0})
                break
            if any(part in skip_parts for part in path.parts):
                continue
            scanned += 1
            if path.is_file() and path.suffix.lower() in {".csv", ".json", ".html"}:
                rows.append({
                    "domain": domain,
                    "type": path.suffix.lower().lstrip("."),
                    "name": path.name,
                    "path": str(path),
                    "size_kb": round(path.stat().st_size / 1024, 2),
                    "modified": pd.Timestamp(path.stat().st_mtime, unit="s").isoformat(),
                })
    return pd.DataFrame(rows).sort_values(["domain", "type", "name"]).reset_index(drop=True) if rows else pd.DataFrame()


def friendly_domain(domain: str) -> str:
    labels = {
        "company": "Equity Valuation",
        "portfolio": "Portfolio Research",
        "workspace": "Platform Workspace",
        "financial_db": "Database Finanziario",
    }
    return labels.get(str(domain), str(domain).replace("_", " ").title())


def latest_modified_label(paths_or_df: Any) -> str:
    if isinstance(paths_or_df, pd.DataFrame):
        if paths_or_df.empty or "modified" not in paths_or_df.columns:
            return "n/a"
        values = pd.to_datetime(paths_or_df["modified"], errors="coerce").dropna()
        if values.empty:
            return "n/a"
        latest = values.max()
    else:
        root = Path(paths_or_df)
        if not root.exists():
            return "missing"
        # Database Finanziario can contain thousands of Drive-backed files. Keep
        # Overview responsive by sampling a bounded number of file mtimes.
        mtimes = []
        for idx, path in enumerate(root.rglob("*")):
            if idx >= 2000:
                break
            if path.is_file():
                try:
                    mtimes.append(path.stat().st_mtime)
                except OSError:
                    continue
        if not mtimes:
            return "empty"
        latest = pd.Timestamp(max(mtimes), unit="s")
    return latest.strftime("%Y-%m-%d %H:%M")


def status_counts_label(df: pd.DataFrame, column: str = "status") -> str:
    if df.empty or column not in df.columns:
        return "n/a"
    counts = df[column].astype(str).value_counts().to_dict()
    ok = counts.get("OK", 0)
    bad = sum(v for k, v in counts.items() if k != "OK")
    return f"{ok} OK / {bad} watch"


def load_ohlcv_coverage_manifest(roots: dict[str, Path]) -> pd.DataFrame:
    candidates = [
        roots.get("workspace", PROJECT_ROOT / "output") / "tables" / "OHLCV_daily_manifest.csv",
        roots.get("financial_db", PROJECT_ROOT) / "MarketData" / "OHLCV" / "manifests" / "ohlcv_daily_manifest.csv",
    ]
    for path in candidates:
        if path.exists():
            try:
                return pd.read_csv(path)
            except Exception:
                continue
    return pd.DataFrame()


def ohlcv_coverage_counts(manifest: pd.DataFrame) -> dict[str, int]:
    if manifest.empty:
        return {"OK": 0, "LIMITED_HISTORY": 0, "DELISTED": 0, "NETWORK_TIMEOUT": 0, "NO_PRICE_DATA": 0}
    manifest = manifest.copy()
    if "ticker" in manifest.columns:
        manifest = manifest[manifest["ticker"].fillna("").astype(str).str.len().gt(0)]
    dedupe_cols = [col for col in ["ticker", "provider_symbol", "coverage_status"] if col in manifest.columns]
    if dedupe_cols:
        manifest = manifest.drop_duplicates(dedupe_cols)
    column = "coverage_status" if "coverage_status" in manifest.columns else "status"
    counts = manifest[column].fillna("UNKNOWN").astype(str).str.upper().value_counts().to_dict()
    return {
        "OK": counts.get("OK", 0) + counts.get("DOWNLOADED", 0) + counts.get("FALLBACK", 0),
        "LIMITED_HISTORY": counts.get("LIMITED_HISTORY", 0),
        "DELISTED": counts.get("DELISTED", 0) + counts.get("DELISTED_NO_PRICE_DATA", 0),
        "NETWORK_TIMEOUT": counts.get("NETWORK_TIMEOUT", 0),
        "NO_PRICE_DATA": counts.get("NO_PRICE_DATA", 0) + counts.get("BULK_EMPTY", 0),
    }


def run_history_frame() -> pd.DataFrame:
    try:
        from research_platform_app.orchestration import JobStore
    except Exception:
        try:
            from orchestration import JobStore
        except Exception:
            return pd.DataFrame()
    try:
        runs = JobStore().list_runs()
        return pd.DataFrame([run.to_dict() for run in runs])
    except Exception:
        return pd.DataFrame()


def run_metrics(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {"last7": 0, "success_rate": "n/a", "top_job": "n/a", "latest_core": "n/a"}
    view = df.copy()
    created = pd.to_datetime(view.get("created_at"), errors="coerce", utc=True)
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=7)
    last7 = view[created >= cutoff] if not created.empty else view.iloc[0:0]
    status = view.get("status", pd.Series(dtype=str)).astype(str)
    success_rate = "n/a"
    if len(view):
        success_rate = f"{round(100 * status.eq('SUCCESS').sum() / len(view), 1)}%"
    top_job = "n/a"
    if "job_id" in view.columns and view["job_id"].notna().any():
        top_job = str(view["job_id"].value_counts().index[0])
    latest_core = "n/a"
    finished = pd.to_datetime(view.get("finished_at"), errors="coerce")
    if finished.notna().any():
        latest_core = finished.max().strftime("%Y-%m-%d %H:%M")
    return {"last7": len(last7), "success_rate": success_rate, "top_job": top_job, "latest_core": latest_core}


def used_by_for_artifact(name: str, domain: str = "") -> str:
    text = f"{name} {domain}".lower()
    if "screener" in text:
        return "Screeners, Valuation"
    if "smartmoney" in text or "smart_money" in text:
        return "Smart Money, Data Platform"
    if "mlstocklab" in text or "ml_stock_lab" in text:
        return "ML Lab, Valuation, Portfolio"
    if "portfolio" in text or "selection" in text or "allocation" in text:
        return "Portfolio, Screeners"
    if "valuation" in text or "fair" in text:
        return "Valuation"
    if "dataplatform" in text or "data_platform" in text:
        return "Data Platform, Home"
    if "dashboard" in text or name.lower().endswith(".html"):
        return "Artifacts, Demo"
    return "Artifacts"


def add_artifact_usage(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["used_by"] = [used_by_for_artifact(row.get("name", ""), row.get("domain", "")) for _, row in out.iterrows()]
    return out


def load_company_artifacts(root: Path) -> dict[str, Any]:
    return {
        "screener_results": read_csv_any(root, ["tables/ScreenerResults.csv", "tables/screener_results.csv"]),
        "screener_summary": read_csv_any(root, ["tables/ScreenerSummary.csv", "tables/screener_summary.csv"]),
        "screener_audit": read_csv_any(root, ["tables/ScreenerAudit.csv", "tables/screener_audit.csv"]),
        "screener_progression": read_csv_any(root, ["tables/ScreenerProgression.csv", "tables/screener_progression.csv"]),
        "screener_schema": read_csv_any(root, ["tables/ScreenerSchema.csv"]),
        "screener_presets": read_csv_any(root, ["tables/ScreenerPresets.csv"]),
        "finviz_groups": read_csv_any(root, ["tables/FinvizGroupSummaries.csv"]),
        "finviz_maps": read_csv_any(root, ["tables/FinvizMapPayload.csv"]),
        "finviz_watchlists": read_csv_any(root, ["tables/FinvizWatchlists.csv"]),
        "finviz_events": read_csv_any(root, ["tables/FinvizEvents.csv"]),
        "finviz_alerts": read_csv_any(root, ["tables/FinvizAlerts.csv"]),
        "finviz_manifest": read_csv_any(root, ["tables/FinvizExportManifest.csv"]),
        "valuation_gap": read_csv_any(root, ["tables/valuation_gap_table.csv", "tables/ValuationGapTable.csv"]),
        "valuation_models": read_csv_any(root, ["tables/valuation_model_registry.csv", "tables/ValuationModelRegistry.csv"]),
        "extended_valuation": read_csv_any(root, ["tables/extended_valuation_results.csv", "tables/ExtendedValuationResults.csv"]),
        "valuation_assumptions": read_csv_any(root, ["tables/valuation_assumption_table.csv"]),
        "ml_leaderboard": read_csv_any(root, ["tables/ml_leaderboard.csv"]),
        "ml_predictions": read_csv_any(root, ["tables/ml_predictions.csv"]),
        "refresh_provenance": read_csv_any(root, ["tables/refresh_provenance.csv"]),
        "provider_registry": read_csv_any(root, ["tables/provider_registry.csv"]),
        "dcf_mc_summary": read_csv_any(root, ["tables/DCFMonteCarloSummary.csv"]),
        "dcf_mc_simulations": read_csv_any(root, ["tables/DCFMonteCarloSimulations.csv"]),
        "dcf_mc_scenarios": read_csv_any(root, ["tables/DCFMonteCarloScenarios.csv"]),
        "dcf_mc_assumptions": read_csv_any(root, ["tables/DCFMonteCarloAssumptions.csv"]),
        "dcf_model_factors": read_csv_any(root, ["tables/DCFModelBasedFactors.csv"]),
        "dcf_market_parameters": read_csv_any(root, ["tables/DCFMarketParameters.csv"]),
        "ri_mc_summary": read_csv_any(root, ["tables/ResidualIncomeMonteCarloSummary.csv"]),
        "ri_mc_simulations": read_csv_any(root, ["tables/ResidualIncomeMonteCarloSimulations.csv"]),
        "ri_mc_scenarios": read_csv_any(root, ["tables/ResidualIncomeMonteCarloScenarios.csv"]),
        "ri_mc_assumptions": read_csv_any(root, ["tables/ResidualIncomeMonteCarloAssumptions.csv"]),
        "ri_model_factors": read_csv_any(root, ["tables/ResidualIncomeModelBasedFactors.csv"]),
        "eva_scenario_bands": read_csv_any(root, ["tables/EVAScenarioBands.csv"]),
        "eva_scenario_summary": read_csv_any(root, ["tables/EVAScenarioSummary.csv"]),
        "eva_scenario_assumptions": read_csv_any(root, ["tables/EVAScenarioAssumptions.csv"]),
        "eva_model_factors": read_csv_any(root, ["tables/EVAModelBasedFactors.csv"]),
    }


def load_portfolio_artifacts(root: Path) -> dict[str, Any]:
    return {
        "selection_results": read_csv_any(root, ["tables/PortfolioSelectionResults.csv"]),
        "selection_summary": read_csv_any(root, ["tables/PortfolioSelectionSummary.csv"]),
        "selection_audit": read_csv_any(root, ["tables/PortfolioSelectionAudit.csv"]),
        "selection_progression": read_csv_any(root, ["tables/PortfolioSelectionProgression.csv"]),
        "selection_schema": read_csv_any(root, ["tables/PortfolioSelectionSchema.csv"]),
        "selection_presets": read_csv_any(root, ["tables/PortfolioSelectionPresets.csv"]),
        "allocation": read_csv_any(root, ["tables/portfolio_allocation.csv", "tables/PortfolioAllocation.csv"]),
        "performance": read_csv_any(root, ["tables/performance_summary.csv", "tables/backtest_performance.csv"]),
        "risk": read_csv_any(root, ["tables/risk_dashboard.csv"]),
        "scenarios": read_csv_any(root, ["tables/portfolio_scenarios.csv"]),
        "engine_weights": read_csv_any(root, ["tables/portfolio_engine_weights.csv"]),
        "engine_metrics": read_csv_any(root, ["tables/portfolio_engine_metrics.csv"]),
        "diagnostics": read_csv_any(root, ["tables/diagnostics.csv"]),
    }


def load_smart_money_artifacts(root: Path) -> dict[str, Any]:
    smart_root = root / "smart_money"
    return {
        "scores": read_csv_any(smart_root, ["tables/SmartMoney_smart_money_scores.csv"]),
        "events": read_csv_any(smart_root, ["tables/SmartMoney_event_feed.csv"]),
        "coverage": read_csv_any(smart_root, ["tables/SmartMoney_coverage.csv"]),
        "source_registry": read_csv_any(smart_root, ["tables/SmartMoney_source_registry.csv"]),
        "entity_master": read_csv_any(smart_root, ["tables/SmartMoney_entity_master.csv"]),
        "holdings": read_csv_any(smart_root, ["tables/SmartMoney_holdings.csv"]),
        "insiders": read_csv_any(smart_root, ["tables/SmartMoney_insiders.csv"]),
        "beneficial_events": read_csv_any(smart_root, ["tables/SmartMoney_beneficial_events.csv"]),
        "macro_positioning": read_csv_any(smart_root, ["tables/SmartMoney_macro_positioning.csv"]),
        "capital_flows": read_csv_any(smart_root, ["tables/SmartMoney_capital_flows.csv"]),
        "cot_history": read_csv_any(smart_root, ["tables/SmartMoney_COT_history.csv"]),
        "cot_snapshot": read_csv_any(smart_root, ["tables/SmartMoney_COT_snapshot.csv"]),
        "cot_history_sample": read_csv_any(smart_root, ["tables/SmartMoney_COT_history_sample.csv"]),
        "government_spending": read_csv_any(smart_root, ["tables/SmartMoney_government_spending_matched.csv", "tables/SmartMoney_government_spending.csv"]),
        "sector_monitor": read_csv_any(smart_root, ["tables/SmartMoney_sector_monitor.csv"]),
        "government_dataset_matrix": read_csv_any(smart_root, ["tables/SmartMoney_government_dataset_matrix.csv"]),
        "open_source_pattern_matrix": read_csv_any(smart_root, ["tables/SmartMoney_open_source_pattern_matrix.csv"]),
        "mvp_30_day_plan": read_csv_any(smart_root, ["tables/SmartMoney_mvp_30_day_plan.csv"]),
        "dataset_priority_ranking": read_csv_any(smart_root, ["tables/SmartMoney_dataset_priority_ranking.csv"]),
        "analyst_quick_wins": read_csv_any(smart_root, ["tables/SmartMoney_analyst_quick_wins.csv"]),
        "manifest": read_json_any(smart_root, ["SmartMoneyManifest.json"]),
    }


def load_ml_stock_lab_artifacts(root: Path) -> dict[str, Any]:
    lab_root = root / "ml_stock_lab"
    training_root = root / "ml_training_lab"
    return {
        "panel": read_csv_any(lab_root, ["tables/MLStockLab_panel.csv"]),
        "signals": read_csv_any(lab_root, ["tables/MLStockLab_signals.csv"]),
        "top": read_csv_any(lab_root, ["tables/MLStockLab_top.csv"]),
        "bottom": read_csv_any(lab_root, ["tables/MLStockLab_bottom.csv"]),
        "quintile_returns": read_csv_any(lab_root, ["tables/MLStockLab_quintile_returns.csv"]),
        "quintile_metrics": read_csv_any(lab_root, ["tables/MLStockLab_quintile_metrics.csv"]),
        "metrics": read_csv_any(lab_root, ["tables/MLStockLab_metrics.csv"]),
        "model_comparison": read_csv_any(lab_root, ["tables/MLStockLab_model_comparison.csv"]),
        "prediction_metrics": read_csv_any(lab_root, ["tables/MLStockLab_prediction_metrics.csv"]),
        "long_short_diagnostics": read_csv_any(lab_root, ["tables/MLStockLab_long_short_diagnostics.csv"]),
        "status": read_csv_any(lab_root, ["tables/MLStockLab_status.csv"]),
        "experiment_card": read_csv_any(lab_root, ["tables/MLStockLab_experiment_card.csv"]),
        "training_metrics": read_csv_any(training_root, ["tables/MLTraining_metrics.csv"]),
        "training_predictions": read_csv_any(training_root, ["tables/MLTraining_predictions_wide.csv", "tables/MLTraining_predictions.csv"]),
        "training_feature_importance": read_csv_any(training_root, ["tables/MLTraining_feature_importance.csv"]),
        "training_model_cards": read_csv_any(training_root, ["tables/MLTraining_model_cards.csv"]),
    }


def metric_value(df: pd.DataFrame, columns: Iterable[str], default: Any = "n/a") -> Any:
    if df.empty:
        return default
    for col in columns:
        if col in df.columns and df[col].notna().any():
            return df[col].dropna().iloc[0]
    return default


def numeric_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def show_empty(label: str, root: Path | None = None) -> None:
    import streamlit as st

    suffix = f" Current root: `{root}`." if root is not None else ""
    st.info(f"{label} is not available yet. Run the corresponding notebook export cells first.{suffix}")


def dataframe_with_download(label: str, df: pd.DataFrame, file_name: str) -> None:
    import streamlit as st

    if df.empty:
        show_empty(label)
        return
    st.caption(f"{label}: {len(df):,} rows · {len(df.columns):,} columns")
    st.dataframe(df, width="stretch", hide_index=True)
    st.download_button(
        f"Download {label}",
        df.to_csv(index=False),
        file_name=file_name,
        mime="text/csv",
        width="content",
    )
