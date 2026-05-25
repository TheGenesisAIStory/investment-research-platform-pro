from __future__ import annotations

import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

from data_bootstrap import render_bootstrap_banner
from operations import generate_all_artifacts, generate_company_artifacts, generate_portfolio_artifacts
from orchestration import build_freshness_table
from orchestration.scheduler import scheduler_tick
from orchestration.scheduler_process import scheduler_status, start_scheduler_process, stop_scheduler_process
from support import (
    FOOTER_TEXT,
    add_artifact_usage,
    find_artifacts,
    friendly_domain,
    latest_modified_label,
    load_company_artifacts,
    load_ml_stock_lab_artifacts,
    load_ohlcv_coverage_manifest,
    load_portfolio_artifacts,
    load_smart_money_artifacts,
    metric_value,
    ohlcv_coverage_counts,
    render_context_bar,
    render_footer,
    render_page_intro,
    render_section_kicker,
    render_selected_ticker_context,
    run_history_frame,
    run_metrics,
    safe_page_link,
    sidebar_roots,
    status_counts_label,
)

from research_platform_core.data_explorer import list_available_tickers
from research_platform_core.data_health import get_stage_health_for_universes


@st.cache_data(ttl=300, show_spinner=False)
def load_market_snapshot() -> dict[str, object]:
    symbols = {"^GSPC": "S&P 500", "^VIX": "VIX"}
    frames: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        try:
            frame = yf.Ticker(symbol).history(period="35d", interval="1d", auto_adjust=False)
            frames[symbol] = frame.dropna(subset=["Close"]) if not frame.empty else pd.DataFrame()
        except Exception:
            frames[symbol] = pd.DataFrame()

    spx = frames.get("^GSPC", pd.DataFrame())
    vix = frames.get("^VIX", pd.DataFrame())
    sentiment = "Waiting for data"
    delta = "n/a"
    if len(spx) >= 2:
        spx_delta = float(spx["Close"].iloc[-1] - spx["Close"].iloc[-2])
        spx_delta_pct = spx_delta / float(spx["Close"].iloc[-2])
        vix_last = float(vix["Close"].iloc[-1]) if not vix.empty else 20.0
        if spx_delta_pct > 0 and vix_last < 20:
            sentiment = "Risk-on"
        elif spx_delta_pct < 0 or vix_last > 25:
            sentiment = "Risk-off"
        else:
            sentiment = "Neutral"
        delta = f"{spx_delta_pct:.2%} vs prev close"
    return {"sentiment": sentiment, "delta": delta, "spx": spx}


@st.cache_data(ttl=300, show_spinner=False)
def _cached_stage_health(financial_db_root: str, output_root: str) -> pd.DataFrame:
    rows = [
        get_stage_health_for_universes("equity_fundamentals", [], financial_db_root, output_root),
        get_stage_health_for_universes("equity_prices", [], financial_db_root, output_root),
    ]
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def _cached_available_tickers(financial_db_root: str, output_root: str) -> pd.DataFrame:
    return list_available_tickers(financial_db_root, output_root, limit=25000)


@st.cache_data(ttl=300, show_spinner=False)
def _cached_factor_snapshot(output_root: str) -> dict[str, Any]:
    path = Path(output_root) / "ml_training_lab" / "tables" / "FactorUniversePanel.csv"
    if not path.exists() or path.stat().st_size <= 1:
        return {"rows": 0, "tickers": 0, "latest_date": "n/a", "status": "MISSING"}
    rows = 0
    tickers: set[str] = set()
    latest_date = ""
    try:
        header = pd.read_csv(path, nrows=0).columns.tolist()
        usecols = [col for col in ["ticker", "date"] if col in header]
        if not usecols:
            return {"rows": 0, "tickers": 0, "latest_date": "n/a", "status": "MISSING"}
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=100_000):
            rows += len(chunk)
            if "ticker" in chunk.columns:
                tickers.update(chunk["ticker"].dropna().astype(str).str.upper().unique().tolist())
            if "date" in chunk.columns:
                dates = pd.to_datetime(chunk["date"], errors="coerce")
                if dates.notna().any():
                    candidate = dates.max().strftime("%Y-%m-%d")
                    latest_date = max(latest_date, candidate)
    except Exception:
        return {"rows": rows, "tickers": len(tickers), "latest_date": latest_date or "n/a", "status": "PARTIAL"}
    return {"rows": rows, "tickers": len(tickers), "latest_date": latest_date or "n/a", "status": "OK" if rows else "MISSING"}


@st.cache_data(ttl=60, show_spinner=False)
def _cached_ollama_status() -> dict[str, str]:
    base_url = (os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=1.5) as response:
            if 200 <= int(response.status) < 300:
                return {"status": "AVAILABLE", "detail": base_url}
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return {"status": "UNAVAILABLE", "detail": f"{base_url} · {type(exc).__name__}"}
    return {"status": "UNAVAILABLE", "detail": base_url}


def _stage_status(health: pd.DataFrame, stage: str) -> str:
    if health.empty or "stage" not in health.columns or "status" not in health.columns:
        return "MISSING"
    values = health[health["stage"].astype(str).eq(stage)]["status"].astype(str).str.upper().tolist()
    return values[0] if values else "MISSING"


def _workflow_card(
    icon: str,
    title: str,
    status: str,
    description: str,
    page: str,
    secondary_page: str | None = None,
    secondary_label: str = "Vai ai dati",
) -> None:
    status = str(status or "UNKNOWN").upper()
    if status in {"OK", "READY", "AVAILABLE"}:
        label = "READY"
        color = "#067647"
        bg = "#ECFDF3"
    elif status in {"PARTIAL", "RUNNING", "WIP"}:
        label = status
        color = "#8A4B00"
        bg = "#FFF4DF"
    elif status in {"PLANNED", "COMING SOON"}:
        label = "PLANNED"
        color = "#006D77"
        bg = "#E6F4F7"
    else:
        label = status
        color = "#8A1F11"
        bg = "#FDECEC"

    with st.container(border=True):
        st.markdown(f"### {icon} {title}")
        st.markdown(
            f"<span style='background:{bg};color:{color};border-radius:999px;padding:4px 9px;font-weight:700;font-size:12px'>{label}</span>",
            unsafe_allow_html=True,
        )
        st.caption(description)
        c1, c2 = st.columns(2)
        if c1.button("Apri modulo", key=f"open_{title}", width="stretch"):
            st.switch_page(page)
        with c2:
            if secondary_page:
                safe_page_link(secondary_page, secondary_label)


def _model_status_label(ml_lab: dict[str, pd.DataFrame]) -> tuple[str, str]:
    metrics = ml_lab.get("training_metrics", pd.DataFrame())
    cards = ml_lab.get("training_model_cards", pd.DataFrame())
    if not metrics.empty and "model" in metrics.columns:
        ok = metrics[metrics.get("status", pd.Series("", index=metrics.index)).astype(str).str.upper().eq("OK")] if "status" in metrics.columns else metrics
        models = ", ".join(ok["model"].dropna().astype(str).unique().tolist()[:5])
        if not cards.empty and {"training_window", "test_window"}.issubset(cards.columns):
            row = cards.iloc[0]
            return "OK", f"{models or len(metrics)} · train {row.get('training_window')} · test {row.get('test_window')}"
        return "OK", models or f"{len(metrics)} model rows"
    signals = ml_lab.get("signals", pd.DataFrame())
    if not signals.empty:
        return "PARTIAL", f"{len(signals):,} signal rows; training cards missing"
    return "MISSING", "no ML training artifacts found"


def _render_start_here() -> None:
    render_section_kicker("Start here")
    with st.container(border=True):
        st.markdown("**Choose the research path**")
        st.caption("These actions set shared context when useful and jump to the right workstation page.")
        actions = st.columns(5)
        if actions[0].button("Analizza un singolo ticker", width="stretch", help="Use the ticker search below and keep the selected ticker across modules."):
            st.session_state["home_focus"] = "ticker"
            st.session_state["selected_ticker"] = ""
            st.rerun()
        if actions[1].button("Costruisci uno screener", width="stretch"):
            st.session_state["active_screener_name"] = "Custom"
            st.switch_page("pages/4_🔍_Screener_Builder.py")
        if actions[2].button("Esplora segnali ML", width="stretch"):
            st.switch_page("pages/9_ML_Stock_Lab.py")
        if actions[3].button("Valuta una company", width="stretch"):
            st.switch_page("pages/2_🔬_Valuation_Research.py")
        if actions[4].button("Controlla stato dati", width="stretch"):
            st.switch_page("pages/8_🗄️_Data_Platform.py")


def _render_ticker_search(roots: dict[str, Path]) -> None:
    render_section_kicker("Cerca un ticker")
    with st.container(border=True):
        st.markdown("**Single ticker shortcut**")
        st.caption("Search uses Data Platform manifests and ML/factor artifacts. The selected ticker becomes shared app context.")
        tickers = _cached_available_tickers(str(roots["financial_db"]), str(roots["workspace"]))
        current = str(st.session_state.get("selected_ticker", "") or "").strip().upper()
        options = tickers["ticker"].dropna().astype(str).str.upper().drop_duplicates().sort_values().tolist() if not tickers.empty and "ticker" in tickers.columns else []
        if current and current not in options:
            options = [current, *options]
        selected = ""
        if options:
            index = options.index(current) if current in options else 0
            selected = st.selectbox("Ticker nel database", options, index=index, help="Type to search tickers already present in coverage/model manifests.")
        manual = st.text_input("Oppure inserisci un ticker manuale", value="" if selected else current, placeholder="AAPL, MSFT, ENEL.MI...")
        chosen = str(manual or selected or "").strip().upper()
        c1, c2 = st.columns([0.22, 0.78])
        if c1.button("Usa ticker", disabled=not bool(chosen), width="stretch"):
            st.session_state["selected_ticker"] = chosen
            st.rerun()
        if chosen and chosen != current:
            c2.info(f"Ticker pronto: {chosen}. Premi `Usa ticker` per renderlo il contesto condiviso.")
        render_selected_ticker_context(roots, st.session_state.get("selected_ticker", ""), expanded=False)


def _render_platform_status(
    roots: dict[str, Path],
    ml_lab: dict[str, pd.DataFrame],
    ohlcv_counts: dict[str, int],
) -> None:
    render_section_kicker("Platform status")
    health = _cached_stage_health(str(roots["financial_db"]), str(roots["workspace"]))
    factor = _cached_factor_snapshot(str(roots["workspace"]))
    ollama = _cached_ollama_status()
    model_status, model_detail = _model_status_label(ml_lab)

    with st.container(border=True):
        cols = st.columns(6)
        cols[0].metric("Fundamentals", _stage_status(health, "equity_fundamentals"))
        cols[1].metric("Prices", _stage_status(health, "equity_prices"), f"{ohlcv_counts.get('LIMITED_HISTORY', 0):,} limited")
        cols[2].metric("Factor Panel", factor["status"], f"{factor['tickers']:,} tickers")
        cols[3].metric("ML Models", model_status, model_detail[:42] + ("..." if len(model_detail) > 42 else ""))
        cols[4].metric("Ollama", ollama["status"])
        cols[5].metric("Data Issues", ohlcv_counts.get("NETWORK_TIMEOUT", 0), "network timeouts")
        st.caption(
            f"Factor rows: {factor['rows']:,} · latest factor date: {factor['latest_date']} · Ollama endpoint: {ollama['detail']}. "
            "Open Data Platform for full failure breakdown and restart controls."
        )


def _render_workflows(company: dict[str, pd.DataFrame], portfolio: dict[str, pd.DataFrame], smart_money: dict[str, pd.DataFrame], ml_lab: dict[str, pd.DataFrame], factor_status: str) -> None:
    render_section_kicker("Workflows")
    valuation_status = "READY" if not company.get("screener_results", pd.DataFrame()).empty else "PARTIAL"
    portfolio_status = "READY" if not (portfolio.get("allocation", pd.DataFrame()).empty and portfolio.get("selection_results", pd.DataFrame()).empty) else "PARTIAL"
    ml_status, _ = _model_status_label(ml_lab)
    smart_status = "WIP" if not smart_money.get("scores", pd.DataFrame()).empty or not smart_money.get("events", pd.DataFrame()).empty else "PLANNED"

    row1 = st.columns(3)
    with row1[0]:
        _workflow_card(
            "🗄️",
            "Data Platform & Health",
            "READY",
            "Explore ticker-level data, check coverage, inspect failures and restart backfills from a human-friendly console.",
            "pages/8_🗄️_Data_Platform.py",
        )
    with row1[1]:
        _workflow_card(
            "🔍",
            "Factor & Screener",
            "READY" if factor_status == "OK" else "PARTIAL",
            "Build factor/ML/valuation screeners over global universes and carry selected names into analysis modules.",
            "pages/4_🔍_Screener_Builder.py",
            "pages/8_🗄️_Data_Platform.py",
        )
    with row1[2]:
        _workflow_card(
            "🧠",
            "ML Stock Lab",
            ml_status,
            "Compare OLS/RF/GBRT/ensemble models, inspect factor blocks and use Ollama for model governance notes.",
            "pages/9_ML_Stock_Lab.py",
            "pages/8_🗄️_Data_Platform.py",
        )

    row2 = st.columns(3)
    with row2[0]:
        _workflow_card(
            "🔬",
            "Valuation",
            valuation_status,
            "Open fair-value, DCF/multiples and valuation reasoning for the selected company or a broader candidate set.",
            "pages/2_🔬_Valuation_Research.py",
        )
    with row2[1]:
        _workflow_card(
            "📁",
            "Portfolio",
            portfolio_status,
            "Review allocation, selection scores, risk diagnostics and factor/ML overlays for portfolio construction.",
            "pages/3_📁_Portfolio_Research.py",
        )
    with row2[2]:
        _workflow_card(
            "📡",
            "Smart Money / FX / Macro",
            smart_status,
            "Issuer events, macro overlays and FX context are available in partial form; FX/macro ticker linkage remains roadmap.",
            "pages/1_📡_Smart_Money_Macro.py",
            "pages/8_🗄️_Data_Platform.py",
        )


def _render_advanced_ops(roots: dict[str, Path], artifacts: pd.DataFrame, freshness: pd.DataFrame, runs: pd.DataFrame) -> None:
    with st.expander("Advanced platform contracts", expanded=False):
        st.markdown("Use this when you need roots, artifact contracts, freshness and recent runs. Daily research starts above.")
        root_cols = st.columns(4)
        for col, domain in zip(root_cols, ["company", "portfolio", "workspace", "financial_db"]):
            root = roots[domain]
            domain_rows = artifacts[artifacts["domain"].eq(domain)] if not artifacts.empty and "domain" in artifacts.columns else pd.DataFrame()
            col.markdown(f"**{friendly_domain(domain)}**")
            freshness_label = ("available" if root.exists() else "missing") if domain == "financial_db" else latest_modified_label(domain_rows)
            file_count = "shared" if domain == "financial_db" else len(domain_rows)
            col.metric("Files", file_count, freshness_label)
            with col.expander("Path", expanded=False):
                st.code(str(root))

        tab_artifacts, tab_freshness, tab_runs = st.tabs(["Artifact Availability", "Freshness", "Recent Runs"])
        with tab_artifacts:
            if artifacts.empty:
                st.info("No exported artifacts found yet. Run the company valuation and/or portfolio notebooks, then refresh this app.")
            else:
                st.dataframe(artifacts.head(200), width="stretch", hide_index=True)
        with tab_freshness:
            if freshness.empty:
                st.info("No artifact contracts found.")
            else:
                status_counts = freshness["status"].value_counts().to_dict() if "status" in freshness.columns else {}
                fcols = st.columns(4)
                for col, status_name in zip(fcols, ["OK", "STALE", "MISSING_REQUIRED", "MISSING_OPTIONAL"]):
                    col.metric(status_name, status_counts.get(status_name, 0))
                st.dataframe(freshness, width="stretch", hide_index=True)
        with tab_runs:
            if runs.empty:
                st.info("No run history yet. Launch a job from Notebook Runner or Data Platform.")
            else:
                cols = ["run_id", "job_id", "status", "created_at", "finished_at", "runner_type", "error_message"]
                st.dataframe(runs[[c for c in cols if c in runs.columns]].head(20), width="stretch", hide_index=True)


def _render_lightweight_generation(roots: dict[str, Path]) -> None:
    with st.expander("Generate missing lightweight artifacts", expanded=False):
        st.markdown(
            """
            These actions rebuild low-cost screener/selection/platform layers from existing notebook exports.
            They do not execute full notebooks or trigger heavy provider refreshes.
            """
        )
        c1, c2, c3 = st.columns(3)
        if c1.button("Generate Company Layer", width="stretch"):
            result = generate_company_artifacts(roots["company"])
            (st.success if result["ok"] else st.warning)(result["message"])
            if result.get("details"):
                st.json(result["details"])
        if c2.button("Generate Portfolio Selection", width="stretch"):
            result = generate_portfolio_artifacts(roots["portfolio"], roots["workspace"])
            (st.success if result["ok"] else st.warning)(result["message"])
            if result.get("details"):
                st.json(result["details"])
        if c3.button("Generate All Lightweight Layers", width="stretch"):
            results = generate_all_artifacts(roots)
            for name, result in results.items():
                (st.success if result["ok"] else st.warning)(f"{name.title()}: {result['message']}")
                if result.get("details"):
                    st.json(result["details"])


def _render_scheduler(roots: dict[str, Path]) -> None:
    with st.expander("Safe scheduler and maintenance", expanded=False):
        st.markdown(
            """
            Safe scheduler ticks refresh Data Platform status and lightweight screener/selection artifacts when
            required outputs are missing or stale. Full notebook jobs should run from Notebook Runner or Data Platform.
            """
        )
        if st.button("Run safe freshness tick now", width="stretch"):
            st.json(scheduler_tick(roots, ["data_platform_status_refresh", "screener_refresh"]))
        status = scheduler_status()
        st.write(f"Scheduler status: `{'RUNNING' if status.get('alive') else 'STOPPED'}`")
        scol1, scol2 = st.columns(2)
        if scol1.button("Start safe local scheduler", width="stretch"):
            st.json(start_scheduler_process(roots, ["data_platform_status_refresh", "screener_refresh"], interval_seconds=900))
        if scol2.button("Stop local scheduler", width="stretch"):
            st.json(stop_scheduler_process())
        st.code("python research_platform_app/scheduler.py --interval-seconds 900 --jobs data_platform_status_refresh screener_refresh")


def render_home_command_center() -> None:
    roots = sidebar_roots()
    artifacts = add_artifact_usage(find_artifacts(roots))
    freshness = build_freshness_table(roots)
    runs = run_history_frame()
    metrics = run_metrics(runs)
    company = load_company_artifacts(roots["company"])
    portfolio = load_portfolio_artifacts(roots["portfolio"])
    smart_money = load_smart_money_artifacts(roots["workspace"])
    ml_lab = load_ml_stock_lab_artifacts(roots["workspace"])
    ohlcv_manifest = load_ohlcv_coverage_manifest(roots)
    ohlcv_counts = ohlcv_coverage_counts(ohlcv_manifest)
    market = load_market_snapshot()
    factor = _cached_factor_snapshot(str(roots["workspace"]))

    st.title("Investment Research Command Center")
    st.caption("TheGenesisAI research workstation: data, factors, ML, valuation, portfolio construction and LLM-assisted governance.")
    render_context_bar()
    render_page_intro(
        "Choose a ticker, workflow or health check from one screen. Advanced jobs and artifact contracts stay available, but no longer block daily research.",
        "Search a ticker or start with Screener, ML Stock Lab, Valuation, Portfolio or Data Platform.",
    )
    render_bootstrap_banner(roots, required=["equity_metadata", "company_screener", "ml_signals", "smart_money_scores"])

    _render_start_here()
    _render_ticker_search(roots)
    _render_platform_status(roots, ml_lab, ohlcv_counts)

    top_cols = st.columns(4)
    top_cols[0].metric("Market Sentiment", str(market["sentiment"]), str(market["delta"]))
    top_cols[1].metric("Runs · 7d", metrics["last7"])
    top_cols[2].metric("Artifact Contracts", status_counts_label(freshness))
    top_cols[3].metric("ML Signals", len(ml_lab["signals"]))

    pulse_left, pulse_right = st.columns([0.58, 0.42])
    with pulse_left:
        render_section_kicker("Market pulse")
        spx = market["spx"]
        if isinstance(spx, pd.DataFrame) and not spx.empty:
            chart = spx.tail(30).reset_index()
            st.plotly_chart(
                px.line(chart, x=chart.columns[0], y="Close", title="S&P 500 · last 30 trading days", template="plotly_white"),
                width="stretch",
            )
        else:
            st.info("Market data is temporarily unavailable. Cached artifacts remain usable.")
    with pulse_right:
        render_section_kicker("Coverage snapshot")
        dc1, dc2 = st.columns(2)
        dc1.metric("OHLCV OK", ohlcv_counts["OK"])
        dc2.metric("Limited History", ohlcv_counts["LIMITED_HISTORY"], help="Active recent listings with no valid pre-listing history.")
        dc3, dc4 = st.columns(2)
        dc3.metric("Delisted", ohlcv_counts["DELISTED"])
        dc4.metric("Network Timeouts", ohlcv_counts["NETWORK_TIMEOUT"])
        st.caption(f"Factor panel: {factor['rows']:,} rows across {factor['tickers']:,} tickers.")
        safe_page_link("pages/8_🗄️_Data_Platform.py", "Open Data Platform")

    _render_workflows(company, portfolio, smart_money, ml_lab, str(factor["status"]))
    _render_advanced_ops(roots, artifacts, freshness, runs)
    _render_lightweight_generation(roots)
    _render_scheduler(roots)

    with st.expander("How to use this command center", expanded=True):
        st.markdown(
            """
            1. Search a ticker to set the shared context used by Screener, ML Stock Lab, Valuation and Portfolio.
            2. Use **Workflows** to choose the right module; labels show READY, WIP or PLANNED status.
            3. Use **Platform status** for a quick health read; Data Platform remains the detailed control room.
            4. Keep notebook and CLI work for experiments or maintenance, not daily first-click research.
            """
        )

    st.caption(FOOTER_TEXT)
