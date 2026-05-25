from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from app_settings import load_platform_settings, save_platform_settings
from support import configure_page, load_ohlcv_coverage_manifest, ohlcv_coverage_counts, render_context_bar, render_footer, render_page_header, render_page_intro, sidebar_roots

from research_platform_core.data_platform import (
    dataset_status,
    load_catalog_tables,
    provider_fallback_plan,
    publish_artifacts_to_financial_db,
    recent_run_logs,
    refresh_europe_stoxx_prices_incremental,
    resolve_data_platform_roots,
    shared_database_contract,
    write_data_platform_status,
)
from research_platform_core.data_health import get_data_health_summary, list_ohlcv_failures, list_ohlcv_provider_failures, load_run_events
from research_platform_core.data_explorer import get_single_ticker_snapshot, list_available_tickers, load_data_explorer_preview
from research_platform_core.multi_asset_universe import load_multi_asset_universe_manifest, summarize_multi_asset_universe
from research_platform_core.run_lock import is_stage_locked, read_stage_lock, stage_lock_path
from research_platform_core.smart_money import load_smart_money_source_manifest, summarize_smart_money_sources

configure_page("Data Platform")

import plotly.express as px
import pandas as pd
import streamlit as st


roots = sidebar_roots()
platform_settings = load_platform_settings(roots["workspace"])
platform_roots = resolve_data_platform_roots(
    financial_db_root=roots["financial_db"],
    repo_output_root=roots["workspace"],
)


@st.cache_data(ttl=300, show_spinner=False)
def _cached_ticker_options(financial_db_root: str, output_root: str) -> pd.DataFrame:
    return list_available_tickers(financial_db_root, output_root)


@st.cache_data(ttl=300, show_spinner=False)
def _cached_ticker_snapshot(ticker: str, financial_db_root: str, output_root: str) -> dict[str, object]:
    return get_single_ticker_snapshot(ticker, financial_db_root, output_root)


@st.cache_data(ttl=300, show_spinner=False)
def _cached_data_preview(
    dataset: str,
    financial_db_root: str,
    output_root: str,
    universe: str,
    start_date: str,
    end_date: str,
    max_rows: int,
) -> pd.DataFrame:
    return load_data_explorer_preview(dataset, financial_db_root, output_root, universe, start_date, end_date, max_rows)


def _status_tone(status: str) -> str:
    value = str(status or "").upper()
    if value in {"OK", "DONE", "SUCCESS"}:
        return "normal"
    if value in {"RUNNING", "PLANNED", "PARTIAL", "LIMITED_HISTORY"}:
        return "off"
    return "inverse"


def _safe_json_loads(value: object) -> dict[str, object]:
    if not value:
        return {}
    try:
        parsed = json.loads(str(value))
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _format_breakdown(value: object, limit: int = 4) -> str:
    data = _safe_json_loads(value)
    if not data:
        return "No universe breakdown yet"
    ordered = sorted(data.items(), key=lambda item: int(item[1]) if str(item[1]).isdigit() else 0, reverse=True)
    return " · ".join(f"{key}: {value}" for key, value in ordered[:limit])


def _status_message(status: object) -> tuple[str, str]:
    value = str(status or "UNKNOWN").upper()
    if value in {"OK", "DONE", "SUCCESS"}:
        return "OK", "success"
    if value in {"PARTIAL", "LIMITED_HISTORY", "PLANNED", "NOT_TICKER_SPECIFIC", "SKIPPED"}:
        return value, "warning"
    if value in {"MISSING", "FAILED", "ERROR"}:
        return value, "error"
    if value == "RUNNING":
        return value, "info"
    return value, "info"


def _render_status_notice(status: object, text: str) -> None:
    label, tone = _status_message(status)
    message = f"**{label}** · {text}"
    if tone == "success":
        st.success(message)
    elif tone == "warning":
        st.warning(message)
    elif tone == "error":
        st.error(message)
    else:
        st.info(message)


def _switch_to_page(page: str, ticker: str) -> None:
    st.session_state["selected_ticker"] = ticker
    try:
        st.switch_page(page)
    except Exception:
        st.info(f"Ticker `{ticker}` salvato nel contesto. Apri `{page}` dal menu laterale.")


def _compact_columns(frame: pd.DataFrame, preferred: list[str], max_cols: int = 14) -> pd.DataFrame:
    if frame.empty:
        return frame
    cols = [col for col in preferred if col in frame.columns]
    cols.extend([col for col in frame.columns if col not in cols][: max(0, max_cols - len(cols))])
    return frame[cols[:max_cols]]


def _run_dir(workspace_root: Path, prefix: str) -> Path:
    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S")
    run_dir = Path(workspace_root) / "runs" / f"{prefix}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _launch_equity_prices_restart(
    *,
    workspace_root: Path,
    financial_db_root: Path,
    config: dict[str, object],
    run_prefix: str,
) -> dict[str, str]:
    """Launch a data-health restart in a detached Python worker.

    The worker calls `research_platform_core.data_health.restart_equity_prices`;
    the Streamlit page only creates a user-friendly Run ID and log files.
    """
    existing_lock = read_stage_lock("equity_prices", workspace_root)
    if existing_lock is not None and is_stage_locked("equity_prices", workspace_root):
        return {
            "run_id": existing_lock.run_id,
            "pid": str(existing_lock.pid),
            "log_path": "",
            "status": "LOCKED_RUNNING",
            "lock_path": existing_lock.lock_path,
        }
    try:
        health = get_data_health_summary(financial_db_root, workspace_root)
        running_row = health[health["stage"].astype(str).eq("equity_prices") & health["status"].astype(str).eq("RUNNING")]
        if not running_row.empty:
            row = running_row.iloc[0]
            return {
                "run_id": str(row.get("run_id", "")),
                "pid": str(row.get("pid", "")),
                "log_path": str(row.get("log_path", "")),
                "status": "LOCKED_RUNNING",
                "lock_path": str(row.get("lock_path", "")),
            }
    except Exception:
        pass
    run_dir = _run_dir(workspace_root, run_prefix)
    log_path = run_dir / "backfill.log"
    metadata_path = run_dir / "run_metadata.json"
    payload = {
        **config,
        "financial_db_root": str(financial_db_root),
        "output_root": str(workspace_root),
        "run_id": run_dir.name,
        "run_events_path": str(run_dir / "progress.jsonl"),
    }
    metadata = {
        "run_id": run_dir.name,
        "job": "restart_equity_prices",
        "status": "RUNNING",
        "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "config": payload,
        "log_path": str(log_path),
        "lock_path": str(stage_lock_path("equity_prices", workspace_root)),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    code = """
from __future__ import annotations
import json
import os
import sys
import traceback
from pathlib import Path
import pandas as pd
from research_platform_core.data_health import restart_equity_prices

payload = json.loads(sys.argv[1])
run_dir = Path(sys.argv[2])
os.environ["RESEARCH_PLATFORM_RUN_ID"] = run_dir.name
metadata_path = run_dir / "run_metadata.json"
print(f"[data_platform] run_id={run_dir.name} start", flush=True)
print(f"[data_platform] config={json.dumps(payload, default=str)}", flush=True)
try:
    manifest = restart_equity_prices(payload)
    summary_path = run_dir / "equity_prices_restart_manifest.csv"
    manifest.to_csv(summary_path, index=False)
    status = str(manifest.get("status", pd.Series(["SUCCESS"])).iloc[0]) if not manifest.empty else "SUCCESS"
    status = "SUCCESS" if status not in {"LOCKED_RUNNING"} else status
    rows = len(manifest)
    print(f"[data_platform] status={status} rows={rows} manifest={summary_path}", flush=True)
except Exception as exc:
    status = "FAILED"
    rows = 0
    print(f"[data_platform] status=FAILED error={type(exc).__name__}: {exc}", flush=True)
    print(traceback.format_exc(), flush=True)
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update({
    "status": status,
    "finished_at": pd.Timestamp.now(tz="UTC").isoformat(),
    "rows": rows,
})
metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
sys.exit(0 if status == "SUCCESS" else 1)
"""
    env = os.environ.copy()
    env["RESEARCH_PLATFORM_RUN_ID"] = run_dir.name
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            [sys.executable, "-c", code, json.dumps(payload, default=str), str(run_dir)],
            cwd=str(APP_DIR.parent),
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
    (run_dir / "backfill.pid").write_text(str(proc.pid), encoding="utf-8")
    metadata["pid"] = str(proc.pid)
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return {"run_id": run_dir.name, "pid": str(proc.pid), "log_path": str(log_path)}

render_page_header(
    "Data Platform",
    "Single shared Database Finanziario control room: explore one ticker, inspect domain coverage, monitor runs and restart data jobs safely.",
    "▦",
    module="PLATFORM OPS",
    status="READY",
)
render_context_bar()
render_page_intro(
    "Monitor the single shared Database Finanziario, coverage manifests, provider health and refresh jobs.",
    "Check coverage first; launch incremental refreshes only when freshness or status cards show a watch item.",
)

cols = st.columns(4)
cols[0].metric("Financial DB", "available" if platform_roots.available else "missing")
cols[1].metric("Source", platform_roots.source)
cols[2].metric("Local Cache", platform_roots.local_cache.name)
cols[3].metric("Repo Output", platform_roots.repo_output.name)

st.code(str(platform_roots.financial_db))

max_files = st.sidebar.slider("Inventory file scan limit", min_value=500, max_value=12000, value=5000, step=500)
contract = shared_database_contract(platform_roots.financial_db, roots["workspace"])
run_logs = recent_run_logs(roots["workspace"], limit=12)
ohlcv_manifest = load_ohlcv_coverage_manifest(roots)
ohlcv_counts = ohlcv_coverage_counts(ohlcv_manifest)
multi_asset_manifest = load_multi_asset_universe_manifest(platform_roots.financial_db, roots["workspace"])
multi_asset_summary = summarize_multi_asset_universe(multi_asset_manifest)
smart_money_source_manifest = load_smart_money_source_manifest(platform_roots.financial_db, roots["workspace"])
smart_money_source_summary = summarize_smart_money_sources(smart_money_source_manifest)
try:
    data_health = get_data_health_summary(platform_roots.financial_db, roots["workspace"])
    ohlcv_failures = list_ohlcv_failures(platform_roots.financial_db, roots["workspace"])
    ohlcv_provider_failures = list_ohlcv_provider_failures(platform_roots.financial_db, roots["workspace"])
except Exception as exc:
    data_health = pd.DataFrame()
    ohlcv_failures = pd.DataFrame()
    ohlcv_provider_failures = pd.DataFrame()
    st.warning(f"Data Health temporarily unavailable: {type(exc).__name__}. Check run logs from the Run Logs tab.")

if not platform_roots.available:
    st.warning("Database Finanziario is not available. Mount Google Drive or set FINANCIAL_DB_ROOT.")
    render_footer()
    st.stop()

top = st.columns(4)
top[0].metric("Shared Domains", len(contract))
top[1].metric("Domains Available", int(contract["exists"].sum()) if not contract.empty else 0)
top[2].metric("Markers Available", int(contract["marker_exists"].sum()) if not contract.empty else 0)
top[3].metric("Recent Run Logs", len(run_logs))

with st.container(border=True):
    st.markdown("**OHLCV coverage states**")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("OK", ohlcv_counts["OK"])
    c2.metric("Limited History", ohlcv_counts["LIMITED_HISTORY"], help="Active listing, but first available price is after 2000-01-01.")
    c3.metric("Delisted", ohlcv_counts["DELISTED"])
    c4.metric("Network Timeout", ohlcv_counts["NETWORK_TIMEOUT"])
    c5.metric("No Price Data", ohlcv_counts["NO_PRICE_DATA"])

with st.container(border=True):
    st.markdown("**Data Health Overview**")
    st.caption("Stage-level status from `research_platform_core.data_health`; no direct file logic is duplicated in the UI.")
    if data_health.empty:
        st.info("No data-health summary is available yet. Run the bootstrap or coverage validator first.")
    else:
        health_cards = st.columns(4)
        stage_labels = {
            "equity_fundamentals": "Fundamentals",
            "equity_prices": "Prices",
            "macro_fx": "Macro / FX",
            "factor_libraries": "Factors",
            "smart_money": "Smart Money",
            "banking": "Banking",
            "factor_universe_panel": "Factor Panel",
        }
        for idx, row in enumerate(data_health.to_dict("records")):
            with health_cards[idx % 4]:
                stage = str(row.get("stage", "stage"))
                status = str(row.get("status", "UNKNOWN"))
                covered = int(row.get("covered_assets") or 0)
                failures = int(row.get("failure_count") or 0)
                delta = "watch" if failures else status
                st.metric(stage_labels.get(stage, stage.replace("_", " ").title()), f"{covered:,}", delta=delta, delta_color=_status_tone(status))
                st.caption(_format_breakdown(row.get("coverage_breakdown")))

with st.container(border=True):
    st.subheader("Single Ticker Explorer")
    st.caption("Search one issuer and see immediately which data layers are available before jumping into research pages.")
    ticker_options = _cached_ticker_options(str(platform_roots.financial_db), str(roots["workspace"]))
    known_tickers = ticker_options["ticker"].dropna().astype(str).tolist() if not ticker_options.empty and "ticker" in ticker_options.columns else []
    default_ticker = st.session_state.get("data_platform_ticker") or st.session_state.get("selected_ticker") or ""
    search_cols = st.columns([0.42, 0.36, 0.22])
    ticker_query = search_cols[0].text_input(
        "Ticker",
        value=str(default_ticker),
        placeholder="AAPL, FBYD, ORGN...",
        help="Ticker search uses OHLCV coverage, ML signals and factor artifacts.",
    ).strip().upper()
    quick_pick = search_cols[1].selectbox(
        "Known tickers",
        [""] + known_tickers[:5000],
        index=0,
        help="Loaded from lightweight manifests; type in the text box for tickers outside the first 5,000 options.",
    )
    if quick_pick:
        ticker_query = quick_pick
    if search_cols[2].button("Load ticker data", disabled=not ticker_query, width="stretch"):
        st.session_state["data_platform_ticker"] = ticker_query
        st.session_state["selected_ticker"] = ticker_query
    active_ticker = str(st.session_state.get("data_platform_ticker") or ticker_query or "").strip().upper()
    if not active_ticker:
        st.info("Enter a ticker to inspect OHLCV, fundamentals, factors, ML signals, valuation proxies and Smart Money coverage.")
    else:
        with st.spinner(f"Loading data map for {active_ticker}..."):
            snapshot = _cached_ticker_snapshot(active_ticker, str(platform_roots.financial_db), str(roots["workspace"]))
        availability = snapshot.get("availability", pd.DataFrame())
        if isinstance(availability, pd.DataFrame) and not availability.empty:
            status_cols = st.columns(min(4, len(availability)))
            for idx, row in enumerate(availability.head(4).to_dict("records")):
                with status_cols[idx % len(status_cols)]:
                    st.metric(str(row.get("domain", "")), str(row.get("status", "UNKNOWN")), help=str(row.get("detail", "")))
            st.dataframe(_compact_columns(availability, ["domain", "status", "rows", "detail", "path"], max_cols=5), width="stretch", hide_index=True)
        nav_cols = st.columns(4)
        if nav_cols[0].button("Open in ML Stock Lab", width="stretch"):
            _switch_to_page("pages/9_ML_Stock_Lab.py", active_ticker)
        if nav_cols[1].button("Open Valuation", width="stretch"):
            _switch_to_page("pages/2_🔬_Valuation_Research.py", active_ticker)
        if nav_cols[2].button("Open Screener", width="stretch"):
            _switch_to_page("pages/4_🔍_Screener_Builder.py", active_ticker)
        if nav_cols[3].button("Use in Portfolio", width="stretch"):
            _switch_to_page("pages/3_📁_Portfolio_Research.py", active_ticker)

        ticker_tabs = st.tabs(["OHLCV", "Factors", "ML / Valuation", "Fundamentals", "Smart Money"])
        with ticker_tabs[0]:
            ohlcv = snapshot.get("ohlcv", pd.DataFrame())
            if isinstance(ohlcv, pd.DataFrame) and not ohlcv.empty:
                price_col = "adjclose" if "adjclose" in ohlcv.columns else "close" if "close" in ohlcv.columns else None
                if price_col and "date" in ohlcv.columns:
                    st.plotly_chart(px.line(ohlcv, x="date", y=price_col, title=f"{active_ticker} OHLCV price preview", template="plotly_white"), width="stretch")
                st.dataframe(_compact_columns(ohlcv.tail(20), ["date", "open", "high", "low", "close", "adjclose", "volume", "source_path"]), width="stretch", hide_index=True)
            else:
                _render_status_notice("MISSING", "No local OHLCV parquet found for this ticker. Check Data Health & Restart for retry options.")
        with ticker_tabs[1]:
            factors = snapshot.get("factors", pd.DataFrame())
            if isinstance(factors, pd.DataFrame) and not factors.empty:
                latest = factors.tail(1)
                score_cols = [col for col in ["value_score", "quality_score", "momentum_score", "risk_score", "size_score", "growth_score", "factor_composite_score"] if col in factors.columns]
                if score_cols:
                    chart = latest.melt(id_vars=[col for col in ["ticker", "date"] if col in latest.columns], value_vars=score_cols, var_name="factor", value_name="score")
                    st.plotly_chart(px.bar(chart, x="factor", y="score", title="Latest factor scores", template="plotly_white"), width="stretch")
                st.dataframe(_compact_columns(factors.tail(30), ["date", "ticker", *score_cols, "ret21d", "ret63d", "ret252d", "vol63d", "vol252d"]), width="stretch", hide_index=True)
            else:
                _render_status_notice("MISSING", "No factor rows found. Run `factor_universe_panel` after OHLCV backfill.")
        with ticker_tabs[2]:
            ml_signals = snapshot.get("ml_signals", pd.DataFrame())
            if isinstance(ml_signals, pd.DataFrame) and not ml_signals.empty:
                preferred = ["date", "ticker", "score_ols", "score_rf", "score_gbrt", "score_composite", "ml_score", "fair_value_hat", "valuation_signal_score", "mispricing_rel", "artifact"]
                st.dataframe(_compact_columns(ml_signals.tail(50), preferred), width="stretch", hide_index=True)
            else:
                _render_status_notice("PLANNED", "No trained ML signal found for this ticker yet. Run ML training or open ML Stock Lab.")
        with ticker_tabs[3]:
            fundamentals = snapshot.get("fundamentals", pd.DataFrame())
            if isinstance(fundamentals, pd.DataFrame) and not fundamentals.empty:
                st.dataframe(_compact_columns(fundamentals.tail(30), ["ticker", "statement_artifact", "asOfDate", "periodType", "currencyCode", "source_path"]), width="stretch", hide_index=True)
            else:
                _render_status_notice("MISSING", "No statement-level fundamentals artifact found for this ticker.")
        with ticker_tabs[4]:
            smart_money = snapshot.get("smart_money", pd.DataFrame())
            if isinstance(smart_money, pd.DataFrame) and not smart_money.empty:
                st.dataframe(_compact_columns(smart_money, ["ticker", "issuer_ticker", "score", "smart_money_score", "event_type", "event_date", "artifact"]), width="stretch", hide_index=True)
            else:
                _render_status_notice("PLANNED", "Smart Money engine exists, but no ticker-level issuer/event row is loaded for this ticker.")

tabs = st.tabs([
    "Data Explorer",
    "Domain Status",
    "Data Health & Restart",
    "Shared DB Contract",
    "OHLCV Coverage",
    "Run Monitor",
    "Run Logs",
    "Inventory Scan",
    "Providers",
    "Credentials",
    "Price Refresh",
    "Sync / Contracts",
    "Data Settings",
])

with tabs[0]:
    st.subheader("Data Explorer")
    st.caption("Touch the data directly: pick a dataset, scope it lightly, preview rows and inspect a simple coverage chart.")
    dataset_map = {
        "Factor panel": "factor_panel",
        "OHLCV coverage manifest": "ohlcv_manifest",
        "Equity fundamentals manifest": "equity_fundamentals_manifest",
        "ML trained signals": "ml_signals",
        "Smart Money artifacts": "smart_money",
    }
    explorer_cols = st.columns([0.28, 0.22, 0.2, 0.2, 0.1])
    dataset_label = explorer_cols[0].selectbox("Dataset", list(dataset_map.keys()), help="Bounded preview; heavy datasets are sampled server-side.")
    universe_values = ["All"]
    if not ticker_options.empty and "universe" in ticker_options.columns:
        universe_values.extend(sorted(ticker_options["universe"].dropna().astype(str).replace("", pd.NA).dropna().unique().tolist())[:200])
    universe_filter = explorer_cols[1].selectbox("Universe / source", universe_values, help="Uses manifest universe/exchange/source columns when available.")
    start_filter = explorer_cols[2].date_input("From", value=pd.Timestamp("2000-01-01").date())
    end_filter = explorer_cols[3].date_input("To", value=pd.Timestamp.now().date())
    max_preview_rows = explorer_cols[4].number_input("Rows", min_value=100, max_value=10000, value=2000, step=100)
    if st.button("Load preview", width="stretch"):
        st.session_state["data_platform_preview_request"] = {
            "dataset": dataset_map[dataset_label],
            "universe": universe_filter,
            "start": str(start_filter),
            "end": str(end_filter),
            "rows": int(max_preview_rows),
        }
    request = st.session_state.get("data_platform_preview_request")
    if not request:
        st.info("Load a preview to inspect data rows without opening notebooks. Factor panel previews are sampled to keep the app responsive.")
    else:
        with st.spinner("Loading bounded data preview..."):
            preview = _cached_data_preview(
                str(request["dataset"]),
                str(platform_roots.financial_db),
                str(roots["workspace"]),
                str(request["universe"]),
                str(request["start"]),
                str(request["end"]),
                int(request["rows"]),
            )
        if preview.empty:
            _render_status_notice("MISSING", "No rows available for this dataset/scope. Check Domain Status or run the relevant job.")
        else:
            metric_cols = st.columns(4)
            metric_cols[0].metric("Rows previewed", f"{len(preview):,}")
            metric_cols[1].metric("Columns", f"{len(preview.columns):,}")
            metric_cols[2].metric("Tickers", f"{preview['ticker'].nunique():,}" if "ticker" in preview.columns else "n/a")
            metric_cols[3].metric("Dataset", dataset_label)
            if "date" in preview.columns:
                chart_frame = preview.copy()
                chart_frame["date"] = pd.to_datetime(chart_frame["date"], errors="coerce")
                metric = next((col for col in ["factor_composite_score", "score_composite", "ml_score", "value_score", "momentum_score"] if col in chart_frame.columns), None)
                if metric:
                    chart = chart_frame.dropna(subset=["date"]).set_index("date").resample("ME")[metric].mean().reset_index()
                    if not chart.empty:
                        st.plotly_chart(px.line(chart, x="date", y=metric, title=f"Monthly average {metric}", template="plotly_white"), width="stretch")
                elif "ticker" in chart_frame.columns:
                    chart = chart_frame.dropna(subset=["date"]).set_index("date").resample("ME")["ticker"].nunique().reset_index(name="tickers")
                    if not chart.empty:
                        st.plotly_chart(px.bar(chart, x="date", y="tickers", title="Ticker coverage over time", template="plotly_white"), width="stretch")
            st.dataframe(preview, width="stretch", hide_index=True)
            st.download_button("Download preview CSV", preview.to_csv(index=False), "data_platform_preview.csv", "text/csv")

with tabs[1]:
    st.subheader("Domain Status")
    st.caption("Every data domain should say whether it is active, planned, missing data, or waiting for a job. No silent empty boxes.")
    health_by_stage = {str(row.get("stage")): row for row in data_health.to_dict("records")} if not data_health.empty else {}
    domain_specs = [
        {
            "label": "Equity Fundamentals",
            "stage": "equity_fundamentals",
            "what": "Universe constituents, yfinance/FMP/Finnhub-style fundamentals, valuation inputs and statement snapshots.",
            "jobs": "`bootstrap_equity_fundamentals`, Data Platform fundamentals stage, Notebook Runner maintenance jobs.",
            "missing": "If empty, run fundamentals backfill for the target universes.",
        },
        {
            "label": "Equity Prices / OHLCV",
            "stage": "equity_prices",
            "what": "Daily OHLCV parquet files with LIMITED_HISTORY / DELISTED / provider failure classification.",
            "jobs": "`bootstrap_equity_prices`, Data Health restart, failed-only retry.",
            "missing": "If partial, inspect OHLCV provider/write failures and retry only failed assets.",
        },
        {
            "label": "FX / Macro",
            "stage": "macro_fx",
            "what": "Benchmark/context series such as DX-Y.NYB, UUP, TLT, GLD, DBC, rates, ECB/Banca d'Italia macro panels.",
            "jobs": "`bootstrap_macro_fx_series`, ECB/Banca d'Italia clients, yfinance macro overlays.",
            "missing": "Currently may be PLANNED/PARTIAL until the macro stage is run; not a ticker-level failure.",
        },
        {
            "label": "Factor Libraries",
            "stage": "factor_libraries",
            "what": "Fama-French/AQR-style factor libraries and internal factor panels for model benchmarking.",
            "jobs": "`bootstrap_factor_libraries`, `build_factor_universe_panel`.",
            "missing": "Run factor libraries for external benchmarks; run factor panel after OHLCV backfill for internal scores.",
        },
        {
            "label": "Smart Money / Flow / Positioning",
            "stage": "smart_money",
            "what": "Issuer scores, government/flow/event artifacts, macro positioning and source registry.",
            "jobs": "`bootstrap_smart_money_data`, Smart Money engine pipeline.",
            "missing": "If no ticker rows appear, the module exists but issuer-level coverage is not populated for that ticker yet.",
        },
        {
            "label": "Banking Data Lab",
            "stage": "banking",
            "what": "Banking universe, macro/regulatory panels, market data and diagnostics for banks.",
            "jobs": "`bootstrap_banking_universe`, Banking Data Lab pipeline.",
            "missing": "If empty, run Banking Data Lab Pipeline; previous stable artifacts are reused when a run fails.",
        },
    ]
    for spec in domain_specs:
        row = health_by_stage.get(spec["stage"], {})
        status = str(row.get("status", "MISSING"))
        with st.container(border=True):
            c1, c2, c3 = st.columns([0.24, 0.18, 0.58])
            c1.metric(spec["label"], status)
            c2.metric("Covered", f"{int(row.get('covered_assets') or 0):,}")
            with c3:
                _render_status_notice(status, spec["missing"])
                st.markdown(f"**What it covers:** {spec['what']}")
                st.markdown(f"**Jobs / hooks:** {spec['jobs']}")
    st.markdown("### Multi-Asset Universe & Coverage")
    st.caption("FX, commodities, crypto, ETF and fixed-income coverage derived from the Macro DB catalog/manifest.")
    if multi_asset_summary.empty:
        st.info("No multi-asset manifest available yet. Compile Macro View or refresh the manifest.")
    else:
        st.dataframe(multi_asset_summary, width="stretch", hide_index=True)
        if {"domain", "instrument_count", "ok_count"}.issubset(multi_asset_summary.columns):
            chart_frame = multi_asset_summary.melt(
                id_vars=["domain"],
                value_vars=[col for col in ["ok_count", "planned_count", "partial_count"] if col in multi_asset_summary.columns],
                var_name="coverage_state",
                value_name="count",
            )
            st.plotly_chart(px.bar(chart_frame, x="domain", y="count", color="coverage_state", title="Multi-asset coverage by domain", template="plotly_white"), width="stretch")
        with st.expander("Multi-asset manifest detail", expanded=False):
            st.dataframe(multi_asset_manifest, width="stretch", hide_index=True)
    st.markdown("### Smart Money Source Coverage")
    st.caption("Explicit source readiness for COT, ETF flows, options positioning and issuer-event evidence.")
    if smart_money_source_summary.empty:
        st.info("No Smart Money source manifest available yet.")
    else:
        st.dataframe(smart_money_source_summary, width="stretch", hide_index=True)
        smart_chart_cols = [col for col in ["ok_count", "ready_optional_count", "planned_count", "partial_count"] if col in smart_money_source_summary.columns]
        if smart_chart_cols and "domain" in smart_money_source_summary.columns:
            smart_chart = smart_money_source_summary.melt(
                id_vars=["domain"],
                value_vars=smart_chart_cols,
                var_name="coverage_state",
                value_name="count",
            )
            st.plotly_chart(px.bar(smart_chart, x="domain", y="count", color="coverage_state", title="Smart Money sources by readiness", template="plotly_white"), width="stretch")
        with st.expander("Smart Money source manifest detail", expanded=False):
            st.dataframe(smart_money_source_manifest, width="stretch", hide_index=True)

with tabs[2]:
    st.subheader("Backfill Controls / Restart")
    st.caption("Friendly controls for restarting data jobs without opening notebooks or calling CLI commands directly.")
    latest_run = st.session_state.get("last_data_restart_run")
    if latest_run:
        st.success(f"Job started: Run ID={latest_run.get('run_id')} · PID={latest_run.get('pid')}")
        st.code(latest_run.get("log_path", ""), language=None)

    with st.expander("Advanced options", expanded=False):
        a1, a2, a3, a4 = st.columns(4)
        restart_start_year = a1.number_input("Start year", min_value=1990, max_value=2026, value=2000, step=1)
        restart_end_year = a2.number_input("End year", min_value=2000, max_value=2035, value=2026, step=1)
        restart_batch_size = a3.number_input("Batch size", min_value=1, max_value=300, value=80, step=5)
        restart_max_assets_raw = a4.number_input("Max assets cap", min_value=0, max_value=10000, value=0, step=25, help="0 means no cap.")
        dry_run_restart = st.toggle("Dry run only", value=False, help="Validate asset selection and write manifests without provider downloads.")
        parquet_override = st.text_input(
            "OHLCV parquet root override",
            value=str(platform_settings.get("data", {}).get("ohlcv_parquet_root") or ""),
            help="Optional. Leave empty to use env/settings/manifest-driven root resolution.",
        ).strip()

    retry_tickers = st.session_state.get("data_retry_tickers", [])
    base_restart_config = {
        "start_year": int(restart_start_year),
        "end_year": int(restart_end_year),
        "batch_size": int(restart_batch_size),
        "max_assets": int(restart_max_assets_raw),
        "dry_run": bool(dry_run_restart),
    }
    if parquet_override:
        base_restart_config["parquet_root"] = parquet_override

    left_restart, right_restart = st.columns([1.1, 1])
    with left_restart:
        with st.container(border=True):
            st.markdown("**Riprova download per titoli falliti**")
            st.caption("Uses `OHLCV_write_failures.csv` and retries only failed assets; selected tickers are optional.")
            retry_disabled = ohlcv_failures.empty
            retry_label = f"Restart equity_prices (failed only) · {len(retry_tickers) or len(ohlcv_failures)} assets"
            if st.button(retry_label, disabled=retry_disabled, width="stretch"):
                config = {
                    **base_restart_config,
                    "failed_only": True,
                    "mode": "full",
                }
                if retry_tickers:
                    config["tickers"] = retry_tickers
                with st.spinner("Starting retry job..."):
                    launched = _launch_equity_prices_restart(
                        workspace_root=roots["workspace"],
                        financial_db_root=platform_roots.financial_db,
                        config=config,
                        run_prefix="equity_prices_retry_failed",
                    )
                st.session_state["last_data_restart_run"] = launched
                if launched.get("status") == "LOCKED_RUNNING":
                    st.warning(f"Run già in esecuzione: Run ID={launched['run_id']} · PID={launched.get('pid')}. Apri Run Monitor.")
                else:
                    st.success(f"Job started: Run ID={launched['run_id']}. Monitor Run Monitor for progress.")

        with st.container(border=True):
            st.markdown("**Avvia backfill prezzi per universi selezionati**")
            markets = st.multiselect(
                "Universes / market groups",
                ["us_all", "europe_major", "japan_major", "global_etfs"],
                default=["us_all"],
                help="These groups map to the MarketUniverseBuilder used by the core data pipeline.",
            )
            mode = st.radio("Run mode", ["incremental", "full"], horizontal=True, help="Incremental resumes from latest stored price; full requests the configured history window.")
            if st.button("Run equity_prices for selected universes", disabled=not markets, width="stretch"):
                config = {
                    **base_restart_config,
                    "failed_only": False,
                    "markets": markets,
                    "mode": mode,
                }
                with st.spinner("Starting universe price job..."):
                    launched = _launch_equity_prices_restart(
                        workspace_root=roots["workspace"],
                        financial_db_root=platform_roots.financial_db,
                        config=config,
                        run_prefix="equity_prices_universe",
                    )
                st.session_state["last_data_restart_run"] = launched
                if launched.get("status") == "LOCKED_RUNNING":
                    st.warning(f"Run già in esecuzione: Run ID={launched['run_id']} · PID={launched.get('pid')}.")
                else:
                    st.success(f"Job started: Run ID={launched['run_id']}.")

    with right_restart:
        with st.container(border=True):
            st.markdown("**Avvia backfill prezzi completo**")
            st.caption("Heavy operation. Use after provider/root checks, preferably with local OHLCV parquet storage.")
            confirm_full = st.checkbox("I understand this can run for a long time and call external providers.")
            if st.button("Avvia backfill prezzi (completo)", disabled=not confirm_full, width="stretch"):
                config = {
                    **base_restart_config,
                    "failed_only": False,
                    "markets": ["us_all", "europe_major", "japan_major", "global_etfs"],
                    "mode": "full",
                }
                with st.spinner("Starting full price backfill..."):
                    launched = _launch_equity_prices_restart(
                        workspace_root=roots["workspace"],
                        financial_db_root=platform_roots.financial_db,
                        config=config,
                        run_prefix="equity_prices_full",
                    )
                st.session_state["last_data_restart_run"] = launched
                if launched.get("status") == "LOCKED_RUNNING":
                    st.warning(f"Run già in esecuzione: Run ID={launched['run_id']} · PID={launched.get('pid')}.")
                else:
                    st.success(f"Full price backfill started: Run ID={launched['run_id']}.")

        with st.container(border=True):
            st.markdown("**Data Health table**")
            if data_health.empty:
                st.info("No health rows available.")
            else:
                display_cols = [
                    "stage",
                    "status",
                    "covered_assets",
                    "failure_count",
                    "ohlcv_parquet_files",
                    "run_id",
                    "log_path",
                ]
                st.dataframe(data_health[[c for c in display_cols if c in data_health.columns]], width="stretch", hide_index=True)
            if st.button("Controlla copertura dati", width="stretch"):
                st.rerun()

    st.subheader("OHLCV Failures Panel")
    failure_tabs = st.tabs(["Write failures", "Provider failures"])
    with failure_tabs[0]:
        st.caption("Retry-ready table from `list_ohlcv_failures()`. It stays empty when all parquet writes succeeded.")
        if ohlcv_failures.empty:
            st.success("No OHLCV parquet write failures are currently recorded.")
        else:
            f1, f2 = st.columns([0.24, 0.76])
            failure_days = f1.number_input("Only last N days", min_value=1, max_value=365, value=30, step=1)
            failures_view = ohlcv_failures.copy()
            if "last_failure_at" in failures_view.columns:
                failure_dates = pd.to_datetime(failures_view["last_failure_at"], errors="coerce", utc=True)
                cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=int(failure_days))
                failures_view = failures_view[failure_dates.ge(cutoff).fillna(True)].copy()
            ticker_options = sorted(failures_view.get("ticker", pd.Series(dtype=str)).dropna().astype(str).unique())
            selected_failures = f2.multiselect(
                "Prepare retry list",
                ticker_options,
                default=ticker_options[: min(25, len(ticker_options))],
                help="Leave all selected for a focused retry, or clear the list to retry every recorded failure.",
            )
            if st.button("Prepare retry list", disabled=failures_view.empty, width="stretch"):
                st.session_state["data_retry_tickers"] = selected_failures
                st.success(f"Retry list prepared: {len(selected_failures)} ticker(s).")
            show_failure_cols = ["ticker", "asset", "universe", "exchange", "target_path", "error_summary", "last_failure_at", "retry_candidate"]
            st.dataframe(failures_view[[c for c in show_failure_cols if c in failures_view.columns]], width="stretch", hide_index=True)
            st.download_button("Download OHLCV write failures CSV", failures_view.to_csv(index=False), "OHLCV_write_failures_filtered.csv", "text/csv")
    with failure_tabs[1]:
        st.caption("Provider-side coverage/failure table from `OHLCV_provider_failures.csv`.")
        if ohlcv_provider_failures.empty:
            st.success("No provider-side OHLCV failures are currently recorded.")
        else:
            p1, p2 = st.columns(2)
            error_filter = p1.multiselect("Error type", sorted(ohlcv_provider_failures["error_type"].dropna().astype(str).unique()), default=[])
            category_filter = p2.multiselect("Category", sorted(ohlcv_provider_failures["category"].dropna().astype(str).unique()), default=[])
            provider_view = ohlcv_provider_failures.copy()
            if error_filter:
                provider_view = provider_view[provider_view["error_type"].astype(str).isin(error_filter)]
            if category_filter:
                provider_view = provider_view[provider_view["category"].astype(str).isin(category_filter)]
            counts = provider_view.groupby("error_type", dropna=False)["ticker"].count().reset_index(name="count") if "error_type" in provider_view.columns else pd.DataFrame()
            if not counts.empty:
                st.dataframe(counts, width="stretch", hide_index=True)
            st.dataframe(provider_view.head(5000), width="stretch", hide_index=True)
            st.download_button("Download OHLCV provider failures CSV", provider_view.to_csv(index=False), "OHLCV_provider_failures_filtered.csv", "text/csv")

with tabs[3]:
    st.subheader("Shared Database Contract")
    st.caption("One Database Finanziario root feeds core, Streamlit pages, ML Stock Lab, Notebook Runner and scripts.")
    st.dataframe(contract, width="stretch", hide_index=True)
    with st.expander("Operating policy", expanded=True):
        st.markdown(
            """
            - **Drive-first:** Database Finanziario is the canonical data lake.
            - **Cache-second:** local/repo artifacts are reused before provider calls.
            - **API-last:** providers are called only for missing, stale or incomplete data.
            - **Single root:** changing `FINANCIAL_DB_ROOT` changes the DB for core, app, scripts and notebooks together.
            - **Weekly refresh + staleness refresh:** recurring checks avoid repeated full downloads.
            - **Credit discipline:** incremental writes, deduplication and provenance are preferred over bulk refresh.
            """
        )

with tabs[4]:
    st.subheader("OHLCV Coverage")
    if ohlcv_manifest.empty:
        st.info("No OHLCV daily manifest found yet. Run the OHLCV or historical data bootstrap job first.")
    else:
        view = ohlcv_manifest.copy()
        status_filter = st.multiselect(
            "Coverage status",
            sorted(view.get("coverage_status", view.get("status", pd.Series(dtype=str))).dropna().astype(str).unique()),
            default=[],
        )
        if status_filter:
            column = "coverage_status" if "coverage_status" in view.columns else "status"
            view = view[view[column].astype(str).isin(status_filter)]
        show_cols = [
            "ticker",
            "provider_symbol",
            "resolved_provider_symbol",
            "status",
            "coverage_status",
            "first_price_date",
            "last_price_date",
            "coverage_ratio",
            "rows",
            "provider",
            "coverage_reason",
            "error",
        ]
        st.dataframe(view[[c for c in show_cols if c in view.columns]].head(5000), width="stretch", hide_index=True)
        st.download_button("Download OHLCV coverage manifest", ohlcv_manifest.to_csv(index=False), "ohlcv_coverage_manifest.csv", "text/csv")

with tabs[5]:
    st.subheader("Run Monitor")
    st.caption("Structured progress from `progress.jsonl` when available. Text logs remain available in Run Logs.")
    runs_root = roots["workspace"] / "runs"
    run_dirs = sorted([path for path in runs_root.glob("*") if path.is_dir()], key=lambda path: path.stat().st_mtime, reverse=True)[:30] if runs_root.exists() else []
    if not run_dirs:
        st.info("No run directories found yet.")
    else:
        selected_run = st.selectbox("Run ID", [path.name for path in run_dirs], index=0)
        run_dir = runs_root / selected_run
        metadata_path = run_dir / "run_metadata.json"
        metadata = _safe_json_loads(metadata_path.read_text(encoding="utf-8", errors="replace")) if metadata_path.exists() else {}
        events = load_run_events(run_dir, limit=1000)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Status", metadata.get("status", "unknown"))
        m2.metric("PID", metadata.get("pid", "n/a"))
        m3.metric("Events", len(events))
        m4.metric("Lock", "tracked" if metadata.get("lock_path") else "not set")
        if events.empty:
            st.info("No structured progress file found yet for this run.")
        else:
            latest = events.tail(1).iloc[0]
            processed = int(latest.get("processed", 0) or 0)
            success = int(latest.get("success", 0) or 0)
            failed = int(latest.get("failed", 0) or 0)
            total_hint = max(processed, success + failed, 1)
            st.progress(min(processed / total_hint, 1.0), text=f"Processed {processed:,} · Success {success:,} · Failed/Watch {failed:,}")
            show_event_cols = [
                "timestamp",
                "event",
                "provider_status",
                "provider",
                "processed",
                "success",
                "failed",
                "limited_history",
                "parquet_files",
                "latest_parquet",
                "symbols",
            ]
            st.dataframe(events[[col for col in show_event_cols if col in events.columns]].tail(200), width="stretch", hide_index=True)

with tabs[6]:
    st.subheader("Recent Run Logs")
    st.caption("Debug links for advanced review. The UI does not print raw stacktraces by default.")
    if run_logs.empty:
        st.info("No run logs found yet.")
    else:
        st.dataframe(run_logs, width="stretch", hide_index=True)
        selected_log = st.selectbox("Preview log", [""] + run_logs["log_path"].dropna().astype(str).tolist())
        if selected_log:
            path = Path(selected_log)
            if path.exists():
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[-120:]
                st.code("\n".join(lines), language="text")

with tabs[7]:
    st.subheader("Inventory Scan")
    st.caption("Run this on demand. It can touch many Drive-backed files, so the shared contract above is the default lightweight view.")
    run_inventory = st.button("Run bounded inventory scan", width="stretch")
    if run_inventory:
        status = dataset_status(platform_roots.financial_db, max_files=max_files)
        inventory = status["inventory"]
        summary = status["summary"]
        st.dataframe(summary, width="stretch", hide_index=True)
        if not summary.empty and {"role", "files"}.issubset(summary.columns):
            st.plotly_chart(px.bar(summary, x="role", y="files", color="freshness_status", template="plotly_white", title="Dataset Inventory by Role"), width="stretch")
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
        st.dataframe(view.head(2000), width="stretch", hide_index=True)
        st.download_button("Download inventory CSV", inventory.to_csv(index=False), "data_platform_inventory.csv", "text/csv")

with tabs[8]:
    st.subheader("Providers")
    st.caption("Provider fallback reads are on-demand because registry files can live on slow Drive-backed storage.")
    if st.button("Load provider fallback plan", width="stretch"):
        fallback = provider_fallback_plan(platform_roots.financial_db)
        st.dataframe(fallback, width="stretch", hide_index=True)

with tabs[9]:
    st.subheader("Credentials")
    st.caption("Credential registry reads are on-demand to keep the page responsive when Drive is mounted remotely.")
    if st.button("Load credential and provider tables", width="stretch"):
        status = load_catalog_tables(platform_roots.financial_db)
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

with tabs[10]:
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

with tabs[11]:
    st.subheader("Status Exports")
    if st.button("Write DataPlatform status to workspace output", width="stretch"):
        paths = write_data_platform_status(platform_roots.financial_db, roots["workspace"], max_files=max_files)
        st.json({k: str(v) for k, v in paths.items()})
    st.subheader("API-ready contracts")
    if st.button("Load API-ready provider contracts", width="stretch"):
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

with tabs[12]:
    st.subheader("Data Settings")
    st.caption("Desk-facing refresh policy stored under output/config. The DB root itself remains read-only here and is controlled by FINANCIAL_DB_ROOT/storage policy.")
    data_settings = platform_settings.get("data", {})
    st.code(str(platform_roots.financial_db), language=None)
    d1, d2, d3 = st.columns(3)
    refresh_policy = d1.selectbox(
        "Refresh policy",
        ["daily_incremental", "weekly_full_validation", "manual_only"],
        index=["daily_incremental", "weekly_full_validation", "manual_only"].index(data_settings.get("refresh_policy", "daily_incremental"))
        if data_settings.get("refresh_policy", "daily_incremental") in ["daily_incremental", "weekly_full_validation", "manual_only"]
        else 0,
        help="Operational default shown to Notebook Runner and Data Platform users.",
    )
    strictness = d2.selectbox(
        "Coverage strictness",
        ["non-strict", "strict"],
        index=0 if data_settings.get("coverage_strictness", "non-strict") == "non-strict" else 1,
        help="Non-strict accepts LIMITED_HISTORY and focuses on actionable provider issues.",
    )
    limited_history_ok = d3.toggle(
        "LIMITED_HISTORY is acceptable",
        value=bool(data_settings.get("limited_history_ok", True)),
        help="Recent listings are monitored but not treated as data failures.",
    )
    threshold_cols = st.columns(2)
    coverage_watch_threshold = threshold_cols[0].slider(
        "Coverage watch threshold (%)",
        0,
        100,
        int(data_settings.get("coverage_watch_threshold", 80)),
        help="Universe coverage below this level should raise a desk watch item.",
    )
    network_timeout_watch_threshold = threshold_cols[1].number_input(
        "Network timeout watch count",
        min_value=0,
        max_value=10000,
        value=int(data_settings.get("network_timeout_watch_threshold", 10)),
        step=1,
        help="Timeout count above this level deserves retry/provider review.",
    )
    if st.button("Save Data Settings", width="stretch"):
        platform_settings["data"] = {
            **data_settings,
            "refresh_policy": refresh_policy,
            "coverage_strictness": strictness,
            "coverage_watch_threshold": int(coverage_watch_threshold),
            "limited_history_ok": bool(limited_history_ok),
            "network_timeout_watch_threshold": int(network_timeout_watch_threshold),
        }
        save_path = save_platform_settings(roots["workspace"], platform_settings)
        st.success(f"Data settings saved: {save_path.name}")

render_footer()
