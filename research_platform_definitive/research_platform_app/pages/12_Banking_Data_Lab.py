from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_DIR.parents[0]
SRC_DIR = PROJECT_ROOT / "src"
for path in [APP_DIR, PROJECT_ROOT, SRC_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data_bootstrap import render_bootstrap_banner
from support import configure_page, render_context_bar, render_footer, render_page_header, render_page_intro, safe_page_link, sidebar_roots
from ui_ops import render_missing_data_cta, render_safe_log_preview

try:
    from research_platform_core import run_banks_data_pipeline
except Exception:
    run_banks_data_pipeline = None


configure_page("Banking Data Lab")


def _read_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def _artifact_rows(output_dir: Path) -> pd.DataFrame:
    rows = []
    for name in ["banks_universe", "banks_macro_regulatory", "banks_fundamentals_panel", "banks_market_panel"]:
        path = output_dir / f"{name}.csv"
        rows.append(
            {
                "artifact": f"{name}.csv",
                "exists": path.exists(),
                "rows": len(_read_csv(path)),
                "path": str(path),
                "modified": pd.Timestamp(path.stat().st_mtime, unit="s").isoformat() if path.exists() else "",
            }
        )
    sqlite_path = output_dir / "banks_data.sqlite"
    rows.append(
        {
            "artifact": "banks_data.sqlite",
            "exists": sqlite_path.exists(),
            "rows": None,
            "path": str(sqlite_path),
            "modified": pd.Timestamp(sqlite_path.stat().st_mtime, unit="s").isoformat() if sqlite_path.exists() else "",
        }
    )
    return pd.DataFrame(rows)


def _log_pipeline_error(output_dir: Path, exc: Exception) -> tuple[str, Path]:
    logs = output_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    run_id = f"banking_{pd.Timestamp.now(tz='UTC').strftime('%Y%m%d_%H%M%S')}"
    path = logs / f"{run_id}.log"
    path.write_text(
        "Banking Data Lab pipeline failure\n"
        f"run_id={run_id}\n"
        f"exception={type(exc).__name__}: {exc}\n\n"
        + traceback.format_exc(),
        encoding="utf-8",
    )
    return run_id, path


def _friendly_error(exc: Exception) -> str:
    message = str(exc)
    if isinstance(exc, TypeError) and ("np.select" in message or "condlist" in message or "choicelist" in message):
        return (
            "Pipeline failed while building the bank universe classification. "
            "Likely cause: `np.select` received conditions/choices with inconsistent lengths. "
            "Check `build_banks_universe` inputs and confirm the source DataFrame columns are aligned."
        )
    return (
        f"Pipeline failed cleanly with `{type(exc).__name__}`. "
        "Open the diagnostics tab for the log path and rerun after checking input artifacts."
    )


def _last_run_status(output_dir: Path) -> dict[str, object]:
    path = output_dir / "banking_last_run.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_last_run(output_dir: Path, payload: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "banking_last_run.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


roots = sidebar_roots()

default_output = roots["workspace"] / "banks_pipeline"
default_cache = roots["workspace"] / "banks_pipeline" / "_cache"

render_page_header(
    "Banking Data Lab",
    "Open-source Italian and euro-area bank universe, regulatory/macro artifacts and listed-bank market panels.",
    "▤",
    module="LABS",
    status="WIP",
)
render_context_bar()
render_page_intro(
    "Refresh and inspect the banking universe, listed-bank market panel and diagnostics from one controlled interface.",
    "Run the pipeline only when artifacts are stale, then use Universe for analyst-facing review.",
)

with st.sidebar:
    st.header("Banking Pipeline")
    market_start = st.text_input("Market start", "2015-01-01", help="Start date for listed-bank market panels when market refresh is enabled.")
    include_market = st.checkbox("Refresh listed-bank market panel", value=False)
    include_ecb_macro = st.checkbox("Call ECB BSI macro now", value=False)
    with st.expander("Advanced paths", expanded=False):
        output_dir = Path(st.text_input("Output directory", str(default_output))).expanduser()
        cache_dir = Path(st.text_input("Cache directory", str(default_cache))).expanduser()

render_bootstrap_banner(roots, required=["banking_universe"])

st.markdown(
    """
    <div class="rp-note">
    <b>Data policy:</b> Drive/repo cache first, public official sources next, API fallbacks only for listed-bank market data.
    ECB/Wikipedia/Banca d'Italia outputs are stored as auditable CSV/SQLite artifacts.
    </div>
    """,
    unsafe_allow_html=True,
)

universe = _read_csv(output_dir / "banks_universe.csv")
macro = _read_csv(output_dir / "banks_macro_regulatory.csv")
fundamentals = _read_csv(output_dir / "banks_fundamentals_panel.csv")
market = _read_csv(output_dir / "banks_market_panel.csv")

cols = st.columns(4)
cols[0].metric("Universe Banks", len(universe))
cols[1].metric("Italian Banks", int(universe["country"].astype(str).eq("IT").sum()) if "country" in universe else 0)
cols[2].metric("Listed Banks", int(universe["listed_flag"].fillna(False).astype(bool).sum()) if "listed_flag" in universe else 0)
cols[3].metric("Market Rows", len(market))

tab_pipeline, tab_universe, tab_diagnostics = st.tabs(["Pipeline", "Universe", "Diagnostics"])

with tab_pipeline:
    left, right = st.columns([0.42, 0.58], gap="large")
    with left:
        st.subheader("Run parameters")
        st.write(f"Output: `{output_dir}`")
        st.write(f"Cache: `{cache_dir}`")
        st.write(f"Market start: `{market_start}`")
        st.write(f"Listed-bank market refresh: `{'on' if include_market else 'off'}`")
        st.write(f"ECB macro call: `{'on' if include_ecb_macro else 'off'}`")
        safe_page_link("pages/6_🧪_Notebook_Runner.py", "Open Banking Data job")
        if st.button("Run / Refresh Banking Pipeline", width="stretch", disabled=run_banks_data_pipeline is None):
            if run_banks_data_pipeline is None:
                st.error("Banking data engine is not importable in this app session.")
            else:
                try:
                    with st.spinner("Refreshing banking universe and artifacts..."):
                        artifacts = run_banks_data_pipeline(
                            output_dir=output_dir,
                            cache_dir=cache_dir,
                            market_start=market_start,
                            include_market=include_market,
                            include_ecb_macro=include_ecb_macro,
                        )
                    payload = {
                        "status": "SUCCESS",
                        "finished_at": pd.Timestamp.now(tz="UTC").isoformat(),
                        "output_dir": str(output_dir),
                        "rows": {name: len(df) for name, df in artifacts.items()},
                    }
                    _write_last_run(output_dir, payload)
                    st.success("Banking artifacts refreshed.")
                    st.json(payload)
                    st.rerun()
                except Exception as exc:
                    run_id, log_path = _log_pipeline_error(output_dir, exc)
                    payload = {
                        "status": "FAILED",
                        "finished_at": pd.Timestamp.now(tz="UTC").isoformat(),
                        "run_id": run_id,
                        "output_dir": str(output_dir),
                        "log_path": str(log_path),
                        "message": _friendly_error(exc),
                    }
                    _write_last_run(output_dir, payload)
                    st.error(payload["message"])
                    st.caption(f"Diagnostic log: {log_path}")
    with right:
        st.subheader("Last run status")
        last = _last_run_status(output_dir)
        if not last:
            st.info("No recorded run yet. Run the pipeline or load an existing artifact directory.")
        else:
            status = str(last.get("status", "UNKNOWN"))
            (st.success if status == "SUCCESS" else st.warning)(f"Last run: {status}")
            st.json(last)
        st.dataframe(_artifact_rows(output_dir), width="stretch", hide_index=True)

with tab_universe:
    if universe.empty:
        render_missing_data_cta(
            "Banking universe",
            job_id="banking_data_refresh",
            output_path=output_dir / "banks_universe.csv",
            cli_hint="python research_platform_app/scheduler.py --once --jobs banking_data_refresh",
        )
    else:
        c1, c2, c3 = st.columns(3)
        country_filter = c1.multiselect("Country", sorted(universe["country"].dropna().astype(str).unique()), default=["IT"] if "country" in universe and "IT" in set(universe["country"].astype(str)) else [])
        status_filter = c2.multiselect("Status", sorted(universe["status"].dropna().astype(str).unique()) if "status" in universe else [])
        query = c3.text_input("Bank / ticker search", "")
        view = universe.copy()
        if country_filter and "country" in view:
            view = view[view["country"].astype(str).isin(country_filter)]
        if status_filter and "status" in view:
            view = view[view["status"].astype(str).isin(status_filter)]
        if query:
            haystack = " ".join([c for c in ["bank_name", "name", "ticker", "isin"] if c in view.columns])
            if haystack:
                q = query.lower()
                text = view[[c for c in ["bank_name", "name", "ticker", "isin"] if c in view.columns]].astype(str).agg(" ".join, axis=1).str.lower()
                view = view[text.str.contains(q, regex=False)]
        st.success(f"Universe filters applied to {len(universe):,} banks - {len(view):,} match criteria.")
        st.dataframe(view, width="stretch", hide_index=True)
        if "ticker" in view.columns:
            selected_bank = st.selectbox("Selected bank ticker", ["", *sorted(view["ticker"].dropna().astype(str).unique().tolist())])
            if selected_bank:
                st.session_state["selected_ticker"] = selected_bank.upper().strip()
                st.caption("Selected ticker context updated for Screener, Smart Money and Portfolio pages.")
                a1, a2, a3 = st.columns(3)
                if a1.button("Open in Screener", width="stretch"):
                    st.switch_page("pages/4_🔍_Screener_Builder.py")
                if a2.button("Open Smart Money", width="stretch"):
                    st.switch_page("pages/1_📡_Smart_Money_Macro.py")
                if a3.button("Open Portfolio", width="stretch"):
                    st.switch_page("pages/3_📁_Portfolio_Research.py")

with tab_diagnostics:
    st.subheader("Artifacts")
    st.dataframe(_artifact_rows(output_dir), width="stretch", hide_index=True)
    st.subheader("Regulatory / macro")
    if macro.empty:
        st.info("No ECB/Banca d'Italia macro rows yet. ECB calls are optional; source links can be configured in notebook/user config.")
    else:
        st.dataframe(macro.head(500), width="stretch", hide_index=True)
    if fundamentals.empty:
        st.caption("No external bank fundamentals loaded yet.")
    else:
        st.subheader("Fundamentals")
        st.dataframe(fundamentals.head(500), width="stretch", hide_index=True)
    if market.empty:
        st.caption("No listed-bank market panel yet.")
    else:
        st.subheader("Market panel")
        st.dataframe(market.head(500), width="stretch", hide_index=True)
    st.subheader("Logs")
    logs = sorted((output_dir / "logs").glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True) if (output_dir / "logs").exists() else []
    if not logs:
        st.info("No pipeline logs found.")
    else:
        selected_log = st.selectbox("Log file", logs, format_func=lambda p: p.name)
        st.caption(str(selected_log))
        render_safe_log_preview(selected_log.read_text(encoding="utf-8", errors="replace"))

render_footer()
