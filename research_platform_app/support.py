"""Read-only Streamlit adapters for notebook-generated research artifacts."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.research_platform_core import as_df, discover_financial_database_root
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


APP_TITLE = "Research Platform"


def default_roots() -> dict[str, Path]:
    financial_db, _, _ = discover_financial_database_root()
    return {
        "company": Path(os.getenv("COMPANY_VALUATION_OUTPUT_ROOT", PROJECT_ROOT / "company_valuation" / "output")),
        "portfolio": Path(os.getenv("PORTFOLIO_OUTPUT_ROOT", PROJECT_ROOT / "portfolio_analysis" / "output")),
        "workspace": Path(os.getenv("RESEARCH_PLATFORM_OUTPUT_ROOT", PROJECT_ROOT / "output")),
        "financial_db": Path(os.getenv("FINANCIAL_DB_ROOT", financial_db)),
    }


def configure_page(page_title: str) -> None:
    import streamlit as st

    st.set_page_config(page_title=f"{APP_TITLE} · {page_title}", layout="wide")
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.8rem; }
        div[data-testid="stMetric"] { background:#ffffff; border:1px solid #d9e2ec; border-radius:10px; padding:12px; }
        .rp-note { background:#f6f8fb; border-left:5px solid #01696f; padding:12px 14px; border-radius:8px; color:#344054; }
        .rp-warn { background:#fff7ed; border-left:5px solid #da7101; padding:12px 14px; border-radius:8px; color:#344054; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def sidebar_roots() -> dict[str, Path]:
    import streamlit as st

    roots = default_roots()
    with st.sidebar:
        st.header("Artifact Roots")
        st.caption("Notebooks generate the data. Streamlit reads the exported CSV/JSON/HTML files.")
        company = st.text_input("Company output root", str(roots["company"]))
        portfolio = st.text_input("Portfolio output root", str(roots["portfolio"]))
        workspace = st.text_input("Workspace output root", str(roots["workspace"]))
        financial_db = st.text_input("Financial DB root", str(roots["financial_db"]))
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
    for domain, root in roots.items():
        if domain == "financial_db":
            continue
        if not root.exists():
            rows.append({"domain": domain, "type": "missing_root", "name": root.name, "path": str(root), "size_kb": 0.0})
            continue
        for path in root.rglob("*"):
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
        mtimes = [p.stat().st_mtime for p in root.rglob("*") if p.is_file()]
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
    cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=7)
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
    st.dataframe(df, width="stretch", hide_index=True)
    st.download_button(
        f"Download {label}",
        df.to_csv(index=False),
        file_name=file_name,
        mime="text/csv",
        width="content",
    )
