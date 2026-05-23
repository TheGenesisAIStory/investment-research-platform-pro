"""Final dashboard and report builders for the portfolio analysis notebook.

This module mirrors the company valuation dashboard pattern: notebook cells
produce canonical tables/charts, while this module turns them into a navigable
HTML dashboard and companion report without changing the research methodology.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


PORTFOLIO_PARAMETER_GUIDE: list[dict[str, str]] = [
    {
        "parameter": "main_ticker / benchmark / peers",
        "where": "MASTERREQUEST / UNIVERSE",
        "meaning": "Defines the focal security, benchmark reference, and peer/watchlist universe.",
        "impact": "Controls allocation candidates, benchmark comparison, factor dispersion, and dashboard context.",
    },
    {
        "parameter": "data_source / data_layers",
        "where": "MASTERREQUEST / PROJECT_DATABASE_CONFIG",
        "meaning": "Prioritizes project DB, Database Finanziario, local cache, API registry, yfinance, and fallback sources.",
        "impact": "Changes freshness, provenance, coverage, and diagnostic warnings.",
    },
    {
        "parameter": "model_depth / detail_level",
        "where": "ML_CONFIG / EXPERIMENT",
        "meaning": "Controls model breadth, governance detail, chart density, and table verbosity.",
        "impact": "Higher levels expose more diagnostics and slower model comparisons.",
    },
    {
        "parameter": "transaction_cost_bps / slippage_bps / turnover_limit",
        "where": "PORTFOLIO_CONFIG",
        "meaning": "Portfolio implementation assumptions used by net backtest and optimization diagnostics.",
        "impact": "Higher costs reduce net performance and can change portfolio feasibility.",
    },
    {
        "parameter": "scenario framework",
        "where": "VALUATION_CONFIG / scenario table",
        "meaning": "Bear, base, and bull return assumptions translated from company-level target proxies.",
        "impact": "Drives scenario distribution, asymmetry, and downside-risk interpretation.",
    },
]


PORTFOLIO_SOURCE_GUIDE: list[dict[str, str]] = [
    {"source": "Project DB / data catalog", "use": "First-priority local metadata, cached datasets, and inventory evidence."},
    {"source": "Database Finanziario / Google Drive", "use": "Persistent heavy data and generated research artifacts outside Git."},
    {"source": "Local market cache", "use": "Fast offline price/fundamental lookups where present."},
    {"source": "API registry / yfinance", "use": "Fresh market and metadata fallback for missing or stale tickers."},
    {"source": "Synthetic fallback", "use": "Last resort only, logged through diagnostics and source summary."},
]


@dataclass
class PortfolioDashboardArtifacts:
    dashboard_path: Path
    report_path: Path
    dashboard_html: str
    report_html: str


def _first(namespace: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        value = namespace.get(name)
        if value is not None:
            return value
    return default


def _as_df(value: Any) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, pd.Series):
        return value.to_frame().T
    if isinstance(value, list):
        try:
            return pd.DataFrame(value)
        except Exception:
            return pd.DataFrame()
    if isinstance(value, dict):
        try:
            return pd.DataFrame([value])
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def _format_value(value: Any) -> str:
    try:
        if pd.isna(value):
            return "n/a"
    except Exception:
        pass
    if isinstance(value, float):
        if abs(value) <= 2:
            return f"{value:,.2%}"
        return f"{value:,.2f}"
    return escape(str(value))


def _table_html(df: Any, title: str, max_rows: int = 40) -> str:
    frame = _as_df(df)
    if frame.empty:
        return f"<section class='card'><h2>{escape(title)}</h2><p class='muted'>No data available.</p></section>"
    return (
        f"<section class='card'><h2>{escape(title)}</h2>"
        + frame.head(max_rows).to_html(index=False, classes="data-table", border=0, escape=True)
        + "</section>"
    )


def _plotly_html(fig: Any, title: str) -> str:
    if fig is None:
        return f"<section class='card chart-card'><h2>{escape(title)}</h2><p class='muted'>No chart available.</p></section>"
    try:
        inner = fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True})
    except Exception as exc:
        return f"<section class='card chart-card'><h2>{escape(title)}</h2><p class='muted'>Chart unavailable: {escape(str(exc))}</p></section>"
    return f"<section class='card chart-card'><h2>{escape(title)}</h2>{inner}</section>"


def _output_root(namespace: Mapping[str, Any]) -> Path:
    for name in ["OUTPUT_ROOT", "output_root", "OUTPUT_DIR", "output_dir"]:
        value = namespace.get(name)
        if value is not None:
            return Path(value)
    return Path.cwd() / "output"


def _tables(namespace: Mapping[str, Any]) -> Mapping[str, Any]:
    result = namespace.get("RESEARCH_RESULT") or namespace.get("result") or {}
    if isinstance(result, Mapping) and isinstance(result.get("tables"), Mapping):
        return result["tables"]
    tables = namespace.get("tables")
    return tables if isinstance(tables, Mapping) else {}


def _charts(namespace: Mapping[str, Any]) -> Mapping[str, Any]:
    result = namespace.get("RESEARCH_RESULT") or namespace.get("result") or {}
    if isinstance(result, Mapping) and isinstance(result.get("charts"), Mapping):
        return result["charts"]
    charts = namespace.get("charts")
    return charts if isinstance(charts, Mapping) else {}


def _config(namespace: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = namespace.get(name)
    return value if isinstance(value, Mapping) else {}


def dashboard_css() -> str:
    return """
:root { --bg:#f6f8fb; --ink:#172033; --muted:#667085; --card:#fff; --line:#d9e2ec; --primary:#01696f; --accent:#da7101; --soft:#eef7f8; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font-family:Inter,Segoe UI,Arial,sans-serif; }
.shell { max-width:1420px; margin:0 auto; padding:24px; }
.hero { background:linear-gradient(135deg,#013f43,#01696f); color:white; padding:30px; border-radius:10px; }
.hero h1 { margin:0 0 8px; font-size:32px; }
.muted { color:var(--muted); }
.hero .muted { color:#c8f0ee; }
.grid { display:grid; gap:14px; }
.kpi-grid { grid-template-columns:repeat(auto-fit,minmax(185px,1fr)); margin:18px 0; }
.card { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:18px; box-shadow:0 1px 2px rgba(16,24,40,.04); overflow:auto; }
.kpi .label { color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.04em; }
.kpi .value { font-size:24px; font-weight:750; margin-top:6px; color:var(--primary); }
.nav { position:sticky; top:0; z-index:5; background:rgba(246,248,251,.96); border-bottom:1px solid var(--line); padding:10px 0; margin:16px 0; }
.nav button { border:1px solid var(--line); background:white; border-radius:7px; padding:9px 12px; margin:3px; cursor:pointer; color:var(--ink); }
.nav button.active { background:var(--primary); color:white; border-color:var(--primary); }
.tab-panel { display:none; }
.tab-panel.active { display:block; }
.data-table { border-collapse:collapse; width:100%; font-size:13px; }
.data-table th { background:var(--primary); color:white; text-align:left; padding:8px; position:sticky; top:0; }
.data-table td { border-bottom:1px solid var(--line); padding:7px; vertical-align:top; }
.pill { display:inline-block; padding:4px 8px; border-radius:999px; background:var(--soft); color:var(--primary); margin:2px; font-size:12px; }
.two { grid-template-columns:repeat(auto-fit,minmax(340px,1fr)); }
.chart-card { min-height:360px; }
.note { background:#fff7ed; border-left:4px solid var(--accent); padding:12px; border-radius:6px; }
.selection-chip { display:inline-flex; align-items:center; gap:6px; border:1px solid var(--line); background:white; border-radius:999px; padding:5px 9px; margin:3px; font-size:12px; }
.selection-chip b { color:var(--primary); }
.toolbar { display:flex; flex-wrap:wrap; gap:10px; align-items:center; margin:12px 0; }
.toolbar input, .toolbar select { padding:8px 10px; border:1px solid var(--line); border-radius:7px; min-width:220px; background:white; color:var(--ink); }
"""


def dashboard_js() -> str:
    return """
function showTab(id) {
  document.querySelectorAll('.tab-panel').forEach(function(el) { el.classList.remove('active'); });
  document.querySelectorAll('.nav button').forEach(function(el) { el.classList.remove('active'); });
  document.getElementById(id).classList.add('active');
  document.querySelector('[data-tab=\"' + id + '\"]').classList.add('active');
}
window.addEventListener('DOMContentLoaded', function() { showTab('tab-executive'); });
"""


def build_kpis(namespace: Mapping[str, Any]) -> str:
    tables = _tables(namespace)
    universe = _config(namespace, "UNIVERSE")
    allocation = _as_df(tables.get("portfolio_allocation"))
    performance = _as_df(tables.get("performance_summary", tables.get("backtest_performance")))
    risk = _as_df(tables.get("risk_dashboard"))
    scenarios = _as_df(tables.get("portfolio_scenarios"))
    kpis = [
        ("Main ticker", universe.get("main_ticker", "n/a")),
        ("Benchmark", universe.get("benchmark", "n/a")),
        ("Universe", len(universe.get("all_tickers", []) or [])),
        ("Top allocation", allocation.iloc[0].get("ticker", "n/a") if not allocation.empty else "n/a"),
        ("Net annual return", performance.iloc[0].get("annual_return_net", pd.NA) if not performance.empty else pd.NA),
        ("Max drawdown", performance.iloc[0].get("max_drawdown_net", pd.NA) if not performance.empty else pd.NA),
        ("Risk warnings", int((risk.get("status", pd.Series(dtype=str)) == "WARN").sum()) if not risk.empty else 0),
        ("Base scenario", scenarios.loc[scenarios.get("scenario", pd.Series(dtype=str)).eq("base"), "portfolio_return"].iloc[0] if not scenarios.empty and "portfolio_return" in scenarios.columns and "scenario" in scenarios.columns and scenarios["scenario"].eq("base").any() else pd.NA),
    ]
    return "<section class='grid kpi-grid'>" + "".join(
        f"<div class='card kpi'><div class='label'>{escape(str(label))}</div><div class='value'>{_format_value(value)}</div></div>"
        for label, value in kpis
    ) + "</section>"


def build_selection_lab_section(namespace: Mapping[str, Any]) -> str:
    tables = _tables(namespace)
    results = _as_df(tables.get("portfolio_selection_results", namespace.get("portfolio_selection_results")))
    audit = _as_df(tables.get("portfolio_selection_audit", namespace.get("portfolio_selection_audit")))
    progression = _as_df(tables.get("portfolio_selection_progression", namespace.get("portfolio_selection_progression")))
    summary = _as_df(tables.get("portfolio_selection_summary", namespace.get("portfolio_selection_summary")))
    schema = _as_df(tables.get("portfolio_selection_schema", namespace.get("portfolio_selection_schema")))
    presets = _as_df(tables.get("portfolio_selection_presets", namespace.get("portfolio_selection_presets")))
    config = namespace.get("portfolio_selection_config", {}) or {}
    filters = pd.DataFrame(config.get("filters", [])) if isinstance(config, Mapping) else pd.DataFrame()
    kpi = summary.iloc[0].to_dict() if not summary.empty else {}
    kpis = [
        ("Preset", kpi.get("preset", config.get("preset", "n/a") if isinstance(config, Mapping) else "n/a")),
        ("Selected", kpi.get("selected_rows", len(results))),
        ("Applied", kpi.get("filters_applied", 0)),
        ("Skipped", kpi.get("filters_skipped", 0)),
        ("Ranking", kpi.get("ranking_mode", "n/a")),
    ]
    kpi_html = "<section class='grid kpi-grid'>" + "".join(
        f"<div class='card kpi'><div class='label'>{escape(str(label))}</div><div class='value'>{_format_value(value)}</div></div>"
        for label, value in kpis
    ) + "</section>"
    chips = ""
    if not filters.empty:
        for _, row in filters.iterrows():
            chips += f"<span class='selection-chip'><b>{escape(str(row.get('field', 'field')))}</b>{escape(str(row.get('op', '')))} {escape(str(row.get('value', '')))}</span>"
    else:
        chips = "<span class='selection-chip'><b>Preset only</b>No custom filters</span>"
    interpretation_html = (
        "<section class='card'><h2>How To Read Portfolio Selection</h2>"
        "<p class='muted'>This layer ranks allocation candidates, not isolated stocks. A high selection score means the asset fits the active portfolio objective and available constraints better than peers in this run. Review weights, risk score, optimizer preference, scenario downside and diagnostics together before changing an allocation.</p>"
        "</section>"
    )
    chart_html = ""
    try:
        import plotly.express as px
        if not results.empty and {"selection_score", "ticker"}.issubset(results.columns):
            chart_html += _plotly_html(px.bar(results.head(25), x="ticker", y="selection_score", color="sector" if "sector" in results.columns else None, title="Selection Score Ranking", template="plotly_white"), "Selection Score Ranking")
        if not results.empty and {"risk_score", "quality_score"}.issubset(results.columns):
            chart_html += _plotly_html(px.scatter(results, x="risk_score", y="quality_score", size="weight" if "weight" in results.columns else None, color="selection_score" if "selection_score" in results.columns else None, hover_name="ticker" if "ticker" in results.columns else None, title="Quality vs Risk Selection Map", template="plotly_white"), "Quality vs Risk Map")
    except Exception:
        chart_html = ""
    return (
        "<section class='card'><h2>Portfolio Selection Lab</h2><p class='muted'>Finviz-inspired filtering adapted to allocation research: candidates are selected from ranking, allocation, optimizer, ML, scenario and diagnostics tables.</p></section>"
        + kpi_html
        + interpretation_html
        + f"<section class='card'><h2>Active Selection Filters</h2>{chips}</section>"
        + _table_html(summary, "Selection Summary", 20)
        + _table_html(results, "Selection Results", 120)
        + chart_html
        + _table_html(audit, "Selection Audit", 120)
        + _table_html(progression, "Selection Progression", 120)
        + _table_html(presets, "Selection Presets", 100)
        + _table_html(schema, "Selection Filter Schema", 200)
    )


def build_dashboard_html(namespace: Mapping[str, Any]) -> str:
    tables = _tables(namespace)
    charts = _charts(namespace)
    master = _config(namespace, "MASTERREQUEST")
    experiment = _config(namespace, "EXPERIMENT")
    universe = _config(namespace, "UNIVERSE")

    tabs = {
        "tab-executive": build_kpis(namespace)
        + "<section class='card'><h2>Executive Summary</h2><p>This dashboard translates the company valuation benchmark into portfolio analytics: allocation, performance, benchmark comparison, factor exposures, optimization, scenarios, diagnostics, and reproducible exports.</p></section>"
        + _table_html(tables.get("sws_portfolio_snapshot"), "Portfolio KPI Snapshot", 40),
        "tab-allocation": _table_html(tables.get("portfolio_allocation"), "Target Allocation", 80)
        + _plotly_html(charts.get("allocation_pie"), "Allocation Weights"),
        "tab-selection": build_selection_lab_section(namespace),
        "tab-performance": _table_html(tables.get("performance_summary", tables.get("backtest_performance")), "Performance Summary", 40)
        + _plotly_html(charts.get("backtest"), "Strategy vs Benchmark")
        + _plotly_html(charts.get("drawdown"), "Drawdown"),
        "tab-benchmark": _table_html(tables.get("benchmark_comparison"), "Benchmark Comparison", 60)
        + _plotly_html(charts.get("benchmark_comparison"), "Benchmark Comparison Chart"),
        "tab-factors": _table_html(tables.get("portfolio_factor_exposure"), "Portfolio Factor Exposures", 40)
        + _plotly_html(charts.get("portfolio_factor_exposure"), "Weighted Factor Exposure")
        + _plotly_html(charts.get("factor_exposure"), "Security Factor Exposure"),
        "tab-optimization": _table_html(tables.get("optimization_summary"), "Optimization Summary", 80)
        + _plotly_html(charts.get("optimization_weights"), "Optimized Allocation")
        + _plotly_html(charts.get("efficient_frontier_proxy"), "Efficient Frontier Proxy"),
        "tab-engine": _table_html(tables.get("portfolio_engine_weights"), "Portfolio Engine Weights", 100)
        + _table_html(tables.get("portfolio_engine_metrics"), "Portfolio Engine Metrics", 50)
        + _table_html(tables.get("portfolio_engine_diagnostics"), "Portfolio Engine Diagnostics", 20)
        + _plotly_html(charts.get("engine_weights_bar"), "Engine Allocation Weights")
        + _plotly_html(charts.get("engine_weights_treemap"), "Engine Allocation Treemap")
        + _plotly_html(charts.get("engine_frontier"), "Engine Efficient Frontier")
        + _plotly_html(charts.get("engine_backtest"), "Engine Backtest"),
        "tab-ml-ts": _table_html(tables.get("time_series_diagnostics"), "Time Series Diagnostics", 80)
        + _table_html(tables.get("time_series_forecast"), "Forecast vs Actual", 100)
        + _table_html(tables.get("time_series_regimes"), "Regime Detection", 100)
        + _table_html(tables.get("time_series_anomalies"), "Anomaly Flags", 100)
        + _table_html(tables.get("deep_stock_signals"), "Deep Stock Signals", 100)
        + _table_html(tables.get("crypto_signals"), "Crypto ML Signals", 100)
        + _plotly_html(charts.get("ts_forecast_actual"), "Forecast vs Actual")
        + _plotly_html(charts.get("ts_regime_detection"), "Regime Detection")
        + _plotly_html(charts.get("ts_anomaly_flags"), "Anomaly Flags")
        + _plotly_html(charts.get("deep_stock_scores"), "Deep Stock Score Ranking")
        + _plotly_html(charts.get("crypto_ml_signals"), "Crypto ML Signals"),
        "tab-scenarios": _table_html(tables.get("portfolio_scenarios"), "Portfolio Scenario Analysis", 40)
        + _plotly_html(charts.get("portfolio_scenario"), "Portfolio Scenario Returns")
        + _plotly_html(charts.get("scenario"), "Security Scenario Returns"),
        "tab-diagnostics": _table_html(tables.get("diagnostics"), "Diagnostics", 100)
        + _table_html(tables.get("data_quality"), "Data Quality", 100)
        + _table_html(tables.get("data_source_summary"), "Data Source Summary", 100),
        "tab-outputs": _table_html(pd.DataFrame([master]), "MASTERREQUEST", 5)
        + _table_html(pd.DataFrame([experiment]), "EXPERIMENT", 5)
        + _table_html(pd.DataFrame([universe]), "UNIVERSE", 5)
        + _table_html(pd.DataFrame(PORTFOLIO_PARAMETER_GUIDE), "Parameter Guide", 100)
        + _table_html(pd.DataFrame(PORTFOLIO_SOURCE_GUIDE), "Source Guide", 100),
    }
    nav = [
        ("tab-executive", "Executive"),
        ("tab-allocation", "Allocation"),
        ("tab-selection", "Selection Lab"),
        ("tab-performance", "Performance"),
        ("tab-benchmark", "Benchmark"),
        ("tab-factors", "Factors"),
        ("tab-optimization", "Optimization"),
        ("tab-engine", "Engine Lab"),
        ("tab-ml-ts", "ML / TS"),
        ("tab-scenarios", "Scenarios"),
        ("tab-diagnostics", "Diagnostics"),
        ("tab-outputs", "Outputs"),
    ]
    nav_html = "<nav class='nav'>" + "".join(
        f"<button data-tab='{tab_id}' onclick=\"showTab('{tab_id}')\">{escape(label)}</button>"
        for tab_id, label in nav
    ) + "</nav>"
    panels = "".join(f"<main id='{tab_id}' class='tab-panel'>{html}</main>" for tab_id, html in tabs.items())
    generated = datetime.now().isoformat(timespec="seconds")
    return f"""<!doctype html>
<html>
<head>
  <meta charset='utf-8'>
  <meta name='viewport' content='width=device-width, initial-scale=1'>
  <title>Portfolio Analysis Dashboard</title>
  <script src='https://cdn.plot.ly/plotly-latest.min.js'></script>
  <style>{dashboard_css()}</style>
</head>
<body>
  <div class='shell'>
    <header class='hero'>
      <h1>Portfolio Analysis Research Platform</h1>
      <p class='muted'>Company valuation UX standard translated into allocation, performance, benchmark, factors, optimization, scenarios and diagnostics. Generated: {escape(generated)}</p>
    </header>
    {nav_html}
    {panels}
  </div>
  <script>{dashboard_js()}</script>
</body>
</html>"""


def build_report_html(namespace: Mapping[str, Any]) -> str:
    tables = _tables(namespace)
    sections = [
        "<section class='card'><h2>Methodology</h2><p>The portfolio report preserves the notebook pipeline: central config, cache/API ingestion, diagnostics, allocation scoring, risk/return analytics, benchmark comparison, factor exposures, optimization proxy, scenario analysis, and Plotly dashboard export.</p></section>",
        _table_html(tables.get("portfolio_allocation"), "Allocation", 80),
        _table_html(tables.get("portfolio_selection_summary"), "Portfolio Selection Summary", 20),
        _table_html(tables.get("portfolio_selection_results"), "Portfolio Selection Results", 100),
        _table_html(tables.get("portfolio_selection_audit"), "Portfolio Selection Audit", 100),
        _table_html(tables.get("performance_summary", tables.get("backtest_performance")), "Performance", 40),
        _table_html(tables.get("portfolio_factor_exposure"), "Factor Exposures", 40),
        _table_html(tables.get("optimization_summary"), "Optimization", 80),
        _table_html(tables.get("portfolio_engine_weights"), "Portfolio Engine Weights", 100),
        _table_html(tables.get("portfolio_engine_metrics"), "Portfolio Engine Metrics", 50),
        _table_html(tables.get("time_series_diagnostics"), "Time Series Diagnostics", 80),
        _table_html(tables.get("time_series_forecast"), "Time Series Forecasts", 100),
        _table_html(tables.get("deep_stock_signals"), "Deep Stock Signals", 100),
        _table_html(tables.get("crypto_signals"), "Crypto ML Signals", 100),
        _table_html(tables.get("portfolio_scenarios"), "Scenarios", 40),
        _table_html(tables.get("diagnostics"), "Diagnostics", 100),
        _table_html(pd.DataFrame(PORTFOLIO_PARAMETER_GUIDE), "Parameter Guide", 100),
    ]
    return f"""<!doctype html>
<html>
<head>
  <meta charset='utf-8'>
  <meta name='viewport' content='width=device-width, initial-scale=1'>
  <title>Portfolio Analysis Report</title>
  <style>{dashboard_css()}</style>
</head>
<body>
  <div class='shell'>
    <header class='hero'>
      <h1>Portfolio Analysis Research Report</h1>
      <p class='muted'>Generated: {escape(datetime.now().isoformat(timespec="seconds"))}</p>
    </header>
    {''.join(sections)}
  </div>
</body>
</html>"""


def export_dashboard_and_report(namespace: Mapping[str, Any]) -> PortfolioDashboardArtifacts:
    output_root = _output_root(namespace)
    dashboard_dir = output_root / "dashboard"
    reports_dir = output_root / "reports"
    dashboard_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    dashboard_html = build_dashboard_html(namespace)
    report_html = build_report_html(namespace)
    dashboard_path = dashboard_dir / "portfolio_analysis_navigable_dashboard.html"
    report_path = reports_dir / "portfolio_analysis_research_report.html"
    dashboard_path.write_text(dashboard_html, encoding="utf-8")
    report_path.write_text(report_html, encoding="utf-8")
    return PortfolioDashboardArtifacts(dashboard_path, report_path, dashboard_html, report_html)
