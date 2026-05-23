"""Final dashboard and report builders for the company valuation notebook.

The functions in this module are intentionally defensive: the notebook has
several historical variable names, so the builders accept a namespace dict and
reuse whichever canonical objects are available without changing the valuation
methodology that produced them.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


PARAMETER_GUIDE: list[dict[str, str]] = [
    {
        "parameter": "ticker",
        "where": "MASTER_REQUEST",
        "meaning": "Company ticker used as the focal security for valuation, peer comparison, and dashboard KPI extraction.",
        "impact": "Changes the company shown in the executive view. The notebook must be rerun to rebuild market/fundamental data.",
    },
    {
        "parameter": "start_date / end_date",
        "where": "MASTER_REQUEST / EXPERIMENT",
        "meaning": "Date window for market data, factors, features, validation, and exported diagnostics.",
        "impact": "Longer windows improve history but may increase missingness and stale fundamental observations.",
    },
    {
        "parameter": "dcf_horizon_years",
        "where": "EXPERIMENT",
        "meaning": "Explicit forecast horizon used by DCF-style models before terminal value.",
        "impact": "Higher values give more weight to explicit cash-flow assumptions; lower values emphasize terminal assumptions.",
    },
    {
        "parameter": "dcf_perpetual_growth",
        "where": "EXPERIMENT",
        "meaning": "Long-run terminal growth assumption for Gordon-growth terminal value.",
        "impact": "Higher terminal growth increases intrinsic value when discount rate remains fixed.",
    },
    {
        "parameter": "discount_rate / cost_of_equity / WACC",
        "where": "EXPERIMENT / valuation config",
        "meaning": "Required return used to discount future cash flows, residual income, and dividends.",
        "impact": "Higher discount rates reduce intrinsic value and make the model more conservative.",
    },
    {
        "parameter": "peer_selection_method / n_peers / manual_peers",
        "where": "MASTER_REQUEST",
        "meaning": "Rules for choosing comparable companies in relative valuation and benchmarking.",
        "impact": "Peer set controls median multiples, relative valuation, and ranking context.",
    },
    {
        "parameter": "active_feature_blocks",
        "where": "EXPERIMENT",
        "meaning": "High-level groups of features used in modelling and interpretability panels.",
        "impact": "Controls which economic signals are emphasized by scoring and diagnostics.",
    },
    {
        "parameter": "data_source_choice",
        "where": "USER_SELECTION",
        "meaning": "Preferred source family for market/fundamental inputs and fallback logic.",
        "impact": "Changes provenance, coverage, and API/cache dependencies.",
    },
]


OPEN_SOURCE_SOURCES: list[dict[str, str]] = [
    {
        "source": "Yahoo Finance / yfinance",
        "type": "Open market-data API wrapper",
        "use": "Daily prices, market caps, dividends, metadata, option/earnings fallbacks where available.",
    },
    {
        "source": "Financial Modeling Prep (FMP)",
        "type": "Fundamental-data API",
        "use": "Income statement, balance sheet, cash-flow statement, ratios, company profiles when API key is configured.",
    },
    {
        "source": "SEC EDGAR",
        "type": "Open regulatory source",
        "use": "Filings and XBRL fundamentals for US-listed companies when connectors are available.",
    },
    {
        "source": "Fama-French Data Library",
        "type": "Open academic factor dataset",
        "use": "Asset-pricing factors and factor-model diagnostics.",
    },
    {
        "source": "Stooq / pandas-datareader compatible feeds",
        "type": "Open market-data fallback",
        "use": "Index, macro, and market history fallback where available.",
    },
    {
        "source": "FRED / central-bank macro series",
        "type": "Open macroeconomic data",
        "use": "Rates, inflation, liquidity, and macro scenario context when connectors are configured.",
    },
]


FORMULA_GUIDE: list[dict[str, str]] = [
    {
        "area": "Returns",
        "formula": "r_t = P_t / P_{t-1} - 1",
        "plain_english": "Daily return measures the one-period percentage move in adjusted price.",
        "used_for": "Momentum, volatility, forward-return targets, backtest diagnostics.",
        "watch_out": "Use adjusted prices when possible; raw prices can create false jumps around dividends and splits.",
    },
    {
        "area": "Forward return target",
        "formula": "R_{t,h} = P_{t+h} / P_t - 1",
        "plain_english": "The supervised ML target is the future return over a chosen horizon.",
        "used_for": "Regression/ranking targets such as 21d, 63d, and 252d forward returns.",
        "watch_out": "Never use future target information in features; split chronologically and respect embargo windows.",
    },
    {
        "area": "Volatility",
        "formula": "sigma_ann = std(r_t) * sqrt(252)",
        "plain_english": "Annualized volatility scales daily return dispersion to a yearly risk estimate.",
        "used_for": "Risk dashboard, risk-adjusted returns, volatility features.",
        "watch_out": "Short rolling windows are noisy; long windows react slowly to regime changes.",
    },
    {
        "area": "Drawdown",
        "formula": "DD_t = P_t / max(P_0...P_t) - 1",
        "plain_english": "Drawdown measures loss from the previous peak.",
        "used_for": "Downside risk, robustness checks, investor-friendly risk communication.",
        "watch_out": "Drawdown depends heavily on the selected date range.",
    },
    {
        "area": "DCF",
        "formula": "EV = sum(FCF_t / (1+WACC)^t) + TV / (1+WACC)^T",
        "plain_english": "Discounted cash flow values the company from forecast cash flows plus terminal value.",
        "used_for": "Intrinsic value estimate and scenario analysis.",
        "watch_out": "Terminal growth and discount rate dominate long-horizon value; expose them clearly.",
    },
    {
        "area": "Terminal value",
        "formula": "TV = FCF_{T+1} / (WACC - g)",
        "plain_english": "Gordon-growth terminal value capitalizes normalized cash flow after the explicit forecast period.",
        "used_for": "DCF terminal value.",
        "watch_out": "The formula is unstable when WACC is close to terminal growth.",
    },
    {
        "area": "Residual income",
        "formula": "Value = Book Value + sum((ROE - CoE) * Book Value / (1+CoE)^t)",
        "plain_english": "Residual income values equity from book value plus economic profit above cost of equity.",
        "used_for": "Bank/financial company valuation and accounting-driven valuation.",
        "watch_out": "Book value quality and ROE normalization matter more than point estimates.",
    },
    {
        "area": "Relative valuation",
        "formula": "Fair Value = Peer Median Multiple * Company Fundamental",
        "plain_english": "Comparable valuation prices the target using peer multiples.",
        "used_for": "PE, PB, EV/EBITDA, EV/Sales, peer premium/discount.",
        "watch_out": "Peer selection quality is the model; bad comparables create misleading precision.",
    },
    {
        "area": "Composite score",
        "formula": "Score = w_value*Value + w_quality*Quality + w_momentum*Momentum + w_risk*Risk",
        "plain_english": "The ranking score combines standardized signals into an analyst-friendly score.",
        "used_for": "Company ranking, shortlist generation, dashboard KPIs.",
        "watch_out": "Weights should match the investment profile and must be documented.",
    },
]


FEATURE_GUIDE: list[dict[str, str]] = [
    {
        "block": "Market",
        "examples": "ret_21d, ret_63d, volume, adj_close",
        "intuition": "Captures recent price behavior and liquidity.",
        "risk": "Can overfit recent market noise if used without time-aware validation.",
    },
    {
        "block": "Momentum",
        "examples": "mom_3m, mom_6m, mom_12_1",
        "intuition": "Measures continuation strength while avoiding immediate reversal windows.",
        "risk": "Momentum can crash during sharp regime reversals.",
    },
    {
        "block": "Valuation",
        "examples": "pe_ratio, pb_ratio, ev_ebitda, fcf_yield, upside_to_fair_value",
        "intuition": "Compares price paid to earnings, book value, cash flow, or model fair value.",
        "risk": "Cheap stocks can be value traps when quality and growth are weak.",
    },
    {
        "block": "Quality",
        "examples": "roe, roa, gross_margin, operating_margin, free_cash_flow_margin",
        "intuition": "Measures profitability, resilience, and business quality.",
        "risk": "Accounting distortions and one-off items can inflate quality measures.",
    },
    {
        "block": "Growth",
        "examples": "revenue_growth, eps_growth, net_income_growth, fcf_growth",
        "intuition": "Captures fundamental expansion and earnings momentum.",
        "risk": "High growth without cash conversion can destroy value.",
    },
    {
        "block": "Leverage",
        "examples": "debt_equity, net_debt_ebitda, interest_coverage",
        "intuition": "Measures balance-sheet risk and financial flexibility.",
        "risk": "Sector differences are large; banks and industrials need different interpretation.",
    },
    {
        "block": "Macro / Risk Factors",
        "examples": "rate_10y, yield_curve_slope, inflation_proxy, credit_spread",
        "intuition": "Links company valuation to rates, inflation, credit and factor regimes.",
        "risk": "Macro series can be stale or low-frequency relative to market prices.",
    },
    {
        "block": "Peers",
        "examples": "peer_similarity_score, peer_discount, peer_rank",
        "intuition": "Makes relative valuation and comparable-company selection auditable.",
        "risk": "Small peer sets create unstable median multiples.",
    },
]


@dataclass
class DashboardArtifacts:
    dashboard_path: Path
    report_path: Path
    dashboard_html: str
    report_html: str
    control_center_path: Path | None = None
    control_center_html: str | None = None


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
    if pd.isna(value):
        return "n/a"
    if isinstance(value, float):
        if abs(value) <= 2:
            return f"{value:,.2%}"
        return f"{value:,.2f}"
    return escape(str(value))


def _table_html(df: Any, title: str, max_rows: int = 30) -> str:
    frame = _as_df(df)
    if frame.empty:
        return f"<section class='card'><h2>{escape(title)}</h2><p class='muted'>No data available.</p></section>"
    clean = frame.head(max_rows).copy()
    return (
        f"<section class='card'><h2>{escape(title)}</h2>"
        + clean.to_html(index=False, classes="data-table", border=0, escape=True)
        + "</section>"
    )


def _plotly_html(fig: Any, title: str) -> str:
    if fig is None:
        return f"<section class='card'><h2>{escape(title)}</h2><p class='muted'>No chart available.</p></section>"
    try:
        inner = fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True})
    except Exception as exc:
        return f"<section class='card'><h2>{escape(title)}</h2><p class='muted'>Chart unavailable: {escape(str(exc))}</p></section>"
    return f"<section class='card chart-card'><h2>{escape(title)}</h2>{inner}</section>"


def _figure_dict(namespace: Mapping[str, Any]) -> dict[str, Any]:
    figures = _first(namespace, "interactivefigures", "interactive_figures", default={})
    return figures if isinstance(figures, dict) else {}


def _target_ticker(namespace: Mapping[str, Any]) -> str:
    master = _first(namespace, "MASTER_REQUEST", default={}) or {}
    user_selection = _first(namespace, "USER_SELECTION", default={}) or {}
    return (
        master.get("ticker")
        or master.get("target_ticker")
        or user_selection.get("selected_company")
        or user_selection.get("target_ticker")
        or "N/A"
    )


def _output_root(namespace: Mapping[str, Any]) -> Path:
    output_root = _first(namespace, "OUTPUT_ROOT", "output_root", default=None)
    if output_root is not None:
        return Path(output_root)
    output_dir = _first(namespace, "OUTPUT_DIR", "output_dir", default=None)
    if output_dir is not None:
        return Path(output_dir)
    return Path.cwd() / "output"


def _latest_row(namespace: Mapping[str, Any]) -> pd.Series:
    latest = _as_df(_first(namespace, "latestcrosssection", "latest_cross_section", "latest_crosssection"))
    ticker = _target_ticker(namespace)
    if latest.empty:
        return pd.Series(dtype="object")
    if "ticker" in latest.columns and ticker != "N/A":
        match = latest.loc[latest["ticker"].astype(str).str.upper() == ticker.upper()]
        if not match.empty:
            return match.iloc[0]
    return latest.iloc[0]


def dashboard_css() -> str:
    return """
:root {
  --bg:#f6f8fb; --ink:#172033; --muted:#667085; --card:#ffffff;
  --line:#d9e2ec; --primary:#01696f; --accent:#da7101; --soft:#eef7f8;
}
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font-family:Inter,Segoe UI,Arial,sans-serif; }
.shell { max-width:1380px; margin:0 auto; padding:24px; }
.hero { background:linear-gradient(135deg,#013f43,#01696f); color:white; padding:28px; border-radius:10px; }
.hero h1 { margin:0 0 8px; font-size:30px; }
.muted { color:var(--muted); }
.hero .muted { color:#c8f0ee; }
.grid { display:grid; gap:14px; }
.kpi-grid { grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); margin:18px 0; }
.card { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:18px; box-shadow:0 1px 2px rgba(16,24,40,.04); overflow:auto; }
.kpi .label { color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.04em; }
.kpi .value { font-size:24px; font-weight:700; margin-top:6px; }
.nav { position:sticky; top:0; z-index:5; background:rgba(246,248,251,.96); border-bottom:1px solid var(--line); padding:10px 0; margin:16px 0; }
.nav button { border:1px solid var(--line); background:white; border-radius:7px; padding:9px 12px; margin:3px; cursor:pointer; color:var(--ink); }
.nav button.active { background:var(--primary); color:white; border-color:var(--primary); }
.tab-panel { display:none; }
.tab-panel.active { display:block; }
.data-table { border-collapse:collapse; width:100%; font-size:13px; }
.data-table th { background:var(--primary); color:white; text-align:left; padding:8px; position:sticky; top:0; }
.data-table td { border-bottom:1px solid var(--line); padding:7px; vertical-align:top; }
.pill { display:inline-block; padding:4px 8px; border-radius:999px; background:var(--soft); color:var(--primary); margin:2px; font-size:12px; }
.param { display:grid; grid-template-columns:220px 1fr; gap:10px; align-items:center; margin:10px 0; }
.param input, .param select { width:100%; padding:8px; border:1px solid var(--line); border-radius:6px; }
.note { background:#fff7ed; border-left:4px solid var(--accent); padding:12px; border-radius:6px; }
.two { grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); }
.three { grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); margin:16px 0; }
.control-hero { background:linear-gradient(135deg,#ffffff 0%,#f7f6f2 100%); border:1px solid #d9d2c4; border-left:7px solid var(--accent); border-radius:12px; padding:20px; margin:18px 0; box-shadow:0 8px 24px rgba(31,41,51,.06); }
.control-hero h2 { margin:0 0 8px; color:var(--primary); font-size:24px; }
.step-ribbon { display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:10px; margin-top:16px; }
.step-ribbon div { background:white; border:1px solid var(--line); border-radius:8px; padding:10px; }
.step-ribbon span { display:inline-flex; align-items:center; justify-content:center; width:24px; height:24px; border-radius:999px; background:var(--primary); color:white; font-weight:800; margin-right:6px; }
.step-ribbon b { color:var(--primary); }
.step-ribbon small { display:block; color:var(--muted); margin-top:4px; }
.metric { background:white; border:1px solid var(--line); border-radius:8px; padding:14px; }
.metric span { display:block; color:var(--muted); font-size:12px; text-transform:uppercase; font-weight:700; letter-spacing:.04em; }
.metric strong { display:block; color:var(--primary); font-size:22px; margin-top:5px; }
.status { display:inline-block; border-radius:999px; padding:4px 9px; font-weight:800; font-size:12px; }
.status.ok { background:#e8f5e9; color:#1b5e20; border:1px solid #c8e6c9; }
.status.warn { background:#fff4df; color:#8a4b00; border:1px solid #ffd699; }
.status.bad { background:#fdeaea; color:#7f1d1d; border:1px solid #f3b9b9; }
.status.neutral { background:#eef2f6; color:#3f4d5a; border:1px solid #d9e2ec; }
.guide-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:14px; margin:16px 0; }
.guide-card { background:white; border:1px solid var(--line); border-radius:8px; padding:16px; }
.guide-card h3 { margin:0 0 8px; color:var(--primary); }
.formula { font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; background:#f3f6f8; border:1px solid var(--line); border-radius:6px; padding:9px; color:#243447; overflow:auto; }
.text-box { width:100%; min-height:110px; border:1px solid var(--line); border-radius:8px; padding:10px; font-family:inherit; resize:vertical; }
details.guide-detail { background:white; border:1px solid var(--line); border-radius:8px; padding:12px 14px; margin:10px 0; }
details.guide-detail summary { cursor:pointer; color:var(--primary); font-weight:800; }
.chart-card { min-height:360px; }
.toolbar { display:flex; flex-wrap:wrap; align-items:center; gap:10px; margin:12px 0; }
.toolbar input, .toolbar select { padding:8px 10px; border:1px solid var(--line); border-radius:7px; background:white; color:var(--ink); min-width:220px; }
.mini-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:12px; margin:14px 0; }
.screener-chip { display:inline-flex; align-items:center; gap:6px; border:1px solid var(--line); background:white; border-radius:999px; padding:5px 9px; margin:3px; font-size:12px; }
.screener-chip b { color:var(--primary); }
.rank-badge { display:inline-flex; align-items:center; justify-content:center; width:28px; height:28px; border-radius:999px; background:var(--primary); color:white; font-weight:800; }
.table-note { font-size:12px; color:var(--muted); margin:8px 0 0; }
@media (max-width:900px) { .step-ribbon { grid-template-columns:1fr; } }
"""


def dashboard_js() -> str:
    return """
function showTab(id) {
  document.querySelectorAll('.tab-panel').forEach(function(el) { el.classList.remove('active'); });
  document.querySelectorAll('.nav button').forEach(function(el) { el.classList.remove('active'); });
  document.getElementById(id).classList.add('active');
  document.querySelector('[data-tab=\"' + id + '\"]').classList.add('active');
}
function refreshSensitivity() {
  const base = parseFloat(document.getElementById('baseFairValue').value || '0');
  const discount = parseFloat(document.getElementById('discountRate').value || '0.09');
  const growth = parseFloat(document.getElementById('terminalGrowth').value || '0.02');
  const horizon = parseFloat(document.getElementById('dcfHorizon').value || '10');
  const spread = Math.max(discount - growth, 0.005);
  const adjusted = base * (0.07 / spread) * (1 + (horizon - 10) * 0.015);
  document.getElementById('sensitivityResult').innerText = isFinite(adjusted) ? adjusted.toFixed(2) : 'n/a';
}
function filterScreenerTable() {
  const q = (document.getElementById('screenerSearch') || {}).value || '';
  const sector = (document.getElementById('screenerSector') || {}).value || '';
  const minScoreRaw = (document.getElementById('screenerMinScore') || {}).value || '';
  const minScore = minScoreRaw === '' ? null : parseFloat(minScoreRaw);
  const table = document.querySelector('#screenerInteractiveTable table');
  if (!table) return;
  const headers = Array.from(table.querySelectorAll('thead th')).map(function(th) { return th.innerText.trim().toLowerCase(); });
  const tickerIdx = headers.indexOf('ticker');
  const nameIdx = headers.indexOf('company_name');
  const sectorIdx = headers.indexOf('sector');
  const scoreIdx = headers.indexOf('screener_score');
  let shown = 0;
  Array.from(table.querySelectorAll('tbody tr')).forEach(function(row) {
    const cells = Array.from(row.children).map(function(td) { return td.innerText.trim(); });
    const text = [cells[tickerIdx] || '', cells[nameIdx] || ''].join(' ').toLowerCase();
    const rowSector = (cells[sectorIdx] || '').toLowerCase();
    const score = parseFloat(cells[scoreIdx] || 'NaN');
    const matchText = !q || text.indexOf(q.toLowerCase()) >= 0;
    const matchSector = !sector || rowSector === sector.toLowerCase();
    const matchScore = minScore === null || (!isNaN(score) && score >= minScore);
    const visible = matchText && matchSector && matchScore;
    row.style.display = visible ? '' : 'none';
    if (visible) shown += 1;
  });
  const count = document.getElementById('screenerVisibleRows');
  if (count) count.innerText = shown.toString();
}
window.addEventListener('DOMContentLoaded', function() {
  showTab('tab-control');
  ['baseFairValue','discountRate','terminalGrowth','dcfHorizon'].forEach(function(id) {
    const el = document.getElementById(id);
    if (el) el.addEventListener('input', refreshSensitivity);
  });
  refreshSensitivity();
  ['screenerSearch','screenerSector','screenerMinScore'].forEach(function(id) {
    const el = document.getElementById(id);
    if (el) el.addEventListener('input', filterScreenerTable);
    if (el) el.addEventListener('change', filterScreenerTable);
  });
  filterScreenerTable();
});
"""


def build_kpis(namespace: Mapping[str, Any]) -> str:
    row = _latest_row(namespace)
    ranking = _as_df(_first(namespace, "companyranking", "company_ranking"))
    qa = _as_df(_first(namespace, "qa_report", "robustness_table", "diagnostics_table"))
    kpis = [
        ("Target ticker", _target_ticker(namespace)),
        ("Latest price", row.get("adj_close", row.get("price", row.get("close", pd.NA))) if not row.empty else pd.NA),
        ("Fair value", row.get("fair_value_estimate", row.get("target_price", row.get("fair_value", pd.NA))) if not row.empty else pd.NA),
        ("Upside", row.get("upside_to_fair_value", row.get("upside", pd.NA)) if not row.empty else pd.NA),
        ("Ranked companies", len(ranking) if not ranking.empty else 0),
        ("QA rows", len(qa) if not qa.empty else 0),
    ]
    return "<section class='grid kpi-grid'>" + "".join(
        f"<div class='card kpi'><div class='label'>{escape(label)}</div><div class='value'>{_format_value(value)}</div></div>"
        for label, value in kpis
    ) + "</section>"


def build_parameter_lab(namespace: Mapping[str, Any]) -> str:
    experiment = _first(namespace, "EXPERIMENT", default={}) or {}
    master = _first(namespace, "MASTER_REQUEST", default={}) or {}
    user_selection = _first(namespace, "USER_SELECTION", default={}) or {}
    row = _latest_row(namespace)
    base_fv = row.get("fair_value_estimate", row.get("target_price", row.get("fair_value", 100.0))) if not row.empty else 100.0
    try:
        base_fv = float(base_fv)
    except Exception:
        base_fv = 100.0
    active_blocks = experiment.get("active_feature_blocks") or experiment.get("feature_blocks") or []
    if isinstance(active_blocks, str):
        active_blocks = [active_blocks]
    chips = "".join(f"<span class='pill'>{escape(str(block))}</span>" for block in active_blocks) or "<span class='pill'>not configured</span>"
    guide = pd.DataFrame(PARAMETER_GUIDE)
    current_params = pd.DataFrame([
        {"scope": "MASTER_REQUEST", "parameter": k, "value": v}
        for k, v in master.items()
        if k in {"ticker", "target_ticker", "start_date", "end_date", "peer_selection_method", "n_peers", "manual_peers", "market", "currency"}
    ] + [
        {"scope": "EXPERIMENT", "parameter": k, "value": v}
        for k, v in experiment.items()
        if k in {"dcf_horizon_years", "dcf_perpetual_growth", "discount_rate", "cost_of_equity", "wacc", "active_feature_blocks", "feature_blocks", "test_start", "embargo"}
    ] + [
        {"scope": "USER_SELECTION", "parameter": k, "value": v}
        for k, v in user_selection.items()
        if k in {"data_source_choice", "selected_company", "valuation_target", "ranking_metric"}
    ])
    return f"""
<section class='grid two'>
  <div class='card'>
    <h2>Parameter Lab</h2>
    <p class='muted'>These controls are an exported HTML sensitivity layer. To change the actual model run, update MASTER_REQUEST / EXPERIMENT in the notebook and rerun the pipeline.</p>
    <div class='param'><label>Base fair value</label><input id='baseFairValue' type='number' step='0.01' value='{base_fv:.4f}'></div>
    <div class='param'><label>Discount rate</label><input id='discountRate' type='number' step='0.001' value='{float(experiment.get("discount_rate", experiment.get("cost_of_equity", 0.09)) or 0.09):.4f}'></div>
    <div class='param'><label>Terminal growth</label><input id='terminalGrowth' type='number' step='0.001' value='{float(experiment.get("dcf_perpetual_growth", experiment.get("terminal_growth", 0.02)) or 0.02):.4f}'></div>
    <div class='param'><label>DCF horizon years</label><input id='dcfHorizon' type='number' min='1' max='30' step='1' value='{int(experiment.get("dcf_horizon_years", 10) or 10)}'></div>
    <div class='note'><strong>Illustrative sensitivity fair value:</strong> <span id='sensitivityResult'>n/a</span></div>
    <p><strong>Active feature blocks:</strong><br>{chips}</p>
  </div>
  {_table_html(current_params, "Current Run Parameters", 50)}
</section>
{_table_html(guide, "Parameter Explanations", 100)}
"""


def build_methodology_guide_section(namespace: Mapping[str, Any]) -> str:
    formula_cards = []
    for row in FORMULA_GUIDE:
        formula_cards.append(f"""
<div class='guide-card'>
  <h3>{escape(row['area'])}</h3>
  <div class='formula'>{escape(row['formula'])}</div>
  <p>{escape(row['plain_english'])}</p>
  <p><b>Used for:</b> {escape(row['used_for'])}</p>
  <p class='muted'><b>Watch out:</b> {escape(row['watch_out'])}</p>
</div>
""")
    feature_details = []
    for row in FEATURE_GUIDE:
        feature_details.append(f"""
<details class='guide-detail'>
  <summary>{escape(row['block'])}</summary>
  <p><b>Examples:</b> {escape(row['examples'])}</p>
  <p><b>Intuition:</b> {escape(row['intuition'])}</p>
  <p class='muted'><b>Risk:</b> {escape(row['risk'])}</p>
</details>
""")
    return f"""
<section class='control-hero'>
  <h2>Methodology and Feature Guide</h2>
  <p>Use this section as the friendly map of the notebook: formulas, feature blocks, modelling intuition and analyst notes in one place.</p>
</section>
<section class='card'>
  <h2>Analyst Notes</h2>
  <p class='muted'>This text area is exported for review notes. It does not change model results; update notebook configs and rerun for actual changes.</p>
  <textarea class='text-box' placeholder='Write thesis notes, caveats, assumptions, or questions for the next run...'></textarea>
</section>
<section class='card'>
  <h2>Core Formulas</h2>
  <div class='guide-grid'>{''.join(formula_cards)}</div>
</section>
<section class='card'>
  <h2>Feature Blocks Explained</h2>
  {''.join(feature_details)}
</section>
{_table_html(pd.DataFrame(PARAMETER_GUIDE), "Parameter Dictionary", 100)}
"""


def _status_badge(status: Any) -> str:
    text = escape(str(status if status is not None else "n/a"))
    cls = "ok" if text.upper() in {"OK", "PASS", "READY", "AVAILABLE", "LOADED"} else "warn" if text.upper() in {"WARN", "WARNING", "PARTIAL", "SKIP"} else "bad" if text.upper() in {"FAIL", "FAILED", "ERROR"} else "neutral"
    return f"<span class='status {cls}'>{text}</span>"


def build_control_center_panel(namespace: Mapping[str, Any]) -> str:
    """Build the notebook operating surface inspired by the portfolio platform.

    This is an exported, HTML-safe control center: it does not mutate the model,
    but it shows what must be changed in the notebook configs before rerunning.
    """
    master = _first(namespace, "MASTER_REQUEST", "MASTERREQUEST", default={}) or {}
    user_selection = _first(namespace, "USER_SELECTION", "USERSELECTION", default={}) or {}
    experiment = _first(namespace, "EXPERIMENT", default={}) or {}
    valuation_config = _first(namespace, "VALUATIONCONFIG", "VALUATION_CONFIG", default={}) or {}
    ml_config = _first(namespace, "MLCONFIG", "ML_CONFIG", default={}) or {}
    governance_config = _first(namespace, "GOVERNANCECONFIG", "GOVERNANCE_CONFIG", default={}) or {}
    fundamentals_config = _first(namespace, "FUNDAMENTALSCONFIG", "FUNDAMENTALS_CONFIG", default={}) or {}
    universe_config = _first(namespace, "UNIVERSECONFIG", "UNIVERSE_CONFIG", default={}) or {}

    provider_registry = _as_df(namespace.get("provider_registry"))
    api_key_status = _as_df(namespace.get("api_key_status"))
    universe_master = _as_df(namespace.get("universe_master"))
    ml_leaderboard = _as_df(namespace.get("ml_leaderboard"))
    valuation_model_registry = _as_df(namespace.get("valuation_model_registry"))
    extended_valuation_results = _as_df(namespace.get("extended_valuation_results"))
    qa = _as_df(_first(namespace, "qa_report", "robustness_table", "diagnostics_table"))
    refresh_plan = _as_df(namespace.get("refresh_plan"))
    runtime_environment = _as_df(namespace.get("runtime_environment"))
    source_provenance = _as_df(namespace.get("refresh_provenance"))
    drive_sync_manifest = _as_df(namespace.get("drive_sync_manifest"))

    dashboard_path = namespace.get("final_dashboard_path") or namespace.get("dashboard_path")
    report_path = namespace.get("final_report_path") or namespace.get("report_path")

    readiness = pd.DataFrame([
        {"area": "Configuration", "status": "PASS" if master or experiment else "WARN", "detail": "MASTER_REQUEST / EXPERIMENT available" if master or experiment else "Run the setup/control cells first"},
        {"area": "Universe", "status": "PASS" if not universe_master.empty else "WARN", "detail": f"{len(universe_master)} securities in universe_master"},
        {"area": "Providers", "status": "PASS" if not provider_registry.empty else "WARN", "detail": f"{len(provider_registry)} provider registry rows"},
        {"area": "API keys", "status": "PASS" if not api_key_status.empty and api_key_status.get("configured", pd.Series(dtype=bool)).any() else "WARN", "detail": f"{int(api_key_status.get('configured', pd.Series(dtype=bool)).sum()) if not api_key_status.empty else 0} provider keys configured"},
        {"area": "Valuation models", "status": "PASS" if not valuation_model_registry.empty else "WARN", "detail": f"{len(valuation_model_registry)} valuation models registered; {len(extended_valuation_results)} result rows"},
        {"area": "ML layer", "status": "PASS" if not ml_leaderboard.empty and not (ml_leaderboard.get("status", pd.Series()).astype(str).eq("WARN").all()) else "WARN", "detail": f"{len(ml_leaderboard)} model rows"},
        {"area": "QA", "status": "PASS" if not qa.empty else "WARN", "detail": f"{len(qa)} diagnostic rows"},
        {"area": "Artifacts", "status": "PASS" if dashboard_path and report_path else "WARN", "detail": "Dashboard/report paths available" if dashboard_path and report_path else "Export cells not run yet"},
    ])

    config_snapshot = pd.DataFrame([
        {"scope": "MASTER_REQUEST", "keys": len(master), "sample": str(dict(list(master.items())[:8])) if isinstance(master, dict) else str(master)},
        {"scope": "USER_SELECTION", "keys": len(user_selection), "sample": str(dict(list(user_selection.items())[:8])) if isinstance(user_selection, dict) else str(user_selection)},
        {"scope": "EXPERIMENT", "keys": len(experiment), "sample": str(dict(list(experiment.items())[:8])) if isinstance(experiment, dict) else str(experiment)},
        {"scope": "VALUATIONCONFIG", "keys": len(valuation_config), "sample": str(dict(list(valuation_config.items())[:8])) if isinstance(valuation_config, dict) else str(valuation_config)},
        {"scope": "MLCONFIG", "keys": len(ml_config), "sample": str(dict(list(ml_config.items())[:8])) if isinstance(ml_config, dict) else str(ml_config)},
        {"scope": "GOVERNANCECONFIG", "keys": len(governance_config), "sample": str(dict(list(governance_config.items())[:8])) if isinstance(governance_config, dict) else str(governance_config)},
        {"scope": "FUNDAMENTALSCONFIG", "keys": len(fundamentals_config), "sample": str(dict(list(fundamentals_config.items())[:8])) if isinstance(fundamentals_config, dict) else str(fundamentals_config)},
        {"scope": "UNIVERSECONFIG", "keys": len(universe_config), "sample": str(dict(list(universe_config.items())[:8])) if isinstance(universe_config, dict) else str(universe_config)},
    ])

    workflow = pd.DataFrame([
        {"step": 1, "action": "Input Setup", "notebook cell": "0.1.5 / MASTER_REQUEST UI", "output": "Target, peers, models, valuation parameters"},
        {"step": 2, "action": "Apply Configuration", "notebook cell": "config sync cells", "output": "MASTER_REQUEST, USER_SELECTION, EXPERIMENT"},
        {"step": 3, "action": "Run Valuation Core", "notebook cell": "data, valuation, scenario, QA sections", "output": "dfmerged, latestcrosssection, companyranking"},
        {"step": 4, "action": "Run Research Platform", "notebook cell": "36.1-36.5", "output": "provider_registry, universe_master, ml_leaderboard"},
        {"step": 5, "action": "Review Outputs", "notebook cell": "final export cells", "output": "Dashboard, report, control center HTML"},
    ])

    artifact_links = pd.DataFrame([
        {"artifact": "Navigable Dashboard", "path": str(dashboard_path or "not exported yet")},
        {"artifact": "Research Report", "path": str(report_path or "not exported yet")},
        {"artifact": "Output Root", "path": str(_output_root(namespace))},
    ])

    readiness_html = readiness.copy()
    if not readiness_html.empty:
        readiness_html["status"] = readiness_html["status"].map(_status_badge)

    return f"""
<section class='control-hero'>
  <h2>Company Valuation Control Center</h2>
  <p>Start here: configure the company, universe, providers, model depth, refresh behavior, diagnostics and final exports. This page mirrors the Portfolio Research Platform Pro operating surface while preserving the company valuation methodology.</p>
  <div class='step-ribbon'>
    <div><span>1</span><b>Input</b><small>ticker, peers, dates</small></div>
    <div><span>2</span><b>Sync</b><small>canonical configs</small></div>
    <div><span>3</span><b>Research</b><small>valuation + ML</small></div>
    <div><span>4</span><b>Diagnose</b><small>coverage, QA, leakage</small></div>
    <div><span>5</span><b>Export</b><small>dashboard + report</small></div>
  </div>
</section>
<section class='grid three'>
  <div class='metric'><span>Target</span><strong>{escape(_target_ticker(namespace))}</strong></div>
  <div class='metric'><span>Universe</span><strong>{len(universe_master):,}</strong></div>
  <div class='metric'><span>Providers</span><strong>{len(provider_registry):,}</strong></div>
  <div class='metric'><span>API Keys</span><strong>{int(api_key_status.get('configured', pd.Series(dtype=bool)).sum()) if not api_key_status.empty else 0}</strong></div>
  <div class='metric'><span>Valuation Models</span><strong>{len(valuation_model_registry):,}</strong></div>
  <div class='metric'><span>ML Rows</span><strong>{len(ml_leaderboard):,}</strong></div>
  <div class='metric'><span>QA Checks</span><strong>{len(qa):,}</strong></div>
  <div class='metric'><span>Runtime</span><strong>{escape(str(runtime_environment.iloc[0].get("runtime", "n/a")) if not runtime_environment.empty else "n/a")}</strong></div>
</section>
{_table_html(readiness_html, "Control Center Readiness", 20)}
{_table_html(workflow, "Operating Workflow", 20)}
{_table_html(config_snapshot, "Configuration Snapshot", 20)}
{_table_html(refresh_plan, "Refresh and Scheduling Plan", 20)}
{_table_html(provider_registry, "Provider Registry", 60)}
{_table_html(api_key_status, "API Key Status", 30)}
{_table_html(valuation_model_registry, "Valuation Model Registry", 100)}
{_table_html(extended_valuation_results, "Expanded Valuation Results", 100)}
{_table_html(universe_master, "Universe Preview", 80)}
{_table_html(source_provenance, "Refresh Provenance", 80)}
{_table_html(drive_sync_manifest, "Google Drive Database Sync Manifest", 100)}
{_table_html(artifact_links, "Artifact Links", 20)}
{build_methodology_guide_section(namespace)}
"""


def build_control_center_html(namespace: Mapping[str, Any]) -> str:
    ticker = _target_ticker(namespace)
    generated = datetime.now().isoformat(timespec="seconds")
    return f"""<!doctype html>
<html>
<head>
  <meta charset='utf-8'>
  <meta name='viewport' content='width=device-width, initial-scale=1'>
  <title>Company Valuation Control Center - {escape(ticker)}</title>
  <style>{dashboard_css()}</style>
  <script>{dashboard_js()}</script>
</head>
<body>
  <div class='shell'>
    <header class='hero'>
      <h1>Company Valuation Research Platform</h1>
      <p>Control Center · {escape(ticker)} · Generated: {escape(generated)}</p>
    </header>
    {build_control_center_panel(namespace)}
  </div>
</body>
</html>"""


def build_sources_section(namespace: Mapping[str, Any]) -> str:
    sources = pd.DataFrame(OPEN_SOURCE_SOURCES)
    platform = namespace.get("research_platform_outputs", {}) or {}
    provider_registry = _as_df(namespace.get("provider_registry"))
    api_key_status = _as_df(namespace.get("api_key_status"))
    universe_provenance = _as_df(namespace.get("universe_provenance"))
    refresh_provenance = _as_df(namespace.get("refresh_provenance"))
    drive_sync_manifest = _as_df(namespace.get("drive_sync_manifest"))
    runtime_environment = _as_df(namespace.get("runtime_environment"))
    availability = []
    for env_name in ["FMP_API_KEY", "fmp_api_key", "FRED_API_KEY", "ALPHAVANTAGE_API_KEY"]:
        availability.append({"api_key": env_name, "configured": "checked at runtime by notebook / environment"})
    return (
        _table_html(sources, "Open-source and API Data Sources", 100)
        + _table_html(provider_registry, "Provider Registry and Fallback Order", 100)
        + _table_html(api_key_status, "API Key Status", 30)
        + _table_html(runtime_environment, "Runtime Environment", 20)
        + _table_html(pd.DataFrame(availability), "API Configuration Notes", 20)
        + _table_html(platform.get("source_provenance"), "Runtime Source Provenance", 100)
        + _table_html(universe_provenance, "Universe Source Provenance", 100)
        + _table_html(refresh_provenance, "Refresh Provenance", 100)
        + _table_html(drive_sync_manifest, "Google Drive Database Sync Manifest", 100)
        + _table_html(platform.get("data_inventory"), "Local Database Inventory", 100)
    )


def build_screener_section(namespace: Mapping[str, Any]) -> str:
    results = _as_df(namespace.get("screener_results"))
    audit = _as_df(namespace.get("screener_audit"))
    progression = _as_df(namespace.get("screener_progression"))
    summary = _as_df(namespace.get("screener_summary"))
    config = namespace.get("screener_config", {}) or {}
    schema = _as_df(namespace.get("screener_schema"))
    presets = _as_df(namespace.get("screener_presets"))
    data_dictionary = _as_df(namespace.get("screener_data_dictionary"))
    export_paths = namespace.get("screener_export_paths", {}) or {}
    active_filters = pd.DataFrame(config.get("filters", [])) if isinstance(config, dict) else pd.DataFrame()
    preset = config.get("preset", "custom") if isinstance(config, dict) else "custom"
    ranking = config.get("ranking", {}) if isinstance(config, dict) else {}
    metrics = [
        ("Active preset", preset),
        ("Matches", len(results) if not results.empty else 0),
        ("Filters", len(active_filters) if not active_filters.empty else 0),
        ("Skipped filters", int((audit.get("status", pd.Series(dtype=str)) == "SKIP").sum()) if not audit.empty else 0),
        ("Ranking", ranking.get("mode", "preset") if isinstance(ranking, dict) else "preset"),
    ]
    kpi_html = "<section class='grid kpi-grid'>" + "".join(
        f"<div class='card kpi'><div class='label'>{escape(label)}</div><div class='value'>{_format_value(value)}</div></div>"
        for label, value in metrics
    ) + "</section>"
    filter_chips = ""
    if not active_filters.empty:
        for _, row in active_filters.iterrows():
            filter_chips += (
                "<span class='screener-chip'>"
                f"<b>{escape(str(row.get('field', 'field')))}</b>"
                f"{escape(str(row.get('op', row.get('operator', ''))))}"
                f"{escape(str(row.get('value', '')))}"
                "</span>"
            )
    else:
        filter_chips = "<span class='screener-chip'><b>Preset only</b>No custom filters</span>"
    interpretation_html = (
        "<section class='card'><h2>How To Read This Screener</h2>"
        "<p class='muted'>The screener is a research shortlist, not a trading signal. Ranking modes combine available valuation, quality, momentum, risk and ML fields after alias resolution. Missing fields are skipped and shown in the audit trail, so coverage and staleness should be reviewed before acting on any result.</p>"
        "</section>"
    )
    export_links = pd.DataFrame(
        [{"artifact": key, "path": str(path)} for key, path in export_paths.items()]
    )
    sector_options = ""
    if not results.empty and "sector" in results.columns:
        sectors = sorted(x for x in results["sector"].dropna().astype(str).unique() if x)
        sector_options = "".join(f"<option value='{escape(sector)}'>{escape(sector)}</option>" for sector in sectors)
    result_html = "<section class='card' id='screenerInteractiveTable'><h2>Interactive Screener Results</h2>"
    if results.empty:
        result_html += "<p class='muted'>No screener results available.</p></section>"
    else:
        display_cols = [c for c in [
            "screener_rank", "ticker", "company_name", "sector", "country", "market_cap",
            "pe_ratio", "pb_ratio", "ev_ebitda", "quality_score", "valuation_score",
            "momentum_score", "risk_score", "upside_to_fair_value", "screener_score",
            "robustness_status",
        ] if c in results.columns]
        result_table = results[display_cols or list(results.columns)].head(250).copy()
        if "screener_rank" in result_table.columns:
            result_table["screener_rank"] = result_table["screener_rank"].map(lambda x: f"<span class='rank-badge'>{escape(str(x))}</span>")
            table = result_table.to_html(index=False, classes="data-table", border=0, escape=False)
        else:
            table = result_table.to_html(index=False, classes="data-table", border=0, escape=True)
        result_html += f"""
<div class='toolbar'>
  <input id='screenerSearch' type='search' placeholder='Search ticker or company'>
  <select id='screenerSector'><option value=''>All sectors</option>{sector_options}</select>
  <input id='screenerMinScore' type='number' step='0.01' placeholder='Min screener_score'>
  <span class='pill'><span id='screenerVisibleRows'>{len(result_table)}</span> visible</span>
</div>
{table}
<p class='table-note'>Client-side filters affect only this exported dashboard table. Re-run the notebook UI to change official screener outputs.</p>
</section>
"""
    chart_html = ""
    try:
        import plotly.express as px
        if not results.empty and "sector" in results.columns:
            sector_counts = results["sector"].fillna("Unknown").value_counts().reset_index()
            sector_counts.columns = ["sector", "count"]
            chart_html += _plotly_html(px.bar(sector_counts, x="sector", y="count", title="Screener Sector Mix", template="plotly_white"), "Sector Mix")
        if not results.empty and {"valuation_score", "quality_score"}.issubset(results.columns):
            chart_html += _plotly_html(px.scatter(results, x="valuation_score", y="quality_score", hover_name="ticker" if "ticker" in results.columns else None, color="screener_score" if "screener_score" in results.columns else None, title="Valuation vs Quality", template="plotly_white"), "Valuation vs Quality")
    except Exception:
        chart_html = ""
    return (
        "<section class='control-hero'><h2>Integrated Stock Screener</h2><p>FINVIZ-style functionality using the notebook's own data, valuation, ML and diagnostics layers first. Missing fields are skipped and recorded in the audit table.</p></section>"
        + kpi_html
        + interpretation_html
        + f"<section class='card'><h2>Active Filter Set</h2>{filter_chips}</section>"
        + _table_html(summary, "Screener Summary", 20)
        + _table_html(active_filters, "Active Structured Filters", 100)
        + result_html
        + chart_html
        + _table_html(audit, "Filter Audit", 100)
        + _table_html(progression, "Count Progression", 100)
        + _table_html(data_dictionary, "Screener Data Dictionary and Availability", 200)
        + _table_html(export_links, "Screener Export Paths", 30)
        + _table_html(presets, "Preset Library", 100)
        + _table_html(schema, "Filter Schema", 200)
    )


def build_finviz_platform_section(namespace: Mapping[str, Any], view: str = "overview") -> str:
    architecture = _as_df(namespace.get("finviz_architecture_review"))
    blueprint = _as_df(namespace.get("finviz_module_blueprint"))
    schema = _as_df(namespace.get("finviz_data_model_schema"))
    securities = _as_df(namespace.get("finviz_securities_master"))
    groups = _as_df(namespace.get("finviz_group_summaries"))
    maps = _as_df(namespace.get("finviz_map_payload"))
    watchlists = _as_df(namespace.get("finviz_watchlists"))
    events = _as_df(namespace.get("finviz_events"))
    macro = _as_df(namespace.get("finviz_macro_risk"))
    alerts = _as_df(namespace.get("finviz_alerts"))
    quick = _as_df(namespace.get("finviz_quick_view"))
    manifest = _as_df(namespace.get("finviz_export_manifest"))
    figures = namespace.get("finviz_figures", {}) if isinstance(namespace.get("finviz_figures"), dict) else {}

    hero = (
        "<section class='control-hero'><h2>Finviz-Inspired Research Layer</h2>"
        "<p>Reproducible Finviz-style concepts implemented on top of this notebook's own data lake, valuation engine, ML outputs, QA and dashboard artifacts. Missing vendor-only datasets are exposed as unavailable instead of fabricated.</p></section>"
    )
    caveat = (
        "<section class='card'><h2>Interpretation Notes</h2>"
        "<p class='muted'>Maps and groups summarize the current cross-section. They are useful for triage and coverage diagnosis, but they compress uncertainty: stale fundamentals, partial event data and model dispersion remain visible in the detail tables and should guide final analyst judgment.</p>"
        "</section>"
    )
    kpis = [
        ("Securities", len(securities)),
        ("Groups", len(groups)),
        ("Map Rows", len(maps)),
        ("Watchlist Rows", len(watchlists)),
        ("Events", len(events)),
        ("Alerts", len(alerts)),
    ]
    kpi_html = "<section class='grid kpi-grid'>" + "".join(
        f"<div class='card kpi'><div class='label'>{escape(label)}</div><div class='value'>{_format_value(value)}</div></div>"
        for label, value in kpis
    ) + "</section>"

    if view == "maps":
        chart_html = ""
        if "finviz_sector_map" in figures:
            chart_html += _plotly_html(figures["finviz_sector_map"], "Sector / Industry Market Map")
        return hero + kpi_html + caveat + chart_html + _table_html(maps, "Map Payload and Drill-Down Table", 250)
    if view == "groups":
        chart_html = ""
        if "finviz_groups_bar" in figures:
            chart_html += _plotly_html(figures["finviz_groups_bar"], "Group Ranking")
        return hero + caveat + chart_html + _table_html(groups, "Group Summaries", 250)
    if view == "watchlist":
        return hero + caveat + _table_html(watchlists, "Watchlists and Candidate Portfolio", 250) + _table_html(quick, "Selected Ticker Quick View", 20)
    if view == "events":
        return hero + caveat + _table_html(events, "Events / News / Insider / Calendar Payload", 250) + _table_html(alerts, "Alert Log", 250)
    if view == "macro":
        return hero + caveat + _table_html(macro, "Futures / Forex / Crypto / Macro-Risk Board", 250)
    return (
        hero
        + kpi_html
        + caveat
        + _table_html(architecture, "Architecture Review", 50)
        + _table_html(blueprint, "Module Blueprint", 50)
        + _table_html(schema, "Unified Data Model", 100)
        + _table_html(securities, "Securities Master", 100)
        + _table_html(manifest, "API-Ready Export Manifest", 100)
    )


def build_dashboard_html(namespace: Mapping[str, Any]) -> str:
    ticker = _target_ticker(namespace)
    ranking = _as_df(_first(namespace, "companyranking", "company_ranking"))
    latest = _as_df(_first(namespace, "latestcrosssection", "latest_cross_section", "latest_crosssection"))
    valuation = _as_df(_first(namespace, "valuation_output", "valuationoutput", "dfvaluation", "df_valuation"))
    qa = _as_df(_first(namespace, "qa_report", "robustness_table", "diagnostics_table"))
    merged = _as_df(_first(namespace, "dfmerged", "df_merged"))
    model = _as_df(_first(namespace, "dfmodel", "df_model"))
    platform = namespace.get("research_platform_outputs", {}) or {}
    peeranalysis = platform.get("peeranalysis", namespace.get("peeranalysis", {})) or {}
    macro_risk = _as_df(platform.get("macro_risk", namespace.get("macro_risk")))
    model_results = _as_df(platform.get("model_results", namespace.get("model_results")))
    feature_catalog = _as_df(platform.get("feature_catalog", namespace.get("feature_catalog")))
    target_catalog = _as_df(platform.get("target_catalog", namespace.get("target_catalog")))
    recommendations = _as_df(platform.get("recommendations", namespace.get("platform_recommendations")))
    valuation_model_registry = _as_df(namespace.get("valuation_model_registry"))
    extended_valuation_results = _as_df(namespace.get("extended_valuation_results"))
    valuation_assumption_table = _as_df(namespace.get("valuation_assumption_table"))
    valuation_gap_table = _as_df(namespace.get("valuation_gap_table"))
    universe_master = _as_df(namespace.get("universe_master"))
    ml_leaderboard = _as_df(namespace.get("ml_leaderboard"))
    ml_predictions = _as_df(namespace.get("ml_predictions"))
    ml_feature_importance = _as_df(namespace.get("ml_feature_importance"))
    ml_validation_windows = _as_df(namespace.get("ml_validation_windows"))
    refresh_plan = _as_df(namespace.get("refresh_plan"))
    figures = _figure_dict(namespace)
    figure_blocks = "\n".join(
        _plotly_html(fig, str(name).replace("_", " ").title())
        for name, fig in figures.items()
    ) or "<section class='card'><h2>Interactive Figures</h2><p class='muted'>No Plotly figures were found in interactivefigures / interactive_figures.</p></section>"

    tabs = {
        "tab-control": build_control_center_panel(namespace),
        "tab-guide": build_methodology_guide_section(namespace),
        "tab-executive": build_kpis(namespace)
        + "<section class='card'><h2>Executive Summary</h2><p>This dashboard consolidates market prices, fundamentals, valuation models, QA diagnostics, and open-source/API provenance for the selected company.</p></section>"
        + _table_html(ranking, "Company Ranking", 25),
        "tab-valuation": _table_html(valuation if not valuation.empty else latest, "Valuation and Fundamentals", 30)
        + _table_html(valuation_model_registry, "Expanded Valuation Model Registry", 100)
        + _table_html(extended_valuation_results, "Expanded Valuation Results", 200)
        + _table_html(valuation_gap_table, "Blended Extended Fair Value", 100)
        + _table_html(valuation_assumption_table, "Valuation Assumptions", 20),
        "tab-peers": _table_html(peeranalysis.get("selected_peers"), "Selected Comparable Companies", 30)
        + _table_html(peeranalysis.get("peer_similarity"), "Peer Similarity Diagnostics", 30)
        + _table_html(peeranalysis.get("target_vs_peers"), "Target vs Peer Median Multiples", 30)
        + _table_html(peeranalysis.get("peer_warnings"), "Peer Warnings", 20),
        "tab-models": _table_html(model_results, "Model Registry Results", 50)
        + _table_html(ml_leaderboard, "Advanced ML Leaderboard", 50)
        + _table_html(ml_validation_windows, "Time-Aware Validation Windows", 50)
        + _table_html(ml_feature_importance, "ML Feature Importance", 100)
        + _table_html(ml_predictions, "ML Holdout Predictions", 100)
        + _table_html(feature_catalog, "Feature Block Coverage", 50)
        + _table_html(target_catalog, "Return Target Catalog", 50),
        "tab-macro": _table_html(macro_risk, "Macro-Risk Layer: Rates, FX, Commodities", 100),
        "tab-universe": _table_html(universe_master, "Universe Master Table", 200)
        + _table_html(refresh_plan, "Continuous Refresh Plan", 50),
        "tab-screener": build_screener_section(namespace),
        "tab-market-overview": build_finviz_platform_section(namespace, "overview"),
        "tab-maps": build_finviz_platform_section(namespace, "maps"),
        "tab-groups": build_finviz_platform_section(namespace, "groups"),
        "tab-watchlist": build_finviz_platform_section(namespace, "watchlist"),
        "tab-events": build_finviz_platform_section(namespace, "events"),
        "tab-figures": figure_blocks,
        "tab-qa": _table_html(qa, "Diagnostics and QA", 50)
        + _table_html(merged.tail(20) if not merged.empty else pd.DataFrame(), "Merged Panel Sample", 20)
        + _table_html(model.tail(20) if not model.empty else pd.DataFrame(), "Model Panel Sample", 20)
        + _table_html(recommendations, "Platform Recommendations", 20),
        "tab-parameters": build_parameter_lab(namespace),
        "tab-sources": build_sources_section(namespace),
    }
    nav = [
        ("tab-control", "Control Center"),
        ("tab-guide", "Guide"),
        ("tab-executive", "Executive"),
        ("tab-valuation", "Valuation"),
        ("tab-peers", "Peers"),
        ("tab-models", "Models"),
        ("tab-macro", "Macro Risk"),
        ("tab-universe", "Universe"),
        ("tab-screener", "Screener"),
        ("tab-market-overview", "Market Overview"),
        ("tab-maps", "Maps"),
        ("tab-groups", "Groups"),
        ("tab-watchlist", "Watchlist"),
        ("tab-events", "Events & Alerts"),
        ("tab-figures", "Visuals"),
        ("tab-qa", "QA"),
        ("tab-parameters", "Parameters"),
        ("tab-sources", "Sources & APIs"),
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
  <title>Company Valuation Dashboard - {escape(ticker)}</title>
  <script src='https://cdn.plot.ly/plotly-latest.min.js'></script>
  <style>{dashboard_css()}</style>
</head>
<body>
  <div class='shell'>
    <header class='hero'>
      <h1>Company Valuation Market/Fundamentals Dashboard</h1>
      <p class='muted'>Ticker: {escape(ticker)} · Generated: {escape(generated)}</p>
    </header>
    {nav_html}
    {panels}
  </div>
  <script>{dashboard_js()}</script>
</body>
</html>"""


def build_report_html(namespace: Mapping[str, Any]) -> str:
    ticker = _target_ticker(namespace)
    experiment = _first(namespace, "EXPERIMENT", default={}) or {}
    master = _first(namespace, "MASTER_REQUEST", default={}) or {}
    ranking = _as_df(_first(namespace, "companyranking", "company_ranking"))
    latest = _as_df(_first(namespace, "latestcrosssection", "latest_cross_section", "latest_crosssection"))
    qa = _as_df(_first(namespace, "qa_report", "robustness_table", "diagnostics_table"))
    platform = namespace.get("research_platform_outputs", {}) or {}
    peeranalysis = platform.get("peeranalysis", namespace.get("peeranalysis", {})) or {}
    valuation_model_registry = _as_df(namespace.get("valuation_model_registry"))
    extended_valuation_results = _as_df(namespace.get("extended_valuation_results"))
    valuation_gap_table = _as_df(namespace.get("valuation_gap_table"))
    universe_master = _as_df(namespace.get("universe_master"))
    ml_leaderboard = _as_df(namespace.get("ml_leaderboard"))
    ml_feature_importance = _as_df(namespace.get("ml_feature_importance"))
    refresh_plan = _as_df(namespace.get("refresh_plan"))
    screener_results = _as_df(namespace.get("screener_results"))
    screener_summary = _as_df(namespace.get("screener_summary"))
    screener_audit = _as_df(namespace.get("screener_audit"))
    finviz_groups = _as_df(namespace.get("finviz_group_summaries"))
    finviz_maps = _as_df(namespace.get("finviz_map_payload"))
    finviz_watchlists = _as_df(namespace.get("finviz_watchlists"))
    finviz_events = _as_df(namespace.get("finviz_events"))
    finviz_alerts = _as_df(namespace.get("finviz_alerts"))
    finviz_manifest = _as_df(namespace.get("finviz_export_manifest"))
    report_sections = [
        "<section class='card'><h2>Methodology</h2><p>The report preserves the notebook methodology: DCF, residual income, dividend discount model, relative valuation, scenario analysis, ML overlay, and time-aware as-of market/fundamentals merge with effective fundamental dates.</p></section>",
        _table_html(pd.DataFrame([master]), "MASTER_REQUEST", 5),
        _table_html(pd.DataFrame([experiment]), "EXPERIMENT", 5),
        _table_html(latest, "Latest Cross Section", 30),
        _table_html(ranking, "Company Ranking", 30),
        _table_html(qa, "QA and Diagnostics", 80),
        _table_html(platform.get("feature_catalog"), "Feature Block Coverage", 100),
        _table_html(platform.get("target_catalog"), "Return Target Catalog", 100),
        _table_html(platform.get("model_results"), "Model Registry Results", 100),
        _table_html(valuation_model_registry, "Expanded Valuation Model Registry", 100),
        _table_html(extended_valuation_results, "Expanded Valuation Results", 200),
        _table_html(valuation_gap_table, "Blended Extended Fair Value", 100),
        _table_html(screener_summary, "Screener Summary", 30),
        _table_html(screener_results, "Screener Results", 100),
        _table_html(screener_audit, "Screener Audit", 100),
        _table_html(finviz_groups, "Finviz-Style Group Summaries", 100),
        _table_html(finviz_maps, "Finviz-Style Map Payload", 100),
        _table_html(finviz_watchlists, "Watchlists and Candidate Portfolio", 100),
        _table_html(finviz_events, "Events / News / Insider / Calendar Payload", 100),
        _table_html(finviz_alerts, "Alert Log", 100),
        _table_html(finviz_manifest, "Finviz Layer Export Manifest", 100),
        _table_html(ml_leaderboard, "Advanced ML Leaderboard", 100),
        _table_html(ml_feature_importance, "ML Feature Importance", 100),
        _table_html(universe_master, "Universe Master Table", 200),
        _table_html(peeranalysis.get("selected_peers"), "Selected Comparable Companies", 50),
        _table_html(peeranalysis.get("target_vs_peers"), "Target vs Peer Multiples", 50),
        _table_html(platform.get("macro_risk"), "Macro-Risk Layer", 100),
        _table_html(platform.get("recommendations"), "Platform Recommendations", 50),
        _table_html(pd.DataFrame(PARAMETER_GUIDE), "Parameter Guide", 100),
        _table_html(pd.DataFrame(OPEN_SOURCE_SOURCES), "Open-source/API Source Inventory", 100),
        _table_html(platform.get("source_provenance"), "Runtime Source Provenance", 100),
        _table_html(namespace.get("provider_registry"), "Provider Registry and Fallback Order", 100),
        _table_html(namespace.get("api_key_status"), "API Key Status", 30),
        _table_html(namespace.get("refresh_provenance"), "Refresh Provenance", 100),
        _table_html(namespace.get("drive_sync_manifest"), "Google Drive Database Sync Manifest", 100),
        _table_html(refresh_plan, "Continuous Refresh Plan", 50),
    ]
    return f"""<!doctype html>
<html>
<head>
  <meta charset='utf-8'>
  <meta name='viewport' content='width=device-width, initial-scale=1'>
  <title>Company Valuation Report - {escape(ticker)}</title>
  <style>{dashboard_css()}</style>
</head>
<body>
  <div class='shell'>
    <header class='hero'>
      <h1>Company Valuation Research Report</h1>
      <p class='muted'>Ticker: {escape(ticker)} · Generated: {escape(datetime.now().isoformat(timespec="seconds"))}</p>
    </header>
    {''.join(report_sections)}
  </div>
</body>
</html>"""


def export_dashboard_and_report(namespace: Mapping[str, Any]) -> DashboardArtifacts:
    output_root = _output_root(namespace)
    dashboard_dir = output_root / "dashboard"
    reports_dir = output_root / "reports"
    dashboard_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    dashboard_html = build_dashboard_html(namespace)
    report_html = build_report_html(namespace)
    control_center_html = build_control_center_html(namespace)
    dashboard_path = dashboard_dir / "company_valuation_navigable_dashboard.html"
    report_path = reports_dir / "company_valuation_research_report.html"
    control_center_path = dashboard_dir / "company_valuation_control_center.html"
    dashboard_path.write_text(dashboard_html, encoding="utf-8")
    report_path.write_text(report_html, encoding="utf-8")
    control_center_path.write_text(control_center_html, encoding="utf-8")
    return DashboardArtifacts(dashboard_path, report_path, dashboard_html, report_html, control_center_path, control_center_html)
