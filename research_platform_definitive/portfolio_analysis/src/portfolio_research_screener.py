"""Portfolio-native research selection layer.

This is a Finviz-inspired layer adapted to portfolio construction: it filters,
ranks and audits allocation candidates using the notebook's own ranking,
allocation, optimization, risk, scenario and ML tables.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

try:
    from research_platform_core import (
        as_df as _core_as_df,
        first_available as _core_first,
        normalize_name,
        normalize_ticker,
        resolve_alias,
        safe_write_csv,
        safe_write_json,
    )
except ModuleNotFoundError:
    from src.research_platform_core import (
        as_df as _core_as_df,
        first_available as _core_first,
        normalize_name,
        normalize_ticker,
        resolve_alias,
        safe_write_csv,
        safe_write_json,
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def as_df(value: Any) -> pd.DataFrame:
    return _core_as_df(value)


def first(namespace: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    return _core_first(namespace, *names, default=default)


PORTFOLIO_FIELD_ALIASES = {
    "ticker": ["symbol", "asset"],
    "company_name": ["name", "company", "short_name", "long_name"],
    "sector": ["sector", "gics_sector"],
    "industry": ["industry", "gics_industry"],
    "country": ["country", "region"],
    "weight": ["weight", "target_weight", "allocation_weight", "portfolio_weight"],
    "engine_weight": ["engine_weight", "optimized_weight", "weight_engine"],
    "rank": ["rank", "ranking"],
    "composite_score": ["composite_score", "compositescore", "score", "selection_score"],
    "value_score": ["value_score", "valuation_score", "valuationscore"],
    "quality_score": ["quality_score", "qualityscore"],
    "momentum_score": ["momentum_score", "momentumscore"],
    "risk_score": ["risk_score", "riskscore"],
    "growth_score": ["growth_score", "growthscore"],
    "ml_score": ["ml_score", "model_score", "prediction_score", "deep_stock_score"],
    "annual_return": ["annual_return", "return_ann", "expected_return"],
    "volatility": ["volatility", "annual_volatility", "vol"],
    "max_drawdown": ["max_drawdown", "drawdown"],
    "sharpe": ["sharpe", "sharpe_ratio"],
    "upside_base": ["upside_base", "upside", "base_upside"],
    "downside_bear": ["downside_bear", "downside", "bear_downside"],
    "market_cap": ["market_cap", "marketcap", "marketcapest"],
    "trailing_pe": ["trailing_pe", "pe_ratio", "peratio"],
    "price_to_book": ["price_to_book", "pb_ratio", "pbratio"],
    "ev_to_ebitda_proxy": ["ev_to_ebitda_proxy", "evebitda", "ev_ebitda"],
    "fcf_yield": ["fcf_yield", "free_cash_flow_yield"],
    "source": ["source", "data_source"],
    "quality_flag": ["quality_flag", "status", "diagnostic_status"],
}


PORTFOLIO_FILTER_SCHEMA = {
    "identity": {
        "ticker": {"type": "text", "operators": ["equals", "in", "contains"]},
        "sector": {"type": "category", "operators": ["equals", "in", "contains"]},
        "industry": {"type": "category", "operators": ["equals", "in", "contains"]},
        "country": {"type": "category", "operators": ["equals", "in"]},
    },
    "allocation": {
        "weight": {"type": "numeric", "operators": ["gt", "lt", "between", "top_pct"]},
        "engine_weight": {"type": "numeric", "operators": ["gt", "lt", "between", "top_pct"]},
        "rank": {"type": "numeric", "operators": ["lt", "between"]},
    },
    "score": {
        "composite_score": {"type": "numeric", "operators": ["gt", "lt", "between", "top_pct"]},
        "value_score": {"type": "numeric", "operators": ["gt", "lt", "between", "top_pct"]},
        "quality_score": {"type": "numeric", "operators": ["gt", "lt", "between", "top_pct"]},
        "momentum_score": {"type": "numeric", "operators": ["gt", "lt", "between", "top_pct"]},
        "risk_score": {"type": "numeric", "operators": ["gt", "lt", "between", "top_pct"]},
        "growth_score": {"type": "numeric", "operators": ["gt", "lt", "between", "top_pct"]},
        "ml_score": {"type": "numeric", "operators": ["gt", "lt", "between", "top_pct"]},
    },
    "risk_return": {
        "annual_return": {"type": "numeric", "operators": ["gt", "lt", "between", "top_pct"]},
        "volatility": {"type": "numeric", "operators": ["gt", "lt", "between", "bottom_pct"]},
        "max_drawdown": {"type": "numeric", "operators": ["gt", "lt", "between"]},
        "sharpe": {"type": "numeric", "operators": ["gt", "lt", "between", "top_pct"]},
        "upside_base": {"type": "numeric", "operators": ["gt", "lt", "between", "top_pct"]},
        "downside_bear": {"type": "numeric", "operators": ["gt", "lt", "between"]},
    },
    "valuation": {
        "trailing_pe": {"type": "numeric", "operators": ["gt", "lt", "between", "bottom_pct"]},
        "price_to_book": {"type": "numeric", "operators": ["gt", "lt", "between", "bottom_pct"]},
        "ev_to_ebitda_proxy": {"type": "numeric", "operators": ["gt", "lt", "between", "bottom_pct"]},
        "fcf_yield": {"type": "numeric", "operators": ["gt", "lt", "between", "top_pct"]},
    },
}


PORTFOLIO_PRESETS = {
    "Core quality allocation": {
        "description": "High quality, diversified candidates with positive composite score.",
        "filters": [{"field": "quality_score", "op": "gt", "value": 0.55}, {"field": "composite_score", "op": "gt", "value": 0.55}],
        "ranking": {"mode": "weighted_score", "weights": {"quality_score": 0.35, "composite_score": 0.30, "risk_score": 0.20, "momentum_score": 0.15}},
    },
    "Risk controlled allocation": {
        "description": "Low volatility / drawdown candidates for defensive construction.",
        "filters": [{"field": "risk_score", "op": "gt", "value": 0.55}, {"field": "volatility", "op": "bottom_pct", "value": 0.50}],
        "ranking": {"mode": "weighted_score", "weights": {"risk_score": 0.45, "quality_score": 0.25, "composite_score": 0.20, "upside_base": 0.10}},
    },
    "Value with upside": {
        "description": "Value score and upside support with quality floor.",
        "filters": [{"field": "value_score", "op": "gt", "value": 0.55}, {"field": "upside_base", "op": "gt", "value": 0.02}, {"field": "quality_score", "op": "gt", "value": 0.35}],
        "ranking": {"mode": "weighted_score", "weights": {"value_score": 0.35, "upside_base": 0.30, "quality_score": 0.20, "risk_score": 0.15}},
    },
    "Momentum sleeve": {
        "description": "Momentum-driven candidates with portfolio risk guardrails.",
        "filters": [{"field": "momentum_score", "op": "gt", "value": 0.60}, {"field": "risk_score", "op": "gt", "value": 0.35}],
        "ranking": {"mode": "weighted_score", "weights": {"momentum_score": 0.45, "composite_score": 0.25, "annual_return": 0.20, "risk_score": 0.10}},
    },
    "Optimizer candidates": {
        "description": "Names preferred by the optimizer / engine weights.",
        "filters": [{"field": "engine_weight", "op": "gt", "value": 0.0}],
        "ranking": {"mode": "raw_sort", "sort_by": "engine_weight", "ascending": False},
    },
    "Top ranked allocation": {
        "description": "Use the notebook-native composite/rank view.",
        "filters": [{"field": "composite_score", "op": "top_pct", "value": 0.35}],
        "ranking": {"mode": "raw_sort", "sort_by": "composite_score", "ascending": False},
    },
}


DEFAULT_COLUMNS = [
    "selection_rank", "ticker", "company_name", "sector", "industry", "country",
    "weight", "engine_weight", "rank", "composite_score", "value_score", "quality_score",
    "momentum_score", "risk_score", "growth_score", "ml_score", "annual_return",
    "volatility", "max_drawdown", "sharpe", "upside_base", "downside_bear",
    "selection_score", "quality_flag",
]


@dataclass
class PortfolioSelectionResult:
    results: pd.DataFrame
    audit: pd.DataFrame
    progression: pd.DataFrame
    summary: pd.DataFrame
    config: dict[str, Any]
    schema: pd.DataFrame
    presets: pd.DataFrame


def resolve_column(df: pd.DataFrame, field: str) -> str | None:
    return resolve_alias(df, field, PORTFOLIO_FIELD_ALIASES)


def ensure_canonical_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    fields = {field for group in PORTFOLIO_FILTER_SCHEMA.values() for field in group}
    fields.update(PORTFOLIO_FIELD_ALIASES)
    for field in sorted(fields):
        if field not in out.columns:
            col = resolve_column(out, field)
            if col is not None:
                out[field] = out[col]
    if "ticker" in out.columns:
        out["ticker"] = out["ticker"].map(normalize_ticker)
    return out


def safe_join(base: pd.DataFrame, aux: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    if base.empty or aux.empty:
        return base
    base = ensure_canonical_columns(base)
    aux = ensure_canonical_columns(aux)
    if "ticker" not in base.columns or "ticker" not in aux.columns:
        return base
    aux = aux.loc[:, ~aux.columns.duplicated()].dropna(subset=["ticker"]).drop_duplicates("ticker", keep="last")
    existing = {normalize_name(c) for c in base.columns}
    add_cols = [c for c in aux.columns if c != "ticker" and normalize_name(c) not in existing]
    if not add_cols:
        return base
    rename = {c: f"{prefix}{c}" for c in add_cols if prefix}
    return base.merge(aux[["ticker", *add_cols]].rename(columns=rename), on="ticker", how="left")


def build_base_frame(tables: Mapping[str, Any]) -> pd.DataFrame:
    """Build the allocation-candidate frame from notebook-native portfolio tables."""
    base = as_df(tables.get("valuation_ranking"))
    if base.empty:
        base = as_df(tables.get("portfolio_allocation"))
    if base.empty:
        return pd.DataFrame()
    base = ensure_canonical_columns(base)
    allocation = as_df(tables.get("portfolio_allocation"))
    engine_weights = as_df(tables.get("portfolio_engine_weights"))
    if not engine_weights.empty:
        engine_weights = ensure_canonical_columns(engine_weights)
        ticker_col = resolve_column(engine_weights, "ticker")
        weight_col = resolve_column(engine_weights, "weight")
        if ticker_col and weight_col:
            engine_weights = engine_weights[[ticker_col, weight_col]].rename(columns={ticker_col: "ticker", weight_col: "engine_weight"})
        else:
            engine_weights = pd.DataFrame()
    deep_signals = as_df(tables.get("deep_stock_signals"))
    if not deep_signals.empty:
        score_col = resolve_column(deep_signals, "ml_score")
        if score_col and score_col != "ml_score":
            deep_signals = deep_signals.rename(columns={score_col: "ml_score"})
    for aux in [allocation, engine_weights, deep_signals, as_df(tables.get("optimization_summary"))]:
        base = safe_join(base, aux)
    base = ensure_canonical_columns(base)
    base = base.loc[:, ~base.columns.duplicated()]
    if "weight" not in base.columns:
        score = pd.to_numeric(base.get("composite_score", pd.Series(1.0, index=base.index)), errors="coerce").clip(lower=0).fillna(0)
        base["weight"] = score / score.sum() if score.sum() else 1 / len(base)
    if "engine_weight" not in base.columns:
        base["engine_weight"] = np.nan
    if "sharpe" not in base.columns and {"annual_return", "volatility"}.issubset(base.columns):
        base["sharpe"] = pd.to_numeric(base["annual_return"], errors="coerce") / pd.to_numeric(base["volatility"], errors="coerce").replace(0, np.nan)
    base["quality_flag"] = np.where(base["ticker"].astype(str).str.len() > 0, "PASS", "WARN") if "ticker" in base.columns else "WARN"
    base["updated_at"] = utc_now()
    return base


def apply_filter(df: pd.DataFrame, spec: Mapping[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    before = len(df)
    field = str(spec.get("field", ""))
    op = str(spec.get("op", spec.get("operator", "equals"))).lower()
    value = spec.get("value")
    col = resolve_column(df, field)
    audit = {"field": field, "operator": op, "value": value, "resolved_column": col, "before": before, "after": before, "dropped": 0, "status": "SKIP", "detail": "", "updated_at": utc_now()}
    if before == 0:
        audit["detail"] = "No rows available"
        return df, audit
    if col is None:
        audit["detail"] = "Field unavailable"
        return df, audit
    s = df[col]
    try:
        if op in {"gt", ">"}:
            mask = pd.to_numeric(s, errors="coerce") > float(value)
        elif op in {"lt", "<"}:
            mask = pd.to_numeric(s, errors="coerce") < float(value)
        elif op == "between":
            lo, hi = value
            mask = pd.to_numeric(s, errors="coerce").between(float(lo), float(hi), inclusive="both")
        elif op in {"equals", "eq"}:
            mask = s.astype(str).str.upper() == str(value).upper()
        elif op in {"in", "in_list"}:
            values = [str(v).upper() for v in (value if isinstance(value, list) else [value])]
            mask = s.astype(str).str.upper().isin(values)
        elif op == "contains":
            mask = s.astype(str).str.contains(str(value), case=False, regex=True, na=False)
        elif op in {"top_pct", "bottom_pct"}:
            nums = pd.to_numeric(s, errors="coerce")
            valid = nums.dropna()
            if valid.empty:
                audit["detail"] = "No numeric values"
                return df, audit
            pct = max(0.0, min(1.0, float(value)))
            n_keep = max(1, int(np.ceil(len(valid) * pct)))
            threshold = valid.nlargest(n_keep).min() if op == "top_pct" else valid.nsmallest(n_keep).max()
            mask = nums >= threshold if op == "top_pct" else nums <= threshold
        else:
            audit["detail"] = f"Unsupported operator: {op}"
            return df, audit
    except Exception as exc:
        audit["detail"] = f"Filter failed: {exc}"
        return df, audit
    out = df.loc[mask.fillna(False)].copy()
    audit.update({"after": len(out), "dropped": before - len(out), "status": "PASS", "detail": "Applied"})
    return out, audit


def percentile_score(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    nums = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if nums.notna().sum() == 0:
        return pd.Series(0.5, index=series.index)
    ranks = nums.rank(pct=True)
    return ranks if higher_is_better else 1 - ranks


def apply_ranking(df: pd.DataFrame, ranking: Mapping[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    mode = str(ranking.get("mode", "raw_sort")).lower()
    skipped: list[str] = []
    if out.empty:
        out["selection_score"] = []
        out["selection_rank"] = []
        return out, skipped
    if mode == "raw_sort":
        field = str(ranking.get("sort_by", "composite_score"))
        col = resolve_column(out, field)
        if col:
            out["selection_score"] = pd.to_numeric(out[col], errors="coerce")
            out = out.sort_values(col, ascending=bool(ranking.get("ascending", False)), na_position="last")
        else:
            skipped.append(field)
            out["selection_score"] = 0.0
    elif mode in {"weighted_score", "composite"}:
        score = pd.Series(0.0, index=out.index)
        total = 0.0
        for field, weight in (ranking.get("weights") or {}).items():
            col = resolve_column(out, field)
            if not col:
                skipped.append(field)
                continue
            lower = bool((ranking.get("lower_is_better") or {}).get(field, False))
            score += abs(float(weight)) * percentile_score(out[col], higher_is_better=not lower)
            total += abs(float(weight))
        out["selection_score"] = score / total if total else 0.0
        out = out.sort_values("selection_score", ascending=False, na_position="last")
    elif mode == "risk_budget":
        risk = resolve_column(out, "risk_score")
        comp = resolve_column(out, "composite_score")
        if risk and comp:
            out["selection_score"] = 0.55 * percentile_score(out[comp], True) + 0.45 * percentile_score(out[risk], True)
            out = out.sort_values("selection_score", ascending=False)
        else:
            skipped.extend([x for x, ok in [("risk_score", risk), ("composite_score", comp)] if not ok])
            out["selection_score"] = 0.0
    else:
        skipped.append(f"unsupported_ranking_mode:{mode}")
        out["selection_score"] = 0.0
    out["selection_rank"] = np.arange(1, len(out) + 1)
    return out, skipped


def schema_frame() -> pd.DataFrame:
    return pd.DataFrame([
        {"group": group, "field": field, "type": spec["type"], "operators": ", ".join(spec["operators"]), "aliases": ", ".join(PORTFOLIO_FIELD_ALIASES.get(field, []))}
        for group, fields in PORTFOLIO_FILTER_SCHEMA.items()
        for field, spec in fields.items()
    ])


def presets_frame() -> pd.DataFrame:
    return pd.DataFrame([
        {"preset": name, "description": cfg.get("description", ""), "filters": len(cfg.get("filters", [])), "ranking": cfg.get("ranking", {}).get("mode", "")}
        for name, cfg in PORTFOLIO_PRESETS.items()
    ])


def run_selection(base: pd.DataFrame, config: Mapping[str, Any]) -> PortfolioSelectionResult:
    work = ensure_canonical_columns(base)
    preset_name = config.get("preset")
    preset = PORTFOLIO_PRESETS.get(preset_name, {}) if preset_name else {}
    filters = list(preset.get("filters", [])) + list(config.get("filters", []))
    ranking = config.get("ranking") or preset.get("ranking") or {"mode": "raw_sort", "sort_by": "composite_score", "ascending": False}
    audit_rows = []
    progression = [{"step": "start", "field": "", "operator": "", "rows": len(work), "dropped": 0, "status": "START", "detail": ""}]
    for idx, spec in enumerate(filters, start=1):
        work, audit = apply_filter(work, spec)
        audit["step"] = idx
        audit_rows.append(audit)
        progression.append({"step": idx, "field": audit["field"], "operator": audit["operator"], "rows": audit["after"], "dropped": audit["dropped"], "status": audit["status"], "detail": audit["detail"]})
    ranked, ranking_skips = apply_ranking(work, ranking)
    for field in ranking_skips:
        audit_rows.append({"field": field, "operator": "ranking", "value": ranking.get("mode"), "resolved_column": None, "before": len(work), "after": len(ranked), "dropped": 0, "status": "SKIP", "detail": "Ranking field unavailable", "updated_at": utc_now(), "step": "ranking"})
    cols = [c for c in DEFAULT_COLUMNS if c in ranked.columns]
    results = ranked[cols + [c for c in ranked.columns if c not in cols]].copy()
    audit = pd.DataFrame(audit_rows)
    summary = pd.DataFrame([{
        "preset": preset_name or "custom",
        "input_rows": len(base),
        "selected_rows": len(results),
        "filters_requested": len(filters),
        "filters_applied": int((audit["status"] == "PASS").sum()) if not audit.empty else 0,
        "filters_skipped": int((audit["status"] == "SKIP").sum()) if not audit.empty else 0,
        "ranking_mode": ranking.get("mode"),
        "ranking_skipped_fields": ", ".join(ranking_skips),
        "updated_at": utc_now(),
    }])
    full_config = {"preset": preset_name, "filters": filters, "ranking": ranking, "description": preset.get("description", config.get("description", ""))}
    return PortfolioSelectionResult(results, audit, pd.DataFrame(progression), summary, full_config, schema_frame(), presets_frame())


def html_table(df: pd.DataFrame, title: str, max_rows: int = 50) -> str:
    if df is None or df.empty:
        return f"<section class='card'><h2>{escape(title)}</h2><p class='muted'>No data available.</p></section>"
    return f"<section class='card'><h2>{escape(title)}</h2>{df.head(max_rows).to_html(index=False, classes='data-table', border=0, escape=True)}</section>"


def build_html_hooks(namespace: dict[str, Any]) -> dict[str, str]:
    results = as_df(namespace.get("portfolio_selection_results"))
    audit = as_df(namespace.get("portfolio_selection_audit"))
    summary = as_df(namespace.get("portfolio_selection_summary"))
    kpi = summary.iloc[0].to_dict() if not summary.empty else {}
    cards = "".join(
        f"<div class='card kpi'><div class='label'>{escape(label)}</div><div class='value'>{escape(str(value))}</div></div>"
        for label, value in [
            ("Preset", kpi.get("preset", "n/a")),
            ("Selected", kpi.get("selected_rows", 0)),
            ("Applied", kpi.get("filters_applied", 0)),
            ("Skipped", kpi.get("filters_skipped", 0)),
            ("Ranking", kpi.get("ranking_mode", "n/a")),
        ]
    )
    return {
        "portfolio_selection_kpi_html": f"<section class='grid kpi-grid'>{cards}</section>",
        "portfolio_selection_results_html": html_table(results, "Portfolio Selection Results", 100),
        "portfolio_selection_audit_html": html_table(audit, "Portfolio Selection Audit", 100),
        "portfolio_selection_summary_html": html_table(summary, "Portfolio Selection Summary", 20),
    }


def export_outputs(result: PortfolioSelectionResult, output_root: str | Path) -> dict[str, Path]:
    root = Path(output_root)
    if root.name.lower() in {"tables", "figures", "logs", "config"}:
        root = root.parent
    table_dir = root / "tables"
    config_dir = root / "config"
    paths = {
        "portfolio_selection_results": table_dir / "PortfolioSelectionResults.csv",
        "portfolio_selection_audit": table_dir / "PortfolioSelectionAudit.csv",
        "portfolio_selection_progression": table_dir / "PortfolioSelectionProgression.csv",
        "portfolio_selection_summary": table_dir / "PortfolioSelectionSummary.csv",
        "portfolio_selection_schema": table_dir / "PortfolioSelectionSchema.csv",
        "portfolio_selection_presets": table_dir / "PortfolioSelectionPresets.csv",
        "portfolio_selection_config": config_dir / "PortfolioSelectionConfig.json",
    }
    for key, df in [
        ("portfolio_selection_results", result.results),
        ("portfolio_selection_audit", result.audit),
        ("portfolio_selection_progression", result.progression),
        ("portfolio_selection_summary", result.summary),
        ("portfolio_selection_schema", result.schema),
        ("portfolio_selection_presets", result.presets),
    ]:
        safe_write_csv(df, paths[key])
    safe_write_json(result.config, paths["portfolio_selection_config"])
    return paths


def run_portfolio_selection_layer(namespace: dict[str, Any], config: Mapping[str, Any] | None = None) -> PortfolioSelectionResult:
    result_obj = namespace.get("RESEARCH_RESULT") or namespace.get("result") or {}
    tables = result_obj.get("tables", {}) if isinstance(result_obj, Mapping) else namespace.get("tables", {})
    tables = tables if isinstance(tables, Mapping) else {}
    base = build_base_frame(tables)
    cfg = dict(config or namespace.get("PORTFOLIOSELECTIONCONFIG", {}) or {})
    if not cfg:
        cfg = {"preset": "Top ranked allocation", "filters": [], "ranking": {"mode": "raw_sort", "sort_by": "composite_score", "ascending": False}}
    result = run_selection(base, cfg)
    output_root = first(namespace, "OUTPUT_ROOT", "OUTPUTROOT", "OUTPUT_DIR", "TABLES_DIR", default=Path.cwd() / "output")
    paths = export_outputs(result, output_root)
    namespace.update({
        "portfolio_selection_base": base,
        "portfolio_selection_results": result.results,
        "portfolio_selection_audit": result.audit,
        "portfolio_selection_progression": result.progression,
        "portfolio_selection_summary": result.summary,
        "portfolio_selection_schema": result.schema,
        "portfolio_selection_presets": result.presets,
        "portfolio_selection_config": result.config,
        "portfolio_selection_export_paths": paths,
    })
    namespace.update(build_html_hooks(namespace))
    if isinstance(tables, dict):
        tables.update({
            "portfolio_selection_results": result.results,
            "portfolio_selection_audit": result.audit,
            "portfolio_selection_progression": result.progression,
            "portfolio_selection_summary": result.summary,
            "portfolio_selection_schema": result.schema,
            "portfolio_selection_presets": result.presets,
        })
    return result
