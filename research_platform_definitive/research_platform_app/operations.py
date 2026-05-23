"""Operational adapters for regenerating lightweight research artifacts.

These functions intentionally run only modular, low-cost layers. They do not
execute notebooks or rebuild the full data pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from research_platform_app.support import PROJECT_ROOT, read_csv_any
except Exception:
    from support import PROJECT_ROOT, read_csv_any


if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _nonempty(df: pd.DataFrame) -> bool:
    return isinstance(df, pd.DataFrame) and not df.empty


def _company_namespace(company_root: Path) -> dict[str, Any]:
    latest = read_csv_any(company_root, [
        "tables/latestcrosssection.csv",
        "tables/latest_cross_section.csv",
        "tables/latest_cross_section_enriched.csv",
        "tables/screener_results.csv",
        "tables/ScreenerResults.csv",
        "tables/universe_master.csv",
    ])
    universe = read_csv_any(company_root, ["tables/universe_master.csv"])
    latest = _enrich_company_base(latest, universe)
    valuation = read_csv_any(company_root, [
        "tables/valuation_gap_table.csv",
        "tables/ValuationGapTable.csv",
        "tables/extended_valuation_results.csv",
        "tables/ExtendedValuationResults.csv",
    ])
    ranking = read_csv_any(company_root, [
        "tables/companyranking.csv",
        "tables/company_ranking.csv",
        "tables/screener_results.csv",
        "tables/ScreenerResults.csv",
    ])
    ml = read_csv_any(company_root, ["tables/ml_predictions.csv", "tables/model_predictions.csv"])
    return {
        "OUTPUTROOT": company_root,
        "latestcrosssection": latest,
        "companyranking": ranking,
        "valuationoutput": valuation,
        "ml_predictions": ml,
    }


def _country_from_ticker(ticker: str) -> str | float:
    ticker = str(ticker).upper()
    suffix_map = {
        ".MI": "Italy",
        ".PA": "France",
        ".DE": "Germany",
        ".AS": "Netherlands",
        ".BR": "Belgium",
        ".MC": "Spain",
        ".LS": "Portugal",
        ".SW": "Switzerland",
        ".L": "United Kingdom",
    }
    for suffix, country in suffix_map.items():
        if ticker.endswith(suffix):
            return country
    return float("nan")


def _enrich_company_base(base: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    if base.empty:
        return base
    out = base.copy()
    if "ticker" not in out.columns:
        return out
    out["ticker"] = out["ticker"].astype(str).str.upper().str.strip()
    if not universe.empty and "ticker" in universe.columns:
        uni = universe.copy()
        uni["ticker"] = uni["ticker"].astype(str).str.upper().str.strip()
        add_cols = [c for c in ["universe_source", "source_count", "country", "sector", "industry", "index_membership"] if c in uni.columns and c not in out.columns]
        if add_cols:
            out = out.merge(uni[["ticker", *add_cols]].drop_duplicates("ticker", keep="last"), on="ticker", how="left")
    if "index_membership" not in out.columns and "universe_source" in out.columns:
        out["index_membership"] = out["universe_source"]
    if "country" not in out.columns:
        out["country"] = out["ticker"].map(_country_from_ticker)
    elif out["country"].isna().all():
        out["country"] = out["ticker"].map(_country_from_ticker)
    return out


def generate_company_artifacts(company_root: Path, preset: str = "Top-ranked internal model ideas") -> dict[str, Any]:
    """Regenerate company screener and Finviz-style layer from available exports."""
    namespace = _company_namespace(company_root)
    if not _nonempty(namespace["latestcrosssection"]) and not _nonempty(namespace["companyranking"]):
        return {
            "ok": False,
            "message": "No company base table found. Run the valuation notebook export cells first.",
            "details": {},
        }
    try:
        from company_valuation.src.company_valuation_screener import run_integrated_screener
        from company_valuation.src.company_valuation_finviz_layer import run_finviz_platform_layer

        config = {"preset": preset, "filters": []}
        screener = run_integrated_screener(namespace, config)
        finviz = run_finviz_platform_layer(namespace, config)
        return {
            "ok": True,
            "message": "Company screener and platform artifacts regenerated.",
            "details": {
                "screener_rows": len(screener.filtered),
                "audit_rows": len(screener.audit),
                "groups_rows": len(finviz.group_summaries),
                "map_rows": len(finviz.map_payload),
            },
        }
    except Exception as exc:
        return {"ok": False, "message": f"Company generation failed: {exc}", "details": {}}


def _workspace_portfolio_rankings(workspace_root: Path) -> pd.DataFrame:
    raw = read_csv_any(workspace_root, ["tables/Table_14_portfolio_rankings.csv"])
    if raw.empty:
        return pd.DataFrame()
    out = raw.copy()
    if "date" in out.columns:
        dates = pd.to_datetime(out["date"], errors="coerce")
        if dates.notna().any():
            out = out.loc[dates.eq(dates.max())].copy()
    if "ticker" not in out.columns and "portfolio" in out.columns:
        out["ticker"] = out["portfolio"].astype(str)
    if "company_name" not in out.columns and "portfolio" in out.columns:
        out["company_name"] = out["portfolio"].astype(str)
    if "composite_score" not in out.columns and "prediction" in out.columns:
        out["composite_score"] = pd.to_numeric(out["prediction"], errors="coerce")
    if "annual_return" not in out.columns and "actual" in out.columns:
        out["annual_return"] = pd.to_numeric(out["actual"], errors="coerce")
    if "rank" not in out.columns and "prediction_rank" in out.columns:
        out["rank"] = pd.to_numeric(out["prediction_rank"], errors="coerce")
    if "quality_flag" not in out.columns:
        out["quality_flag"] = "BACKTEST_EXPORT"
    if "sector" not in out.columns:
        out["sector"] = "Portfolio Sleeve"
    if "country" not in out.columns:
        out["country"] = "n/a"
    return out


def _portfolio_allocation_from_rankings(rankings: pd.DataFrame) -> pd.DataFrame:
    if rankings.empty or "ticker" not in rankings.columns:
        return pd.DataFrame()
    latest = rankings.copy()
    if "date" in latest.columns:
        latest["_date"] = pd.to_datetime(latest["date"], errors="coerce")
        latest = latest[latest["_date"].eq(latest["_date"].max())].drop(columns=["_date"])
    if "selected_topk" in latest.columns:
        selected = latest[latest["selected_topk"].astype(str).str.lower().isin(["true", "1", "yes"])]
        if not selected.empty:
            latest = selected
    tickers = latest["ticker"].dropna().astype(str).drop_duplicates().tolist()
    if not tickers:
        return pd.DataFrame()
    weight = 1.0 / len(tickers)
    return pd.DataFrame({"ticker": tickers, "weight": weight, "source": "workspace_backtest_topk"})


def _portfolio_namespace(portfolio_root: Path, workspace_root: Path) -> dict[str, Any]:
    valuation_ranking = read_csv_any(portfolio_root, [
        "tables/valuation_ranking.csv",
        "tables/PortfolioSelectionResults.csv",
        "tables/portfolio_selection_results.csv",
    ])
    if valuation_ranking.empty:
        valuation_ranking = _workspace_portfolio_rankings(workspace_root)
    allocation = read_csv_any(portfolio_root, [
        "tables/portfolio_allocation.csv",
        "tables/PortfolioAllocation.csv",
    ])
    if allocation.empty:
        allocation = _portfolio_allocation_from_rankings(valuation_ranking)
    tables = {
        "valuation_ranking": valuation_ranking,
        "portfolio_allocation": allocation,
        "portfolio_engine_weights": read_csv_any(portfolio_root, ["tables/portfolio_engine_weights.csv"]),
        "deep_stock_signals": read_csv_any(portfolio_root, ["tables/deep_stock_signals.csv"]),
        "optimization_summary": read_csv_any(portfolio_root, ["tables/optimization_summary.csv"]),
    }
    return {"OUTPUT_ROOT": portfolio_root, "RESEARCH_RESULT": {"tables": tables}, "tables": tables}


def generate_portfolio_artifacts(
    portfolio_root: Path,
    workspace_root: Path,
    preset: str = "Top ranked allocation",
) -> dict[str, Any]:
    """Regenerate portfolio selection artifacts from portfolio or workspace exports."""
    namespace = _portfolio_namespace(portfolio_root, workspace_root)
    base = namespace["RESEARCH_RESULT"]["tables"].get("valuation_ranking", pd.DataFrame())
    if not _nonempty(base):
        return {
            "ok": False,
            "message": "No portfolio ranking/allocation table found. Run the portfolio notebook or workspace export first.",
            "details": {},
        }
    try:
        from portfolio_analysis.src.portfolio_research_screener import run_portfolio_selection_layer

        result = run_portfolio_selection_layer(namespace, {"preset": preset, "filters": []})
        return {
            "ok": True,
            "message": "Portfolio selection artifacts regenerated.",
            "details": {
                "selection_rows": len(result.results),
                "audit_rows": len(result.audit),
                "base_rows": len(base),
            },
        }
    except Exception as exc:
        return {"ok": False, "message": f"Portfolio generation failed: {exc}", "details": {}}


def generate_all_artifacts(roots: dict[str, Path]) -> dict[str, dict[str, Any]]:
    return {
        "company": generate_company_artifacts(roots["company"]),
        "portfolio": generate_portfolio_artifacts(roots["portfolio"], roots["workspace"]),
    }
