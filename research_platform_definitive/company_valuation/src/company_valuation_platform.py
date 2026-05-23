"""Research-platform layer for the company valuation notebook.

This module upgrades the notebook from a linear valuation script into a
diagnostics-first research platform while preserving the notebook's existing
economic methodology and variable names. Every function is defensive so it can
run in Colab with partial data, local database files, or API fallbacks.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


CRITICAL_FUNDAMENTAL_FIELDS = [
    "ticker",
    "effective_date",
    "revenue",
    "free_cash_flow",
    "shares_outstanding",
    "eps",
    "book_value_per_share",
    "roe",
    "total_debt",
    "cash",
]


FEATURE_BLOCK_LIBRARY = {
    "market": ["adj_close", "close", "volume", "ret_1d", "ret_21d", "ret_63d"],
    "valuation": ["pe_ratio", "pb_ratio", "ev_ebitda", "fcf_yield", "fair_value_estimate", "upside_to_fair_value"],
    "quality": ["roe", "roa", "gross_margin", "operating_margin", "free_cash_flow_margin"],
    "growth": ["revenue_growth", "eps_growth", "net_income_growth", "free_cash_flow_growth"],
    "leverage": ["debt_equity", "net_debt_ebitda", "interest_coverage", "total_debt"],
    "dividend": ["dividend_yield", "payout_ratio", "dividend_growth"],
    "momentum": ["mom_3m", "mom_6m", "mom_12m"],
    "risk": ["vol_21d", "vol_63d", "beta", "drawdown_252d"],
    "macro": ["rate_10y", "rate_2y", "yield_curve_slope", "inflation_proxy", "credit_spread"],
    "fx": ["eurusd", "gbpusd", "usdjpy", "fx_return_63d"],
    "commodities": ["brent", "wti", "gold", "copper", "commodity_return_63d"],
    "peers": ["peer_discount", "peer_rank", "peer_similarity_score"],
}


RETURN_TARGETS = {
    "raw_forward_return_1m": {"horizon": 21, "benchmark": None},
    "raw_forward_return_3m": {"horizon": 63, "benchmark": None},
    "raw_forward_return_12m": {"horizon": 252, "benchmark": None},
    "excess_vs_market_12m": {"horizon": 252, "benchmark": "market"},
    "excess_vs_sector_12m": {"horizon": 252, "benchmark": "sector"},
    "risk_adjusted_return_12m": {"horizon": 252, "vol_adjust": True},
    "valuation_gap_closure": {"target": "upside_to_fair_value"},
}


MODEL_FAMILIES = {
    "linear": "Linear regression baseline",
    "ridge": "Regularized linear model",
    "elastic_net": "Sparse regularized linear model",
    "random_forest": "Nonlinear tree ensemble",
    "gradient_boosting": "Boosting-style nonlinear model",
    "bayesian_ridge": "Bayesian linear model with predictive uncertainty proxy",
}


OPEN_SOURCE_SERIES = {
    "rates": {
        "yfinance": {"us_2y": "^IRX", "us_10y": "^TNX"},
        "stooq": {"us_10y": "10usy.b", "us_2y": "02usy.b"},
    },
    "fx": {
        "yfinance": {"eurusd": "EURUSD=X", "gbpusd": "GBPUSD=X", "usdjpy": "JPY=X"},
        "stooq": {"eurusd": "eurusd", "gbpusd": "gbpusd", "usdjpy": "usdjpy"},
    },
    "commodities": {
        "yfinance": {"wti": "CL=F", "brent": "BZ=F", "gold": "GC=F", "copper": "HG=F"},
        "stooq": {"gold": "xauusd", "copper": "hg.f", "wti": "cl.f"},
    },
}


@dataclass
class ResearchPlatformOutputs:
    data_inventory: pd.DataFrame
    data_quality: pd.DataFrame
    feature_catalog: pd.DataFrame
    target_catalog: pd.DataFrame
    model_results: pd.DataFrame
    peeranalysis: dict[str, pd.DataFrame]
    macro_risk: pd.DataFrame
    source_provenance: pd.DataFrame
    recommendations: pd.DataFrame


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


def _first(namespace: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        value = namespace.get(name)
        if value is not None:
            return value
    return default


def get_db_base(namespace: Mapping[str, Any] | None = None) -> Path:
    namespace = namespace or {}
    for key in ["ML_TRADING_DB_BASE", "GENESIS_DB_BASE", "DATA_PATH", "DB_BASE"]:
        value = os.environ.get(key)
        if value:
            return Path(value).expanduser().resolve()
    for name in ["DB_BASE", "DATA_PATH", "db_base"]:
        value = namespace.get(name)
        if value:
            return Path(value).expanduser().resolve()
    drive = Path("/content/drive/MyDrive/Database Finanziario")
    if drive.exists():
        return drive
    return Path.cwd().resolve()


def normalize_ticker(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().upper().replace("/", "-")


def safe_numeric(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = frame.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def discover_local_database(db_base: Path) -> pd.DataFrame:
    rows = []
    if not db_base.exists():
        return pd.DataFrame([{"path": str(db_base), "kind": "missing_root", "size_mb": 0.0, "matched_role": "none"}])
    patterns = ["*.parquet", "*.csv", "*.json", "*.xlsx"]
    keywords = {
        "market": ["price", "market", "ohlcv", "equity"],
        "fundamentals": ["fundamental", "financial", "income", "balance", "cashflow", "ratio"],
        "risk_factors": ["factor", "fama", "risk", "beta"],
        "rates": ["rate", "yield", "curve", "treasury", "ecb"],
        "fx": ["fx", "forex", "currency", "eurusd", "gbpusd"],
        "commodities": ["commodity", "oil", "brent", "wti", "gold", "copper"],
        "macro": ["macro", "fred", "inflation", "gdp", "cpi"],
    }
    files = []
    for pattern in patterns:
        files.extend(db_base.rglob(pattern))
    for path in files[:2000]:
        lower = path.name.lower()
        roles = [role for role, words in keywords.items() if any(word in lower for word in words)]
        rows.append(
            {
                "path": str(path),
                "file": path.name,
                "suffix": path.suffix.lower(),
                "size_mb": round(path.stat().st_size / 1_000_000, 3) if path.exists() else np.nan,
                "matched_role": ",".join(roles) if roles else "unclassified",
            }
        )
    return pd.DataFrame(rows).sort_values(["matched_role", "file"]).reset_index(drop=True) if rows else pd.DataFrame()


def read_tabular_file(path: Path) -> pd.DataFrame:
    try:
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        if path.suffix.lower() == ".json":
            return pd.read_json(path)
        if path.suffix.lower() in [".xlsx", ".xls"]:
            return pd.read_excel(path)
    except Exception:
        return pd.DataFrame()
    return pd.DataFrame()


def load_local_risk_factor_candidates(inventory: pd.DataFrame, max_files: int = 8) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    if inventory.empty or "matched_role" not in inventory.columns:
        return out
    roles = ["risk_factors", "rates", "fx", "commodities", "macro"]
    candidates = inventory[inventory["matched_role"].astype(str).apply(lambda x: any(role in x for role in roles))]
    for _, row in candidates.head(max_files).iterrows():
        path = Path(str(row["path"]))
        data = read_tabular_file(path)
        if not data.empty:
            out[path.stem] = data
    return out


def fetch_yfinance_risk_series(start_date: str | None = None, end_date: str | None = None) -> dict[str, pd.DataFrame]:
    try:
        import yfinance as yf
    except Exception:
        return {}
    outputs = {}
    for group, providers in OPEN_SOURCE_SERIES.items():
        symbols = providers.get("yfinance", {})
        for name, symbol in symbols.items():
            try:
                df = yf.download(symbol, start=start_date, end=end_date, progress=False, auto_adjust=False)
                if df is not None and not df.empty:
                    df = df.reset_index()
                    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
                    df["factor_name"] = name
                    df["factor_group"] = group
                    outputs[f"{group}_{name}"] = df
            except Exception:
                continue
    return outputs


def build_source_provenance(local_inventory: pd.DataFrame, api_series: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    if not local_inventory.empty:
        for role, count in local_inventory["matched_role"].value_counts().items():
            rows.append({"source": "local_database", "role": role, "objects": int(count), "status": "available"})
    for name, df in api_series.items():
        rows.append({"source": "open_source_api", "role": name, "objects": len(df), "status": "loaded" if not df.empty else "empty"})
    if not rows:
        rows.append({"source": "none", "role": "none", "objects": 0, "status": "no sources discovered"})
    return pd.DataFrame(rows)


def run_data_quality(namespace: Mapping[str, Any]) -> pd.DataFrame:
    rows = []
    frames = {
        "dfmarket": _as_df(_first(namespace, "dfmarket", "df_market", "dfpanel", "df_panel")),
        "dffund": _as_df(_first(namespace, "dffund", "df_fund", "fundamentals")),
        "dfmerged": _as_df(_first(namespace, "dfmerged", "df_merged")),
        "dffeat": _as_df(_first(namespace, "dffeat", "df_feat")),
        "dfmodel": _as_df(_first(namespace, "dfmodel", "df_model")),
    }
    for name, frame in frames.items():
        if frame.empty:
            rows.append({"check": f"{name}_exists", "status": "WARN", "metric": "rows", "value": 0, "detail": f"{name} is missing or empty"})
            continue
        rows.append({"check": f"{name}_exists", "status": "PASS", "metric": "rows", "value": len(frame), "detail": f"{name} available"})
        if "ticker" in frame.columns:
            rows.append({"check": f"{name}_ticker_coverage", "status": "PASS", "metric": "unique_tickers", "value": frame["ticker"].nunique(), "detail": "unique ticker count"})
        if "date" in frame.columns:
            dates = pd.to_datetime(frame["date"], errors="coerce")
            rows.append({"check": f"{name}_date_coverage", "status": "PASS" if dates.notna().any() else "FAIL", "metric": "date_range", "value": f"{dates.min()} -> {dates.max()}", "detail": "date coverage"})
        missing_ratio = frame.isna().mean().mean()
        rows.append({"check": f"{name}_missingness", "status": "PASS" if missing_ratio < 0.25 else "WARN", "metric": "missing_ratio", "value": float(missing_ratio), "detail": "average missingness across columns"})

    dffund = frames["dffund"]
    if not dffund.empty:
        for field in CRITICAL_FUNDAMENTAL_FIELDS:
            if field not in dffund.columns:
                rows.append({"check": "fundamental_field", "status": "FAIL", "metric": field, "value": "missing", "detail": "critical valuation field is absent"})
            else:
                miss = float(dffund[field].isna().mean())
                rows.append({"check": "fundamental_field", "status": "PASS" if miss < 0.40 else "WARN", "metric": field, "value": miss, "detail": "field missingness"})

    dfmerged = frames["dfmerged"]
    if not dfmerged.empty and {"date", "effective_date"}.issubset(dfmerged.columns):
        dates = pd.to_datetime(dfmerged["date"], errors="coerce")
        effective = pd.to_datetime(dfmerged["effective_date"], errors="coerce")
        violations = int((effective > dates).fillna(False).sum())
        rows.append({"check": "no_lookahead_effective_date", "status": "PASS" if violations == 0 else "FAIL", "metric": "violations", "value": violations, "detail": "effective_date must be <= market date"})
    return pd.DataFrame(rows)


def build_feature_catalog(namespace: Mapping[str, Any]) -> pd.DataFrame:
    frame = _as_df(_first(namespace, "dffeat", "dfmerged", "dfmodel", "df_feat", "df_merged", "df_model"))
    rows = []
    for block, columns in FEATURE_BLOCK_LIBRARY.items():
        available = [col for col in columns if col in frame.columns]
        rows.append({"feature_block": block, "available_features": ", ".join(available), "n_available": len(available), "n_defined": len(columns), "coverage": len(available) / max(len(columns), 1)})
    return pd.DataFrame(rows)


def build_target_catalog(namespace: Mapping[str, Any]) -> pd.DataFrame:
    df = _as_df(_first(namespace, "dfmodel", "dffeat", "df_model", "df_feat"))
    rows = []
    for name, spec in RETURN_TARGETS.items():
        existing_cols = [col for col in df.columns if name in col or str(spec.get("horizon", "")) in col and "ret" in col.lower()]
        rows.append({"target_name": name, "definition": str(spec), "available_proxy_columns": ", ".join(existing_cols[:8]), "status": "available" if existing_cols else "needs_build"})
    return pd.DataFrame(rows)


def add_return_targets(df: pd.DataFrame, price_col: str = "adj_close") -> pd.DataFrame:
    out = df.copy()
    if out.empty or "ticker" not in out.columns:
        return out
    if price_col not in out.columns:
        price_col = "close" if "close" in out.columns else ""
    if not price_col:
        return out
    out = out.sort_values(["ticker", "date"] if "date" in out.columns else ["ticker"]).copy()
    for horizon in [21, 63, 252]:
        out[f"raw_forward_return_{horizon}d"] = out.groupby("ticker")[price_col].shift(-horizon) / out[price_col] - 1
    if "raw_forward_return_252d" in out.columns:
        out["target_raw_forward_return_12m"] = out["raw_forward_return_252d"]
    return out


def build_model_frame(namespace: Mapping[str, Any], active_blocks: list[str] | None = None, target_col: str | None = None) -> tuple[pd.DataFrame, list[str], str | None]:
    df = _as_df(_first(namespace, "dfmodel", "dffeat", "dfmerged", "df_model", "df_feat", "df_merged"))
    if df.empty:
        return pd.DataFrame(), [], None
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = add_return_targets(df)
    active_blocks = active_blocks or ["market", "valuation", "quality", "growth", "leverage", "risk", "macro", "fx", "commodities", "peers"]
    feature_candidates = []
    for block in active_blocks:
        feature_candidates.extend(FEATURE_BLOCK_LIBRARY.get(block, []))
    feature_candidates = [col for col in dict.fromkeys(feature_candidates) if col in df.columns]
    numeric = [col for col in df.select_dtypes(include=[np.number]).columns if col not in feature_candidates and not col.startswith("target_")]
    features = feature_candidates + [col for col in numeric if col not in feature_candidates][:20]
    if target_col is None:
        for candidate in ["target_raw_forward_return_12m", "raw_forward_return_252d", "upside_to_fair_value", "upside"]:
            if candidate in df.columns and df[candidate].notna().sum() >= 20:
                target_col = candidate
                break
    if target_col is None:
        return df, features, None
    usable = df[features + [target_col] + (["date"] if "date" in df.columns else [])].replace([np.inf, -np.inf], np.nan).dropna()
    return usable, features, target_col


def run_model_registry(namespace: Mapping[str, Any]) -> pd.DataFrame:
    experiment = _first(namespace, "EXPERIMENT", default={}) or {}
    active_blocks = experiment.get("active_feature_blocks") or experiment.get("feature_blocks")
    model_frame, features, target_col = build_model_frame(namespace, active_blocks=active_blocks)
    if model_frame.empty or not features or target_col is None or len(model_frame) < 40:
        return pd.DataFrame([{"model": "not_run", "status": "WARN", "detail": "Not enough complete feature/target rows for model registry", "rows": len(model_frame), "target": target_col or "none"}])
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
        from sklearn.linear_model import BayesianRidge, ElasticNet, LinearRegression, Ridge
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:
        return pd.DataFrame([{"model": "sklearn_unavailable", "status": "WARN", "detail": str(exc), "rows": len(model_frame), "target": target_col}])

    data = model_frame.sort_values("date") if "date" in model_frame.columns else model_frame.copy()
    split_idx = int(len(data) * 0.75)
    train = data.iloc[:split_idx].copy()
    test = data.iloc[split_idx:].copy()
    if len(train) < 20 or len(test) < 10:
        return pd.DataFrame([{"model": "not_run", "status": "WARN", "detail": "Insufficient train/test rows after time split", "rows": len(data), "target": target_col}])
    X_train, y_train = train[features], train[target_col]
    X_test, y_test = test[features], test[target_col]
    models = {
        "linear": make_pipeline(StandardScaler(), LinearRegression()),
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "elastic_net": make_pipeline(StandardScaler(), ElasticNet(alpha=0.05, l1_ratio=0.5, random_state=42)),
        "random_forest": RandomForestRegressor(n_estimators=120, max_depth=5, min_samples_leaf=5, random_state=42, n_jobs=-1),
        "gradient_boosting": HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, max_leaf_nodes=15, random_state=42),
        "bayesian_ridge": make_pipeline(StandardScaler(), BayesianRidge()),
    }
    rows = []
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
            mae = float(mean_absolute_error(y_test, pred))
            r2 = float(r2_score(y_test, pred))
            hit = float((np.sign(pred) == np.sign(y_test)).mean())
            ic = float(pd.Series(pred).corr(pd.Series(y_test).reset_index(drop=True), method="spearman"))
            rows.append({"model": name, "status": "PASS", "target": target_col, "n_features": len(features), "train_rows": len(train), "test_rows": len(test), "rmse": rmse, "mae": mae, "r2": r2, "hit_rate": hit, "spearman_ic": ic})
        except Exception as exc:
            rows.append({"model": name, "status": "FAIL", "target": target_col, "detail": str(exc)})
    return pd.DataFrame(rows)


def build_peer_analysis(namespace: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
    latest = _as_df(_first(namespace, "latestcrosssection", "latest_cross_section", "companyranking", "company_ranking"))
    master = _first(namespace, "MASTER_REQUEST", default={}) or {}
    target = normalize_ticker(master.get("ticker") or master.get("target_ticker") or "")
    if latest.empty or "ticker" not in latest.columns:
        empty = pd.DataFrame([{"status": "WARN", "detail": "latest cross-section unavailable for peer analysis"}])
        return {"selected_peers": empty, "peer_similarity": empty, "relative_multiples": empty, "target_vs_peers": empty, "peer_warnings": empty}
    data = latest.copy()
    data["ticker"] = data["ticker"].map(normalize_ticker)
    target_row = data[data["ticker"] == target]
    if target_row.empty:
        target_row = data.head(1)
        target = str(target_row.iloc[0]["ticker"])
    target_row = target_row.iloc[0]
    manual = master.get("manual_peers") or []
    if isinstance(manual, str):
        manual = [x.strip().upper() for x in manual.split(",") if x.strip()]
    peer_pool = data[data["ticker"] != target].copy()
    if manual:
        peer_pool = peer_pool[peer_pool["ticker"].isin(manual)].copy()
    score_cols = [col for col in ["market_cap", "revenue", "roe", "debt_equity", "pe_ratio", "pb_ratio", "revenue_growth"] if col in peer_pool.columns and col in data.columns]
    for col in score_cols:
        base = pd.to_numeric(target_row.get(col), errors="coerce")
        vals = pd.to_numeric(peer_pool[col], errors="coerce")
        denom = np.nanmedian(np.abs(vals)) or 1.0
        peer_pool[f"sim_{col}"] = 1.0 / (1.0 + np.abs(vals - base) / denom)
    if "sector" in peer_pool.columns and "sector" in data.columns:
        peer_pool["sim_sector"] = (peer_pool["sector"].astype(str) == str(target_row.get("sector"))).astype(float)
    if "country" in peer_pool.columns and "country" in data.columns:
        peer_pool["sim_country"] = (peer_pool["country"].astype(str) == str(target_row.get("country"))).astype(float)
    sim_cols = [col for col in peer_pool.columns if col.startswith("sim_")]
    peer_pool["peer_similarity_score"] = peer_pool[sim_cols].mean(axis=1) if sim_cols else 0.5
    n_peers = int(master.get("n_peers") or 10)
    selected = peer_pool.sort_values("peer_similarity_score", ascending=False).head(n_peers).copy()
    multiple_cols = [col for col in ["pe_ratio", "pb_ratio", "ev_ebitda", "fcf_yield"] if col in data.columns]
    multiples = selected[["ticker", "peer_similarity_score"] + multiple_cols].copy() if not selected.empty else pd.DataFrame()
    comparison_rows = []
    for col in multiple_cols:
        target_val = pd.to_numeric(pd.Series([target_row.get(col)]), errors="coerce").iloc[0]
        peer_median = pd.to_numeric(selected[col], errors="coerce").median() if col in selected.columns else np.nan
        comparison_rows.append({"metric": col, "target": target_val, "peer_median": peer_median, "target_premium_discount": target_val / peer_median - 1 if pd.notna(target_val) and pd.notna(peer_median) and peer_median else np.nan})
    warnings = []
    if len(selected) < max(3, min(n_peers, 5)):
        warnings.append({"warning": "low_peer_count", "detail": f"Only {len(selected)} peers selected"})
    if not multiple_cols:
        warnings.append({"warning": "missing_multiples", "detail": "No PE/PB/EV_EBITDA/FCF yield columns available"})
    return {
        "selected_peers": selected,
        "peer_similarity": selected[["ticker", "peer_similarity_score"] + sim_cols] if not selected.empty else pd.DataFrame(),
        "relative_multiples": multiples,
        "target_vs_peers": pd.DataFrame(comparison_rows),
        "peer_warnings": pd.DataFrame(warnings),
    }


def build_macro_risk_layer(namespace: Mapping[str, Any], api_series: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    api_series = api_series or {}
    rows = []
    for name, df in api_series.items():
        if df.empty:
            continue
        value_col = "adj_close" if "adj_close" in df.columns else "close" if "close" in df.columns else None
        if value_col is None:
            continue
        values = pd.to_numeric(df[value_col], errors="coerce").dropna()
        if values.empty:
            continue
        ret_63d = values.iloc[-1] / values.iloc[max(0, len(values) - 64)] - 1 if len(values) > 2 else np.nan
        group = str(df.get("factor_group", pd.Series(["unknown"])).iloc[0])
        rows.append({"factor": name, "group": group, "latest": float(values.iloc[-1]), "change_63d": float(ret_63d) if pd.notna(ret_63d) else np.nan, "risk_signal": "rising" if pd.notna(ret_63d) and ret_63d > 0.03 else "falling" if pd.notna(ret_63d) and ret_63d < -0.03 else "neutral"})
    if not rows:
        rows.append({"factor": "macro_risk_layer", "group": "status", "latest": np.nan, "change_63d": np.nan, "risk_signal": "No local/API macro-risk series loaded"})
    return pd.DataFrame(rows)


def build_recommendations(data_quality: pd.DataFrame, feature_catalog: pd.DataFrame, model_results: pd.DataFrame, peeranalysis: dict[str, pd.DataFrame], macro_risk: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if not data_quality.empty and (data_quality["status"] == "FAIL").any():
        rows.append({"priority": 1, "area": "data_quality", "recommendation": "Fix failed data-quality checks before trusting valuation or ML outputs."})
    if not feature_catalog.empty and feature_catalog["coverage"].mean() < 0.35:
        rows.append({"priority": 2, "area": "feature_engineering", "recommendation": "Increase local/API coverage for selected feature blocks; too many defined features are unavailable."})
    if not model_results.empty and not (model_results["status"] == "PASS").any():
        rows.append({"priority": 3, "area": "modelling", "recommendation": "Model registry did not produce a valid train/test run; inspect target availability and NaNs."})
    peer_warnings = peeranalysis.get("peer_warnings", pd.DataFrame())
    if not peer_warnings.empty:
        rows.append({"priority": 4, "area": "peers", "recommendation": "Peer analysis raised warnings; review selected peers and comparable multiples."})
    if not macro_risk.empty and "No local/API" in str(macro_risk.iloc[0].get("risk_signal", "")):
        rows.append({"priority": 5, "area": "macro_risk", "recommendation": "Connect local rates/FX/commodity files or enable API refresh for macro-risk integration."})
    if not rows:
        rows.append({"priority": 1, "area": "platform", "recommendation": "No blocking platform issues detected by the defensive checks."})
    return pd.DataFrame(rows).sort_values("priority").reset_index(drop=True)


def build_research_platform(namespace: Mapping[str, Any], refresh_api: bool = False) -> ResearchPlatformOutputs:
    db_base = get_db_base(namespace)
    inventory = discover_local_database(db_base)
    local_risk = load_local_risk_factor_candidates(inventory)
    experiment = _first(namespace, "EXPERIMENT", default={}) or {}
    start_date = experiment.get("start_date")
    end_date = experiment.get("end_date")
    api_series = fetch_yfinance_risk_series(start_date=start_date, end_date=end_date) if refresh_api else {}
    if not api_series and local_risk:
        api_series = {name: df for name, df in local_risk.items()}
    source_provenance = build_source_provenance(inventory, api_series)
    data_quality = run_data_quality(namespace)
    feature_catalog = build_feature_catalog(namespace)
    target_catalog = build_target_catalog(namespace)
    model_results = run_model_registry(namespace)
    peeranalysis = build_peer_analysis(namespace)
    macro_risk = build_macro_risk_layer(namespace, api_series=api_series)
    recommendations = build_recommendations(data_quality, feature_catalog, model_results, peeranalysis, macro_risk)
    return ResearchPlatformOutputs(
        data_inventory=inventory,
        data_quality=data_quality,
        feature_catalog=feature_catalog,
        target_catalog=target_catalog,
        model_results=model_results,
        peeranalysis=peeranalysis,
        macro_risk=macro_risk,
        source_provenance=source_provenance,
        recommendations=recommendations,
    )


def inject_platform_outputs(namespace: dict[str, Any], outputs: ResearchPlatformOutputs) -> dict[str, Any]:
    namespace["research_platform_outputs"] = {
        "data_inventory": outputs.data_inventory,
        "data_quality": outputs.data_quality,
        "feature_catalog": outputs.feature_catalog,
        "target_catalog": outputs.target_catalog,
        "model_results": outputs.model_results,
        "peeranalysis": outputs.peeranalysis,
        "macro_risk": outputs.macro_risk,
        "source_provenance": outputs.source_provenance,
        "recommendations": outputs.recommendations,
    }
    namespace["qa_report"] = outputs.data_quality
    namespace["feature_catalog"] = outputs.feature_catalog
    namespace["target_catalog"] = outputs.target_catalog
    namespace["model_results"] = outputs.model_results
    namespace["peeranalysis"] = outputs.peeranalysis
    namespace["macro_risk"] = outputs.macro_risk
    namespace["source_provenance"] = outputs.source_provenance
    namespace["platform_recommendations"] = outputs.recommendations
    return namespace
