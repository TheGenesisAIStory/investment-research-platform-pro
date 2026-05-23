"""Incremental research-platform extensions for the valuation notebook.

The functions here are intentionally notebook-friendly: they preserve the
existing globals and add source registries, cache-aware refresh hooks, broader
universe construction, richer ML validation, and cloud portability metadata
without changing the economics of the valuation cells.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
import shutil

import numpy as np
import pandas as pd


UTC_NOW = lambda: datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class ProviderSpec:
    provider: str
    domains: list[str]
    priority: int
    requires_key: bool
    key_env: str | None
    free_tier: str
    rate_limit_estimate: str
    cache_policy: str
    fallback_after: list[str]


PROVIDER_REGISTRY: list[ProviderSpec] = [
    ProviderSpec("local_cache", ["all"], 0, False, None, "primary local lake", "filesystem", "read first, write every successful refresh", []),
    ProviderSpec("google_drive_database", ["all"], 1, False, None, "primary persistent lake in Colab", "drive/filesystem", "read first, write curated outputs", ["local_cache"]),
    ProviderSpec("yfinance", ["prices", "fundamentals_light", "metadata", "fx", "rates", "commodities", "benchmarks"], 2, False, None, "free unofficial", "polite batching", "cache parquet/csv per request", ["local_cache"]),
    ProviderSpec("stooq", ["prices", "indices", "fx", "rates", "commodities"], 3, False, None, "free", "polite batching", "cache csv per symbol", ["yfinance"]),
    ProviderSpec("fred", ["macro", "rates"], 4, False, None, "free API/key optional", "polite requests", "cache csv/json per series", ["stooq"]),
    ProviderSpec("ecb", ["rates", "fx", "macro_europe"], 5, False, None, "free", "polite requests", "cache response per series", ["fred"]),
    ProviderSpec("eurostat", ["macro_europe"], 6, False, None, "free", "polite requests", "cache response per dataset", ["ecb"]),
    ProviderSpec("sec_edgar", ["us_filings", "fundamentals_us"], 7, False, None, "free", "SEC fair access", "cache submissions/company facts", ["yfinance"]),
    ProviderSpec("fmp", ["fundamentals", "profiles", "ratios", "prices"], 8, True, "FMP_API_KEY", "free tier if key available", "provider tier dependent", "cache every endpoint", ["yfinance"]),
    ProviderSpec("finnhub", ["fundamentals", "profiles", "estimates"], 9, True, "FINNHUB_API_KEY", "free tier if key available", "provider tier dependent", "cache every endpoint", ["fmp"]),
    ProviderSpec("alpha_vantage", ["prices", "fundamentals", "fx", "macro"], 10, True, "ALPHA_VANTAGE_API_KEY", "free tier if key available", "low free quota", "cache every endpoint", ["finnhub"]),
    ProviderSpec("eodhd", ["prices", "fundamentals", "indices"], 11, True, "EODHD_API_KEY", "trial/free-compatible if key available", "provider tier dependent", "cache every endpoint", ["alpha_vantage"]),
]


API_KEY_ENV_VARS = [
    {"provider": "FMP", "env_var": "FMP_API_KEY", "domain": "fundamentals, ratios, profiles"},
    {"provider": "Finnhub", "env_var": "FINNHUB_API_KEY", "domain": "profiles, fundamentals, estimates"},
    {"provider": "Alpha Vantage", "env_var": "ALPHA_VANTAGE_API_KEY", "domain": "prices, fundamentals, FX, macro"},
    {"provider": "EODHD", "env_var": "EODHD_API_KEY", "domain": "prices, fundamentals, indices"},
    {"provider": "FRED", "env_var": "FRED_API_KEY", "domain": "macro, rates"},
]


VALUATION_MODEL_REGISTRY = [
    {"model": "dcf_fcff", "family": "intrinsic", "required_fields": "free_cash_flow or operating_cash_flow/capex, discount_rate, growth", "output": "equity_value_per_share", "status": "active_if_data_available"},
    {"model": "dcf_fcfe", "family": "intrinsic", "required_fields": "free_cash_flow_to_equity or free_cash_flow, cost_of_equity", "output": "equity_value_per_share", "status": "active_if_data_available"},
    {"model": "two_stage_dcf", "family": "intrinsic", "required_fields": "free_cash_flow, high_growth, terminal_growth", "output": "equity_value_per_share", "status": "active_if_data_available"},
    {"model": "reverse_dcf", "family": "expectations", "required_fields": "current_price, free_cash_flow, shares_outstanding", "output": "implied_growth_rate", "status": "diagnostic"},
    {"model": "residual_income", "family": "accounting", "required_fields": "book_value_per_share, roe, cost_of_equity", "output": "equity_value_per_share", "status": "active_if_data_available"},
    {"model": "economic_profit_eva", "family": "accounting", "required_fields": "nopat, invested_capital, wacc", "output": "enterprise_value_proxy", "status": "active_if_data_available"},
    {"model": "ddm_gordon", "family": "dividend", "required_fields": "dividend_per_share, cost_of_equity, dividend_growth", "output": "equity_value_per_share", "status": "active_if_data_available"},
    {"model": "ddm_two_stage", "family": "dividend", "required_fields": "dividend_per_share, near_growth, terminal_growth", "output": "equity_value_per_share", "status": "active_if_data_available"},
    {"model": "relative_pe", "family": "relative", "required_fields": "eps, peer pe_ratio", "output": "peer_implied_price", "status": "active_if_data_available"},
    {"model": "relative_pb", "family": "relative", "required_fields": "book_value_per_share, peer pb_ratio", "output": "peer_implied_price", "status": "active_if_data_available"},
    {"model": "relative_ev_ebitda", "family": "relative", "required_fields": "ebitda, net_debt, peer ev_ebitda", "output": "peer_implied_price", "status": "active_if_data_available"},
    {"model": "relative_sales", "family": "relative", "required_fields": "revenue, peer sales_multiple", "output": "peer_implied_price", "status": "active_if_data_available"},
    {"model": "asset_based_book", "family": "balance_sheet", "required_fields": "book_value_per_share or equity/shares", "output": "equity_floor_value", "status": "diagnostic"},
    {"model": "analyst_consensus", "family": "external", "required_fields": "target_price or analyst_target", "output": "consensus_value", "status": "active_if_data_available"},
    {"model": "sotp_placeholder", "family": "segment", "required_fields": "segment revenue/EBITDA if available", "output": "sum_of_the_parts_value", "status": "scaffold"},
]


UNIVERSE_STATIC_FALLBACKS = {
    "ftse_mib": ["A2A.MI", "AMP.MI", "ATL.MI", "AZM.MI", "BAMI.MI", "BMED.MI", "BMPS.MI", "BPE.MI", "CPR.MI", "DIA.MI", "ENEL.MI", "ENI.MI", "ERG.MI", "FBK.MI", "G.MI", "HER.MI", "IG.MI", "INW.MI", "ISP.MI", "IVG.MI", "LDO.MI", "MB.MI", "MONC.MI", "NEXI.MI", "PIRC.MI", "PST.MI", "PRY.MI", "RACE.MI", "REC.MI", "SRG.MI", "STLAM.MI", "STM.MI", "TEN.MI", "TIT.MI", "TRN.MI", "UCG.MI", "UNI.MI"],
    "euro_stoxx_50": ["ASML.AS", "MC.PA", "SAP.DE", "OR.PA", "TTE.PA", "SAN.PA", "SIE.DE", "SU.PA", "AIR.PA", "ALV.DE", "BNP.PA", "IBE.MC", "ENEL.MI", "ITX.MC", "DTE.DE", "DG.PA", "AI.PA", "EL.PA", "ENI.MI", "ISP.MI"],
    "sp500_core_sample": ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B", "LLY", "AVGO", "JPM", "XOM", "UNH", "V", "PG", "MA", "COST", "HD", "MRK", "ABBV", "CRM"],
}


def _first(namespace: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in namespace and namespace[name] is not None:
            return namespace[name]
    return default


def _as_df(value: Any) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, pd.Series):
        return value.to_frame().T
    if isinstance(value, list):
        return pd.DataFrame(value)
    if isinstance(value, dict):
        return pd.DataFrame([value])
    return pd.DataFrame()


def normalize_ticker(ticker: Any) -> str:
    if pd.isna(ticker):
        return ""
    return str(ticker).strip().upper().replace("/", "-")


def detect_runtime_environment() -> dict[str, Any]:
    cwd = Path.cwd().resolve()
    is_colab = "COLAB_RELEASE_TAG" in os.environ or Path("/content").exists()
    is_azure = any(k in os.environ for k in ["AZUREML_RUN_ID", "AZUREML_ARM_SUBSCRIPTION"])
    is_aws = any(k in os.environ for k in ["SAGEMAKER_JOB_NAME", "AWS_EXECUTION_ENV", "AWS_REGION"])
    return {
        "runtime": "colab" if is_colab else "azure_ml" if is_azure else "aws" if is_aws else "local",
        "cwd": str(cwd),
        "is_colab": is_colab,
        "is_azure": is_azure,
        "is_aws": is_aws,
        "updated_at": UTC_NOW(),
    }


def sync_config_aliases(namespace: dict[str, Any]) -> dict[str, Any]:
    if "MASTER_REQUEST" not in namespace and "MASTERREQUEST" in namespace:
        namespace["MASTER_REQUEST"] = namespace["MASTERREQUEST"]
    if "MASTERREQUEST" not in namespace and "MASTER_REQUEST" in namespace:
        namespace["MASTERREQUEST"] = namespace["MASTER_REQUEST"]
    if "USER_SELECTION" not in namespace and "USERSELECTION" in namespace:
        namespace["USER_SELECTION"] = namespace["USERSELECTION"]
    if "USERSELECTION" not in namespace and "USER_SELECTION" in namespace:
        namespace["USERSELECTION"] = namespace["USER_SELECTION"]
    namespace.setdefault("EXPERIMENT", {})
    namespace.setdefault("VALUATIONCONFIG", {})
    namespace.setdefault("MLCONFIG", {})
    namespace.setdefault("GOVERNANCECONFIG", {})
    namespace.setdefault("FUNDAMENTALSCONFIG", {})
    namespace.setdefault("UNIVERSECONFIG", {})
    return namespace


def resolve_platform_paths(namespace: dict[str, Any]) -> dict[str, Path]:
    output_root = Path(_first(namespace, "OUTPUTROOT", "OUTPUT_ROOT", "OUTPUT_DIR", default=Path.cwd() / "output")).expanduser()
    paths = {
        "OUTPUTROOT": output_root,
        "FIGURESDIR": Path(_first(namespace, "FIGURESDIR", "FIGURES_DIR", default=output_root / "figures")),
        "TABLESDIR": Path(_first(namespace, "TABLESDIR", "TABLES_DIR", default=output_root / "tables")),
        "LOGSDIR": Path(_first(namespace, "LOGSDIR", "LOGS_DIR", default=output_root / "logs")),
        "CACHEDIR": Path(_first(namespace, "CACHEDIR", "CACHE_DIR", default=output_root / "cache")),
        "MANIFESTDIR": Path(_first(namespace, "MANIFESTDIR", "MANIFEST_DIR", default=output_root / "manifests")),
    }
    for key, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        namespace[key] = path
    namespace.setdefault("OUTPUT_ROOT", output_root)
    namespace.setdefault("OUTPUT_DIR", output_root)
    return paths


def provider_registry_frame() -> pd.DataFrame:
    rows = [asdict(p) for p in PROVIDER_REGISTRY]
    for row in rows:
        row["domains"] = ", ".join(row["domains"])
        row["fallback_after"] = ", ".join(row["fallback_after"])
        row["key_available"] = bool(row["key_env"] and os.environ.get(row["key_env"]))
    return pd.DataFrame(rows).sort_values("priority").reset_index(drop=True)


def api_key_status_frame() -> pd.DataFrame:
    rows = []
    for item in API_KEY_ENV_VARS:
        value = os.environ.get(item["env_var"])
        rows.append(
            {
                **item,
                "configured": bool(value),
                "masked_value": f"{value[:3]}...{value[-3:]}" if value and len(value) >= 8 else "not_set",
                "updated_at": UTC_NOW(),
            }
        )
    return pd.DataFrame(rows)


def valuation_model_registry_frame() -> pd.DataFrame:
    return pd.DataFrame(VALUATION_MODEL_REGISTRY)


def _num(row: Mapping[str, Any], names: Iterable[str], default: float = np.nan) -> float:
    for name in names:
        if name in row and pd.notna(row[name]):
            try:
                return float(row[name])
            except Exception:
                continue
    return default


def _latest_valuation_rows(namespace: Mapping[str, Any]) -> pd.DataFrame:
    for name in ["valuation_output", "valuationoutput", "latestcrosssection", "latest_cross_section", "companyranking", "dfvaluation"]:
        df = _as_df(namespace.get(name))
        if not df.empty:
            return df.copy()
    return pd.DataFrame()


def run_extended_valuation_models(namespace: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
    """Run additional valuation diagnostics when data exists.

    The function does not fabricate missing fundamentals. Each unavailable model
    returns an explicit SKIP row with required fields.
    """
    data = _latest_valuation_rows(namespace)
    registry = valuation_model_registry_frame()
    experiment = _first(namespace, "EXPERIMENT", default={}) or {}
    valuation_config = _first(namespace, "VALUATIONCONFIG", "VALUATION_CONFIG", default={}) or {}
    if data.empty:
        return {
            "valuation_model_registry": registry,
            "extended_valuation_results": pd.DataFrame([{"model": "all", "status": "WARN", "reason": "No latest valuation/cross-section table available"}]),
            "valuation_assumption_table": pd.DataFrame(),
            "valuation_gap_table": pd.DataFrame(),
        }

    discount_rate = float(experiment.get("discount_rate", experiment.get("cost_of_equity", valuation_config.get("discount_rate", 0.09))) or 0.09)
    terminal_growth = float(experiment.get("dcf_perpetual_growth", experiment.get("terminal_growth", valuation_config.get("terminal_growth", 0.02))) or 0.02)
    high_growth = float(valuation_config.get("high_growth_rate", experiment.get("high_growth_rate", max(terminal_growth + 0.02, 0.04))) or 0.04)
    horizon = int(experiment.get("dcf_horizon_years", valuation_config.get("dcf_horizon_years", 10)) or 10)
    cost_of_equity = float(experiment.get("cost_of_equity", discount_rate) or discount_rate)

    peer_medians = {}
    for col in ["pe_ratio", "pb_ratio", "ev_ebitda", "sales_multiple_proxy", "ev_sales"]:
        if col in data.columns:
            peer_medians[col] = pd.to_numeric(data[col], errors="coerce").replace([np.inf, -np.inf], np.nan).median()

    rows = []
    gap_rows = []
    for _, row in data.iterrows():
        ticker = normalize_ticker(row.get("ticker", ""))
        price = _num(row, ["current_price", "adj_close", "close", "last_price", "price"])
        shares = _num(row, ["shares_outstanding", "weighted_average_shares", "shares"], default=1.0)
        if not np.isfinite(shares) or shares <= 0:
            shares = 1.0
        fcf = _num(row, ["free_cash_flow", "fcf", "free_cashflow", "operating_cash_flow"])
        eps = _num(row, ["eps", "trailing_eps", "earnings_per_share"])
        bvps = _num(row, ["book_value_per_share", "bvps"])
        if not np.isfinite(bvps):
            equity = _num(row, ["equity", "total_equity", "book_value"])
            if np.isfinite(equity) and shares:
                bvps = equity / shares
        roe = _num(row, ["roe", "return_on_equity"])
        dps = _num(row, ["dividend_per_share", "annual_dividend", "dividend_rate"])
        ebitda = _num(row, ["ebitda"])
        net_debt = _num(row, ["net_debt"], default=0.0)
        revenue = _num(row, ["revenue", "total_revenue"])
        target_price = _num(row, ["target_price", "target_mean_price", "analyst_target", "consensus_target"])

        def add(model: str, value: float, status: str = "OK", reason: str = ""):
            upside = value / price - 1 if np.isfinite(value) and np.isfinite(price) and price else np.nan
            rows.append({"ticker": ticker, "model": model, "status": status, "value_per_share": value, "current_price": price, "upside": upside, "reason": reason, "updated_at": UTC_NOW()})

        if np.isfinite(fcf) and discount_rate > terminal_growth:
            fcf_per_share = fcf / shares if abs(fcf) > price * 20 else fcf
            tv = fcf_per_share * (1 + terminal_growth) / (discount_rate - terminal_growth)
            pv_stage = sum((fcf_per_share * ((1 + terminal_growth) ** year)) / ((1 + discount_rate) ** year) for year in range(1, horizon + 1))
            add("dcf_fcff", pv_stage + tv / ((1 + discount_rate) ** horizon))
            high_stage = sum((fcf_per_share * ((1 + high_growth) ** year)) / ((1 + discount_rate) ** year) for year in range(1, min(5, horizon) + 1))
            mature_fcf = fcf_per_share * ((1 + high_growth) ** min(5, horizon))
            mature_tv = mature_fcf * (1 + terminal_growth) / (discount_rate - terminal_growth)
            add("two_stage_dcf", high_stage + mature_tv / ((1 + discount_rate) ** min(5, horizon)))
            if np.isfinite(price) and price > 0 and fcf_per_share > 0:
                implied_growth = discount_rate - (fcf_per_share / price)
                rows.append({"ticker": ticker, "model": "reverse_dcf", "status": "OK", "value_per_share": np.nan, "current_price": price, "upside": np.nan, "implied_growth_rate": implied_growth, "reason": "single-stage approximation", "updated_at": UTC_NOW()})
        else:
            add("dcf_fcff", np.nan, "SKIP", "free_cash_flow unavailable or discount_rate <= terminal_growth")
            add("two_stage_dcf", np.nan, "SKIP", "free_cash_flow unavailable or discount_rate <= terminal_growth")

        if np.isfinite(fcf) and cost_of_equity > terminal_growth:
            fcf_per_share = fcf / shares if abs(fcf) > price * 20 else fcf
            add("dcf_fcfe", fcf_per_share * (1 + terminal_growth) / (cost_of_equity - terminal_growth))
        else:
            add("dcf_fcfe", np.nan, "SKIP", "free_cash_flow unavailable or cost_of_equity <= terminal_growth")

        if np.isfinite(bvps) and np.isfinite(roe) and cost_of_equity > terminal_growth:
            residual = ((roe - cost_of_equity) * bvps) / (cost_of_equity - terminal_growth)
            add("residual_income", bvps + residual)
        else:
            add("residual_income", np.nan, "SKIP", "book_value_per_share/roe unavailable")

        if np.isfinite(dps) and cost_of_equity > terminal_growth:
            add("ddm_gordon", dps * (1 + terminal_growth) / (cost_of_equity - terminal_growth))
            near = sum((dps * ((1 + high_growth) ** year)) / ((1 + cost_of_equity) ** year) for year in range(1, 6))
            terminal_dps = dps * ((1 + high_growth) ** 5) * (1 + terminal_growth)
            add("ddm_two_stage", near + (terminal_dps / (cost_of_equity - terminal_growth)) / ((1 + cost_of_equity) ** 5))
        else:
            add("ddm_gordon", np.nan, "SKIP", "dividend_per_share unavailable or cost_of_equity <= terminal_growth")
            add("ddm_two_stage", np.nan, "SKIP", "dividend_per_share unavailable or cost_of_equity <= terminal_growth")

        if np.isfinite(eps) and np.isfinite(peer_medians.get("pe_ratio", np.nan)):
            add("relative_pe", eps * peer_medians["pe_ratio"])
        else:
            add("relative_pe", np.nan, "SKIP", "eps or peer PE unavailable")
        if np.isfinite(bvps) and np.isfinite(peer_medians.get("pb_ratio", np.nan)):
            add("relative_pb", bvps * peer_medians["pb_ratio"])
        else:
            add("relative_pb", np.nan, "SKIP", "book value or peer PB unavailable")
        if np.isfinite(ebitda) and np.isfinite(peer_medians.get("ev_ebitda", np.nan)):
            ev = ebitda * peer_medians["ev_ebitda"]
            add("relative_ev_ebitda", (ev - net_debt) / shares)
        else:
            add("relative_ev_ebitda", np.nan, "SKIP", "EBITDA or peer EV/EBITDA unavailable")
        sales_multiple = peer_medians.get("sales_multiple_proxy", peer_medians.get("ev_sales", np.nan))
        if np.isfinite(revenue) and np.isfinite(sales_multiple):
            add("relative_sales", (revenue * sales_multiple - net_debt) / shares)
        else:
            add("relative_sales", np.nan, "SKIP", "revenue or peer sales multiple unavailable")
        if np.isfinite(bvps):
            add("asset_based_book", bvps)
        else:
            add("asset_based_book", np.nan, "SKIP", "book_value_per_share unavailable")
        if np.isfinite(target_price):
            add("analyst_consensus", target_price)
        else:
            add("analyst_consensus", np.nan, "SKIP", "analyst target unavailable")
        add("sotp_placeholder", np.nan, "SCAFFOLD", "requires segment-level revenue/EBITDA inputs")

    results = pd.DataFrame(rows)
    if not results.empty:
        ok = results[results["status"].eq("OK") & results["value_per_share"].notna()]
        if not ok.empty:
            summary = ok.groupby("ticker")["value_per_share"].agg(["count", "median", "mean", "min", "max"]).reset_index()
            price_map = ok.groupby("ticker")["current_price"].first()
            summary["current_price"] = summary["ticker"].map(price_map)
            summary["blended_extended_fair_value"] = summary["median"]
            summary["extended_upside"] = summary["blended_extended_fair_value"] / summary["current_price"] - 1
            gap_rows = summary.to_dict("records")
    assumptions = pd.DataFrame([{
        "discount_rate": discount_rate,
        "cost_of_equity": cost_of_equity,
        "terminal_growth": terminal_growth,
        "high_growth": high_growth,
        "dcf_horizon_years": horizon,
        "updated_at": UTC_NOW(),
    }])
    return {
        "valuation_model_registry": registry,
        "extended_valuation_results": results,
        "valuation_assumption_table": assumptions,
        "valuation_gap_table": pd.DataFrame(gap_rows),
    }


def _cache_key(provider: str, domain: str, payload: Any) -> str:
    raw = json.dumps({"provider": provider, "domain": domain, "payload": payload}, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def cache_paths(cache_dir: Path, provider: str, domain: str, payload: Any, ext: str = "parquet") -> tuple[Path, Path]:
    key = _cache_key(provider, domain, payload)
    folder = cache_dir / provider / domain
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{key}.{ext}", folder / f"{key}.metadata.json"


def read_cached(path: Path, max_age_hours: float | None = None) -> pd.DataFrame | None:
    if not path.exists():
        return None
    if max_age_hours is not None:
        age_hours = (time.time() - path.stat().st_mtime) / 3600
        if age_hours > max_age_hours:
            return None
    try:
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        if path.suffix == ".csv":
            return pd.read_csv(path)
        if path.suffix == ".json":
            return pd.read_json(path)
    except Exception:
        return None
    return None


def write_cached(df: pd.DataFrame, path: Path, meta_path: Path, metadata: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if path.suffix == ".parquet":
            df.to_parquet(path, index=False)
        else:
            df.to_csv(path, index=False)
    except Exception:
        fallback = path.with_suffix(".csv")
        df.to_csv(fallback, index=False)
        path = fallback
    meta = dict(metadata)
    meta.update({"cache_path": str(path), "updated_at": UTC_NOW(), "rows": int(len(df)), "columns": list(map(str, df.columns))})
    meta_path.write_text(json.dumps(meta, indent=2, default=str))


def cached_provider_call(
    namespace: Mapping[str, Any],
    provider: str,
    domain: str,
    payload: Mapping[str, Any],
    loader: Callable[[], pd.DataFrame],
    max_age_hours: float = 24 * 7,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cache_dir = Path(_first(namespace, "CACHEDIR", "CACHE_DIR", default=Path.cwd() / "output" / "cache"))
    path, meta_path = cache_paths(cache_dir, provider, domain, payload)
    cached = read_cached(path, max_age_hours=max_age_hours)
    if cached is not None and not cached.empty:
        return cached, {"provider": provider, "domain": domain, "status": "cache_hit", "cache_path": str(path), "updated_at": UTC_NOW()}
    try:
        df = loader()
        if df is None:
            df = pd.DataFrame()
        if not df.empty:
            write_cached(df, path, meta_path, {"provider": provider, "domain": domain, "payload": dict(payload), "status": "refreshed"})
            return df, {"provider": provider, "domain": domain, "status": "refreshed", "cache_path": str(path), "updated_at": UTC_NOW()}
        return pd.DataFrame(), {"provider": provider, "domain": domain, "status": "empty_response", "cache_path": str(path), "updated_at": UTC_NOW()}
    except Exception as exc:
        return pd.DataFrame(), {"provider": provider, "domain": domain, "status": "failed", "error": str(exc), "cache_path": str(path), "updated_at": UTC_NOW()}


def load_wikipedia_constituents(label: str, url: str, ticker_column_candidates: Iterable[str]) -> pd.DataFrame:
    try:
        tables = pd.read_html(url)
    except Exception:
        return pd.DataFrame()
    for table in tables:
        cols = {str(c).strip().lower(): c for c in table.columns}
        for candidate in ticker_column_candidates:
            if candidate.lower() in cols:
                col = cols[candidate.lower()]
                out = table.copy()
                out["ticker"] = out[col].map(normalize_ticker)
                out["universe_source"] = label
                out["source_url"] = url
                out["updated_at"] = UTC_NOW()
                return out
    return pd.DataFrame()


def local_database_universe(namespace: Mapping[str, Any]) -> pd.DataFrame:
    frames = []
    for name in ["dfmarket", "dfpanel", "dffund", "dfmerged", "latestcrosssection", "companyranking"]:
        df = _as_df(namespace.get(name))
        if not df.empty and "ticker" in df.columns:
            tmp = df[["ticker"]].dropna().drop_duplicates().copy()
            tmp["universe_source"] = f"local_{name}"
            tmp["updated_at"] = UTC_NOW()
            frames.append(tmp)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_universe_master(namespace: dict[str, Any], refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = resolve_platform_paths(namespace)
    cache_file = paths["CACHEDIR"] / "universe" / "universe_master.parquet"
    cached = None if refresh else read_cached(cache_file, max_age_hours=24 * 14)
    if cached is not None and not cached.empty:
        return cached, pd.DataFrame([{"source": "cache", "status": "cache_hit", "rows": len(cached), "updated_at": UTC_NOW()}])

    frames = [local_database_universe(namespace)]
    metadata_rows = [{"source": "local_database", "status": "loaded", "rows": len(frames[0]), "updated_at": UTC_NOW()}]

    wikipedia_sources = [
        ("sp500", "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", ["Symbol"]),
        ("euro_stoxx_50", "https://en.wikipedia.org/wiki/EURO_STOXX_50", ["Ticker", "Ticker symbol"]),
        ("ftse_mib", "https://en.wikipedia.org/wiki/FTSE_MIB", ["Ticker", "Ticker symbol"]),
        ("stoxx_europe_600", "https://en.wikipedia.org/wiki/STOXX_Europe_600", ["Ticker", "Ticker symbol"]),
    ]
    for label, url, candidates in wikipedia_sources:
        df = load_wikipedia_constituents(label, url, candidates)
        frames.append(df)
        metadata_rows.append({"source": label, "status": "loaded" if not df.empty else "unavailable", "rows": len(df), "updated_at": UTC_NOW(), "url": url})

    for label, tickers in UNIVERSE_STATIC_FALLBACKS.items():
        df = pd.DataFrame({"ticker": tickers, "universe_source": f"static_{label}", "updated_at": UTC_NOW()})
        frames.append(df)
        metadata_rows.append({"source": f"static_{label}", "status": "loaded", "rows": len(df), "updated_at": UTC_NOW()})

    master = pd.concat([f for f in frames if not f.empty], ignore_index=True) if frames else pd.DataFrame()
    if master.empty:
        master = pd.DataFrame(columns=["ticker", "universe_source", "updated_at"])
    master["ticker"] = master["ticker"].map(normalize_ticker)
    master = master[master["ticker"].ne("")].copy()
    master["source_count"] = master.groupby("ticker")["universe_source"].transform("nunique")
    if "Security" in master.columns and "company_name" not in master.columns:
        master["company_name"] = master["Security"]
    master = master.sort_values(["source_count", "ticker"], ascending=[False, True]).drop_duplicates("ticker").reset_index(drop=True)
    write_cached(master, cache_file, cache_file.with_suffix(".metadata.json"), {"provider": "universe_builder", "domain": "benchmark_constituents", "status": "refreshed"})
    return master, pd.DataFrame(metadata_rows)


def fetch_yfinance_prices(namespace: Mapping[str, Any], tickers: list[str], start: str | None, end: str | None, max_tickers: int = 100) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        import yfinance as yf
    except Exception as exc:
        return pd.DataFrame(), pd.DataFrame([{"provider": "yfinance", "domain": "prices", "status": "unavailable", "error": str(exc), "updated_at": UTC_NOW()}])
    rows, meta = [], []
    for ticker in tickers[:max_tickers]:
        def loader(t=ticker):
            data = yf.download(t, start=start, end=end, progress=False, auto_adjust=False)
            if data is None or data.empty:
                return pd.DataFrame()
            data = data.reset_index()
            data.columns = [str(c).lower().replace(" ", "_") for c in data.columns]
            data["ticker"] = t
            data["source"] = "yfinance"
            data["updated_at"] = UTC_NOW()
            return data
        df, m = cached_provider_call(namespace, "yfinance", "prices", {"ticker": ticker, "start": start, "end": end}, loader, max_age_hours=24)
        rows.append(df)
        meta.append(m)
    return (pd.concat([r for r in rows if not r.empty], ignore_index=True) if rows else pd.DataFrame(), pd.DataFrame(meta))


def fetch_yfinance_fundamentals_light(namespace: Mapping[str, Any], tickers: list[str], max_tickers: int = 100) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        import yfinance as yf
    except Exception as exc:
        return pd.DataFrame(), pd.DataFrame([{"provider": "yfinance", "domain": "fundamentals_light", "status": "unavailable", "error": str(exc), "updated_at": UTC_NOW()}])
    rows, meta = [], []
    wanted = ["marketCap", "enterpriseValue", "trailingPE", "priceToBook", "returnOnEquity", "debtToEquity", "sector", "industry", "country", "currency", "dividendYield"]
    for ticker in tickers[:max_tickers]:
        def loader(t=ticker):
            info = yf.Ticker(t).get_info()
            row = {"ticker": t, "source": "yfinance_info", "updated_at": UTC_NOW()}
            row.update({k: info.get(k) for k in wanted})
            return pd.DataFrame([row])
        df, m = cached_provider_call(namespace, "yfinance", "fundamentals_light", {"ticker": ticker}, loader, max_age_hours=24 * 14)
        rows.append(df)
        meta.append(m)
    return (pd.concat([r for r in rows if not r.empty], ignore_index=True) if rows else pd.DataFrame(), pd.DataFrame(meta))


def add_forward_return_targets(df: pd.DataFrame, horizons: Iterable[int] = (21, 63, 252), price_col: str = "close") -> pd.DataFrame:
    out = df.copy()
    if out.empty or "ticker" not in out.columns or price_col not in out.columns:
        return out
    date_col = "date" if "date" in out.columns else "Date" if "Date" in out.columns else None
    if date_col:
        out = out.sort_values(["ticker", date_col])
    for h in horizons:
        out[f"fwd_return_{h}d"] = out.groupby("ticker")[price_col].shift(-h) / out[price_col] - 1
    return out


def build_ml_feature_frame(namespace: Mapping[str, Any]) -> pd.DataFrame:
    df = _as_df(_first(namespace, "dfmodel", "dffeat", "dfmerged", "dfpanel"))
    if df.empty:
        return df
    config = _first(namespace, "MLCONFIG", default={}) or {}
    blocks = config.get("feature_blocks") or _first(namespace, "EXPERIMENT", default={}).get("active_feature_blocks") or []
    block_cols = {
        "market": ["ret_1d", "ret_21d", "ret_63d", "volume", "close", "adj_close"],
        "valuation": ["pe_ratio", "pb_ratio", "ev_ebitda", "fcf_yield", "upside_to_fair_value"],
        "quality": ["roe", "roa", "gross_margin", "operating_margin", "free_cash_flow_margin"],
        "leverage": ["debt_equity", "net_debt_ebitda", "interest_coverage", "total_debt"],
        "factor": ["beta", "size_factor", "value_factor", "quality_factor", "momentum_factor"],
        "macro": ["rate_10y", "rate_2y", "yield_curve_slope", "inflation_proxy", "credit_spread"],
        "forensic": ["accruals_ratio", "beneish_m_score", "altman_z_score"],
        "sentiment": ["news_sentiment", "filing_sentiment", "analyst_sentiment"],
    }
    selected = []
    for block in blocks:
        selected.extend(block_cols.get(block, []))
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    selected = [c for c in dict.fromkeys(selected) if c in df.columns]
    if not selected:
        selected = [c for c in numeric if c not in {"target", "forward_return", "fwd_return_21d", "fwd_return_63d", "fwd_return_252d"}]
    keep = [c for c in ["ticker", "date", "effective_date"] if c in df.columns] + selected
    target = config.get("target") or _first(namespace, "EXPERIMENT", default={}).get("return_target")
    if target in df.columns:
        keep.append(target)
    return df[list(dict.fromkeys(keep))].copy()


def time_aware_split(df: pd.DataFrame, date_col: str | None, test_size: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    if df.empty:
        return np.array([], dtype=int), np.array([], dtype=int)
    if date_col and date_col in df.columns:
        ordered = df.sort_values(date_col).index.to_numpy()
    else:
        ordered = df.index.to_numpy()
    cut = max(1, int(len(ordered) * (1 - test_size)))
    return ordered[:cut], ordered[cut:]


def run_ml_research_layer(namespace: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
    frame = build_ml_feature_frame(namespace)
    if frame.empty:
        warn = pd.DataFrame([{"model": "not_run", "status": "WARN", "reason": "No ML frame available"}])
        return {"ml_feature_frame": frame, "ml_leaderboard": warn, "ml_predictions": pd.DataFrame(), "ml_feature_importance": pd.DataFrame(), "ml_validation_windows": pd.DataFrame()}
    config = _first(namespace, "MLCONFIG", default={}) or {}
    target = config.get("target") or _first(namespace, "EXPERIMENT", default={}).get("return_target") or "fwd_return_252d"
    if target not in frame.columns:
        price_col = "close" if "close" in frame.columns else "adj_close" if "adj_close" in frame.columns else None
        if price_col:
            frame = add_forward_return_targets(frame, horizons=[21, 63, 252], price_col=price_col)
            target = "fwd_return_252d"
    if target not in frame.columns:
        warn = pd.DataFrame([{"model": "not_run", "status": "WARN", "reason": f"Target {target} unavailable"}])
        return {"ml_feature_frame": frame, "ml_leaderboard": warn, "ml_predictions": pd.DataFrame(), "ml_feature_importance": pd.DataFrame(), "ml_validation_windows": pd.DataFrame()}
    features = [c for c in frame.columns if c not in {"ticker", "date", "effective_date", target} and pd.api.types.is_numeric_dtype(frame[c])]
    id_cols = [c for c in ["ticker", "date"] if c in frame.columns]
    data = frame[id_cols + features + [target]].replace([np.inf, -np.inf], np.nan).dropna(subset=[target])
    if len(data) < int(config.get("min_train_rows", 40)):
        warn = pd.DataFrame([{"model": "not_run", "status": "WARN", "reason": "Insufficient labelled rows", "rows": len(data), "target": target}])
        return {"ml_feature_frame": frame, "ml_leaderboard": warn, "ml_predictions": pd.DataFrame(), "ml_feature_importance": pd.DataFrame(), "ml_validation_windows": pd.DataFrame()}
    X = data[features].fillna(data[features].median(numeric_only=True))
    y = pd.to_numeric(data[target], errors="coerce")
    train_idx, test_idx = time_aware_split(data, "date" if "date" in data.columns else None, float(config.get("test_size", 0.25)))
    if len(test_idx) == 0 or len(train_idx) < 10:
        train_idx, test_idx = np.arange(max(1, len(data) - 10)), np.arange(max(1, len(data) - 10), len(data))

    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import BayesianRidge, ElasticNet, Lasso, LinearRegression, Ridge
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    model_specs = {
        "linear": LinearRegression(),
        "ridge": Ridge(alpha=1.0),
        "lasso": Lasso(alpha=0.001, max_iter=5000),
        "elastic_net": ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000),
        "random_forest": RandomForestRegressor(n_estimators=150, random_state=42, min_samples_leaf=5),
        "gradient_boosting": GradientBoostingRegressor(random_state=42),
        "bayesian_ridge": BayesianRidge(),
    }
    optional_imports = {
        "xgboost": ("xgboost", "XGBRegressor", {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05, "random_state": 42}),
        "lightgbm": ("lightgbm", "LGBMRegressor", {"n_estimators": 200, "learning_rate": 0.05, "random_state": 42, "verbose": -1}),
        "catboost": ("catboost", "CatBoostRegressor", {"iterations": 200, "learning_rate": 0.05, "depth": 4, "random_seed": 42, "verbose": False}),
    }
    for name, (module, cls, kwargs) in optional_imports.items():
        try:
            mod = __import__(module, fromlist=[cls])
            model_specs[name] = getattr(mod, cls)(**kwargs)
        except Exception:
            pass
    requested = config.get("model_families") or _first(namespace, "EXPERIMENT", default={}).get("model_families") or list(model_specs)
    rows, preds, imps = [], [], []
    for name in requested:
        if name not in model_specs:
            rows.append({"model": name, "status": "SKIP", "reason": "library unavailable or unsupported"})
            continue
        model = model_specs[name]
        steps = [("imputer", SimpleImputer(strategy="median"))]
        if name in {"linear", "ridge", "lasso", "elastic_net", "bayesian_ridge"}:
            steps.append(("scaler", StandardScaler()))
        steps.append(("model", model))
        pipe = Pipeline(steps)
        try:
            pipe.fit(X.loc[train_idx], y.loc[train_idx])
            pred = pipe.predict(X.loc[test_idx])
            mse = float(mean_squared_error(y.loc[test_idx], pred))
            rmse = float(np.sqrt(mse))
            rows.append({"model": name, "status": "OK", "target": target, "features": len(features), "train_rows": len(train_idx), "test_rows": len(test_idx), "rmse": rmse, "mae": float(mean_absolute_error(y.loc[test_idx], pred)), "r2": float(r2_score(y.loc[test_idx], pred)) if len(test_idx) > 1 else np.nan, "updated_at": UTC_NOW()})
            pred_df = data.loc[test_idx, [c for c in ["ticker", "date"] if c in data.columns]].copy()
            pred_df["model"] = name
            pred_df["y_true"] = y.loc[test_idx].to_numpy()
            pred_df["y_pred"] = pred
            preds.append(pred_df)
            fitted = pipe.named_steps["model"]
            if hasattr(fitted, "feature_importances_"):
                imp_values = fitted.feature_importances_
            elif hasattr(fitted, "coef_"):
                imp_values = np.ravel(np.abs(fitted.coef_))
            else:
                imp_values = np.zeros(len(features))
            imps.append(pd.DataFrame({"model": name, "feature": features, "importance": imp_values}).sort_values("importance", ascending=False).head(50))
        except Exception as exc:
            rows.append({"model": name, "status": "FAIL", "reason": str(exc), "updated_at": UTC_NOW()})
    leaderboard = pd.DataFrame(rows)
    if "rmse" not in leaderboard.columns:
        leaderboard["rmse"] = np.nan
    if "status" not in leaderboard.columns:
        leaderboard["status"] = "WARN"
    leaderboard = leaderboard.sort_values(["status", "rmse"], ascending=[True, True], na_position="last")
    windows = pd.DataFrame([{"window": "holdout", "train_start": data.loc[train_idx, "date"].min() if "date" in data.columns and len(train_idx) else None, "train_end": data.loc[train_idx, "date"].max() if "date" in data.columns and len(train_idx) else None, "test_start": data.loc[test_idx, "date"].min() if "date" in data.columns and len(test_idx) else None, "test_end": data.loc[test_idx, "date"].max() if "date" in data.columns and len(test_idx) else None, "embargo_rows": int(config.get("embargo_rows", 0)), "updated_at": UTC_NOW()}])
    return {
        "ml_feature_frame": frame,
        "ml_leaderboard": leaderboard,
        "ml_predictions": pd.concat(preds, ignore_index=True) if preds else pd.DataFrame(),
        "ml_feature_importance": pd.concat(imps, ignore_index=True) if imps else pd.DataFrame(),
        "ml_validation_windows": windows,
    }


def build_refresh_plan(namespace: Mapping[str, Any]) -> pd.DataFrame:
    config = _first(namespace, "GOVERNANCECONFIG", default={}) or {}
    return pd.DataFrame([
        {"job": "daily_prices_fx_rates_commodities", "cadence": config.get("market_refresh_cadence", "daily"), "mode": "cache-first then free API", "owner": "notebook/batch hook"},
        {"job": "weekly_fundamentals_profiles", "cadence": config.get("fundamentals_refresh_cadence", "weekly"), "mode": "local cache then yfinance/FMP/Finnhub/AlphaVantage/EODHD", "owner": "notebook/batch hook"},
        {"job": "weekly_universe_constituents", "cadence": "weekly", "mode": "local cache then benchmark pages/static fallbacks", "owner": "notebook/batch hook"},
        {"job": "monthly_research_artifacts", "cadence": "monthly or pre-demo", "mode": "rerun diagnostics, ML, dashboard, report", "owner": "analyst"},
        {"job": "drive_database_sync", "cadence": config.get("drive_sync_cadence", "after every successful run"), "mode": "copy curated tables/cache/manifests to Google Drive data lake", "owner": "notebook/batch hook"},
    ])


def resolve_google_drive_database_root(namespace: Mapping[str, Any]) -> Path:
    for key in ["GOOGLE_DRIVE_DB_ROOT", "DB_BASE", "DATA_PATH", "ML_TRADING_DB_BASE"]:
        value = os.environ.get(key) or namespace.get(key)
        if value:
            return Path(value).expanduser().resolve()
    candidates = [
        Path("/content/drive/MyDrive/Database Finanziario"),
        Path("/content/drive/MyDrive/GitHub/Database Finanziario"),
        Path.home() / "Library/CloudStorage/GoogleDrive-sfn.gns@gmail.com/Il mio Drive/Database Finanziario",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return Path(_first(namespace, "OUTPUTROOT", "OUTPUT_ROOT", "OUTPUT_DIR", default=Path.cwd() / "output")).expanduser().resolve() / "google_drive_database_sync"


def sync_outputs_to_google_drive_database(namespace: Mapping[str, Any], include_cache: bool = True) -> pd.DataFrame:
    """Copy generated research artifacts into the Google Drive financial DB area."""
    output_root = Path(_first(namespace, "OUTPUTROOT", "OUTPUT_ROOT", "OUTPUT_DIR", default=Path.cwd() / "output")).expanduser().resolve()
    drive_root = resolve_google_drive_database_root(namespace)
    sync_root = drive_root / "company_valuation_platform" / "auto_refresh"
    sync_root.mkdir(parents=True, exist_ok=True)
    source_dirs = [
        output_root / "tables",
        output_root / "manifests",
        output_root / "dashboard",
        output_root / "reports",
    ]
    if include_cache:
        source_dirs.append(output_root / "cache")
    rows = []
    for source_dir in source_dirs:
        if not source_dir.exists():
            rows.append({"source": str(source_dir), "target": "", "status": "missing_source", "updated_at": UTC_NOW()})
            continue
        target_dir = sync_root / source_dir.name
        target_dir.mkdir(parents=True, exist_ok=True)
        copied = 0
        for path in source_dir.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(source_dir)
            target = target_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(path, target)
                copied += 1
            except Exception as exc:
                rows.append({"source": str(path), "target": str(target), "status": "failed", "error": str(exc), "updated_at": UTC_NOW()})
        rows.append({"source": str(source_dir), "target": str(target_dir), "status": "synced", "files": copied, "updated_at": UTC_NOW()})
    manifest = pd.DataFrame(rows)
    manifest_path = sync_root / "drive_sync_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    return manifest


def run_research_extension_layer(namespace: dict[str, Any], refresh: bool = False, max_api_tickers: int = 75) -> dict[str, Any]:
    sync_config_aliases(namespace)
    paths = resolve_platform_paths(namespace)
    runtime = pd.DataFrame([detect_runtime_environment()])
    registry = provider_registry_frame()
    api_keys = api_key_status_frame()
    universe, universe_meta = build_universe_master(namespace, refresh=refresh)
    master = _first(namespace, "MASTER_REQUEST", "MASTERREQUEST", default={}) or {}
    start = master.get("start_date") or _first(namespace, "EXPERIMENT", default={}).get("start_date")
    end = master.get("end_date") or _first(namespace, "EXPERIMENT", default={}).get("end_date")
    refresh_enabled = bool(refresh or _first(namespace, "FUNDAMENTALSCONFIG", default={}).get("auto_refresh", False))
    tickers = universe["ticker"].dropna().astype(str).head(max_api_tickers).tolist() if not universe.empty else []
    prices, price_meta = (pd.DataFrame(), pd.DataFrame())
    fundamentals, fundamental_meta = (pd.DataFrame(), pd.DataFrame())
    if refresh_enabled and tickers:
        prices, price_meta = fetch_yfinance_prices(namespace, tickers, start, end, max_tickers=max_api_tickers)
        fundamentals, fundamental_meta = fetch_yfinance_fundamentals_light(namespace, tickers, max_tickers=max_api_tickers)
    ml_outputs = run_ml_research_layer(namespace)
    valuation_outputs = run_extended_valuation_models(namespace)
    refresh_plan = build_refresh_plan(namespace)
    provenance = pd.concat([universe_meta, price_meta, fundamental_meta], ignore_index=True) if any(not x.empty for x in [universe_meta, price_meta, fundamental_meta]) else pd.DataFrame()
    outputs = {
        "runtime_environment": runtime,
        "provider_registry": registry,
        "api_key_status": api_keys,
        "universe_master": universe,
        "universe_provenance": universe_meta,
        "refreshed_prices": prices,
        "refreshed_fundamentals_light": fundamentals,
        "refresh_provenance": provenance,
        "refresh_plan": refresh_plan,
        **valuation_outputs,
        **ml_outputs,
    }
    for name, value in outputs.items():
        namespace[name] = value
        if isinstance(value, pd.DataFrame):
            out = paths["TABLESDIR"] / f"{name}.csv"
            try:
                value.to_csv(out, index=False)
            except Exception:
                pass
    manifest = {
        "updated_at": UTC_NOW(),
        "runtime": runtime.to_dict("records"),
        "tables": {k: int(len(v)) for k, v in outputs.items() if isinstance(v, pd.DataFrame)},
        "refresh_enabled": refresh_enabled,
        "max_api_tickers": max_api_tickers,
    }
    (paths["MANIFESTDIR"] / "research_extension_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    if bool((_first(namespace, "GOVERNANCECONFIG", default={}) or {}).get("sync_google_drive_database", True)):
        drive_sync_manifest = sync_outputs_to_google_drive_database(namespace, include_cache=bool((_first(namespace, "GOVERNANCECONFIG", default={}) or {}).get("sync_cache_to_drive", True)))
        outputs["drive_sync_manifest"] = drive_sync_manifest
        namespace["drive_sync_manifest"] = drive_sync_manifest
        try:
            drive_sync_manifest.to_csv(paths["TABLESDIR"] / "drive_sync_manifest.csv", index=False)
        except Exception:
            pass
    return outputs
