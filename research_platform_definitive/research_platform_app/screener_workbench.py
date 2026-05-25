"""Reusable screening workbench helpers.

Assumptions:
- The canonical source is the artifact layer already exposed by support.py:
  company screener/valuation tables, portfolio selection tables, ML Stock Lab
  tables and Smart Money / Gov Data tables.
- Database Finanziario or local database mirrors may expose additional metadata
  such as `us_equities_meta_data.csv`; when they are missing, the builder keeps
  the artifact-only universe instead of fabricating fundamentals.
- SHAP/local feature attribution may not exist in the current bundle. In that
  case, explanation uses a transparent proxy decomposition over available
  feature, score, valuation, ML and Smart Money columns.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


NUMERIC_ALIASES: dict[str, list[str]] = {
    "pe": ["pe", "p_e", "trailing_pe", "trailingPE", "price_earnings", "price_to_earnings"],
    "ev_ebit": ["ev_ebit", "ev_to_ebit", "enterprise_to_ebit"],
    "ev_ebitda": ["ev_ebitda", "ev_to_ebitda", "enterpriseToEbitda", "enterprise_to_ebitda"],
    "pb": ["pb", "p_b", "price_to_book", "pricebook", "priceToBook"],
    "dividend_yield": ["dividend_yield", "dividendyield", "yield"],
    "revenue_cagr": ["revenue_cagr", "sales_cagr", "growth_cagr", "cagr"],
    "roe": ["roe", "return_on_equity"],
    "roic": ["roic", "return_on_invested_capital"],
    "gross_margin": ["gross_margin", "grossmargin"],
    "operating_margin": ["operating_margin", "operatingmargin", "ebit_margin"],
    "debt_to_equity": ["debt_to_equity", "debt_equity", "leverage"],
    "volatility": ["volatility", "annual_volatility", "risk"],
    "marketcap": ["marketcap", "market_cap", "marketCapitalization"],
}


PRESETS: dict[str, dict[str, Any]] = {
    "Custom": {},
    "Quality at reasonable price": {
        "min_quality": 50.0,
        "max_pe": 30.0,
        "max_pb": 8.0,
        "min_screener_score": 0.0,
        "sort_by": "quality_proxy",
    },
    "Dividend compounders": {
        "min_dividend_yield": 1.5,
        "min_quality": 45.0,
        "max_debt_to_equity": 250.0,
        "sort_by": "dividend_yield",
    },
    "ML high conviction": {
        "min_ml_score": 60.0,
        "ml_quintiles": ["Q5"],
        "sort_by": "ml_score",
    },
    "Value + improving momentum": {
        "max_pe": 25.0,
        "max_pb": 4.0,
        "min_ml_score": 50.0,
        "sort_by": "composite_conviction_score",
    },
    "Smart Money confirmed": {
        "min_smart_money_score": 50.0,
        "require_smart_events": True,
        "sort_by": "smart_money_score",
    },
}


@dataclass(frozen=True)
class ScreenerArtifactPaths:
    root: Path

    @property
    def screeners_dir(self) -> Path:
        return self.root / "screeners"


def slugify(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip().lower())
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "untitled_screener"


def normalize_ticker(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip().upper()


def _first_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    lookup = {str(col).lower(): col for col in df.columns}
    for name in candidates:
        if name in df.columns:
            return name
        found = lookup.get(str(name).lower())
        if found is not None:
            return str(found)
    return None


def _series_from_alias(df: pd.DataFrame, aliases: Iterable[str], default: object = np.nan) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=object)
    col = _first_column(df, aliases)
    if col is None:
        return pd.Series([default] * len(df), index=df.index)
    return df[col]


def _safe_numeric(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 1:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def load_local_equity_metadata(financial_db_root: Path | None = None) -> pd.DataFrame:
    candidates = []
    if financial_db_root is not None:
        financial_db_root = Path(financial_db_root).expanduser()
        candidates.extend(
            [
                financial_db_root / "data" / "us_equities_meta_data.csv",
                financial_db_root / "us_equities_meta_data.csv",
            ]
        )
    candidates.append(PROJECT_ROOT / "archive" / "local_databases_not_on_drive" / "data" / "us_equities_meta_data.csv")
    for candidate in candidates:
        frame = _safe_read_csv(candidate)
        if not frame.empty and "ticker" in frame.columns:
            frame = frame.copy()
            frame["ticker"] = frame["ticker"].map(normalize_ticker)
            return frame.drop_duplicates("ticker")
    return pd.DataFrame()


def _prepare_base(company: dict[str, pd.DataFrame], metadata: pd.DataFrame) -> pd.DataFrame:
    sources = [
        company.get("screener_results", pd.DataFrame()),
        company.get("extended_valuation", pd.DataFrame()),
        company.get("valuation_gap", pd.DataFrame()),
        metadata,
    ]
    frames = []
    for source in sources:
        if source is None or source.empty or "ticker" not in source.columns:
            continue
        frame = source.copy()
        frame["ticker"] = frame["ticker"].map(normalize_ticker)
        frames.append(frame[frame["ticker"].ne("")])
    if not frames:
        return pd.DataFrame(columns=["ticker"])
    base = pd.concat(frames, ignore_index=True, sort=False)
    base = base.sort_values([col for col in ["screener_rank", "ticker"] if col in base.columns])
    base = base.drop_duplicates("ticker", keep="first").reset_index(drop=True)
    return base


def _merge_by_ticker(left: pd.DataFrame, right: pd.DataFrame, suffix: str) -> pd.DataFrame:
    if right.empty or "ticker" not in right.columns:
        return left
    clean = right.copy()
    clean["ticker"] = clean["ticker"].map(normalize_ticker)
    clean = clean[clean["ticker"].ne("")].drop_duplicates("ticker", keep="first")
    return left.merge(clean, on="ticker", how="left", suffixes=("", suffix))


def _percentile(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == 0:
        return pd.Series(np.nan, index=series.index)
    ranked = numeric.rank(pct=True, ascending=not higher_is_better) * 100
    return ranked.round(1)


def _quintile(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()
    out = pd.Series("n/a", index=series.index)
    if valid.nunique() < 2:
        return out
    rank = numeric.rank(method="first", ascending=not higher_is_better)
    try:
        bins = pd.qcut(rank.dropna(), 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
        out.loc[bins.index] = bins.astype(str)
    except Exception:
        out.loc[valid.index] = "Q3"
    return out


def build_screening_frame(
    company: dict[str, pd.DataFrame],
    portfolio: dict[str, pd.DataFrame],
    smart_money: dict[str, pd.DataFrame],
    ml_lab: dict[str, pd.DataFrame],
    financial_db_root: Path | None = None,
) -> pd.DataFrame:
    metadata = load_local_equity_metadata(financial_db_root)
    frame = _prepare_base(company, metadata)
    if frame.empty:
        return frame

    frame = _merge_by_ticker(frame, company.get("valuation_gap", pd.DataFrame()), "_valuation")
    frame = _merge_by_ticker(frame, company.get("extended_valuation", pd.DataFrame()), "_extended")
    frame = _merge_by_ticker(frame, portfolio.get("selection_results", pd.DataFrame()), "_portfolio")
    frame = _merge_by_ticker(frame, portfolio.get("allocation", pd.DataFrame()), "_allocation")
    frame = _merge_by_ticker(frame, ml_lab.get("signals", pd.DataFrame()), "_ml")
    frame = _merge_by_ticker(frame, smart_money.get("scores", pd.DataFrame()), "_smart")

    frame["ticker"] = frame["ticker"].map(normalize_ticker)
    frame["company_name"] = _series_from_alias(frame, ["company_name", "name", "issuer_name", "shortName"], "")
    frame["sector"] = _series_from_alias(frame, ["sector", "sector_smart", "sector_portfolio"], "")
    frame["industry"] = _series_from_alias(frame, ["industry"], "")
    frame["country"] = _series_from_alias(frame, ["country", "country_smart", "country_portfolio"], "")
    frame["universe"] = _series_from_alias(frame, ["index_membership", "universe_source", "source"], "Current Export")

    for canonical, aliases in NUMERIC_ALIASES.items():
        frame[canonical] = pd.to_numeric(_series_from_alias(frame, aliases), errors="coerce")

    frame["screener_score"] = pd.to_numeric(_series_from_alias(frame, ["screener_score", "blended_score", "blendedscore"]), errors="coerce")
    frame["quality_proxy"] = pd.to_numeric(
        _series_from_alias(frame, ["quality_score", "qualityscore", "composite_score", "selection_score"]),
        errors="coerce",
    )
    frame["ml_score"] = pd.to_numeric(
        _series_from_alias(
            frame,
            [
                "ml_score",
                "score_composite",
                "expected_return_score",
                "score_rf",
                "score_ols",
                "score_gbrt",
                "score_lasso",
                "score_ensemble",
                "score",
                "score_ml",
            ],
        ),
        errors="coerce",
    )
    expected_rank = pd.to_numeric(_series_from_alias(frame, ["expected_return_rank", "expected_return_rank_rf", "expected_return_rank_ols"]), errors="coerce")
    if frame["ml_score"].notna().sum() == 0 and expected_rank.notna().sum() > 0:
        frame["ml_score"] = expected_rank * 100
    frame["ml_zscore"] = pd.to_numeric(_series_from_alias(frame, ["zscore", "zscore_ml"]), errors="coerce")
    frame["ml_percentile"] = pd.to_numeric(_series_from_alias(frame, ["percentile_rank", "prediction_rank"]), errors="coerce")
    frame["ml_signal"] = _series_from_alias(frame, ["signal", "signal_ml", "quality_flag"], "")
    frame["ml_quintile"] = _quintile(frame["ml_score"], higher_is_better=True)
    frame["smart_money_score"] = pd.to_numeric(
        _series_from_alias(frame, ["composite_institutional_interest_score", "smart_money_score"]),
        errors="coerce",
    )
    frame["smart_money_coverage"] = _series_from_alias(frame, ["coverage_note", "coverage_note_smart"], "missing")
    frame["selection_score"] = pd.to_numeric(_series_from_alias(frame, ["selection_score", "composite_score"]), errors="coerce")
    frame["portfolio_weight"] = pd.to_numeric(_series_from_alias(frame, ["weight", "engine_weight"]), errors="coerce")
    frame["fair_value_hat"] = pd.to_numeric(_series_from_alias(frame, ["fair_value_hat", "fair_value", "target_price"]), errors="coerce")
    frame["mispricing_rel"] = pd.to_numeric(_series_from_alias(frame, ["mispricing_rel", "upside", "valuation_gap"]), errors="coerce")
    frame["valuation_signal_score"] = _percentile(frame["mispricing_rel"], higher_is_better=True)
    frame["updated_fundamentals"] = _series_from_alias(frame, ["updated_at", "updated_at_valuation"], "")
    frame["updated_ml"] = _series_from_alias(frame, ["updated_at_ml", "date"], "")
    frame["updated_smart_money"] = _series_from_alias(frame, ["updated_at_smart", "event_date"], "")

    events = smart_money.get("events", pd.DataFrame())
    if events is not None and not events.empty and "ticker" in events.columns:
        event_counts = events.assign(ticker=events["ticker"].map(normalize_ticker)).groupby("ticker").size()
        frame["smart_recent_events"] = frame["ticker"].map(event_counts).fillna(0).astype(int)
    else:
        frame["smart_recent_events"] = 0

    available_scores = [
        frame["screener_score"].fillna(0),
        frame["quality_proxy"].fillna(0),
        frame["selection_score"].fillna(0),
        frame["ml_score"].fillna(0),
        frame["valuation_signal_score"].fillna(0),
        frame["smart_money_score"].fillna(0),
    ]
    frame["composite_conviction_score"] = np.nanmean(np.vstack([s.to_numpy(dtype=float) for s in available_scores]), axis=0).round(2)
    frame["screening_percentile"] = _percentile(frame["composite_conviction_score"], higher_is_better=True)
    frame["screening_quintile"] = _quintile(frame["composite_conviction_score"], higher_is_better=True)
    return frame.sort_values("composite_conviction_score", ascending=False).reset_index(drop=True)


def numeric_range(frame: pd.DataFrame, column: str, fallback: tuple[float, float]) -> tuple[float, float]:
    if frame.empty or column not in frame.columns:
        return fallback
    values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return fallback
    lo = float(np.nanpercentile(values, 1))
    hi = float(np.nanpercentile(values, 99))
    if lo == hi:
        hi = lo + 1.0
    return round(lo, 2), round(hi, 2)


def apply_screening_filters(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    view = frame.copy()
    if view.empty:
        return view
    for col, values in {
        "universe": config.get("universes"),
        "country": config.get("countries"),
        "sector": config.get("sectors"),
        "industry": config.get("industries"),
        "ml_quintile": config.get("ml_quintiles"),
    }.items():
        if values and col in view.columns and "All" not in values:
            view = view[view[col].astype(str).isin([str(v) for v in values])]

    custom_tickers = [normalize_ticker(ticker) for ticker in config.get("custom_tickers", []) if normalize_ticker(ticker)]
    if custom_tickers and "ticker" in view.columns:
        view = view[view["ticker"].map(normalize_ticker).isin(custom_tickers)]

    numeric_rules = {
        "pe": (None, config.get("max_pe")),
        "ev_ebit": (None, config.get("max_ev_ebit")),
        "ev_ebitda": (None, config.get("max_ev_ebitda")),
        "pb": (None, config.get("max_pb")),
        "dividend_yield": (config.get("min_dividend_yield"), None),
        "roe": (config.get("min_roe"), None),
        "roic": (config.get("min_roic"), None),
        "debt_to_equity": (None, config.get("max_debt_to_equity")),
        "volatility": (None, config.get("max_volatility")),
        "screener_score": (config.get("min_screener_score"), None),
        "quality_proxy": (config.get("min_quality"), None),
        "ml_score": (config.get("min_ml_score"), None),
        "valuation_signal_score": (config.get("min_valuation_signal"), None),
        "smart_money_score": (config.get("min_smart_money_score"), None),
        "composite_conviction_score": (config.get("min_conviction"), None),
    }
    for col, (min_value, max_value) in numeric_rules.items():
        if col not in view.columns:
            continue
        values = pd.to_numeric(view[col], errors="coerce")
        if min_value is not None:
            view = view[values.fillna(-np.inf) >= float(min_value)]
            values = pd.to_numeric(view[col], errors="coerce")
        if max_value is not None:
            view = view[values.fillna(np.inf) <= float(max_value)]

    if config.get("require_smart_events"):
        view = view[pd.to_numeric(view.get("smart_recent_events", 0), errors="coerce").fillna(0) > 0]

    query = str(config.get("query") or "").strip().lower()
    if query:
        haystack = (
            view.get("ticker", pd.Series("", index=view.index)).astype(str)
            + " "
            + view.get("company_name", pd.Series("", index=view.index)).astype(str)
            + " "
            + view.get("sector", pd.Series("", index=view.index)).astype(str)
            + " "
            + view.get("industry", pd.Series("", index=view.index)).astype(str)
        ).str.lower()
        view = view[haystack.str.contains(query, regex=False)]

    sort_by = config.get("sort_by") or "composite_conviction_score"
    if sort_by in view.columns:
        view = view.sort_values(sort_by, ascending=bool(config.get("sort_ascending", False)))
    return view.reset_index(drop=True)


def get_screeners_dir(workspace_root: Path) -> Path:
    root = Path(workspace_root).expanduser() / "screeners"
    root.mkdir(parents=True, exist_ok=True)
    return root


def save_screener_config(workspace_root: Path, name: str, description: str, config: dict[str, Any], result_count: int) -> Path:
    screeners_dir = get_screeners_dir(workspace_root)
    path = screeners_dir / f"{slugify(name)}.json"
    payload = {
        "name": name.strip() or "Untitled screener",
        "description": description.strip(),
        "created_or_updated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "result_count_at_save": int(result_count),
        "config": config,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path


def load_screener_config(path: Path) -> dict[str, Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def list_saved_screeners(workspace_root: Path, current_frame: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    for path in sorted(get_screeners_dir(workspace_root).glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        payload = load_screener_config(path)
        config = payload.get("config", {})
        current_count = None
        if current_frame is not None and not current_frame.empty:
            current_count = len(apply_screening_filters(current_frame, config))
        rows.append(
            {
                "name": payload.get("name", path.stem),
                "description": payload.get("description", ""),
                "created_or_updated_at": payload.get("created_or_updated_at", ""),
                "saved_count": payload.get("result_count_at_save", ""),
                "current_count": current_count,
                "path": str(path),
            }
        )
    return pd.DataFrame(rows)


def delete_screener(path: str | Path) -> bool:
    target = Path(path)
    if target.exists() and target.suffix == ".json":
        target.unlink()
        return True
    return False


def _feature_drivers(row: pd.Series) -> pd.DataFrame:
    candidates = [
        ("ML expected-return score", row.get("ml_score"), "Model signal"),
        ("Valuation signal score", row.get("valuation_signal_score"), "Valuation signal"),
        ("Mispricing", row.get("mispricing_rel"), "Fair value gap"),
        ("Screening score", row.get("screener_score"), "Core screener"),
        ("Quality proxy", row.get("quality_proxy"), "Fundamental quality"),
        ("Smart Money score", row.get("smart_money_score"), "Official-source confirmation"),
        ("Portfolio selection", row.get("selection_score"), "Portfolio-aware ranking"),
        ("ROE", row.get("roe"), "Profitability"),
        ("ROIC", row.get("roic"), "Capital efficiency"),
        ("Dividend yield", row.get("dividend_yield"), "Income profile"),
    ]
    rows = []
    for label, value, family in candidates:
        numeric = _safe_numeric(value)
        if numeric is None:
            continue
        rows.append({"driver": label, "value": numeric, "family": family, "abs_value": abs(numeric)})
    out = pd.DataFrame(rows)
    return out.sort_values("abs_value", ascending=False).head(8) if not out.empty else out


def _fmt(value: Any, suffix: str = "") -> str:
    numeric = _safe_numeric(value)
    if numeric is None:
        return "n/a"
    return f"{numeric:.2f}{suffix}"


def _direction(value: Any, threshold: float = 0.0, higher_is_better: bool = True) -> str:
    numeric = _safe_numeric(value)
    if numeric is None:
        return "neutral"
    positive = numeric >= threshold if higher_is_better else numeric <= threshold
    return "up" if positive else "down"


def format_ml_explanation(
    ticker: str,
    row: pd.Series | None,
    feature_contribs: pd.DataFrame | None = None,
    universe_stats: dict[str, Any] | None = None,
) -> str:
    if row is None:
        return f"**Upshot:** {ticker} has no ML artifact row in the current universe."
    drivers = feature_contribs if feature_contribs is not None and not feature_contribs.empty else _feature_drivers(row)
    top = drivers.drop(columns=["abs_value"], errors="ignore").head(5) if not drivers.empty else pd.DataFrame()
    driver_lines = []
    for _, item in top.iterrows():
        direction = "↑" if _direction(item.get("value")) == "up" else "↓"
        driver_lines.append(f"- {item.get('driver')}: `{_fmt(item.get('value'))}` ({item.get('family', 'driver')}) {direction}")
    if not driver_lines:
        driver_lines = ["- Attribution table unavailable; using score/quintile context only."]
    percentile = row.get("ml_percentile")
    if pd.isna(percentile):
        percentile = row.get("screening_percentile", "n/a")
    return (
        f"**Upshot:** {ticker} sits in `{row.get('ml_quintile', 'n/a')}` with ML score `{_fmt(row.get('ml_score'))}`. "
        "The signal is a research ranking input, not a trade instruction.\n\n"
        "**Driver principali:**\n"
        + "\n".join(driver_lines)
        + "\n\n"
        f"**Posizionamento nell'universo:** percentile `{percentile}`, screening quintile `{row.get('screening_quintile', 'n/a')}`, "
        f"Smart Money `{_fmt(row.get('smart_money_score'))}`."
    )


def format_smart_money_explanation(ticker: str, issuer_row: pd.Series | None, events: pd.DataFrame) -> str:
    score = issuer_row.get("smart_money_score") if issuer_row is not None else np.nan
    coverage = issuer_row.get("smart_money_coverage") if issuer_row is not None else "missing"
    event_count = len(events) if events is not None else 0
    event_sample = []
    if events is not None and not events.empty:
        for _, item in events.head(5).iterrows():
            event_sample.append(f"- {item.get('event_date', 'n/a')}: {item.get('event_label', item.get('event_source', 'event'))}")
    if not event_sample:
        event_sample = ["- No issuer-level official-source event is present in current artifacts."]
    return (
        f"**Upshot:** {ticker} has Smart Money score `{_fmt(score)}` with `{event_count}` event rows. "
        f"Coverage note: `{coverage}`.\n\n"
        "**Driver principali:**\n"
        + "\n".join(event_sample)
        + "\n\n"
        "**Posizionamento nell'universo:** use this as confirmation only when source coverage is available and recent."
    )


def format_valuation_explanation(ticker: str, valuation_artifact: pd.DataFrame | pd.Series | None) -> str:
    if valuation_artifact is None or (hasattr(valuation_artifact, "empty") and valuation_artifact.empty):
        return (
            f"**Upshot:** {ticker} has no detailed valuation driver artifact yet. "
            "Use live multiples as a pre-read and run the valuation job for DCF/WACC/terminal value detail.\n\n"
            "**Driver principali:**\n- DCF assumptions unavailable.\n- Multiples/fair-value fields unavailable or sparse.\n\n"
            "**Posizionamento nell'universo:** unavailable until valuation artifacts are refreshed."
        )
    if isinstance(valuation_artifact, pd.DataFrame):
        rows = valuation_artifact.head(5)
        lines = [f"- {row.get('driver', row.get('metric', 'driver'))}: `{_fmt(row.get('value', row.get('metric_value')) )}`" for _, row in rows.iterrows()]
    else:
        lines = [
            f"- Fair value proxy: `{_fmt(valuation_artifact.get('fair_value_hat'))}`",
            f"- Mispricing/upside: `{_fmt(valuation_artifact.get('mispricing_rel'))}`",
            f"- P/E: `{_fmt(valuation_artifact.get('pe'))}`",
            f"- P/B: `{_fmt(valuation_artifact.get('pb'))}`",
        ]
    return (
        f"**Upshot:** {ticker} valuation should be interpreted as a range of assumptions, not a point forecast.\n\n"
        "**Driver principali:**\n"
        + "\n".join(lines[:5])
        + "\n\n"
        "**Posizionamento nell'universo:** compare value, quality and risk percentiles before promotion to portfolio work."
    )


def explain_ml_signal(frame: pd.DataFrame, ticker: str) -> tuple[pd.DataFrame, str]:
    row = _row_for_ticker(frame, ticker)
    if row is None:
        return pd.DataFrame(), "No ML signal row is available for the selected ticker."
    drivers = _feature_drivers(row)
    text = format_ml_explanation(ticker, row, drivers)
    return drivers.drop(columns=["abs_value"], errors="ignore"), text


def explain_smart_money(smart_money: dict[str, pd.DataFrame], ticker: str, frame: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    clean_ticker = normalize_ticker(ticker)
    events = smart_money.get("events", pd.DataFrame())
    if events is not None and not events.empty and "ticker" in events.columns:
        event_view = events[events["ticker"].map(normalize_ticker).eq(clean_ticker)].copy()
    else:
        event_view = pd.DataFrame()
    row = _row_for_ticker(frame, clean_ticker)
    score = row.get("smart_money_score") if row is not None else np.nan
    coverage = row.get("smart_money_coverage") if row is not None else "missing"
    event_count = len(event_view)
    text = format_smart_money_explanation(clean_ticker, row, event_view)
    return event_view, text


def explain_valuation(frame: pd.DataFrame, ticker: str) -> tuple[pd.DataFrame, str]:
    row = _row_for_ticker(frame, ticker)
    if row is None:
        return pd.DataFrame(), "No valuation row is available for the selected ticker."
    drivers = [
        ("Fair value proxy", row.get("fair_value_hat")),
        ("Mispricing / upside", row.get("mispricing_rel")),
        ("P/E", row.get("pe")),
        ("EV/EBITDA", row.get("ev_ebitda")),
        ("P/B", row.get("pb")),
        ("ROE", row.get("roe")),
        ("Debt / Equity", row.get("debt_to_equity")),
    ]
    rows = [{"driver": label, "value": value} for label, value in drivers if _safe_numeric(value) is not None]
    table = pd.DataFrame(rows)
    text = format_valuation_explanation(ticker, table if not table.empty else row)
    return table, text


def screening_zero_result_suggestions(config: dict[str, Any], total_names: int) -> list[str]:
    suggestions = []
    if config.get("require_smart_events"):
        suggestions.append("Remove `Require recent Smart Money event`; official-source event coverage is often sparse.")
    if float(config.get("min_ml_score") or 0) >= 60:
        suggestions.append(f"Lower Min ML score from {config.get('min_ml_score')} to 40.")
    if float(config.get("min_smart_money_score") or 0) >= 50:
        suggestions.append(f"Lower Min Smart Money score from {config.get('min_smart_money_score')} to 20.")
    if float(config.get("min_conviction") or 0) >= 50:
        suggestions.append(f"Lower Min composite conviction from {config.get('min_conviction')} to 25.")
    if config.get("ml_quintiles"):
        suggestions.append("Clear ML quintile filters and sort by ML score instead.")
    for key, label in [("sectors", "sector"), ("industries", "industry"), ("countries", "country"), ("universes", "universe")]:
        if config.get(key):
            suggestions.append(f"Broaden the {label} selection.")
            break
    if not suggestions:
        suggestions.append(f"Reset filters to inspect the full universe ({total_names:,} names).")
    return suggestions[:3]


def _row_for_ticker(frame: pd.DataFrame, ticker: str) -> pd.Series | None:
    if frame.empty or "ticker" not in frame.columns:
        return None
    view = frame[frame["ticker"].map(normalize_ticker).eq(normalize_ticker(ticker))]
    if view.empty:
        return None
    return view.iloc[0]


def display_columns(frame: pd.DataFrame) -> list[str]:
    preferred = [
        "ticker",
        "company_name",
        "sector",
        "industry",
        "country",
        "universe",
        "pe",
        "ev_ebitda",
        "pb",
        "dividend_yield",
        "roe",
        "debt_to_equity",
        "screener_score",
        "ml_score",
        "ml_quintile",
        "valuation_signal_score",
        "smart_money_score",
        "smart_recent_events",
        "selection_score",
        "composite_conviction_score",
        "screening_percentile",
        "updated_fundamentals",
        "updated_ml",
        "updated_smart_money",
    ]
    return [col for col in preferred if col in frame.columns]
