"""Data-centric exploration helpers for the Streamlit Data Platform.

These helpers intentionally stay Streamlit-free. They provide a thin,
best-effort read layer over existing manifests/artifacts so the Data Platform
can answer a human question quickly: "what do we have for this ticker?"
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .data_health import get_data_status_for_tickers
from .data_platform import get_ohlcv_daily_search_roots, resolve_data_platform_roots


FACTOR_PANEL_REL = Path("ml_training_lab") / "tables" / "FactorUniversePanel.csv"
ML_SIGNALS_RELS = [
    Path("ml_stock_lab") / "tables" / "MLStockLab_trained_model_signals.csv",
    Path("ml_training_lab") / "tables" / "MLTraining_predictions_wide.csv",
    Path("ml_stock_lab") / "tables" / "MLStockLab_signals.csv",
]
SMART_MONEY_RELS = [
    Path("smart_money") / "tables" / "SmartMoney_smart_money_scores.csv",
    Path("smart_money") / "tables" / "SmartMoney_event_feed.csv",
    Path("smart_money") / "tables" / "SmartMoney_entity_master.csv",
]


def _read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    if not path.exists() or not path.is_file() or path.stat().st_size <= 1:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, **kwargs)
    except Exception:
        return pd.DataFrame()


def _normalize_ticker(value: Any) -> str:
    return str(value or "").strip().upper()


def _ticker_aliases(ticker: str) -> set[str]:
    clean = _normalize_ticker(ticker)
    if not clean:
        return set()
    aliases = {
        clean,
        clean.replace(".", "_").replace("-", "_").replace("/", "_"),
        clean.replace("_", "."),
        clean.replace("-", "."),
    }
    return {item for item in aliases if item}


def _ticker_mask(frame: pd.DataFrame, ticker: str) -> pd.Series:
    aliases = _ticker_aliases(ticker)
    if frame.empty or not aliases:
        return pd.Series(False, index=frame.index)
    mask = pd.Series(False, index=frame.index)
    for col in ["ticker", "provider_symbol", "resolved_provider_symbol", "symbol", "issuer_ticker"]:
        if col in frame.columns:
            mask = mask | frame[col].fillna("").astype(str).str.upper().isin(aliases)
    return mask


def _first_existing(paths: list[Path]) -> Path | None:
    return next((path for path in paths if path.exists() and path.stat().st_size > 1), None)


def list_available_tickers(
    financial_db_root: str | Path | None = None,
    output_root: str | Path | None = None,
    limit: int = 20000,
) -> pd.DataFrame:
    """Return tickers discoverable from coverage/model manifests.

    The OHLCV manifest is the primary source because it is lightweight and
    contains the broadest ticker universe. Model and factor artifacts are used
    as enrichment/fallbacks.
    """
    roots = resolve_data_platform_roots(financial_db_root=financial_db_root, repo_output_root=output_root)
    rows: list[dict[str, Any]] = []
    manifest = _read_csv(roots.repo_output / "tables" / "OHLCV_daily_manifest.csv")
    if not manifest.empty:
        for _, row in manifest.head(int(limit)).iterrows():
            ticker = _normalize_ticker(row.get("ticker") or row.get("provider_symbol"))
            if ticker:
                rows.append(
                    {
                        "ticker": ticker,
                        "source": "ohlcv_manifest",
                        "universe": row.get("universe", row.get("exchange", "")),
                        "coverage_status": row.get("coverage_status", row.get("status", "")),
                        "last_price_date": row.get("last_price_date", ""),
                    }
                )
    for rel in ML_SIGNALS_RELS:
        frame = _read_csv(roots.repo_output / rel, usecols=lambda col: col in {"ticker", "date", "score_composite", "ml_score"})
        if frame.empty or "ticker" not in frame.columns:
            continue
        for ticker in frame["ticker"].dropna().astype(str).str.upper().unique()[: int(limit)]:
            rows.append({"ticker": ticker, "source": rel.name, "universe": "", "coverage_status": "MODEL_SIGNAL", "last_price_date": ""})
    if not rows:
        return pd.DataFrame(columns=["ticker", "source", "universe", "coverage_status", "last_price_date"])
    out = pd.DataFrame(rows)
    out = out.drop_duplicates("ticker", keep="first").sort_values("ticker").head(int(limit)).reset_index(drop=True)
    return out


def find_ohlcv_parquet(
    ticker: str,
    financial_db_root: str | Path | None = None,
    output_root: str | Path | None = None,
) -> Path | None:
    """Find the local parquet file for a ticker if it exists."""
    roots = resolve_data_platform_roots(financial_db_root=financial_db_root, repo_output_root=output_root)
    aliases = _ticker_aliases(ticker)
    if not aliases:
        return None
    for daily_root in get_ohlcv_daily_search_roots(roots.financial_db, roots.repo_output):
        if not daily_root.exists():
            continue
        for alias in aliases:
            try:
                found = next(daily_root.glob(f"*/{alias}.parquet"), None)
            except Exception:
                found = None
            if found is not None:
                return found
    return None


def load_ticker_ohlcv(
    ticker: str,
    financial_db_root: str | Path | None = None,
    output_root: str | Path | None = None,
    tail_rows: int = 252,
) -> pd.DataFrame:
    path = find_ohlcv_parquet(ticker, financial_db_root, output_root)
    if path is None:
        return pd.DataFrame()
    try:
        frame = pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame.sort_values("date")
    frame["ticker"] = _normalize_ticker(ticker)
    frame["source_path"] = str(path)
    return frame.tail(int(tail_rows)).reset_index(drop=True)


def load_factor_rows_for_ticker(
    ticker: str,
    output_root: str | Path | None = None,
    max_rows: int = 500,
) -> pd.DataFrame:
    roots = resolve_data_platform_roots(repo_output_root=output_root)
    path = roots.repo_output / FACTOR_PANEL_REL
    if not path.exists() or path.stat().st_size <= 1:
        return pd.DataFrame()
    aliases = _ticker_aliases(ticker)
    if not aliases:
        return pd.DataFrame()
    preferred = [
        "date",
        "ticker",
        "price",
        "ret21d",
        "ret63d",
        "ret252d",
        "vol63d",
        "vol252d",
        "value_score",
        "quality_score",
        "momentum_score",
        "risk_score",
        "size_score",
        "growth_score",
        "factor_composite_score",
        "forward_return_21d",
        "forward_return_63d",
        "forward_return_252d",
    ]
    try:
        header = pd.read_csv(path, nrows=0).columns.tolist()
    except Exception:
        return pd.DataFrame()
    usecols = [col for col in preferred if col in header]
    rows: list[pd.DataFrame] = []
    try:
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=100_000):
            if "ticker" not in chunk.columns:
                continue
            match = chunk[chunk["ticker"].fillna("").astype(str).str.upper().isin(aliases)]
            if not match.empty:
                rows.append(match)
    except Exception:
        return pd.DataFrame()
    if not rows:
        return pd.DataFrame(columns=usecols)
    out = pd.concat(rows, ignore_index=True, sort=False)
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out = out.sort_values("date")
    return out.tail(int(max_rows)).reset_index(drop=True)


def load_ml_signal_rows_for_ticker(
    ticker: str,
    output_root: str | Path | None = None,
    max_rows: int = 500,
) -> pd.DataFrame:
    roots = resolve_data_platform_roots(repo_output_root=output_root)
    frames: list[pd.DataFrame] = []
    for rel in ML_SIGNALS_RELS:
        frame = _read_csv(roots.repo_output / rel)
        if frame.empty:
            continue
        match = frame[_ticker_mask(frame, ticker)].copy()
        if match.empty:
            continue
        match["artifact"] = str(rel)
        frames.append(match)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out = out.sort_values("date")
    return out.tail(int(max_rows)).reset_index(drop=True)


def load_fundamental_rows_for_ticker(
    ticker: str,
    financial_db_root: str | Path | None = None,
    max_files: int = 4,
) -> pd.DataFrame:
    roots = resolve_data_platform_roots(financial_db_root=financial_db_root)
    aliases = _ticker_aliases(ticker)
    equities = roots.financial_db / "Equities"
    if not equities.exists() or not aliases:
        return pd.DataFrame()
    paths: list[Path] = []
    for alias in aliases:
        try:
            paths.extend(equities.glob(f"*/*/fundamentals/{alias}_*.parquet"))
        except Exception:
            continue
    rows: list[pd.DataFrame] = []
    for path in sorted(set(paths))[: int(max_files)]:
        try:
            frame = pd.read_parquet(path)
        except Exception:
            continue
        frame = frame.tail(5).copy()
        frame["ticker"] = _normalize_ticker(ticker)
        frame["statement_artifact"] = path.name
        frame["source_path"] = str(path)
        rows.append(frame)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def load_smart_money_rows_for_ticker(
    ticker: str,
    output_root: str | Path | None = None,
    max_rows: int = 500,
) -> pd.DataFrame:
    roots = resolve_data_platform_roots(repo_output_root=output_root)
    frames: list[pd.DataFrame] = []
    for rel in SMART_MONEY_RELS:
        frame = _read_csv(roots.repo_output / rel)
        if frame.empty:
            continue
        match = frame[_ticker_mask(frame, ticker)].copy()
        if match.empty:
            continue
        match["artifact"] = str(rel)
        frames.append(match)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False).head(int(max_rows)).reset_index(drop=True)


def get_single_ticker_snapshot(
    ticker: str,
    financial_db_root: str | Path | None = None,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    """Return availability and lightweight data samples for one ticker."""
    roots = resolve_data_platform_roots(financial_db_root=financial_db_root, repo_output_root=output_root)
    ticker = _normalize_ticker(ticker)
    status = get_data_status_for_tickers([ticker], roots.financial_db, roots.repo_output).get(ticker)
    ohlcv = load_ticker_ohlcv(ticker, roots.financial_db, roots.repo_output, tail_rows=260)
    fundamentals = load_fundamental_rows_for_ticker(ticker, roots.financial_db)
    factors = load_factor_rows_for_ticker(ticker, roots.repo_output)
    ml_signals = load_ml_signal_rows_for_ticker(ticker, roots.repo_output)
    smart_money = load_smart_money_rows_for_ticker(ticker, roots.repo_output)
    parquet_path = find_ohlcv_parquet(ticker, roots.financial_db, roots.repo_output)
    latest_factor = factors.tail(1).copy()
    latest_ml = ml_signals.tail(1).copy()

    availability = [
        {
            "domain": "OHLCV",
            "status": status.prices_status if status else ("OK" if not ohlcv.empty else "MISSING"),
            "rows": len(ohlcv),
            "detail": (status.price_coverage_status if status else "") or ("local parquet found" if parquet_path else "no local parquet"),
            "path": str(parquet_path or ""),
        },
        {
            "domain": "Fundamentals",
            "status": status.fundamentals_status if status else ("OK" if not fundamentals.empty else "MISSING"),
            "rows": len(fundamentals),
            "detail": "statement parquet rows" if not fundamentals.empty else "no statement artifact found",
            "path": str(fundamentals.get("source_path", pd.Series([""])).iloc[-1]) if not fundamentals.empty and "source_path" in fundamentals.columns else "",
        },
        {
            "domain": "Factor Panel",
            "status": "OK" if not factors.empty else "MISSING",
            "rows": len(factors),
            "detail": "value/quality/momentum/risk scores available" if not factors.empty else "run factor_universe_panel to populate",
            "path": str(roots.repo_output / FACTOR_PANEL_REL),
        },
        {
            "domain": "ML Signals",
            "status": "OK" if not ml_signals.empty else "MISSING",
            "rows": len(ml_signals),
            "detail": "trained model predictions available" if not ml_signals.empty else "run ML training or ML Stock Lab",
            "path": str(_first_existing([roots.repo_output / rel for rel in ML_SIGNALS_RELS]) or ""),
        },
        {
            "domain": "Valuation",
            "status": "OK" if not latest_ml.empty and any(col in latest_ml.columns for col in ["fair_value_hat", "valuation_signal_score", "mispricing_rel"]) else "PLANNED",
            "rows": int(not latest_ml.empty),
            "detail": "valuation proxy in ML/Screener artifacts" if not latest_ml.empty else "dedicated valuation artifact not linked for this ticker",
            "path": "",
        },
        {
            "domain": "Smart Money",
            "status": "OK" if not smart_money.empty else "PLANNED",
            "rows": len(smart_money),
            "detail": "issuer/events found" if not smart_money.empty else "module present; no ticker-level event loaded",
            "path": str(_first_existing([roots.repo_output / rel for rel in SMART_MONEY_RELS]) or ""),
        },
        {
            "domain": "FX / Macro",
            "status": "NOT_TICKER_SPECIFIC",
            "rows": 0,
            "detail": "macro/FX overlays are platform context, not issuer rows yet",
            "path": "",
        },
    ]
    return {
        "ticker": ticker,
        "status": status.to_dict() if status else {},
        "availability": pd.DataFrame(availability),
        "ohlcv": ohlcv,
        "fundamentals": fundamentals,
        "factors": factors,
        "ml_signals": ml_signals,
        "smart_money": smart_money,
    }


def load_data_explorer_preview(
    dataset: str,
    financial_db_root: str | Path | None = None,
    output_root: str | Path | None = None,
    universe: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    max_rows: int = 5000,
) -> pd.DataFrame:
    """Load a bounded preview for the Data Explorer tab."""
    roots = resolve_data_platform_roots(financial_db_root=financial_db_root, repo_output_root=output_root)
    dataset_key = str(dataset or "").lower()
    if dataset_key == "ohlcv_manifest":
        frame = _read_csv(roots.repo_output / "tables" / "OHLCV_daily_manifest.csv")
    elif dataset_key == "equity_fundamentals_manifest":
        frame = _read_csv(roots.repo_output / "data_completion" / "equity_fundamentals_2000_2026.csv")
    elif dataset_key == "ml_signals":
        frame = _read_csv(_first_existing([roots.repo_output / rel for rel in ML_SIGNALS_RELS]) or Path(""))
    elif dataset_key == "smart_money":
        frame = _read_csv(_first_existing([roots.repo_output / rel for rel in SMART_MONEY_RELS]) or Path(""))
    else:
        path = roots.repo_output / FACTOR_PANEL_REL
        if not path.exists() or path.stat().st_size <= 1:
            frame = pd.DataFrame()
        else:
            chunks: list[pd.DataFrame] = []
            try:
                for chunk in pd.read_csv(path, chunksize=100_000):
                    chunks.append(chunk.head(max(1, int(max_rows) // 20)))
                    if sum(len(part) for part in chunks) >= int(max_rows):
                        break
            except Exception:
                chunks = []
            frame = pd.concat(chunks, ignore_index=True, sort=False) if chunks else pd.DataFrame()
    if frame.empty:
        return frame
    if universe and universe != "All":
        for col in ["universe", "exchange", "market", "primary_source"]:
            if col in frame.columns:
                frame = frame[frame[col].fillna("").astype(str).str.lower().eq(str(universe).lower())].copy()
                break
    if "date" in frame.columns and (start_date or end_date):
        dates = pd.to_datetime(frame["date"], errors="coerce")
        if start_date:
            frame = frame[dates.ge(pd.to_datetime(start_date, errors="coerce")).fillna(False)].copy()
            dates = pd.to_datetime(frame["date"], errors="coerce")
        if end_date:
            frame = frame[dates.le(pd.to_datetime(end_date, errors="coerce")).fillna(False)].copy()
    return frame.head(int(max_rows)).reset_index(drop=True)
