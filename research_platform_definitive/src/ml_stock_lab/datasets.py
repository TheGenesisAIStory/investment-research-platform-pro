"""Dataset adapters for ml_stock_lab.

The module is Drive-first/cache-second/API-last by construction: it uses the
existing research platform roots and local artifacts before attempting any
external provider work. The returned panel is notebook-friendly and uses
`date, ticker` columns rather than requiring a fragile MultiIndex.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from src.research_platform_core.data_platform import discover_financial_database_root, read_dataset_drive_first
    from src.research_platform_core import get_aqr_factor_panel
except Exception:  # Colab/local package fallback
    try:
        from research_platform_core.data_platform import discover_financial_database_root, read_dataset_drive_first
        from research_platform_core import get_aqr_factor_panel
    except Exception:
        discover_financial_database_root = None
        read_dataset_drive_first = None
        get_aqr_factor_panel = None


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _candidate_artifacts(output_root: Path | None = None) -> list[Path]:
    root = output_root or _project_root()
    candidates = [
        root / "output" / "smart_money" / "tables" / "SmartMoney_smart_money_scores.csv",
        root / "company_valuation" / "output" / "tables" / "ScreenerResults.csv",
        root / "company_valuation" / "output" / "tables" / "screener_results.csv",
        root / "company_valuation" / "output" / "tables" / "extended_valuation_results.csv",
        root / "company_valuation" / "output" / "tables" / "valuation_gap_table.csv",
        root / "portfolio_analysis" / "output" / "tables" / "PortfolioSelectionResults.csv",
    ]
    return [p for p in candidates if p.exists()]


def normalize_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common notebook/export columns into an ML-ready panel."""
    if df.empty:
        return pd.DataFrame(columns=["date", "ticker", "market_value"])
    out = df.copy()
    lower = {str(c).lower(): c for c in out.columns}
    if "ticker" not in out.columns:
        for name in ["symbol", "canonical_ticker"]:
            if name in lower:
                out["ticker"] = out[lower[name]]
                break
    if "date" not in out.columns:
        for name in ["report_date", "asofdate", "as_of_date", "filing_date"]:
            if name in lower:
                out["date"] = out[lower[name]]
                break
    if "date" not in out.columns:
        out["date"] = pd.Timestamp.today().normalize().date().isoformat()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["ticker"] = out.get("ticker", pd.Series(index=out.index, dtype=object)).astype(str).str.upper()
    aliases = {
        "market_value": ["market_value", "marketcap", "marketcapest", "market_cap", "mktcap", "price"],
        "forward_return": ["forward_return", "fwd_return", "ret_fwd", "next_return", "ret21d", "ret63d"],
        "price": ["price", "last_price", "close", "adj_close"],
    }
    for target, names in aliases.items():
        if target not in out.columns:
            for name in names:
                if name.lower() in lower:
                    out[target] = out[lower[name.lower()]]
                    break
    if "market_value" not in out.columns and "price" in out.columns:
        out["market_value"] = out["price"]
    if "market_value" not in out.columns:
        for proxy in ["actual", "composite_score", "selection_score", "prediction", "screener_score"]:
            if proxy in out.columns:
                out["market_value"] = pd.to_numeric(out[proxy], errors="coerce")
                out["market_value_proxy_source"] = proxy
                break
    for col in out.columns:
        if col not in {"ticker", "date", "sector", "industry", "country", "issuer_name", "company_name", "coverage_note"}:
            converted = pd.to_numeric(out[col], errors="coerce")
            if converted.notna().sum() > 0:
                out[col] = converted
    out = out[out["ticker"].notna() & out["ticker"].ne("")]
    return out.reset_index(drop=True)


def validate_panel_coverage(
    panel: pd.DataFrame,
    min_tickers: int = 5,
    min_dates: int = 2,
    min_rows: int | None = None,
    date_col: str = "date",
    ticker_col: str = "ticker",
    required_columns: Iterable[str] = ("market_value",),
) -> pd.DataFrame:
    """Return a one-row coverage/should-run diagnostic for an ML panel."""
    min_rows = int(min_rows if min_rows is not None else max(1, min_tickers))
    rows = int(len(panel))
    reasons: list[str] = []

    if rows == 0:
        reasons.append("PANEL_EMPTY")

    ticker_count = 0
    if ticker_col in panel.columns:
        tickers = panel[ticker_col].dropna().astype(str)
        ticker_count = int(tickers[tickers.ne("")].nunique())
    else:
        reasons.append(f"MISSING_{ticker_col.upper()}")

    if date_col in panel.columns:
        dates = pd.to_datetime(panel[date_col], errors="coerce")
        valid_dates = dates.dropna()
        date_count = int(valid_dates.nunique())
        min_date = valid_dates.min().date().isoformat() if not valid_dates.empty else pd.NA
        max_date = valid_dates.max().date().isoformat() if not valid_dates.empty else pd.NA
    else:
        date_count = 0
        min_date = pd.NA
        max_date = pd.NA
        reasons.append(f"MISSING_{date_col.upper()}")

    for col in required_columns:
        if col not in panel.columns:
            reasons.append(f"MISSING_{col.upper()}")
        elif pd.to_numeric(panel[col], errors="coerce").notna().sum() == 0:
            reasons.append(f"EMPTY_{col.upper()}")

    if rows < min_rows:
        reasons.append(f"ROWS_LT_{min_rows}")
    if ticker_count < min_tickers:
        reasons.append(f"TICKERS_LT_{min_tickers}")
    if date_count < min_dates:
        reasons.append(f"DATES_LT_{min_dates}")

    should_run = not reasons
    return pd.DataFrame([{
        "status": "OK" if should_run else "INSUFFICIENT_PANEL",
        "should_run": bool(should_run),
        "reason": "OK" if should_run else ";".join(dict.fromkeys(reasons)),
        "panel_rows": rows,
        "ticker_count": ticker_count,
        "date_count": date_count,
        "min_date": min_date,
        "max_date": max_date,
        "min_required_rows": min_rows,
        "min_required_tickers": int(min_tickers),
        "min_required_dates": int(min_dates),
    }])


def load_artifact_panel(output_root: str | Path | None = None) -> pd.DataFrame:
    """Load the best available panel-like artifact from existing notebooks."""
    root = Path(output_root) if output_root else _project_root()
    frames: list[pd.DataFrame] = []
    for path in _candidate_artifacts(root):
        df = _read_csv(path)
        if not df.empty:
            df["source_artifact"] = str(path)
            frames.append(normalize_panel(df))
    if not frames:
        return pd.DataFrame(columns=["date", "ticker", "market_value"])
    panel = pd.concat(frames, ignore_index=True, sort=False)
    panel = panel.drop_duplicates(["date", "ticker"], keep="first")
    return panel.sort_values(["date", "ticker"]).reset_index(drop=True)


def load_financial_db_panel(
    financial_db_root: str | Path | None = None,
    identifiers: Iterable[str] | None = None,
    output_root: str | Path | None = None,
) -> pd.DataFrame:
    """Load an ML-ready panel from Database Finanziario or existing artifacts.

    Current implementation is conservative: it first uses project artifacts that
    already represent valuation/screener/portfolio outputs, then optionally
    attempts known Drive-first price paths via the existing data platform helper.
    """
    artifact_panel = load_artifact_panel(output_root)
    if not artifact_panel.empty:
        return artifact_panel
    if discover_financial_database_root is None or read_dataset_drive_first is None:
        return artifact_panel
    db_root, _, _ = discover_financial_database_root([Path(financial_db_root)] if financial_db_root else None)
    frames: list[pd.DataFrame] = []
    for identifier in identifiers or []:
        try:
            df = read_dataset_drive_first(db_root, "prices", identifier)
            if isinstance(df, pd.DataFrame) and not df.empty:
                df = df.copy()
                df["ticker"] = identifier
                frames.append(normalize_panel(df))
        except Exception:
            continue
    return pd.concat(frames, ignore_index=True, sort=False) if frames else artifact_panel


def load_aqr_factor_panel(
    slugs: Iterable[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    refresh: bool = False,
    max_datasets: int | None = None,
    financial_db_root: str | Path | None = None,
    output_root: str | Path | None = None,
) -> pd.DataFrame:
    """Load AQR factors through the shared Drive-first factor provider."""
    if get_aqr_factor_panel is None:
        return pd.DataFrame()
    return get_aqr_factor_panel(
        slugs=slugs,
        start=start,
        end=end,
        refresh=refresh,
        max_datasets=max_datasets,
        financial_db_root=financial_db_root,
        output_root=output_root,
    )


class FundamentalDatasetBuilder:
    """Build time-aware ML panels with feature/target columns."""

    def __init__(self, feature_columns: list[str] | None = None, target_col: str = "market_value") -> None:
        self.feature_columns = feature_columns
        self.target_col = target_col

    def build(self, panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
        panel = normalize_panel(panel)
        numeric_cols = [c for c in panel.columns if pd.api.types.is_numeric_dtype(panel[c])]
        excluded = {"market_value", "forward_return", "date"}
        features = self.feature_columns or [c for c in numeric_cols if c not in excluded]
        features = [c for c in features if c in panel.columns]
        if self.target_col not in panel.columns and "market_value" in panel.columns:
            self.target_col = "market_value"
        X = panel[features].replace([np.inf, -np.inf], np.nan).fillna(panel[features].median(numeric_only=True))
        y = pd.to_numeric(panel[self.target_col], errors="coerce") if self.target_col in panel.columns else pd.Series(index=panel.index, dtype=float)
        mask = y.notna()
        return X.loc[mask], y.loc[mask], features
