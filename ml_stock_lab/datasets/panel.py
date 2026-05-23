"""Panel loading, normalization, feature engineering and ML dataset helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def normalize_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Return a normalized stock panel with `date`, `ticker` and numeric fields."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "ticker", "market_value"])

    out = df.copy()
    lower = {str(c).lower(): c for c in out.columns}

    if "ticker" not in out.columns:
        for name in ["symbol", "canonical_ticker", "asset", "security"]:
            if name in lower:
                out["ticker"] = out[lower[name]]
                break

    if "date" not in out.columns:
        for name in ["report_date", "asofdate", "as_of_date", "filing_date", "timestamp"]:
            if name in lower:
                out["date"] = out[lower[name]]
                break
    if "date" not in out.columns:
        out["date"] = pd.Timestamp.today().normalize()

    aliases = {
        "market_value": ["market_value", "marketcap", "market_cap", "mktcap", "price", "close", "adj_close"],
        "forward_return": ["forward_return", "fwd_return", "ret_fwd", "next_return", "ret21d", "ret63d"],
        "price": ["price", "last_price", "close", "adj_close"],
    }
    for target, names in aliases.items():
        if target not in out.columns:
            for name in names:
                if name in lower:
                    out[target] = out[lower[name]]
                    break

    if "market_value" not in out.columns and "price" in out.columns:
        out["market_value"] = out["price"]

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["ticker"] = out.get("ticker", pd.Series(index=out.index, dtype=object)).astype(str).str.upper()
    out = out[out["ticker"].notna() & out["ticker"].ne("")]

    protected = {"ticker", "date", "sector", "industry", "country", "issuer_name", "company_name"}
    for col in out.columns:
        if col not in protected:
            converted = pd.to_numeric(out[col], errors="coerce")
            if converted.notna().sum() > 0:
                out[col] = converted

    return out.reset_index(drop=True)


def load_panel_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV file and normalize it into the standard panel schema."""
    return normalize_panel(pd.read_csv(path))


def load_financial_db_panel(
    output_root: str | Path | None = None,
    financial_db_root: str | Path | None = None,
    identifiers: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Load the best available local stock panel from known CSV artifacts.

    The loader is intentionally conservative and offline-friendly: it scans
    project/output folders and optional identifier CSVs, then normalizes the
    first usable panel-like data it finds.
    """
    roots = [Path(output_root)] if output_root else [Path.cwd()]
    if financial_db_root:
        roots.append(Path(financial_db_root))

    candidate_names = [
        "MLStockLab_panel.csv",
        "ScreenerResults.csv",
        "screener_results.csv",
        "extended_valuation_results.csv",
        "valuation_gap_table.csv",
        "PortfolioSelectionResults.csv",
    ]
    frames: list[pd.DataFrame] = []
    for root in roots:
        search_roots = [root, root / "output", root / "tables"]
        for search_root in search_roots:
            if not search_root.exists():
                continue
            for name in candidate_names:
                for path in search_root.rglob(name):
                    try:
                        frame = load_panel_csv(path)
                    except Exception:
                        continue
                    if not frame.empty:
                        frame["source_artifact"] = str(path)
                        frames.append(frame)

        for identifier in identifiers or []:
            for path in root.rglob(f"{identifier}.csv"):
                try:
                    frame = load_panel_csv(path)
                except Exception:
                    continue
                if not frame.empty:
                    frame["ticker"] = frame["ticker"].where(frame["ticker"].ne(""), str(identifier).upper())
                    frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=["date", "ticker", "market_value"])

    panel = pd.concat(frames, ignore_index=True, sort=False)
    panel = normalize_panel(panel)
    if {"date", "ticker"}.issubset(panel.columns):
        panel = panel.drop_duplicates(["date", "ticker"], keep="first").sort_values(["date", "ticker"])
    return panel.reset_index(drop=True)


def add_basic_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Add simple numeric features commonly used by fundamental stock models."""
    if panel is None or panel.empty:
        return normalize_panel(pd.DataFrame())
    out = normalize_panel(panel)
    lower = {str(c).lower(): c for c in out.columns}
    aliases = {
        "pe": ["peratio", "pe_ratio", "p_e"],
        "pb": ["pbratio", "pb_ratio", "p_b"],
        "ev_ebitda": ["evebitda", "ev_ebitda"],
        "revenue_growth": ["revenuegrowth", "revenue_growth"],
        "debt_to_equity": ["debttoequity", "debt_equity"],
        "quality_score": ["qualityscore", "quality_score"],
        "valuation_score": ["valuationscore", "valuation_score"],
        "momentum_score": ["momentumscore", "momentum_score"],
        "risk_score": ["riskscore", "risk_score"],
    }
    for canonical, names in aliases.items():
        if canonical not in out.columns:
            for name in names:
                if name in lower:
                    out[canonical] = pd.to_numeric(out[lower[name]], errors="coerce")
                    break

    if "market_value" in out.columns:
        mv = pd.to_numeric(out["market_value"], errors="coerce")
        out["log_market_value"] = np.log(mv.where(mv > 0))
    if {"quality_score", "valuation_score"}.issubset(out.columns):
        out["quality_value_blend"] = pd.to_numeric(out["quality_score"], errors="coerce") + pd.to_numeric(out["valuation_score"], errors="coerce")
    if {"momentum_score", "risk_score"}.issubset(out.columns):
        out["risk_adjusted_momentum"] = pd.to_numeric(out["momentum_score"], errors="coerce") - pd.to_numeric(out["risk_score"], errors="coerce")
    for ret_col in ["ret21d", "ret63d", "ret126d"]:
        if ret_col in out.columns:
            values = pd.to_numeric(out[ret_col], errors="coerce")
            out[f"{ret_col}_rank"] = values.groupby(out["date"]).rank(pct=True) if "date" in out.columns else values.rank(pct=True)
    return out


def make_forward_returns(panel: pd.DataFrame, price_col: str = "price", horizon: int = 1) -> pd.DataFrame:
    """Create forward returns by ticker using a price-like column."""
    out = normalize_panel(panel)
    if price_col not in out.columns or "ticker" not in out.columns:
        return out
    sort_cols = ["ticker", "date"] if "date" in out.columns else ["ticker"]
    out = out.sort_values(sort_cols).copy()
    out["forward_return"] = out.groupby("ticker")[price_col].transform(
        lambda s: pd.to_numeric(s, errors="coerce").shift(-horizon) / pd.to_numeric(s, errors="coerce") - 1
    )
    return out


def select_numeric_features(panel: pd.DataFrame, target: str = "market_value", min_non_null: int = 5) -> list[str]:
    """Select numeric feature columns with enough non-null observations."""
    out = normalize_panel(panel)
    excluded = {target, "forward_return", "date"}
    return [
        col
        for col in out.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(out[col]) and out[col].notna().sum() >= min_non_null
    ]


class FundamentalDatasetBuilder:
    """Build feature and target matrices from a normalized fundamental panel."""

    def __init__(self, feature_columns: list[str] | None = None, target_col: str = "market_value") -> None:
        self.feature_columns = feature_columns
        self.target_col = target_col

    def build(self, panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
        """Return `(X, y, features)` with numeric values cleaned for modeling."""
        out = normalize_panel(panel)
        features = self.feature_columns or select_numeric_features(out, target=self.target_col)
        features = [col for col in features if col in out.columns]
        if self.target_col not in out.columns and "market_value" in out.columns:
            self.target_col = "market_value"
        X = out[features].replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
        y = pd.to_numeric(out[self.target_col], errors="coerce") if self.target_col in out.columns else pd.Series(index=out.index, dtype=float)
        mask = y.notna()
        return X.loc[mask], y.loc[mask], features
