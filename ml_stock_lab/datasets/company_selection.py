"""Company universe selection helpers built from local research artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DISPLAY_COLUMNS = [
    "selected",
    "ticker",
    "company_name",
    "country",
    "sector",
    "industry",
    "index_membership",
    "market_cap",
    "price",
    "screener_score",
    "screener_rank",
    "source_count",
    "data_sources",
]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def company_source_paths(project_root: str | Path | None = None) -> list[Path]:
    """Return the standard cached company/universe artifact paths."""
    root = Path(project_root).expanduser() if project_root else _project_root()
    return [
        root / "papers" / "italian_banks_ml_stock_screening" / "data" / "italian_banks_panel.csv",
        root / "research_platform_definitive" / "company_valuation" / "output" / "tables" / "FinvizSecuritiesMaster.csv",
        root / "research_platform_definitive" / "company_valuation" / "output" / "tables" / "FinvizLatestCrossSection.csv",
        root / "research_platform_definitive" / "company_valuation" / "output" / "tables" / "ScreenerResults.csv",
        root / "research_platform_definitive" / "company_valuation" / "output" / "tables" / "screener_results.csv",
        root / "research_platform_definitive" / "company_valuation" / "output" / "tables" / "universe_master.csv",
        root / "research_platform_definitive" / "company_valuation" / "output" / "tables" / "refreshed_fundamentals_light.csv",
        root / "research_platform_definitive" / "output" / "ml_stock_lab" / "tables" / "MLStockLab_panel.csv",
    ]


def _first_existing_column(df: pd.DataFrame, names: Iterable[str]) -> str | None:
    lower = {str(col).lower(): col for col in df.columns}
    for name in names:
        if name in df.columns:
            return name
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def _read_company_source(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        raw = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    if raw.empty:
        return pd.DataFrame()

    out = pd.DataFrame(index=raw.index)
    aliases = {
        "ticker": ["ticker", "symbol", "canonical_ticker"],
        "company_name": ["company_name", "issuer_name", "name", "security", "Security"],
        "country": ["country", "meta_country"],
        "sector": ["sector", "meta_sector"],
        "industry": ["industry", "meta_industry"],
        "exchange": ["exchange"],
        "index_membership": ["index_membership", "universe_source", "portfolio"],
        "universe_source": ["universe_source", "source", "source_artifact"],
        "market_cap": ["market_cap", "market_value", "mkt_market_cap", "marketcap"],
        "price": ["price", "mkt_price", "last_price", "close", "adj_close"],
        "liquidity": ["liquidity", "turnover", "volume"],
        "screener_score": ["screener_score", "composite_score", "selection_score"],
        "screener_rank": ["screener_rank", "rank", "selection_rank"],
        "quality_flag": ["quality_flag", "robustness_status", "status"],
        "updated_at": ["updated_at", "date", "timestamp"],
    }
    for target, names in aliases.items():
        col = _first_existing_column(raw, names)
        out[target] = raw[col] if col else pd.NA

    out["ticker"] = out["ticker"].astype(str).str.strip().str.upper()
    out = out[out["ticker"].ne("") & out["ticker"].ne("NAN")].copy()
    if out.empty:
        return out

    for col in ["market_cap", "price", "liquidity", "screener_score", "screener_rank", "source_count"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out["source_file"] = path.name
    out["source_path"] = str(path)
    return out.reset_index(drop=True)


def _first_valid(series: pd.Series):
    valid = series.dropna()
    valid = valid[valid.astype(str).str.strip().ne("")]
    return valid.iloc[0] if not valid.empty else pd.NA


def _max_numeric(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.max()) if not values.empty else np.nan


def build_company_selection_panel(
    project_root: str | Path | None = None,
    source_paths: Iterable[str | Path] | None = None,
    country: str | None = None,
    sector: str | None = None,
    tickers: Iterable[str] | None = None,
    default_selected: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Build a deduplicated company-choice panel from standard local artifacts."""
    paths = [Path(p).expanduser() for p in (source_paths or company_source_paths(project_root))]
    selected_set = {str(ticker).strip().upper() for ticker in (default_selected or [])}
    frames = [_read_company_source(path) for path in paths]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        if not selected_set:
            return pd.DataFrame(columns=DISPLAY_COLUMNS)
        manual = pd.DataFrame({"ticker": sorted(selected_set)})
        manual["selected"] = True
        manual["data_sources"] = "manual_default"
        manual["source_count"] = 1
        return manual[[col for col in DISPLAY_COLUMNS if col in manual.columns] + [col for col in manual.columns if col not in DISPLAY_COLUMNS]]

    raw = pd.concat(frames, ignore_index=True, sort=False)
    grouped = raw.groupby("ticker", dropna=False)
    panel = grouped.agg(
        company_name=("company_name", _first_valid),
        country=("country", _first_valid),
        sector=("sector", _first_valid),
        industry=("industry", _first_valid),
        exchange=("exchange", _first_valid),
        index_membership=("index_membership", _first_valid),
        universe_source=("universe_source", _first_valid),
        market_cap=("market_cap", _max_numeric),
        price=("price", _max_numeric),
        liquidity=("liquidity", _max_numeric),
        screener_score=("screener_score", _max_numeric),
        screener_rank=("screener_rank", _max_numeric),
        quality_flag=("quality_flag", _first_valid),
        updated_at=("updated_at", _first_valid),
        source_count=("source_file", "nunique"),
        data_sources=("source_file", lambda s: ", ".join(sorted(set(s.dropna().astype(str))))),
    ).reset_index()

    if country:
        country_key = str(country).lower()
        panel = panel[panel["country"].astype(str).str.lower().eq(country_key)]
    if sector:
        sector_key = str(sector).lower()
        panel = panel[panel["sector"].astype(str).str.lower().eq(sector_key)]
    if tickers:
        ticker_set = {str(ticker).strip().upper() for ticker in tickers}
        panel = panel[panel["ticker"].isin(ticker_set)]

    missing_selected = sorted(selected_set - set(panel["ticker"]))
    if missing_selected:
        manual = pd.DataFrame({
            "ticker": missing_selected,
            "company_name": pd.NA,
            "country": pd.NA,
            "sector": pd.NA,
            "industry": pd.NA,
            "index_membership": "manual_default",
            "market_cap": np.nan,
            "price": np.nan,
            "screener_score": np.nan,
            "screener_rank": np.nan,
            "source_count": 1,
            "data_sources": "manual_default",
        })
        panel = pd.concat([panel, manual], ignore_index=True, sort=False)

    panel["selected"] = panel["ticker"].isin(selected_set) if selected_set else False
    sort_cols = ["selected", "source_count", "screener_score", "ticker"]
    panel = panel.sort_values(sort_cols, ascending=[False, False, False, True], na_position="last")
    cols = [col for col in DISPLAY_COLUMNS if col in panel.columns] + [col for col in panel.columns if col not in DISPLAY_COLUMNS]
    return panel[cols].reset_index(drop=True)


def filter_company_selection_panel(
    panel: pd.DataFrame,
    query: str | None = None,
    country: str | None = None,
    sector: str | None = None,
    selected_only: bool = False,
) -> pd.DataFrame:
    """Filter a company-choice panel by free text and metadata."""
    out = panel.copy()
    if query:
        q = str(query).strip().lower()
        searchable = out[["ticker", "company_name", "country", "sector", "industry", "index_membership"]].fillna("").astype(str)
        mask = searchable.apply(lambda col: col.str.lower().str.contains(q, regex=False)).any(axis=1)
        out = out[mask]
    if country and country != "All":
        out = out[out["country"].astype(str).eq(country)]
    if sector and sector != "All":
        out = out[out["sector"].astype(str).eq(sector)]
    if selected_only and "selected" in out.columns:
        out = out[out["selected"].astype(bool)]
    return out.reset_index(drop=True)


def build_company_selection_widget(
    panel: pd.DataFrame,
    default_tickers: Iterable[str] | None = None,
    max_options: int = 500,
):
    """Create an ipywidgets company selector and return a small controller dict."""
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except Exception as exc:
        raise ImportError("ipywidgets is required for the interactive selector.") from exc

    working = panel.copy()
    default_set = {str(ticker).strip().upper() for ticker in (default_tickers or [])}
    if default_set:
        working["selected"] = working["ticker"].isin(default_set)

    country_options = ["All"] + sorted([x for x in working["country"].dropna().astype(str).unique() if x and x != "nan"])
    sector_options = ["All"] + sorted([x for x in working["sector"].dropna().astype(str).unique() if x and x != "nan"])

    query = widgets.Text(value="", placeholder="Cerca ticker, nome, settore...", description="Cerca")
    country = widgets.Dropdown(options=country_options, value="All", description="Paese")
    sector = widgets.Dropdown(options=sector_options, value="All", description="Settore")
    selected_only = widgets.Checkbox(value=False, description="Solo selezionate")
    selector = widgets.SelectMultiple(options=[], value=(), rows=14, description="Aziende", layout=widgets.Layout(width="100%"))
    output = widgets.Output()

    def option_label(row: pd.Series) -> str:
        name = "" if pd.isna(row.get("company_name")) else str(row.get("company_name"))
        sector_name = "" if pd.isna(row.get("sector")) else str(row.get("sector"))
        country_name = "" if pd.isna(row.get("country")) else str(row.get("country"))
        return f"{row['ticker']} | {name} | {country_name} | {sector_name}".strip()

    def refresh(*_):
        filtered = filter_company_selection_panel(working, query.value, country.value, sector.value, selected_only.value)
        filtered = filtered.head(max_options)
        options = [(option_label(row), row["ticker"]) for _, row in filtered.iterrows()]
        current = tuple(ticker for ticker in selector.value if ticker in {opt[1] for opt in options})
        selector.options = options
        selector.value = current
        with output:
            output.clear_output()
            display_cols = [col for col in DISPLAY_COLUMNS if col in filtered.columns]
            display(filtered[display_cols].head(25))

    def mark_selected(change):
        selected = set(change.get("new", ()))
        working["selected"] = working["ticker"].isin(selected)

    for widget in [query, country, sector, selected_only]:
        widget.observe(refresh, names="value")
    selector.observe(mark_selected, names="value")

    refresh()
    if default_set:
        available = {value for _, value in selector.options}
        selector.value = tuple(sorted(default_set & available))

    ui = widgets.VBox([widgets.HBox([query, country, sector, selected_only]), selector, output])

    def selected_tickers() -> list[str]:
        return list(selector.value)

    return {"ui": ui, "selector": selector, "panel": working, "selected_tickers": selected_tickers, "display": lambda: display(ui)}
