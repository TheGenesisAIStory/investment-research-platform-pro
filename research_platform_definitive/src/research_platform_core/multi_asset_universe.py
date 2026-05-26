"""Multi-asset universe and coverage manifest helpers.

The equity universe has a dedicated OHLCV manifest.  This module gives the
non-equity side (FX, commodities, ETF, fixed income, crypto) an equivalent,
lightweight manifest derived from the Macro DB catalog/manifest.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd

from .data_platform import resolve_data_platform_roots, utc_now
from .macro_market import macro_asset_catalog


TABLE_REL = Path("macro_market") / "tables" / "MultiAssetUniverseManifest.csv"
INGEST_ROOT_NAME = "multi_asset_universe"
INGEST_MANIFEST_NAME = "MultiAssetUniverseManifest.csv"
INGEST_SNAPSHOT_NAME = "MultiAssetLatestSnapshot.csv"


ASSET_CLASS_MAP = {
    "fx": "fx",
    "commodity_future": "commodity",
    "commodity_etf": "etf_commodity",
    "crypto": "crypto",
    "crypto_etf": "crypto",
    "crypto_equity": "crypto",
    "equity_index_etf": "etf_equity",
    "sector_etf": "etf_equity",
    "equity_index": "etf_equity",
    "fixed_income_etf": "etf_fi",
    "rate_index": "yield_proxy",
    "volatility_index": "yield_proxy",
}

EXPANDED_MULTI_ASSET_ROWS: tuple[dict[str, Any], ...] = (
    # Equity indices and regional ETFs.
    {"symbol": "GSPC", "provider_symbol": "^GSPC", "name": "S&P 500 Index", "asset_class": "equity_index", "region": "us", "currency": "USD", "category": "large_cap", "exposure": "US equity index", "priority": 1},
    {"symbol": "IXIC", "provider_symbol": "^IXIC", "name": "NASDAQ Composite", "asset_class": "equity_index", "region": "us", "currency": "USD", "category": "growth", "exposure": "US technology/growth index", "priority": 1},
    {"symbol": "DJI", "provider_symbol": "^DJI", "name": "Dow Jones Industrial Average", "asset_class": "equity_index", "region": "us", "currency": "USD", "category": "blue_chip", "exposure": "US blue-chip index", "priority": 2},
    {"symbol": "RUT", "provider_symbol": "^RUT", "name": "Russell 2000 Index", "asset_class": "equity_index", "region": "us", "currency": "USD", "category": "small_cap", "exposure": "US small-cap index", "priority": 2},
    {"symbol": "DIA", "provider_symbol": "DIA", "name": "SPDR Dow Jones Industrial Average ETF", "asset_class": "equity_index_etf", "region": "us", "currency": "USD", "category": "blue_chip", "exposure": "US Dow ETF", "priority": 2},
    {"symbol": "URTH", "provider_symbol": "URTH", "name": "iShares MSCI World ETF", "asset_class": "equity_index_etf", "region": "global", "currency": "USD", "category": "developed_markets", "exposure": "MSCI World proxy", "priority": 1},
    {"symbol": "VEA", "provider_symbol": "VEA", "name": "Vanguard Developed Markets ETF", "asset_class": "equity_index_etf", "region": "global", "currency": "USD", "category": "developed_ex_us", "exposure": "Developed markets ex-US", "priority": 2},
    {"symbol": "STOXX50E", "provider_symbol": "^STOXX50E", "name": "EURO STOXX 50 Index", "asset_class": "equity_index", "region": "eu", "currency": "EUR", "category": "large_cap", "exposure": "Eurozone large-cap index", "priority": 1},
    {"symbol": "GDAXI", "provider_symbol": "^GDAXI", "name": "DAX Index", "asset_class": "equity_index", "region": "eu", "currency": "EUR", "category": "country_equity", "exposure": "Germany equity index", "priority": 2},
    {"symbol": "FCHI", "provider_symbol": "^FCHI", "name": "CAC 40 Index", "asset_class": "equity_index", "region": "eu", "currency": "EUR", "category": "country_equity", "exposure": "France equity index", "priority": 2},
    {"symbol": "IBEX", "provider_symbol": "^IBEX", "name": "IBEX 35 Index", "asset_class": "equity_index", "region": "eu", "currency": "EUR", "category": "country_equity", "exposure": "Spain equity index", "priority": 2},
    {"symbol": "AEX", "provider_symbol": "^AEX", "name": "AEX Index", "asset_class": "equity_index", "region": "eu", "currency": "EUR", "category": "country_equity", "exposure": "Netherlands equity index", "priority": 3},
    {"symbol": "EWN", "provider_symbol": "EWN", "name": "iShares MSCI Netherlands ETF", "asset_class": "equity_index_etf", "region": "eu", "currency": "USD", "category": "country_equity", "exposure": "Netherlands ETF", "priority": 3},
    {"symbol": "EWC", "provider_symbol": "EWC", "name": "iShares MSCI Canada ETF", "asset_class": "equity_index_etf", "region": "global", "currency": "USD", "category": "country_equity", "exposure": "Canada ETF", "priority": 3},
    {"symbol": "EWZ", "provider_symbol": "EWZ", "name": "iShares MSCI Brazil ETF", "asset_class": "equity_index_etf", "region": "em", "currency": "USD", "category": "country_equity", "exposure": "Brazil ETF", "priority": 2},
    {"symbol": "EWW", "provider_symbol": "EWW", "name": "iShares MSCI Mexico ETF", "asset_class": "equity_index_etf", "region": "em", "currency": "USD", "category": "country_equity", "exposure": "Mexico ETF", "priority": 3},
    {"symbol": "EWA", "provider_symbol": "EWA", "name": "iShares MSCI Australia ETF", "asset_class": "equity_index_etf", "region": "apac", "currency": "USD", "category": "country_equity", "exposure": "Australia ETF", "priority": 3},
    {"symbol": "EWH", "provider_symbol": "EWH", "name": "iShares MSCI Hong Kong ETF", "asset_class": "equity_index_etf", "region": "apac", "currency": "USD", "category": "country_equity", "exposure": "Hong Kong ETF", "priority": 3},
    {"symbol": "EWT", "provider_symbol": "EWT", "name": "iShares MSCI Taiwan ETF", "asset_class": "equity_index_etf", "region": "apac", "currency": "USD", "category": "country_equity", "exposure": "Taiwan ETF", "priority": 3},
    {"symbol": "EWY", "provider_symbol": "EWY", "name": "iShares MSCI South Korea ETF", "asset_class": "equity_index_etf", "region": "apac", "currency": "USD", "category": "country_equity", "exposure": "Korea ETF", "priority": 3},
    {"symbol": "EWS", "provider_symbol": "EWS", "name": "iShares MSCI Singapore ETF", "asset_class": "equity_index_etf", "region": "apac", "currency": "USD", "category": "country_equity", "exposure": "Singapore ETF", "priority": 3},
    {"symbol": "AAXJ", "provider_symbol": "AAXJ", "name": "iShares MSCI All Country Asia ex Japan ETF", "asset_class": "equity_index_etf", "region": "apac", "currency": "USD", "category": "regional_equity", "exposure": "Asia Pacific ex Japan", "priority": 2},
    {"symbol": "EWJ", "provider_symbol": "EWJ", "name": "iShares MSCI Japan ETF", "asset_class": "equity_index_etf", "region": "jp", "currency": "USD", "category": "country_equity", "exposure": "Japan ETF", "priority": 1},
    {"symbol": "DXJ", "provider_symbol": "DXJ", "name": "WisdomTree Japan Hedged Equity Fund", "asset_class": "equity_index_etf", "region": "jp", "currency": "USD", "category": "currency_hedged", "exposure": "Japan hedged equity ETF", "priority": 2},
    {"symbol": "N225", "provider_symbol": "^N225", "name": "Nikkei 225 Index", "asset_class": "equity_index", "region": "jp", "currency": "JPY", "category": "large_cap", "exposure": "Japan large-cap index", "priority": 1},
    # FX.
    {"symbol": "USDBRL", "provider_symbol": "USDBRL=X", "name": "USD / BRL", "asset_class": "fx", "region": "em", "currency": "BRL", "category": "em_fx", "exposure": "Brazilian real", "priority": 2},
    {"symbol": "USDMXN", "provider_symbol": "USDMXN=X", "name": "USD / MXN", "asset_class": "fx", "region": "em", "currency": "MXN", "category": "em_fx", "exposure": "Mexican peso", "priority": 2},
    {"symbol": "USDCNY", "provider_symbol": "USDCNY=X", "name": "USD / CNY", "asset_class": "fx", "region": "em", "currency": "CNY", "category": "em_fx", "exposure": "Chinese yuan", "priority": 2},
    {"symbol": "USDINR", "provider_symbol": "USDINR=X", "name": "USD / INR", "asset_class": "fx", "region": "em", "currency": "INR", "category": "em_fx", "exposure": "Indian rupee", "priority": 2},
    {"symbol": "USDTRY", "provider_symbol": "USDTRY=X", "name": "USD / TRY", "asset_class": "fx", "region": "em", "currency": "TRY", "category": "em_fx", "exposure": "Turkish lira", "priority": 3},
    {"symbol": "USDZAR", "provider_symbol": "USDZAR=X", "name": "USD / ZAR", "asset_class": "fx", "region": "em", "currency": "ZAR", "category": "em_fx", "exposure": "South African rand", "priority": 3},
    {"symbol": "USDKRW", "provider_symbol": "USDKRW=X", "name": "USD / KRW", "asset_class": "fx", "region": "em", "currency": "KRW", "category": "em_fx", "exposure": "Korean won", "priority": 3},
    # Fixed income and yield proxies.
    {"symbol": "TYX", "provider_symbol": "^TYX", "name": "US 30Y Treasury Yield Index", "asset_class": "rate_index", "region": "us", "currency": "percent", "category": "rates", "exposure": "US 30Y yield proxy", "priority": 2},
    {"symbol": "BNDW", "provider_symbol": "BNDW", "name": "Vanguard Total World Bond ETF", "asset_class": "fixed_income_etf", "region": "global", "currency": "USD", "category": "global_bonds", "exposure": "Global aggregate bond ETF", "priority": 2},
    {"symbol": "IAGG", "provider_symbol": "IAGG", "name": "iShares Core International Aggregate Bond ETF", "asset_class": "fixed_income_etf", "region": "global", "currency": "USD", "category": "global_bonds_ex_us", "exposure": "International aggregate bonds", "priority": 3},
    {"symbol": "IBCI_AS", "provider_symbol": "IBCI.AS", "name": "iShares EUR Government Bond ETF proxy", "asset_class": "fixed_income_etf", "region": "eu", "currency": "EUR", "category": "duration", "exposure": "Euro government bond proxy", "priority": 2},
    {"symbol": "BTPI_MI", "provider_symbol": "BTPI.MI", "name": "BTP inflation-linked bond ETF proxy", "asset_class": "fixed_income_etf", "region": "it", "currency": "EUR", "category": "sovereign", "exposure": "Italy BTP proxy", "priority": 2},
    # Commodity ETFs and futures.
    {"symbol": "DJP", "provider_symbol": "DJP", "name": "iPath Bloomberg Commodity Index ETN", "asset_class": "commodity_etf", "region": "global", "currency": "USD", "category": "broad_commodities", "exposure": "Broad commodity basket", "priority": 2},
    {"symbol": "PDBC", "provider_symbol": "PDBC", "name": "Invesco Optimum Yield Diversified Commodity Strategy", "asset_class": "commodity_etf", "region": "global", "currency": "USD", "category": "broad_commodities", "exposure": "Broad commodity ETF", "priority": 2},
    {"symbol": "IAU", "provider_symbol": "IAU", "name": "iShares Gold Trust", "asset_class": "commodity_etf", "region": "global", "currency": "USD", "category": "precious_metals", "exposure": "Gold ETF", "priority": 2},
    {"symbol": "CPER", "provider_symbol": "CPER", "name": "United States Copper Index Fund", "asset_class": "commodity_etf", "region": "global", "currency": "USD", "category": "industrial_metals", "exposure": "Copper ETF", "priority": 3},
    {"symbol": "GC_F", "provider_symbol": "GC=F", "name": "Gold future", "asset_class": "commodity_future", "region": "global", "currency": "USD", "category": "precious_metals", "exposure": "Gold futures proxy", "priority": 1},
    {"symbol": "SI_F", "provider_symbol": "SI=F", "name": "Silver future", "asset_class": "commodity_future", "region": "global", "currency": "USD", "category": "precious_metals", "exposure": "Silver futures proxy", "priority": 1},
    # Crypto.
    {"symbol": "MATIC", "provider_symbol": "MATIC-USD", "name": "Polygon", "asset_class": "crypto", "region": "global", "currency": "USD", "category": "crypto_alt", "exposure": "Polygon price", "priority": 3},
    {"symbol": "IBIT", "provider_symbol": "IBIT", "name": "iShares Bitcoin Trust", "asset_class": "crypto_etf", "region": "us", "currency": "USD", "category": "crypto_etf", "exposure": "US spot bitcoin ETF", "priority": 1},
    {"symbol": "FBTC", "provider_symbol": "FBTC", "name": "Fidelity Wise Origin Bitcoin Fund", "asset_class": "crypto_etf", "region": "us", "currency": "USD", "category": "crypto_etf", "exposure": "US spot bitcoin ETF", "priority": 2},
)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 1:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _safe_symbol(symbol: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(symbol).upper())


def _normalized_asset_class(value: object) -> str:
    raw = str(value or "").lower()
    return ASSET_CLASS_MAP.get(raw, raw or "other")


def _priority_for_asset(asset_class: object, region: object, symbol: object) -> int:
    asset = _normalized_asset_class(asset_class)
    reg = str(region or "").lower()
    sym = str(symbol or "").upper()
    core_symbols = {
        "SPY",
        "QQQ",
        "IWM",
        "ACWI",
        "VT",
        "EURUSD",
        "DXY",
        "WTI",
        "BRENT",
        "GLD",
        "TLT",
        "HYG",
        "TNX",
        "BTC",
        "ETH",
    }
    if sym in core_symbols or asset in {"fx", "commodity", "yield_proxy"}:
        return 1
    if reg in {"usa", "us", "eu", "italy", "global"}:
        return 2
    return 3


def _ingest_root(output_dir: str | Path | None = None) -> Path:
    roots = resolve_data_platform_roots(repo_output_root=output_dir)
    if output_dir is None:
        return roots.repo_output / INGEST_ROOT_NAME
    base = Path(output_dir).expanduser()
    return base if base.name == INGEST_ROOT_NAME else base / INGEST_ROOT_NAME


def _normalize_history(raw: pd.DataFrame, asset: pd.Series) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    frame = raw.reset_index().copy()
    frame = frame.rename(
        columns={
            "Date": "date",
            "Datetime": "date",
            "index": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adjclose",
            "Volume": "volume",
        }
    )
    if "adjclose" not in frame.columns and "close" in frame.columns:
        frame["adjclose"] = frame["close"]
    if "close" not in frame.columns and "adjclose" in frame.columns:
        frame["close"] = frame["adjclose"]
    for col in ["date", "open", "high", "low", "close", "adjclose", "volume"]:
        if col not in frame.columns:
            frame[col] = pd.NA
    out = frame[["date", "open", "high", "low", "close", "adjclose", "volume"]].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date")
    for col in ["open", "high", "low", "close", "adjclose", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in ["symbol", "provider_symbol", "name", "asset_class", "region", "currency", "source"]:
        out[col] = asset.get(col, "")
    return out[
        [
            "date",
            "symbol",
            "provider_symbol",
            "name",
            "asset_class",
            "region",
            "currency",
            "source",
            "open",
            "high",
            "low",
            "close",
            "adjclose",
            "volume",
        ]
    ].reset_index(drop=True)


def _rolling_snapshot(history: pd.DataFrame, asset: pd.Series, target_path: Path) -> dict[str, Any]:
    close = pd.to_numeric(history.get("close", pd.Series(dtype=float)), errors="coerce")
    clean = history.assign(close=close).dropna(subset=["date", "close"]).sort_values("date")

    def pct(days: int) -> float | None:
        if len(clean) <= days:
            return None
        base = float(clean["close"].iloc[-days - 1])
        if base == 0:
            return None
        return float(clean["close"].iloc[-1] / base - 1.0)

    if clean.empty:
        latest_date = ""
        latest_close = None
    else:
        latest_date = pd.to_datetime(clean["date"].iloc[-1]).date().isoformat()
        latest_close = float(clean["close"].iloc[-1])
    return {
        "symbol": asset.get("symbol", ""),
        "name": asset.get("name", ""),
        "asset_class": asset.get("asset_class", ""),
        "region": asset.get("region", ""),
        "date": latest_date,
        "close": latest_close,
        "pct_1d": pct(1),
        "pct_5d": pct(5),
        "pct_21d": pct(21),
        "pct_63d": pct(63),
        "target_path": str(target_path),
        "updated_at": utc_now(),
    }


def multi_asset_universe_catalog(
    symbols: Iterable[str] | None = None,
    *,
    regions: Iterable[str] | None = None,
    asset_classes: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Return the normalized multi-asset catalog used by ingestion and UI."""
    catalog = macro_asset_catalog(regions=regions, asset_classes=asset_classes).copy()
    extra = pd.DataFrame(EXPANDED_MULTI_ASSET_ROWS)
    if not extra.empty:
        extra["source"] = "yfinance"
        extra["status"] = "READY"
        extra["country"] = extra.get("country", extra["region"].astype(str).str.upper())
        catalog = pd.concat([catalog, extra], ignore_index=True, sort=False)
        catalog = catalog.drop_duplicates("symbol", keep="first")
    catalog["asset_class"] = catalog["asset_class"].map(_normalized_asset_class)
    catalog["region"] = catalog["region"].astype(str).str.lower().replace({"usa": "us", "italy": "it", "crypto": "global"})
    catalog["source"] = catalog.get("source", "yfinance").fillna("yfinance")
    catalog["data_status"] = catalog.get("status", "READY").fillna("READY").replace({"READY": "PLANNED", "PROXY": "PLANNED"})
    catalog["priority"] = [
        int(row.get("priority")) if pd.notna(row.get("priority", pd.NA)) else _priority_for_asset(row.get("asset_class"), row.get("region"), row.get("symbol"))
        for _, row in catalog.iterrows()
    ]
    catalog["notes"] = catalog.apply(
        lambda row: f"{row.get('category', '')} / {row.get('exposure', '')}".strip(" /"),
        axis=1,
    )
    if symbols:
        wanted = {str(symbol).upper() for symbol in symbols}
        catalog = catalog[
            catalog["symbol"].astype(str).str.upper().isin(wanted)
            | catalog["provider_symbol"].astype(str).str.upper().isin(wanted)
        ]
    columns = [
        "symbol",
        "provider_symbol",
        "name",
        "asset_class",
        "region",
        "currency",
        "source",
        "data_status",
        "priority",
        "notes",
    ]
    return catalog[[col for col in columns if col in catalog.columns]].sort_values(["asset_class", "region", "priority", "symbol"]).reset_index(drop=True)


def ingest_multi_asset_universe(
    symbols_list: Iterable[str] | None = None,
    start_date: str = "2000-01-01",
    end_date: str | None = None,
    output_dir: str | Path | None = None,
    *,
    download_fn: Callable[[str, str, str | None], pd.DataFrame] | None = None,
    refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """Ingest the multi-asset catalog to local parquet + manifest artifacts.

    The function is deliberately backend-only. Streamlit pages call it through
    explicit user actions; tests can pass `download_fn` to avoid network I/O.
    """
    if download_fn is None:
        import yfinance as yf

        def download_fn(provider_symbol: str, start: str, end: str | None) -> pd.DataFrame:
            return yf.Ticker(provider_symbol).history(start=start, end=end, interval="1d", auto_adjust=True)

    root = _ingest_root(output_dir)
    ohlcv_root = root / "ohlcv"
    log_root = root / "logs"
    ohlcv_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    catalog = multi_asset_universe_catalog(symbols_list)
    manifest_rows: list[dict[str, Any]] = []
    snapshot_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    for _, asset in catalog.iterrows():
        target = ohlcv_root / f"{_safe_symbol(str(asset['symbol']))}.parquet"
        status = "PARTIAL"
        error = ""
        if target.exists() and not refresh:
            try:
                history = pd.read_parquet(target)
                status = "OK" if not history.empty else "PARTIAL"
            except Exception as exc:
                history = pd.DataFrame()
                status = "PARTIAL"
                error = f"{type(exc).__name__}: {exc}"
        else:
            try:
                raw = download_fn(str(asset["provider_symbol"]), start_date, end_date)
                history = _normalize_history(raw, asset)
                if not history.empty:
                    history.to_parquet(target, index=False)
                    status = "OK"
                else:
                    status = "PARTIAL"
                    error = "NO_DATA"
            except Exception as exc:
                history = pd.DataFrame()
                status = "PARTIAL"
                error = f"{type(exc).__name__}: {exc}"
        if history.empty and error:
            failure_rows.append(
                {
                    "symbol": asset.get("symbol", ""),
                    "provider_symbol": asset.get("provider_symbol", ""),
                    "asset_class": asset.get("asset_class", ""),
                    "region": asset.get("region", ""),
                    "error": error,
                    "updated_at": utc_now(),
                }
            )
        first_date = ""
        last_date = ""
        if not history.empty and "date" in history.columns:
            dates = pd.to_datetime(history["date"], errors="coerce")
            if dates.notna().any():
                first_date = dates.min().date().isoformat()
                last_date = dates.max().date().isoformat()
        manifest_rows.append(
            {
                **asset.to_dict(),
                "data_status": status,
                "first_date": first_date,
                "last_date": last_date,
                "n_rows": int(len(history)),
                "source": asset.get("source", "yfinance"),
                "target_path": str(target),
                "error": error,
                "updated_at": utc_now(),
            }
        )
        if not history.empty:
            snapshot_rows.append(_rolling_snapshot(history, asset, target))

    manifest = pd.DataFrame(manifest_rows)
    snapshot = pd.DataFrame(snapshot_rows)
    failures = pd.DataFrame(failure_rows)
    manifest.to_csv(root / INGEST_MANIFEST_NAME, index=False)
    snapshot.to_csv(root / INGEST_SNAPSHOT_NAME, index=False)
    if not failures.empty:
        failures.to_csv(log_root / "MultiAssetIngestFailures.csv", index=False)
    return {"catalog": catalog, "manifest": manifest, "snapshot": snapshot, "failures": failures}
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _domain_for_asset_class(asset_class: str) -> str:
    value = str(asset_class or "").lower()
    if value == "fx":
        return "FX"
    if value == "commodity_etf":
        return "ETF Commodity"
    if value == "etf_commodity":
        return "ETF Commodity"
    if value.startswith("commodity"):
        return "Commodities"
    if value == "crypto_etf":
        return "ETF Crypto"
    if "crypto" in value:
        return "Crypto"
    if value == "fixed_income_etf":
        return "ETF Fixed Income"
    if value == "etf_fi":
        return "ETF Fixed Income"
    if "fixed_income" in value:
        return "Fixed Income"
    if value == "rate_index":
        return "Fixed Income"
    if value == "yield_proxy":
        return "Fixed Income"
    if value == "volatility_index":
        return "Volatility"
    if value in {"equity_index_etf", "sector_etf"}:
        return "ETF Equity"
    if value == "etf_equity":
        return "ETF Equity"
    if value.endswith("_etf") or "etf" in value:
        return "ETF"
    if "equity_index" in value:
        return "Equity Index / ETF"
    return "Other"


def compile_multi_asset_universe_manifest(
    financial_db_root: str | Path | None = None,
    output_root: str | Path | None = None,
) -> pd.DataFrame:
    """Build a manifest for non-equity/macro tradable proxies."""
    roots = resolve_data_platform_roots(financial_db_root=financial_db_root, repo_output_root=output_root)
    table_root = roots.repo_output / TABLE_REL.parent
    table_root.mkdir(parents=True, exist_ok=True)
    catalog = multi_asset_universe_catalog()
    catalog.to_csv(table_root / "MacroAssetCatalog.csv", index=False)
    manifest_paths = [
        roots.repo_output / "macro_market" / "tables" / "MacroAssetManifest.csv",
        roots.financial_db / "MarketData" / "Macro" / "MacroAssetManifest.csv",
    ]
    macro_manifest = next((frame for frame in (_read_csv(path) for path in manifest_paths) if not frame.empty), pd.DataFrame())
    if not macro_manifest.empty and "symbol" in macro_manifest.columns:
        macro_manifest = macro_manifest.copy()
        macro_manifest["symbol"] = macro_manifest["symbol"].astype(str).str.upper()
        catalog = catalog.merge(
            macro_manifest[
                [
                    col
                    for col in [
                        "symbol",
                        "status",
                        "rows",
                        "last_date",
                        "target_path",
                        "updated_at",
                    ]
                    if col in macro_manifest.columns
                ]
            ],
            on="symbol",
            how="left",
            suffixes=("", "_manifest"),
        )
    else:
        catalog["status"] = "PLANNED"
        catalog["rows"] = 0
        catalog["last_date"] = ""
        catalog["target_path"] = ""
        catalog["updated_at"] = ""

    out = catalog.copy()
    out["symbol"] = out["symbol"].astype(str).str.upper()
    out["domain"] = out["asset_class"].map(_domain_for_asset_class)
    status_source = (
        out["status_manifest"]
        if "status_manifest" in out.columns
        else out["status"]
        if "status" in out.columns
        else pd.Series("PLANNED", index=out.index)
    )
    out["data_status"] = status_source.fillna("PLANNED").astype(str).str.upper()
    out["data_status"] = out["data_status"].replace({"READY": "PLANNED", "PROXY": "PLANNED"})
    out["rows"] = pd.to_numeric(out.get("rows", 0), errors="coerce").fillna(0).astype(int)
    out["coverage_label"] = out.apply(
        lambda row: "OK" if str(row["data_status"]).upper() == "OK" and int(row["rows"]) > 0 else str(row["data_status"]).upper(),
        axis=1,
    )
    out["first_date"] = ""
    last_date = out["last_date"] if "last_date" in out.columns else pd.Series("", index=out.index)
    out["last_date"] = last_date.fillna("").astype(str)
    out["manifest_updated_at"] = utc_now()
    columns = [
        "domain",
        "symbol",
        "provider_symbol",
        "name",
        "asset_class",
        "region",
        "country",
        "category",
        "exposure",
        "currency",
        "source",
        "coverage_label",
        "data_status",
        "rows",
        "first_date",
        "last_date",
        "target_path",
        "manifest_updated_at",
    ]
    out = out[[col for col in columns if col in out.columns]].sort_values(["domain", "region", "asset_class", "symbol"]).reset_index(drop=True)
    out.to_csv(table_root / TABLE_REL.name, index=False)
    return out


def load_multi_asset_universe_manifest(
    financial_db_root: str | Path | None = None,
    output_root: str | Path | None = None,
    *,
    refresh_if_missing: bool = True,
) -> pd.DataFrame:
    roots = resolve_data_platform_roots(financial_db_root=financial_db_root, repo_output_root=output_root)
    ingest_path = roots.repo_output / INGEST_ROOT_NAME / INGEST_MANIFEST_NAME
    path = ingest_path if ingest_path.exists() and ingest_path.stat().st_size > 1 else roots.repo_output / TABLE_REL
    frame = _read_csv(path)
    if frame.empty and refresh_if_missing:
        frame = compile_multi_asset_universe_manifest(roots.financial_db, roots.repo_output)
    if not frame.empty:
        frame = frame.copy()
        if "domain" not in frame.columns and "asset_class" in frame.columns:
            frame["domain"] = frame["asset_class"].map(_domain_for_asset_class)
        if "rows" not in frame.columns and "n_rows" in frame.columns:
            frame["rows"] = frame["n_rows"]
        if "coverage_label" not in frame.columns and "data_status" in frame.columns:
            frame["coverage_label"] = frame["data_status"]
    return frame


def summarize_multi_asset_universe(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a multi-asset manifest by domain and status."""
    if frame.empty:
        return pd.DataFrame(columns=["domain", "instrument_count", "ok_count", "planned_count", "partial_count", "last_date"])
    view = frame.copy()
    if "domain" not in view.columns and "asset_class" in view.columns:
        view["domain"] = view["asset_class"].map(_domain_for_asset_class)
    if "coverage_label" not in view.columns:
        view["coverage_label"] = view["data_status"] if "data_status" in view.columns else "PLANNED"
    view["coverage_label"] = view["coverage_label"].fillna("PLANNED").astype(str).str.upper()
    grouped = view.groupby("domain", dropna=False)
    rows: list[dict[str, Any]] = []
    for domain, group in grouped:
        rows.append(
            {
                "domain": domain,
                "instrument_count": int(len(group)),
                "ok_count": int(group["coverage_label"].eq("OK").sum()),
                "planned_count": int(group["coverage_label"].eq("PLANNED").sum()),
                "partial_count": int(group["coverage_label"].isin(["PARTIAL", "NO_DATA", "FAILED"]).sum()),
                "last_date": group["last_date"].dropna().astype(str).max() if "last_date" in group.columns and group["last_date"].notna().any() else "",
            }
        )
    return pd.DataFrame(rows).sort_values("domain").reset_index(drop=True)
