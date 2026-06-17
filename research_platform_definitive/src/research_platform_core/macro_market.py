"""Macro market metadata, snapshots and lightweight database compiler.

The module is intentionally provider-light: yfinance is used for broad market
proxies that do not require keys, while official macro connectors remain in
the existing ECB/Banca d'Italia/FRED loaders. Outputs are small, table-first
artifacts that Streamlit can explore without scanning the whole Drive.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .data_platform import resolve_data_platform_roots, utc_now


@dataclass(frozen=True)
class MacroAsset:
    symbol: str
    provider_symbol: str
    name: str
    asset_class: str
    region: str
    country: str
    category: str
    exposure: str
    currency: str
    source: str = "yfinance"
    status: str = "READY"


MACRO_ASSETS: tuple[MacroAsset, ...] = (
    MacroAsset("SPY", "SPY", "SPDR S&P 500 ETF", "equity_index_etf", "usa", "US", "large_cap", "US equity beta", "USD"),
    MacroAsset("QQQ", "QQQ", "Invesco QQQ Trust", "equity_index_etf", "usa", "US", "growth", "US Nasdaq / growth beta", "USD"),
    MacroAsset("IWM", "IWM", "iShares Russell 2000 ETF", "equity_index_etf", "usa", "US", "small_cap", "US small-cap beta", "USD"),
    MacroAsset("VTI", "VTI", "Vanguard Total Stock Market ETF", "equity_index_etf", "usa", "US", "broad_market", "US total equity market", "USD"),
    MacroAsset("ACWI", "ACWI", "iShares MSCI ACWI ETF", "equity_index_etf", "global", "World", "global_equity", "Global developed/emerging equity", "USD"),
    MacroAsset("VT", "VT", "Vanguard Total World Stock ETF", "equity_index_etf", "global", "World", "global_equity", "Global all-country equity", "USD"),
    MacroAsset("EFA", "EFA", "iShares MSCI EAFE ETF", "equity_index_etf", "global", "World", "developed_ex_us", "Developed markets ex-US", "USD"),
    MacroAsset("EEM", "EEM", "iShares MSCI Emerging Markets ETF", "equity_index_etf", "global", "World", "emerging_markets", "Emerging markets equity", "USD"),
    MacroAsset("FEZ", "FEZ", "SPDR EURO STOXX 50 ETF", "equity_index_etf", "eu", "Euro Area", "large_cap", "Eurozone blue-chip equity", "USD"),
    MacroAsset("VGK", "VGK", "Vanguard FTSE Europe ETF", "equity_index_etf", "eu", "Europe", "broad_market", "European equity beta", "USD"),
    MacroAsset("EZU", "EZU", "iShares MSCI Eurozone ETF", "equity_index_etf", "eu", "Euro Area", "broad_market", "Eurozone equity beta", "USD"),
    MacroAsset("EWI", "EWI", "iShares MSCI Italy ETF", "equity_index_etf", "italy", "Italy", "country_equity", "Italy equity proxy", "USD"),
    MacroAsset("FTSEMIB", "FTSEMIB.MI", "FTSE MIB Index proxy", "equity_index", "italy", "Italy", "country_equity", "Italian large-cap equity", "EUR", status="PROXY"),
    MacroAsset("EURUSD", "EURUSD=X", "EUR / USD", "fx", "global", "Euro Area / US", "major_fx", "EURUSD exchange rate", "USD"),
    MacroAsset("GBPUSD", "GBPUSD=X", "GBP / USD", "fx", "global", "UK / US", "major_fx", "GBPUSD exchange rate", "USD"),
    MacroAsset("USDJPY", "USDJPY=X", "USD / JPY", "fx", "global", "Japan / US", "major_fx", "USDJPY exchange rate", "JPY"),
    MacroAsset("USDCHF", "USDCHF=X", "USD / CHF", "fx", "global", "Switzerland / US", "major_fx", "USDCHF exchange rate", "CHF"),
    MacroAsset("DXY", "DX-Y.NYB", "US Dollar Index", "fx", "global", "US", "dollar_index", "Broad USD strength", "USD"),
    MacroAsset("VIX", "^VIX", "CBOE Volatility Index", "volatility_index", "usa", "US", "implied_volatility", "US equity implied volatility", "index"),
    MacroAsset("GLD", "GLD", "SPDR Gold Shares", "commodity_etf", "global", "World", "precious_metals", "Gold proxy", "USD"),
    MacroAsset("SLV", "SLV", "iShares Silver Trust", "commodity_etf", "global", "World", "precious_metals", "Silver proxy", "USD"),
    MacroAsset("DBC", "DBC", "Invesco DB Commodity Index ETF", "commodity_etf", "global", "World", "broad_commodities", "Broad commodity basket", "USD"),
    MacroAsset("WTI", "CL=F", "WTI crude oil future", "commodity_future", "global", "World", "energy", "Oil price proxy", "USD"),
    MacroAsset("BRENT", "BZ=F", "Brent crude oil future", "commodity_future", "global", "World", "energy", "Brent oil proxy", "USD"),
    MacroAsset("COPPER", "HG=F", "Copper future", "commodity_future", "global", "World", "industrial_metals", "Cyclical metal proxy", "USD"),
    MacroAsset("TLT", "TLT", "iShares 20+ Year Treasury Bond ETF", "fixed_income_etf", "usa", "US", "duration", "Long-duration US Treasuries", "USD"),
    MacroAsset("IEF", "IEF", "iShares 7-10 Year Treasury Bond ETF", "fixed_income_etf", "usa", "US", "duration", "Intermediate US Treasuries", "USD"),
    MacroAsset("SHY", "SHY", "iShares 1-3 Year Treasury Bond ETF", "fixed_income_etf", "usa", "US", "front_end_rates", "Short US Treasuries", "USD"),
    MacroAsset("AGG", "AGG", "iShares Core US Aggregate Bond ETF", "fixed_income_etf", "usa", "US", "aggregate_bonds", "US aggregate fixed income", "USD"),
    MacroAsset("LQD", "LQD", "iShares iBoxx Investment Grade Corporate Bond ETF", "fixed_income_etf", "usa", "US", "credit", "US investment-grade credit", "USD"),
    MacroAsset("HYG", "HYG", "iShares iBoxx High Yield Corporate Bond ETF", "fixed_income_etf", "usa", "US", "credit", "US high-yield credit", "USD"),
    MacroAsset("TIP", "TIP", "iShares TIPS Bond ETF", "fixed_income_etf", "usa", "US", "inflation_linked", "US inflation-linked bonds", "USD"),
    MacroAsset("TNX", "^TNX", "US 10Y Treasury Yield Index", "rate_index", "usa", "US", "rates", "US 10Y yield proxy", "percent"),
    MacroAsset("IRX", "^IRX", "US 13W Treasury Yield Index", "rate_index", "usa", "US", "rates", "US front-end yield proxy", "percent"),
    MacroAsset("BUND_PROXY", "IBGL.AS", "Euro government bond ETF proxy", "fixed_income_etf", "eu", "Euro Area", "duration", "Euro government bond duration proxy", "EUR", status="PROXY"),
    MacroAsset("BTP_PROXY", "IBTM.MI", "Italy government bond ETF proxy", "fixed_income_etf", "italy", "Italy", "sovereign", "Italy BTP duration proxy", "EUR", status="PROXY"),
    MacroAsset("BTC", "BTC-USD", "Bitcoin", "crypto", "crypto", "Global", "crypto_major", "Bitcoin price", "USD"),
    MacroAsset("ETH", "ETH-USD", "Ethereum", "crypto", "crypto", "Global", "crypto_major", "Ethereum price", "USD"),
    MacroAsset("SOL", "SOL-USD", "Solana", "crypto", "crypto", "Global", "crypto_alt", "Solana price", "USD"),
    MacroAsset("BNB", "BNB-USD", "BNB", "crypto", "crypto", "Global", "crypto_alt", "BNB price", "USD"),
    MacroAsset("XRP", "XRP-USD", "XRP", "crypto", "crypto", "Global", "crypto_alt", "XRP price", "USD"),
    MacroAsset("COIN", "COIN", "Coinbase", "crypto_equity", "usa", "US", "crypto_equity", "Listed crypto exchange proxy", "USD"),
    MacroAsset("BITO", "BITO", "ProShares Bitcoin Strategy ETF", "crypto_etf", "usa", "US", "crypto_etf", "US bitcoin-linked ETF proxy", "USD"),
    MacroAsset("AUDUSD", "AUDUSD=X", "AUD / USD", "fx", "global", "Australia / US", "major_fx", "AUDUSD exchange rate", "USD"),
    MacroAsset("NZDUSD", "NZDUSD=X", "NZD / USD", "fx", "global", "New Zealand / US", "major_fx", "NZDUSD exchange rate", "USD"),
    MacroAsset("USDCAD", "CAD=X", "USD / CAD", "fx", "global", "Canada / US", "major_fx", "USDCAD exchange rate", "CAD"),
    MacroAsset("EURGBP", "EURGBP=X", "EUR / GBP", "fx", "eu", "Euro Area / UK", "cross_fx", "EURGBP exchange rate", "GBP"),
    MacroAsset("EURJPY", "EURJPY=X", "EUR / JPY", "fx", "global", "Euro Area / Japan", "cross_fx", "EURJPY exchange rate", "JPY"),
    MacroAsset("EURCHF", "EURCHF=X", "EUR / CHF", "fx", "eu", "Euro Area / Switzerland", "cross_fx", "EURCHF exchange rate", "CHF"),
    MacroAsset("EURSEK", "EURSEK=X", "EUR / SEK", "fx", "eu", "Euro Area / Sweden", "cross_fx", "EURSEK exchange rate", "SEK"),
    MacroAsset("EURPLN", "EURPLN=X", "EUR / PLN", "fx", "eu", "Euro Area / Poland", "cross_fx", "EURPLN exchange rate", "PLN"),
    MacroAsset("XLF", "XLF", "Financial Select Sector SPDR", "sector_etf", "usa", "US", "financials", "US financials sector", "USD"),
    MacroAsset("XLK", "XLK", "Technology Select Sector SPDR", "sector_etf", "usa", "US", "technology", "US technology sector", "USD"),
    MacroAsset("XLE", "XLE", "Energy Select Sector SPDR", "sector_etf", "usa", "US", "energy", "US energy sector", "USD"),
    MacroAsset("XLY", "XLY", "Consumer Discretionary Select Sector SPDR", "sector_etf", "usa", "US", "consumer_discretionary", "US discretionary sector", "USD"),
    MacroAsset("XLP", "XLP", "Consumer Staples Select Sector SPDR", "sector_etf", "usa", "US", "consumer_staples", "US staples sector", "USD"),
    MacroAsset("XLV", "XLV", "Health Care Select Sector SPDR", "sector_etf", "usa", "US", "healthcare", "US healthcare sector", "USD"),
    MacroAsset("XLI", "XLI", "Industrial Select Sector SPDR", "sector_etf", "usa", "US", "industrials", "US industrials sector", "USD"),
    MacroAsset("XLU", "XLU", "Utilities Select Sector SPDR", "sector_etf", "usa", "US", "utilities", "US utilities sector", "USD"),
    MacroAsset("XLB", "XLB", "Materials Select Sector SPDR", "sector_etf", "usa", "US", "materials", "US materials sector", "USD"),
    MacroAsset("XLRE", "XLRE", "Real Estate Select Sector SPDR", "sector_etf", "usa", "US", "real_estate", "US real estate sector", "USD"),
    MacroAsset("XLC", "XLC", "Communication Services Select Sector SPDR", "sector_etf", "usa", "US", "communication_services", "US communication services sector", "USD"),
    MacroAsset("IEUR", "IEUR", "iShares Core MSCI Europe ETF", "equity_index_etf", "eu", "Europe", "broad_market", "European equity ETF proxy", "USD"),
    MacroAsset("HEZU", "HEZU", "iShares Currency Hedged MSCI Eurozone ETF", "equity_index_etf", "eu", "Euro Area", "currency_hedged", "Hedged Eurozone equity proxy", "USD"),
    MacroAsset("EWG", "EWG", "iShares MSCI Germany ETF", "equity_index_etf", "eu", "Germany", "country_equity", "Germany equity proxy", "USD"),
    MacroAsset("EWQ", "EWQ", "iShares MSCI France ETF", "equity_index_etf", "eu", "France", "country_equity", "France equity proxy", "USD"),
    MacroAsset("EWP", "EWP", "iShares MSCI Spain ETF", "equity_index_etf", "eu", "Spain", "country_equity", "Spain equity proxy", "USD"),
    MacroAsset("EWU", "EWU", "iShares MSCI United Kingdom ETF", "equity_index_etf", "eu", "UK", "country_equity", "UK equity proxy", "USD"),
    MacroAsset("EWJ", "EWJ", "iShares MSCI Japan ETF", "equity_index_etf", "global", "Japan", "country_equity", "Japan equity proxy", "USD"),
    MacroAsset("MCHI", "MCHI", "iShares MSCI China ETF", "equity_index_etf", "global", "China", "country_equity", "China equity proxy", "USD"),
    MacroAsset("INDA", "INDA", "iShares MSCI India ETF", "equity_index_etf", "global", "India", "country_equity", "India equity proxy", "USD"),
    MacroAsset("BND", "BND", "Vanguard Total Bond Market ETF", "fixed_income_etf", "usa", "US", "aggregate_bonds", "US broad fixed income", "USD"),
    MacroAsset("GOVT", "GOVT", "iShares U.S. Treasury Bond ETF", "fixed_income_etf", "usa", "US", "treasuries", "US Treasury curve proxy", "USD"),
    MacroAsset("SHV", "SHV", "iShares Short Treasury Bond ETF", "fixed_income_etf", "usa", "US", "cash_like", "US T-bill proxy", "USD"),
    MacroAsset("VCIT", "VCIT", "Vanguard Intermediate-Term Corporate Bond ETF", "fixed_income_etf", "usa", "US", "credit", "US intermediate credit", "USD"),
    MacroAsset("VCSH", "VCSH", "Vanguard Short-Term Corporate Bond ETF", "fixed_income_etf", "usa", "US", "short_credit", "US short corporate credit", "USD"),
    MacroAsset("MUB", "MUB", "iShares National Muni Bond ETF", "fixed_income_etf", "usa", "US", "municipal", "US municipal bond proxy", "USD"),
    MacroAsset("EMB", "EMB", "iShares J.P. Morgan USD Emerging Markets Bond ETF", "fixed_income_etf", "global", "Emerging Markets", "em_debt", "EM sovereign debt proxy", "USD"),
    MacroAsset("BNDX", "BNDX", "Vanguard Total International Bond ETF", "fixed_income_etf", "global", "World", "global_bonds_ex_us", "Global ex-US bonds", "USD"),
    MacroAsset("USO", "USO", "United States Oil Fund", "commodity_etf", "global", "World", "energy", "WTI oil ETF proxy", "USD"),
    MacroAsset("UNG", "UNG", "United States Natural Gas Fund", "commodity_etf", "global", "World", "energy", "Natural gas ETF proxy", "USD"),
    MacroAsset("DBB", "DBB", "Invesco DB Base Metals Fund", "commodity_etf", "global", "World", "industrial_metals", "Base metals ETF proxy", "USD"),
    MacroAsset("DBA", "DBA", "Invesco DB Agriculture Fund", "commodity_etf", "global", "World", "agriculture", "Agriculture ETF proxy", "USD"),
    MacroAsset("GSG", "GSG", "iShares S&P GSCI Commodity-Indexed Trust", "commodity_etf", "global", "World", "broad_commodities", "GSCI commodity basket", "USD"),
    MacroAsset("NATGAS", "NG=F", "Natural gas future", "commodity_future", "global", "World", "energy", "Natural gas futures proxy", "USD"),
    MacroAsset("PLATINUM", "PL=F", "Platinum future", "commodity_future", "global", "World", "precious_metals", "Platinum futures proxy", "USD"),
    MacroAsset("PALLADIUM", "PA=F", "Palladium future", "commodity_future", "global", "World", "precious_metals", "Palladium futures proxy", "USD"),
    MacroAsset("CORN", "ZC=F", "Corn future", "commodity_future", "global", "World", "agriculture", "Corn futures proxy", "USD"),
    MacroAsset("WHEAT", "ZW=F", "Wheat future", "commodity_future", "global", "World", "agriculture", "Wheat futures proxy", "USD"),
    MacroAsset("SOYBEANS", "ZS=F", "Soybean future", "commodity_future", "global", "World", "agriculture", "Soybean futures proxy", "USD"),
    MacroAsset("ADA", "ADA-USD", "Cardano", "crypto", "crypto", "Global", "crypto_alt", "Cardano price", "USD"),
    MacroAsset("DOGE", "DOGE-USD", "Dogecoin", "crypto", "crypto", "Global", "crypto_alt", "Dogecoin price", "USD"),
    MacroAsset("LINK", "LINK-USD", "Chainlink", "crypto", "crypto", "Global", "crypto_alt", "Chainlink price", "USD"),
    MacroAsset("AVAX", "AVAX-USD", "Avalanche", "crypto", "crypto", "Global", "crypto_alt", "Avalanche price", "USD"),
    MacroAsset("DOT", "DOT-USD", "Polkadot", "crypto", "crypto", "Global", "crypto_alt", "Polkadot price", "USD"),
)


def macro_asset_catalog(
    regions: Iterable[str] | None = None,
    asset_classes: Iterable[str] | None = None,
) -> pd.DataFrame:
    frame = pd.DataFrame([asdict(asset) for asset in MACRO_ASSETS])
    if regions:
        allowed = {str(item).lower() for item in regions}
        frame = frame[frame["region"].str.lower().isin(allowed)]
    if asset_classes:
        allowed = {str(item).lower() for item in asset_classes}
        frame = frame[frame["asset_class"].str.lower().isin(allowed)]
    return frame.sort_values(["region", "asset_class", "symbol"]).reset_index(drop=True)


def _safe_symbol(symbol: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(symbol))


def _normalize_history(raw: pd.DataFrame, asset: pd.Series) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    frame = raw.reset_index().copy()
    frame = frame.rename(
        columns={
            "Date": "date",
            "Datetime": "date",
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
    for col in ["date", "open", "high", "low", "close", "adjclose", "volume"]:
        if col not in frame.columns:
            frame[col] = pd.NA
    for col in ["symbol", "provider_symbol", "name", "asset_class", "region", "country", "category", "exposure", "currency"]:
        frame[col] = asset.get(col, "")
    return frame[["date", "symbol", "provider_symbol", "name", "asset_class", "region", "country", "category", "exposure", "currency", "open", "high", "low", "close", "adjclose", "volume"]]


def _returns(frame: pd.DataFrame) -> dict[str, float | None]:
    if frame.empty or "close" not in frame.columns:
        return {"return_1d": None, "return_1m": None, "return_3m": None, "return_1y": None}
    close = pd.to_numeric(frame["close"], errors="coerce").dropna()
    if close.empty:
        return {"return_1d": None, "return_1m": None, "return_3m": None, "return_1y": None}

    def ret(days: int) -> float | None:
        if len(close) <= days:
            return None
        base = close.iloc[-days - 1]
        if not base:
            return None
        return float(close.iloc[-1] / base - 1.0)

    return {"return_1d": ret(1), "return_1m": ret(21), "return_3m": ret(63), "return_1y": ret(252)}


def compile_macro_asset_database(
    financial_db_root: str | Path | None = None,
    output_root: str | Path | None = None,
    *,
    regions: Iterable[str] | None = None,
    asset_classes: Iterable[str] | None = None,
    period: str = "5y",
    interval: str = "1d",
    start: str | None = None,
    end: str | None = None,
    refresh: bool = False,
    max_assets: int | None = None,
) -> dict[str, Any]:
    """Download/write a broad macro asset database and lightweight manifests."""
    import yfinance as yf

    roots = resolve_data_platform_roots(financial_db_root=financial_db_root, repo_output_root=output_root)
    catalog = macro_asset_catalog(regions=regions, asset_classes=asset_classes)
    if max_assets:
        catalog = catalog.head(int(max_assets))
    db_root = roots.financial_db / "MarketData" / "Macro"
    table_root = roots.repo_output / "macro_market" / "tables"
    db_root.mkdir(parents=True, exist_ok=True)
    table_root.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(db_root / "MacroAssetCatalog.csv", index=False)
    catalog.to_csv(table_root / "MacroAssetCatalog.csv", index=False)

    manifest_rows: list[dict[str, Any]] = []
    latest_rows: list[dict[str, Any]] = []
    sample_frames: list[pd.DataFrame] = []
    for _, asset in catalog.iterrows():
        target = db_root / str(asset["asset_class"]) / f"{_safe_symbol(asset['symbol'])}.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and not refresh:
            try:
                history = pd.read_parquet(target)
            except Exception:
                history = pd.DataFrame()
        else:
            try:
                history_kwargs: dict[str, Any] = {"interval": interval, "auto_adjust": False}
                if start or end:
                    if start:
                        history_kwargs["start"] = start
                    if end:
                        history_kwargs["end"] = end
                else:
                    history_kwargs["period"] = period
                raw = yf.Ticker(str(asset["provider_symbol"])).history(**history_kwargs)
                history = _normalize_history(raw, asset)
                if not history.empty:
                    history.to_parquet(target, index=False)
            except Exception as exc:
                history = pd.DataFrame()
                manifest_rows.append(
                    {
                        **asset.to_dict(),
                        "status": "FAILED",
                        "error": f"{type(exc).__name__}: {exc}",
                        "rows": 0,
                        "target_path": str(target),
                        "updated_at": utc_now(),
                    }
                )
                continue
        if history.empty:
            status = "NO_DATA"
            last_date = ""
            last_close = None
        else:
            status = "OK"
            history_dates = pd.to_datetime(history["date"], errors="coerce")
            last_date = history_dates.max().date().isoformat() if history_dates.notna().any() else ""
            last_close = pd.to_numeric(history["close"], errors="coerce").dropna().iloc[-1] if pd.to_numeric(history["close"], errors="coerce").dropna().size else None
            sample_frames.append(history.tail(260))
            latest_rows.append({**asset.to_dict(), "last_date": last_date, "last_close": last_close, **_returns(history)})
        manifest_rows.append(
            {
                **asset.to_dict(),
                "status": status,
                "rows": len(history),
                "last_date": last_date,
                "target_path": str(target),
                "updated_at": utc_now(),
            }
        )
    manifest = pd.DataFrame(manifest_rows)
    latest = pd.DataFrame(latest_rows)
    history_sample = pd.concat(sample_frames, ignore_index=True, sort=False) if sample_frames else pd.DataFrame()
    manifest.to_csv(db_root / "MacroAssetManifest.csv", index=False)
    manifest.to_csv(table_root / "MacroAssetManifest.csv", index=False)
    latest.to_csv(table_root / "MacroLatestSnapshot.csv", index=False)
    history_sample.to_csv(table_root / "MacroHistorySample.csv", index=False)
    summary = {
        "status": "OK" if not manifest.empty and manifest["status"].astype(str).eq("OK").any() else "NO_DATA",
        "asset_count": int(len(catalog)),
        "ok_count": int(manifest["status"].astype(str).eq("OK").sum()) if not manifest.empty else 0,
        "manifest_path": str(table_root / "MacroAssetManifest.csv"),
        "catalog_path": str(table_root / "MacroAssetCatalog.csv"),
        "updated_at": utc_now(),
    }
    (table_root / "MacroAssetSummary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"catalog": catalog, "manifest": manifest, "latest": latest, "history_sample": history_sample, "summary": summary}


def load_macro_market_artifacts(
    financial_db_root: str | Path | None = None,
    output_root: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    roots = resolve_data_platform_roots(financial_db_root=financial_db_root, repo_output_root=output_root)
    table_root = roots.repo_output / "macro_market" / "tables"
    db_root = roots.financial_db / "MarketData" / "Macro"

    def read_csv(path: Path) -> pd.DataFrame:
        if not path.exists() or path.stat().st_size <= 1:
            return pd.DataFrame()
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()

    catalog = read_csv(table_root / "MacroAssetCatalog.csv")
    if catalog.empty:
        catalog = read_csv(db_root / "MacroAssetCatalog.csv")
    if catalog.empty:
        catalog = macro_asset_catalog()
    return {
        "catalog": catalog,
        "manifest": read_csv(table_root / "MacroAssetManifest.csv") if (table_root / "MacroAssetManifest.csv").exists() else read_csv(db_root / "MacroAssetManifest.csv"),
        "latest": read_csv(table_root / "MacroLatestSnapshot.csv"),
        "history_sample": read_csv(table_root / "MacroHistorySample.csv"),
    }
