"""Import local Kaggle market-data dumps as static OHLCV seeds.

The loader intentionally works from files already present in the local data
lake. It does not call Kaggle APIs, because dataset slugs and downloaded file
layouts vary by user and over time. Configure or place downloaded CSV/TXT/ZIP
files under ``Database Finanziario/Kaggle`` and this module normalizes them into
the existing OHLCV database tables.
"""

from __future__ import annotations

import logging
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from ..data_platform import resolve_data_platform_roots, utc_now
from ..ohlcv_store import ASSET_COLUMNS, OhlcvDatabase


LOGGER = logging.getLogger(__name__)


DEFAULT_KAGGLE_DATASETS: dict[str, dict[str, Any]] = {
    "us_stocks_etfs": {
        "slug": "borismarjanovic/price-volume-data-for-all-us-stocks-etfs",
        "path": "price-volume-data-for-all-us-stocks-etfs",
        "patterns": ["Stocks/*.txt", "ETFs/*.txt", "stocks/*.csv", "etfs/*.csv", "*.zip"],
        "frequency": "daily",
        "asset_type": "infer",
        "country": "US",
        "exchange": "US",
        "source": "kaggle_seed_stocks",
        "source_by_type": {"stock": "kaggle_seed_stocks", "etf": "kaggle_seed_etf"},
        "primary_source": "kaggle_seed_us_stocks_etfs",
    },
    "us_stocks_meta": {
        "slug": "jacksoncrow/stock-market-dataset",
        "path": "stock-market-dataset",
        "patterns": ["stocks/*.csv", "etfs/*.csv", "*.zip"],
        "frequency": "daily",
        "asset_type": "infer",
        "country": "US",
        "exchange": "US",
        "source": "kaggle_seed_stocks",
        "source_by_type": {"stock": "kaggle_seed_stocks", "etf": "kaggle_seed_etf"},
        "primary_source": "kaggle_seed_us_stocks_meta",
    },
    "historical_etf_data": {
        "slug": "stefanoleone992/mutual-funds-and-etfs",
        "path": "historical-etf-data",
        "patterns": ["*.csv", "**/*.csv", "*.zip"],
        "frequency": "daily",
        "asset_type": "etf",
        "country": "US",
        "exchange": "US",
        "source": "kaggle_seed_etf",
        "primary_source": "kaggle_seed_historical_etf",
    },
    "world_stock_prices": {
        "slug": "nelgiriyewithana/world-stock-prices-daily-updating",
        "path": "world-stock-prices-daily-updating",
        "patterns": ["*.csv", "**/*.csv", "*.zip"],
        "frequency": "daily",
        "asset_type": "stock",
        "country": "GLOBAL",
        "exchange": "GLOBAL",
        "source": "kaggle_seed_stocks",
        "primary_source": "kaggle_seed_world_stock_prices",
    },
    "crypto_daily": {
        "slug": "sudalairajkumar/cryptocurrencypricehistory",
        "path": "cryptocurrencypricehistory",
        "patterns": ["*.csv", "**/*.csv", "*.zip"],
        "frequency": "daily",
        "asset_type": "crypto",
        "country": "GLOBAL",
        "exchange": "CRYPTO",
        "source": "kaggle_seed_crypto_daily",
        "primary_source": "kaggle_seed_crypto_daily",
    },
    "crypto_5m": {
        "slug": "varpit94/bitcoin-data-updated-till-26jun2021",
        "path": "crypto-5m",
        "patterns": ["*.csv", "**/*.csv", "*.zip"],
        "frequency": "5m",
        "asset_type": "crypto",
        "country": "GLOBAL",
        "exchange": "CRYPTO",
        "source": "kaggle_seed_crypto_5m",
        "primary_source": "kaggle_seed_crypto_5m",
    },
    "btcusdt_5m": {
        "slug": "kaggle-local/btcusdt-5-minute-ohlcv",
        "path": "btcusdt-5-minute-ohlcv",
        "patterns": ["*.csv", "**/*.csv", "*.zip"],
        "frequency": "5m",
        "asset_type": "crypto",
        "country": "GLOBAL",
        "exchange": "BINANCE",
        "source": "kaggle_seed_crypto_5m",
        "primary_source": "kaggle_seed_btcusdt_5m",
        "ticker": "BTCUSDT",
    },
}


@dataclass(frozen=True)
class KaggleDatasetConfig:
    name: str
    slug: str = ""
    path: str = ""
    patterns: list[str] = field(default_factory=lambda: ["*.csv", "**/*.csv", "*.txt", "**/*.txt", "*.zip"])
    frequency: str = "daily"
    asset_type: str = "stock"
    country: str = "GLOBAL"
    exchange: str = "GLOBAL"
    source: str = "kaggle_seed_stocks"
    primary_source: str = ""
    enabled: bool = True
    ticker: str = ""
    source_by_type: dict[str, str] = field(default_factory=dict)


def _load_yaml_config() -> dict[str, Any]:
    config_path = Path(__file__).resolve().parents[3] / "config" / "ohlcv_data_sources.yaml"
    if not config_path.exists():
        return {}
    try:
        import yaml

        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _dataset_configs(config: dict[str, Any] | None = None) -> dict[str, KaggleDatasetConfig]:
    raw = dict(DEFAULT_KAGGLE_DATASETS)
    configured = (((config or _load_yaml_config()).get("ohlcv", {}) or {}).get("kaggle", {}) or {}).get("datasets", {})
    for name, values in configured.items():
        if isinstance(values, dict):
            base = raw.get(name, {})
            raw[name] = {**base, **values}
    out = {}
    for name, values in raw.items():
        values = dict(values)
        values.setdefault("name", name)
        values.setdefault("primary_source", f"kaggle_seed_{name}")
        out[name] = KaggleDatasetConfig(**{k: v for k, v in values.items() if k in KaggleDatasetConfig.__dataclass_fields__})
    return out


def _normalized_col(name: str) -> str:
    return "".join(ch for ch in str(name).strip().lower() if ch.isalnum())


def _find_column(columns: Iterable[str], aliases: Iterable[str]) -> str | None:
    by_name = {_normalized_col(col): col for col in columns}
    for alias in aliases:
        found = by_name.get(_normalized_col(alias))
        if found is not None:
            return found
    return None


def _infer_symbol_from_path(path: Path, fallback: str = "") -> str:
    stem = path.name
    for suffix in [".csv", ".txt", ".parquet"]:
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
    if stem.lower().endswith(".us"):
        stem = stem[:-3]
    for token in ["_prices", "-prices", "_ohlcv", "-ohlcv", "_5m", "-5m"]:
        stem = stem.replace(token, "")
    return str(stem or fallback).strip().upper().replace(".", "-")


def _infer_asset_type(path: Path, dataset: KaggleDatasetConfig, chunk: pd.DataFrame | None = None, meta: dict[str, Any] | None = None) -> str:
    if dataset.asset_type != "infer":
        return dataset.asset_type
    lowered = "/".join(part.lower() for part in path.parts)
    if "etf" in lowered:
        return "etf"
    if "crypto" in lowered or "btc" in lowered:
        return "crypto"
    if meta and str(meta.get("type", "")).lower() in {"stock", "etf", "index", "crypto"}:
        return str(meta["type"]).lower()
    if chunk is not None:
        etf_col = _find_column(chunk.columns, ["ETF", "is_etf"])
        if etf_col and str(chunk[etf_col].dropna().astype(str).head(1).squeeze()).upper() == "Y":
            return "etf"
    return "stock"


def _parse_datetime(value: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(value, errors="coerce")
    if numeric.notna().any():
        median = numeric.dropna().median()
        if 19000101 <= median <= 21001231:
            parsed_numeric_date = pd.to_datetime(numeric.dropna().astype("Int64").astype(str), format="%Y%m%d", errors="coerce", utc=True)
            out = pd.Series(pd.NaT, index=value.index, dtype="datetime64[ns, UTC]")
            out.loc[numeric.dropna().index] = parsed_numeric_date
            return out
        if median > 1_000_000_000:
            unit = "ms" if median > 10_000_000_000 else "s"
            return pd.to_datetime(numeric, errors="coerce", utc=True, unit=unit)
    return pd.to_datetime(value, errors="coerce", utc=True)


def _is_price_file(path: Path) -> bool:
    lowered = str(path).lower()
    if not lowered.endswith((".csv", ".txt", ".zip", ".parquet")):
        return False
    skip_tokens = ["symbols_valid_meta", "metadata", "readme", "license", "sample_submission"]
    return not any(token in lowered for token in skip_tokens)


class KaggleSeedLoader:
    """Normalize and import Kaggle seed datasets into the OHLCV store."""

    def __init__(
        self,
        financial_db_root: Path | str | None = None,
        output_root: Path | str | None = None,
        database_url: str | None = None,
        kaggle_root: Path | str | None = None,
        chunk_size: int = 200_000,
    ):
        roots = resolve_data_platform_roots(financial_db_root=financial_db_root, repo_output_root=output_root)
        self.financial_db_root = roots.financial_db
        self.output_root = roots.repo_output
        config_root = (((_load_yaml_config().get("ohlcv", {}) or {}).get("kaggle", {}) or {}).get("root", "Kaggle"))
        self.kaggle_root = Path(kaggle_root).expanduser() if kaggle_root else self.financial_db_root / str(config_root)
        self.database = OhlcvDatabase(database_url=database_url, financial_db_root=roots.financial_db)
        self.chunk_size = int(chunk_size)
        self.manifest_dir = self.financial_db_root / "MarketData" / "OHLCV" / "manifests"
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        (self.output_root / "tables").mkdir(parents=True, exist_ok=True)

    def _candidate_roots(self, dataset: KaggleDatasetConfig) -> list[Path]:
        candidates = []
        for value in [dataset.path, dataset.name, dataset.slug.split("/")[-1] if dataset.slug else ""]:
            if not value:
                continue
            path = Path(value).expanduser()
            candidates.append(path if path.is_absolute() else self.kaggle_root / path)
        return list(dict.fromkeys(candidates))

    def _dataset_root(self, dataset: KaggleDatasetConfig) -> Path | None:
        for candidate in self._candidate_roots(dataset):
            if candidate.exists():
                return candidate
        return None

    def discover_files(self, dataset: KaggleDatasetConfig) -> list[Path]:
        root = self._dataset_root(dataset)
        if root is None:
            return []
        if root.is_file():
            return [root] if _is_price_file(root) else []
        files: list[Path] = []
        for pattern in dataset.patterns:
            files.extend(path for path in root.glob(pattern) if path.is_file() and _is_price_file(path))
        if not files:
            files = [path for path in root.rglob("*") if path.is_file() and _is_price_file(path)]
        return sorted(dict.fromkeys(files))

    def _load_symbol_meta(self, root: Path | None) -> dict[str, dict[str, Any]]:
        if root is None or root.is_file():
            return {}
        candidates = list(root.rglob("symbols_valid_meta.csv")) + list(root.rglob("*meta*.csv"))
        if not candidates:
            return {}
        try:
            meta = pd.read_csv(candidates[0], low_memory=False)
        except Exception:
            return {}
        symbol_col = _find_column(meta.columns, ["Symbol", "ticker", "Ticker"])
        if not symbol_col:
            return {}
        name_col = _find_column(meta.columns, ["Security Name", "name", "Name"])
        exchange_col = _find_column(meta.columns, ["Listing Exchange", "exchange", "Exchange"])
        etf_col = _find_column(meta.columns, ["ETF", "is_etf"])
        out = {}
        for _, item in meta.iterrows():
            symbol = str(item.get(symbol_col, "")).strip().upper().replace(".", "-")
            if not symbol:
                continue
            asset_type = "etf" if etf_col and str(item.get(etf_col, "")).upper() == "Y" else "stock"
            out[symbol] = {
                "name": item.get(name_col, symbol) if name_col else symbol,
                "exchange": item.get(exchange_col, "") if exchange_col else "",
                "type": asset_type,
            }
        return out

    def _iter_frames(self, path: Path):
        if path.suffix.lower() == ".parquet":
            yield path, pd.read_parquet(path)
            return
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as archive:
                for member in archive.namelist():
                    member_path = Path(member)
                    if not _is_price_file(member_path):
                        continue
                    with archive.open(member) as handle:
                        for chunk in pd.read_csv(handle, chunksize=self.chunk_size, low_memory=False):
                            yield member_path, chunk
            return
        for chunk in pd.read_csv(path, chunksize=self.chunk_size, low_memory=False):
            yield path, chunk

    def _ticker_series(self, frame: pd.DataFrame, path: Path, dataset: KaggleDatasetConfig) -> pd.Series:
        ticker_col = _find_column(frame.columns, ["ticker", "symbol", "Symbol", "pair", "Pair", "asset", "Asset"])
        if ticker_col:
            return frame[ticker_col].astype(str).str.strip().str.upper().str.replace(".", "-", regex=False)
        inferred = dataset.ticker or _infer_symbol_from_path(path, dataset.name)
        return pd.Series([inferred] * len(frame), index=frame.index)

    def _asset_frame(self, frame: pd.DataFrame, path: Path, dataset: KaggleDatasetConfig, meta_map: dict[str, dict[str, Any]]) -> pd.DataFrame:
        tickers = self._ticker_series(frame, path, dataset)
        rows = []
        for ticker in sorted(set(tickers.dropna().astype(str))):
            meta = meta_map.get(ticker, {})
            asset_type = _infer_asset_type(path, dataset, frame, meta)
            exchange_col = _find_column(frame.columns, ["exchange", "Exchange", "market", "Market"])
            country_col = _find_column(frame.columns, ["country", "Country"])
            name_col = _find_column(frame.columns, ["name", "Name", "Security Name"])
            rows.append(
                {
                    "ticker": ticker,
                    "provider_symbol": ticker,
                    "exchange": str(frame[exchange_col].dropna().iloc[0]) if exchange_col and frame[exchange_col].notna().any() else str(meta.get("exchange") or dataset.exchange),
                    "country": str(frame[country_col].dropna().iloc[0]) if country_col and frame[country_col].notna().any() else dataset.country,
                    "type": asset_type,
                    "name": str(frame[name_col].dropna().iloc[0]) if name_col and frame[name_col].notna().any() else str(meta.get("name") or ticker),
                    "primary_source": dataset.primary_source or f"kaggle_seed_{dataset.name}",
                    "active_flag": True,
                }
            )
        return pd.DataFrame(rows, columns=ASSET_COLUMNS)

    def _price_frame(self, frame: pd.DataFrame, path: Path, dataset: KaggleDatasetConfig, assets: pd.DataFrame) -> pd.DataFrame:
        date_col = _find_column(frame.columns, ["date", "Date", "datetime", "Datetime", "timestamp", "Timestamp", "time", "Open time", "open_time"])
        open_col = _find_column(frame.columns, ["open", "Open"])
        high_col = _find_column(frame.columns, ["high", "High"])
        low_col = _find_column(frame.columns, ["low", "Low"])
        close_col = _find_column(frame.columns, ["close", "Close", "price", "Price"])
        adj_col = _find_column(frame.columns, ["adj close", "Adj Close", "adjusted close", "Adjusted Close", "adjclose"])
        volume_col = _find_column(frame.columns, ["volume", "Volume", "volumefrom", "Volume USD", "Volume_(Currency)", "Volume BTC"])
        required = [date_col, open_col, high_col, low_col, close_col]
        if any(col is None for col in required):
            return pd.DataFrame()
        out = pd.DataFrame(index=frame.index)
        tickers = self._ticker_series(frame, path, dataset)
        asset_lookup = assets.set_index("ticker")[["exchange", "primary_source", "type"]].to_dict("index")
        out["ticker"] = tickers
        out["exchange"] = out["ticker"].map(lambda ticker: asset_lookup.get(str(ticker), {}).get("exchange", dataset.exchange))
        out["primary_source"] = out["ticker"].map(lambda ticker: asset_lookup.get(str(ticker), {}).get("primary_source", dataset.primary_source))
        out["open"] = pd.to_numeric(frame[open_col], errors="coerce")
        out["high"] = pd.to_numeric(frame[high_col], errors="coerce")
        out["low"] = pd.to_numeric(frame[low_col], errors="coerce")
        out["close"] = pd.to_numeric(frame[close_col], errors="coerce")
        out["volume"] = pd.to_numeric(frame[volume_col], errors="coerce") if volume_col else 0
        source = dataset.source
        if dataset.source_by_type:
            out["source"] = out["ticker"].map(lambda ticker: dataset.source_by_type.get(asset_lookup.get(str(ticker), {}).get("type", ""), source))
        else:
            out["source"] = source
        out["ingestion_ts"] = utc_now()
        parsed_time = _parse_datetime(frame[date_col])
        if dataset.frequency == "5m":
            out["ts"] = parsed_time
            return out.dropna(subset=["ticker", "ts", "close"]).reset_index(drop=True)
        out["date"] = parsed_time.dt.date.astype("string")
        out["adjclose"] = pd.to_numeric(frame[adj_col], errors="coerce") if adj_col else out["close"]
        return out.dropna(subset=["ticker", "date", "close"]).reset_index(drop=True)

    def import_dataset(
        self,
        dataset: KaggleDatasetConfig,
        dry_run: bool = True,
        limit_files: int | None = None,
    ) -> dict[str, Any]:
        root = self._dataset_root(dataset)
        if root is None:
            return {
                "dataset_name": dataset.name,
                "slug": dataset.slug,
                "status": "missing_root",
                "root_path": str(self._candidate_roots(dataset)[0]) if self._candidate_roots(dataset) else "",
                "num_assets": 0,
                "num_rows": 0,
                "last_data_date": None,
                "last_ingestion_ts": utc_now(),
            }
        files = self.discover_files(dataset)
        if limit_files:
            files = files[: int(limit_files)]
        meta_map = self._load_symbol_meta(root)
        assets_seen: set[str] = set()
        rows_inserted = 0
        last_data_date: str | None = None
        status = "dry_run" if dry_run else "imported"
        for path in files:
            try:
                for virtual_path, frame in self._iter_frames(path):
                    if frame.empty:
                        continue
                    assets = self._asset_frame(frame, virtual_path, dataset, meta_map)
                    prices = self._price_frame(frame, virtual_path, dataset, assets)
                    if prices.empty:
                        continue
                    assets_seen.update(assets["ticker"].dropna().astype(str).tolist())
                    if dataset.frequency == "5m":
                        max_ts = pd.to_datetime(prices["ts"], errors="coerce").max()
                        if pd.notna(max_ts):
                            last_data_date = max(str(max_ts.date()), last_data_date or str(max_ts.date()))
                    else:
                        max_date = prices["date"].dropna().max()
                        if pd.notna(max_date):
                            last_data_date = max(str(max_date), last_data_date or str(max_date))
                    if dry_run:
                        rows_inserted += int(len(prices))
                        continue
                    asset_map = self.database.upsert_assets(assets)
                    prices["asset_id"] = [
                        asset_map.get((row.ticker, row.exchange, row.primary_source))
                        for row in prices[["ticker", "exchange", "primary_source"]].itertuples(index=False)
                    ]
                    if dataset.frequency == "5m":
                        rows_inserted += self.database.upsert_intraday_5m(
                            prices[["asset_id", "ts", "open", "high", "low", "close", "volume", "source", "ingestion_ts"]].dropna(subset=["asset_id"])
                        )
                    else:
                        rows_inserted += self.database.upsert_daily_prices(
                            prices[["asset_id", "date", "open", "high", "low", "close", "adjclose", "volume", "source", "ingestion_ts"]].dropna(subset=["asset_id"])
                        )
            except Exception as exc:
                LOGGER.warning("Kaggle seed file skipped dataset=%s path=%s error=%s", dataset.name, path, exc)
                status = "partial_error" if status != "dry_run" else "dry_run_partial_error"
        return {
            "dataset_name": dataset.name,
            "slug": dataset.slug,
            "status": status,
            "frequency": dataset.frequency,
            "root_path": str(root),
            "num_files": len(files),
            "num_assets": len(assets_seen),
            "num_rows": rows_inserted,
            "last_data_date": last_data_date,
            "last_ingestion_ts": utc_now(),
        }

    def import_seeds(
        self,
        datasets: list[str] | None = None,
        dry_run: bool = True,
        limit_files: int | None = None,
    ) -> pd.DataFrame:
        configs = _dataset_configs()
        selected = datasets or [name for name, cfg in configs.items() if cfg.enabled]
        rows = []
        for name in selected:
            cfg = configs.get(name)
            if cfg is None:
                rows.append({"dataset_name": name, "status": "unknown_dataset", "num_assets": 0, "num_rows": 0, "last_ingestion_ts": utc_now()})
                continue
            rows.append(self.import_dataset(cfg, dry_run=dry_run, limit_files=limit_files))
        manifest = pd.DataFrame(rows)
        manifest.to_csv(self.manifest_dir / "kaggle_seed_manifest.csv", index=False)
        manifest.to_csv(self.output_root / "tables" / "Kaggle_seed_manifest.csv", index=False)
        return manifest


def import_kaggle_seeds(
    datasets: list[str] | None = None,
    kaggle_root: Path | str | None = None,
    financial_db_root: Path | str | None = None,
    output_root: Path | str | None = None,
    database_url: str | None = None,
    dry_run: bool = True,
    limit_files: int | None = None,
    chunk_size: int = 200_000,
) -> pd.DataFrame:
    """Import configured local Kaggle seed files and write a manifest."""
    loader = KaggleSeedLoader(
        financial_db_root=financial_db_root,
        output_root=output_root,
        database_url=database_url,
        kaggle_root=kaggle_root,
        chunk_size=chunk_size,
    )
    return loader.import_seeds(datasets=datasets, dry_run=dry_run, limit_files=limit_files)
