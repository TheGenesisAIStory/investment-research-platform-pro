"""Historical and incremental OHLCV ingestion jobs."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd

from .data_platform import resolve_data_platform_roots, utc_now
from .loaders.market_universe import MarketUniverseBuilder
from .loaders.ohlcv_client import OhlcvClient
from .ohlcv_store import OhlcvDatabase


LOGGER = logging.getLogger(__name__)


def _chunks(values: list[Any], size: int) -> list[list[Any]]:
    size = max(int(size), 1)
    return [values[idx : idx + size] for idx in range(0, len(values), size)]


def _next_day(date_value: str | None, fallback: str) -> str:
    if not date_value or str(date_value).lower() == "nan":
        return fallback
    return (pd.to_datetime(date_value) + pd.Timedelta(days=1)).date().isoformat()


def _safe_exchange(value: Any) -> str:
    text = str(value or "unknown").strip().replace("/", "_").replace(" ", "_")
    return text or "unknown"


class OhlcvIngestJob:
    """Coordinate universe discovery, provider calls, DB upserts and manifests."""

    def __init__(
        self,
        financial_db_root: Path | str | None = None,
        output_root: Path | str | None = None,
        database_url: str | None = None,
        client: OhlcvClient | None = None,
        database: OhlcvDatabase | None = None,
    ):
        roots = resolve_data_platform_roots(financial_db_root=financial_db_root, repo_output_root=output_root)
        self.financial_db_root = roots.financial_db
        self.output_root = roots.repo_output
        self.client = client or OhlcvClient(roots.financial_db, roots.repo_output)
        self.database = database or OhlcvDatabase(database_url=database_url, financial_db_root=roots.financial_db)
        self.universe_builder = MarketUniverseBuilder(roots.financial_db, roots.repo_output)
        self.base_dir = roots.financial_db / "MarketData" / "OHLCV"
        self.daily_dir = self.base_dir / "daily"
        self.intraday_dir = self.base_dir / "intraday_5m"
        self.manifest_dir = self.base_dir / "manifests"
        for path in [self.daily_dir, self.intraday_dir, self.manifest_dir, roots.repo_output / "logs"]:
            path.mkdir(parents=True, exist_ok=True)

    def build_assets(self, markets: list[str] | None = None, max_assets: int | None = None, include_etfs: bool = True) -> pd.DataFrame:
        assets = self.universe_builder.build_universe(markets=markets, include_etfs=include_etfs)
        if max_assets:
            assets = assets.head(max_assets).copy()
        if not assets.empty:
            assets.to_csv(self.base_dir / "asset_master_candidates.csv", index=False)
        return assets

    def _attach_asset_ids(self, assets: pd.DataFrame) -> pd.DataFrame:
        asset_map = self.database.upsert_assets(assets)
        rows = assets.copy()
        rows["asset_id"] = [
            asset_map.get((row.ticker, row.exchange, row.primary_source))
            for row in rows[["ticker", "exchange", "primary_source"]].itertuples(index=False)
        ]
        return rows.dropna(subset=["asset_id"]).reset_index(drop=True)

    def _write_daily_file(self, asset: pd.Series, frame: pd.DataFrame) -> Path:
        target = self.daily_dir / _safe_exchange(asset.get("exchange")) / f"{str(asset['ticker']).replace('/', '_')}.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        out = frame.copy()
        if target.exists():
            try:
                old = pd.read_parquet(target)
                out = pd.concat([old, out], ignore_index=True, sort=False)
                out = out.drop_duplicates(["date"], keep="last").sort_values("date")
            except Exception:
                pass
        out.to_parquet(target, index=False)
        return target

    def _write_intraday_file(self, asset: pd.Series, frame: pd.DataFrame) -> Path:
        target = self.intraday_dir / _safe_exchange(asset.get("exchange")) / f"{str(asset['ticker']).replace('/', '_')}.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        out = frame.copy()
        if target.exists():
            try:
                old = pd.read_parquet(target)
                out = pd.concat([old, out], ignore_index=True, sort=False)
                out = out.drop_duplicates(["ts"], keep="last").sort_values("ts")
            except Exception:
                pass
        out.to_parquet(target, index=False)
        return target

    def run_daily(
        self,
        markets: list[str] | None = None,
        start_date: str = "2000-01-01",
        end_date: str | None = None,
        mode: str = "incremental",
        max_assets: int | None = None,
        batch_size: int = 80,
        include_etfs: bool = True,
        dry_run: bool = False,
        preferred_providers: list[str] | None = None,
    ) -> pd.DataFrame:
        """Run full-history or incremental daily OHLCV ingestion."""
        started = time.time()
        assets = self.build_assets(markets=markets, max_assets=max_assets, include_etfs=include_etfs)
        if assets.empty:
            manifest = pd.DataFrame([{"status": "no_assets", "updated_at": utc_now()}])
            manifest.to_csv(self.manifest_dir / "ohlcv_daily_manifest.csv", index=False)
            return manifest
        if dry_run:
            manifest = assets.assign(status="dry_run", rows=0)
            manifest.to_csv(self.manifest_dir / "ohlcv_daily_manifest.csv", index=False)
            return manifest
        assets = self._attach_asset_ids(assets)
        latest = {} if mode == "full" else self.database.latest_daily_dates(assets["asset_id"].astype(int).tolist())
        latest_by_symbol = {} if mode == "full" else self.database.latest_daily_dates_by_provider_symbol(assets["provider_symbol"].astype(str).tolist())
        starts = []
        for row in assets[["asset_id", "provider_symbol"]].itertuples(index=False):
            if mode == "full":
                starts.append(start_date)
                continue
            asset_latest = latest.get(int(row.asset_id))
            symbol_latest = latest_by_symbol.get(str(row.provider_symbol))
            candidates = [value for value in [asset_latest, symbol_latest] if value and str(value).lower() != "nan"]
            max_latest = max(candidates) if candidates else None
            starts.append(_next_day(max_latest, start_date))
        assets["start_for_job"] = starts
        today = pd.Timestamp.utcnow().date()
        if end_date:
            today = min(today, pd.to_datetime(end_date).date())
        assets = assets[pd.to_datetime(assets["start_for_job"]).dt.date.le(today)].copy()
        if assets.empty:
            manifest = pd.DataFrame([{"status": "already_current", "updated_at": utc_now()}])
            manifest.to_csv(self.manifest_dir / "ohlcv_daily_manifest.csv", index=False)
            return manifest
        rows: list[dict[str, Any]] = []
        grouped = assets.groupby("start_for_job", dropna=False)
        for group_start, group in grouped:
            group_assets = group.to_dict("records")
            for batch_no, batch in enumerate(_chunks(group_assets, batch_size), start=1):
                symbols = [row["provider_symbol"] for row in batch]
                batch_map = {row["provider_symbol"]: row for row in batch}
                try:
                    batch_frames, bulk_meta = self.client.fetch_daily_bulk(
                        symbols,
                        start=str(group_start),
                        end=end_date,
                        assets=batch,
                        preferred_providers=preferred_providers,
                    )
                except Exception as exc:
                    batch_frames = {}
                    bulk_meta = {"provider": None}
                    rows.append({"ticker": "", "provider_symbol": ",".join(symbols[:10]), "status": "bulk_failed", "provider": "policy_engine", "rows": 0, "error": str(exc), "start": group_start, "batch": batch_no})
                if not batch_frames:
                    rows.append({"ticker": "", "provider_symbol": ",".join(symbols[:10]), "status": "bulk_empty", "provider": bulk_meta.get("provider"), "rows": 0, "start": group_start, "batch": batch_no})
                missing = [symbol for symbol in symbols if symbol not in batch_frames]
                for symbol in missing:
                    try:
                        frame, meta = self.client.fetch_daily_one(
                            symbol,
                            str(group_start),
                            end_date,
                            asset=batch_map[symbol],
                            preferred_providers=preferred_providers,
                        )
                        if not frame.empty:
                            batch_frames[symbol] = frame
                        rows.append({"ticker": batch_map[symbol]["ticker"], "provider_symbol": symbol, "status": "fallback", "provider": meta.get("provider"), "rows": len(frame), "start": group_start})
                    except Exception as exc:
                        rows.append({"ticker": batch_map[symbol]["ticker"], "provider_symbol": symbol, "status": "failed", "error": str(exc), "start": group_start})
                for symbol, frame in batch_frames.items():
                    asset = pd.Series(batch_map[symbol])
                    asset_id = int(asset["asset_id"])
                    clean = frame.copy()
                    clean["asset_id"] = asset_id
                    clean["ingestion_ts"] = utc_now()
                    inserted = self.database.upsert_daily_prices(clean)
                    path = self._write_daily_file(asset, clean)
                    rows.append(
                        {
                            "ticker": asset["ticker"],
                            "provider_symbol": symbol,
                            "asset_id": asset_id,
                            "exchange": asset.get("exchange", ""),
                            "status": "downloaded",
                            "provider": clean["source"].dropna().iloc[-1] if "source" in clean and not clean.empty else "unknown",
                            "rows": inserted,
                            "start": group_start,
                            "end": end_date or today.isoformat(),
                            "target_path": str(path),
                            "batch": batch_no,
                        }
                    )
        manifest = pd.DataFrame(rows)
        manifest["duration_seconds"] = round(time.time() - started, 2)
        manifest["updated_at"] = utc_now()
        manifest.to_csv(self.manifest_dir / "ohlcv_daily_manifest.csv", index=False)
        (self.output_root / "tables").mkdir(parents=True, exist_ok=True)
        manifest.to_csv(self.output_root / "tables" / "OHLCV_daily_manifest.csv", index=False)
        return manifest

    def run_intraday_5m(
        self,
        markets: list[str] | None = None,
        month: str | None = None,
        max_assets: int | None = 25,
        include_etfs: bool = True,
        dry_run: bool = False,
        preferred_providers: list[str] | None = None,
    ) -> pd.DataFrame:
        """Run policy-selected 5m ingestion for a small configured slice."""
        assets = self.build_assets(markets=markets, max_assets=max_assets, include_etfs=include_etfs)
        if assets.empty or dry_run:
            manifest = assets.assign(status="dry_run" if dry_run else "no_assets", rows=0)
            manifest.to_csv(self.manifest_dir / "ohlcv_intraday_5m_manifest.csv", index=False)
            return manifest
        assets = self._attach_asset_ids(assets)
        rows: list[dict[str, Any]] = []
        for row in assets.to_dict("records"):
            symbol = row["provider_symbol"]
            try:
                frame, meta = self.client.fetch_intraday_5m_one(symbol, month=month, asset=row, preferred_providers=preferred_providers)
                frame["asset_id"] = int(row["asset_id"])
                inserted = self.database.upsert_intraday_5m(frame)
                path = self._write_intraday_file(pd.Series(row), frame)
                rows.append({"ticker": row["ticker"], "provider_symbol": symbol, "asset_id": row["asset_id"], "status": "downloaded", "provider": meta.get("provider"), "rows": inserted, "target_path": str(path)})
            except Exception as exc:
                rows.append({"ticker": row["ticker"], "provider_symbol": symbol, "asset_id": row["asset_id"], "status": "failed", "error": str(exc)})
        manifest = pd.DataFrame(rows)
        manifest["updated_at"] = utc_now()
        manifest.to_csv(self.manifest_dir / "ohlcv_intraday_5m_manifest.csv", index=False)
        return manifest


def summarize_ohlcv_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    if manifest.empty or "status" not in manifest:
        return pd.DataFrame()
    out = manifest.groupby("status", dropna=False).agg(symbols=("ticker", "count"), rows=("rows", "sum")).reset_index()
    if "duration_seconds" in manifest:
        out["duration_seconds"] = manifest["duration_seconds"].max()
    return out
