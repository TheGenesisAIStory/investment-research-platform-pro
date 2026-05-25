"""Historical and incremental OHLCV ingestion jobs."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any

import pandas as pd

from .data_platform import get_ohlcv_parquet_root_info, resolve_data_platform_roots, utc_now
from .loaders.market_universe import MarketUniverseBuilder
from .loaders.ohlcv_client import (
    OhlcvClient,
    classify_history_window,
    classify_provider_error,
    is_low_priority_structured_symbol,
    is_variant_sensitive_symbol,
)
from .ohlcv_store import OhlcvDatabase


LOGGER = logging.getLogger(__name__)

INVALID_PROVIDER_SYMBOL_WORDS = {
    "ABOUT",
    "ADDRESS",
    "CONTACT",
    "DESCRIPTION",
    "FOUNDATION",
    "HOMEPAGE",
    "HTTP",
    "HTTPS",
    "INVESTOR",
    "CONSTITUENTS",
    "EXCHANGES",
    "NAN",
    "OPERATOR",
    "RELATIONS",
    "WEBSITE",
    "WWW",
}

_PROVIDER_SYMBOL_PATTERN = re.compile(r"^[A-Za-z0-9.^=$_\-]+$")


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


def _manifest_history_fields(frame: pd.DataFrame, start: str | None, end: str | None, error: str | Exception | None = None) -> dict[str, Any]:
    return classify_history_window(frame, start=start, end=end, error=error)


def _status_from_history(history: dict[str, Any], base_status: str = "downloaded") -> str:
    coverage_status = str(history.get("coverage_status") or "").upper()
    if coverage_status == "LIMITED_HISTORY":
        return "limited_history"
    if coverage_status == "DELISTED":
        return "delisted"
    if coverage_status == "NETWORK_TIMEOUT":
        return "network_timeout"
    if coverage_status in {"NO_PRICE_DATA", "PROVIDER_ERROR"}:
        return coverage_status.lower()
    return base_status


def _prefer_variant_probe(symbol: str) -> bool:
    """Symbols like `ABR$F` or `ACHR.W` need bounded variant probes.

    The bulk request already tried the raw symbol. For single-name fallback,
    probing Yahoo-style variants first avoids repeated raw-provider retries.
    """
    return is_variant_sensitive_symbol(symbol)


def validate_provider_symbol(symbol: Any) -> tuple[bool, str]:
    """Validate provider symbols before expensive provider calls.

    This gate catches universe/metadata contamination such as `FOUNDATION` or
    `WEBSITE` while keeping common exchange suffixes (`.L`, `.T`), preferred
    markers (`$F`), indices (`^GSPC`) and FX/future-like tickers (`DX-Y.NYB`).
    """
    text = str(symbol or "").strip()
    upper = text.upper()
    if not text:
        return False, "empty provider symbol"
    if len(text) > 24:
        return False, "provider symbol too long"
    if any(char.isspace() for char in text):
        return False, "provider symbol contains whitespace"
    if upper in INVALID_PROVIDER_SYMBOL_WORDS:
        return False, f"reserved metadata token: {upper}"
    if any(marker in upper for marker in ["HTTP://", "HTTPS://", ".COM/", "WWW."]):
        return False, "provider symbol looks like a URL or website field"
    if not _PROVIDER_SYMBOL_PATTERN.match(text):
        return False, "provider symbol contains unsupported characters"
    if not any(char.isdigit() or char.isalpha() for char in text):
        return False, "provider symbol has no alphanumeric characters"
    return True, "valid"


def _asset_category(symbol: str, asset: dict[str, Any] | pd.Series | None = None) -> str:
    asset_type = ""
    if asset is not None:
        try:
            asset_type = str(asset.get("type", "") or "").lower()
        except Exception:
            asset_type = ""
    if is_low_priority_structured_symbol(symbol):
        return "structured"
    if is_variant_sensitive_symbol(symbol):
        return "preferred"
    if "etf" in asset_type:
        return "etf"
    return "common"


class OhlcvIngestJob:
    """Coordinate universe discovery, provider calls, DB upserts and manifests."""

    def __init__(
        self,
        financial_db_root: Path | str | None = None,
        output_root: Path | str | None = None,
        database_url: str | None = None,
        client: OhlcvClient | None = None,
        database: OhlcvDatabase | None = None,
        parquet_root: Path | str | None = None,
        run_id: str | None = None,
        run_events_path: Path | str | None = None,
    ):
        roots = resolve_data_platform_roots(financial_db_root=financial_db_root, repo_output_root=output_root)
        self.financial_db_root = roots.financial_db
        self.output_root = roots.repo_output
        self.client = client or OhlcvClient(roots.financial_db, roots.repo_output)
        self.database = database or OhlcvDatabase(database_url=database_url, financial_db_root=roots.financial_db)
        self.universe_builder = MarketUniverseBuilder(roots.financial_db, roots.repo_output)
        self.base_dir = roots.financial_db / "MarketData" / "OHLCV"
        parquet_info = get_ohlcv_parquet_root_info(roots.financial_db, roots.repo_output, explicit_root=parquet_root, create=True)
        self.parquet_root = parquet_info.path
        self.parquet_storage_mode = parquet_info.storage_mode
        self.parquet_root_source = parquet_info.source
        self.run_id = run_id or os.environ.get("RESEARCH_PLATFORM_RUN_ID", "")
        if run_events_path:
            self.run_events_path = Path(run_events_path).expanduser()
        elif self.run_id:
            self.run_events_path = self.output_root / "runs" / self.run_id / "progress.jsonl"
        else:
            self.run_events_path = None
        self.daily_dir = self.parquet_root / "daily"
        self.intraday_dir = self.parquet_root / "intraday_5m"
        self.manifest_dir = self.base_dir / "manifests"
        for path in [self.base_dir, self.daily_dir, self.intraday_dir, self.manifest_dir, roots.repo_output / "logs"]:
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
        self._write_parquet_resilient(out, target)
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
        self._write_parquet_resilient(out, target)
        return target

    def _write_parquet_resilient(self, frame: pd.DataFrame, target: Path, retries: int = 3, sleep_seconds: float = 1.5) -> None:
        """Write parquet through a local temp file with retry/copy fallback."""
        tmp_dir = self.output_root / "_tmp_ohlcv_parquet"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp = tmp_dir / f"{target.stem}.{os.getpid()}.{time.time_ns()}.parquet"
        last_error: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                if tmp.exists():
                    tmp.unlink()
                frame.to_parquet(tmp, index=False)
                target.parent.mkdir(parents=True, exist_ok=True)
                try:
                    tmp.replace(target)
                except OSError:
                    shutil.copy2(tmp, target)
                    tmp.unlink(missing_ok=True)
                return
            except OSError as exc:
                last_error = exc
                LOGGER.warning("OHLCV parquet write failed attempt=%s target=%s error=%s", attempt, target, exc)
                time.sleep(sleep_seconds * attempt)
            finally:
                if tmp.exists() and attempt == retries:
                    tmp.unlink(missing_ok=True)
        if last_error is not None:
            raise last_error

    def _write_manifest_csv(self, frame: pd.DataFrame, path: Path) -> bool:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(path, index=False)
            return True
        except OSError as exc:
            LOGGER.warning("OHLCV manifest write failed path=%s error=%s", path, exc)
            return False

    def _write_run_event(self, event: dict[str, Any]) -> None:
        if self.run_events_path is None:
            return
        payload = {
            "run_id": self.run_id,
            "stage": "equity_prices",
            "timestamp": utc_now(),
            **event,
        }
        try:
            self.run_events_path.parent.mkdir(parents=True, exist_ok=True)
            with self.run_events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, default=str, sort_keys=True) + "\n")
        except OSError as exc:
            LOGGER.warning("OHLCV run event write failed path=%s error=%s", self.run_events_path, exc)

    def _write_provider_failures(self, manifest: pd.DataFrame) -> bool:
        if manifest.empty:
            return False
        frame = manifest.copy()
        coverage = frame.get("coverage_status", pd.Series("", index=frame.index)).fillna("").astype(str).str.upper()
        status = frame.get("status", pd.Series("", index=frame.index)).fillna("").astype(str).str.lower()
        write_status = frame.get("write_status", pd.Series("", index=frame.index)).fillna("").astype(str).str.upper()
        provider_mask = coverage.isin(
            {
                "NO_PRICE_DATA",
                "NETWORK_TIMEOUT",
                "PROVIDER_ERROR",
                "INVALID_SYMBOL",
                "LIMITED_HISTORY",
                "DELISTED",
            }
        ) | status.isin({"bulk_empty", "failed", "invalid_symbol", "network_timeout", "no_price_data", "limited_history", "delisted"})
        provider_mask = provider_mask & ~write_status.eq("FAILED")
        failures = frame[provider_mask].copy()
        if failures.empty:
            return False
        failures["error_type"] = failures.get("error_type", failures.get("coverage_status", pd.Series("", index=failures.index))).fillna("").astype(str)
        missing_error_type = failures["error_type"].str.len().eq(0)
        failures.loc[missing_error_type, "error_type"] = failures.get("coverage_status", pd.Series("PROVIDER_ERROR", index=failures.index)).astype(str)
        failures["category"] = failures.get("category", pd.Series("", index=failures.index)).fillna("").astype(str)
        missing_category = failures["category"].str.len().eq(0)
        if missing_category.any():
            failures.loc[missing_category, "category"] = [
                _asset_category(symbol, None) for symbol in failures.loc[missing_category, "provider_symbol"].fillna("").astype(str)
            ]
        failures["attempts"] = pd.to_numeric(failures.get("attempts", pd.Series(1, index=failures.index)), errors="coerce").fillna(1).astype(int)
        failures["last_seen"] = failures.get("updated_at", pd.Series(utc_now(), index=failures.index)).fillna(utc_now()).astype(str)
        columns = [
            "ticker",
            "provider_symbol",
            "resolved_provider_symbol",
            "exchange",
            "universe",
            "category",
            "provider",
            "error_type",
            "coverage_status",
            "coverage_reason",
            "error",
            "probe_error",
            "attempts",
            "rows",
            "start",
            "end",
            "first_price_date",
            "last_price_date",
            "last_seen",
        ]
        out = failures[[col for col in columns if col in failures.columns]].reset_index(drop=True)
        output_path = self.output_root / "tables" / "OHLCV_provider_failures.csv"
        self._write_manifest_csv(out, output_path)
        self._write_manifest_csv(out, self.manifest_dir / "OHLCV_provider_failures.csv")
        return True

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
        assets_override: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Run full-history or incremental daily OHLCV ingestion."""
        started = time.time()
        assets = assets_override.copy() if assets_override is not None else self.build_assets(markets=markets, max_assets=max_assets, include_etfs=include_etfs)
        if assets_override is not None and max_assets:
            assets = assets.head(int(max_assets)).copy()
        if assets.empty:
            manifest = pd.DataFrame([{"status": "no_assets", "updated_at": utc_now()}])
            self._write_manifest_csv(manifest, self.manifest_dir / "ohlcv_daily_manifest.csv")
            self._write_manifest_csv(manifest, self.output_root / "tables" / "OHLCV_daily_manifest.csv")
            return manifest
        if dry_run:
            manifest = assets.assign(status="dry_run", rows=0, parquet_root=str(self.parquet_root), parquet_storage_mode=self.parquet_storage_mode)
            self._write_manifest_csv(manifest, self.manifest_dir / "ohlcv_daily_manifest.csv")
            self._write_manifest_csv(manifest, self.output_root / "tables" / "OHLCV_daily_manifest.csv")
            return manifest
        rows: list[dict[str, Any]] = []
        assets = self._attach_asset_ids(assets)
        validation = assets["provider_symbol"].apply(validate_provider_symbol)
        assets["provider_symbol_valid"] = validation.apply(lambda item: bool(item[0]))
        assets["provider_symbol_validation_error"] = validation.apply(lambda item: str(item[1]))
        invalid_assets = assets[~assets["provider_symbol_valid"]].copy()
        if not invalid_assets.empty:
            for row in invalid_assets.to_dict("records"):
                rows.append(
                    {
                        "ticker": row.get("ticker", ""),
                        "provider_symbol": row.get("provider_symbol", ""),
                        "asset_id": row.get("asset_id", pd.NA),
                        "exchange": row.get("exchange", ""),
                        "status": "invalid_symbol",
                        "coverage_status": "INVALID_SYMBOL",
                        "coverage_reason": row.get("provider_symbol_validation_error", "invalid provider symbol"),
                        "provider": "validation_gate",
                        "rows": 0,
                        "start": start_date,
                        "end": end_date,
                        "first_price_date": pd.NA,
                        "last_price_date": pd.NA,
                        "coverage_ratio": 0.0,
                        "listing_gap_days": pd.NA,
                        "error_type": "INVALID_SYMBOL",
                        "category": _asset_category(str(row.get("provider_symbol", "")), row),
                        "attempts": 0,
                    }
                )
            self._write_run_event(
                {
                    "event": "invalid_symbols_filtered",
                    "provider_status": "invalid_symbol",
                    "provider": "validation_gate",
                    "processed": 0,
                    "success": 0,
                    "failed": int(len(invalid_assets)),
                    "limited_history": 0,
                    "symbols": ",".join(invalid_assets["provider_symbol"].astype(str).head(25).tolist()),
                }
            )
        assets = assets[assets["provider_symbol_valid"]].drop(columns=["provider_symbol_valid", "provider_symbol_validation_error"]).copy()
        if assets.empty:
            manifest = pd.DataFrame(rows)
            manifest["duration_seconds"] = round(time.time() - started, 2)
            manifest["updated_at"] = utc_now()
            self._write_manifest_csv(manifest, self.manifest_dir / "ohlcv_daily_manifest.csv")
            self._write_manifest_csv(manifest, self.output_root / "tables" / "OHLCV_daily_manifest.csv")
            self._write_provider_failures(manifest)
            return manifest
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
        today = pd.Timestamp.now(tz="UTC").date()
        if end_date:
            today = min(today, pd.to_datetime(end_date).date())
        provider_end = today.isoformat()
        assets = assets[pd.to_datetime(assets["start_for_job"]).dt.date.le(today)].copy()
        if assets.empty:
            current_rows = rows or [{"status": "already_current"}]
            manifest = pd.DataFrame(current_rows)
            if "status" not in manifest.columns:
                manifest["status"] = "already_current"
            manifest["updated_at"] = utc_now()
            self._write_manifest_csv(manifest, self.manifest_dir / "ohlcv_daily_manifest.csv")
            self._write_manifest_csv(manifest, self.output_root / "tables" / "OHLCV_daily_manifest.csv")
            self._write_provider_failures(manifest)
            return manifest
        grouped = assets.groupby("start_for_job", dropna=False)
        processed_count = 0
        success_count = 0
        failed_count = len(invalid_assets)
        limited_history_count = 0
        for group_start, group in grouped:
            group_assets = group.to_dict("records")
            for batch_no, batch in enumerate(_chunks(group_assets, batch_size), start=1):
                symbols = [row["provider_symbol"] for row in batch]
                batch_map = {row["provider_symbol"]: row for row in batch}
                batch_success = 0
                batch_failed = 0
                batch_limited = 0
                try:
                    batch_frames, bulk_meta = self.client.fetch_daily_bulk(
                        symbols,
                        start=str(group_start),
                        end=provider_end,
                        assets=batch,
                        preferred_providers=preferred_providers,
                    )
                except Exception as exc:
                    batch_frames = {}
                    bulk_meta = {"provider": None}
                    error_type = classify_provider_error(exc)
                    rows.append({"ticker": "", "provider_symbol": ",".join(symbols[:10]), "status": _status_from_history(_manifest_history_fields(pd.DataFrame(), str(group_start), provider_end, exc), "bulk_failed"), "coverage_status": error_type, "provider": "policy_engine", "rows": 0, "error": str(exc), "error_type": error_type, "category": "batch", "attempts": 1, "start": group_start, "batch": batch_no})
                    batch_failed += len(symbols)
                if not batch_frames:
                    rows.append({"ticker": "", "provider_symbol": ",".join(symbols[:10]), "status": "bulk_empty", "coverage_status": "NO_PRICE_DATA", "provider": bulk_meta.get("provider"), "rows": 0, "error_type": "NO_PRICE_DATA", "category": "batch", "attempts": 1, "start": group_start, "batch": batch_no})
                missing = [symbol for symbol in symbols if symbol not in batch_frames]
                for symbol in missing:
                    if is_low_priority_structured_symbol(symbol):
                        batch_failed += 1
                        rows.append(
                            {
                                "ticker": batch_map[symbol]["ticker"],
                                "provider_symbol": symbol,
                                "status": "special_security_no_price_data",
                                "coverage_status": "NO_PRICE_DATA",
                                "coverage_reason": "warrant/unit/right skipped after bulk miss",
                                "provider": bulk_meta.get("provider"),
                                "rows": 0,
                                "start": group_start,
                                "end": provider_end,
                                "first_price_date": pd.NA,
                                "last_price_date": pd.NA,
                                "coverage_ratio": 0.0,
                                "listing_gap_days": pd.NA,
                                "error_type": "NO_PRICE_DATA",
                                "category": _asset_category(symbol, batch_map[symbol]),
                                "attempts": 1,
                                "batch": batch_no,
                            }
                        )
                        continue
                    if _prefer_variant_probe(symbol):
                        try:
                            probe_frame, probe_meta = self.client.probe_recent_listing(
                                symbol,
                                start=str(group_start),
                                end=provider_end,
                                asset=batch_map[symbol],
                                preferred_providers=preferred_providers,
                            )
                            if not probe_frame.empty:
                                batch_frames[symbol] = probe_frame
                                history = dict(probe_meta.get("history") or _manifest_history_fields(probe_frame, str(group_start), provider_end))
                                if history.get("coverage_status") == "LIMITED_HISTORY":
                                    batch_limited += 1
                                rows.append({"ticker": batch_map[symbol]["ticker"], "provider_symbol": symbol, "resolved_provider_symbol": probe_meta.get("probe_symbol", symbol), "status": _status_from_history(history, "variant_probe"), "provider": probe_meta.get("provider"), "rows": len(probe_frame), "start": group_start, "error_type": history.get("coverage_status"), "category": _asset_category(symbol, batch_map[symbol]), "attempts": 1, **history})
                                continue
                        except Exception as probe_exc:
                            history = _manifest_history_fields(pd.DataFrame(), str(group_start), provider_end, probe_exc)
                            batch_failed += 1
                            rows.append({"ticker": batch_map[symbol]["ticker"], "provider_symbol": symbol, "status": _status_from_history(history, "failed"), "error": str(probe_exc), "start": group_start, "error_type": history.get("coverage_status"), "category": _asset_category(symbol, batch_map[symbol]), "attempts": 1, **history})
                            continue
                    try:
                        frame, meta = self.client.fetch_daily_one(
                            symbol,
                            str(group_start),
                            provider_end,
                            asset=batch_map[symbol],
                            preferred_providers=preferred_providers,
                        )
                        if not frame.empty:
                            batch_frames[symbol] = frame
                        history = dict(meta.get("history") or _manifest_history_fields(frame, str(group_start), provider_end))
                        if frame.empty:
                            batch_failed += 1
                        if history.get("coverage_status") == "LIMITED_HISTORY":
                            batch_limited += 1
                        rows.append({"ticker": batch_map[symbol]["ticker"], "provider_symbol": symbol, "status": _status_from_history(history, "fallback"), "provider": meta.get("provider"), "rows": len(frame), "start": group_start, "error_type": history.get("coverage_status"), "category": _asset_category(symbol, batch_map[symbol]), "attempts": 1, **history})
                    except Exception as exc:
                        history = _manifest_history_fields(pd.DataFrame(), str(group_start), provider_end, exc)
                        if history["coverage_status"] in {"DELISTED", "NO_PRICE_DATA"}:
                            try:
                                probe_frame, probe_meta = self.client.probe_recent_listing(
                                    symbol,
                                    start="2020-01-01",
                                    end=provider_end,
                                    asset=batch_map[symbol],
                                    preferred_providers=preferred_providers,
                                )
                                if not probe_frame.empty:
                                    batch_frames[symbol] = probe_frame
                                    history = dict(probe_meta.get("history") or _manifest_history_fields(probe_frame, str(group_start), provider_end))
                                    if history.get("coverage_status") == "LIMITED_HISTORY":
                                        batch_limited += 1
                                    rows.append({"ticker": batch_map[symbol]["ticker"], "provider_symbol": symbol, "resolved_provider_symbol": probe_meta.get("probe_symbol", symbol), "status": _status_from_history(history, "limited_history"), "provider": probe_meta.get("provider"), "rows": len(probe_frame), "start": group_start, "error_type": history.get("coverage_status"), "category": _asset_category(symbol, batch_map[symbol]), "attempts": 2, **history})
                                    continue
                            except Exception as probe_exc:
                                history["probe_error"] = str(probe_exc)
                        batch_failed += 1
                        rows.append({"ticker": batch_map[symbol]["ticker"], "provider_symbol": symbol, "status": _status_from_history(history, "failed"), "error": str(exc), "start": group_start, "error_type": history.get("coverage_status"), "category": _asset_category(symbol, batch_map[symbol]), "attempts": 1, **history})
                for symbol, frame in batch_frames.items():
                    asset = pd.Series(batch_map[symbol])
                    asset_id = int(asset["asset_id"])
                    clean = frame.copy()
                    clean["asset_id"] = asset_id
                    clean["ingestion_ts"] = utc_now()
                    inserted = self.database.upsert_daily_prices(clean)
                    path: Path | None = None
                    write_status = "OK"
                    write_error = ""
                    try:
                        path = self._write_daily_file(asset, clean)
                    except OSError as exc:
                        write_status = "FAILED"
                        write_error = f"{type(exc).__name__}: {exc}"
                        LOGGER.warning("OHLCV parquet write failed ticker=%s provider_symbol=%s error=%s", asset.get("ticker"), symbol, exc)
                    history = _manifest_history_fields(clean, str(group_start), provider_end)
                    rows.append(
                        {
                            "ticker": asset["ticker"],
                            "provider_symbol": symbol,
                            "asset_id": asset_id,
                            "exchange": asset.get("exchange", ""),
                            "status": _status_from_history(history, "downloaded" if write_status == "OK" else "downloaded_db_only"),
                            **history,
                            "provider": clean["source"].dropna().iloc[-1] if "source" in clean and not clean.empty else "unknown",
                            "rows": inserted,
                            "start": group_start,
                            "end": provider_end,
                            "target_path": str(path) if path is not None else "",
                            "parquet_root": str(self.parquet_root),
                            "parquet_storage_mode": self.parquet_storage_mode,
                            "write_status": write_status,
                            "write_error": write_error,
                            "error_type": "WRITE_FAILED" if write_status == "FAILED" else history.get("coverage_status", "OK"),
                            "category": _asset_category(symbol, asset),
                            "attempts": 1,
                            "batch": batch_no,
                        }
                    )
                    batch_success += 1
                    if history.get("coverage_status") == "LIMITED_HISTORY":
                        batch_limited += 1
                    if write_status == "FAILED":
                        batch_failed += 1
                processed_count += len(batch)
                success_count += batch_success
                failed_count += batch_failed
                limited_history_count += batch_limited
                parquet_files = list(self.daily_dir.glob("*/*.parquet")) if self.daily_dir.exists() else []
                latest_parquet = max(parquet_files, key=lambda path: path.stat().st_mtime).name if parquet_files else ""
                self._write_run_event(
                    {
                        "event": "batch_complete",
                        "start": str(group_start),
                        "end": provider_end,
                        "batch": batch_no,
                        "provider": bulk_meta.get("provider"),
                        "provider_status": "success" if batch_success else "failed",
                        "symbols": ",".join(symbols[:25]),
                        "processed": processed_count,
                        "success": success_count,
                        "failed": failed_count,
                        "limited_history": limited_history_count,
                        "parquet_files": len(parquet_files),
                        "latest_parquet": latest_parquet,
                    }
                )
        manifest = pd.DataFrame(rows)
        manifest["duration_seconds"] = round(time.time() - started, 2)
        manifest["updated_at"] = utc_now()
        self._write_manifest_csv(manifest, self.manifest_dir / "ohlcv_daily_manifest.csv")
        (self.output_root / "tables").mkdir(parents=True, exist_ok=True)
        self._write_manifest_csv(manifest, self.output_root / "tables" / "OHLCV_daily_manifest.csv")
        if "write_status" in manifest.columns:
            failures = manifest[manifest["write_status"].astype(str).eq("FAILED")].copy()
            if not failures.empty:
                self._write_manifest_csv(failures, self.output_root / "tables" / "OHLCV_write_failures.csv")
        self._write_provider_failures(manifest)
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
            manifest = assets.assign(status="dry_run" if dry_run else "no_assets", rows=0, parquet_root=str(self.parquet_root), parquet_storage_mode=self.parquet_storage_mode)
            self._write_manifest_csv(manifest, self.manifest_dir / "ohlcv_intraday_5m_manifest.csv")
            return manifest
        assets = self._attach_asset_ids(assets)
        rows: list[dict[str, Any]] = []
        for row in assets.to_dict("records"):
            symbol = row["provider_symbol"]
            try:
                frame, meta = self.client.fetch_intraday_5m_one(symbol, month=month, asset=row, preferred_providers=preferred_providers)
                frame["asset_id"] = int(row["asset_id"])
                inserted = self.database.upsert_intraday_5m(frame)
                path = ""
                write_status = "OK"
                write_error = ""
                try:
                    path = str(self._write_intraday_file(pd.Series(row), frame))
                except OSError as exc:
                    write_status = "FAILED"
                    write_error = f"{type(exc).__name__}: {exc}"
                rows.append({"ticker": row["ticker"], "provider_symbol": symbol, "asset_id": row["asset_id"], "status": "downloaded" if write_status == "OK" else "downloaded_db_only", "provider": meta.get("provider"), "rows": inserted, "target_path": path, "parquet_root": str(self.parquet_root), "parquet_storage_mode": self.parquet_storage_mode, "write_status": write_status, "write_error": write_error})
            except Exception as exc:
                rows.append({"ticker": row["ticker"], "provider_symbol": symbol, "asset_id": row["asset_id"], "status": "failed", "error": str(exc)})
        manifest = pd.DataFrame(rows)
        manifest["updated_at"] = utc_now()
        self._write_manifest_csv(manifest, self.manifest_dir / "ohlcv_intraday_5m_manifest.csv")
        return manifest


def summarize_ohlcv_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    if manifest.empty or "status" not in manifest:
        return pd.DataFrame()
    out = manifest.groupby("status", dropna=False).agg(symbols=("ticker", "count"), rows=("rows", "sum")).reset_index()
    if "coverage_status" in manifest.columns:
        coverage_counts = manifest["coverage_status"].fillna("UNKNOWN").astype(str).value_counts().to_dict()
        for status_name in ["OK", "LIMITED_HISTORY", "DELISTED", "NETWORK_TIMEOUT", "NO_PRICE_DATA", "PROVIDER_ERROR"]:
            out[f"coverage_{status_name.lower()}"] = coverage_counts.get(status_name, 0)
    if "duration_seconds" in manifest:
        out["duration_seconds"] = manifest["duration_seconds"].max()
    return out
