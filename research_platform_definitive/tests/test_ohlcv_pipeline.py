from __future__ import annotations

from pathlib import Path
import sys
import tempfile

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for candidate in [PROJECT_ROOT, PROJECT_ROOT / "src"]:
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from research_platform_core.loaders.ohlcv_client import (
    classify_history_window,
    classify_provider_error,
    is_low_priority_structured_symbol,
    is_variant_sensitive_symbol,
    normalize_ohlcv_frame,
    parse_yfinance_bulk,
    provider_symbol_variants,
)
from research_platform_core.loaders.market_universe import _provider_symbol_for_index
from research_platform_core.loaders.kaggle_seed_loader import import_kaggle_seeds
from research_platform_core.api_orchestrator import DataProviderPolicyEngine, DataProviderRegistry, ProviderRequest, ProviderSpec, ProviderUsageTracker
from research_platform_core.data_health import list_ohlcv_provider_failures, load_run_events
from research_platform_core.ohlcv_ingest import OhlcvIngestJob, validate_provider_symbol
from research_platform_core.ohlcv_store import OhlcvDatabase


class FakeOhlcvClient:
    def fetch_daily_bulk(self, symbols: list[str], start: str, end: str | None = None, assets=None, preferred_providers=None):
        return self.fetch_yfinance_daily_bulk(symbols, start, end), {"provider": "fake_bulk"}

    def fetch_yfinance_daily_bulk(self, symbols: list[str], start: str, end: str | None = None):
        return {
            symbol: pd.DataFrame(
                {
                    "date": ["2024-01-02", "2024-01-03"],
                    "open": [10.0, 11.0],
                    "high": [11.0, 12.0],
                    "low": [9.5, 10.5],
                    "close": [10.5, 11.5],
                    "adjclose": [10.5, 11.5],
                    "volume": [1000, 1100],
                    "provider_symbol": symbol,
                    "source": "fake",
                    "ingestion_ts": "2024-01-03T00:00:00+00:00",
                }
            )
            for symbol in symbols
        }

    def fetch_daily_one(self, symbol: str, start: str, end: str | None = None, asset=None, preferred_providers=None):
        return self.fetch_yfinance_daily_bulk([symbol], start, end)[symbol], {"provider": "fake"}


def test_normalize_ohlcv_frame() -> None:
    raw = pd.DataFrame({"Date": ["2024-01-02"], "Open": [1], "High": [2], "Low": [0.5], "Close": [1.5], "Volume": [100]})
    out = normalize_ohlcv_frame(raw, "AAA", "unit")
    assert list(out[["provider_symbol", "source"]].iloc[0]) == ["AAA", "unit"]
    assert float(out["close"].iloc[0]) == 1.5


def test_ohlcv_history_classification_handles_recent_listing_and_timeout() -> None:
    recent_listing = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03"],
            "close": [10.0, 10.5],
            "provider_symbol": ["FBYD", "FBYD"],
            "source": ["unit", "unit"],
        }
    )
    history = classify_history_window(recent_listing, "2000-01-01", "2024-01-10")
    assert history["coverage_status"] == "LIMITED_HISTORY"
    assert classify_provider_error("curl: (28) Operation timed out") == "NETWORK_TIMEOUT"
    assert classify_provider_error("possibly delisted; no price data found") == "DELISTED"


def test_recent_listing_probe_prioritizes_common_ticker_for_preferred_like_symbol() -> None:
    variants = provider_symbol_variants("FBYDP", prefer_common=True)
    assert variants[0] == "FBYD"
    assert "FBYDP" in variants


def test_preferred_share_variants_prioritize_yahoo_dash_format() -> None:
    variants = provider_symbol_variants("ARES$B", prefer_common=True)
    assert variants[:3] == ["ARES-PB", "ARES-B", "ARES.B"]
    assert "ARES$B" in variants


def test_special_security_variants_skip_raw_when_requested() -> None:
    warrant_variants = provider_symbol_variants("ACHR.W", prefer_common=True, include_raw=False)
    assert warrant_variants[:3] == ["ACHR-WT", "ACHR-WS", "ACHR-W"]
    assert "ACHR.W" not in warrant_variants
    assert is_variant_sensitive_symbol("ACHR.W")
    assert is_low_priority_structured_symbol("ACHR.W")

    unit_variants = provider_symbol_variants("AIIA.U", prefer_common=True, include_raw=False)
    assert unit_variants[0] == "AIIA-U"
    assert "AIIA.U" not in unit_variants
    assert is_variant_sensitive_symbol("AIIA.U")
    assert is_low_priority_structured_symbol("AIIA.U")
    assert not is_low_priority_structured_symbol("BRK.B")


def test_validate_provider_symbol_rejects_metadata_tokens() -> None:
    assert validate_provider_symbol("FOUNDATION")[0] is False
    assert validate_provider_symbol("WEBSITE")[0] is False
    assert validate_provider_symbol("OPERATOR")[0] is False
    assert validate_provider_symbol("EXCHANGES")[0] is False
    assert validate_provider_symbol("CONSTITUENTS")[0] is False
    assert validate_provider_symbol("NAN")[0] is False
    assert validate_provider_symbol("Investor Relations")[0] is False
    assert validate_provider_symbol("ARES$B")[0] is True
    assert validate_provider_symbol("DX-Y.NYB")[0] is True
    assert validate_provider_symbol("ENEL.MI")[0] is True
    assert validate_provider_symbol("III.L")[0] is True


def test_provider_symbol_for_index_uses_yahoo_suffixes() -> None:
    assert _provider_symbol_for_index("ENEL-MI", "ftsemib") == "ENEL.MI"
    assert _provider_symbol_for_index("ADS-DE", "dax40") == "ADS.DE"
    assert _provider_symbol_for_index("AC-PA", "cac40") == "AC.PA"
    assert _provider_symbol_for_index("ACS-MC", "ibex35") == "ACS.MC"
    assert _provider_symbol_for_index("III", "ftse100") == "III.L"
    assert _provider_symbol_for_index("BT-A", "ftse100") == "BT-A.L"


def test_sqlite_store_upserts(tmp_path: Path) -> None:
    db = OhlcvDatabase(database_url=f"sqlite:///{tmp_path / 'ohlcv.sqlite'}")
    assets = pd.DataFrame(
        [{"ticker": "AAA", "provider_symbol": "AAA", "exchange": "NASDAQ", "country": "US", "type": "stock", "name": "AAA Inc", "primary_source": "unit", "active_flag": True}]
    )
    asset_map = db.upsert_assets(assets)
    asset_id = next(iter(asset_map.values()))
    prices = pd.DataFrame(
        [{"asset_id": asset_id, "date": "2024-01-02", "open": 1, "high": 2, "low": 0.5, "close": 1.5, "adjclose": 1.5, "volume": 100, "source": "unit", "ingestion_ts": "now"}]
    )
    assert db.upsert_daily_prices(prices) == 1
    assert db.latest_daily_dates([asset_id])[asset_id] == "2024-01-02"


def test_ingest_job_uses_incremental_db(tmp_path: Path) -> None:
    job = OhlcvIngestJob(financial_db_root=tmp_path / "db", output_root=tmp_path / "out", database_url=f"sqlite:///{tmp_path / 'ohlcv.sqlite'}", client=FakeOhlcvClient())
    assets = pd.DataFrame(
        [
            {"ticker": "AAA", "provider_symbol": "AAA", "exchange": "NASDAQ", "country": "US", "type": "stock", "name": "AAA Inc", "primary_source": "unit", "active_flag": True},
            {"ticker": "BBB", "provider_symbol": "BBB", "exchange": "NYSE", "country": "US", "type": "etf", "name": "BBB ETF", "primary_source": "unit", "active_flag": True},
        ]
    )
    job.build_assets = lambda *args, **kwargs: assets
    manifest = job.run_daily(markets=["unit"], start_date="2024-01-01", end_date="2024-01-10", mode="full", max_assets=2, batch_size=2)
    assert set(manifest["status"]) == {"downloaded"}
    assert int(manifest["rows"].sum()) == 4


def test_ingest_job_can_stage_parquet_outside_financial_db(tmp_path: Path, monkeypatch) -> None:
    parquet_root = tmp_path / "local_ohlcv"
    monkeypatch.setenv("RESEARCH_PLATFORM_OHLCV_PARQUET_ROOT", str(parquet_root))
    job = OhlcvIngestJob(financial_db_root=tmp_path / "db", output_root=tmp_path / "out", database_url=f"sqlite:///{tmp_path / 'ohlcv.sqlite'}", client=FakeOhlcvClient())
    assets = pd.DataFrame(
        [{"ticker": "AAA", "provider_symbol": "AAA", "exchange": "NASDAQ", "country": "US", "type": "stock", "name": "AAA Inc", "primary_source": "unit", "active_flag": True}]
    )
    job.build_assets = lambda *args, **kwargs: assets
    manifest = job.run_daily(markets=["unit"], start_date="2024-01-01", end_date="2024-01-10", mode="full", max_assets=1, batch_size=1)
    target_path = Path(str(manifest.loc[0, "target_path"]))
    assert str(target_path).startswith(str(parquet_root))
    assert target_path.exists()
    assert manifest.loc[0, "parquet_storage_mode"] == "env_override"


def test_ingest_job_records_parquet_write_failure_without_crashing(tmp_path: Path, monkeypatch) -> None:
    job = OhlcvIngestJob(financial_db_root=tmp_path / "db", output_root=tmp_path / "out", database_url=f"sqlite:///{tmp_path / 'ohlcv.sqlite'}", client=FakeOhlcvClient())
    assets = pd.DataFrame(
        [{"ticker": "AAA", "provider_symbol": "AAA", "exchange": "NASDAQ", "country": "US", "type": "stock", "name": "AAA Inc", "primary_source": "unit", "active_flag": True}]
    )
    job.build_assets = lambda *args, **kwargs: assets

    def fail_write(*args, **kwargs):
        raise OSError(89, "Operation canceled")

    monkeypatch.setattr(job, "_write_daily_file", fail_write)
    manifest = job.run_daily(markets=["unit"], start_date="2024-01-01", end_date="2024-01-10", mode="full", max_assets=1, batch_size=1)
    assert manifest.loc[0, "status"] == "downloaded_db_only"
    assert manifest.loc[0, "write_status"] == "FAILED"
    assert (tmp_path / "out" / "tables" / "OHLCV_write_failures.csv").exists()


def test_ingest_job_records_invalid_symbol_provider_failure(tmp_path: Path) -> None:
    job = OhlcvIngestJob(financial_db_root=tmp_path / "db", output_root=tmp_path / "out", database_url=f"sqlite:///{tmp_path / 'ohlcv.sqlite'}", client=FakeOhlcvClient())
    assets = pd.DataFrame(
        [{"ticker": "WEBSITE", "provider_symbol": "WEBSITE", "exchange": "NASDAQ", "country": "US", "type": "stock", "name": "bad metadata", "primary_source": "unit", "active_flag": True}]
    )
    job.build_assets = lambda *args, **kwargs: assets
    manifest = job.run_daily(markets=["unit"], start_date="2024-01-01", end_date="2024-01-10", mode="full", max_assets=1, batch_size=1)
    failures = list_ohlcv_provider_failures(tmp_path / "db", tmp_path / "out")

    assert manifest.loc[0, "status"] == "invalid_symbol"
    assert failures.loc[0, "error_type"] == "INVALID_SYMBOL"
    assert failures.loc[0, "provider_symbol"] == "WEBSITE"


def test_ingest_job_writes_structured_progress_jsonl(tmp_path: Path) -> None:
    events_path = tmp_path / "out" / "runs" / "unit_run" / "progress.jsonl"
    job = OhlcvIngestJob(
        financial_db_root=tmp_path / "db",
        output_root=tmp_path / "out",
        database_url=f"sqlite:///{tmp_path / 'ohlcv.sqlite'}",
        client=FakeOhlcvClient(),
        run_id="unit_run",
        run_events_path=events_path,
    )
    assets = pd.DataFrame(
        [{"ticker": "AAA", "provider_symbol": "AAA", "exchange": "NASDAQ", "country": "US", "type": "stock", "name": "AAA Inc", "primary_source": "unit", "active_flag": True}]
    )
    job.build_assets = lambda *args, **kwargs: assets
    job.run_daily(markets=["unit"], start_date="2024-01-01", end_date="2024-01-10", mode="full", max_assets=1, batch_size=1)

    events = load_run_events(events_path)
    assert not events.empty
    assert events.loc[0, "run_id"] == "unit_run"
    assert "processed" in events.columns


def test_provider_policy_respects_daily_budget(tmp_path: Path) -> None:
    registry = DataProviderRegistry(
        {
            "cheap": ProviderSpec("cheap", priority=1, requests_per_day=1, quality_score=0.5, coverage_score=0.5, cost_score=1.0),
            "quality": ProviderSpec("quality", priority=2, requests_per_day=10, quality_score=0.9, coverage_score=0.9, cost_score=0.3),
        }
    )
    tracker = ProviderUsageTracker(tmp_path / "usage.json")
    engine = DataProviderPolicyEngine(registry=registry, usage_tracker=tracker)
    request = ProviderRequest(provider_symbol="AAA", interval="1d", start="2000-01-01")
    order = engine.provider_order(request)
    assert order[0] == "quality"
    assert engine.reserve("cheap")
    order_after_budget = engine.provider_order(request)
    assert "cheap" not in order_after_budget


def test_kaggle_seed_import_uses_store_and_symbol_latest(tmp_path: Path) -> None:
    kaggle_dir = tmp_path / "Kaggle" / "price-volume-data-for-all-us-stocks-etfs" / "Stocks"
    kaggle_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "Date": ["2024-01-02", "2024-01-03"],
            "Open": [1, 2],
            "High": [2, 3],
            "Low": [0.5, 1.5],
            "Close": [1.5, 2.5],
            "Volume": [100, 200],
        }
    ).to_csv(kaggle_dir / "aapl.us.txt", index=False)
    db_url = f"sqlite:///{tmp_path / 'ohlcv.sqlite'}"
    manifest = import_kaggle_seeds(
        datasets=["us_stocks_etfs"],
        kaggle_root=tmp_path / "Kaggle",
        financial_db_root=tmp_path / "db",
        output_root=tmp_path / "out",
        database_url=db_url,
        dry_run=False,
    )
    assert int(manifest["num_rows"].sum()) == 2
    db = OhlcvDatabase(database_url=db_url)
    assert db.latest_daily_dates_by_provider_symbol(["AAPL"])["AAPL"] == "2024-01-03"


if __name__ == "__main__":
    test_normalize_ohlcv_frame()
    with tempfile.TemporaryDirectory() as tmp:
        test_sqlite_store_upserts(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_ingest_job_uses_incremental_db(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_provider_policy_respects_daily_budget(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_kaggle_seed_import_uses_store_and_symbol_latest(Path(tmp))
    print("ohlcv_pipeline_tests_OK")
