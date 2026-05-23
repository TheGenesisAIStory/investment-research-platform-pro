from __future__ import annotations

from pathlib import Path
import sys
import tempfile

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for candidate in [PROJECT_ROOT, PROJECT_ROOT / "src"]:
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from research_platform_core.loaders.ohlcv_client import normalize_ohlcv_frame, parse_yfinance_bulk
from research_platform_core.loaders.kaggle_seed_loader import import_kaggle_seeds
from research_platform_core.api_orchestrator import DataProviderPolicyEngine, DataProviderRegistry, ProviderRequest, ProviderSpec, ProviderUsageTracker
from research_platform_core.ohlcv_ingest import OhlcvIngestJob
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
    manifest = job.run_daily(markets=["unit"], start_date="2024-01-01", mode="full", max_assets=2, batch_size=2)
    assert set(manifest["status"]) == {"downloaded"}
    assert int(manifest["rows"].sum()) == 4


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
