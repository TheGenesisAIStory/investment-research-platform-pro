from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for candidate in [PROJECT_ROOT, PROJECT_ROOT / "src"]:
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from research_platform_core.data_health import (
    get_data_health_summary,
    get_data_status_for_tickers,
    get_stage_health_for_universes,
    list_ohlcv_failures,
    list_ohlcv_provider_failures,
    restart_equity_prices,
)
from research_platform_core.run_lock import acquire_stage_lock, release_stage_lock
from research_platform_core.data_platform import get_ohlcv_daily_search_roots, get_ohlcv_parquet_root_info


def _clear_ohlcv_env(monkeypatch) -> None:
    monkeypatch.delenv("RESEARCH_PLATFORM_OHLCV_PARQUET_ROOT", raising=False)
    monkeypatch.delenv("RESEARCH_PLATFORM_OHLCV_WRITE_MODE", raising=False)


def test_ohlcv_root_uses_env_override(tmp_path: Path, monkeypatch) -> None:
    _clear_ohlcv_env(monkeypatch)
    root = tmp_path / "stable_ohlcv"
    monkeypatch.setenv("RESEARCH_PLATFORM_OHLCV_PARQUET_ROOT", str(root))

    info = get_ohlcv_parquet_root_info(tmp_path / "db", tmp_path / "out", create=True)

    assert info.path == root
    assert info.source == "env:RESEARCH_PLATFORM_OHLCV_PARQUET_ROOT"
    assert info.storage_mode == "env_override"
    assert root.exists()


def test_ohlcv_root_redirects_cloud_db_to_local_staging(tmp_path: Path, monkeypatch) -> None:
    _clear_ohlcv_env(monkeypatch)
    drive_root = tmp_path / "CloudStorage" / "GoogleDrive-test" / "Il mio Drive" / "Database Finanziario"
    output_root = tmp_path / "out"

    info = get_ohlcv_parquet_root_info(drive_root, output_root)
    search_roots = get_ohlcv_daily_search_roots(drive_root, output_root)

    assert info.storage_mode == "local_staging_for_cloud_db"
    assert info.path == output_root / "data_cache" / "financial_db_mirror" / "MarketData" / "OHLCV"
    assert search_roots[0] == info.path / "daily"


def test_ohlcv_root_reads_persistent_app_setting(tmp_path: Path, monkeypatch) -> None:
    _clear_ohlcv_env(monkeypatch)
    configured = tmp_path / "configured_ohlcv"
    settings = tmp_path / "out" / "config" / "platform_settings.json"
    settings.parent.mkdir(parents=True)
    settings.write_text('{"data": {"ohlcv_parquet_root": "%s"}}' % configured, encoding="utf-8")

    info = get_ohlcv_parquet_root_info(tmp_path / "db", tmp_path / "out")

    assert info.path == configured
    assert info.storage_mode == "settings_override"


def test_ohlcv_root_can_recover_last_run_root_from_manifest(tmp_path: Path, monkeypatch) -> None:
    _clear_ohlcv_env(monkeypatch)
    output_root = tmp_path / "out"
    parquet_root = tmp_path / "local_financial_db" / "MarketData" / "OHLCV"
    (parquet_root / "daily" / "NASDAQ").mkdir(parents=True)
    manifest_path = output_root / "tables" / "OHLCV_daily_manifest.csv"
    manifest_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "provider_symbol": "AAA",
                "exchange": "NASDAQ",
                "parquet_root": str(parquet_root),
                "target_path": str(parquet_root / "daily" / "NASDAQ" / "AAA.parquet"),
            }
        ]
    ).to_csv(manifest_path, index=False)

    info = get_ohlcv_parquet_root_info(tmp_path / "CloudStorage" / "Database Finanziario", output_root)

    assert info.path == parquet_root
    assert info.storage_mode == "manifest_override"


def test_list_ohlcv_failures_normalizes_retry_table(tmp_path: Path, monkeypatch) -> None:
    _clear_ohlcv_env(monkeypatch)
    output_root = tmp_path / "out"
    failures_path = output_root / "tables" / "OHLCV_write_failures.csv"
    failures_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "provider_symbol": "AAA",
                "exchange": "NASDAQ",
                "universe": "us_all",
                "parquet_root": str(tmp_path / "ohlcv"),
                "write_error": "OSError: [Errno 89] Operation canceled",
                "updated_at": "2026-05-24T18:00:00+00:00",
            }
        ]
    ).to_csv(failures_path, index=False)

    failures = list_ohlcv_failures(tmp_path / "db", output_root)

    assert list(failures["ticker"]) == ["AAA"]
    assert bool(failures.loc[0, "retry_candidate"])
    assert failures.loc[0, "target_path"].endswith("/daily/NASDAQ/AAA.parquet")
    assert "Operation canceled" in failures.loc[0, "error_summary"]


def test_data_health_summary_counts_ohlcv_failures_and_root(tmp_path: Path, monkeypatch) -> None:
    _clear_ohlcv_env(monkeypatch)
    parquet_root = tmp_path / "stable_ohlcv"
    monkeypatch.setenv("RESEARCH_PLATFORM_OHLCV_PARQUET_ROOT", str(parquet_root))
    output_root = tmp_path / "out"
    (output_root / "data_completion").mkdir(parents=True)
    pd.DataFrame([{"stage": "equity_prices", "status": "DONE", "rows": 1}]).to_csv(
        output_root / "data_completion" / "equity_prices_2000_2026.csv",
        index=False,
    )
    (output_root / "tables").mkdir(parents=True)
    pd.DataFrame([{"ticker": "AAA", "provider_symbol": "AAA", "exchange": "NASDAQ", "status": "downloaded", "rows": 2}]).to_csv(
        output_root / "tables" / "OHLCV_daily_manifest.csv",
        index=False,
    )
    pd.DataFrame([{"ticker": "BBB", "provider_symbol": "BBB", "exchange": "NYSE", "write_error": "OSError"}]).to_csv(
        output_root / "tables" / "OHLCV_write_failures.csv",
        index=False,
    )
    (parquet_root / "daily" / "NASDAQ").mkdir(parents=True)
    (parquet_root / "daily" / "NASDAQ" / "AAA.parquet").write_bytes(b"stub")

    health = get_data_health_summary(tmp_path / "db", output_root)
    equity_prices = health[health["stage"].eq("equity_prices")].iloc[0]

    assert equity_prices["status"] == "FAILED"
    assert int(equity_prices["failure_count"]) == 1
    assert int(equity_prices["critical_failure_count"]) == 1
    assert equity_prices["ohlcv_parquet_root"] == str(parquet_root)
    assert int(equity_prices["ohlcv_parquet_files"]) == 1


def test_list_ohlcv_provider_failures_normalizes_table(tmp_path: Path, monkeypatch) -> None:
    _clear_ohlcv_env(monkeypatch)
    output_root = tmp_path / "out"
    failures_path = output_root / "tables" / "OHLCV_provider_failures.csv"
    failures_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "ticker": "FOUNDATION",
                "provider_symbol": "FOUNDATION",
                "exchange": "NASDAQ",
                "category": "common",
                "provider": "validation_gate",
                "error_type": "INVALID_SYMBOL",
                "coverage_reason": "reserved metadata token",
                "last_seen": "2026-05-25T00:00:00+00:00",
            }
        ]
    ).to_csv(failures_path, index=False)

    failures = list_ohlcv_provider_failures(tmp_path / "db", output_root)

    assert failures.loc[0, "error_type"] == "INVALID_SYMBOL"
    assert bool(failures.loc[0, "retry_candidate"]) is False


def test_data_health_status_ok_partial_failed_running(tmp_path: Path, monkeypatch) -> None:
    _clear_ohlcv_env(monkeypatch)
    output_root = tmp_path / "out"
    (output_root / "tables").mkdir(parents=True)
    (output_root / "data_completion").mkdir(parents=True)
    pd.DataFrame([{"stage": "equity_prices", "status": "DONE", "rows": 100}]).to_csv(
        output_root / "data_completion" / "equity_prices_2000_2026.csv",
        index=False,
    )
    pd.DataFrame(
        [{"ticker": f"AAA{i}", "provider_symbol": f"AAA{i}", "exchange": "NASDAQ", "coverage_status": "OK", "status": "downloaded", "rows": 1} for i in range(100)]
    ).to_csv(output_root / "tables" / "OHLCV_daily_manifest.csv", index=False)
    assert get_data_health_summary(tmp_path / "db", output_root).query("stage == 'equity_prices'").iloc[0]["status"] == "OK"

    pd.DataFrame(
        [{"ticker": f"AAA{i}", "provider_symbol": f"AAA{i}", "exchange": "NASDAQ", "coverage_status": "OK" if i < 85 else "NO_PRICE_DATA", "status": "downloaded" if i < 85 else "no_price_data", "rows": 1 if i < 85 else 0} for i in range(100)]
    ).to_csv(output_root / "tables" / "OHLCV_daily_manifest.csv", index=False)
    assert get_data_health_summary(tmp_path / "db", output_root).query("stage == 'equity_prices'").iloc[0]["status"] == "PARTIAL"

    pd.DataFrame(
        [{"ticker": f"AAA{i}", "provider_symbol": f"AAA{i}", "exchange": "NASDAQ", "coverage_status": "NETWORK_TIMEOUT" if i < 25 else "OK", "status": "network_timeout" if i < 25 else "downloaded", "rows": 0 if i < 25 else 1} for i in range(100)]
    ).to_csv(output_root / "tables" / "OHLCV_daily_manifest.csv", index=False)
    assert get_data_health_summary(tmp_path / "db", output_root).query("stage == 'equity_prices'").iloc[0]["status"] == "FAILED"

    lock = acquire_stage_lock("equity_prices", output_root, run_id="running_unit")
    try:
        assert get_data_health_summary(tmp_path / "db", output_root).query("stage == 'equity_prices'").iloc[0]["status"] == "RUNNING"
    finally:
        release_stage_lock("equity_prices", output_root, lock_info=lock)


def test_stage_health_for_universes_uses_stage_manifests(tmp_path: Path, monkeypatch) -> None:
    _clear_ohlcv_env(monkeypatch)
    output_root = tmp_path / "out"
    (output_root / "data_completion").mkdir(parents=True)
    (output_root / "tables").mkdir(parents=True)
    pd.DataFrame(
        [
            {"stage": "equity_fundamentals", "status": "DONE", "universe": "sp500", "rows": 500},
            {"stage": "equity_fundamentals", "status": "FAILED", "universe": "nasdaq100", "rows": 0},
        ]
    ).to_csv(output_root / "data_completion" / "equity_fundamentals_2000_2026.csv", index=False)
    pd.DataFrame(
        [
            *[
                {"ticker": f"AAA{i}", "provider_symbol": f"AAA{i}", "exchange": "sp500", "coverage_status": "OK", "status": "downloaded", "rows": 1}
                for i in range(9)
            ],
            {"ticker": "BBB", "provider_symbol": "BBB", "exchange": "sp500", "coverage_status": "NO_PRICE_DATA", "status": "no_price_data", "rows": 0},
        ]
    ).to_csv(output_root / "tables" / "OHLCV_daily_manifest.csv", index=False)

    fundamentals = get_stage_health_for_universes("equity_fundamentals", ["sp500", "nasdaq100"], tmp_path / "db", output_root)
    prices = get_stage_health_for_universes("equity_prices", ["sp500"], tmp_path / "db", output_root)

    assert fundamentals.set_index("universe").loc["sp500", "status"] == "OK"
    assert fundamentals.set_index("universe").loc["nasdaq100", "status"] == "FAILED"
    assert prices.loc[0, "status"] == "PARTIAL"


def test_data_status_for_tickers_combines_fundamentals_prices_and_failures(tmp_path: Path, monkeypatch) -> None:
    _clear_ohlcv_env(monkeypatch)
    output_root = tmp_path / "out"
    db_root = tmp_path / "db"
    parquet_root = tmp_path / "ohlcv"
    monkeypatch.setenv("RESEARCH_PLATFORM_OHLCV_PARQUET_ROOT", str(parquet_root))
    (output_root / "tables").mkdir(parents=True)
    (parquet_root / "daily" / "NASDAQ").mkdir(parents=True)
    (parquet_root / "daily" / "NASDAQ" / "AAA.parquet").write_bytes(b"stub")
    fundamentals_dir = db_root / "Equities" / "US" / "sp500" / "fundamentals"
    fundamentals_dir.mkdir(parents=True)
    (fundamentals_dir / "AAA_income_statement_quarterly.parquet").write_bytes(b"stub")
    pd.DataFrame(
        [
            {"ticker": "AAA", "provider_symbol": "AAA", "exchange": "NASDAQ", "coverage_status": "OK", "status": "downloaded", "last_price_date": "2026-05-22", "rows": 10},
            {"ticker": "BBB", "provider_symbol": "BBB", "exchange": "NASDAQ", "coverage_status": "NETWORK_TIMEOUT", "status": "network_timeout", "rows": 0},
        ]
    ).to_csv(output_root / "tables" / "OHLCV_daily_manifest.csv", index=False)
    pd.DataFrame(
        [
            {"ticker": "BBB", "provider_symbol": "BBB", "exchange": "NASDAQ", "category": "common", "error_type": "NETWORK_TIMEOUT", "last_seen": "2026-05-25T00:00:00+00:00"}
        ]
    ).to_csv(output_root / "tables" / "OHLCV_provider_failures.csv", index=False)

    statuses = get_data_status_for_tickers(["AAA", "BBB"], db_root, output_root)

    assert statuses["AAA"].overall_status == "OK"
    assert statuses["AAA"].has_price_parquet
    assert statuses["BBB"].overall_status == "FAILED"
    assert statuses["BBB"].provider_error_type == "NETWORK_TIMEOUT"


def test_restart_equity_prices_failed_only_dry_run_uses_failure_assets(tmp_path: Path, monkeypatch) -> None:
    _clear_ohlcv_env(monkeypatch)
    output_root = tmp_path / "out"
    failures_path = output_root / "tables" / "OHLCV_write_failures.csv"
    failures_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "provider_symbol": "AAA",
                "exchange": "NASDAQ",
                "universe": "us_all",
                "write_error": "OSError",
            }
        ]
    ).to_csv(failures_path, index=False)

    manifest = restart_equity_prices(
        {
            "financial_db_root": tmp_path / "db",
            "output_root": output_root,
            "parquet_root": tmp_path / "stable_ohlcv",
            "failed_only": True,
            "dry_run": True,
            "max_assets": 1,
        }
    )

    assert len(manifest) == 1
    assert manifest.loc[0, "ticker"] == "AAA"
    assert manifest.loc[0, "status"] == "dry_run"


def test_restart_equity_prices_failed_only_can_filter_tickers(tmp_path: Path, monkeypatch) -> None:
    _clear_ohlcv_env(monkeypatch)
    output_root = tmp_path / "out"
    failures_path = output_root / "tables" / "OHLCV_write_failures.csv"
    failures_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {"ticker": "AAA", "provider_symbol": "AAA", "exchange": "NASDAQ", "write_error": "OSError"},
            {"ticker": "BBB", "provider_symbol": "BBB", "exchange": "NYSE", "write_error": "OSError"},
        ]
    ).to_csv(failures_path, index=False)

    manifest = restart_equity_prices(
        {
            "financial_db_root": tmp_path / "db",
            "output_root": output_root,
            "parquet_root": tmp_path / "stable_ohlcv",
            "failed_only": True,
            "tickers": ["BBB"],
            "dry_run": True,
        }
    )

    assert list(manifest["ticker"]) == ["BBB"]
