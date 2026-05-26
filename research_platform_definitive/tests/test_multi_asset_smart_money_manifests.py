from __future__ import annotations

import pandas as pd

from research_platform_core import (
    compile_multi_asset_universe_manifest,
    compile_smart_money_source_manifest,
    load_multi_asset_universe_manifest,
    load_smart_money_source_manifest,
    summarize_multi_asset_universe,
    summarize_smart_money_sources,
)


def test_multi_asset_manifest_covers_non_equity_domains(tmp_path) -> None:
    financial_db = tmp_path / "financial_db"
    output = tmp_path / "output"
    macro_table = output / "macro_market" / "tables"
    macro_table.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"symbol": "EURUSD", "status": "OK", "rows": 1200, "last_date": "2026-01-01", "target_path": "eurusd.parquet"},
            {"symbol": "GLD", "status": "OK", "rows": 1200, "last_date": "2026-01-01", "target_path": "gld.parquet"},
        ]
    ).to_csv(macro_table / "MacroAssetManifest.csv", index=False)

    manifest = compile_multi_asset_universe_manifest(financial_db, output)
    assert {"FX", "Commodities", "Fixed Income", "Crypto", "ETF", "Equity Index / ETF"}.intersection(set(manifest["domain"]))
    assert "EURUSD" in set(manifest["symbol"])
    assert (output / "macro_market" / "tables" / "MultiAssetUniverseManifest.csv").exists()
    loaded = load_multi_asset_universe_manifest(financial_db, output, refresh_if_missing=False)
    summary = summarize_multi_asset_universe(loaded)
    assert not summary.empty
    assert int(summary["instrument_count"].sum()) == len(manifest)


def test_smart_money_source_manifest_without_network(tmp_path) -> None:
    output = tmp_path / "output"
    manifest = compile_smart_money_source_manifest(tmp_path / "financial_db", output, fetch_cot=False)
    assert {"COT", "ETF_FLOWS", "OPTIONS", "ISSUER_EVENTS"}.issubset(set(manifest["domain"]))
    assert (output / "smart_money" / "tables" / "SmartMoneySourceManifest.csv").exists()
    loaded = load_smart_money_source_manifest(tmp_path / "financial_db", output, refresh_if_missing=False)
    summary = summarize_smart_money_sources(loaded)
    assert not summary.empty
    assert int(summary["source_count"].sum()) == len(manifest)
