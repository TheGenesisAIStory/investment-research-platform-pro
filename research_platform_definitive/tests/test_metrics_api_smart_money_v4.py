from __future__ import annotations

import pandas as pd


def test_79_metrics_registered_and_complete():
    from research_platform_core.metrics_metadata import METRICS_METADATA

    assert len(METRICS_METADATA) >= 79
    for metric_id, meta in METRICS_METADATA.items():
        assert meta.formula
        assert meta.formula_latex, metric_id
        assert meta.source_paper, metric_id
        assert meta.source_doi, metric_id
        assert meta.interpretation_range, metric_id
        assert meta.metric_family, metric_id


def test_new_metric_families_present():
    from research_platform_core.metrics_metadata import METRICS_METADATA

    for metric_id in [
        "sterling_ratio",
        "burke_ratio",
        "martin_ratio",
        "historical_var_95",
        "ff3_r_squared",
        "ic_mean_21d",
        "cot_net_speculator_zscore",
        "dark_pool_activity_proxy",
    ]:
        assert metric_id in METRICS_METADATA


def test_api_orchestrator_fallback_without_keys():
    from research_platform_core.api_orchestrator import ApiOrchestrator

    orch = ApiOrchestrator(api_keys={})
    result = orch.get_fundamentals_waterfall("AAPL")
    assert result["source_provider"] == "none"
    assert result["data_completeness_pct"] == 0.0


def test_cboe_pcr_parser_fallback(tmp_path):
    from research_platform_core.smart_money import get_pcr_cboe_bulk

    result = get_pcr_cboe_bulk(("2024-01-01", "2024-01-31"), output_root=tmp_path)
    assert {"date", "total_pcr", "equity_pcr", "index_pcr", "data_status"}.issubset(result.columns)
    assert len(result) > 0


def test_italy_local_factors_schema(tmp_path):
    from research_platform_core.italy_local_factors import build_italy_factor_panel, fetch_ecb_sovereign_spread

    spread = fetch_ecb_sovereign_spread(tmp_path)
    assert {"date", "btp_bund_10y", "data_status"}.issubset(spread.columns)
    panel = build_italy_factor_panel("2024-01-01", tmp_path)
    assert {"date", "ticker", "price", "ret_21d", "momentum_12m_1m", "btp_bund_10y"}.issubset(panel.columns)
    assert panel["ticker"].nunique() >= 10


def test_aqr_parser_records_failures(tmp_path):
    from research_platform_core.aqr_factors import parse_aqr_excel_robust

    bad_path = tmp_path / "missing_or_bad.xlsx"
    result = parse_aqr_excel_robust(bad_path, output_root=tmp_path)
    assert result.empty
    assert (tmp_path / "aqr_factors" / "parse_failures.csv").exists()
