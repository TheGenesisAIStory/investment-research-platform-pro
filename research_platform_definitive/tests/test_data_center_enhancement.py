from __future__ import annotations

from pathlib import Path
import tempfile

from research_platform_core.cache_manager import DataCache
from research_platform_core.data_center_catalog import build_target_catalog, summarize_target_catalog
from research_platform_core.loaders.fama_french import parse_fama_french_csv


def test_fama_french_parser_monthly() -> None:
    text = "Header line\n,Mkt-RF,SMB,HML,RF\n202001,1.00,2.00,-3.00,0.10\n202002,0.50,0.25,0.10,0.01\n\nAnnual Factors:\n"
    df = parse_fama_french_csv(text, "monthly")
    assert len(df) == 2
    assert round(float(df["Mkt-RF"].iloc[0]), 4) == 0.01


def test_target_catalog_has_required_domains(tmp_path: Path) -> None:
    catalog = build_target_catalog(tmp_path)
    domains = set(catalog["domain"])
    assert {"equity_universe", "factor_data", "fx", "commodities", "risk_factors"}.issubset(domains)
    summary = summarize_target_catalog(catalog)
    assert not summary.empty


def test_data_cache_ttl(tmp_path: Path) -> None:
    cache = DataCache(tmp_path)
    cache.set("answer", {"x": 42})
    assert cache.get("answer", ttl_hours=1)["x"] == 42


if __name__ == "__main__":
    test_fama_french_parser_monthly()
    with tempfile.TemporaryDirectory() as tmp:
        test_target_catalog_has_required_domains(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_data_cache_ttl(Path(tmp))
    print("data_center_enhancement_tests_OK")
