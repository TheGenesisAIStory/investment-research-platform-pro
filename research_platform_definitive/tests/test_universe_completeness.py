from __future__ import annotations

from research_platform_core.multi_asset_universe import multi_asset_universe_catalog


def test_fx_universe_contains_g10_and_em_symbols() -> None:
    catalog = multi_asset_universe_catalog()
    fx = catalog[catalog["asset_class"].eq("fx")]
    symbols = set(fx["provider_symbol"].astype(str))
    for symbol in ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X", "AUDUSD=X", "USDCAD=X", "NZDUSD=X", "USDMXN=X"]:
        if symbol == "USDCAD=X":
            assert symbol in symbols or "CAD=X" in symbols
        else:
            assert symbol in symbols


def test_fixed_income_universe_covers_us_eu_it_global() -> None:
    catalog = multi_asset_universe_catalog()
    fi = catalog[catalog["asset_class"].isin(["etf_fi", "yield_proxy"])]
    regions = set(fi["region"].astype(str))
    assert {"us", "eu", "it", "global"}.issubset(regions)
    assert {"TLT", "IEF", "HYG", "LQD"}.issubset(set(fi["provider_symbol"].astype(str)))


def test_crypto_universe_contains_majors_and_spot_etfs() -> None:
    catalog = multi_asset_universe_catalog()
    crypto = catalog[catalog["asset_class"].eq("crypto")]
    providers = set(crypto["provider_symbol"].astype(str))
    assert {"BTC-USD", "ETH-USD", "SOL-USD", "IBIT", "FBTC", "BITO"}.issubset(providers)


def test_multi_asset_regions_include_global_eu_it_jp_apac_em() -> None:
    catalog = multi_asset_universe_catalog()
    regions = set(catalog["region"].astype(str))
    assert {"global", "eu", "it", "jp", "apac", "em"}.issubset(regions)
