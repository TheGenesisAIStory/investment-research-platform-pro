from __future__ import annotations

import pandas as pd

from research_platform_core.smart_money import fetch_cot_data, normalize_cot_data


def _raw_cot() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "market_and_exchange_names": "E-MINI S&P 500 - CHICAGO MERCANTILE EXCHANGE",
                "report_date_as_yyyy_mm_dd": "2024-01-02T00:00:00.000",
                "lev_money_positions_long": 1000,
                "lev_money_positions_short": 700,
                "open_interest_all": 10000,
            },
            {
                "market_and_exchange_names": "E-MINI S&P 500 - CHICAGO MERCANTILE EXCHANGE",
                "report_date_as_yyyy_mm_dd": "2024-01-09T00:00:00.000",
                "lev_money_positions_long": 1200,
                "lev_money_positions_short": 650,
                "open_interest_all": 11000,
            },
            {
                "market_and_exchange_names": "GOLD - COMMODITY EXCHANGE INC.",
                "report_date_as_yyyy_mm_dd": "2024-01-09T00:00:00.000",
                "lev_money_positions_long": 500,
                "lev_money_positions_short": 800,
                "open_interest_all": 5000,
            },
        ]
    )


def test_normalize_cot_data_maps_core_markets() -> None:
    normalized = normalize_cot_data(_raw_cot())
    assert {"SP500", "GOLD"}.issubset(set(normalized["instrument"]))
    latest_sp = normalized[normalized["instrument"].eq("SP500")].tail(1).iloc[0]
    assert latest_sp["net_noncommercial"] == 550
    assert "weekly_change_net" in normalized.columns


def test_fetch_cot_data_writes_snapshot_without_network(tmp_path) -> None:
    bundle = fetch_cot_data(tmp_path, raw_frame=_raw_cot())
    assert not bundle["history"].empty
    assert not bundle["snapshot"].empty
    assert (tmp_path / "smart_money" / "tables" / "SmartMoney_COT_history.csv").exists()
    assert (tmp_path / "smart_money" / "tables" / "SmartMoney_COT_snapshot.csv").exists()

