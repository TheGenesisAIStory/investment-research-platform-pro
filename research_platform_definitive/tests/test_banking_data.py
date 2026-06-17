from __future__ import annotations

import pandas as pd

from research_platform_core.banking_data import (
    ITALIAN_LISTED_BANK_TICKERS,
    build_market_panel,
    curated_listed_bank_universe,
)
import research_platform_core.banking_data as banking_data


def test_curated_listed_bank_universe_contains_core_tickers() -> None:
    universe = curated_listed_bank_universe()
    assert len(universe) >= len(ITALIAN_LISTED_BANK_TICKERS)
    assert {"legal_name", "ticker", "listed_flag", "country", "source"}.issubset(universe.columns)
    assert {"ISP.MI", "UCG.MI", "BAMI.MI"}.issubset(set(universe["ticker"]))
    assert universe["listed_flag"].all()


def test_build_market_panel_falls_back_to_curated_listed_banks(monkeypatch) -> None:
    def fake_yfinance_panel(tickers: list[str], start: str = "2015-01-01") -> pd.DataFrame:
        return pd.DataFrame(
            {
                "ticker": tickers[:2],
                "date": pd.to_datetime(["2024-01-31", "2024-01-31"]),
                "mkt_price": [1.0, 2.0],
            }
        )

    monkeypatch.setattr(banking_data, "build_yfinance_bank_panel", fake_yfinance_panel)
    raw_universe = pd.DataFrame({"listed_flag": [False], "ticker": [pd.NA], "bank_id": ["BANK_001"]})
    panel = build_market_panel(raw_universe, start="2024-01-01")

    assert not panel.empty
    assert panel["bank_id"].notna().all()
    assert set(panel["ticker"]).issubset(set(ITALIAN_LISTED_BANK_TICKERS.values()))
