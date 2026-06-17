from __future__ import annotations

import numpy as np
import pandas as pd

from ml_stock_lab.macro_features import build_macro_context_features


def _macro_history(symbol: str, start: float, periods: int = 90) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=periods, freq="B")
    close = start * (1 + np.linspace(0, 0.25, periods))
    return pd.DataFrame({"date": dates, "symbol": symbol, "close": close})


def test_macro_context_features_smoke_and_missing_proxies() -> None:
    macro = pd.concat([_macro_history("SPY", 100.0), _macro_history("TLT", 90.0)], ignore_index=True)
    panel = pd.DataFrame(
        {
            "date": pd.date_range("2024-03-01", periods=5, freq="B"),
            "ticker": ["AAA"] * 5,
            "price": np.linspace(10, 11, 5),
        }
    )

    enriched = build_macro_context_features(panel, macro)

    assert len(enriched) == len(panel)
    assert "macro_spy_ret21d" in enriched.columns
    assert "macro_tlt_ret21d" in enriched.columns


def test_macro_context_features_are_lagged() -> None:
    dates = pd.date_range("2024-01-01", periods=60, freq="B")
    spy = pd.DataFrame({"date": dates, "symbol": "SPY", "close": np.arange(100.0, 160.0)})
    tlt = pd.DataFrame({"date": dates, "symbol": "TLT", "close": np.arange(90.0, 150.0)})
    hyg = pd.DataFrame({"date": dates, "symbol": "HYG", "close": np.arange(80.0, 140.0)})
    macro = pd.concat([spy, tlt, hyg], ignore_index=True)
    target_date = dates[30]
    panel = pd.DataFrame({"date": [target_date], "ticker": ["AAA"], "price": [10.0]})

    enriched = build_macro_context_features(panel, macro)
    expected = spy.set_index("date")["close"].pct_change(21).shift(1).loc[target_date]

    assert np.isclose(enriched.loc[0, "macro_spy_ret21d"], expected, equal_nan=False)


def test_macro_context_from_existing_regime_history() -> None:
    macro = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=4, freq="B"),
            "macro_spy_ret21d": [0.01, 0.02, 0.03, 0.04],
            "macro_credit_spread": [0.0, 0.01, 0.02, 0.03],
            "regime": ["risk_on", "risk_on", "risk_off", "recovery"],
        }
    )
    panel = pd.DataFrame({"date": [pd.Timestamp("2024-01-04")], "ticker": ["AAA"], "price": [10.0]})

    enriched = build_macro_context_features(panel, macro)

    assert "macro_regime_encoded" in enriched.columns
    assert enriched.loc[0, "macro_regime_encoded"] == -0.5
