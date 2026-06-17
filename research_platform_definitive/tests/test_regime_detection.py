from __future__ import annotations

import pandas as pd

from research_platform_core.regime_detection import build_regime_history, detect_market_regime, get_current_regime


def test_regime_detection_synthetic_all_four_labels(tmp_path) -> None:
    macro = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=4, freq="B"),
            "equity_momentum_21d": [0.03, -0.03, -0.08, 0.02],
            "equity_momentum_63d": [0.06, -0.01, -0.12, -0.04],
            "credit_spread": [0.01, 0.00, -0.05, 0.00],
            "vix_proxy": [15.0, 18.0, 35.0, 22.0],
            "yield_slope": [1.0, 0.3, -0.2, 0.1],
            "commodity_momentum": [0.02, -0.01, -0.04, 0.01],
        }
    )
    path = tmp_path / "macro_signals.csv"
    macro.to_csv(path, index=False)

    history = build_regime_history(macro_history_path=path, output_root=tmp_path / "output", write=True)

    assert history["regime"].tolist() == ["risk_on", "risk_off", "crisis", "recovery"]
    assert {"date", "regime", "equity_momentum_21d", "credit_spread", "vix_proxy", "yield_slope", "commodity_momentum"}.issubset(history.columns)


def test_detect_market_regime_asof_from_snapshot_path(tmp_path) -> None:
    macro = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=2, freq="B"),
            "equity_momentum_21d": [0.02, -0.06],
            "credit_spread": [0.01, -0.04],
            "vix_proxy": [16.0, 34.0],
        }
    )
    path = tmp_path / "macro.csv"
    macro.to_csv(path, index=False)

    latest = detect_market_regime("2024-01-02", macro_snapshot_path=path)

    assert latest["regime"] == "crisis"
    assert latest["confidence"] > 0
    assert "signals" in latest


def test_get_current_regime_missing_csv_is_unknown(tmp_path) -> None:
    latest = get_current_regime(output_root=tmp_path / "missing_output")

    assert latest["regime"] == "unknown"
    assert latest["status"] == "MISSING"
