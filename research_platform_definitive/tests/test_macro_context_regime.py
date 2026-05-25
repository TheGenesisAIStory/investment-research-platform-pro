from __future__ import annotations

import numpy as np
import pandas as pd

from research_platform_core.macro_context import add_macro_context_features, build_macro_context_panel
from research_platform_core.regime_detection import build_market_regime_history, detect_market_regime


def _macro_history(symbol: str, start: float, periods: int = 320) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=periods, freq="B")
    close = start + np.linspace(0, start * 0.2, periods)
    return pd.DataFrame({"date": dates, "close": close, "symbol": symbol})


def test_macro_context_features_are_lagged_and_joinable(tmp_path) -> None:
    financial_db = tmp_path / "financial_db"
    output = tmp_path / "output"
    macro_dir = financial_db / "MarketData" / "Macro" / "equity_index_etf"
    macro_dir.mkdir(parents=True)
    manifest_rows = []
    for symbol, start in [("SPY", 100.0), ("HYG", 80.0), ("TLT", 90.0), ("DXY", 100.0), ("VIX", 20.0)]:
        path = macro_dir / f"{symbol}.parquet"
        _macro_history(symbol, start).to_parquet(path, index=False)
        manifest_rows.append({"symbol": symbol, "status": "OK", "rows": 320, "target_path": str(path), "last_date": "2025-03-21"})
    table_root = output / "macro_market" / "tables"
    table_root.mkdir(parents=True)
    pd.DataFrame(manifest_rows).to_csv(table_root / "MacroAssetManifest.csv", index=False)

    context = build_macro_context_panel(financial_db, output, write=True)
    assert "macro_spy_ret63d" in context.columns
    assert "macro_credit_hyg_tlt_ret63d" in context.columns
    assert context["macro_spy_ret63d"].notna().any()

    panel = pd.DataFrame({"date": pd.date_range("2024-09-01", periods=5, freq="B"), "ticker": ["AAA"] * 5, "price": range(5)})
    enriched = add_macro_context_features(panel, financial_db, output)
    assert "macro_context_score" in enriched.columns
    assert len(enriched) == len(panel)


def test_regime_detection_builds_latest_label(tmp_path) -> None:
    output = tmp_path / "output"
    table_root = output / "macro_market" / "tables"
    table_root.mkdir(parents=True)
    pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=3, freq="B"),
            "macro_spy_ret63d": [0.02, 0.05, -0.2],
            "macro_credit_hyg_tlt_ret63d": [0.01, 0.02, -0.08],
            "macro_vix_level": [18.0, 17.0, 40.0],
            "macro_dxy_ret63d": [0.0, -0.01, 0.05],
            "macro_risk_on_score": [75.0, 80.0, 10.0],
        }
    ).to_csv(table_root / "MacroContextPanel.csv", index=False)
    history = build_market_regime_history(output_root=output, write=True)
    assert not history.empty
    latest = detect_market_regime(output_root=output)
    assert latest["regime"] == "crisis"

