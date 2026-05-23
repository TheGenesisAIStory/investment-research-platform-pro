#!/usr/bin/env python3
"""Smoke test for ml_stock_lab."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ml_stock_lab import (
    FundamentalDatasetBuilder,
    PeerImpliedValuator,
    compute_relative_mispricing,
    cross_sectional_zscore,
    make_quantile_portfolios,
    run_ml_stock_lab_experiment,
)


def main() -> None:
    df = pd.DataFrame({
        "date": ["2026-01-31"] * 10,
        "ticker": [f"T{i}" for i in range(10)],
        "market_value": [100 + i + (i % 3) * 2 for i in range(10)],
        "quality_score": [i / 10 for i in range(10)],
        "valuation_score": [1 - i / 10 for i in range(10)],
        "forward_return": [i / 100 for i in range(10)],
    })
    X, y, cols = FundamentalDatasetBuilder(["quality_score", "valuation_score"]).build(df)
    fv = PeerImpliedValuator("ols").fit_predict(X, y)
    mp = compute_relative_mispricing(fv, y)
    z = cross_sectional_zscore(mp, df.loc[X.index, "date"])
    test = df.loc[X.index].copy()
    test["zscore"] = z
    qret = make_quantile_portfolios(test, "zscore", "forward_return", q=5)
    assert len(fv) == len(y)
    assert "long_short" in qret["quantile"].astype(str).tolist()
    assert "name_count" in qret.columns
    with tempfile.TemporaryDirectory() as tmp:
        result = run_ml_stock_lab_experiment(Path(tmp), model="ols", max_rows=100)
        assert result["status"] in {"OK", "NO_PANEL", "INSUFFICIENT_PANEL", "NO_SIGNALS", "NO_QUINTILE_RETURNS", "NO_LONG_SHORT_LEG"}
    print("ml_stock_lab_smoke_OK")


if __name__ == "__main__":
    main()
