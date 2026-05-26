from __future__ import annotations

import numpy as np
import pandas as pd

from research_platform_core.market_statistics import (
    compute_correlation_matrix,
    compute_ticker_market_statistics,
    summarize_correlation_matrix,
)


def _write_ohlcv(root, exchange: str, ticker: str, returns: np.ndarray) -> None:
    dates = pd.bdate_range("2024-01-01", periods=len(returns))
    close = 100.0 * (1.0 + pd.Series(returns)).cumprod()
    path = root / "MarketData" / "OHLCV" / "daily" / exchange
    path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"date": dates, "close": close}).to_parquet(path / f"{ticker}.parquet", index=False)


def test_compute_ticker_market_statistics_and_correlation(tmp_path, monkeypatch) -> None:
    financial_db = tmp_path / "financial_db"
    output = tmp_path / "output"
    monkeypatch.setenv("RESEARCH_PLATFORM_OHLCV_PARQUET_ROOT", str(financial_db / "MarketData" / "OHLCV"))
    rng = np.random.default_rng(11)
    bench = rng.normal(0.0002, 0.01, 320)
    asset_a = 1.2 * bench + rng.normal(0, 0.006, 320)
    asset_b = -0.2 * bench + rng.normal(0, 0.009, 320)
    _write_ohlcv(financial_db, "TEST", "SPY", bench)
    _write_ohlcv(financial_db, "TEST", "AAA", asset_a)
    _write_ohlcv(financial_db, "TEST", "BBB", asset_b)

    stats = compute_ticker_market_statistics("AAA", benchmark="SPY", financial_db_root=financial_db, output_root=output)
    assert {"annualized_volatility", "annualized_variance", "beta_to_benchmark", "correlation_to_benchmark"}.issubset(stats.columns)
    assert stats["beta_to_benchmark"].dropna().iloc[-1] > 0

    corr = compute_correlation_matrix(["AAA", "BBB", "SPY"], financial_db_root=financial_db, output_root=output, window=252)
    assert set(corr.columns) == {"AAA", "BBB", "SPY"}
    summary = summarize_correlation_matrix(corr, benchmark="SPY")
    assert summary["status"] == "OK"
    assert summary["asset_count"] == 3
