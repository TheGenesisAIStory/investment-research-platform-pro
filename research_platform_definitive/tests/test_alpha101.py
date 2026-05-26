from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research_platform_core.alpha101 import Alpha101Suite, alpha001, alpha002


N_ASSETS = 10
T_DAYS = 150


@pytest.fixture
def mock_panel():
    rng = np.random.default_rng(42)
    idx = pd.date_range("2020-01-01", periods=T_DAYS, freq="B")
    cols = [f"TICKER_{i:03d}" for i in range(N_ASSETS)]
    close = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.01, size=(T_DAYS, N_ASSETS)), axis=0), index=idx, columns=cols)
    open_ = close * (1 + rng.normal(0, 0.003, size=(T_DAYS, N_ASSETS)))
    high = pd.concat([open_, close], axis=0).groupby(level=0).max() * (1 + np.abs(rng.normal(0, 0.005, size=(T_DAYS, N_ASSETS))))
    low = pd.concat([open_, close], axis=0).groupby(level=0).min() * (1 - np.abs(rng.normal(0, 0.005, size=(T_DAYS, N_ASSETS))))
    volume = pd.DataFrame(np.abs(rng.normal(0, 1, size=(T_DAYS, N_ASSETS))) * 1e6 + 5e5, index=idx, columns=cols)
    vwap = (high + low + close) / 3
    returns = close.pct_change()
    return dict(close=close, open_=open_, high=high, low=low, volume=volume, vwap=vwap, returns=returns)


def test_alpha001_shape(mock_panel):
    result = alpha001(**mock_panel)
    assert result.shape == (T_DAYS, N_ASSETS)


def test_alpha002_shape(mock_panel):
    result = alpha002(**mock_panel)
    assert result.shape == (T_DAYS, N_ASSETS)


def test_alpha001_no_lookahead(mock_panel):
    base = alpha001(**mock_panel)
    modified = {key: value.copy() for key, value in mock_panel.items()}
    modified["close"].iloc[-1] = modified["close"].iloc[-1] * 10
    changed = alpha001(**modified)
    pd.testing.assert_frame_equal(base.iloc[:-1], changed.iloc[:-1])


def test_all_101_compute_without_error(mock_panel):
    suite = Alpha101Suite()
    results = suite.compute_all(**mock_panel, verbose=False)
    assert len(results) == 101
    assert not suite.failed_alphas
    for name, frame in results.items():
        assert frame.shape == (T_DAYS, N_ASSETS), name
        valid_rows = frame.iloc[60:].dropna(how="all")
        assert len(valid_rows) > 0, f"{name} is all NaN after warmup"


def test_alpha_suite_to_flat_panel(mock_panel):
    suite = Alpha101Suite()
    results = suite.compute_subset(["alpha001", "alpha005", "alpha101"], **mock_panel)
    flat = suite.to_flat_panel(results)
    assert isinstance(flat.index, pd.MultiIndex)
    assert "alpha001" in flat.columns
    assert "alpha101" in flat.columns


def test_feature_metadata_integration():
    from research_platform_core.feature_metadata import FEATURE_METADATA

    alpha101_features = [key for key, meta in FEATURE_METADATA.items() if meta.category == "alpha101"]
    assert len(alpha101_features) == 101
    assert "alpha001" in FEATURE_METADATA
    assert "alpha101" in FEATURE_METADATA


def test_alpha_outputs_cross_sectional_rank(mock_panel):
    result = alpha001(**mock_panel)
    last_valid = result.dropna(how="all").iloc[-1]
    assert last_valid.max() <= 1.0
    assert last_valid.min() >= 0.0
