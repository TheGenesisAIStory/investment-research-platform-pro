from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from factors.eps_factors import EPS_FACTOR_COLUMNS, build_eps_factors
from ml_stock_lab.factor_registry import FACTOR_BLOCKS, FACTOR_REGISTRY
from research_platform_core.feature_metadata import metadata_for_feature
from research_platform_core.metrics_metadata import metadata_for_metric
from simulation.monte_carlo import monte_carlo_factor_uncertainty, monte_carlo_returns


def _eps_panel() -> pd.DataFrame:
    dates = pd.date_range("2021-03-31", periods=8, freq="QE")
    rows = []
    for ticker, base in [("AAA", 1.0), ("BBB", 2.0)]:
        for i, date in enumerate(dates):
            actual = base + i * 0.1
            rows.append(
                {
                    "ticker": ticker,
                    "report_date": date,
                    "actual_eps": actual,
                    "consensus_eps": actual * 0.95,
                    "price": 20 + i,
                }
            )
    return pd.DataFrame(rows)


def test_eps_factors_are_lagged_and_complete() -> None:
    frame = _eps_panel()
    result = build_eps_factors(frame)
    assert set(EPS_FACTOR_COLUMNS).issubset(result.columns)
    first_by_ticker = result.groupby("ticker").head(1)
    assert first_by_ticker["eps_surprise"].isna().all()
    assert result["eps_surprise"].dropna().iloc[0] > 0
    assert result["earnings_yield"].dropna().iloc[-1] > 0


def test_eps_factors_use_naive_consensus_proxy_when_missing() -> None:
    frame = _eps_panel().drop(columns=["consensus_eps"])
    result = build_eps_factors(frame)
    assert "eps_surprise" in result.columns
    assert result.groupby("ticker")["eps_surprise"].apply(lambda s: s.notna().sum()).min() > 0


def test_monte_carlo_returns_reproducible() -> None:
    rng = np.random.default_rng(7)
    scores = pd.DataFrame(rng.normal(0.001, 0.02, size=(260, 4)), columns=list("ABCD"))
    first = monte_carlo_returns(scores, n_sim=500, horizon=30, seed=42)
    second = monte_carlo_returns(scores, n_sim=500, horizon=30, seed=42)
    assert list(first["paths"].columns) == ["p05", "p25", "p50", "p75", "p95"]
    assert first["terminal_percentiles"] == second["terminal_percentiles"]
    assert "var_95" in first and "cvar_95" in first


def test_monte_carlo_factor_uncertainty_shape() -> None:
    frame = pd.DataFrame(np.arange(60, dtype=float).reshape(20, 3), columns=["value", "momentum", "quality"])
    result = monte_carlo_factor_uncertainty(frame, n_sim=200, seed=42)
    assert result.shape == (3, 5)
    assert {"p05", "p50", "p95"}.issubset(result.columns)


def test_registry_metadata_and_notebook_skeleton() -> None:
    assert "eps_factors" in FACTOR_BLOCKS
    assert "eps_factors" in FACTOR_REGISTRY
    assert "monte_carlo_returns" in FACTOR_REGISTRY
    assert metadata_for_feature("eps_surprise") is not None
    assert metadata_for_feature("eps_revision_3m") is not None
    assert metadata_for_metric("mc_p50") is not None
    notebook_path = Path("research_platform_definitive/colab/ml_training_pipeline.ipynb")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    sources = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    for section in [
        "Mount Google Drive",
        "Feature Engineering",
        "Model Training",
        "Walk-Forward",
        "SHAP Feature Importance",
        "Factor IC Analysis",
        "Export Artifacts",
    ]:
        assert section in sources
