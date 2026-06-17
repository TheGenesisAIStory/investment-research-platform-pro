from __future__ import annotations

import pandas as pd

from research_platform_core.factor_benchmarks import (
    compute_factor_benchmark_summary,
    load_factor_benchmark_summary,
)


def test_factor_benchmarks_compute_and_persist(tmp_path):
    table_root = tmp_path / "ml_training_lab" / "tables"
    table_root.mkdir(parents=True)
    rows = []
    for date in pd.date_range("2024-01-01", periods=6, freq="D"):
        for idx in range(10):
            rows.append(
                {
                    "date": date.date().isoformat(),
                    "ticker": f"T{idx}",
                    "value_score": idx * 10,
                    "quality_score": 100 - idx * 5,
                    "forward_return_21d": (idx - 4) / 100,
                }
            )
    pd.DataFrame(rows).to_csv(table_root / "FactorUniversePanel.csv", index=False)

    summary = compute_factor_benchmark_summary(
        tmp_path,
        factors=["value_score", "quality_score", "missing_score"],
        target_col="forward_return_21d",
        max_rows=None,
        write=True,
    )

    assert set(summary["factor"]) == {"value_score", "quality_score", "missing_score"}
    assert summary.loc[summary["factor"].eq("value_score"), "status"].iloc[0] == "OK"
    assert summary.loc[summary["factor"].eq("missing_score"), "status"].iloc[0] == "MISSING_FACTOR"
    assert (table_root / "FactorBenchmarkSummary.csv").exists()
    loaded = load_factor_benchmark_summary(tmp_path)
    assert not loaded.empty
    assert "long_short_mean_return" in loaded.columns

