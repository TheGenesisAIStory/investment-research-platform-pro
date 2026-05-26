from __future__ import annotations

from pathlib import Path

from research_platform_core.research_database import ResearchDatabase, populate_research_database


def test_populate_research_database_creates_core_tables(tmp_path: Path) -> None:
    db_path = tmp_path / "research.sqlite"
    sample_dir = tmp_path / "sample"

    summary = populate_research_database(
        db_path=db_path,
        sample_dir=sample_dir,
        start="2024-01-02",
        end="2024-12-31",
        seed=7,
        sample_rows_per_symbol=20,
    )

    db = ResearchDatabase(db_path)
    counts = db.table_counts()
    assert counts["instruments"] >= 8
    assert counts["ohlcv_daily"] > 500
    assert counts["features_labels"] > 200
    assert counts["ml_signals"] > 0
    assert counts["backtest_results"] == 1
    assert counts["strategy_registry"] >= 3
    assert summary["table_counts"]["data_catalog_entries"] >= 5
    assert (sample_dir / "instruments_sample.csv").exists()
    assert (sample_dir / "research_database_summary.json").exists()


def test_research_database_snapshots_are_queryable(tmp_path: Path) -> None:
    db_path = tmp_path / "research.sqlite"
    populate_research_database(db_path=db_path, sample_dir=tmp_path / "sample", start="2024-01-02", end="2024-09-30")

    db = ResearchDatabase(db_path)
    strategies = db.list_strategies()
    signals = db.latest_signals(limit=5)
    backtests = db.latest_backtests(limit=5)

    assert "risk_balanced_blend" in set(strategies["strategy_name"])
    assert {"symbol", "score", "target_weight", "reasoning"}.issubset(signals.columns)
    assert float(backtests["sharpe"].iloc[0]) == backtests["sharpe"].iloc[0]
