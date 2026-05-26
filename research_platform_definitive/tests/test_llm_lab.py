from __future__ import annotations

from pathlib import Path

from research_platform_core.llm_lab import build_context_from_database, render_prompt, save_prompt_packet
from research_platform_core.research_database import populate_research_database


def test_render_strategy_review_prompt_from_database(tmp_path: Path) -> None:
    db_path = tmp_path / "research.sqlite"
    populate_research_database(db_path=db_path, sample_dir=tmp_path / "sample", start="2024-01-02", end="2024-12-31")

    context = build_context_from_database(db_path)
    prompt = render_prompt("strategy_review_it", context)

    assert "Investment Research Platform Pro" in prompt
    assert "risk_balanced_blend" in prompt
    assert "Non proporre live trading" in prompt


def test_save_prompt_packet(tmp_path: Path) -> None:
    db_path = tmp_path / "research.sqlite"
    populate_research_database(db_path=db_path, sample_dir=tmp_path / "sample", start="2024-01-02", end="2024-10-31")

    paths = save_prompt_packet(["data_quality_brief_it"], output_dir=tmp_path / "llm_lab", db_path=db_path)

    assert "data_quality_brief_it" in paths
    assert Path(paths["data_quality_brief_it"]).exists()
    assert Path(paths["manifest"]).exists()
