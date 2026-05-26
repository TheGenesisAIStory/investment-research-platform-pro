from __future__ import annotations

from pathlib import Path

from research_platform_core import FEATURE_METADATA, METRICS_METADATA, metadata_frame, metrics_metadata_frame


def test_all_features_have_academic_metadata_fields() -> None:
    missing = [
        key
        for key, meta in FEATURE_METADATA.items()
        if not meta.formula or not meta.formula_latex or not meta.source_paper or not meta.economic_rationale
    ]
    assert missing == []


def test_all_metrics_have_formula_latex_and_source() -> None:
    missing = [key for key, meta in METRICS_METADATA.items() if not meta.formula or not meta.formula_latex or not meta.paper]
    assert missing == []


def test_metadata_frames_expose_academic_columns() -> None:
    features = metadata_frame(["momentum_12m_1m", "piotroski_f_score", "amihud_illiquidity"])
    metrics = metrics_metadata_frame(["information_ratio", "calmar_ratio", "cvar_95"])
    assert {"formula_latex", "source_paper", "economic_rationale", "point_in_time_safe"}.issubset(features.columns)
    assert {"formula_latex", "paper", "typical_range", "annualized"}.issubset(metrics.columns)
    assert features["source_paper"].astype(str).str.len().gt(0).all()
    assert metrics["paper"].astype(str).str.len().gt(0).all()


def test_academic_reference_page_and_docs_are_wired() -> None:
    root = Path(__file__).resolve().parents[1]
    page = root / "research_platform_app" / "pages" / "15_📚_Academic_Reference.py"
    doc = root / "docs" / "FEATURE_ACADEMIC_REFERENCE.md"
    page_text = page.read_text(encoding="utf-8")
    doc_text = doc.read_text(encoding="utf-8")

    assert page.exists()
    assert doc.exists()
    assert "Factor Browser" in page_text
    assert "Fama-French & AQR Factor Returns" in page_text
    assert "FEATURE_ACADEMIC_REFERENCE.md" in page_text
    assert "Point-in-time safety" in doc_text
