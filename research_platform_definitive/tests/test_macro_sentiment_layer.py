from __future__ import annotations

from research_platform_core.macro_market import macro_asset_catalog
from research_platform_core.sentiment_analysis import score_sentiment_frame, score_text


def test_macro_asset_catalog_covers_requested_regions_and_asset_classes() -> None:
    catalog = macro_asset_catalog()

    assert {"global", "usa", "eu", "italy", "crypto"}.issubset(set(catalog["region"]))
    assert {"fx", "commodity_etf", "fixed_income_etf", "crypto"}.issubset(set(catalog["asset_class"]))
    assert catalog["provider_symbol"].notna().all()


def test_macro_asset_catalog_filters_region_and_asset_class() -> None:
    catalog = macro_asset_catalog(regions=["italy"], asset_classes=["fixed_income_etf"])

    assert not catalog.empty
    assert set(catalog["region"]) == {"italy"}
    assert set(catalog["asset_class"]) == {"fixed_income_etf"}


def test_sentiment_scoring_basic_direction() -> None:
    assert score_text("strong bullish upside upgrade") > 0
    assert score_text("weak bearish downgrade downside risk") < 0


def test_sentiment_frame_labels_mentions() -> None:
    frame = score_sentiment_frame(
        [
            {"provider": "test", "ticker": "AAA", "text": "bullish strong upside", "created_at": "2026-01-01"},
            {"provider": "test", "ticker": "AAA", "text": "bearish weak downside", "created_at": "2026-01-01"},
        ]
    )

    assert set(frame["sentiment_label"]) == {"positive", "negative"}
