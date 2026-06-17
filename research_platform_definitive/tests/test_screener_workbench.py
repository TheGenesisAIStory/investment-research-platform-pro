from __future__ import annotations

import pandas as pd

from research_platform_app.screener_workbench import (
    apply_screening_filters,
    build_screening_frame,
    explain_ml_signal,
    screening_zero_result_suggestions,
)


def test_build_screening_frame_merges_research_layers(tmp_path) -> None:
    company = {
        "screener_results": pd.DataFrame(
            [
                {
                    "ticker": "AAA",
                    "company_name": "Alpha",
                    "sector": "Tech",
                    "country": "US",
                    "index_membership": "sp500",
                    "screener_score": 70,
                }
            ]
        ),
        "valuation_gap": pd.DataFrame(),
        "extended_valuation": pd.DataFrame(),
    }
    portfolio = {"selection_results": pd.DataFrame([{"ticker": "AAA", "selection_score": 80}]), "allocation": pd.DataFrame()}
    smart_money = {
        "scores": pd.DataFrame([{"ticker": "AAA", "composite_institutional_interest_score": 60}]),
        "events": pd.DataFrame([{"ticker": "AAA", "event_label": "insider_buy"}]),
    }
    ml_lab = {"signals": pd.DataFrame([{"ticker": "AAA", "score_rf": 90, "mispricing_rel": 0.12}])}

    frame = build_screening_frame(company, portfolio, smart_money, ml_lab, tmp_path)

    row = frame[frame["ticker"].eq("AAA")].iloc[0]
    assert row["smart_recent_events"] == 1
    assert row["ml_score"] == 90
    assert row["valuation_signal_score"] > 0
    assert row["composite_conviction_score"] > 0


def test_screening_frame_does_not_treat_fair_value_as_ml_score(tmp_path) -> None:
    company = {
        "screener_results": pd.DataFrame([{"ticker": "AAA", "screener_score": 50}]),
        "valuation_gap": pd.DataFrame(),
        "extended_valuation": pd.DataFrame(),
    }
    frame = build_screening_frame(
        company,
        {"selection_results": pd.DataFrame(), "allocation": pd.DataFrame()},
        {"scores": pd.DataFrame(), "events": pd.DataFrame()},
        {"signals": pd.DataFrame([{"ticker": "AAA", "fair_value_hat": 250.0, "mispricing_rel": 0.25}])},
        tmp_path,
    )

    row = frame.iloc[0]
    assert pd.isna(row["ml_score"])
    assert row["valuation_signal_score"] > 0


def test_apply_screening_filters_and_explanation(tmp_path) -> None:
    frame = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "sector": "Tech",
                "universe": "sp500",
                "ml_quintile": "Q5",
                "ml_score": 90.0,
                "smart_money_score": 55.0,
                "smart_recent_events": 2,
                "composite_conviction_score": 75.0,
                "mispricing_rel": 0.2,
            },
            {
                "ticker": "BBB",
                "sector": "Utilities",
                "universe": "ftsemib",
                "ml_quintile": "Q2",
                "ml_score": 30.0,
                "smart_money_score": 0.0,
                "smart_recent_events": 0,
                "composite_conviction_score": 20.0,
            },
        ]
    )

    filtered = apply_screening_filters(
        frame,
        {"min_ml_score": 60.0, "ml_quintiles": ["Q5"], "require_smart_events": True, "sort_by": "ml_score"},
    )
    drivers, text = explain_ml_signal(filtered, "AAA")

    assert filtered["ticker"].tolist() == ["AAA"]
    assert not drivers.empty
    assert "AAA" in text


def test_zero_result_suggestions_are_filter_specific() -> None:
    suggestions = screening_zero_result_suggestions(
        {
            "min_ml_score": 80,
            "min_smart_money_score": 60,
            "require_smart_events": True,
            "ml_quintiles": ["Q5"],
        },
        total_names=6890,
    )

    joined = " ".join(suggestions)
    assert "Require recent Smart Money event" in joined
    assert "Lower Min ML score" in joined
