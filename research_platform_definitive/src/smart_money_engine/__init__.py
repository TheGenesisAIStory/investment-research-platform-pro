"""Smart Money Government Data Engine.

Official-source-first analytics for institutional ownership, insider activity,
activism, macro positioning, cross-border flows and public spending exposure.
"""

from .analytics import (
    beneficial_ownership_analytics,
    event_feed,
    government_spending_analytics,
    insider_analytics,
    issuer_deep_dive,
    manager_deep_dive,
    ownership_analytics,
    sector_theme_monitor,
    smart_money_screener,
)
from .ingestion import ingest_official_sources
from .pipeline import run_smart_money_engine
from .resolution import build_entity_master, match_awards_to_issuers
from .schemas import (
    analyst_quick_wins,
    dataset_priority_ranking,
    government_dataset_matrix,
    mvp_30_day_plan,
    open_source_pattern_matrix,
    source_registry_frame,
)

__all__ = [
    "beneficial_ownership_analytics",
    "event_feed",
    "government_spending_analytics",
    "insider_analytics",
    "issuer_deep_dive",
    "manager_deep_dive",
    "ownership_analytics",
    "sector_theme_monitor",
    "smart_money_screener",
    "ingest_official_sources",
    "run_smart_money_engine",
    "build_entity_master",
    "match_awards_to_issuers",
    "analyst_quick_wins",
    "dataset_priority_ranking",
    "government_dataset_matrix",
    "mvp_30_day_plan",
    "open_source_pattern_matrix",
    "source_registry_frame",
]
