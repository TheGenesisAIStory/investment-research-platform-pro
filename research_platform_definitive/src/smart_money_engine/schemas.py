"""Schemas and source registry for the Smart Money Government Data Engine.

The engine is official-source-first. It records dataset caveats explicitly
because regulatory data are delayed, partial, and not equivalent across the
United States and Europe.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd


UNIFIED_ENTITY_COLUMNS = [
    "entity_id",
    "issuer_name",
    "filer_name",
    "cik",
    "lei",
    "isin",
    "cusip",
    "ticker",
    "country",
    "sector",
    "industry",
    "source",
    "match_confidence",
    "match_method",
]

HOLDING_COLUMNS = [
    "issuer_name",
    "filer_name",
    "cik",
    "lei",
    "isin",
    "cusip",
    "ticker",
    "country",
    "sector",
    "industry",
    "filing_date",
    "report_date",
    "source_form",
    "position_value_usd",
    "shares",
    "percent_portfolio",
    "ownership_percent",
    "long_short_flag",
    "instrument_type",
    "source",
    "source_path",
]

INSIDER_COLUMNS = [
    "issuer_name",
    "insider_name",
    "cik",
    "ticker",
    "filing_date",
    "report_date",
    "source_form",
    "insider_role",
    "transaction_code",
    "shares",
    "position_value_usd",
    "ownership_percent",
    "source",
    "source_path",
]

BENEFICIAL_EVENT_COLUMNS = [
    "issuer_name",
    "filer_name",
    "cik",
    "ticker",
    "filing_date",
    "report_date",
    "source_form",
    "ownership_percent",
    "event_type",
    "active_passive_flag",
    "source",
    "source_path",
]

MACRO_POSITIONING_COLUMNS = [
    "market",
    "asset_class",
    "report_date",
    "source_form",
    "managed_money_net",
    "commercial_net",
    "net_position_percentile",
    "trend_regime",
    "source",
    "source_path",
]

CAPITAL_FLOW_COLUMNS = [
    "country",
    "region",
    "asset_class",
    "report_date",
    "flow_usd",
    "holding_usd",
    "flow_percentile",
    "source",
    "source_path",
]

GOV_SPENDING_COLUMNS = [
    "recipient_name",
    "issuer_name",
    "ticker",
    "country",
    "sector",
    "award_date",
    "award_amount",
    "procurement_agency",
    "award_type",
    "source",
    "source_path",
]

SCORE_COLUMNS = [
    "ticker",
    "issuer_name",
    "country",
    "sector",
    "new_13f_holders",
    "ownership_change_score",
    "insider_conviction_score",
    "activist_pressure_score",
    "government_demand_tailwind_score",
    "cross_border_flow_support_score",
    "macro_positioning_score",
    "composite_institutional_interest_score",
    "coverage_count",
    "coverage_note",
]


@dataclass(frozen=True)
class SourceSpec:
    dataset_id: str
    region: str
    authority: str
    name: str
    use_case: str
    frequency: str
    format: str
    granularity: str
    key_fields: str
    official_url: str
    limitations: str
    priority: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


SOURCE_SPECS: list[SourceSpec] = [
    SourceSpec(
        "sec_13f",
        "USA",
        "SEC",
        "Form 13F Data Sets",
        "Institutional holdings, position changes, clone portfolios, crowdedness.",
        "Quarterly with filing lag",
        "ZIP/XML/TSV/CSV depending on SEC release",
        "manager-security holding snapshot",
        "cik, cusip, issuer_name, report_date",
        "https://www.sec.gov/data-research/sec-markets-data/form-13f-data-sets",
        "Delayed snapshots; no intraperiod trades; long US-listed holdings focus.",
        1,
    ),
    SourceSpec(
        "sec_nport",
        "USA",
        "SEC",
        "Form N-PORT Data Sets",
        "Registered fund holdings, liquidity profile, ETF/mutual fund transparency.",
        "Monthly/SEC dataset cadence",
        "ZIP/XML/CSV",
        "fund portfolio snapshot",
        "cik, series, isin/cusip, report_date",
        "https://www.sec.gov/data-research/sec-markets-data/form-n-port-data-sets",
        "Complex fund taxonomy; disclosure lag; not hedge fund transparency.",
        2,
    ),
    SourceSpec(
        "sec_13dg",
        "USA",
        "SEC EDGAR",
        "Schedules 13D / 13G",
        "Beneficial ownership above 5%, activist/passive event monitor.",
        "Event-driven",
        "EDGAR filings XML/HTML/text",
        "issuer-owner filing event",
        "cik, accession, filing_date",
        "https://www.sec.gov/edgar/search/",
        "Text extraction varies; amendments require careful event state tracking.",
        1,
    ),
    SourceSpec(
        "sec_form4",
        "USA",
        "SEC",
        "Insider Transactions Data Sets / Forms 3, 4, 5",
        "Insider buys/sells, cluster buying, officer/director conviction.",
        "Event-driven",
        "XML/CSV data sets",
        "insider transaction",
        "cik, accession, transaction_code, filing_date",
        "https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets",
        "Transaction intent is context-dependent; plans and grants need filtering.",
        1,
    ),
    SourceSpec(
        "cftc_cot",
        "USA",
        "CFTC",
        "Commitments of Traders",
        "Macro futures positioning, managed money trend/extreme percentiles.",
        "Weekly",
        "CSV/TXT",
        "market participant aggregate",
        "market, report_date",
        "https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm",
        "Aggregate futures positioning; not single-fund or issuer-level data.",
        2,
    ),
    SourceSpec(
        "treasury_tic",
        "USA",
        "U.S. Treasury",
        "Treasury International Capital",
        "Cross-border holdings and capital flow support by country/asset bucket.",
        "Monthly/quarterly depending table",
        "CSV/XLS",
        "country-asset flow/holding",
        "country, asset_class, report_date",
        "https://home.treasury.gov/data/treasury-international-capital-tic-system",
        "Macro flow proxy; not security-level trade attribution.",
        2,
    ),
    SourceSpec(
        "usaspending",
        "USA",
        "USAspending.gov",
        "USAspending API",
        "Federal contract/grant awards and government demand beneficiaries.",
        "Daily/API updates",
        "JSON API/CSV download",
        "award recipient",
        "recipient_name, award_id, award_date",
        "https://api.usaspending.gov/",
        "Recipient-to-listed-parent mapping is non-trivial and must be audited.",
        1,
    ),
    SourceSpec(
        "esma_firds",
        "EU",
        "ESMA",
        "FIRDS financial instrument reference data",
        "ISIN/MIC/issuer reference and EU security normalization.",
        "Daily/reference data cadence",
        "XML/CSV files",
        "instrument reference",
        "isin, lei, mic",
        "https://registers.esma.europa.eu/publication/",
        "Reference data, not ownership transparency.",
        3,
    ),
    SourceSpec(
        "ecb_sdw",
        "EU",
        "ECB",
        "Statistical Data Warehouse",
        "Rates, macro-financial aggregates and flow proxies.",
        "Series-dependent",
        "API/CSV",
        "macro time series",
        "series_key, date",
        "https://sdw.ecb.europa.eu/",
        "Macro context only; no issuer-level smart money visibility.",
        4,
    ),
    SourceSpec(
        "ted_procurement",
        "EU",
        "European Union",
        "TED public procurement",
        "EU public procurement awards and contractor exposure.",
        "Daily/publication cadence",
        "CSV/XML/API depending release",
        "procurement notice/award",
        "notice_id, recipient_name, date",
        "https://ted.europa.eu/",
        "Recipient normalization and listed-parent mapping are hard; country coverage differs.",
        3,
    ),
    SourceSpec(
        "national_disclosures",
        "EU/UK",
        "National regulators",
        "Major holdings / insider dealing / regulated disclosures",
        "Country-level beneficial ownership and insider proxies.",
        "Event-driven/country-specific",
        "HTML/CSV/PDF/API depending regulator",
        "filing event",
        "issuer, holder, date",
        "https://www.esma.europa.eu/",
        "Fragmented; no single EDGAR equivalent; each connector must carry caveats.",
        4,
    ),
]


def source_registry_frame() -> pd.DataFrame:
    """Return the canonical government/regulatory source registry."""
    return pd.DataFrame([spec.to_dict() for spec in SOURCE_SPECS]).sort_values(["priority", "region", "dataset_id"]).reset_index(drop=True)


def empty_frame(columns: list[str]) -> pd.DataFrame:
    """Create an empty analytics-ready frame with a stable schema."""
    return pd.DataFrame(columns=columns)


def government_dataset_matrix() -> pd.DataFrame:
    """Matrix requested by the product spec: dataset to use case, cadence and limits."""
    cols = ["dataset_id", "region", "authority", "name", "use_case", "frequency", "limitations", "priority"]
    return source_registry_frame()[cols]


def open_source_pattern_matrix() -> pd.DataFrame:
    """Map open-source project patterns to concrete uses in this platform."""
    rows = [
        {"repository": "QuantStats", "project_function": "HTML monitoring reports, summary cards, drawdown/performance table style.", "implementation": "report_builder emits compact HTML sections and exportable score tables."},
        {"repository": "pyfolio / pyfolio-reloaded", "project_function": "Tear-sheet discipline, rolling diagnostics, caveat-first analytics.", "implementation": "issuer/manager deep dives are split into overview, exposures, events and diagnostics."},
        {"repository": "Riskfolio-Lib", "project_function": "Concentration, overlap and contribution analysis patterns.", "implementation": "ownership analytics exposes concentration and manager overlap-ready tables."},
        {"repository": "Ghostfolio", "project_function": "Investment dashboard UX, positions, allocation navigation.", "implementation": "Streamlit page uses Overview, Screener, Issuer, Manager, Sector and Governance tabs."},
        {"repository": "Reverse-DCF-Data-Analysis", "project_function": "Scenario narrative and valuation context UX.", "implementation": "signals are designed to join valuation and portfolio modules by ticker/issuer."},
        {"repository": "Vibe-Trading", "project_function": "Research orchestration, repeatable agentic workflow, auditability.", "implementation": "pipeline exports manifests, caveats and report blocks for notebook/app orchestration."},
    ]
    return pd.DataFrame(rows)


def mvp_30_day_plan() -> pd.DataFrame:
    """Concrete 30-day build plan."""
    rows = [
        {"phase": "Week 1", "deliverable": "13F + Form 4 local parsers, source registry, schemas, Drive-first cache.", "success_metric": "Parse official files into holdings/insider tables with provenance."},
        {"phase": "Week 2", "deliverable": "Entity resolution CIK/CUSIP/ticker/name audit trail, smart money screener v1.", "success_metric": "Issuer-level score table with coverage notes and no fabricated fields."},
        {"phase": "Week 3", "deliverable": "13D/13G event monitor, USAspending recipient mapping, COT/TIC macro overlays.", "success_metric": "Event feed and government/macro tailwind columns exported."},
        {"phase": "Week 4", "deliverable": "Streamlit dashboard, notebook sections, HTML weekly report, EU proxy registry.", "success_metric": "Analyst can run weekly refresh and review top issuers/managers/sectors."},
    ]
    return pd.DataFrame(rows)


def dataset_priority_ranking() -> pd.DataFrame:
    """Priority ranking for implementation and refresh scheduling."""
    rows = [
        {"rank": 1, "dataset_id": "sec_13f", "why": "Highest signal for institutional positioning and clone portfolios.", "mvp_status": "implement parser first"},
        {"rank": 2, "dataset_id": "sec_form4", "why": "Event-driven insider conviction, especially cluster buying.", "mvp_status": "implement parser first"},
        {"rank": 3, "dataset_id": "sec_13dg", "why": "Activist/passive ownership shocks above 5%.", "mvp_status": "event monitor"},
        {"rank": 4, "dataset_id": "usaspending", "why": "Government demand tailwind and industrial policy beneficiaries.", "mvp_status": "API/cache connector"},
        {"rank": 5, "dataset_id": "cftc_cot", "why": "Macro positioning overlay for commodities/rates/FX themes.", "mvp_status": "weekly macro overlay"},
        {"rank": 6, "dataset_id": "treasury_tic", "why": "Cross-border capital flow support proxy.", "mvp_status": "monthly macro overlay"},
        {"rank": 7, "dataset_id": "sec_nport", "why": "Fund transparency beyond 13F, but more complex.", "mvp_status": "phase 2"},
        {"rank": 8, "dataset_id": "esma_firds", "why": "EU reference data and identifier resolution.", "mvp_status": "EU foundation"},
        {"rank": 9, "dataset_id": "ted_procurement", "why": "EU procurement exposure proxy.", "mvp_status": "EU phase 1"},
        {"rank": 10, "dataset_id": "national_disclosures", "why": "EU/UK ownership/insider proxies; fragmented connectors.", "mvp_status": "country-by-country"},
    ]
    return pd.DataFrame(rows)


def analyst_quick_wins() -> pd.DataFrame:
    """High-value quick wins for buy-side, macro and event-driven analysts."""
    rows = [
        {"persona": "Buy-side equity", "quick_win": "Top issuers with rising 13F holders and insider net buying.", "output": "Smart Money Screener"},
        {"persona": "Event-driven", "quick_win": "13D/13G activist event feed with ownership acceleration.", "output": "Activist Monitor"},
        {"persona": "Macro desk", "quick_win": "COT extreme percentiles plus TIC foreign demand shifts.", "output": "Macro Positioning Panel"},
        {"persona": "Industrial policy", "quick_win": "USAspending award acceleration by listed contractor ecosystem.", "output": "Government Demand Tailwind"},
        {"persona": "Portfolio manager", "quick_win": "Manager clone portfolio and crowdedness concentration table.", "output": "Manager Deep Dive"},
        {"persona": "EU analyst", "quick_win": "ESMA identifier mapping plus TED procurement proxy coverage map.", "output": "EU Proxy Monitor"},
    ]
    return pd.DataFrame(rows)
