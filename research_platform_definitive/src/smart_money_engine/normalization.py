"""Normalization utilities for government/regulatory smart-money data."""

from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd

from .schemas import (
    BENEFICIAL_EVENT_COLUMNS,
    CAPITAL_FLOW_COLUMNS,
    GOV_SPENDING_COLUMNS,
    HOLDING_COLUMNS,
    INSIDER_COLUMNS,
    MACRO_POSITIONING_COLUMNS,
    empty_frame,
)


def normalize_label(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).upper().strip()
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    text = re.sub(r"\b(INC|CORP|CORPORATION|PLC|SA|SPA|NV|AG|LTD|LLC|LP|CO|COMPANY)\b", "", text)
    return re.sub(r"\s+", " ", text).strip()


def first_col(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    lookup = {c.lower().replace("_", "").replace(" ", ""): c for c in df.columns}
    for candidate in candidates:
        key = candidate.lower().replace("_", "").replace(" ", "")
        if key in lookup:
            return lookup[key]
    return None


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False), errors="coerce")


def date_series(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.date.astype("string")


def _stable_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if df.empty:
        return empty_frame(columns)
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = pd.NA
    return out[columns]


def standardize_13f(raw: pd.DataFrame, source_path: str = "") -> pd.DataFrame:
    """Normalize common SEC 13F flattened fields into holding schema."""
    if raw.empty:
        return empty_frame(HOLDING_COLUMNS)
    out = pd.DataFrame()
    out["issuer_name"] = raw[first_col(raw, ["nameofissuer", "name_of_issuer", "issuer_name", "issuer"]) or raw.columns[0]].astype("string")
    filer = first_col(raw, ["filingmanagername", "manager_name", "filer_name", "name"])
    out["filer_name"] = raw[filer].astype("string") if filer else pd.NA
    for target, aliases in {
        "cik": ["cik", "manager_cik", "filer_cik"],
        "cusip": ["cusip", "cusip6"],
        "ticker": ["ticker", "symbol"],
        "shares": ["sshprnamt", "shares", "share_amount"],
        "position_value_usd": ["value", "value_usd", "market_value", "position_value_usd"],
        "report_date": ["periodofreport", "report_date", "period_date"],
        "filing_date": ["filing_date", "filed_date"],
    }.items():
        col = first_col(raw, aliases)
        out[target] = raw[col] if col else pd.NA
    out["position_value_usd"] = numeric(out["position_value_usd"]) * 1000
    out["shares"] = numeric(out["shares"])
    out["filing_date"] = date_series(out["filing_date"]) if out["filing_date"].notna().any() else pd.NA
    out["report_date"] = date_series(out["report_date"]) if out["report_date"].notna().any() else pd.NA
    out["source_form"] = "13F"
    out["source"] = "SEC 13F"
    out["source_path"] = source_path
    return _stable_frame(out, HOLDING_COLUMNS)


def standardize_form4(raw: pd.DataFrame, source_path: str = "") -> pd.DataFrame:
    """Normalize SEC Form 4 / insider transaction-like files."""
    if raw.empty:
        return empty_frame(INSIDER_COLUMNS)
    out = pd.DataFrame()
    for target, aliases in {
        "issuer_name": ["issuer_name", "issuer", "company_name", "nameofissuer"],
        "insider_name": ["rptownername", "insider_name", "owner_name", "reporting_owner"],
        "cik": ["issuer_cik", "cik"],
        "ticker": ["ticker", "symbol"],
        "filing_date": ["filing_date", "filed_date"],
        "report_date": ["transaction_date", "report_date", "periodofreport"],
        "insider_role": ["officer_title", "relationship", "insider_role"],
        "transaction_code": ["transaction_code", "transactioncode", "code"],
        "shares": ["transaction_shares", "shares", "shares_owned"],
        "position_value_usd": ["transaction_value", "value_usd", "price_total"],
    }.items():
        col = first_col(raw, aliases)
        out[target] = raw[col] if col else pd.NA
    out["shares"] = numeric(out["shares"])
    out["position_value_usd"] = numeric(out["position_value_usd"])
    out["filing_date"] = date_series(out["filing_date"]) if out["filing_date"].notna().any() else pd.NA
    out["report_date"] = date_series(out["report_date"]) if out["report_date"].notna().any() else pd.NA
    out["source_form"] = "Form 4"
    out["source"] = "SEC Insider Transactions"
    out["source_path"] = source_path
    return _stable_frame(out, INSIDER_COLUMNS)


def standardize_13dg(raw: pd.DataFrame, source_path: str = "") -> pd.DataFrame:
    if raw.empty:
        return empty_frame(BENEFICIAL_EVENT_COLUMNS)
    out = pd.DataFrame()
    for target, aliases in {
        "issuer_name": ["issuer_name", "subject_company", "company_name"],
        "filer_name": ["filer_name", "owner_name", "reporting_person"],
        "cik": ["cik", "issuer_cik"],
        "ticker": ["ticker", "symbol"],
        "filing_date": ["filing_date", "filed_date"],
        "report_date": ["event_date", "report_date"],
        "ownership_percent": ["ownership_percent", "percent_owned", "percent"],
        "source_form": ["form", "source_form"],
    }.items():
        col = first_col(raw, aliases)
        out[target] = raw[col] if col else pd.NA
    out["ownership_percent"] = numeric(out["ownership_percent"])
    form = out["source_form"].astype(str).str.upper()
    out["event_type"] = np.where(form.str.contains("13D", na=False), "activist_or_active", "beneficial_ownership")
    out["active_passive_flag"] = np.where(form.str.contains("13D", na=False), "active", "passive_or_exempt")
    out["source"] = "SEC EDGAR 13D/13G"
    out["source_path"] = source_path
    return _stable_frame(out, BENEFICIAL_EVENT_COLUMNS)


def standardize_cot(raw: pd.DataFrame, source_path: str = "") -> pd.DataFrame:
    if raw.empty:
        return empty_frame(MACRO_POSITIONING_COLUMNS)
    out = pd.DataFrame()
    market = first_col(raw, ["market_and_exchange_names", "market", "contract_market_name"])
    out["market"] = raw[market].astype("string") if market else pd.NA
    date = first_col(raw, ["report_date_as_yyyy_mm_dd", "report_date", "date"])
    out["report_date"] = date_series(raw[date]) if date else pd.NA
    managed_long = first_col(raw, ["m_money_positions_long_all", "managed_money_long"])
    managed_short = first_col(raw, ["m_money_positions_short_all", "managed_money_short"])
    commercial_long = first_col(raw, ["commercial_positions_long_all", "commercial_long"])
    commercial_short = first_col(raw, ["commercial_positions_short_all", "commercial_short"])
    out["managed_money_net"] = numeric(raw[managed_long]) - numeric(raw[managed_short]) if managed_long and managed_short else pd.NA
    out["commercial_net"] = numeric(raw[commercial_long]) - numeric(raw[commercial_short]) if commercial_long and commercial_short else pd.NA
    out["net_position_percentile"] = out.groupby("market")["managed_money_net"].rank(pct=True) if out["managed_money_net"].notna().any() else pd.NA
    out["trend_regime"] = pd.cut(pd.to_numeric(out["net_position_percentile"], errors="coerce"), bins=[-0.01, 0.2, 0.8, 1.01], labels=["bearish_extreme", "neutral", "bullish_extreme"]).astype("string")
    out["asset_class"] = pd.NA
    out["source_form"] = "COT"
    out["source"] = "CFTC COT"
    out["source_path"] = source_path
    return _stable_frame(out, MACRO_POSITIONING_COLUMNS)


def standardize_tic(raw: pd.DataFrame, source_path: str = "") -> pd.DataFrame:
    if raw.empty:
        return empty_frame(CAPITAL_FLOW_COLUMNS)
    out = pd.DataFrame()
    for target, aliases in {
        "country": ["country", "country_name"],
        "region": ["region"],
        "asset_class": ["asset_class", "security_type", "type"],
        "report_date": ["date", "report_date", "period"],
        "flow_usd": ["flow_usd", "net_flow", "net_purchases"],
        "holding_usd": ["holding_usd", "holdings", "position"],
    }.items():
        col = first_col(raw, aliases)
        out[target] = raw[col] if col else pd.NA
    out["flow_usd"] = numeric(out["flow_usd"])
    out["holding_usd"] = numeric(out["holding_usd"])
    out["flow_percentile"] = out.groupby("asset_class")["flow_usd"].rank(pct=True) if out["flow_usd"].notna().any() else pd.NA
    out["source"] = "Treasury TIC"
    out["source_path"] = source_path
    return _stable_frame(out, CAPITAL_FLOW_COLUMNS)


def standardize_spending(raw: pd.DataFrame, source_path: str = "") -> pd.DataFrame:
    if raw.empty:
        return empty_frame(GOV_SPENDING_COLUMNS)
    out = pd.DataFrame()
    for target, aliases in {
        "recipient_name": ["recipient_name", "awardee_or_recipient_legal", "legal_business_name"],
        "issuer_name": ["issuer_name", "parent_recipient_name", "recipient_parent_name"],
        "ticker": ["ticker", "symbol"],
        "country": ["recipient_country", "country"],
        "sector": ["sector", "naics_description"],
        "award_date": ["award_date", "action_date", "date"],
        "award_amount": ["award_amount", "federal_action_obligation", "amount"],
        "procurement_agency": ["awarding_agency_name", "agency", "procurement_agency"],
        "award_type": ["award_type", "type"],
    }.items():
        col = first_col(raw, aliases)
        out[target] = raw[col] if col else pd.NA
    out["award_amount"] = numeric(out["award_amount"])
    out["source"] = "USAspending"
    out["source_path"] = source_path
    return _stable_frame(out, GOV_SPENDING_COLUMNS)
