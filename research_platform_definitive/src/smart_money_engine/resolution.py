"""Entity resolution with deterministic keys and auditable fuzzy fallback."""

from __future__ import annotations

from difflib import SequenceMatcher

import pandas as pd

from .normalization import normalize_label
from .schemas import UNIFIED_ENTITY_COLUMNS, empty_frame


def build_entity_master(*frames: pd.DataFrame) -> pd.DataFrame:
    """Build a lightweight issuer/filer master from normalized dataframes."""
    rows: list[dict[str, object]] = []
    for frame in frames:
        if frame is None or frame.empty:
            continue
        for _, row in frame.iterrows():
            issuer = row.get("issuer_name", pd.NA)
            filer = row.get("filer_name", pd.NA)
            name = issuer if pd.notna(issuer) and str(issuer).strip() else filer
            def clean(value: object) -> str:
                return "" if pd.isna(value) else str(value).strip()

            key_parts = [clean(row.get("cik", "")), clean(row.get("cusip", "")), clean(row.get("isin", "")), normalize_label(name)]
            entity_id = "|".join([p for p in key_parts if p and p != "<NA>"]) or normalize_label(name)
            rows.append({
                "entity_id": entity_id,
                "issuer_name": issuer,
                "filer_name": filer,
                "cik": row.get("cik", pd.NA),
                "lei": row.get("lei", pd.NA),
                "isin": row.get("isin", pd.NA),
                "cusip": row.get("cusip", pd.NA),
                "ticker": row.get("ticker", pd.NA),
                "country": row.get("country", pd.NA),
                "sector": row.get("sector", pd.NA),
                "industry": row.get("industry", pd.NA),
                "source": row.get("source", pd.NA),
                "match_confidence": 1.0 if any(clean(row.get(k, "")) for k in ["cik", "cusip", "isin", "ticker"]) else 0.75,
                "match_method": "deterministic_id" if any(clean(row.get(k, "")) for k in ["cik", "cusip", "isin", "ticker"]) else "normalized_name",
            })
    if not rows:
        return empty_frame(UNIFIED_ENTITY_COLUMNS)
    out = pd.DataFrame(rows)
    out = out.sort_values(["match_confidence"], ascending=False).drop_duplicates("entity_id", keep="first")
    for col in UNIFIED_ENTITY_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    return out[UNIFIED_ENTITY_COLUMNS].reset_index(drop=True)


def fuzzy_match_name(name: str, candidates: pd.Series, threshold: float = 0.88) -> tuple[str | None, float]:
    """Return best normalized-name match and score, or None if below threshold."""
    needle = normalize_label(name)
    best_name: str | None = None
    best_score = 0.0
    for candidate in candidates.dropna().astype(str).unique():
        score = SequenceMatcher(None, needle, normalize_label(candidate)).ratio()
        if score > best_score:
            best_name, best_score = candidate, score
    if best_score >= threshold:
        return best_name, best_score
    return None, best_score


def match_awards_to_issuers(awards: pd.DataFrame, entity_master: pd.DataFrame) -> pd.DataFrame:
    """Attach issuer candidates to government awards using audited deterministic/fuzzy logic."""
    if awards.empty or entity_master.empty:
        return awards.copy()
    out = awards.copy()
    issuer_names = entity_master["issuer_name"].dropna().astype(str)
    matched_names: list[str | None] = []
    scores: list[float] = []
    methods: list[str] = []
    for _, row in out.iterrows():
        recipient = str(row.get("recipient_name", "") or "")
        existing = row.get("issuer_name", pd.NA)
        if pd.notna(existing) and str(existing).strip():
            matched_names.append(str(existing))
            scores.append(1.0)
            methods.append("provided_issuer")
            continue
        match, score = fuzzy_match_name(recipient, issuer_names, threshold=0.9)
        matched_names.append(match)
        scores.append(score)
        methods.append("fuzzy_name" if match else "unmatched")
    out["issuer_name"] = out.get("issuer_name", pd.Series([pd.NA] * len(out))).fillna(pd.Series(matched_names))
    out["match_confidence"] = scores
    out["match_method"] = methods
    return out
