"""Open banking data pipeline helpers for Italian and euro-area bank research.

The module is intentionally notebook-friendly: it prefers public/open sources,
uses local caching, writes reproducible CSV/SQLite artifacts, and degrades
cleanly when an official source requires manual export configuration.
"""

from __future__ import annotations

import hashlib
from io import StringIO
import json
import logging
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

LOGGER = logging.getLogger(__name__)

USER_AGENT = "research-platform-banking-data/1.0"
REQUEST_TIMEOUT = 30

ITALIAN_LISTED_BANK_TICKERS = {
    "INTESA SANPAOLO": "ISP.MI",
    "UNICREDIT": "UCG.MI",
    "BANCO BPM": "BAMI.MI",
    "BPER BANCA": "BPE.MI",
    "BANCA MONTE DEI PASCHI DI SIENA": "BMPS.MI",
    "FINECOBANK": "FBK.MI",
    "BANCA GENERALI": "BGN.MI",
    "CREDITO EMILIANO": "CE.MI",
    "BANCA MEDIOLANUM": "BMED.MI",
    "MEDIOBANCA": "MB.MI",
    "BANCA IFIS": "IF.MI",
    "ILLIMITY BANK": "ILTY.MI",
}

DEFAULT_BANK_IR_PAGES = {
    "Intesa Sanpaolo": "https://group.intesasanpaolo.com/en/investor-relations",
    "UniCredit": "https://www.unicreditgroup.eu/en/investors.html",
    "Banco BPM": "https://gruppo.bancobpm.it/en/investor-relations/",
    "BPER Banca": "https://istituzionale.bper.it/en/investor-relations",
    "MPS": "https://www.gruppomps.it/en/investor-relations/",
    "FinecoBank": "https://finecobank.com/en/online/investors/",
    "CREDEM": "https://www.credem.it/content/credem/en/investor-relations.html",
}


def curated_listed_bank_universe() -> pd.DataFrame:
    """Return a minimal listed Italian bank universe from curated ticker mappings."""
    rows: list[dict[str, Any]] = []
    for idx, (name, ticker) in enumerate(ITALIAN_LISTED_BANK_TICKERS.items(), start=1):
        rows.append(
            {
                "bank_id": f"LISTED_IT_{idx:03d}",
                "legal_name": name.title(),
                "normalized_name": normalize_bank_name(name),
                "country": "IT",
                "is_significant": name in {"INTESA SANPAOLO", "UNICREDIT", "BANCO BPM", "BPER BANCA"},
                "is_LSI": False,
                "status": "listed_curated",
                "listed_flag": True,
                "ticker": ticker,
                "group_name": infer_group_name(name),
                "bank_category": "listed_bank",
                "license_type": np.nan,
                "lei": np.nan,
                "bic": np.nan,
                "swift": np.nan,
                "source": "curated_italian_listed_bank_tickers",
            }
        )
    return pd.DataFrame(rows)

FUNDAMENTALS_SCHEMA = [
    "bank_id",
    "date",
    "roe",
    "roa",
    "nim",
    "ci_ratio",
    "npl_ratio",
    "npl_gross",
    "npl_net",
    "coverage_ratio",
    "cet1",
    "tier1",
    "total_capital",
    "ldr",
    "loans_to_customers",
    "customer_deposits",
    "total_assets",
    "equity",
    "net_income",
    "sovereign_exposure",
    "source",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def slugify(value: Any) -> str:
    value = str(value or "").strip().lower()
    value = re.sub(r"[^\w\s-]", "", value)
    value = re.sub(r"[\s_-]+", "_", value)
    return value.strip("_")


def normalize_bank_name(name: Any) -> str:
    if pd.isna(name):
        return ""
    text = str(name).upper()
    patterns = [
        r"\bS\.P\.A\.?\b",
        r"\bSPA\b",
        r"\bSOCIETA PER AZIONI\b",
        r"\bS\.C\.P\.A\.?\b",
        r"\bSCPA\b",
        r"\bBANCA\b",
        r"\bBANK\b",
        r"\bGRUPPO\b",
        r"\bGROUP\b",
        r"\bCOOPERATIVA\b",
        r"\bCREDITO COOPERATIVO\b",
        r"\bLIMITED\b",
        r"\bLTD\b",
        r"\bPLC\b",
    ]
    for pattern in patterns:
        text = re.sub(pattern, " ", text)
    text = re.sub(r"[^A-Z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _cache_path(cache_dir: Path, url: str, suffix: str | None = None) -> Path:
    suffix = suffix or Path(url.split("?")[0]).suffix or ".bin"
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    return Path(cache_dir) / f"{digest}{suffix}"


def _is_fresh(path: Path, max_age_days: int) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
    return age <= timedelta(days=max_age_days)


def cached_get(
    url: str,
    cache_dir: str | Path,
    cache_path: str | Path | None = None,
    max_age_days: int = 7,
    force: bool = False,
) -> bytes:
    cache_dir = Path(cache_dir)
    path = Path(cache_path) if cache_path else _cache_path(cache_dir, url)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not force and _is_fresh(path, max_age_days):
        return path.read_bytes()
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    path.write_bytes(response.content)
    return response.content


def fetch_wikipedia_italian_banks(cache_dir: str | Path) -> pd.DataFrame:
    """Fetch a non-regulatory seed list of Italian banks from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_banks_in_Italy"
    html = cached_get(url, cache_dir=cache_dir, max_age_days=14).decode("utf-8", errors="ignore")
    frames: list[pd.DataFrame] = []
    for idx, table in enumerate(pd.read_html(StringIO(html))):
        tbl = table.copy()
        tbl.columns = [slugify(c) for c in tbl.columns]
        name_col = next((c for c in tbl.columns if "name" in c or "bank" in c), None)
        if not name_col:
            continue
        out = pd.DataFrame(
            {
                "legal_name": tbl[name_col].astype(str).str.replace(r"\[.*?\]", "", regex=True).str.strip(),
                "country": "IT",
                "source": "wikipedia_list_of_banks_in_italy",
                "source_table": idx,
            }
        )
        for col in tbl.columns:
            if col != name_col:
                out[f"wiki_{col}"] = tbl[col]
        frames.append(out)
    if not frames:
        return pd.DataFrame(columns=["legal_name", "country", "source", "normalized_name"])
    df = pd.concat(frames, ignore_index=True, sort=False)
    df = df[df["legal_name"].notna() & df["legal_name"].str.len().gt(2)].copy()
    df["normalized_name"] = df["legal_name"].map(normalize_bank_name)
    return df.drop_duplicates(subset=["normalized_name"]).reset_index(drop=True)


def download_ecb_supervised_entities_pdf(
    cache_dir: str | Path,
    force: bool = False,
) -> Path:
    """Download ECB's supervised entities PDF using the stable public PDF fallback."""
    url = "https://www.ecb.europa.eu/pub/pdf/other/intro_list_sse_lsi.en.pdf"
    path = Path(cache_dir) / "ecb_supervised_entities.pdf"
    cached_get(url, cache_dir=cache_dir, cache_path=path, max_age_days=14, force=force)
    return path


def parse_ecb_supervised_entities_pdf(pdf_path: str | Path) -> pd.DataFrame:
    """Parse ECB SI/LSI PDF with optional PDF libraries; returns an auditable raw-text table."""
    path = Path(pdf_path)
    rows: list[dict[str, Any]] = []
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        current_status = None
        for page in reader.pages:
            text = page.extract_text() or ""
            for line in text.splitlines():
                line = re.sub(r"\s+", " ", line).strip()
                if not line:
                    continue
                low = line.lower()
                if "significant supervised entities" in low:
                    current_status = "SI"
                elif "less significant" in low:
                    current_status = "LSI"
                country = re.search(r"\b(IT|DE|FR|ES|NL|BE|AT|IE|PT|FI|GR|LU|SI|SK|EE|LV|LT|CY|MT|HR)\b", line)
                if country and len(line) > 6:
                    rows.append(
                        {
                            "legal_name": line,
                            "country": country.group(1),
                            "is_significant": current_status == "SI",
                            "is_LSI": current_status == "LSI",
                            "source": "ecb_supervised_entities_pdf_text",
                        }
                    )
    except Exception as exc:
        LOGGER.info("ECB PDF text parser unavailable or failed: %s", exc)

    if not rows:
        return pd.DataFrame(columns=["legal_name", "country", "is_significant", "is_LSI", "source", "normalized_name"])
    df = pd.DataFrame(rows)
    df["normalized_name"] = df["legal_name"].map(normalize_bank_name)
    return df[df["normalized_name"].str.len().gt(2)].drop_duplicates(["normalized_name", "country"]).reset_index(drop=True)


def ecb_get_series(
    dataset_id: str,
    key: str,
    cache_dir: str | Path,
    params: dict[str, Any] | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Generic ECB SDMX 2.1 REST CSV call."""
    params = params or {}
    query = "&".join(f"{name}={value}" for name, value in params.items())
    url = f"https://data-api.ecb.europa.eu/service/data/{dataset_id}/{key}"
    if query:
        url = f"{url}?{query}"
    path = Path(cache_dir) / "ecb" / f"{dataset_id}_{slugify(key)}_{hashlib.md5(url.encode()).hexdigest()[:8]}.csv"
    if not force and _is_fresh(path, 7):
        return pd.read_csv(path)
    content = cached_get(url, cache_dir=cache_dir, cache_path=path, max_age_days=7, force=force)
    text = content.decode("utf-8", errors="ignore")
    if text.strip().startswith("<"):
        xml_path = path.with_suffix(".xml")
        xml_path.write_text(text, encoding="utf-8")
        return pd.DataFrame({"dataset_id": [dataset_id], "series_key": [key], "raw_xml_path": [str(xml_path)]})
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, sep=";")


def build_ecb_bsi_macro_panel(
    cache_dir: str | Path,
    countries: list[str] | None = None,
    start_period: str = "2015-01",
) -> pd.DataFrame:
    """Build a small configurable ECB BSI panel; invalid keys are logged and skipped."""
    countries = countries or ["IT", "DE", "FR", "ES", "NL"]
    configs = [
        {
            "item_code": "bsi_total_assets_candidate",
            "dataset_id": "BSI",
            "key_template": "M.{country}.N.A.A20.A.1.U2.2300.Z01.E",
            "description": "MFI BSI candidate key; validate/extend from ECB Data Portal as needed.",
        }
    ]
    frames: list[pd.DataFrame] = []
    for country in countries:
        for spec in configs:
            key = spec["key_template"].format(country=country)
            try:
                df = ecb_get_series(spec["dataset_id"], key, cache_dir=cache_dir, params={"startPeriod": start_period, "format": "csvdata"})
                df["country"] = country
                df["item_code"] = spec["item_code"]
                df["description"] = spec["description"]
                frames.append(df)
            except Exception as exc:
                LOGGER.warning("ECB series skipped country=%s key=%s error=%s", country, key, exc)
    if not frames:
        return pd.DataFrame(columns=["date", "country", "item_code", "value", "unit", "sector", "source"])
    raw = pd.concat(frames, ignore_index=True, sort=False)
    time_col = next((c for c in raw.columns if str(c).upper() in {"TIME_PERIOD", "TIME"}), None)
    value_col = next((c for c in raw.columns if str(c).upper() in {"OBS_VALUE", "VALUE"}), None)
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(raw[time_col], errors="coerce") if time_col else pd.NaT,
            "country": raw.get("country"),
            "item_code": raw.get("item_code"),
            "value": pd.to_numeric(raw[value_col], errors="coerce") if value_col else np.nan,
            "unit": raw.get("UNIT_MEASURE", np.nan),
            "sector": "banks_mfi",
            "source": "ecb_bsi_sdmx",
        }
    )
    return out.dropna(subset=["date"], how="all").reset_index(drop=True)


def load_bancaditalia_bds_file(filepath: str | Path) -> pd.DataFrame:
    """Normalize a CSV/XLSX exported from Banca d'Italia BDS."""
    path = Path(filepath)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        raw = pd.read_excel(path)
    else:
        try:
            raw = pd.read_csv(path)
        except Exception:
            raw = pd.read_csv(path, sep=";")
    raw.columns = [slugify(c) for c in raw.columns]
    date_col = next((c for c in raw.columns if c in {"date", "data", "periodo", "time_period"}), None)
    value_col = next((c for c in raw.columns if c in {"value", "valore", "obs_value"}), None)
    return pd.DataFrame(
        {
            "date": pd.to_datetime(raw[date_col], errors="coerce") if date_col else pd.NaT,
            "item_code": raw.get("item_code", raw.get("codice", path.stem)),
            "value": pd.to_numeric(raw[value_col], errors="coerce") if value_col else np.nan,
            "unit": raw.get("unit", raw.get("unita", np.nan)),
            "sector": raw.get("sector", "banks"),
            "country": "IT",
            "source": "bancaditalia_bds_export",
            "raw_file": str(path),
        }
    )


def fetch_bancaditalia_macro_regulatory(
    cache_dir: str | Path,
    exports: dict[str, str] | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Download and normalize configured public Banca d'Italia BDS CSV/XLSX exports."""
    frames: list[pd.DataFrame] = []
    for name, url in (exports or {}).items():
        try:
            suffix = Path(url.split("?")[0]).suffix or ".csv"
            path = Path(cache_dir) / "bancaditalia" / f"{slugify(name)}{suffix}"
            cached_get(url, cache_dir=cache_dir, cache_path=path, max_age_days=7, force=force)
            df = load_bancaditalia_bds_file(path)
            df["item_code"] = name
            frames.append(df)
        except Exception as exc:
            LOGGER.warning("Banca d'Italia export skipped name=%s error=%s", name, exc)
    if not frames:
        return pd.DataFrame(columns=["date", "item_code", "value", "unit", "sector", "country", "source"])
    return pd.concat(frames, ignore_index=True, sort=False)


def infer_bank_category(row: pd.Series) -> str:
    text = " ".join(str(v) for v in row.values if pd.notna(v)).lower()
    if any(token in text for token in ["cooperative", "credito cooperativo", "bcc", "popolare"]):
        return "cooperative"
    if any(token in text for token in ["online", "fineco", "illimity"]):
        return "online_or_digital"
    if any(token in text for token in ["investment", "mediobanca"]):
        return "investment"
    if any(token in text for token in ["private banking", "wealth"]):
        return "private_banking"
    return "retail_commercial"


def infer_group_name(name: Any) -> str:
    norm = normalize_bank_name(name)
    rules = {
        "INTESA": "Intesa Sanpaolo Group",
        "UNICREDIT": "UniCredit Group",
        "BANCO BPM": "Banco BPM Group",
        "BPER": "BPER Group",
        "MONTE PASCHI": "MPS Group",
        "MEDIOBANCA": "Mediobanca Group",
        "FINECO": "FinecoBank",
        "MEDIOLANUM": "Mediolanum Group",
        "GENERALI": "Generali Group",
        "CREDITO EMILIANO": "CREDEM Group",
        "IFIS": "Banca IFIS Group",
    }
    for token, group in rules.items():
        if token in norm:
            return group
    return str(name)


def attach_listed_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["listed_flag"] = False
    out["ticker"] = np.nan
    tickers = {normalize_bank_name(name): ticker for name, ticker in ITALIAN_LISTED_BANK_TICKERS.items()}
    for idx, row in out.iterrows():
        norm = str(row.get("normalized_name", ""))
        for bank_norm, ticker in tickers.items():
            if bank_norm in norm or norm in bank_norm:
                out.loc[idx, "listed_flag"] = True
                out.loc[idx, "ticker"] = ticker
                break
    return out


def build_banks_universe(
    cache_dir: str | Path,
    include_wikipedia: bool = True,
    include_ecb: bool = True,
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Build and optionally persist a normalized Italian/euro-area bank universe."""
    frames: list[pd.DataFrame] = []
    if include_wikipedia:
        try:
            frames.append(fetch_wikipedia_italian_banks(cache_dir))
        except Exception as exc:
            LOGGER.warning("Wikipedia bank list skipped: %s", exc)
    if include_ecb:
        try:
            pdf = download_ecb_supervised_entities_pdf(cache_dir)
            frames.append(parse_ecb_supervised_entities_pdf(pdf))
        except Exception as exc:
            LOGGER.warning("ECB supervised entities skipped: %s", exc)
    if not frames:
        raise RuntimeError("No bank universe source succeeded.")
    raw = pd.concat(frames, ignore_index=True, sort=False)
    raw["legal_name"] = raw["legal_name"].astype(str).str.strip()
    raw["country"] = raw["country"].fillna("IT")
    raw["normalized_name"] = raw["legal_name"].map(normalize_bank_name)
    raw = raw[raw["normalized_name"].str.len().gt(2)].copy()
    for col in ["is_significant", "is_LSI"]:
        if col not in raw.columns:
            raw[col] = False
    agg = (
        raw.groupby(["normalized_name", "country"], as_index=False)
        .agg(
            legal_name=("legal_name", "first"),
            source=("source", lambda x: "|".join(sorted(set(map(str, x))))),
            is_significant=("is_significant", "max"),
            is_LSI=("is_LSI", "max"),
        )
    )
    agg["is_significant"] = agg["is_significant"].map(lambda value: bool(value) if pd.notna(value) else False)
    agg["is_LSI"] = agg["is_LSI"].map(lambda value: bool(value) if pd.notna(value) else False)
    agg["group_name"] = agg["legal_name"].map(infer_group_name)
    agg["bank_category"] = agg.apply(infer_bank_category, axis=1)
    agg["status"] = np.select(
        [agg["is_significant"], agg["is_LSI"], agg["country"].ne("IT")],
        ["significant_institution", "less_significant_institution", "euro_area_peer"],
        default="unclassified",
    )
    agg = attach_listed_flags(agg)
    agg["bank_id"] = [f"BANK_{i:06d}" for i in range(1, len(agg) + 1)]
    for col in ["lei", "bic", "swift", "license_type"]:
        agg[col] = np.nan
    cols = [
        "bank_id",
        "legal_name",
        "normalized_name",
        "country",
        "is_significant",
        "is_LSI",
        "status",
        "listed_flag",
        "ticker",
        "group_name",
        "bank_category",
        "license_type",
        "lei",
        "bic",
        "swift",
        "source",
    ]
    out = agg[cols].sort_values(["country", "legal_name"]).reset_index(drop=True)
    curated = curated_listed_bank_universe()
    known_tickers = set(out["ticker"].dropna().astype(str))
    known_names = set(out["normalized_name"].dropna().astype(str))
    curated = curated[
        ~curated["ticker"].astype(str).isin(known_tickers)
        & ~curated["normalized_name"].astype(str).isin(known_names)
    ]
    if not curated.empty:
        out = pd.concat([out, curated[cols]], ignore_index=True, sort=False).sort_values(["country", "legal_name"]).reset_index(drop=True)
        out["bank_id"] = [str(value) if str(value).startswith("LISTED_IT_") else f"BANK_{i:06d}" for i, value in enumerate(out["bank_id"], start=1)]
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_dir / "banks_universe.csv", index=False)
    return out


def _safe_info_value(info: dict[str, Any], names: list[str], default: Any = np.nan) -> Any:
    for name in names:
        value = info.get(name)
        if value is not None:
            return value
    return default


def build_yfinance_bank_panel(
    tickers: list[str],
    start: str = "2015-01-01",
    end: str | None = None,
    price_col_name: str = "mkt_price",
    market_cap_col_name: str = "mkt_market_cap",
    save_path: str | Path | None = None,
) -> pd.DataFrame:
    """Build a monthly listed-bank panel from Yahoo Finance with explicit proxy flags."""
    try:
        import yfinance as yf
    except Exception as exc:
        raise ImportError("yfinance non disponibile. Installa con: pip install yfinance oppure usa un CSV.") from exc
    tickers = [str(t).strip().upper() for t in tickers if str(t).strip()]
    if not tickers:
        raise ValueError("Nessun ticker disponibile per costruire il panel yfinance.")
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False, group_by="ticker", threads=True)
    frames: list[pd.DataFrame] = []
    for ticker in tickers:
        if isinstance(raw.columns, pd.MultiIndex):
            if ticker not in raw.columns.get_level_values(0):
                continue
            px = raw[ticker].copy()
        else:
            px = raw.copy()
        if px.empty:
            continue
        price_col = "Close" if "Close" in px.columns else "Adj Close" if "Adj Close" in px.columns else None
        if not price_col:
            continue
        monthly = px[[price_col]].dropna().resample("ME").last().rename(columns={price_col: price_col_name})
        if monthly.empty:
            continue
        monthly["ticker"] = ticker
        monthly["date"] = monthly.index
        try:
            info = getattr(yf.Ticker(ticker), "info", {}) or {}
        except Exception:
            info = {}
        current_price = float(monthly[price_col_name].dropna().iloc[-1]) if monthly[price_col_name].notna().any() else np.nan
        market_cap_now = pd.to_numeric(_safe_info_value(info, ["marketCap"]), errors="coerce")
        shares_proxy = market_cap_now / current_price if pd.notna(market_cap_now) and current_price and current_price > 0 else np.nan
        monthly[market_cap_col_name] = monthly[price_col_name] * shares_proxy if pd.notna(shares_proxy) else monthly[price_col_name]
        monthly["target_source"] = "yfinance_marketcap_proxy" if pd.notna(shares_proxy) else "yfinance_price_proxy"
        monthly["country"] = "IT"
        monthly["sector"] = "Banks"
        monthly["company_name"] = _safe_info_value(info, ["longName", "shortName"], ticker)
        monthly["mkt_pe"] = pd.to_numeric(_safe_info_value(info, ["trailingPE", "forwardPE"]), errors="coerce")
        monthly["mkt_pb"] = pd.to_numeric(_safe_info_value(info, ["priceToBook"]), errors="coerce")
        monthly["mkt_beta_bank"] = pd.to_numeric(_safe_info_value(info, ["beta"]), errors="coerce")
        monthly["fund_roa"] = pd.to_numeric(_safe_info_value(info, ["returnOnAssets"]), errors="coerce")
        monthly["fund_roe"] = pd.to_numeric(_safe_info_value(info, ["returnOnEquity"]), errors="coerce")
        monthly["fund_assets"] = pd.to_numeric(_safe_info_value(info, ["totalAssets"]), errors="coerce")
        monthly["data_source"] = "yfinance"
        frames.append(monthly.reset_index(drop=True))
    if not frames:
        raise ValueError("Nessun prezzo scaricato da yfinance. Controlla ticker o connessione.")
    out = pd.concat(frames, ignore_index=True, sort=False).sort_values(["ticker", "date"])
    out["ret_1m"] = out.groupby("ticker")[price_col_name].pct_change()
    out["mkt_vol_1y"] = out.groupby("ticker")["ret_1m"].rolling(12, min_periods=4).std().reset_index(level=0, drop=True)
    out["mkt_mom_6m"] = out.groupby("ticker")[price_col_name].pct_change(6)
    out["mkt_mom_12m"] = out.groupby("ticker")[price_col_name].pct_change(12)
    roll_max = out.groupby("ticker")[price_col_name].rolling(12, min_periods=4).max().reset_index(level=0, drop=True)
    out["mkt_drawdown_1y"] = out[price_col_name] / roll_max - 1
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(path, index=False)
    return out.reset_index(drop=True)


def build_market_panel(banks_universe: pd.DataFrame, start: str = "2015-01-01") -> pd.DataFrame:
    listed = banks_universe[banks_universe["listed_flag"] & banks_universe["ticker"].notna()]
    if listed.empty:
        listed = curated_listed_bank_universe()
    if listed.empty:
        return pd.DataFrame()
    panel = build_yfinance_bank_panel(listed["ticker"].dropna().unique().tolist(), start=start)
    lookup = listed[["bank_id", "ticker"]].drop_duplicates()
    return panel.merge(lookup, on="ticker", how="left")


def load_external_fundamentals(filepath: str | Path, banks_universe: pd.DataFrame | None = None) -> pd.DataFrame:
    path = Path(filepath)
    raw = pd.read_excel(path) if path.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(path)
    raw.columns = [slugify(c) for c in raw.columns]
    rename_map = {
        "return_on_equity": "roe",
        "return_on_assets": "roa",
        "net_interest_margin": "nim",
        "cost_income": "ci_ratio",
        "cost_income_ratio": "ci_ratio",
        "npl": "npl_ratio",
        "cet1_ratio": "cet1",
        "tier_1": "tier1",
        "total_capital_ratio": "total_capital",
        "loan_deposit_ratio": "ldr",
        "deposits": "customer_deposits",
        "loans": "loans_to_customers",
        "assets": "total_assets",
    }
    df = raw.rename(columns={k: v for k, v in rename_map.items() if k in raw.columns})
    if "bank_id" not in df.columns and banks_universe is not None:
        name_col = next((c for c in df.columns if c in {"bank", "bank_name", "legal_name", "name"}), None)
        if name_col:
            df["normalized_name"] = df[name_col].map(normalize_bank_name)
            df = df.merge(banks_universe[["bank_id", "normalized_name"]], on="normalized_name", how="left")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    elif "year" in df.columns:
        df["date"] = pd.to_datetime(df["year"].astype(str) + "-12-31", errors="coerce")
    else:
        df["date"] = pd.NaT
    for col in FUNDAMENTALS_SCHEMA:
        if col not in df.columns:
            df[col] = np.nan
    df["source"] = df["source"].fillna(f"external_file:{path.name}")
    for col in [c for c in FUNDAMENTALS_SCHEMA if c not in {"bank_id", "date", "source"}]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[FUNDAMENTALS_SCHEMA].copy()


def calculate_bank_ratios(fundamentals_panel: pd.DataFrame) -> pd.DataFrame:
    df = fundamentals_panel.copy()
    safe_div = lambda a, b: np.where((pd.notna(b)) & (b != 0), a / b, np.nan)
    df["roe"] = df["roe"].fillna(safe_div(df["net_income"], df["equity"]))
    df["roa"] = df["roa"].fillna(safe_div(df["net_income"], df["total_assets"]))
    df["ldr"] = df["ldr"].fillna(safe_div(df["loans_to_customers"], df["customer_deposits"]))
    df["npl_ratio"] = df["npl_ratio"].fillna(safe_div(df["npl_gross"], df["loans_to_customers"]))
    df["coverage_ratio"] = df["coverage_ratio"].fillna(safe_div(df["npl_gross"] - df["npl_net"], df["npl_gross"]))
    return df


def discover_bank_report_links(bank_name: str, ir_url: str, cache_dir: str | Path, max_links: int = 20) -> pd.DataFrame:
    try:
        html = cached_get(ir_url, cache_dir=cache_dir, max_age_days=7).decode("utf-8", errors="ignore")
        soup = BeautifulSoup(html, "html.parser")
        rows: list[dict[str, Any]] = []
        for link in soup.find_all("a", href=True):
            href = link["href"]
            text = link.get_text(" ", strip=True)
            full = urljoin(ir_url, href)
            if not re.search(r"\.(pdf|xlsx|xls)(\?|$)", full, re.I):
                continue
            if not re.search(r"annual|report|financial|bilancio|results|consolidated|relazione", text + " " + full, re.I):
                continue
            year = re.search(r"(20\d{2})", text + " " + full)
            rows.append(
                {
                    "bank_name": bank_name,
                    "url": full,
                    "link_text": text,
                    "year": int(year.group(1)) if year else np.nan,
                    "file_type": Path(full.split("?")[0]).suffix.lower(),
                    "source": "bank_investor_relations",
                }
            )
        return pd.DataFrame(rows).drop_duplicates("url").head(max_links)
    except Exception as exc:
        LOGGER.warning("IR discovery failed bank=%s error=%s", bank_name, exc)
        return pd.DataFrame(columns=["bank_name", "url", "link_text", "year", "file_type", "source"])


def parse_bank_annual_report(filepath: str | Path) -> dict[str, Any]:
    """Placeholder for future audited PDF extraction of bank fundamentals."""
    return {"filepath": str(filepath), "parsed": False, "todo": "Implement audited PDF table extraction for banking fundamentals."}


def save_to_sqlite(tables: dict[str, pd.DataFrame], db_path: str | Path) -> None:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        for name, df in tables.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                df.to_sql(name, conn, if_exists="replace", index=False)


def run_banks_data_pipeline(
    output_dir: str | Path,
    cache_dir: str | Path | None = None,
    bds_exports: dict[str, str] | None = None,
    external_fundamentals_path: str | Path | None = None,
    market_start: str = "2015-01-01",
    include_market: bool = True,
    include_ecb_macro: bool = False,
) -> dict[str, pd.DataFrame]:
    """Run the open-source bank universe/regulatory/market data pipeline."""
    output_dir = Path(output_dir)
    cache_dir = Path(cache_dir or output_dir / "_cache")
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    universe = build_banks_universe(cache_dir=cache_dir, output_dir=output_dir)
    ecb_macro = build_ecb_bsi_macro_panel(cache_dir=cache_dir) if include_ecb_macro else pd.DataFrame()
    bdi_macro = fetch_bancaditalia_macro_regulatory(cache_dir=cache_dir, exports=bds_exports or {})
    macro = pd.concat([ecb_macro, bdi_macro], ignore_index=True, sort=False)
    if external_fundamentals_path:
        fundamentals = calculate_bank_ratios(load_external_fundamentals(external_fundamentals_path, banks_universe=universe))
    else:
        fundamentals = pd.DataFrame(columns=FUNDAMENTALS_SCHEMA)
    market = build_market_panel(universe, start=market_start) if include_market else pd.DataFrame()
    artifacts = {
        "banks_universe": universe,
        "banks_macro_regulatory": macro,
        "banks_fundamentals_panel": fundamentals,
        "banks_market_panel": market,
    }
    for name, df in artifacts.items():
        df.to_csv(output_dir / f"{name}.csv", index=False)
    save_to_sqlite(artifacts, output_dir / "banks_data.sqlite")
    manifest = {
        "created_at": utc_now(),
        "output_dir": str(output_dir),
        "cache_dir": str(cache_dir),
        "artifacts": {name: str(output_dir / f"{name}.csv") for name in artifacts},
    }
    (output_dir / "banks_pipeline_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return artifacts
