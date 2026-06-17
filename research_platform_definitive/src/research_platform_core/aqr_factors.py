"""AQR Data Library factor provider.

The provider is intentionally dependency-light and notebook-friendly:
official AQR dataset pages are discovered live, Excel files are cached, and
parsed factor panels are exported as ordinary CSV artifacts.
"""

from __future__ import annotations

import json
import re
import time
import zipfile
from dataclasses import asdict, dataclass
from io import BytesIO
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urljoin, urlparse

import numpy as np
import pandas as pd

from .data_platform import resolve_data_platform_roots, utc_now


AQR_DATASET_PAGES = (
    "https://www.aqr.com/Insights/Datasets",
    "https://www.aqr.com/Insights/Datasets?page=2",
)
AQR_BASE_URL = "https://www.aqr.com"
FRENCH_FTP_BASE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"

FF3_REGIONS = {
    "US": "F-F_Research_Data_Factors",
    "EU": "Europe_3_Factors",
    "JP": "Japan_3_Factors",
    "APAC": "Asia_Pacific_ex_Japan_3_Factors",
    # Ken French publishes emerging-market factors as the 5-factor bundle.
    # The downloader can still return an FF3-compatible subset from it.
    "EM": "Emerging_5_Factors",
    "WORLD": "Developed_ex_US_3_Factors",
}

FF5_REGIONS = {
    "US": "F-F_Research_Data_5_Factors_2x3",
    "EU": "Europe_5_Factors",
    "JP": "Japan_5_Factors",
    "APAC": "Asia_Pacific_ex_Japan_5_Factors",
    "EM": "Emerging_5_Factors",
}

FF_MOM_REGIONS = {
    "US": "F-F_Momentum_Factor",
    "EU": "Europe_Mom_Factor",
    "JP": "Japan_Mom_Factor",
    "APAC": "Asia_Pacific_ex_Japan_Mom_Factor",
}


@dataclass(frozen=True)
class AqrDataset:
    """Discovered AQR Excel dataset."""

    slug: str
    title: str
    page_url: str
    xlsx_url: str
    filename: str
    provider: str = "aqr"


class _AqrLinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[str, str]] = []
        self._href: str | None = None
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "a":
            attrs_dict = {k: v for k, v in attrs}
            self._href = attrs_dict.get("href")
            self._text = []

    def handle_data(self, data: str) -> None:
        if self._href is not None:
            text = data.strip()
            if text:
                self._text.append(text)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a" and self._href is not None:
            self.links.append((self._href, " ".join(self._text).strip()))
            self._href = None
            self._text = []


def slugify(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\.(xlsx|xls)$", "", text)
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def _normalize_region(region: str) -> str:
    text = str(region or "").strip().upper()
    aliases = {"EUROPE": "EU", "JAPAN": "JP", "GLOBAL": "WORLD", "WORLD": "WORLD", "DEVELOPED": "WORLD"}
    return aliases.get(text, text)


def _normalize_factor_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rename = {
        "Mkt-RF": "MKT_RF",
        "Mkt_RF": "MKT_RF",
        "MKT-RF": "MKT_RF",
        "MKT RF": "MKT_RF",
        "Mom": "MOM",
        "UMD": "MOM",
    }
    out = frame.copy()
    out = out.rename(columns={col: rename.get(str(col).strip(), str(col).strip().replace("-", "_").replace(" ", "_").upper()) for col in out.columns})
    if "DATE" in out.columns and "date" not in out.columns:
        out = out.rename(columns={"DATE": "date"})
    if "date" not in out.columns:
        out = out.reset_index().rename(columns={out.index.name or "index": "date"})
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"])
    for col in out.columns:
        if col == "date":
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce")
        max_abs = out[col].abs().max(skipna=True)
        if pd.notna(max_abs) and max_abs > 2:
            out[col] = out[col] / 100.0
    wanted = [col for col in ["MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM", "RF"] if col in out.columns]
    return out[["date", *wanted]].sort_values("date").reset_index(drop=True)


def _ff_cache_path(region: str, factor_set: str, output_root: str | Path | None = None) -> Path:
    roots = resolve_data_platform_roots(repo_output_root=output_root)
    path = roots.repo_output / "ff_factors" / f"{_normalize_region(region)}_{factor_set.upper().replace('+', '_')}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _cache_is_fresh(path: Path, max_age_days: int = 30) -> bool:
    if not path.exists() or path.stat().st_size <= 100:
        return False
    modified = pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC")
    return (pd.Timestamp.now("UTC") - modified).days <= max_age_days


def _read_french_zip(dataset: str) -> pd.DataFrame:
    from .loaders.fama_french import parse_fama_french_csv

    import requests

    url = FRENCH_FTP_BASE_URL + f"{dataset}_CSV.zip"
    response = requests.get(url, timeout=60, headers={"User-Agent": "ResearchPlatform/1.0 FamaFrenchRegional"})
    response.raise_for_status()
    with zipfile.ZipFile(BytesIO(response.content)) as archive:
        csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if not csv_names:
            raise ValueError(f"No CSV found in {dataset}")
        text = archive.read(csv_names[0]).decode("utf-8", errors="ignore")
    return parse_fama_french_csv(text, "monthly")


def _read_french_datareader(dataset: str, start_date: str) -> pd.DataFrame:
    try:
        from pandas_datareader import data as pdr_data
    except Exception as exc:
        raise ImportError("pandas_datareader is unavailable") from exc

    bundle = pdr_data.DataReader(dataset, "famafrench", start=start_date)
    frame = next((item for item in bundle.values() if isinstance(item, pd.DataFrame)), pd.DataFrame())
    if frame.empty:
        return frame
    out = frame.copy()
    if isinstance(out.index, pd.PeriodIndex):
        out.index = out.index.to_timestamp(how="end")
    out.index.name = "date"
    return out.reset_index()


def download_ff_factors(
    region: str,
    factor_set: str = "FF5+MOM",
    start_date: str = "2000-01-01",
    *,
    output_root: str | Path | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Download regional Fama-French factors with local parquet caching.

    Returns columns ``date, MKT_RF, SMB, HML, [RMW, CMA], [MOM], RF`` in
    decimal return units.  ``FF6`` and ``FF5+MOM`` are synonyms.
    """
    region_key = _normalize_region(region)
    factor_key = str(factor_set or "FF5+MOM").strip().upper().replace(" ", "")
    if factor_key == "FF6":
        factor_key = "FF5+MOM"
    cache_path = _ff_cache_path(region_key, factor_key, output_root)
    if not refresh and _cache_is_fresh(cache_path):
        cached = pd.read_parquet(cache_path)
        return cached[cached["date"].ge(pd.to_datetime(start_date))].reset_index(drop=True)

    if region_key == "IT":
        raise ValueError("Italy FF factors are constructed locally with construct_it_local_factors().")
    if region_key not in FF3_REGIONS:
        raise KeyError(f"Unsupported Fama-French region: {region}")

    frames: list[pd.DataFrame] = []
    if factor_key in {"FF3", "FF5+MOM"}:
        dataset = FF3_REGIONS[region_key]
        try:
            frames.append(_read_french_datareader(dataset, start_date))
        except Exception:
            frames.append(_read_french_zip(dataset))
    if factor_key in {"FF5", "FF5+MOM"} and region_key in FF5_REGIONS:
        dataset = FF5_REGIONS[region_key]
        try:
            frames = [_read_french_datareader(dataset, start_date)]
        except Exception:
            frames = [_read_french_zip(dataset)]
    if factor_key == "FF5+MOM" and region_key in FF_MOM_REGIONS:
        dataset = FF_MOM_REGIONS[region_key]
        try:
            frames.append(_read_french_datareader(dataset, start_date))
        except Exception:
            frames.append(_read_french_zip(dataset))

    normalized = [_normalize_factor_columns(frame) for frame in frames if isinstance(frame, pd.DataFrame) and not frame.empty]
    if not normalized:
        return pd.DataFrame(columns=["date", "MKT_RF", "SMB", "HML", "RF"])
    out = normalized[0]
    for frame in normalized[1:]:
        out = out.merge(frame, on="date", how="outer", suffixes=("", "_dup"))
        for col in [c for c in out.columns if c.endswith("_dup")]:
            base = col[:-4]
            if base in out.columns:
                out[base] = out[base].combine_first(out[col])
            out = out.drop(columns=[col])
    out = out[out["date"].ge(pd.to_datetime(start_date))].sort_values("date").reset_index(drop=True)
    out.to_parquet(cache_path, index=False)
    cache_path.with_suffix(".metadata.json").write_text(
        json.dumps({"region": region_key, "factor_set": factor_key, "rows": len(out), "updated_at": utc_now()}, indent=2),
        encoding="utf-8",
    )
    return out


def construct_it_local_factors(
    panel_df: pd.DataFrame,
    start_date: str = "2000-01-01",
    *,
    output_root: str | Path | None = None,
    write: bool = True,
) -> pd.DataFrame:
    """Construct lightweight Italy SMB/HML/MOM factors from the equity panel."""
    if panel_df is None or panel_df.empty or "date" not in panel_df.columns:
        return pd.DataFrame(columns=["date", "MKT_RF", "SMB", "HML", "MOM", "RF"])
    frame = panel_df.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    country = frame.get("country", pd.Series("", index=frame.index)).astype(str).str.upper()
    exchange = frame.get("exchange", pd.Series("", index=frame.index)).astype(str).str.upper()
    symbol = frame.get("ticker", frame.get("symbol", pd.Series("", index=frame.index))).astype(str).str.upper()
    it_mask = country.isin({"IT", "ITALY"}) | exchange.isin({"MIL", "BIT", "MILAN"}) | symbol.str.endswith((".MI", "-MI"))
    frame = frame[it_mask & frame["date"].ge(pd.to_datetime(start_date))].copy()
    if frame.empty:
        return pd.DataFrame(columns=["date", "MKT_RF", "SMB", "HML", "MOM", "RF"])
    ret_col = next((col for col in ["ret_21d", "return_21d", "forward_return_21d", "realized_return"] if col in frame.columns), None)
    size_col = next((col for col in ["market_cap", "marketvalue", "market_value", "log_market_cap"] if col in frame.columns), None)
    bm_col = next((col for col in ["book_to_market", "bm", "value_score"] if col in frame.columns), None)
    mom_col = next((col for col in ["momentum_12m_1m", "momentum_12_1", "momentum_score"] if col in frame.columns), None)
    if ret_col is None:
        return pd.DataFrame(columns=["date", "MKT_RF", "SMB", "HML", "MOM", "RF"])
    frame["month"] = frame["date"].dt.to_period("M").dt.to_timestamp("M")
    for col in [ret_col, size_col, bm_col, mom_col]:
        if col:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    rows: list[dict[str, Any]] = []
    for date, group in frame.groupby("month"):
        group = group.dropna(subset=[ret_col])
        if len(group) < 4:
            continue
        market = float(group[ret_col].mean())

        def spread(sort_col: str | None, low_long: bool = False) -> float | None:
            if sort_col is None or group[sort_col].notna().sum() < 4:
                return None
            low = group[sort_col].quantile(0.3)
            high = group[sort_col].quantile(0.7)
            low_ret = group.loc[group[sort_col].le(low), ret_col].mean()
            high_ret = group.loc[group[sort_col].ge(high), ret_col].mean()
            return float(low_ret - high_ret) if low_long else float(high_ret - low_ret)

        rows.append({
            "date": pd.Timestamp(date),
            "MKT_RF": market,
            "SMB": spread(size_col, low_long=True),
            "HML": spread(bm_col),
            "MOM": spread(mom_col),
            "RF": 0.0,
        })
    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    if write:
        cache_path = _ff_cache_path("IT", "LOCAL_FACTORS", output_root)
        out.to_parquet(cache_path, index=False)
    return out


def _request_text(url: str, timeout: int = 30) -> str:
    import requests

    response = requests.get(url, timeout=timeout, headers={"User-Agent": "ResearchPlatform/1.0 (+AQR factor cache)"})
    response.raise_for_status()
    return response.text


def _extract_links(html: str, base_url: str) -> list[tuple[str, str]]:
    parser = _AqrLinkParser()
    parser.feed(html)
    return [(urljoin(base_url, href), text) for href, text in parser.links if href]


def discover_aqr_dataset_pages(pages: Iterable[str] = AQR_DATASET_PAGES) -> pd.DataFrame:
    """Discover AQR dataset detail pages from the public index pages."""
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for page in pages:
        html = _request_text(page)
        for href, text in _extract_links(html, page):
            parsed = urlparse(href)
            if "/Insights/Datasets/" not in parsed.path:
                continue
            if href in seen:
                continue
            seen.add(href)
            rows.append({
                "page_url": href,
                "title": text or Path(parsed.path).name.replace("-", " "),
                "page_slug": slugify(Path(parsed.path).name),
                "source_index_page": page,
                "discovered_at": utc_now(),
            })
    return pd.DataFrame(rows)


def discover_aqr_datasets(pages: Iterable[str] = AQR_DATASET_PAGES) -> pd.DataFrame:
    """Discover all unique AQR Excel factor files from dataset pages."""
    page_df = discover_aqr_dataset_pages(pages)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _, page in page_df.iterrows():
        page_url = str(page["page_url"])
        try:
            html = _request_text(page_url)
        except Exception as exc:
            rows.append({
                "slug": str(page.get("page_slug", "")),
                "title": str(page.get("title", "")),
                "page_url": page_url,
                "xlsx_url": "",
                "filename": "",
                "status": "page_failed",
                "error": str(exc),
                "discovered_at": utc_now(),
            })
            continue
        for href, text in _extract_links(html, page_url):
            if not href.lower().split("?", 1)[0].endswith((".xlsx", ".xls")):
                continue
            filename = Path(urlparse(href).path).name
            slug = slugify(filename)
            if href in seen:
                continue
            seen.add(href)
            rows.append({
                "slug": slug,
                "title": text or filename.replace("-", " ").rsplit(".", 1)[0],
                "page_url": page_url,
                "page_title": page.get("title", ""),
                "xlsx_url": href,
                "filename": filename,
                "status": "discovered",
                "error": "",
                "discovered_at": utc_now(),
            })
    out = pd.DataFrame(rows)
    if not out.empty and "slug" in out.columns:
        out = out.drop_duplicates("slug", keep="first").sort_values("slug").reset_index(drop=True)
    return out


def _default_aqr_dirs(
    financial_db_root: str | Path | None = None,
    output_root: str | Path | None = None,
    cache_root: str | Path | None = None,
) -> dict[str, Path]:
    roots = resolve_data_platform_roots(financial_db_root=financial_db_root, repo_output_root=output_root)
    if roots.available:
        base = roots.financial_db / "alpha_factor_library" / "aqr"
    else:
        base = roots.local_cache / "aqr_factors"
    dirs = {
        "base": base,
        "raw": Path(cache_root).expanduser() if cache_root else base / "raw",
        "processed": base / "processed",
        "catalog": base / "catalog",
        "output_tables": roots.repo_output / "aqr_factors" / "tables",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _download_file(url: str, path: Path, force: bool = False, timeout: int = 60) -> dict[str, Any]:
    if path.exists() and path.stat().st_size > 1024 and not force:
        return {"status": "cache_hit", "path": str(path), "bytes": path.stat().st_size, "updated_at": utc_now()}
    import requests

    response = requests.get(url, timeout=timeout, headers={"User-Agent": "ResearchPlatform/1.0 (+AQR factor cache)"})
    response.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)
    return {"status": "downloaded", "path": str(path), "bytes": len(response.content), "updated_at": utc_now()}


def _is_date_token(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, (pd.Timestamp,)):
        return True
    text = str(value).strip()
    if not text:
        return False
    if text.upper() in {"DATE", "DATES", "MONTH", "TIME"}:
        return True
    if re.fullmatch(r"\d{6}(\.0)?", text) or re.fullmatch(r"\d{8}(\.0)?", text):
        return True
    return pd.to_datetime(pd.Series([text]), errors="coerce").notna().iloc[0]


def _coerce_aqr_dates(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    out = pd.to_datetime(text, errors="coerce")
    numeric = text.str.fullmatch(r"\d{6}|\d{8}", na=False)
    if numeric.any():
        yyyymm = text[numeric & text.str.len().eq(6)]
        yyyymmdd = text[numeric & text.str.len().eq(8)]
        parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
        if not yyyymm.empty:
            parsed.loc[yyyymm.index] = pd.to_datetime(yyyymm + "01", format="%Y%m%d", errors="coerce") + pd.offsets.MonthEnd(0)
        if not yyyymmdd.empty:
            parsed.loc[yyyymmdd.index] = pd.to_datetime(yyyymmdd, format="%Y%m%d", errors="coerce")
        out.loc[parsed.notna()] = parsed.loc[parsed.notna()]
    return out


def _dedupe_columns(columns: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out: list[str] = []
    for col in columns:
        base = slugify(col) or "value"
        seen[base] = seen.get(base, 0) + 1
        out.append(base if seen[base] == 1 else f"{base}_{seen[base]}")
    return out


def _parse_aqr_sheet(path: Path, sheet_name: str) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=sheet_name, header=None)
    if raw.empty:
        return pd.DataFrame()
    header_row = None
    for idx in range(min(len(raw), 80)):
        row = raw.iloc[idx]
        first = row.iloc[0] if len(row) else np.nan
        if _is_date_token(first) and row.notna().sum() > 1:
            header_row = idx
            break
        if row.astype(str).str.upper().eq("DATE").any() and row.notna().sum() > 1:
            header_row = idx
            break
    if header_row is None:
        return pd.DataFrame()
    header = raw.iloc[header_row].ffill().astype(str).tolist()
    data = raw.iloc[header_row + 1 :].copy()
    data.columns = _dedupe_columns(header)
    date_col = next((c for c in data.columns if c in {"date", "dates", "month", "time"}), data.columns[0])
    data["date"] = _coerce_aqr_dates(data[date_col])
    data = data.dropna(subset=["date"]).drop(columns=[c for c in [date_col] if c != "date"], errors="ignore")
    value_cols = [c for c in data.columns if c != "date"]
    if not value_cols:
        return pd.DataFrame()
    for col in value_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(how="all", subset=value_cols)
    if data.empty:
        return pd.DataFrame()
    data = data.set_index("date").sort_index()
    return data


def parse_aqr_excel(path: str | Path, dataset_slug: str | None = None) -> pd.DataFrame:
    """Parse AQR Excel workbooks into a single date-indexed wide factor panel."""
    path = Path(path)
    dataset_slug = slugify(dataset_slug or path.stem)
    sheets = pd.ExcelFile(path).sheet_names
    parsed: list[pd.DataFrame] = []
    for sheet in sheets:
        frame = _parse_aqr_sheet(path, sheet)
        if frame.empty:
            continue
        prefix = slugify(sheet)
        frame = frame.rename(columns={col: f"{dataset_slug}__{prefix}__{slugify(col)}" for col in frame.columns})
        parsed.append(frame)
    if not parsed:
        return pd.DataFrame()
    out = pd.concat(parsed, axis=1).sort_index()
    out = out.loc[:, ~out.columns.duplicated()]
    out.index.name = "date"
    return out


def _record_aqr_parse_failure(output_root: str | Path | None, row: dict[str, Any]) -> None:
    roots = resolve_data_platform_roots(repo_output_root=output_root)
    failure_path = roots.repo_output / "aqr_factors" / "parse_failures.csv"
    failure_path.parent.mkdir(parents=True, exist_ok=True)
    existing = pd.read_csv(failure_path) if failure_path.exists() else pd.DataFrame()
    pd.concat([existing, pd.DataFrame([row])], ignore_index=True).to_csv(failure_path, index=False)


def parse_aqr_excel_robust(
    filepath: str | Path,
    sheet_name: str | int = 0,
    parser_config: dict[str, Any] | None = None,
    *,
    output_root: str | Path | None = None,
) -> pd.DataFrame:
    """Parse one AQR Excel sheet defensively across header/date variants."""
    path = Path(filepath)
    config = {
        "expected_columns": (),
        "date_column": "date",
        "header_search_rows": [0, 1, 2, 3, 4, 5],
        "min_valid_columns": 3,
        **(parser_config or {}),
    }
    try:
        best: pd.DataFrame | None = None
        best_score = -1
        xls = pd.ExcelFile(path)
        sheet = sheet_name if sheet_name in xls.sheet_names or isinstance(sheet_name, int) else xls.sheet_names[0]
        for header in config["header_search_rows"]:
            try:
                frame = pd.read_excel(path, sheet_name=sheet, header=header)
            except Exception:
                continue
            frame = frame.dropna(axis=0, how="all").dropna(axis=1, how="all")
            if frame.empty:
                continue
            frame.columns = _dedupe_columns([str(col).strip().lower().replace(" ", "_") for col in frame.columns])
            date_col = next((col for col in frame.columns if col in {"date", "dates", "month", "year"} or "date" in col), frame.columns[0])
            parsed_dates = _coerce_aqr_dates(frame[date_col])
            numeric_cols = 0
            for col in frame.columns:
                if col == date_col:
                    continue
                converted = pd.to_numeric(frame[col], errors="coerce")
                if converted.notna().sum() >= max(2, len(frame) // 10):
                    numeric_cols += 1
            score = int(parsed_dates.notna().sum()) + numeric_cols * 10
            if numeric_cols >= int(config["min_valid_columns"]) - 1 and score > best_score:
                candidate = frame.copy()
                candidate["date"] = parsed_dates
                candidate = candidate.dropna(subset=["date"]).drop(columns=[date_col] if date_col != "date" else [], errors="ignore")
                for col in candidate.columns:
                    if col != "date":
                        candidate[col] = pd.to_numeric(candidate[col], errors="coerce")
                candidate = candidate.dropna(axis=1, how="all").sort_values("date").reset_index(drop=True)
                best = candidate
                best_score = score
        if best is None or best.empty:
            raise ValueError("No parseable date/numeric block found")
        return best
    except Exception as exc:
        _record_aqr_parse_failure(
            output_root,
            {
                "file_path": str(path),
                "sheet_name": str(sheet_name),
                "error": f"{type(exc).__name__}: {exc}",
                "updated_at": utc_now(),
            },
        )
        return pd.DataFrame()


def load_all_regional_factors(
    output_root: str | Path | None = None,
    factor_set: str = "FF5+MOM",
    start_date: str = "2000-01-01",
    refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """Load all supported regional Fama-French factor panels."""
    out: dict[str, pd.DataFrame] = {}
    for region in ["US", "EU", "JP", "APAC", "EM", "World"]:
        try:
            out[region] = download_ff_factors(region, factor_set=factor_set, start_date=start_date, output_root=output_root, refresh=refresh)
        except Exception:
            out[region] = pd.DataFrame()
    return out


class AqrFactorProvider:
    """Drive-first/cache-first provider for AQR Data Library factors."""

    def __init__(
        self,
        financial_db_root: str | Path | None = None,
        output_root: str | Path | None = None,
        cache_root: str | Path | None = None,
        pages: Iterable[str] = AQR_DATASET_PAGES,
    ) -> None:
        self.pages = tuple(pages)
        self.dirs = _default_aqr_dirs(financial_db_root, output_root, cache_root)

    @property
    def catalog_path(self) -> Path:
        return self.dirs["catalog"] / "AQRFactorDiscovery.csv"

    def discover(self, force: bool = False) -> pd.DataFrame:
        if self.catalog_path.exists() and not force:
            return pd.read_csv(self.catalog_path)
        discovery = discover_aqr_datasets(self.pages)
        discovery.to_csv(self.catalog_path, index=False)
        return discovery

    def get_dataset(self, slug: str, refresh: bool = False) -> pd.DataFrame:
        discovery = self.discover(force=False)
        if discovery.empty or "slug" not in discovery.columns:
            return pd.DataFrame()
        wanted = slugify(slug)
        row = discovery[discovery["slug"].astype(str).eq(wanted)]
        if row.empty:
            row = discovery[discovery["slug"].astype(str).str.contains(wanted, na=False)]
        if row.empty:
            raise KeyError(f"AQR dataset not found: {slug}")
        info = row.iloc[0]
        raw_path = self.dirs["raw"] / str(info["filename"])
        parsed_path = self.dirs["processed"] / f"{info['slug']}.csv"
        if parsed_path.exists() and not refresh:
            df = pd.read_csv(parsed_path, parse_dates=["date"])
            return df.set_index("date").sort_index()
        _download_file(str(info["xlsx_url"]), raw_path, force=refresh)
        parsed = parse_aqr_excel(raw_path, str(info["slug"]))
        parsed.reset_index().to_csv(parsed_path, index=False)
        return parsed

    def get_all_datasets(self, refresh: bool = False, max_datasets: int | None = None, sleep_seconds: float = 0.25) -> dict[str, pd.DataFrame]:
        discovery = self.discover(force=refresh)
        if max_datasets is not None and max_datasets > 0:
            discovery = discovery.head(max_datasets)
        out: dict[str, pd.DataFrame] = {}
        for _, row in discovery.iterrows():
            slug = str(row["slug"])
            try:
                out[slug] = self.get_dataset(slug, refresh=refresh)
            except Exception:
                out[slug] = pd.DataFrame()
            if sleep_seconds:
                time.sleep(float(sleep_seconds))
        return out

    def get_panel(self, slugs: Iterable[str] | None = None, refresh: bool = False, max_datasets: int | None = None) -> pd.DataFrame:
        if slugs:
            datasets = {slugify(s): self.get_dataset(str(s), refresh=refresh) for s in slugs}
        else:
            datasets = self.get_all_datasets(refresh=refresh, max_datasets=max_datasets)
        frames = [df for df in datasets.values() if isinstance(df, pd.DataFrame) and not df.empty]
        if not frames:
            return pd.DataFrame()
        panel = pd.concat(frames, axis=1).sort_index()
        panel = panel.loc[:, ~panel.columns.duplicated()]
        panel.index.name = "date"
        return panel


def get_aqr_factor_panel(
    slugs: Iterable[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    refresh: bool = False,
    max_datasets: int | None = None,
    financial_db_root: str | Path | None = None,
    output_root: str | Path | None = None,
) -> pd.DataFrame:
    """Return an outer-joined AQR factor panel indexed by date."""
    provider = AqrFactorProvider(financial_db_root=financial_db_root, output_root=output_root)
    panel = provider.get_panel(slugs=slugs, refresh=refresh, max_datasets=max_datasets)
    if panel.empty:
        return panel
    if start:
        panel = panel.loc[pd.to_datetime(start) :]
    if end:
        panel = panel.loc[: pd.to_datetime(end)]
    return panel


def get_all_factors_panel(
    include_aqr: bool = True,
    extra_factor_frames: Iterable[pd.DataFrame] | None = None,
    start: str | None = None,
    end: str | None = None,
    refresh: bool = False,
    max_aqr_datasets: int | None = None,
    financial_db_root: str | Path | None = None,
    output_root: str | Path | None = None,
) -> pd.DataFrame:
    """Outer-join AQR with any additional factor frames supplied by callers."""
    frames: list[pd.DataFrame] = []
    if include_aqr:
        frames.append(get_aqr_factor_panel(start=start, end=end, refresh=refresh, max_datasets=max_aqr_datasets, financial_db_root=financial_db_root, output_root=output_root))
    for frame in extra_factor_frames or []:
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            f = frame.copy()
            if "date" in f.columns:
                f["date"] = pd.to_datetime(f["date"], errors="coerce")
                f = f.dropna(subset=["date"]).set_index("date")
            frames.append(f)
    frames = [f for f in frames if isinstance(f, pd.DataFrame) and not f.empty]
    if not frames:
        return pd.DataFrame()
    panel = pd.concat(frames, axis=1).sort_index()
    panel = panel.loc[:, ~panel.columns.duplicated()]
    panel.index.name = "date"
    return panel


def refresh_aqr_factor_library(
    financial_db_root: str | Path | None = None,
    output_root: str | Path | None = None,
    refresh: bool = False,
    max_datasets: int | None = None,
) -> dict[str, Path]:
    """Refresh AQR discovery, parsed datasets, combined panel and manifest."""
    provider = AqrFactorProvider(financial_db_root=financial_db_root, output_root=output_root)
    discovery = provider.discover(force=refresh)
    tables = provider.dirs["output_tables"]
    tables.mkdir(parents=True, exist_ok=True)
    discovery_path = tables / "AQRFactorDiscovery.csv"
    discovery.to_csv(discovery_path, index=False)

    log_rows: list[dict[str, Any]] = []
    datasets = provider.get_all_datasets(refresh=refresh, max_datasets=max_datasets)
    for slug, df in datasets.items():
        if df.empty:
            log_rows.append({"slug": slug, "status": "empty_or_failed", "rows": 0, "columns": 0, "min_date": "", "max_date": "", "updated_at": utc_now()})
            continue
        csv_path = tables / f"AQR_{slug}.csv"
        df.reset_index().to_csv(csv_path, index=False)
        log_rows.append({
            "slug": slug,
            "status": "ok",
            "rows": len(df),
            "columns": len(df.columns),
            "min_date": df.index.min(),
            "max_date": df.index.max(),
            "output_path": str(csv_path),
            "updated_at": utc_now(),
        })
    panel = provider.get_panel(max_datasets=max_datasets)
    panel_path = tables / "AQRFactorPanel.csv"
    if not panel.empty:
        panel.reset_index().to_csv(panel_path, index=False)
    else:
        pd.DataFrame().to_csv(panel_path, index=False)
    log = pd.DataFrame(log_rows)
    log_path = tables / "AQRFactorRefreshLog.csv"
    log.to_csv(log_path, index=False)
    manifest = {
        "provider": "aqr",
        "generated_at": utc_now(),
        "dataset_count": int(len(discovery)),
        "refreshed_count": int((log["status"].eq("ok")).sum()) if not log.empty and "status" in log else 0,
        "panel_rows": int(len(panel)),
        "panel_columns": int(len(panel.columns)) if not panel.empty else 0,
        "paths": {
            "discovery": str(discovery_path),
            "refresh_log": str(log_path),
            "panel": str(panel_path),
            "drive_catalog": str(provider.catalog_path),
            "drive_processed": str(provider.dirs["processed"]),
        },
    }
    manifest_path = tables / "AQRFactorManifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return {
        "discovery": discovery_path,
        "refresh_log": log_path,
        "panel": panel_path,
        "manifest": manifest_path,
        "drive_catalog": provider.catalog_path,
    }


__all__ = [
    "AqrDataset",
    "AqrFactorProvider",
    "FF3_REGIONS",
    "FF5_REGIONS",
    "FF_MOM_REGIONS",
    "construct_it_local_factors",
    "discover_aqr_dataset_pages",
    "discover_aqr_datasets",
    "download_ff_factors",
    "get_all_factors_panel",
    "get_aqr_factor_panel",
    "load_all_regional_factors",
    "parse_aqr_excel",
    "parse_aqr_excel_robust",
    "refresh_aqr_factor_library",
]
