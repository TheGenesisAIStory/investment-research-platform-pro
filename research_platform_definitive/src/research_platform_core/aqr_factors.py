"""AQR Data Library factor provider.

The provider is intentionally dependency-light and notebook-friendly:
official AQR dataset pages are discovered live, Excel files are cached, and
parsed factor panels are exported as ordinary CSV artifacts.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
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
    "discover_aqr_dataset_pages",
    "discover_aqr_datasets",
    "get_all_factors_panel",
    "get_aqr_factor_panel",
    "parse_aqr_excel",
    "refresh_aqr_factor_library",
]
