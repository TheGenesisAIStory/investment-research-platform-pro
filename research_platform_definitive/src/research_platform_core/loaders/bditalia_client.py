"""Banca d'Italia BDS/Infostat public export client.

The Banca d'Italia Statistical Database (BDS, Infostat inquiry) exposes
application-to-application REST exports as compressed ZIP files:

https://a2a.bancaditalia.it/infostat/dataservices/export/{lang}/{format}/{content}/{object_type}/BANKITALIA/DIFF/{object_id}

Official references:

* Infostat inquiry: https://infostat.bancaditalia.it/inquiry/
* BDS overview: https://www.bancaditalia.it/statistiche/basi-dati/bds/
* Statistics and release schedule:
  https://www.bancaditalia.it/statistiche/
  https://www.bancaditalia.it/statistiche/calendario-pubblicazioni/calendario-pubblicazioni.html
* Methodological references for big data/AI in central-bank statistics:
  https://www.bancaditalia.it/pubblicazioni/qef/2022-0721/QEF_721_EN.pdf?language_id=1
  https://www.bancaditalia.it/pubblicazioni/qef/2022-0693/QEF_693_22.pdf

The parser keeps publication/cube metadata and source files attached to every
row because official statistical releases can be revised.
"""

from __future__ import annotations

import json
import logging
import re
import zipfile
from dataclasses import dataclass
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any

import pandas as pd

from ..data_platform import resolve_data_platform_roots, utc_now


LOGGER = logging.getLogger(__name__)

BDITALIA_A2A_BASE_URL = "https://a2a.bancaditalia.it/infostat/dataservices/export"
BDITALIA_COMMUNITY = "BANKITALIA"
BDITALIA_CONTEXT = "DIFF"

BDITALIA_SERIES_PRESETS: dict[str, dict[str, Any]] = {
    "credit_growth": {
        "object_id": "AGGM0500",
        "object_type": "CUBE",
        "content": "DATA",
        "publication": "BAM",
        "description": "Principali aggregati monetari e creditizi, variazioni percentuali sui 12 mesi",
        "category": "credit",
        "frequency": "monthly",
    },
    "credit_counterparts": {
        "object_id": "AGGM0300",
        "object_type": "CUBE",
        "content": "DATA",
        "publication": "BAM",
        "description": "Contropartite italiane della moneta dell'area dell'euro, consistenze",
        "category": "credit",
        "frequency": "monthly",
    },
    "deposit_rates": {
        "object_id": "MIR0700",
        "object_type": "CUBE",
        "content": "DATA",
        "publication": "BAM",
        "description": "Tassi sui depositi in euro di famiglie e societa non finanziarie, nuove operazioni",
        "category": "deposits",
        "frequency": "monthly",
    },
    "deposit_volumes": {
        "object_id": "MIR0900",
        "object_type": "CUBE",
        "content": "DATA",
        "publication": "BAM",
        "description": "Volumi sui depositi in euro di famiglie e societa non finanziarie, nuove operazioni",
        "category": "deposits",
        "frequency": "monthly",
    },
    "official_rates_bdi": {
        "object_id": "TUFF0100",
        "object_type": "CUBE",
        "content": "DATA",
        "publication": "BAM",
        "description": "Tassi d'interesse ufficiali dell'Eurosistema",
        "category": "policy_rates",
        "frequency": "event",
    },
    "public_debt": {
        "object_id": "TCCE0100",
        "object_type": "CUBE",
        "content": "DATA",
        "publication": "FPI",
        "description": "Finanza pubblica: fabbisogno e debito, tavola principale",
        "category": "public_debt",
        "frequency": "monthly",
    },
    "banks_money_publication": {
        "object_id": "BAM",
        "object_type": "PUBLICATION",
        "content": "ALL",
        "publication": "BAM",
        "description": "Banche e moneta: serie nazionali",
        "category": "publication",
        "frequency": "mixed",
    },
    "public_finance_publication": {
        "object_id": "FPI",
        "object_type": "PUBLICATION",
        "content": "ALL",
        "publication": "FPI",
        "description": "Finanza pubblica: fabbisogno e debito",
        "category": "publication",
        "frequency": "monthly",
    },
}

BDITALIA_ALIAS = {
    "credit": "credit_growth",
    "loans": "credit_counterparts",
    "deposits": "deposit_volumes",
    "deposit": "deposit_volumes",
    "debt": "public_debt",
    "public_debt": "public_debt",
}


@dataclass(frozen=True)
class BancaDItaliaSeriesSpec:
    """Configured Banca d'Italia BDS publication/cube reference."""

    name: str
    object_id: str
    object_type: str
    content: str
    publication: str
    description: str
    category: str
    frequency: str


def _slugify(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "_", text)
    return text.strip("_")


def _decode_bytes(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
        try:
            return raw.decode(encoding)
        except Exception:
            continue
    return raw.decode("latin1", errors="ignore")


def _read_bds_csv(raw: bytes) -> pd.DataFrame:
    text = _decode_bytes(raw)
    for kwargs in ({"sep": ";", "decimal": ","}, {"sep": ",", "decimal": "."}, {"sep": None, "engine": "python"}):
        try:
            df = pd.read_csv(StringIO(text), **kwargs)
            if len(df.columns) > 1:
                return df
        except Exception:
            continue
    return pd.read_csv(StringIO(text), sep=";", decimal=",")


def _coerce_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    cleaned = series.astype(str).str.replace("\u00a0", "", regex=False).str.strip()
    if cleaned.str.contains(",", regex=False).any():
        cleaned = cleaned.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    else:
        cleaned = cleaned.str.replace(",", "", regex=False)
    return pd.to_numeric(cleaned, errors="coerce")


def _extract_csv_frames_from_zip(raw: bytes, parent_name: str = "") -> list[tuple[str, pd.DataFrame]]:
    frames: list[tuple[str, pd.DataFrame]] = []
    with zipfile.ZipFile(BytesIO(raw)) as archive:
        for member in archive.namelist():
            if member.endswith("/"):
                continue
            data = archive.read(member)
            name = f"{parent_name}/{member}".strip("/")
            if member.lower().endswith(".zip"):
                frames.extend(_extract_csv_frames_from_zip(data, parent_name=name))
            elif member.lower().endswith(".csv"):
                frames.append((name, _read_bds_csv(data)))
    return frames


def normalize_bditalia_frame(
    df: pd.DataFrame,
    dataset: str,
    description: str,
    source_url: str,
    source_file: str = "",
    publication: str = "",
    category: str = "",
) -> pd.DataFrame:
    """Convert a BDS CSV table to a long, ML-friendly observation frame."""
    raw = df.copy()
    raw.columns = [_slugify(c) for c in raw.columns]
    date_candidates = [
        "date",
        "data",
        "periodo",
        "time_period",
        "data_decor",
        "data_prov",
        "mese",
        "trimestre",
        "anno",
    ]
    date_col = next((c for c in date_candidates if c in raw.columns), None)
    if date_col is None:
        date_col = next((c for c in raw.columns if c.startswith("data")), None)
    id_like = {date_col, "num_ord", "codice", "serie", "territorio", "ateco", "settore", "strumento", "valuta", "fonte"}
    value_vars: list[str] = []
    for col in raw.columns:
        if col in id_like or col is None:
            continue
        numeric = _coerce_numeric(raw[col])
        if numeric.notna().sum() > 0:
            raw[col] = numeric
            value_vars.append(col)
    if not value_vars:
        return pd.DataFrame(
            columns=["date", "dataset", "item_code", "value", "description", "source", "source_url", "source_file", "publication", "category", "retrieved_at"]
        )
    id_vars = [c for c in raw.columns if c not in value_vars]
    long = raw.melt(id_vars=id_vars, value_vars=value_vars, var_name="item_code", value_name="value")
    if date_col:
        long["date"] = pd.to_datetime(long[date_col], errors="coerce")
    else:
        long["date"] = pd.NaT
    long["dataset"] = dataset
    long["series_code"] = long["item_code"].map(lambda item: f"{dataset}.{item}")
    long["description"] = description
    long["source"] = "bancaditalia_bds_a2a"
    long["source_url"] = source_url
    long["source_file"] = source_file
    long["publication"] = publication
    long["category"] = category
    long["retrieved_at"] = utc_now()
    cols = [
        "date",
        "dataset",
        "series_code",
        "item_code",
        "value",
        "description",
        "source",
        "source_url",
        "source_file",
        "publication",
        "category",
        "retrieved_at",
    ]
    extras = [c for c in long.columns if c not in cols and c != date_col]
    return long[cols + extras].dropna(subset=["value"], how="all").reset_index(drop=True)


class BancaDItaliaClient:
    """Client for public Banca d'Italia BDS/Infostat A2A exports."""

    def __init__(
        self,
        financial_db_root: Path | str | None = None,
        output_root: Path | str | None = None,
        base_url: str = BDITALIA_A2A_BASE_URL,
        timeout: int = 60,
        session: Any | None = None,
    ):
        roots = resolve_data_platform_roots(financial_db_root=financial_db_root, repo_output_root=output_root)
        self.financial_db_root = roots.financial_db
        self.output_root = roots.repo_output
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = session
        self.base_dir = self.financial_db_root / "OfficialMacro" / "BancaItalia"
        self.raw_dir = self.base_dir / "raw"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def specs(self) -> list[BancaDItaliaSeriesSpec]:
        return [BancaDItaliaSeriesSpec(name, **meta) for name, meta in BDITALIA_SERIES_PRESETS.items()]

    def build_url(
        self,
        object_id: str,
        object_type: str = "CUBE",
        content: str = "DATA",
        lang: str = "IT",
        data_format: str = "CSV",
        community: str = BDITALIA_COMMUNITY,
        context: str = BDITALIA_CONTEXT,
    ) -> str:
        return "/".join(
            [
                self.base_url,
                lang.upper(),
                data_format.upper(),
                content.upper(),
                object_type.upper(),
                community.upper(),
                context.upper(),
                object_id.upper(),
            ]
        )

    def download_export(
        self,
        object_id: str,
        object_type: str = "CUBE",
        content: str = "DATA",
        lang: str = "IT",
        data_format: str = "CSV",
    ) -> tuple[str, bytes]:
        """Download a BDS export ZIP for a publication or cube."""
        import requests

        url = self.build_url(object_id, object_type=object_type, content=content, lang=lang, data_format=data_format)
        headers = {"User-Agent": "ResearchPlatform/1.0 BancaDItaliaClient", "Accept": "application/zip,text/csv,*/*"}
        session = self.session or requests
        response = session.get(url, headers=headers, timeout=self.timeout)
        response.raise_for_status()
        return getattr(response, "url", url), response.content

    def get_dataset(
        self,
        object_id: str,
        object_type: str = "CUBE",
        content: str = "DATA",
        lang: str = "IT",
        data_format: str = "CSV",
        description: str = "",
        publication: str = "",
        category: str = "",
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """Download and normalize a BDS cube/publication into long observations."""
        source_url, raw = self.download_export(object_id, object_type=object_type, content=content, lang=lang, data_format=data_format)
        raw_path = self.raw_dir / f"{object_id.upper()}_{content.upper()}.zip"
        raw_path.write_bytes(raw)
        frames = []
        for source_file, frame in _extract_csv_frames_from_zip(raw):
            normalized = normalize_bditalia_frame(
                frame,
                dataset=object_id.upper(),
                description=description or object_id.upper(),
                source_url=source_url,
                source_file=source_file,
                publication=publication,
                category=category,
            )
            frames.append(normalized)
        out = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
        if start and not out.empty and "date" in out:
            out = out[out["date"].ge(pd.to_datetime(start, errors="coerce"))]
        if end and not out.empty and "date" in out:
            out = out[out["date"].le(pd.to_datetime(end, errors="coerce"))]
        out["raw_path"] = str(raw_path)
        return out.reset_index(drop=True)

    def get_preset(self, name: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        preset_name = BDITALIA_ALIAS.get(str(name).lower(), name)
        spec_map = {spec.name: spec for spec in self.specs()}
        if preset_name not in spec_map:
            raise KeyError(f"Unknown Banca d'Italia preset: {name}")
        spec = spec_map[preset_name]
        df = self.get_dataset(
            spec.object_id,
            object_type=spec.object_type,
            content=spec.content,
            description=spec.description,
            publication=spec.publication,
            category=spec.category,
            start=start,
            end=end,
        )
        df["preset"] = preset_name
        df["frequency"] = spec.frequency
        return df

    def get_credit_series(self, code: str = "credit_growth", start: str | None = None, end: str | None = None) -> pd.DataFrame:
        """Return configured credit/loan series from the BDS BAM publication."""
        return self.get_preset(code, start=start, end=end)

    def get_deposits_series(self, code: str = "deposit_volumes", start: str | None = None, end: str | None = None) -> pd.DataFrame:
        """Return configured deposit series from Banca d'Italia BDS."""
        return self.get_preset(code, start=start, end=end)

    def get_public_debt_series(self, code: str = "public_debt", start: str | None = None, end: str | None = None) -> pd.DataFrame:
        """Return configured public finance/debt series from the BDS FPI publication."""
        return self.get_preset(code, start=start, end=end)

    def release_frequency(self, code: str) -> str:
        """Return expected release cadence from the local preset registry."""
        preset_name = BDITALIA_ALIAS.get(str(code).lower(), code)
        meta = BDITALIA_SERIES_PRESETS.get(preset_name, {})
        return str(meta.get("frequency", "unknown"))

    def save_dataset(self, df: pd.DataFrame, name: str, fmt: str = "csv", metadata: dict[str, Any] | None = None) -> Path:
        """Persist a normalized BDS dataset and sidecar metadata."""
        target = self.base_dir / f"{name}.{fmt.lower()}"
        target.parent.mkdir(parents=True, exist_ok=True)
        if fmt.lower() == "parquet":
            df.to_parquet(target, index=False)
        else:
            target = target.with_suffix(".csv")
            df.to_csv(target, index=False)
        meta = {
            "dataset": name,
            "provider": "bancaditalia",
            "source": "Banca d'Italia BDS/Infostat A2A",
            "rows": int(len(df)),
            "columns": list(df.columns),
            "release_calendar": "https://www.bancaditalia.it/statistiche/calendario-pubblicazioni/calendario-pubblicazioni.html",
            "written_at": utc_now(),
            **(metadata or {}),
        }
        target.with_suffix(".metadata.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        return target

    def sync_presets(
        self,
        presets: list[str] | None = None,
        start: str | None = None,
        end: str | None = None,
        refresh: bool = False,
        fmt: str = "csv",
    ) -> pd.DataFrame:
        """Idempotently download configured Banca d'Italia presets."""
        rows: list[dict[str, Any]] = []
        names = presets or ["credit_growth", "deposit_volumes", "public_debt", "official_rates_bdi"]
        for name in names:
            target = self.base_dir / f"{name}.{fmt.lower()}"
            if target.exists() and not refresh:
                rows.append({"dataset": name, "status": "cache_hit", "target_path": str(target), "rows": len(pd.read_csv(target)) if target.suffix == ".csv" else ""})
                continue
            try:
                df = self.get_preset(name, start=start, end=end)
                path = self.save_dataset(df, name, fmt=fmt, metadata={"preset": BDITALIA_SERIES_PRESETS.get(name, {})})
                rows.append({"dataset": name, "status": "downloaded", "target_path": str(path), "rows": len(df), "updated_at": utc_now()})
            except Exception as exc:
                LOGGER.warning("Banca d'Italia preset failed name=%s error=%s", name, exc)
                rows.append({"dataset": name, "status": "failed", "error": str(exc), "updated_at": utc_now()})
        manifest = pd.DataFrame(rows)
        manifest.to_csv(self.base_dir / "bditalia_official_macro_manifest.csv", index=False)
        return manifest
