"""Fama-French Data Library loader."""

from __future__ import annotations

import json
import re
import zipfile
from dataclasses import dataclass
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any

import pandas as pd

from ..data_platform import resolve_data_platform_roots, utc_now


FAMA_FRENCH_BASE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"

FAMA_FRENCH_DATASETS = {
    "FF3_monthly": {"filename": "F-F_Research_Data_Factors_CSV.zip", "frequency": "monthly", "category": "factors"},
    "FF3_daily": {"filename": "F-F_Research_Data_Factors_daily_CSV.zip", "frequency": "daily", "category": "factors"},
    "FF5_monthly": {"filename": "F-F_Research_Data_5_Factors_2x3_CSV.zip", "frequency": "monthly", "category": "factors"},
    "FF5_daily": {"filename": "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip", "frequency": "daily", "category": "factors"},
    "Momentum_monthly": {"filename": "F-F_Momentum_Factor_CSV.zip", "frequency": "monthly", "category": "momentum"},
    "Momentum_daily": {"filename": "F-F_Momentum_Factor_daily_CSV.zip", "frequency": "daily", "category": "momentum"},
    "Industry_10_monthly": {"filename": "10_Industry_Portfolios_CSV.zip", "frequency": "monthly", "category": "industry"},
    "Industry_10_daily": {"filename": "10_Industry_Portfolios_daily_CSV.zip", "frequency": "daily", "category": "industry"},
    "Industry_49_monthly": {"filename": "49_Industry_Portfolios_CSV.zip", "frequency": "monthly", "category": "industry"},
    "Global_3_monthly": {"filename": "Global_3_Factors_CSV.zip", "frequency": "monthly", "category": "international"},
    "Global_3_daily": {"filename": "Global_3_Factors_Daily_CSV.zip", "frequency": "daily", "category": "international"},
    "Developed_3_monthly": {"filename": "Developed_3_Factors_CSV.zip", "frequency": "monthly", "category": "international"},
    "Emerging_5_monthly": {"filename": "Emerging_5_Factors_CSV.zip", "frequency": "monthly", "category": "international"},
}


@dataclass(frozen=True)
class FamaFrenchDatasetSpec:
    name: str
    filename: str
    frequency: str
    category: str

    @property
    def url(self) -> str:
        return FAMA_FRENCH_BASE_URL + self.filename


def _parse_ff_date(value: object, frequency: str) -> pd.Timestamp:
    token = re.sub(r"\.0$", "", str(value).strip())
    if frequency == "daily" and re.fullmatch(r"\d{8}", token):
        return pd.to_datetime(token, format="%Y%m%d", errors="coerce")
    if re.fullmatch(r"\d{6}", token):
        return pd.to_datetime(token + "01", format="%Y%m%d", errors="coerce") + pd.offsets.MonthEnd(0)
    if re.fullmatch(r"\d{4}", token):
        return pd.to_datetime(token + "1231", format="%Y%m%d", errors="coerce")
    return pd.to_datetime(token, errors="coerce")


def parse_fama_french_csv(text: str, frequency: str) -> pd.DataFrame:
    """Parse the first numeric table in a Ken French CSV file."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    header_idx = None
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or "," not in stripped:
            continue
        first = stripped.split(",", 1)[0].strip()
        if first == "" or first.lower() in {"date", "year"}:
            header_idx = idx
            break
    if header_idx is None:
        raise ValueError("Fama-French table header not found")
    table_lines = [lines[header_idx]]
    for line in lines[header_idx + 1 :]:
        stripped = line.strip()
        if not stripped:
            break
        first = stripped.split(",", 1)[0].strip()
        if not re.fullmatch(r"\d{4}|\d{6}|\d{8}", first):
            break
        table_lines.append(line)
    df = pd.read_csv(StringIO("\n".join(table_lines)))
    first_col = df.columns[0]
    df = df.rename(columns={first_col: "date"})
    df["date"] = df["date"].map(lambda value: _parse_ff_date(value, frequency))
    df = df.dropna(subset=["date"])
    for col in df.columns:
        if col == "date":
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce") / 100.0
    return df.sort_values("date").reset_index(drop=True)


class FamaFrenchLoader:
    """Download and normalize Fama-French factor files into Drive."""

    def __init__(self, financial_db_root: Path | str | None = None, output_root: Path | str | None = None):
        roots = resolve_data_platform_roots(financial_db_root=financial_db_root, repo_output_root=output_root)
        self.financial_db_root = roots.financial_db
        self.output_root = roots.repo_output
        self.base_dir = self.financial_db_root / "Factors" / "FamaFrench"
        self.raw_dir = self.base_dir / "raw"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def specs(self) -> list[FamaFrenchDatasetSpec]:
        return [FamaFrenchDatasetSpec(name, **meta) for name, meta in FAMA_FRENCH_DATASETS.items()]

    def catalog(self) -> pd.DataFrame:
        rows = []
        for spec in self.specs():
            target = self.base_dir / f"{spec.name}.csv"
            rows.append(
                {
                    "dataset": spec.name,
                    "provider": "fama_french",
                    "category": spec.category,
                    "frequency": spec.frequency,
                    "url": spec.url,
                    "target_path": str(target),
                    "exists": target.exists(),
                    "rows": len(pd.read_csv(target)) if target.exists() else 0,
                    "updated_at": pd.Timestamp(target.stat().st_mtime, unit="s").isoformat() if target.exists() else "",
                }
            )
        return pd.DataFrame(rows)

    def download_dataset(self, name: str, refresh: bool = False) -> dict[str, Any]:
        spec_map = {spec.name: spec for spec in self.specs()}
        if name not in spec_map:
            raise KeyError(f"Unknown Fama-French dataset: {name}")
        spec = spec_map[name]
        target = self.base_dir / f"{name}.csv"
        raw_path = self.raw_dir / spec.filename
        if target.exists() and target.stat().st_size > 100 and not refresh:
            return {"dataset": name, "status": "cache_hit", "target_path": str(target), "rows": len(pd.read_csv(target)), "updated_at": utc_now()}
        import requests

        response = requests.get(spec.url, timeout=60, headers={"User-Agent": "ResearchPlatform/1.0 FamaFrenchLoader"})
        response.raise_for_status()
        raw_path.write_bytes(response.content)
        with zipfile.ZipFile(BytesIO(response.content)) as archive:
            csv_names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
            if not csv_names:
                raise ValueError(f"No CSV found in {spec.filename}")
            text = archive.read(csv_names[0]).decode("utf-8", errors="ignore")
        df = parse_fama_french_csv(text, spec.frequency)
        df.to_csv(target, index=False)
        metadata = {
            "dataset": name,
            "provider": "fama_french",
            "source_url": spec.url,
            "target_path": str(target),
            "raw_path": str(raw_path),
            "rows": len(df),
            "columns": list(df.columns),
            "frequency": spec.frequency,
            "updated_at": utc_now(),
            "note": "Ken French library uses CRSP CIZ format for US research returns from the January 2025 data release onward.",
        }
        target.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
        return {"status": "downloaded", **metadata}

    def download_all(self, refresh: bool = False, max_datasets: int | None = None) -> pd.DataFrame:
        rows = []
        specs = self.specs()
        if max_datasets:
            specs = specs[:max_datasets]
        for spec in specs:
            try:
                rows.append(self.download_dataset(spec.name, refresh=refresh))
            except Exception as exc:
                rows.append({"dataset": spec.name, "status": "failed", "error": str(exc), "updated_at": utc_now()})
        manifest = pd.DataFrame(rows)
        manifest_path = self.base_dir / "FamaFrench_manifest.csv"
        manifest.to_csv(manifest_path, index=False)
        return manifest
