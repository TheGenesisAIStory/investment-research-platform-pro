"""I/O helpers for Smart Money Government Data Engine."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def discover_financial_db() -> Path:
    """Find the canonical Database Finanziario root without hard-failing."""
    candidates = [
        Path(os.environ["FINANCIAL_DB_ROOT"]) if os.environ.get("FINANCIAL_DB_ROOT") else None,
        Path("/content/drive/MyDrive/Database Finanziario"),
        Path("/content/drive/MyDrive/Il mio Drive/Database Finanziario"),
        Path.home() / "Library/CloudStorage/GoogleDrive-sfn.gns@gmail.com/Il mio Drive/Database Finanziario",
        project_root() / "data",
    ]
    for candidate in [c for c in candidates if c is not None]:
        if candidate.exists():
            return candidate
    return project_root() / "data"


def default_output_root() -> Path:
    return Path(os.environ.get("SMART_MONEY_OUTPUT_ROOT", project_root() / "output" / "smart_money"))


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_write_csv(df: pd.DataFrame, path: Path) -> Path:
    ensure_dir(path.parent)
    df.to_csv(path, index=False)
    return path


def safe_write_json(payload: dict[str, Any], path: Path) -> Path:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def read_table(path: Path) -> pd.DataFrame:
    """Read CSV/TSV/parquet/jsonl files with conservative fallbacks."""
    if not path.exists() or not path.is_file():
        return pd.DataFrame()
    suffix = path.suffix.lower()
    try:
        if suffix == ".parquet":
            return pd.read_parquet(path)
        if suffix in {".tsv", ".txt"}:
            return pd.read_csv(path, sep="\t", low_memory=False)
        if suffix == ".jsonl":
            return pd.read_json(path, lines=True)
        if suffix == ".json":
            obj = json.loads(path.read_text(encoding="utf-8"))
            return pd.DataFrame(obj if isinstance(obj, list) else [obj])
        return pd.read_csv(path, low_memory=False)
    except Exception:
        try:
            return pd.read_csv(path, sep=None, engine="python", low_memory=False)
        except Exception:
            return pd.DataFrame()


def find_candidate_files(root: Path, tokens: list[str], suffixes: tuple[str, ...] = (".csv", ".tsv", ".txt", ".parquet", ".json", ".jsonl")) -> list[Path]:
    """Find local official-source files by token, keeping search lightweight."""
    if not root.exists():
        return []
    lowered = [t.lower() for t in tokens]
    matches: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        rel = str(path.relative_to(root)).lower()
        if any(token in rel for token in lowered):
            matches.append(path)
    return sorted(matches)


def latest_file(root: Path, tokens: list[str]) -> Path | None:
    files = find_candidate_files(root, tokens)
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)
