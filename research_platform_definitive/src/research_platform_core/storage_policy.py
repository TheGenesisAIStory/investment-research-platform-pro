"""Project storage policy shared by notebooks, Streamlit and scripts.

The platform is designed to run Drive-first:

- Google Drive hosts the canonical project bundle for Colab and local runs.
- Database Finanziario remains the canonical data lake.
- Local folders are allowed as a development mirror, but should not be required.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DRIVE_PROJECT_CANDIDATES = [
    Path("/content/drive/MyDrive/GitHub/machine-learning-for-trading/research_platform_definitive"),
    Path("/content/drive/MyDrive/machine-learning-for-trading/research_platform_definitive"),
    Path.home() / "Library/CloudStorage/GoogleDrive-sfn.gns@gmail.com/Il mio Drive/GitHub/machine-learning-for-trading/research_platform_definitive",
    Path.home() / "Library/CloudStorage/GoogleDrive-sfn.gns@gmail.com/Il mio Drive/machine-learning-for-trading/research_platform_definitive",
]

LOCAL_PROJECT_CANDIDATES = [
    Path.home() / "GitHub/machine-learning-for-trading/research_platform_definitive",
    Path.cwd(),
    Path.cwd().parent,
]

FINANCIAL_DB_CANDIDATES = [
    Path("/content/drive/MyDrive/Database Finanziario"),
    Path("/content/drive/MyDrive/GitHub/Database Finanziario"),
    Path.home() / "Library/CloudStorage/GoogleDrive-sfn.gns@gmail.com/Il mio Drive/Database Finanziario",
]


@dataclass(frozen=True)
class StorageRoots:
    project_root: Path
    financial_db_root: Path
    output_root: Path
    cache_root: Path
    source: str
    storage_mode: str


def has_project_sentinel(path: Path) -> bool:
    path = Path(path).expanduser()
    return (path / "src" / "research_platform_core").exists() or (path / "src" / "research_platform_core.py").exists()


def first_existing(candidates: list[Path], fallback: Path) -> Path:
    for candidate in candidates:
        candidate = Path(candidate).expanduser()
        if candidate.exists():
            return candidate
    return fallback


def resolve_project_root(prefer_drive: bool | None = None) -> tuple[Path, str]:
    """Resolve the project root with an explicit Drive-first default."""
    env_root = os.environ.get("RESEARCH_PLATFORM_ROOT")
    if env_root:
        path = Path(env_root).expanduser()
        if has_project_sentinel(path):
            return path.resolve(), "env:RESEARCH_PLATFORM_ROOT"

    if prefer_drive is None:
        mode = os.environ.get("RESEARCH_PLATFORM_STORAGE_MODE", "drive").strip().lower()
        prefer_drive = mode != "local"

    drive = [p for p in DRIVE_PROJECT_CANDIDATES if has_project_sentinel(p)]
    local = [p for p in LOCAL_PROJECT_CANDIDATES if has_project_sentinel(p)]
    ordered = [*drive, *local] if prefer_drive else [*local, *drive]
    if ordered:
        root = ordered[0].resolve()
        return root, "drive_discovered" if root in [p.resolve() for p in drive] else "local_discovered"

    fallback = DRIVE_PROJECT_CANDIDATES[0]
    return fallback, "missing"


def resolve_storage_roots(project_root: Path | str | None = None, prefer_drive: bool | None = None) -> StorageRoots:
    if project_root is None:
        root, source = resolve_project_root(prefer_drive=prefer_drive)
    else:
        root = Path(project_root).expanduser()
        source = "explicit"

    db_candidates = []
    if os.environ.get("FINANCIAL_DB_ROOT"):
        db_candidates.append(Path(os.environ["FINANCIAL_DB_ROOT"]))
    db_candidates.extend(FINANCIAL_DB_CANDIDATES)
    financial_db = first_existing(db_candidates, FINANCIAL_DB_CANDIDATES[0])

    output_root = Path(os.environ.get("RESEARCH_PLATFORM_OUTPUT_ROOT", root / "output")).expanduser()
    cache_root = Path(os.environ.get("RESEARCH_PLATFORM_LOCAL_CACHE", output_root / "data_cache")).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    root_text = str(root)
    storage_mode = "drive" if ("/content/drive/" in root_text or "/CloudStorage/" in root_text) else "local"
    return StorageRoots(root, financial_db, output_root, cache_root, source, storage_mode)


def export_storage_env(roots: StorageRoots) -> dict[str, str]:
    values = {
        "RESEARCH_PLATFORM_ROOT": str(roots.project_root),
        "PROJECT_ROOT": str(roots.project_root),
        "FINANCIAL_DB_ROOT": str(roots.financial_db_root),
        "DB_BASE": str(roots.financial_db_root),
        "DATA_PATH": str(roots.financial_db_root),
        "RESEARCH_PLATFORM_OUTPUT_ROOT": str(roots.output_root),
        "RESEARCH_PLATFORM_LOCAL_CACHE": str(roots.cache_root),
        "DATA_LOCAL": str(roots.cache_root),
        "COMPANY_VALUATION_DATA_LOCAL": str(roots.cache_root),
    }
    os.environ.update(values)
    return values
