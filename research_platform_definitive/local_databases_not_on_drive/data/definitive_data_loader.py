"""Definitive database helpers for ML4T research notebooks.

The loader gives every definitive notebook the same source order:
local cache, Google Drive, then project APIs. It never stores credentials;
API keys stay in environment variables or ignored `.env` files.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "database" / "definitive_database_manifest.json"


@dataclass(frozen=True)
class LoadResult:
    name: str
    source: str
    path: Path | None
    rows: int
    columns: int
    synthetic: bool
    detail: str


def load_manifest(path: Path | str = MANIFEST_PATH) -> dict:
    return json.loads(Path(path).read_text())


def manifest_frame(manifest: dict | None = None) -> pd.DataFrame:
    manifest = manifest or load_manifest()
    rows = []
    for name, spec in manifest["datasets"].items():
        rows.append({"dataset": name, **spec})
    return pd.DataFrame(rows)


def domain_frame(manifest: dict | None = None) -> pd.DataFrame:
    manifest = manifest or load_manifest()
    rows = []
    for name, spec in manifest["domains"].items():
        rows.append({"domain": name, **spec})
    return pd.DataFrame(rows)


def dataset_spec(name: str, manifest: dict | None = None) -> dict:
    manifest = manifest or load_manifest()
    try:
        return manifest["datasets"][name]
    except KeyError as exc:
        available = ", ".join(sorted(manifest["datasets"]))
        raise KeyError(f"Unknown dataset '{name}'. Available datasets: {available}") from exc


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".json":
        return pd.read_json(path)
    raise ValueError(f"Unsupported table suffix for {path}")


def load_local_dataset(name: str, manifest: dict | None = None) -> tuple[pd.DataFrame, LoadResult]:
    spec = dataset_spec(name, manifest)
    path = REPO_ROOT / spec["local_path"]
    if not path.exists():
        raise FileNotFoundError(f"Local dataset not found: {path}")
    data = _read_table(path)
    result = LoadResult(
        name=name,
        source="local",
        path=path,
        rows=len(data),
        columns=len(data.columns),
        synthetic=False,
        detail=f"loaded {spec['local_path']}",
    )
    return data, result


def download_drive_dataset(name: str, manifest: dict | None = None, cache_dir: Path | str | None = None) -> Path:
    spec = dataset_spec(name, manifest)
    file_id = spec.get("drive_file_id")
    if not file_id:
        raise ValueError(f"Dataset '{name}' has a Drive folder URL but no direct file id")

    try:
        import gdown  # type: ignore
    except Exception as exc:
        raise ImportError("Install gdown to download Google Drive files inside the notebook") from exc

    local_suffix = Path(spec["local_path"]).suffix
    target_dir = Path(cache_dir) if cache_dir else REPO_ROOT / "data" / "_drive_cache"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{name}{local_suffix}"
    url = f"https://drive.google.com/uc?id={file_id}"
    downloaded = gdown.download(url, str(target), quiet=True, fuzzy=True)
    if not downloaded or not target.exists():
        raise RuntimeError(f"Drive download failed for dataset '{name}'")
    return target


def load_drive_dataset(name: str, manifest: dict | None = None) -> tuple[pd.DataFrame, LoadResult]:
    path = download_drive_dataset(name, manifest)
    data = _read_table(path)
    result = LoadResult(
        name=name,
        source="drive",
        path=path,
        rows=len(data),
        columns=len(data.columns),
        synthetic=False,
        detail=f"downloaded from Google Drive to {path.relative_to(REPO_ROOT)}",
    )
    return data, result


def load_api_dataset(
    name: str,
    fetchers: dict[str, Callable[[], pd.DataFrame]] | None = None,
) -> tuple[pd.DataFrame, LoadResult]:
    fetchers = fetchers or {}
    if name not in fetchers:
        raise KeyError(f"No API fetcher registered for dataset '{name}'")
    data = fetchers[name]()
    result = LoadResult(
        name=name,
        source="api",
        path=None,
        rows=len(data),
        columns=len(data.columns),
        synthetic=False,
        detail="loaded through registered API fetcher",
    )
    return data, result


def load_dataset(
    name: str,
    source: str = "auto",
    manifest: dict | None = None,
    api_fetchers: dict[str, Callable[[], pd.DataFrame]] | None = None,
    logger: logging.Logger | None = None,
) -> tuple[pd.DataFrame, LoadResult]:
    """Load a dataset from local cache, Drive, or API.

    source values:
    - "auto": local -> drive -> api
    - "local": local cache only
    - "drive": Google Drive only
    - "api": registered API fetcher only
    """

    logger = logger or logging.getLogger(__name__)
    manifest = manifest or load_manifest()
    loaders = {
        "local": lambda: load_local_dataset(name, manifest),
        "drive": lambda: load_drive_dataset(name, manifest),
        "api": lambda: load_api_dataset(name, api_fetchers),
    }
    order = ["local", "drive", "api"] if source == "auto" else [source]
    errors: list[str] = []
    for candidate in order:
        try:
            return loaders[candidate]()
        except Exception as exc:
            message = f"{name}: {candidate} load failed: {exc}"
            logger.warning(message)
            errors.append(message)
    raise RuntimeError(" | ".join(errors))


def load_dotenv_files(paths: list[Path | str] | None = None) -> dict[str, str]:
    paths = paths or [REPO_ROOT / ".env.local", REPO_ROOT / ".env"]
    loaded: dict[str, str] = {}
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        for raw_line in path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
                loaded[key] = value
    return loaded


def synthetic_fallback(name: str, periods: int = 504, seed: int = 42) -> tuple[pd.DataFrame, LoadResult]:
    import numpy as np

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=periods)
    shocks = rng.normal(loc=0.0003, scale=0.012, size=periods)
    price = 100 * (1 + pd.Series(shocks, index=dates)).cumprod()
    data = pd.DataFrame(
        {
            "date": dates,
            f"{name}_return_synthetic": shocks,
            f"{name}_price_synthetic": price.to_numpy(),
            f"{name}_volume_synthetic": rng.lognormal(mean=14.0, sigma=0.35, size=periods),
        }
    )
    result = LoadResult(
        name=name,
        source="synthetic",
        path=None,
        rows=len(data),
        columns=len(data.columns),
        synthetic=True,
        detail="real data unavailable; generated realistic lognormal-volume/geometric-price fallback",
    )
    return data, result


def source_summary(results: list[LoadResult]) -> pd.DataFrame:
    return pd.DataFrame([result.__dict__ for result in results])


def write_table(df: pd.DataFrame, path: Path | str, decimals: int = 4) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.round(decimals).to_csv(path, index=False)


def temp_output_path(name: str) -> Path:
    return Path(tempfile.gettempdir()) / name
