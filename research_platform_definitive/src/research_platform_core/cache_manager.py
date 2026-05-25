"""Small disk cache with TTL metadata for API/data refreshes."""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .data_platform import utc_now


def stable_cache_key(*parts: object) -> str:
    text = "::".join(str(part) for part in parts)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


class DataCache:
    """Dependency-light disk cache for DataFrame/object results."""

    def __init__(self, cache_dir: Path | str):
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _paths(self, key: str) -> tuple[Path, Path]:
        safe_key = stable_cache_key(key)
        return self.cache_dir / f"{safe_key}.pkl", self.cache_dir / f"{safe_key}.json"

    def get(self, key: str, ttl_hours: float | None = None) -> Any | None:
        data_path, meta_path = self._paths(key)
        if not data_path.exists() or not meta_path.exists():
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            created = pd.Timestamp(meta.get("created_at"), tz="UTC")
            age_hours = (pd.Timestamp.now(tz="UTC") - created).total_seconds() / 3600
            if ttl_hours is not None and age_hours > ttl_hours:
                return None
            with data_path.open("rb") as handle:
                return pickle.load(handle)
        except Exception:
            return None

    def set(self, key: str, value: Any, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        data_path, meta_path = self._paths(key)
        with data_path.open("wb") as handle:
            pickle.dump(value, handle)
        meta = {
            "key": key,
            "created_at": utc_now(),
            "data_path": str(data_path),
            **(metadata or {}),
        }
        meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        return meta

    def get_or_fetch(self, key: str, fetch_func: Callable[[], Any], ttl_hours: float = 24, metadata: dict[str, Any] | None = None) -> Any:
        cached = self.get(key, ttl_hours=ttl_hours)
        if cached is not None:
            return cached
        value = fetch_func()
        self.set(key, value, metadata=metadata)
        return value
