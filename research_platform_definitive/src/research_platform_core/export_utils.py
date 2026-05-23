"""Shared export helpers with notebook-safe fallbacks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .data_utils import first_available


def output_root_from_namespace(namespace: Mapping[str, Any], default: str | Path | None = None) -> Path:
    """Resolve the canonical output root from common notebook variables."""
    root = first_available(
        namespace,
        "OUTPUTROOT",
        "OUTPUT_ROOT",
        "OUTPUT_DIR",
        "output_root",
        "output_dir",
        "TABLESDIR",
        "TABLES_DIR",
        default=default or Path.cwd() / "output",
    )
    root = Path(root)
    if root.name.lower() in {"tables", "figures", "logs", "config", "reports", "dashboard"}:
        return root.parent
    return root


def safe_write_csv(df: pd.DataFrame, path: str | Path) -> Path:
    """Write CSV through a temporary file before replacing the destination."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)
    return path


def safe_write_json(obj: Any, path: str | Path) -> Path:
    """Write JSON through a temporary file before replacing the destination."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)
    return path

