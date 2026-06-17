#!/usr/bin/env python3
"""Validate canonical research notebooks without touching legacy backups."""

from __future__ import annotations

import json
import re
import argparse
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class NotebookContract:
    path: str
    required_text: tuple[str, ...]
    setup_markers: tuple[str, ...] = ("PROJECT_ROOT", "sys.path")


NOTEBOOKS = (
    NotebookContract(
        "company_valuation/notebooks/Company_Valuation_Final_Version.ipynb",
        ("ml_stock_lab", "smart_money", "Data Platform"),
    ),
    NotebookContract(
        "portfolio_analysis/notebooks/Portfolio-Analysis-Model_RESEARCH_PLATFORM_PRO.ipynb",
        ("ml_stock_lab", "smart_money", "Data Platform"),
    ),
    NotebookContract(
        "machine_learning_lab/notebooks/ML_Stock_Lab_Experiments.ipynb",
        ("ml_stock_lab", "run_ml_stock_lab_experiment", "ML Stock Lab"),
    ),
    NotebookContract(
        "machine_learning_lab/notebooks/ML_Training_Lab_Colab_Ollama.ipynb",
        ("train_ml_model_suite", "Ollama", "ML Training Lab"),
    ),
)

FORBIDDEN_PATH_PATTERNS = (
    r"/Users/[^\\s'\"]+/",
    r"C:\\\\Users\\\\",
    r"/home/[^\\s'\"]+/",
    r"/Desktop/",
)


def _notebook_text(nb: dict) -> str:
    chunks: list[str] = []
    for cell in nb.get("cells", []):
        source = cell.get("source", "")
        if isinstance(source, list):
            chunks.extend(source)
        else:
            chunks.append(str(source))
    return "\n".join(chunks)


def _first_non_empty_code_cell(nb: dict) -> str:
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        if str(source).strip():
            return str(source)
    return ""


def _early_code_text(nb: dict, limit: int = 12) -> str:
    code_cells: list[str] = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        code_cells.append("".join(source) if isinstance(source, list) else str(source))
        if len(code_cells) >= limit:
            break
    return "\n".join(code_cells)


def validate_notebook(contract: NotebookContract, strict: bool = False) -> dict[str, object]:
    path = ROOT / contract.path
    if not path.exists():
        return {"path": contract.path, "status": "MISSING", "message": "notebook file not found"}

    try:
        nb = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"path": contract.path, "status": "INVALID_JSON", "message": str(exc)}

    if nb.get("nbformat") is None or "cells" not in nb:
        return {"path": contract.path, "status": "INVALID_NOTEBOOK", "message": "missing nbformat/cells"}

    text = _notebook_text(nb)
    warnings: list[str] = []
    missing = [needle for needle in contract.required_text if needle not in text]
    if missing:
        return {
            "path": contract.path,
            "status": "MISSING_BRIDGE_TEXT",
            "message": ", ".join(missing),
            "cells": len(nb.get("cells", [])),
        }

    first_code = _first_non_empty_code_cell(nb)
    if "# Parameters" not in first_code and "Parameters" not in first_code and "PROJECT_ROOT" not in first_code:
        warnings.append("first code cell is not a Parameters/setup cell")

    early_text = _early_code_text(nb)
    missing_setup = [marker for marker in contract.setup_markers if marker not in early_text]
    if missing_setup:
        warnings.append("early setup markers missing: " + ", ".join(missing_setup))

    path_hits: list[str] = []
    for pattern in FORBIDDEN_PATH_PATTERNS:
        path_hits.extend(sorted(set(re.findall(pattern, text))))
    if path_hits:
        warnings.append("hardcoded local path patterns: " + ", ".join(path_hits[:3]))

    if re.search(r"except\s*:\s*(?:\n\s*)?pass\b", text):
        warnings.append("silent bare except/pass detected")

    if warnings and strict:
        return {
            "path": contract.path,
            "status": "WARNING_AS_ERROR",
            "message": " | ".join(warnings),
            "cells": len(nb.get("cells", [])),
        }

    status = "OK_WITH_WARNINGS" if warnings else "OK"
    return {"path": contract.path, "status": status, "message": " | ".join(warnings), "cells": len(nb.get("cells", []))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true", help="Treat notebook hygiene warnings as failures.")
    args = parser.parse_args()

    rows = [validate_notebook(contract, strict=args.strict) for contract in NOTEBOOKS]
    width = max(len(row["path"]) for row in rows)
    failed = False
    for row in rows:
        status = str(row["status"])
        marker = "OK" if status in {"OK", "OK_WITH_WARNINGS"} else "FAIL"
        print(f"{marker:4} {str(row['path']):<{width}}  {status}  {row.get('message', '')}")
        failed = failed or status not in {"OK", "OK_WITH_WARNINGS"}
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
