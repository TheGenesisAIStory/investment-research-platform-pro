#!/usr/bin/env python3
"""Validate restored laboratorio reference notebooks."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAB_ROOT = ROOT / "archive" / "laboratorio"
MANIFEST = LAB_ROOT / "LABORATORIO_MANIFEST.csv"


def main() -> int:
    if not LAB_ROOT.exists():
        print(f"FAIL missing laboratorio folder: {LAB_ROOT}")
        return 1
    if not MANIFEST.exists():
        print(f"FAIL missing manifest: {MANIFEST}")
        return 1

    rows = list(csv.DictReader(MANIFEST.open(encoding="utf-8")))
    notebooks = [row for row in rows if row.get("kind") == "ipynb"]
    failures: list[str] = []

    for row in notebooks:
        path = Path(row["bundle_path"])
        if not path.is_absolute():
            path = ROOT.parent / path
        if not path.exists() and "/laboratorio/" in str(path):
            path = Path(str(path).replace("/laboratorio/", "/archive/laboratorio/"))
        if not path.exists():
            failures.append(f"MISSING {row['relative_path']}")
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            failures.append(f"INVALID_JSON {row['relative_path']}: {type(exc).__name__}")
            continue
        if "cells" not in payload or "nbformat" not in payload:
            failures.append(f"INVALID_SCHEMA {row['relative_path']}")

    if failures:
        for failure in failures[:50]:
            print("FAIL", failure)
        if len(failures) > 50:
            print(f"... {len(failures) - 50} additional failures")
        return 1

    print("laboratorio_OK")
    print(f"files={len(rows)}")
    print(f"notebooks={len(notebooks)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
