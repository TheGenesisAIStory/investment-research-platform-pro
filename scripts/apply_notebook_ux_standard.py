#!/usr/bin/env python3
"""Inject the shared Analysis Studio control panel into notebooks.

This is intentionally conservative: it adds a lightweight setup cell only when
the notebook does not already contain the standard marker. It does not rewrite
analysis logic, outputs, or executed results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MARKER = "Analysis Studio universal controls"


def discover_notebooks(root: Path) -> list[Path]:
    ignored = {".git", ".venv", "output", "__pycache__"}
    notebooks = []
    for path in root.rglob("*.ipynb"):
        rel_parts = path.relative_to(root).parts
        if any(part in ignored for part in rel_parts):
            continue
        notebooks.append(path)
    return sorted(notebooks)


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def standard_cells(analysis_name: str) -> list[dict]:
    md = """### Analysis Studio Controls\n\nUser-friendly parameters and data-source selection shared across the workspace.\n"""
    src = f'''# {MARKER}: parameters, data source, model depth, outputs
import os
import sys
from pathlib import Path

def _find_project_root(start: Path) -> Path:
    for candidate in [start.resolve(), *start.resolve().parents]:
        if (candidate / "README.md").exists() and (candidate / "src").exists():
            return candidate
    return start.resolve()

PROJECT_ROOT = _find_project_root(Path.cwd())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.notebook_standards import (
    NotebookControlDefaults,
    build_control_panel,
    configure_notebook_runtime,
    resolve_notebook_config,
)

runtime = configure_notebook_runtime(project_root=PROJECT_ROOT)
EXPERIMENT = globals().get("EXPERIMENT", {{
    "name": "{analysis_name}",
    "target": "research_target",
    "horizons": [20],
    "test_start": "2021-01-01",
    "embargo": 20,
    "n_quantiles": 5,
    "cost_bps": 10.0,
    "models": ["baseline"],
    "run_ablation": True,
    "run_backtest": True,
    "save_figures": True,
    "feature_blocks": ["controls", "signals", "risk"],
}})
DB_BASE = runtime["DB_BASE"]
DATA_PATH = runtime["DATA_PATH"]

CONTROL_PANEL = build_control_panel(NotebookControlDefaults(analysis_name=EXPERIMENT.get("name", "{analysis_name}")))
NOTEBOOK_CONFIG = resolve_notebook_config(CONTROL_PANEL, EXPERIMENT)
print("DB_BASE:", DB_BASE)
print("DATA_PATH:", DATA_PATH)
print("Notebook config:", NOTEBOOK_CONFIG)
'''
    return [markdown_cell(md), code_cell(src)]


def inject_standard(path: Path) -> bool:
    nb = json.loads(path.read_text(encoding="utf-8"))
    text = "\n".join("".join(cell.get("source", [])) for cell in nb.get("cells", []))
    if MARKER in text or "Analysis Studio user controls" in text:
        return False

    analysis_name = path.stem.replace("-", "_").replace(" ", "_")
    cells = nb.setdefault("cells", [])
    insert_at = 1 if cells and cells[0].get("cell_type") == "markdown" else 0
    for cell in reversed(standard_cells(analysis_name)):
        cells.insert(insert_at, cell)
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Inject shared notebook UX control panel.")
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--apply", action="store_true", help="Write changes. Without this, only prints candidates.")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    candidates = discover_notebooks(args.root)
    changed = []
    for path in candidates[: args.limit]:
        nb_text = path.read_text(encoding="utf-8")
        if MARKER in nb_text or "Analysis Studio user controls" in nb_text:
            continue
        if args.apply:
            if inject_standard(path):
                changed.append(path)
        else:
            changed.append(path)
    verb = "Updated" if args.apply else "Would update"
    print(f"{verb}: {len(changed)} notebooks")
    for path in changed[:200]:
        print(path.relative_to(args.root))
    if len(changed) > 200:
        print(f"... {len(changed) - 200} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

