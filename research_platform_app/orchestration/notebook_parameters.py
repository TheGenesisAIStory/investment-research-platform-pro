"""Notebook parameter-cell inspection and conservative injection."""

from __future__ import annotations

from pathlib import Path


def has_parameters_cell(notebook_path: Path) -> bool:
    try:
        import nbformat

        nb = nbformat.read(notebook_path, as_version=4)
        return any("parameters" in cell.get("metadata", {}).get("tags", []) for cell in nb.cells)
    except Exception:
        return False


def ensure_parameters_cell(notebook_path: Path, parameter_names: list[str], dry_run: bool = False) -> dict:
    """Add a Papermill-compatible parameters cell if one is missing."""
    try:
        import nbformat
    except Exception as exc:
        return {"ok": False, "changed": False, "message": f"nbformat unavailable: {exc}"}
    if not notebook_path.exists():
        return {"ok": False, "changed": False, "message": f"Notebook not found: {notebook_path}"}
    nb = nbformat.read(notebook_path, as_version=4)
    if any("parameters" in cell.get("metadata", {}).get("tags", []) for cell in nb.cells):
        return {"ok": True, "changed": False, "message": "Parameters cell already exists"}
    source = "# Parameters\n" + "\n".join(f"{name} = None" for name in parameter_names) + "\n"
    cell = nbformat.v4.new_code_cell(source, metadata={"tags": ["parameters"]})
    nb.cells.insert(0, cell)
    if not dry_run:
        backup = notebook_path.with_suffix(notebook_path.suffix + ".pre_parameters_backup")
        if not backup.exists():
            backup.write_bytes(notebook_path.read_bytes())
        nbformat.write(nb, notebook_path)
    return {"ok": True, "changed": True, "message": "Parameters cell inserted"}
