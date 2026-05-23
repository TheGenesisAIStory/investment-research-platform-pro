"""nbclient fallback notebook runner."""

from __future__ import annotations

from pathlib import Path

from .models import NotebookJob
from .utils import PROJECT_ROOT, append_log, ensure_dir


def run_with_nbclient(job: NotebookJob, parameters: dict, output_notebook: Path, log_path: Path) -> None:
    if job.notebook_path is None:
        raise ValueError("nbclient runner requires a notebook_path")
    try:
        import nbformat
        from nbclient import NotebookClient
    except Exception as exc:
        raise RuntimeError("nbclient and nbformat are required for fallback notebook execution") from exc

    ensure_dir(output_notebook.parent)
    append_log(log_path, f"Executing with nbclient: {job.notebook_path}")
    notebook = nbformat.read(job.notebook_path, as_version=4)
    if parameters:
        source = "\n".join(f"{key} = {value!r}" for key, value in parameters.items())
        notebook.cells.insert(0, nbformat.v4.new_code_cell(source, metadata={"tags": ["injected-parameters"]}))
        append_log(log_path, f"Injected parameters: {sorted(parameters)}")
    client = NotebookClient(
        notebook,
        timeout=job.timeout_seconds,
        kernel_name="python3",
        resources={"metadata": {"path": str(PROJECT_ROOT)}},
    )
    client.execute()
    nbformat.write(notebook, output_notebook)
    append_log(log_path, f"Executed notebook written to: {output_notebook}")
