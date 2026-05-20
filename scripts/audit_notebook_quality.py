#!/usr/bin/env python3
"""Audit notebook UX, graphics, modeling depth, and export readiness."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.config import DB_BASE

STANDARD_HEADERS = [
    "## 0. Setup & Config",
    "## 1. Data Ingestion",
    "## 2. Cleaning & Alignment",
    "## 3. Feature Engineering",
    "## 4. Targets & Labels",
    "## 5. Descriptive Stats",
    "## 6. Exploratory / Event Study",
    "## 7. Single-Factor Diagnostics",
    "## 8. Statistical Models (regressions / econometrics)",
    "## 9. ML Walk-Forward",
    "## 10. Feature Ablation",
    "## 11. Backtest / Strategy Evaluation",
    "## 12. Interpretability",
    "## 13. Robustness Checks",
    "## 14. Final Summary",
]

PATTERNS = {
    "plotly_hits": r"plotly|go\.Figure|px\.",
    "widget_hits": r"ipywidgets|widgets\.|interact\(|#@param|Dropdown|Select|Slider",
    "source_hits": r"DATA_PATH|DB_BASE|data_source|Database Finanziario|drive\.mount|yfinance|read_csv|read_parquet",
    "export_hits": r"to_csv|write_html|to_html|to_markdown|savefig|export_analysis_report|manifest",
    "model_hits": r"RandomForest|XGB|LightGBM|CatBoost|Ridge|Lasso|ElasticNet|LinearRegression|LogisticRegression|statsmodels|OLS|ARIMA|GARCH|walk_forward|backtest",
    "fallback_hits": r"synthetic|fallback|try:|except|WARNING|logger\.warning",
}


@dataclass
class NotebookAudit:
    path: str
    cells: int
    markdown_cells: int
    code_cells: int
    has_standard_headers: bool
    header_count: int
    plotly_hits: int
    widget_hits: int
    source_hits: int
    export_hits: int
    model_hits: int
    fallback_hits: int
    score: int
    status: str
    recommendations: str


def _read_notebook(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def audit_notebook(path: Path, root: Path = REPO_ROOT) -> NotebookAudit:
    nb = _read_notebook(path)
    cells = nb.get("cells", [])
    text = "\n".join("".join(cell.get("source", [])) for cell in cells)
    headers = ["".join(cell.get("source", [])).split("\n", 1)[0].strip() for cell in cells if cell.get("cell_type") == "markdown"]
    found_headers = sum(1 for header in STANDARD_HEADERS if header in text)
    has_standard_headers = found_headers == len(STANDARD_HEADERS)
    metrics = {name: len(re.findall(pattern, text, flags=re.IGNORECASE)) for name, pattern in PATTERNS.items()}
    score = 0
    score += 25 if has_standard_headers else min(20, found_headers)
    score += min(15, metrics["widget_hits"] * 3)
    score += min(15, metrics["plotly_hits"] * 2)
    score += min(15, metrics["source_hits"] * 2)
    score += min(15, metrics["export_hits"] * 2)
    score += min(10, metrics["model_hits"])
    score += min(5, metrics["fallback_hits"])
    recommendations = []
    if not has_standard_headers:
        recommendations.append("add 0-14 section scaffold")
    if metrics["widget_hits"] == 0:
        recommendations.append("add user-friendly parameter/source controls")
    if metrics["plotly_hits"] < 2:
        recommendations.append("add at least two Plotly charts")
    if metrics["export_hits"] < 3:
        recommendations.append("save CSV/HTML/Markdown outputs under DB_BASE")
    if metrics["source_hits"] < 2:
        recommendations.append("standardize DB_BASE/DATA_PATH and data source choice")
    if metrics["model_hits"] < 3:
        recommendations.append("document or deepen model diagnostics")
    status = "gold" if score >= 85 else "silver" if score >= 65 else "bronze" if score >= 45 else "needs_work"
    return NotebookAudit(
        path=str(path.relative_to(root) if path.is_relative_to(root) else path),
        cells=len(cells),
        markdown_cells=sum(1 for cell in cells if cell.get("cell_type") == "markdown"),
        code_cells=sum(1 for cell in cells if cell.get("cell_type") == "code"),
        has_standard_headers=has_standard_headers,
        header_count=found_headers,
        score=score,
        status=status,
        recommendations="; ".join(recommendations) or "meets professional standard",
        **metrics,
    )


def discover_notebooks(root: Path) -> list[Path]:
    ignored_parts = {".git", ".venv", "output", "__pycache__"}
    notebooks = []
    for path in root.rglob("*.ipynb"):
        if any(part in ignored_parts for part in path.parts):
            continue
        if path.parts[:1] == ("data",):
            continue
        notebooks.append(path)
    return sorted(notebooks)


def write_reports(frame: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "notebook_quality_audit.csv"
    md_path = output_dir / "Notebook_Quality_Audit.md"
    frame.to_csv(csv_path, index=False)
    summary = frame.groupby("status").size().rename("N").reset_index().sort_values("status")
    folder_frame = frame.copy()
    folder_frame["folder"] = folder_frame["path"].map(lambda value: str(value).split("/", 1)[0])
    folder_summary = (
        folder_frame.groupby("folder", as_index=False)
        .agg(notebooks=("path", "count"), avg_score=("score", "mean"), min_score=("score", "min"))
        .sort_values(["avg_score", "folder"], ascending=[False, True])
    )
    lowest = frame.sort_values("score").head(25)[["path", "score", "status", "recommendations"]]
    try:
        summary_md = summary.to_markdown(index=False)
        folder_md = folder_summary.to_markdown(index=False)
        lowest_md = lowest.to_markdown(index=False)
    except Exception:
        summary_md = "```\n" + summary.to_string(index=False) + "\n```"
        folder_md = "```\n" + folder_summary.to_string(index=False) + "\n```"
        lowest_md = "```\n" + lowest.to_string(index=False) + "\n```"
    lines = [
        "# Notebook Quality Audit",
        "",
        f"Notebook count: {len(frame)}",
        "",
        "## Status Summary",
        "",
        summary_md,
        "",
        "## Folder Summary",
        "",
        folder_md,
        "",
        "## Lowest Scoring Notebooks",
        "",
        lowest_md,
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, md_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit notebook UX/graphics/model/export quality.")
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--include-drive-benchmarks", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "output" / "review")
    args = parser.parse_args()

    notebooks = discover_notebooks(args.root)
    if args.include_drive_benchmarks:
        notebooks.extend(
            [
                Path("/Users/itsgennymac/Library/CloudStorage/GoogleDrive-sfn.gns@gmail.com/Il mio Drive/Colab Notebooks/Company_Valuation_Final_Version.ipynb"),
                Path("/Users/itsgennymac/Library/CloudStorage/GoogleDrive-sfn.gns@gmail.com/Il mio Drive/Colab Notebooks/Portfolio-Analysis-Model.ipynb"),
            ]
        )
    audits = [audit_notebook(path, root=args.root) for path in notebooks if path.exists()]
    frame = pd.DataFrame([audit.__dict__ for audit in audits]).sort_values(["score", "path"], ascending=[False, True])
    csv_path, md_path = write_reports(frame, args.output_dir)

    db_output_dir = DB_BASE / "analysis_outputs" / "notebook_quality"
    db_csv, db_md = write_reports(frame, db_output_dir)
    print(f"Audited notebooks: {len(frame)}")
    print(f"Repo CSV: {csv_path}")
    print(f"Repo Markdown: {md_path}")
    print(f"DB CSV: {db_csv}")
    print(f"DB Markdown: {db_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
