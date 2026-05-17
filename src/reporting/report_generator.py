"""CSV, Markdown, HTML, and chart exports for Analysis Studio."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.analysis.common import StudioResult, dataframe_preview_markdown
from src.analysis.config import (
    ANALYSIS_REGISTRY,
    CHART_EXPORTS,
    CSV_EXPORTS,
    HTML_EXPORTS,
    MARKDOWN_EXPORTS,
    analysis_output_dir,
    ensure_base_directories,
)


def _safe_slug(value: str) -> str:
    return value.lower().replace(" ", "_").replace("/", "_")


def _chart_to_html(chart: object) -> str:
    if hasattr(chart, "to_html"):
        return chart.to_html(full_html=False, include_plotlyjs="cdn")
    return ""


def export_analysis_report(result: StudioResult | pd.DataFrame, analysis_name: str | None = None, summary: str | None = None, charts: Iterable[object] | None = None) -> dict[str, str]:
    """Persist one analysis result under DB_BASE and notebook export folders."""
    ensure_base_directories()

    if isinstance(result, StudioResult):
        name = result.analysis_name
        data = result.data
        summary_text = result.summary
        chart_list = list(result.charts)
        metadata = result.metadata
    else:
        if analysis_name is None:
            raise ValueError("analysis_name is required when result is a DataFrame")
        name = analysis_name
        data = result
        summary_text = summary or f"{name} analysis completed."
        chart_list = list(charts or [])
        metadata = {}

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    title = ANALYSIS_REGISTRY.get(name, {}).get("title", name.replace("_", " ").title())
    output_dir = analysis_output_dir(name)
    stem = f"{_safe_slug(name)}_{timestamp}"

    csv_path = output_dir / f"{stem}.csv"
    export_csv_path = CSV_EXPORTS / f"{stem}.csv"
    markdown_path = MARKDOWN_EXPORTS / f"{stem}.md"
    html_path = HTML_EXPORTS / f"{stem}.html"
    manifest_path = output_dir / f"{stem}_manifest.json"

    rounded = data.copy()
    float_cols = rounded.select_dtypes(include=["float", "float64", "float32"]).columns
    rounded[float_cols] = rounded[float_cols].round(6)
    rounded.to_csv(csv_path, index=False)
    rounded.to_csv(export_csv_path, index=False)

    chart_paths: list[str] = []
    chart_html_parts: list[str] = []
    for idx, chart in enumerate(chart_list, start=1):
        chart_html = _chart_to_html(chart)
        if not chart_html:
            continue
        chart_path = CHART_EXPORTS / f"{stem}_chart_{idx}.html"
        chart_path.write_text(f"<html><body>{chart_html}</body></html>", encoding="utf-8")
        chart_paths.append(str(chart_path))
        chart_html_parts.append(chart_html)

    preview = dataframe_preview_markdown(rounded)
    markdown = "\n".join(
        [
            f"# {title}",
            "",
            f"Generated UTC: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
            "",
            "## Executive Summary",
            "",
            summary_text,
            "",
            "## Output Files",
            "",
            f"- Analysis CSV: `{csv_path}`",
            f"- Export CSV: `{export_csv_path}`",
            f"- HTML report: `{html_path}`",
            f"- Charts: {len(chart_paths)}",
            "",
            "## Data Preview",
            "",
            preview,
            "",
        ]
    )
    markdown_path.write_text(markdown, encoding="utf-8")

    html_table = rounded.head(50).to_html(index=False, border=0, classes="analysis-table")
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 32px; color: #1f2933; }}
    h1, h2 {{ color: #01696f; }}
    .summary {{ border-left: 4px solid #da7101; padding-left: 14px; font-size: 1.05rem; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 0.9rem; }}
    th, td {{ border-bottom: 1px solid #d8dee4; padding: 8px; text-align: left; }}
    th {{ background: #f7f6f2; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <p>Generated UTC: {datetime.now(timezone.utc).isoformat(timespec='seconds')}</p>
  <h2>Executive Summary</h2>
  <p class="summary">{summary_text}</p>
  <h2>Charts</h2>
  {''.join(chart_html_parts) if chart_html_parts else '<p>No charts available.</p>'}
  <h2>Data Preview</h2>
  {html_table}
</body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")

    manifest = {
        "analysis_name": name,
        "title": title,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": int(len(rounded)),
        "columns": list(rounded.columns),
        "metadata": metadata,
        "csv_path": str(csv_path),
        "export_csv_path": str(export_csv_path),
        "markdown_path": str(markdown_path),
        "html_path": str(html_path),
        "chart_paths": chart_paths,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return {key: str(value) if isinstance(value, Path) else value for key, value in manifest.items()}

