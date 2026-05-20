"""Reusable user-friendly notebook controls.

The functions in this module are intentionally lightweight: they can be used in
Jupyter, VS Code notebooks, and Colab, while falling back to plain dictionaries
when ipywidgets is not installed or the notebook runs headlessly.
"""

from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DB_BASE = Path(
    os.environ.get(
        "ML_TRADING_DB_BASE",
        "/Users/itsgennymac/Library/CloudStorage/"
        "GoogleDrive-sfn.gns@gmail.com/Il mio Drive/Database Finanziario",
    )
).expanduser()

DATA_SOURCE_OPTIONS = [
    "auto",
    "database_finanziario",
    "local_repo_cache",
    "yfinance",
    "csv_upload",
    "synthetic_fallback",
]

MODEL_DEPTH_OPTIONS = [
    "quick",
    "standard",
    "institutional",
    "research_deep_dive",
]

DEFAULT_OUTPUT_FOLDERS = [
    "analysis_outputs",
    "notebook_exports/html",
    "notebook_exports/markdown",
    "notebook_exports/csv",
    "notebook_exports/charts",
    "logs",
]


@dataclass
class NotebookControlDefaults:
    """Default controls shared by professional research notebooks."""

    analysis_name: str
    ticker: str = "AAPL"
    universe: str = "core"
    sector: str = "technology"
    data_source: str = "auto"
    model_depth: str = "standard"
    risk_profile: str = "moderate"
    start_date: str = "2018-01-01"
    end_date: str = "2026-01-01"
    forecast_horizon: int = 20
    run_backtest: bool = True
    run_ablation: bool = True
    save_outputs: bool = True


def configure_notebook_runtime(project_root: str | Path | None = None, db_base: str | Path | None = None) -> dict[str, Path]:
    """Configure sys.path, DB_BASE, DATA_PATH, and persistent output folders."""
    root = Path(project_root).resolve() if project_root is not None else _find_project_root(Path.cwd())
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    base = Path(db_base).expanduser() if db_base is not None else DB_BASE
    os.environ["ML_TRADING_DB_BASE"] = str(base)
    os.environ["DATA_PATH"] = str(base)
    for relative in DEFAULT_OUTPUT_FOLDERS:
        (base / relative).mkdir(parents=True, exist_ok=True)

    return {"PROJECT_ROOT": root, "DB_BASE": base, "DATA_PATH": base}


def build_control_panel(defaults: NotebookControlDefaults | dict[str, Any]) -> dict[str, Any]:
    """Build and display an ipywidgets control panel when available.

    Returns a dictionary containing either widget objects or plain fallback
    values. Use `resolve_notebook_config()` after users have changed controls.
    """
    values = asdict(defaults) if isinstance(defaults, NotebookControlDefaults) else dict(defaults)
    try:
        import ipywidgets as widgets
        from IPython.display import Markdown, display
    except Exception:
        values["widgets_available"] = False
        return values

    style = {"description_width": "130px"}
    layout = widgets.Layout(width="360px")
    controls = {
        "analysis_name": widgets.Text(value=str(values.get("analysis_name", "analysis")), description="Analysis", style=style, layout=layout),
        "ticker": widgets.Text(value=str(values.get("ticker", "AAPL")), description="Ticker", style=style, layout=layout),
        "universe": widgets.Dropdown(options=["core", "sp500", "technology", "semiconductors", "banks", "macro"], value=values.get("universe", "core"), description="Universe", style=style, layout=layout),
        "sector": widgets.Dropdown(options=["technology", "semiconductors", "financials", "banks", "macro", "none"], value=values.get("sector", "technology"), description="Sector", style=style, layout=layout),
        "data_source": widgets.Dropdown(options=DATA_SOURCE_OPTIONS, value=values.get("data_source", "auto"), description="Data source", style=style, layout=layout),
        "model_depth": widgets.Dropdown(options=MODEL_DEPTH_OPTIONS, value=values.get("model_depth", "standard"), description="Model depth", style=style, layout=layout),
        "risk_profile": widgets.Dropdown(options=["conservative", "moderate", "aggressive"], value=values.get("risk_profile", "moderate"), description="Risk profile", style=style, layout=layout),
        "start_date": widgets.Text(value=str(values.get("start_date", "2018-01-01")), description="Start date", style=style, layout=layout),
        "end_date": widgets.Text(value=str(values.get("end_date", "2026-01-01")), description="End date", style=style, layout=layout),
        "forecast_horizon": widgets.IntSlider(value=int(values.get("forecast_horizon", 20)), min=1, max=252, step=1, description="Horizon", style=style, layout=layout),
        "run_backtest": widgets.Checkbox(value=bool(values.get("run_backtest", True)), description="Run backtest", indent=False),
        "run_ablation": widgets.Checkbox(value=bool(values.get("run_ablation", True)), description="Run ablation", indent=False),
        "save_outputs": widgets.Checkbox(value=bool(values.get("save_outputs", True)), description="Save outputs", indent=False),
    }
    panel = widgets.VBox(
        [
            widgets.HTML("<h3>Analysis Studio Controls</h3>"),
            widgets.HTML("<p>Choose parameters and data source before running the notebook.</p>"),
            widgets.HBox([widgets.VBox(list(controls.values())[:6]), widgets.VBox(list(controls.values())[6:])]),
        ]
    )
    display(Markdown("### User-friendly parameters and data sources"))
    display(panel)
    controls["widgets_available"] = True
    return controls


def resolve_notebook_config(panel: dict[str, Any], experiment: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolve widget/fallback controls into a plain configuration dictionary."""
    resolved: dict[str, Any] = {}
    for key, value in panel.items():
        if key == "widgets_available":
            continue
        resolved[key] = getattr(value, "value", value)
    if resolved.get("sector") == "none":
        resolved["sector"] = None
    if experiment is not None:
        experiment.update(
            {
                "data_source": resolved.get("data_source"),
                "model_depth": resolved.get("model_depth"),
                "start_date": resolved.get("start_date"),
                "end_date": resolved.get("end_date"),
                "horizons": [int(resolved.get("forecast_horizon", 20))],
                "run_backtest": bool(resolved.get("run_backtest", True)),
                "run_ablation": bool(resolved.get("run_ablation", True)),
                "save_figures": bool(resolved.get("save_outputs", True)),
            }
        )
    return resolved


def _find_project_root(start: Path) -> Path:
    for candidate in [start.resolve(), *start.resolve().parents]:
        if (candidate / "README.md").exists() and (candidate / "src").exists():
            return candidate
    return start.resolve()

