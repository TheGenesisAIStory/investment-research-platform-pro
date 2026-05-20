# Analysis Studio

Analysis Studio is the professional analysis layer integrated into this local ML Trading workspace.

It is not a separate project. It reuses the repository root, writes persistent outputs to Google Drive, and exposes the same engines through notebooks, CLI, Streamlit, and the weekly scheduler.

## Canonical Layout

```text
src/analysis/              reusable analysis engines
src/reporting/             CSV, Markdown, HTML, chart exports
src/cli/                   command-line interface
src/dashboard/             shared Streamlit renderer
dashboard/                 Streamlit app and pages
00_notebooks_final/        canonical Analysis Studio notebooks
scripts/weekly_update.py   scheduled batch runner
```

There is no root `notebooks/` folder in this workspace, so `00_notebooks_final/` remains the canonical home for the professional notebooks. Do not duplicate them into another folder unless the project later adopts a single repo-wide notebook convention.

## Persistent Data and Outputs

All persistent Analysis Studio artifacts are written under:

```python
from pathlib import Path

DB_BASE = Path("/Users/itsgennymac/Library/CloudStorage/GoogleDrive-sfn.gns@gmail.com/Il mio Drive/Database Finanziario")
```

Expected output folders:

```text
DB_BASE / "analysis_outputs/stock_screening"
DB_BASE / "analysis_outputs/portfolio_risk"
DB_BASE / "analysis_outputs/earnings_analysis"
DB_BASE / "analysis_outputs/portfolio_builder"
DB_BASE / "analysis_outputs/technical_analysis"
DB_BASE / "analysis_outputs/competitive_analysis"
DB_BASE / "analysis_outputs/quantitative_research"
DB_BASE / "analysis_outputs/macro_analysis"
DB_BASE / "notebook_exports/html"
DB_BASE / "notebook_exports/markdown"
DB_BASE / "notebook_exports/csv"
DB_BASE / "notebook_exports/charts"
```

The repo should not store heavy datasets or generated research exports.

## CLI

Run from the project root:

```bash
python -m src.cli.analysis_commands screen --universe sp500 --sector technology
python -m src.cli.analysis_commands risk --portfolio-file positions.csv
python -m src.cli.analysis_commands earnings --ticker NVDA
python -m src.cli.analysis_commands build --risk-profile moderate
python -m src.cli.analysis_commands technical --ticker AAPL
python -m src.cli.analysis_commands competitive --sector semiconductors
python -m src.cli.analysis_commands quant --ticker MSFT
python -m src.cli.analysis_commands macro --portfolio-file positions.csv
```

There is also a local executable wrapper:

```bash
./analysis screen --universe core --top-n 10
scripts/analysis screen --universe core --top-n 10
```

Use `--offline` to force local/synthetic fallback mode.

## Streamlit Dashboard

Run from the project root:

```bash
streamlit run dashboard/Analysis_Studio.py
```

The app exposes dedicated pages for all eight workflows. Each page has sidebar filters, a run button, Plotly charts, downloadable tables, executive summary text, and report export paths.

## Engines

Each engine follows the same interface:

```python
engine = StockScreenerEngine(universe="core")
result_df = engine.run()
manifest = export_analysis_report(engine.last_result)
```

The `run()` method returns a DataFrame for notebook ergonomics. The richer `engine.last_result` object contains summary text, metadata, and charts for CLI/dashboard/reporting.

## Data Philosophy

The engines try local/remote real data where practical and fall back to realistic synthetic data when unavailable. Synthetic columns are explicitly marked with suffixes such as `_synthetic`, and warnings are logged by the shared data helpers.

## Notebook UX Standard

The shared notebook control layer lives in `src/notebook_standards/`.

Use it directly:

```python
from src.notebook_standards import NotebookControlDefaults, build_control_panel
```

Audit notebooks:

```bash
python scripts/audit_notebook_quality.py --include-drive-benchmarks
```

Apply the common control panel to notebooks that do not yet have it:

```bash
python scripts/apply_notebook_ux_standard.py --apply
```
