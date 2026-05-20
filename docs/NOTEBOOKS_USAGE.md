# Notebook Usage

`00_notebooks_final/` is the canonical notebook folder for Analysis Studio.

The project currently does not use a single root-level `notebooks/` directory, so notebooks are not duplicated elsewhere. This keeps GitHub clean and prevents two versions of the same professional workflow from drifting apart.

## Running Locally

From the repository root:

```bash
jupyter notebook 00_notebooks_final/01_stock_screening_professional.ipynb
```

Each notebook:

- configures `DB_BASE` and `DATA_PATH`
- exposes Analysis Studio controls for parameters, data source, and model depth
- creates missing output folders
- imports the relevant Analysis Studio engine
- runs end-to-end with fallback data
- saves CSV, Markdown, HTML, and chart exports under Google Drive
- ends with an artifact counter

## Running in VS Code

Open the notebook from `00_notebooks_final/` and select the repository virtual environment. If imports fail, run the first setup cell again from the repository root.

## Running in Colab

Upload or clone the whole repository, not only an isolated notebook. The setup cell tries to detect common Colab paths and adds the project root to `sys.path`.

Mount Drive if you need persistent outputs:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Then set:

```python
DB_BASE = Path("/content/drive/MyDrive/Database Finanziario")
DATA_PATH = DB_BASE
```

## Notebook List

- `01_stock_screening_professional.ipynb`
- `02_portfolio_risk_bridgewater.ipynb`
- `03_pre_earnings_jpm.ipynb`
- `04_portfolio_builder_blackrock.ipynb`
- `05_technical_analysis_citadel.ipynb`
- `06_competitive_analysis_bain.ipynb`
- `07_quant_research_renaissance.ipynb`
- `08_macro_strategy_mckinsey.ipynb`

`99_colab_final_patch.ipynb` remains a utility notebook for Colab environment checks.

## Canonical Professional Notebook

The canonical standard for future professional notebooks and dashboards is:

```text
company_valuation/notebooks/Company_Valuation_Final_Version.ipynb
```

Use it as the reference pattern for:

- central `MASTERREQUEST` / config-state design
- visible Control Center and interactive user inputs
- local project database plus API ingestion
- validated data pipeline and diagnostics
- formula-rich methodology sections
- valuation, peer intelligence, model lab, risk, diagnostics, sensitivity, and scenario sections
- final Plotly/HTML dashboard with dedicated result tabs
- CSV, HTML, Markdown, config snapshot, and static dashboard exports under `DB_BASE`

Future portfolio notebooks should reuse the company valuation UX/dashboard pattern and adapt it to portfolio analytics. Do not treat portfolio-specific prototypes as the canonical standard.

## Quality Audit

Run:

```bash
python scripts/audit_notebook_quality.py --include-drive-benchmarks
```

The report is saved to:

- `output/review/notebook_quality_audit.csv`
- `output/review/Notebook_Quality_Audit.md`
- `DB_BASE / "analysis_outputs/notebook_quality/"`

The current standardization pass ensures every notebook has the shared parameter/source control layer. The audit also identifies deeper work still needed, especially adding 0-14 structure, Plotly charts, and exports to older educational notebooks.
