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

