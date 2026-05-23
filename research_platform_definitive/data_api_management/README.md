# Data/API Management

Common operating layer for `Database Finanziario`, private API credentials, batch exports and `ml_stock_lab` handoff.

## Installation

From the repository root:

```bash
source .venv/bin/activate
python3 research_platform_definitive/scripts/migrate_data_api_control.py
streamlit run research_platform_definitive/data_api_app.py
```

The same view is available inside the main Streamlit app:

```text
research_platform_app/pages/11_Data_API_Control_Center.py
```

## Architecture

```text
Database Finanziario
  -> data_platform.py          # inventory, freshness, Drive-first reads
  -> api_management.py         # provider registry and masked credential status
  -> batch_download.py         # CSV/Parquet/JSON/SQLite/Excel/original export runs
  -> data_bridge.py            # manifest bridge for ml_stock_lab
  -> Data/API Control Center   # Streamlit UI and contracts
```

Configuration lives in:

```text
config/data_api_control.yaml
```

## Batch Download

Batch exports write to:

```text
output/batch_downloads/{timestamp}_{format}_export/
├── by_provider/
├── by_role/
├── manifest.csv
└── metadata.json
```

Supported modes:

- `csv`
- `parquet`
- `json`
- `sqlite`
- `excel`
- `original`

The UI caps interactive exports by default. Enable large export only when you really want to process the full filtered inventory.

## Colab Notebook

```text
data_api_management/notebooks/Data_API_Management_Colab.ipynb
```

It uses the same shared modules as Streamlit and writes the same file contracts.

## Internal API Contract

The Streamlit app currently writes file contracts for a future API layer:

```text
output/api_contracts/data_platform_contract.json
output/api_contracts/data_api_control_contract.json
```

Planned REST surface:

```http
GET /api/datasets
GET /api/datasets/{id}/download
POST /api/sync
GET /api/health
GET /api/market-context
```

## Security Policy

- Drive-first lookup from `Database Finanziario`.
- Cache-second reuse from workspace outputs.
- API-last calls only for missing, stale or incomplete datasets.
- Credential values stay in private Drive files or runtime environment.
- Exported contracts contain provider status and masked previews only.
- The app reads only headings from `api_credentials_master.md`.

## Troubleshooting

- Missing Drive root: set `FINANCIAL_DB_ROOT` or mount Google Drive in Colab.
- Missing API folder: set `API_CREDENTIALS_ROOT` or create `Database Finanziario/API`.
- No advanced grid: install `streamlit-aggrid`; the app falls back to native Streamlit tables.
- Parquet export failure: install a parquet engine such as `pyarrow`.
- Excel export failure: install `openpyxl`.
