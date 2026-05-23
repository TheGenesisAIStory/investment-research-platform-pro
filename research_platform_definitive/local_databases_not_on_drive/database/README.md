# Definitive Database Manifest

This folder records the repository-wide database manifest used by the definitive research notebooks.

| File | Purpose |
|---|---|
| `definitive_database_manifest.json` | Maps canonical dataset names to local cache paths, Google Drive ids/URLs, domains, and dataset types |
| `../data/definitive_data_loader.py` | Shared loader for local cache, Google Drive downloads, registered API fetchers, and realistic synthetic fallback data |

Local Google Drive for Desktop path:

```text
/Users/itsgennymac/Library/CloudStorage/GoogleDrive-s.genise50@studenti.poliba.it/Il mio Drive/Database Finanziario
```

The loader follows the notebook data rules in `AGENTS.md`: try real local or remote data first, log failed attempts, and mark synthetic fallback columns with `_synthetic`.

Typical notebook usage:

```python
from data.definitive_data_loader import (
    load_dataset,
    load_manifest,
    manifest_frame,
    source_summary,
    synthetic_fallback,
)

manifest = load_manifest()
catalog = manifest_frame(manifest)
data, result = load_dataset("api_providers", manifest=manifest)
print(source_summary([result]).to_string(index=False))
```

Local curated datasets live under `data/catalog/`, `data/data_providers/api_registry/`, and `data/alternative_data/`. Private credentials stay in environment variables or ignored `.env` files and are never written to this manifest.
