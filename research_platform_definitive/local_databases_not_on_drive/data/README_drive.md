# Google Drive Data Workspace

Heavy datasets for this repository live on Google Drive, not in Git.

- Drive folder: https://drive.google.com/drive/folders/11xTU7S06NBFSEtvHM4CFLab7uqRenlI1?usp=drive_link
- Canonical local Google Drive database folder: `/Users/itsgennymac/Library/CloudStorage/GoogleDrive-sfn.gns@gmail.com/Il mio Drive/Database Finanziario`
- Canonical notebook variable: `DATA_PATH`
- Canonical persistent output variable: `DB_BASE`

## Local Access

1. Install and configure `rclone`.

   ```bash
   brew install rclone
   rclone config
   ```

2. Configure a Google Drive remote named `gdrive`.

3. If the linked folder is only visible under "Shared with me", open it in Google Drive and choose "Add shortcut to Drive". Put the shortcut in My Drive with a stable name such as `Database Finanziario`.

4. Sync data to the external local folder.

   ```bash
   bash scripts/sync_drive_data.sh
   ```

5. In local notebooks, keep data access behind `DATA_PATH`.

   ```python
   from pathlib import Path
   DB_BASE = Path("/Users/itsgennymac/Library/CloudStorage/GoogleDrive-sfn.gns@gmail.com/Il mio Drive/Database Finanziario")
   DATA_PATH = DB_BASE
   ```

You can override the default without editing notebooks:

```bash
export ML_TRADING_DB_BASE="/path/to/local/Database Finanziario"
```

## Colab Access

In Colab, mount Drive and point `DATA_PATH` to the Drive folder or shortcut.

```python
import os
from pathlib import Path
from google.colab import drive

drive.mount("/content/drive")
DB_BASE = Path(os.environ.get("ML_TRADING_DB_BASE", "/content/drive/MyDrive/Database Finanziario"))
DATA_PATH = DB_BASE
```

If the folder is shared with you, add a shortcut to My Drive first, or configure the notebook path to the exact mounted shortcut location.

## rclone Notes

- Prefer `rclone` for large folders and many-file datasets.
- Avoid `gdown` for large Drive folders; it is less reliable with quota prompts, shared-folder edge cases, and many files.
- For shared folders, either add a shortcut to My Drive or configure `rclone` for the shared folder/folder ID explicitly.
- The default script expects `gdrive:Database Finanziario`. Change `REMOTE_NAME`, `REMOTE_PATH`, or `LOCAL_DATA_DIR` at the top of `scripts/sync_drive_data.sh` if your remote is different.

## Git Policy

Do not commit database files to GitHub. Keep only documentation, tiny samples, and link metadata in this repo.

Allowed for the Drive-managed data workspace under `data/`:

- `data/README_drive.md`
- `data/sample/`
- `data/links/`

Existing lightweight catalog/source files already tracked by the repository can remain tracked, but new heavy dataset exports should stay in Drive or in the external local mirror.

Ignored under `data/`:

- local market databases
- parquet/h5/sqlite/feather/pickle/csv dataset exports
- large downloaded caches
