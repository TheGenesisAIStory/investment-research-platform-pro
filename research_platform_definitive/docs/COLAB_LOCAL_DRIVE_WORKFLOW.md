# Colab, Local and Google Drive Workflow

Use one canonical notebook version. Do not maintain separate local and Colab notebooks.

The operating policy is now **Drive-first**:

- Google Drive hosts the canonical runnable project.
- Local folders are development mirrors.
- The large `Database Finanziario` remains a separate Drive data lake.

## Recommended Layout

Keep this folder synced to Google Drive:

```text
MyDrive/machine-learning-for-trading/research_platform_definitive/
```

The Database Finanziario remains separate:

```text
MyDrive/Database Finanziario/
```

This avoids duplicated databases while giving Colab the full codebase it needs.

## Sync From Local Mac To Drive

From the repo root:

```bash
./scripts/sync_repo_to_google_drive.sh
```

This updates:

```text
MyDrive/GitHub/machine-learning-for-trading/
```

From `research_platform_definitive/` you can still sync only the definitive bundle:

```bash
./scripts/sync_definitive_to_drive.sh
```

Default sync excludes virtual environments, runtime run history, Python caches, local archives and transient data cache.

If you really want a full project copy on Drive including local DB mirrors/cache/archive, run:

```bash
SYNC_MODE=full ./scripts/sync_repo_to_google_drive.sh
```

Use full mode sparingly. The preferred data lake is still `MyDrive/Database Finanziario`, not duplicated local cache folders.

## Run Local Tools Against Drive

From the repo root:

```bash
source scripts/use_drive_research_platform.sh
streamlit run research_platform_definitive/research_platform_app/app.py
```

This forces local Streamlit/notebooks to read and write the Drive project copy.

## Colab Bootstrap Behavior

Each canonical notebook now starts with `Portable Research Platform Bootstrap - Canonical v1.2`.

It tries, in order:

1. `RESEARCH_PLATFORM_ROOT` environment variable.
2. Google Drive project paths.
3. Local VS Code / Jupyter paths.
4. `/content/...` Colab clone paths.
5. Optional GitHub clone into `/content/machine-learning-for-trading`.

The notebook also sets writable output/cache folders and avoids writing to read-only `/content` locations when running locally.

## If Colab Still Fails

Run this in a Colab cell before the notebook bootstrap:

```python
import os
os.environ["RESEARCH_PLATFORM_ROOT"] = "/content/drive/MyDrive/machine-learning-for-trading/research_platform_definitive"
```

Or let Colab clone the repo if Drive is not synced:

```python
import os
os.environ["RESEARCH_PLATFORM_GIT_URL"] = "https://github.com/TheGenesisAIStory/ml-trading-thesis-bot.git"
os.environ["RESEARCH_PLATFORM_AUTO_CLONE"] = "1"
```

## Canonical Notebooks

- `company_valuation/notebooks/Company_Valuation_Final_Version.ipynb`
- `portfolio_analysis/notebooks/Portfolio-Analysis-Model_RESEARCH_PLATFORM_PRO.ipynb`
- `machine_learning_lab/notebooks/ML_Stock_Lab_Experiments.ipynb`

`notebooks/README.md` is only an index.
