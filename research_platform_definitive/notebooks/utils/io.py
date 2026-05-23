import os
import shutil
import logging
from pathlib import Path
import subprocess
import sys

def get_project_root() -> Path:
    if Path('/content').exists():
        return Path('/content')
    return Path.cwd()

def get_data_dir() -> Path:
    root = get_project_root()
    data_dir = root / 'data_db'
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir

def mount_drive_and_get_root(logger=None):
    if logger is None:
        logger = logging.getLogger(__name__)
    try:
        from google.colab import drive
        if not Path("/content/drive/MyDrive").exists():
            drive.mount("/content/drive")
        logger.info("Google Drive montato: /content/drive/MyDrive")
    except Exception as e:
        logger.warning(f"Drive mount skipped: {e}")

    mydrive = Path("/content/drive/MyDrive")
    db_name = next(
        (n for n in os.listdir(str(mydrive))
         if "Database" in n and "Finanziario" in n), None
    ) if mydrive.exists() else None

    db_root = mydrive / db_name if db_name else None
    logger.info(f"DB_ROOT: {db_root} | exists: {db_root.exists() if db_root else False}")
    return db_root
