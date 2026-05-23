#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DRIVE_ROOT="${1:-/Users/itsgennymac/Library/CloudStorage/GoogleDrive-sfn.gns@gmail.com/Il mio Drive/machine-learning-for-trading/research_platform_definitive}"

mkdir -p "$DRIVE_ROOT"

rsync -a --delete \
  --exclude ".DS_Store" \
  --exclude "__pycache__/" \
  --exclude "*.pyc" \
  --exclude ".git/" \
  --exclude "archive/" \
  --exclude "local_databases_not_on_drive/" \
  --exclude "output/data_cache/" \
  --exclude "research_platform_app/runs/" \
  --exclude "research_platform_app/state/" \
  "$SOURCE_ROOT/" "$DRIVE_ROOT/"

echo "synced_OK"
echo "source=$SOURCE_ROOT"
echo "drive_root=$DRIVE_ROOT"
