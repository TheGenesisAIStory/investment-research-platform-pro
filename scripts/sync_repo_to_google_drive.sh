#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DRIVE_ROOT="${1:-/Users/itsgennymac/Library/CloudStorage/GoogleDrive-sfn.gns@gmail.com/Il mio Drive/GitHub/investment-research-platform-pro}"
MODE="${SYNC_MODE:-operational}"

mkdir -p "$DRIVE_ROOT"

COMMON_EXCLUDES=(
  --exclude ".git/"
  --exclude ".venv/"
  --exclude "venv/"
  --exclude "env/"
  --exclude "__pycache__/"
  --exclude "*.pyc"
  --exclude ".DS_Store"
  --exclude ".ipynb_checkpoints/"
  --exclude "research_platform_definitive/research_platform_app/runs/"
  --exclude "research_platform_definitive/research_platform_app/state/"
)

OPERATIONAL_EXCLUDES=(
  --exclude "research_platform_definitive/local_databases_not_on_drive/"
  --exclude "research_platform_definitive/output/data_cache/"
  --exclude "research_platform_definitive/archive/"
)

if [[ "$MODE" == "full" ]]; then
  rsync -a --delete "${COMMON_EXCLUDES[@]}" "$SOURCE_ROOT/" "$DRIVE_ROOT/"
else
  rsync -a --delete "${COMMON_EXCLUDES[@]}" "${OPERATIONAL_EXCLUDES[@]}" "$SOURCE_ROOT/" "$DRIVE_ROOT/"
fi

echo "repo_synced_OK"
echo "mode=$MODE"
echo "source=$SOURCE_ROOT"
echo "drive_root=$DRIVE_ROOT"
echo "colab_root=/content/drive/MyDrive/GitHub/investment-research-platform-pro"
echo
echo "Use full mode only if you really want local DB/cache/archive copies on Drive:"
echo "  SYNC_MODE=full $0"
