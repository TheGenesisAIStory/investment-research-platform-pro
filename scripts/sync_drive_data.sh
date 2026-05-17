#!/usr/bin/env bash
set -euo pipefail

# Google Drive sync settings. Override from the shell if needed:
#   REMOTE_PATH="Database Finanziario" bash scripts/sync_drive_data.sh
REMOTE_NAME="${REMOTE_NAME:-gdrive}"
REMOTE_PATH="${REMOTE_PATH:-Database Finanziario}"
LOCAL_DATA_DIR="${LOCAL_DATA_DIR:-/Users/itsgennymac/Library/CloudStorage/GoogleDrive-sfn.gns@gmail.com/Il mio Drive/Database Finanziario}"
DRY_RUN="${DRY_RUN:-0}"
EXTRA_RCLONE_ARGS="${EXTRA_RCLONE_ARGS:-}"

DRIVE_FOLDER_URL="https://drive.google.com/drive/folders/11xTU7S06NBFSEtvHM4CFLab7uqRenlI1?usp=drive_link"
SOURCE="${REMOTE_NAME}:${REMOTE_PATH}"

echo "== ML Trading Drive Data Sync =="
echo "Drive folder : ${DRIVE_FOLDER_URL}"
echo "Remote source: ${SOURCE}"
echo "Local target : ${LOCAL_DATA_DIR}"
echo
echo "Note: if the Drive folder is only under 'Shared with me', add a shortcut"
echo "to My Drive first, then set REMOTE_PATH to that shortcut name."
echo

if ! command -v rclone >/dev/null 2>&1; then
  echo "ERROR: rclone is not installed."
  echo "Install it with: brew install rclone"
  echo "Then configure Google Drive with: rclone config"
  exit 1
fi

mkdir -p "${LOCAL_DATA_DIR}"

RCLONE_ARGS=(sync "${SOURCE}" "${LOCAL_DATA_DIR}" --progress --create-empty-src-dirs)
if [[ "${DRY_RUN}" == "1" ]]; then
  RCLONE_ARGS+=(--dry-run)
fi
if [[ -n "${EXTRA_RCLONE_ARGS}" ]]; then
  read -r -a EXTRA_ARGS <<< "${EXTRA_RCLONE_ARGS}"
  RCLONE_ARGS+=("${EXTRA_ARGS[@]}")
fi

echo "Running: rclone ${RCLONE_ARGS[*]}"
echo "This only writes inside LOCAL_DATA_DIR and never deletes files in the git repo."
rclone "${RCLONE_ARGS[@]}"

echo
echo "Done. Use this in notebooks:"
echo "  ML_TRADING_DB_BASE=${LOCAL_DATA_DIR}"
