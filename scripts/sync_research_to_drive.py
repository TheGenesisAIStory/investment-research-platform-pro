"""Sync research project folders to the local Google Drive database folder.

Default destination:
    /Users/itsgennymac/.../Il mio Drive/Database Finanziario/research_exports

The script exits with a clear warning if the local Drive folder is unavailable
instead of failing silently.
"""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_FOLDERS = ["company_valuation", "pead_european_banks_ifrs9", "portfolio_analysis", "prompts"]
SUPPORT_FILES = ["RESEARCH_PROJECTS.md", "EXPORT_LOCATIONS.md", "data/README_drive.md"]
GENERATED_OUTPUT_NAMES = {"output", ".ipynb_checkpoints", "__pycache__"}
MANIFEST_NAME = "RESEARCH_EXPORT_MANIFEST.md"
DEFAULT_DATABASE_ROOT = Path(
    "/Users/itsgennymac/Library/CloudStorage/"
    "GoogleDrive-sfn.gns@gmail.com/Il mio Drive/Database Finanziario"
)
DEFAULT_DESTINATION = DEFAULT_DATABASE_ROOT / "research_exports"


def _ignore_generated(_: str, names: list[str]) -> set[str]:
    """Skip generated artifacts by default so Drive syncs remain reviewable."""
    return {name for name in names if name in GENERATED_OUTPUT_NAMES}


def _copy_path(src: Path, dst: Path, *, include_output: bool = False) -> Path:
    if dst.exists():
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, ignore=None if include_output else _ignore_generated)
    else:
        shutil.copy2(src, dst)
    return dst


def _write_manifest(destination: Path, copied: list[Path]) -> Path:
    manifest = destination / MANIFEST_NAME
    lines = [
        "# Research Export Manifest",
        "",
        f"Generated UTC: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"Destination: `{destination}`",
        "",
        "## Copied paths",
        "",
    ]
    for path in copied:
        kind = "directory" if path.is_dir() else "file"
        lines.append(f"- `{path}` ({kind})")
    manifest.write_text("\n".join(lines) + "\n")
    return manifest


def sync_to_drive(
    destination: Path,
    folders: list[str],
    *,
    include_output: bool = False,
    include_support_files: bool = True,
    repo_root: Path = REPO_ROOT,
) -> list[Path]:
    """Copy selected research folders to Drive from any current working directory."""
    if not destination.parent.exists():
        raise FileNotFoundError(
            f"Drive parent folder does not exist: {destination.parent}. "
            "Confirm that Google Drive for Desktop is mounted or pass --destination explicitly."
        )
    destination.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for folder in folders:
        src = (repo_root / folder).resolve()
        if not src.exists():
            print(f"⚠️ Skipping missing folder: {src}")
            continue
        if not src.is_dir():
            print(f"⚠️ Skipping non-directory path: {src}")
            continue
        copied.append(_copy_path(src, destination / src.name, include_output=include_output))
    if include_support_files:
        for relative in SUPPORT_FILES:
            src = repo_root / relative
            if src.exists():
                copied.append(_copy_path(src, destination / relative, include_output=include_output))
    copied.append(_write_manifest(destination, copied))
    return copied


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync research project folders to a mounted Google Drive directory.")
    parser.add_argument("--destination", default=str(DEFAULT_DESTINATION))
    parser.add_argument("--folders", nargs="*", default=PROJECT_FOLDERS)
    parser.add_argument("--include-output", action="store_true", help="Also copy generated output/ folders and notebook checkpoints.")
    parser.add_argument("--no-support-files", action="store_true", help="Skip RESEARCH_PROJECTS.md, EXPORT_LOCATIONS.md, and Drive data docs.")
    args = parser.parse_args()
    destination = Path(args.destination)
    try:
        copied = sync_to_drive(
            destination,
            args.folders,
            include_output=args.include_output,
            include_support_files=not args.no_support_files,
        )
    except Exception as exc:
        print(f"⚠️ Drive sync skipped: {exc}")
        return 0
    print("✅ Synced folders to Drive:")
    for path in copied:
        print(f"   {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
