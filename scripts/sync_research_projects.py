"""Copy research project folders to a local destination.

This script is intentionally generic: it can mirror the three canonical research
projects to Google Drive, another checked-out repository, or a local folder such
as `/Users/itsgennymac/GitHub/machine-learning-for-trading`.
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_PROJECT_FOLDERS = ["company_valuation", "pead_european_banks_ifrs9", "portfolio_analysis"]
OPTIONAL_SUPPORT_FILES = ["RESEARCH_PROJECTS.md", "prompts", "scripts", "data/README_drive.md"]
GENERATED_OUTPUT_NAMES = {"output", ".ipynb_checkpoints", "__pycache__"}
DEFAULT_DATABASE_ROOT = Path(
    "/Users/itsgennymac/Library/CloudStorage/"
    "GoogleDrive-sfn.gns@gmail.com/Il mio Drive/Database Finanziario"
)
DEFAULT_LOCAL_DESTINATION = DEFAULT_DATABASE_ROOT / "research_exports"
MANIFEST_NAME = "RESEARCH_EXPORT_MANIFEST.md"
ZIP_NAME = "ml_trading_research_projects_bundle.zip"


def _ignore_generated(_: str, names: list[str]) -> set[str]:
    """Skip generated artifacts by default so mirrors stay lightweight."""
    return {name for name in names if name in GENERATED_OUTPUT_NAMES}


def copy_path(src: Path, dst: Path, *, include_output: bool = False) -> Path:
    """Copy one file or directory, replacing any existing destination path."""
    if not src.exists():
        raise FileNotFoundError(f"Source path does not exist: {src}")
    if dst.exists():
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        ignore = None if include_output else _ignore_generated
        shutil.copytree(src, dst, ignore=ignore)
    else:
        shutil.copy2(src, dst)
    return dst


def write_manifest(destination: Path, copied: list[Path], *, zip_path: Path | None = None) -> Path:
    """Write a visible manifest with exact exported locations."""
    destination.mkdir(parents=True, exist_ok=True)
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
    if zip_path is not None:
        lines.extend(["", "## Bundle", "", f"- `{zip_path}`"])
    manifest.write_text("\n".join(lines) + "\n")
    return manifest


def create_zip_bundle(destination: Path, copied: list[Path], *, zip_name: str = ZIP_NAME) -> Path:
    """Create one easy-to-find ZIP bundle from the copied paths."""
    zip_path = destination / zip_name
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for copied_path in copied:
            if copied_path.is_dir():
                for file_path in copied_path.rglob("*"):
                    if file_path.is_file() and file_path.name != zip_name:
                        zf.write(file_path, file_path.relative_to(destination))
            elif copied_path.is_file():
                zf.write(copied_path, copied_path.relative_to(destination))
    return zip_path


def sync_research_projects(
    destination: Path,
    *,
    include_output: bool = False,
    include_support_files: bool = True,
    make_zip: bool = False,
    repo_root: Path = REPO_ROOT,
) -> list[Path]:
    """Mirror the three research project folders into a local destination."""
    copied: list[Path] = []
    for relative in RESEARCH_PROJECT_FOLDERS:
        copied.append(copy_path(repo_root / relative, destination / relative, include_output=include_output))
    if include_support_files:
        for relative in OPTIONAL_SUPPORT_FILES:
            source = repo_root / relative
            if source.exists():
                copied.append(copy_path(source, destination / relative, include_output=include_output))
    zip_path = create_zip_bundle(destination, copied) if make_zip else None
    manifest = write_manifest(destination, copied, zip_path=zip_path)
    return [*copied, manifest, *([zip_path] if zip_path is not None else [])]


def main() -> int:
    parser = argparse.ArgumentParser(description="Mirror the canonical research project folders to a local destination.")
    parser.add_argument("--destination", type=Path, default=DEFAULT_LOCAL_DESTINATION)
    parser.add_argument("--include-output", action="store_true", help="Also copy generated output/ folders and notebook checkpoints.")
    parser.add_argument("--no-support-files", action="store_true", help="Copy only the three project folders, without RESEARCH_PROJECTS.md/prompts/scripts/Drive data docs.")
    parser.add_argument("--zip", action="store_true", help=f"Also create {ZIP_NAME} in the destination folder.")
    args = parser.parse_args()
    copied = sync_research_projects(
        args.destination,
        include_output=args.include_output,
        include_support_files=not args.no_support_files,
        make_zip=args.zip,
    )
    print(f"✅ Synced {len(copied)} research paths to {args.destination}")
    for path in copied:
        print(f"   {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
