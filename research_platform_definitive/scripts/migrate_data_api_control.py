"""Create/refresh Data/API control artifacts from the existing setup."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for candidate in [PROJECT_ROOT, PROJECT_ROOT / "src", PROJECT_ROOT / "research_platform_app"]:
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from research_platform_core.api_management import resolve_api_control_roots, write_api_control_status
from research_platform_core.data_platform import resolve_data_platform_roots, write_data_platform_status


def main() -> int:
    roots = resolve_data_platform_roots(repo_output_root=PROJECT_ROOT / "output")
    api_roots = resolve_api_control_roots(roots.financial_db)
    data_paths = write_data_platform_status(roots.financial_db, PROJECT_ROOT / "output", max_files=5000)
    api_paths = write_api_control_status(roots.financial_db, PROJECT_ROOT / "output", api_roots.api_root)
    print("Data/API control migration complete")
    print(f"Financial DB: {roots.financial_db} | exists={roots.available}")
    print(f"API root: {api_roots.api_root} | exists={api_roots.api_available}")
    print("Data contracts:")
    for name, path in data_paths.items():
        print(f"- {name}: {path}")
    print("API contracts:")
    for name, path in api_paths.items():
        print(f"- {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
