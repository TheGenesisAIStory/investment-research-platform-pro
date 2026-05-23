"""Smoke checks for the Data/API Control Center."""

from __future__ import annotations

import tempfile
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for candidate in [PROJECT_ROOT, PROJECT_ROOT / "src", PROJECT_ROOT / "research_platform_app"]:
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from research_platform_app.data_api_config import load_data_api_config
from research_platform_core.api_management import api_control_status
from research_platform_core.batch_download import create_batch_download
from research_platform_core.data_bridge import DataBridge
from research_platform_core.data_platform import resolve_data_platform_roots


def main() -> int:
    config = load_data_api_config()
    assert "theme" in config and "batch_download" in config

    roots = resolve_data_platform_roots(repo_output_root=PROJECT_ROOT / "output")
    status = api_control_status(roots.financial_db)
    assert {"summary", "api_providers", "credential_status"}.issubset(status)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        source = tmp_path / "sample.csv"
        pd.DataFrame({"x": [1, 2], "y": [3, 4]}).to_csv(source, index=False)
        inventory = pd.DataFrame(
            [{"path": str(source), "relative_path": "sample.csv", "file": "sample.csv", "suffix": ".csv", "role": "prices", "size_mb": 0.001, "age_hours": 1}]
        )
        batch = create_batch_download(inventory, tmp_path / "output", export_format="csv")
        assert batch["manifest_path"].exists()
        bridge_paths = DataBridge(tmp_path, tmp_path / "output").push_manifest(inventory)
        assert bridge_paths["contract"].exists()

    print("data_api_control_smoke_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
