from __future__ import annotations

from pathlib import Path
import tempfile

import pandas as pd

from research_platform_core.api_management import api_control_status, build_env_template
from research_platform_core.batch_download import create_batch_download, enrich_inventory_for_export
from research_platform_core.data_bridge import DataBridge


def test_api_control_status_uses_masked_catalog(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog"
    api_root = tmp_path / "API"
    catalog.mkdir()
    api_root.mkdir()
    (catalog / "api_providers.csv").write_text(
        "name,category,url,env_var,openbb_credential,data_type,ml_use,quality,free_tier,notes\n"
        "FRED,Macro,https://fred.stlouisfed.org/,FRED_API_KEY,fred_api_key,Macro,Regimes,Alta,Yes,Official\n",
        encoding="utf-8",
    )
    (catalog / "credential_status_masked.csv").write_text(
        "name,env_var,configured,credential_preview,openbb_credential\n"
        "FRED,FRED_API_KEY,True,abc...yz,fred_api_key\n",
        encoding="utf-8",
    )
    status = api_control_status(tmp_path, api_root)
    assert len(status["api_providers"]) == 1
    assert bool(status["credential_status"]["configured"].iloc[0]) is True
    assert "FRED_API_KEY=" in build_env_template(status["api_providers"])


def test_batch_download_exports_csv(tmp_path: Path) -> None:
    source = tmp_path / "prices.csv"
    pd.DataFrame({"date": ["2026-01-01"], "close": [100.0]}).to_csv(source, index=False)
    inventory = pd.DataFrame(
        [
            {
                "path": str(source),
                "relative_path": "prices.csv",
                "file": "prices.csv",
                "suffix": ".csv",
                "role": "prices",
                "size_mb": 0.001,
                "age_hours": 1,
            }
        ]
    )
    result = create_batch_download(inventory, tmp_path / "output", export_format="csv")
    assert result["metadata"]["summary"]["files"] == 1
    assert result["manifest"]["status"].iloc[0] == "written_csv"
    assert Path(result["manifest"]["export_path"].iloc[0]).exists()


def test_data_bridge_writes_manifest(tmp_path: Path) -> None:
    source = tmp_path / "dataset.csv"
    source.write_text("a\n1\n", encoding="utf-8")
    inventory = enrich_inventory_for_export(
        pd.DataFrame(
            [
                {
                    "path": str(source),
                    "relative_path": "dataset.csv",
                    "file": "dataset.csv",
                    "suffix": ".csv",
                    "role": "prices",
                    "size_mb": 0.001,
                    "age_hours": 1,
                }
            ]
        )
    )
    bridge = DataBridge(tmp_path, tmp_path / "output")
    paths = bridge.push_manifest(inventory)
    assert paths["manifest"].exists()
    assert paths["contract"].exists()


if __name__ == "__main__":
    for test in [
        test_api_control_status_uses_masked_catalog,
        test_batch_download_exports_csv,
        test_data_bridge_writes_manifest,
    ]:
        with tempfile.TemporaryDirectory() as tmp:
            test(Path(tmp))
    print("data_api_control_tests_OK")
