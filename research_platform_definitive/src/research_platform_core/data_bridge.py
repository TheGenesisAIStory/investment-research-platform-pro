"""Bridge selected datasets from the Data/API platform into ml_stock_lab."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Iterable

import pandas as pd

from .batch_download import enrich_inventory_for_export, slugify
from .data_platform import utc_now


class DataBridge:
    """Small file-contract bridge between Research Platform and ml_stock_lab."""

    def __init__(self, project_root: Path | str, output_root: Path | str):
        self.project_root = Path(project_root).expanduser()
        self.output_root = Path(output_root).expanduser()
        self.bridge_root = self.output_root / "ml_stock_lab" / "data_bridge"
        self.bridge_root.mkdir(parents=True, exist_ok=True)

    def discover_ml_stock_lab_notebooks(self) -> pd.DataFrame:
        rows = []
        for root in [self.project_root / "ml_stock_lab", self.project_root / "machine_learning_lab"]:
            if not root.exists():
                continue
            for path in root.rglob("*.ipynb"):
                rows.append({"notebook": path.name, "path": str(path), "updated_at": pd.Timestamp(path.stat().st_mtime, unit="s").isoformat()})
        return pd.DataFrame(rows)

    def push_manifest(self, inventory: pd.DataFrame, selected_ids: Iterable[str] | None = None, copy_files: bool = False) -> dict[str, Path]:
        data = enrich_inventory_for_export(inventory)
        if selected_ids is not None:
            selected = set(selected_ids)
            data = data[data["dataset_id"].isin(selected)]
        manifest_dir = self.bridge_root / "manifests"
        dataset_dir = self.bridge_root / "datasets"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        timestamp = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
        manifest_path = manifest_dir / f"ml_stock_lab_dataset_manifest_{timestamp}.csv"
        data.to_csv(manifest_path, index=False)
        copied_rows = []
        if copy_files:
            for _, row in data.iterrows():
                source = Path(str(row.get("path", ""))).expanduser()
                if not source.exists() or not source.is_file():
                    copied_rows.append({"source": str(source), "status": "missing"})
                    continue
                role = slugify(row.get("role"), "other")
                target = dataset_dir / role / source.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                copied_rows.append({"source": str(source), "target": str(target), "status": "copied"})
        copy_manifest = manifest_dir / f"ml_stock_lab_dataset_copy_{timestamp}.csv"
        pd.DataFrame(copied_rows).to_csv(copy_manifest, index=False)
        contract = {
            "generated_at": utc_now(),
            "manifest": str(manifest_path),
            "copy_manifest": str(copy_manifest),
            "rows": len(data),
            "copy_files": copy_files,
        }
        contract_path = self.bridge_root / "data_bridge_contract.json"
        contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
        return {"manifest": manifest_path, "copy_manifest": copy_manifest, "contract": contract_path}
