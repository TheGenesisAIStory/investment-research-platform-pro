"""AQR loader wrapper for the Data Center enhancement."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..aqr_factors import AqrFactorProvider, refresh_aqr_factor_library


class AQRDataLoader:
    """Use the existing AQR discovery/parser with Data Center naming."""

    def __init__(self, financial_db_root: Path | str | None = None, output_root: Path | str | None = None):
        self.provider = AqrFactorProvider(financial_db_root=financial_db_root, output_root=output_root)
        self.financial_db_root = Path(financial_db_root).expanduser() if financial_db_root else None
        self.output_root = Path(output_root).expanduser() if output_root else None

    def catalog(self, refresh: bool = False) -> pd.DataFrame:
        discovery = self.provider.discover(force=refresh)
        if discovery.empty:
            return discovery
        out = discovery.copy()
        out["provider"] = "aqr"
        out["target_family"] = "Factors/AQR"
        return out

    def download_all(self, refresh: bool = False, max_datasets: int | None = None) -> dict[str, Path]:
        return refresh_aqr_factor_library(
            financial_db_root=self.financial_db_root,
            output_root=self.output_root,
            refresh=refresh,
            max_datasets=max_datasets,
        )

    def get_dataset(self, slug: str, refresh: bool = False) -> pd.DataFrame:
        return self.provider.get_dataset(slug, refresh=refresh)
