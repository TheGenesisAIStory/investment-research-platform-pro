"""Smoke checks for the global OHLCV pipeline."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for candidate in [PROJECT_ROOT, PROJECT_ROOT / "src"]:
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from research_platform_core.data_center_catalog import build_target_catalog
from research_platform_core.loaders.ohlcv_client import normalize_ohlcv_frame
from research_platform_core.ohlcv_store import OhlcvDatabase


def main() -> int:
    raw = pd.DataFrame({"Date": ["2024-01-02"], "Open": [1], "High": [2], "Low": [0.5], "Close": [1.5], "Volume": [100]})
    normalized = normalize_ohlcv_frame(raw, "AAA", "smoke")
    assert not normalized.empty
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        db = OhlcvDatabase(database_url=f"sqlite:///{tmp_path / 'ohlcv.sqlite'}")
        asset_map = db.upsert_assets(
            pd.DataFrame(
                [{"ticker": "AAA", "provider_symbol": "AAA", "exchange": "NASDAQ", "country": "US", "type": "stock", "name": "AAA", "primary_source": "smoke", "active_flag": True}]
            )
        )
        normalized["asset_id"] = next(iter(asset_map.values()))
        db.upsert_daily_prices(normalized)
        assert db.latest_daily_dates(list(asset_map.values()))
        catalog = build_target_catalog(tmp_path)
        assert "ohlcv_prices" in set(catalog["domain"])
    print("ohlcv_pipeline_smoke_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
